"""Export scripted AlignBox3D demonstrations for LeRobot pi0.5.

The collector's per-frame task text describes an expert-controlled stage. A
deployed policy receives one high-level goal, so this export replaces only the
language condition. State, actions, images, and episode timing are unchanged.
The training source is written first and the held-out validation source last,
matching LeRobot v0.6.1's per-task tail episode split.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .dataset_format import DATA_PATH, EPISODE_PATH, VIDEO_PATH, validate_dataset


MODEL_FEATURES = (
    "observation.state", "action", "observation.images.camera2",
    "timestamp", "frame_index", "episode_index", "index", "task_index",
)
NUMERIC_FEATURES = tuple(key for key in MODEL_FEATURES if key != "observation.images.camera2")
QUANTILE_FEATURES = ("observation.state", "action")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _episode_rows(root: Path):
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    return [row for file in files for row in pq.read_table(file).to_pylist()]


def _source(root: Path, expected_split: str):
    validate_dataset(root)
    info = _read_json(root / "meta/info.json")
    split = _read_json(root / "meta/data_split.json")
    if split.get("split") != expected_split:
        raise ValueError(f"{root}: expected {expected_split} source, got {split.get('split')!r}.")
    for key in MODEL_FEATURES:
        if key not in info["features"]:
            raise ValueError(f"{root}: missing pi0.5 feature {key}.")
    if (info["features"]["observation.state"]["shape"] != [20]
            or info["features"]["action"]["shape"] != [14]):
        raise ValueError(f"{root}: expected AlignBox3D state[20] and action[14].")
    if info["features"]["observation.images.camera2"]["dtype"] != "video":
        raise ValueError(f"{root}: camera2 must be an RGB video feature.")
    if info["data_path"] != DATA_PATH or info["video_path"] != VIDEO_PATH:
        raise ValueError(f"{root}: data/video path templates differ from this exporter.")
    episodes = _episode_rows(root)
    if len(episodes) != info["total_episodes"]:
        raise ValueError(f"{root}: episode metadata count mismatch.")
    return info, split, episodes


class _Moments:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.count = 0
        self.total = np.zeros(self.shape, dtype=np.float64)
        self.square = np.zeros(self.shape, dtype=np.float64)
        self.minimum = np.full(self.shape, np.inf)
        self.maximum = np.full(self.shape, -np.inf)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, *self.shape)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("A model feature is empty or non-finite.")
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.square += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def result(self):
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.square / self.count - np.square(mean), 0.0))
        return {"min": self.minimum.tolist(), "max": self.maximum.tolist(),
                "mean": mean.tolist(), "std": std.tolist(), "count": [self.count]}


def _video_stats(rows, shape):
    """Combine the collector's exact pre-encode per-episode pixel moments."""
    key = "observation.images.camera2"
    height, width, _ = shape
    moments = _Moments((3, 1, 1))
    for row in rows:
        prefix = f"stats/{key}/"
        mean = np.asarray(row[prefix + "mean"], dtype=np.float64)
        std = np.asarray(row[prefix + "std"], dtype=np.float64)
        low = np.asarray(row[prefix + "min"], dtype=np.float64)
        high = np.asarray(row[prefix + "max"], dtype=np.float64)
        pixels = int(row["length"]) * height * width
        if mean.shape != (3, 1, 1) or not np.isfinite([*mean.ravel(), *std.ravel()]).all():
            raise ValueError("Source camera statistics have an unexpected shape or value.")
        moments.count += pixels
        moments.total += mean * pixels
        moments.square += (np.square(std) + np.square(mean)) * pixels
        moments.minimum = np.minimum(moments.minimum, low)
        moments.maximum = np.maximum(moments.maximum, high)
    result = moments.result()
    result["count"] = [sum(int(row["length"]) for row in rows)]
    return result


def _numeric_array(table, key, shape):
    values = np.asarray(table[key].to_pylist())
    return values.reshape(len(table), *shape)


def _write_export(train: Path, validation: Path, destination: Path, task: str):
    sources = [(train, "train"), (validation, "validation")]
    loaded = [(root, expected, *_source(root, expected)) for root, expected in sources]
    train_info, train_split = loaded[0][2:4]
    validation_info, validation_split = loaded[1][2:4]
    for key in ("fps", "robot_type", "features"):
        if key == "features":
            if any(train_info[key][name] != validation_info[key][name] for name in MODEL_FEATURES):
                raise ValueError("Train and validation model feature schemas differ.")
        elif train_info[key] != validation_info[key]:
            raise ValueError(f"Train and validation {key} differ.")
    if train_split.get("config") != validation_split.get("config"):
        raise ValueError("Train and validation scene configurations differ.")
    train_contract = _read_json(train / "meta/provenance.json")["environment_contract"]
    validation_contract = _read_json(validation / "meta/provenance.json")["environment_contract"]
    if train_contract != validation_contract:
        raise ValueError("Train and validation policy contracts differ.")
    train_range = train_split["config"]["splits"]["train"]["seed_range"]
    validation_range = validation_split["config"]["splits"]["validation"]["seed_range"]
    if max(train_range[0], validation_range[0]) < min(train_range[1], validation_range[1]):
        raise ValueError("Train and validation seed ranges overlap.")

    info = {**train_info, "features": {key: train_info["features"][key] for key in MODEL_FEATURES},
            "total_episodes": sum(len(item[4]) for item in loaded),
            "total_frames": sum(item[2]["total_frames"] for item in loaded),
            "total_tasks": 1}
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    stats = {key: _Moments(info["features"][key]["shape"]) for key in NUMERIC_FEATURES}
    total_frames = info["total_frames"]
    train_frames = train_info["total_frames"]
    quantiles = {key: np.lib.format.open_memmap(destination / f".{key}.npy", mode="w+",
                                                 dtype=np.float32, shape=(train_frames, *info["features"][key]["shape"]))
                 for key in QUANTILE_FEATURES}
    output_rows = []
    video_map = {}
    frame_offset = 0
    all_source_rows = []
    source_summary = []
    for root, split_name, source_info, split, episodes in loaded:
        source_summary.append({"split": split_name, "path": str(root.resolve()),
                               "episodes": len(episodes), "frames": source_info["total_frames"],
                               "seed_range": split["config"]["splits"][split_name]["seed_range"]})
        for source_row in episodes:
            episode_index = len(output_rows)
            length = int(source_row["length"])
            source_file = root / source_info["data_path"].format(
                chunk_index=source_row["data/chunk_index"], file_index=source_row["data/file_index"])
            table = pq.read_table(source_file, columns=list(NUMERIC_FEATURES))
            source_start = int(table["index"][0].as_py())
            offset = int(source_row["dataset_from_index"]) - source_start
            if offset < 0 or offset + length > len(table):
                raise ValueError(f"{source_file}: episode range exceeds its data file.")
            table = table.slice(offset, length)
            if (table["episode_index"][0].as_py() != source_row["episode_index"]
                    or table["index"][0].as_py() != source_row["dataset_from_index"]):
                raise ValueError(f"{source_file}: episode row alignment failed.")
            replacements = {"episode_index": np.full(length, episode_index, dtype=np.int64),
                            "index": np.arange(frame_offset, frame_offset + length, dtype=np.int64),
                            "task_index": np.zeros(length, dtype=np.int64)}
            for key, values in replacements.items():
                column = table.column_names.index(key)
                table = table.set_column(column, key, pa.array(values))
            table = table.select(list(NUMERIC_FEATURES))
            output_file = destination / DATA_PATH.format(chunk_index=0, file_index=episode_index)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_file, compression="zstd")

            row = {"episode_index": episode_index, "tasks": [task], "length": length,
                   "dataset_from_index": frame_offset, "dataset_to_index": frame_offset + length,
                   "data/chunk_index": 0, "data/file_index": episode_index,
                   "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
            for key in NUMERIC_FEATURES:
                values = _numeric_array(table, key, info["features"][key]["shape"])
                if split_name == "train":
                    stats[key].add(values)
                episode_moments = _Moments(info["features"][key]["shape"])
                episode_moments.add(values)
                for stat_name, stat_value in episode_moments.result().items():
                    row[f"stats/{key}/{stat_name}"] = stat_value
                if key in quantiles and split_name == "train":
                    quantiles[key][frame_offset:frame_offset + length] = values

            key = "observation.images.camera2"
            video_source = root / source_info["video_path"].format(
                video_key=key, chunk_index=source_row[f"videos/{key}/chunk_index"],
                file_index=source_row[f"videos/{key}/file_index"])
            if video_source not in video_map:
                video_index = len(video_map)
                video_map[video_source] = video_index
                video_target = destination / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=video_index)
                video_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(video_source, video_target)
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = video_map[video_source]
            for bound in ("from", "to"):
                row[f"videos/{key}/{bound}_timestamp"] = source_row[f"videos/{key}/{bound}_timestamp"]
            for name in ("min", "max", "mean", "std", "count"):
                row[f"stats/{key}/{name}"] = source_row[f"stats/{key}/{name}"]
            output_rows.append(row)
            all_source_rows.append(source_row)
            frame_offset += length

    if frame_offset != total_frames:
        raise ValueError("Exported frame count disagrees with source metadata.")
    stats_result = {key: values.result() for key, values in stats.items()}
    for key in tuple(quantiles):
        values = quantiles.pop(key)
        shape = tuple(info["features"][key]["shape"])
        for percentile, label in ((0.01, "q01"), (0.99, "q99")):
            stats_result[key][label] = np.asarray([
                np.quantile(values[:, channel], percentile) for channel in range(shape[0])
            ], dtype=np.float64).reshape(shape).tolist()
        values.flush()
        del values
        (destination / f".{key}.npy").unlink()
    stats_result["observation.images.camera2"] = _video_stats(all_source_rows[:len(loaded[0][4])],
        info["features"]["observation.images.camera2"]["shape"])

    episode_file = destination / EPISODE_PATH.format(chunk_index=0, file_index=0)
    episode_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), episode_file, compression="zstd")
    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task"))
    tasks.to_parquet(destination / "meta/tasks.parquet")
    _write_json(destination / "meta/info.json", info)
    _write_json(destination / "meta/stats.json", stats_result)
    _write_json(destination / "meta/contract.json", train_contract)
    source_schema = train / "meta/pushenv_schema.json"
    if source_schema.is_file():
        schema = _read_json(source_schema)
        schema["features"] = {key: schema["features"][key] for key in MODEL_FEATURES if key not in
                              ("timestamp", "frame_index", "episode_index", "index", "task_index")}
        schema["feature_semantics"] = {key: value for key, value in schema.get("feature_semantics", {}).items()
                                       if key in schema["features"]}
        schema.pop("language_stages", None)
        schema["data_split"] = {"split": "train_and_validation",
                                "config": train_split["config"],
                                "validation_first_episode": len(loaded[0][4])}
        _write_json(destination / "meta/pushenv_schema.json", schema)
    _write_json(destination / "meta/pi05_export.json", {
        "format": "alignbox3d_pi05_high_level/v1", "task": task,
        "source_datasets": source_summary,
        "validation_first_episode": len(loaded[0][4]),
        # LeRobot takes ceil(total_episodes * eval_split). A half-episode
        # margin avoids a floating-point round-up to one extra episode.
        "eval_split": (len(loaded[1][4]) - 0.5) / info["total_episodes"],
        "policy_features": list(MODEL_FEATURES),
        "actions_unchanged": True, "stage_labels_excluded": True,
        "normalization_stats_scope": "train_episodes_only",
    })
    _write_json(destination / "meta/data_split.json", {
        "train": train_split, "validation": validation_split,
        "validation_first_episode": len(loaded[0][4]),
    })


def check_export(root: Path):
    result = validate_dataset(root)
    info = _read_json(root / "meta/info.json")
    manifest = _read_json(root / "meta/pi05_export.json")
    if set(info["features"]) != set(MODEL_FEATURES):
        raise ValueError("Export has missing or privileged model features.")
    contract = _read_json(root / "meta/contract.json")
    if (contract.get("obs_dim") != 20 or contract.get("action_dim") != 14
            or contract.get("fps") != info["fps"]
            or contract.get("observation_names") != info["features"]["observation.state"]["names"]
            or contract.get("action_names") != info["features"]["action"]["names"]):
        raise ValueError("Exported policy contract disagrees with model features.")
    if manifest["validation_first_episode"] <= 0 or manifest["validation_first_episode"] >= result["episodes"]:
        raise ValueError("Both train and validation episodes are required.")
    stats = _read_json(root / "meta/stats.json")
    for key in QUANTILE_FEATURES:
        shape = info["features"][key]["shape"]
        for name in ("q01", "q99"):
            values = np.asarray(stats[key][name], dtype=np.float64)
            if values.shape != tuple(shape) or not np.isfinite(values).all():
                raise ValueError(f"{key} lacks correctly shaped {name} statistics.")
        if np.any(np.asarray(stats[key]["q01"]) > np.asarray(stats[key]["q99"])):
            raise ValueError(f"{key} quantile order is invalid.")
        train_frames = sum(source["frames"] for source in manifest["source_datasets"]
                           if source["split"] == "train")
        if stats[key]["count"] != [train_frames]:
            raise ValueError(f"{key} normalization statistics must use train frames only.")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if len(tasks) != 1 or tasks.index[0] != manifest["task"]:
        raise ValueError("Export must use the same high-level task for every frame.")
    return {**result, "train_episodes": manifest["validation_first_episode"],
            "validation_episodes": result["episodes"] - manifest["validation_first_episode"],
            "eval_split": manifest["eval_split"], "task": manifest["task"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", help="One high-level command used for every frame and rollout.")
    parser.add_argument("--check", type=Path, help="Validate an existing pi0.5 export instead of exporting.")
    args = parser.parse_args()
    if args.check is not None:
        if any(value is not None for value in (args.train, args.validation, args.output, args.task)):
            parser.error("--check cannot be combined with export arguments.")
        print(json.dumps(check_export(args.check), ensure_ascii=False, indent=2))
        return
    if any(value is None for value in (args.train, args.validation, args.output, args.task)):
        parser.error("Export requires --train, --validation, --output and --task.")
    task = args.task.strip()
    if not task:
        parser.error("--task must be a nonempty high-level instruction.")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pi05-export-", dir=output.parent) as temporary:
        destination = Path(temporary)
        _write_export(args.train.resolve(), args.validation.resolve(), destination, task)
        result = check_export(destination)
        os.rename(destination, output)
    result["root"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
