"""Convert and check AlignBox3D demonstrations with LeRobot 0.6.1.

Run in the LeRobot Python environment. The ReposeEnv collector's stage text is
checked against its stage ID, then replaced by the frozen public task goal.
Only state, camera2, action, and LeRobot's generated indices enter the model
dataset. The source datasets and their stage labels are never modified.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import lerobot
from lerobot.datasets.io_utils import write_info, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np


REPO_ID = "local/alignbox3d_pi05"
STATE = "observation.state"
CAMERA = "observation.images.camera2"
ACTION = "action"
MODEL_KEYS = {STATE, CAMERA, ACTION}
INDEX_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _require_version() -> None:
    if lerobot.__version__ != "0.6.1":
        raise RuntimeError(f"This project requires lerobot 0.6.1, found {lerobot.__version__}.")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _frozen_goal(root: Path) -> str:
    """Extract the collector's task_description literal without importing MuJoCo."""
    path = root / "meta/source_snapshot/alignbox3d/env.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AlignBox3DEnv"]
    if len(classes) != 1:
        raise ValueError(f"{path}: expected one AlignBox3DEnv definition.")
    values = [n.value for n in classes[0].body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "task_description" for t in n.targets)]
    if len(values) != 1:
        raise ValueError(f"{path}: expected one literal task_description.")
    try:
        goal = ast.literal_eval(values[0])
    except (ValueError, TypeError, SyntaxError, RecursionError) as error:
        raise ValueError(f"{path}: task_description must be literal text.") from error
    if not isinstance(goal, str) or not goal or goal != goal.strip():
        raise ValueError(f"{path}: task_description must be nonempty clean text.")
    return goal


def _source(root: Path, split_name: str) -> dict:
    split = _read(root / "meta/data_split.json")
    collection = _read(root / "meta/collection.json")
    info = _read(root / "meta/info.json")
    if split.get("split") != split_name or collection.get("complete") is not True:
        raise ValueError(f"{root}: expected a complete {split_name} collection.")
    attempts = collection.get("episodes", [])
    if not attempts or any(type(row.get("seed")) is not int for row in attempts):
        raise ValueError(f"{root}: missing collection attempt seeds.")
    seed_start, seed_stop = split["config"]["splits"][split_name]["seed_range"]
    if any(not seed_start <= row["seed"] < seed_stop for row in attempts):
        raise ValueError(f"{root}: collection attempt seed is outside the {split_name} range.")
    ds = LeRobotDataset(f"local/alignbox3d_raw_{split_name}", root=root,
                        return_uint8=True, video_backend="pyav",
                        tolerance_s=0.5 / info["fps"])
    if not ds.meta.total_episodes or collection.get("saved_episodes") != ds.meta.total_episodes:
        raise ValueError(f"{root}: saved episode count disagrees with collector report.")
    if not MODEL_KEYS | {"language.stage_id"} <= set(ds.features):
        raise ValueError(f"{root}: missing policy or stage-language features.")
    if tuple(ds.features[STATE]["shape"]) != (20,) or tuple(ds.features[ACTION]["shape"]) != (14,):
        raise ValueError(f"{root}: expected state[20] and action[14].")
    camera_shape = tuple(ds.features[CAMERA]["shape"])
    if ds.features[CAMERA]["dtype"] != "video" or len(camera_shape) != 3 or camera_shape[2] != 3:
        raise ValueError(f"{root}: source camera2 metadata must be HWC RGB.")
    stages = _read(root / "meta/language.json")["labels"]
    stage_for_text = {}
    for stage in stages:
        for sentence in stage["instructions"]:
            if sentence in stage_for_text and stage_for_text[sentence] != stage["id"]:
                raise ValueError(f"{root}: a stage sentence has two IDs.")
            stage_for_text[sentence] = stage["id"]
    if len(stages) != 6 or {stage["id"] for stage in stages} != set(range(6)) or not stage_for_text:
        raise ValueError(f"{root}: expected six stage labels and their sentences.")
    previous = 0
    for number, episode in enumerate(ds.meta.episodes):
        start, stop = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        if (int(episode["episode_index"]) != number or start != previous
                or stop - start != int(episode["length"]) or stop <= start):
            raise ValueError(f"{root}: invalid episode {number} bounds.")
        previous = stop
    if previous != len(ds):
        raise ValueError(f"{root}: episode bounds do not cover the dataset.")
    return {"root": root, "dataset": ds, "split": split, "goal": _frozen_goal(root),
            "contract": _read(root / "meta/provenance.json")["environment_contract"],
            "stage_for_text": stage_for_text, "camera_shape": camera_shape,
            "last_attempt_seed": max(row["seed"] for row in attempts)}


def _frame(source: dict, index: int, episode: int, output_shape: tuple[int, ...], goal: str) -> dict:
    item = source["dataset"][index]
    if int(item["index"].item()) != index or int(item["episode_index"].item()) != episode:
        raise ValueError(f"{source['root']}: frame {index} crossed an episode boundary.")
    stage = int(item["language.stage_id"].item())
    if source["stage_for_text"].get(item["task"]) != stage:
        raise ValueError(f"{source['root']}: frame {index} stage ID and task text disagree.")
    state = item[STATE].cpu().numpy().astype(np.float32, copy=False)
    action = item[ACTION].cpu().numpy().astype(np.float32, copy=False)
    image = item[CAMERA].cpu().numpy()
    if (state.shape != (20,) or action.shape != (14,) or image.shape != output_shape
            or image.dtype != np.uint8 or not np.isfinite(state).all() or not np.isfinite(action).all()):
        raise ValueError(f"{source['root']}: frame {index} has invalid policy input or action.")
    return {STATE: state, CAMERA: np.ascontiguousarray(image), ACTION: action, "task": goal}


def _convert(train_root: Path, val_root: Path, destination: Path) -> None:
    train, val = _source(train_root, "train"), _source(val_root, "validation")
    if train["goal"] != val["goal"] or train["contract"] != val["contract"]:
        raise ValueError("Train and validation task descriptions or environment contracts differ.")
    if train["split"]["config"] != val["split"]["config"]:
        raise ValueError("Train and validation scene configs differ.")
    a = train["split"]["config"]["splits"]["train"]["seed_range"]
    b = val["split"]["config"]["splits"]["validation"]["seed_range"]
    if max(a[0], b[0]) < min(a[1], b[1]):
        raise ValueError("Train and validation seed ranges overlap.")
    left, right, channels = train["camera_shape"]
    if channels != 3 or val["camera_shape"] != train["camera_shape"]:
        raise ValueError("Train and validation camera shapes differ.")
    train_ds, val_ds = train["dataset"], val["dataset"]
    if train_ds.fps != val_ds.fps:
        raise ValueError("Train and validation FPS differ.")
    features = {
        STATE: {k: train_ds.features[STATE][k] for k in ("dtype", "shape", "names")},
        ACTION: {k: train_ds.features[ACTION][k] for k in ("dtype", "shape", "names")},
        CAMERA: {"dtype": "video", "shape": (3, left, right),
                 "names": ["channels", "height", "width"]},
    }
    for key in (STATE, ACTION):
        if val_ds.features[key]["names"] != features[key]["names"]:
            raise ValueError(f"Train and validation {key} names differ.")
    output = LeRobotDataset.create(repo_id=REPO_ID, root=destination, fps=train_ds.fps,
                                   robot_type="alignbox3d", features=features,
                                   use_videos=True, video_backend="pyav", image_writer_threads=4)
    try:
        for source in (train, val):
            for number, episode in enumerate(source["dataset"].meta.episodes):
                for index in range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"])):
                    output.add_frame(_frame(source, index, number, features[CAMERA]["shape"], train["goal"]))
                output.save_episode()
            if source is train:
                train_stats = deepcopy(output.meta.stats)
    finally:
        output.finalize()
    output.meta.stats = train_stats
    write_stats(train_stats, destination)
    train_count, val_count = train_ds.meta.total_episodes, val_ds.meta.total_episodes
    total = train_count + val_count
    output.meta.info.splits = {"train": f"0:{train_count}",
                               "validation": f"{train_count}:{total}"}
    write_info(output.meta.info, destination)
    _write(destination / "meta/contract.json", train["contract"])
    _write(destination / "meta/data_split.json", {"train": train["split"], "validation": val["split"]})
    _write(destination / "meta/pi05_export.json", {
        "repo_id": REPO_ID, "task": train["goal"],
        "task_source": "meta/source_snapshot/alignbox3d/env.py::AlignBox3DEnv.task_description",
        "train_episodes": train_count, "validation_episodes": val_count,
        "train_frames": len(train_ds), "validation_frames": len(val_ds),
        "validation_first_episode": train_count,
        "eval_split": (val_count - 0.5) / total,
        "source_train": str(train_root), "source_validation": str(val_root),
        "validation_last_attempt_seed": val["last_attempt_seed"],
        "stage_labels_excluded": True, "normalization_stats_scope": "train_only",
        "video_reencoded": True,
    })


def check_dataset(root: Path, *, sample_videos: bool = True) -> dict:
    """Inspect tasks and statistics; optionally decode boundary video frames."""
    _require_version()
    root = root.expanduser().resolve()
    manifest = _read(root / "meta/pi05_export.json")
    ds = LeRobotDataset(REPO_ID, root=root, return_uint8=True, video_backend="pyav",
                        tolerance_s=0.5 / _read(root / "meta/info.json")["fps"])
    features = ds.meta.features
    if set(features) != MODEL_KEYS | INDEX_KEYS:
        raise ValueError("Model dataset contains a missing or privileged feature.")
    if (tuple(features[STATE]["shape"]) != (20,) or tuple(features[ACTION]["shape"]) != (14,)
            or tuple(features[CAMERA]["shape"])[0] != 3):
        raise ValueError("Model feature dimensions are wrong.")
    train_count, val_count = manifest["train_episodes"], manifest["validation_episodes"]
    if (train_count < 1 or val_count < 1 or train_count + val_count != ds.meta.total_episodes
            or manifest["validation_first_episode"] != train_count
            or manifest["train_frames"] + manifest["validation_frames"] != len(ds)):
        raise ValueError("Exported frame or episode counts disagree with the manifest.")
    if math.ceil(ds.meta.total_episodes * manifest["eval_split"]) != val_count:
        raise ValueError("LeRobot eval_split would hold out the wrong episodes.")
    if ds.meta.info.splits != {"train": f"0:{train_count}",
                               "validation": f"{train_count}:{train_count + val_count}"}:
        raise ValueError("Episode split metadata is wrong.")
    if len(ds.meta.tasks) != 1 or ds.meta.tasks.index[0] != manifest["task"]:
        raise ValueError("Dataset task differs from the frozen high-level goal.")
    indices = ds.hf_dataset.data.column("task_index").to_numpy(zero_copy_only=False)
    if len(indices) != len(ds) or np.any(indices != 0):
        raise ValueError("Every derived frame must use task_index=0.")
    if any(row["tasks"] != [manifest["task"]] for row in ds.meta.episodes):
        raise ValueError("An episode contains a stage-specific task.")
    for key in (STATE, ACTION):
        stats = ds.meta.stats[key]
        if int(np.asarray(stats["count"])[0]) != manifest["train_frames"]:
            raise ValueError(f"{key} normalization statistics include validation data.")
        for quantile in ("q01", "q99"):
            values = np.asarray(stats[quantile])
            if values.shape != tuple(features[key]["shape"]) or not np.isfinite(values).all():
                raise ValueError(f"{key} lacks usable {quantile}.")
    if sample_videos:
        first_val = int(ds.meta.episodes[train_count]["dataset_from_index"])
        for index in (0, first_val, len(ds) - 1):
            row = ds[index]
            if (row["task"] != manifest["task"] or tuple(row[CAMERA].shape) != tuple(features[CAMERA]["shape"])
                    or str(row[CAMERA].dtype) not in ("torch.uint8", "uint8")):
                raise ValueError(f"Frame {index} is not a valid task-conditioned RGB sample.")
    contract = _read(root / "meta/contract.json")
    if (contract["obs_dim"] != 20 or contract["action_dim"] != 14 or contract["fps"] != ds.fps
            or contract["observation_names"] != features[STATE]["names"]
            or contract["action_names"] != features[ACTION]["names"]):
        raise ValueError("Environment contract and model features disagree.")
    return {"root": str(root), "task": manifest["task"], "frames": len(ds),
            "episodes": ds.meta.total_episodes,
            "train_episodes": train_count, "validation_episodes": val_count,
            "train_frames": manifest["train_frames"], "validation_frames": manifest["validation_frames"],
            "eval_split": manifest["eval_split"]}


def main() -> None:
    _require_version()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Convert two complete raw collections with LeRobot.")
    export.add_argument("--train", type=Path, required=True)
    export.add_argument("--validation", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check", help="Inspect a converted LeRobot dataset.")
    check.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps(check_dataset(args.dataset), ensure_ascii=False, indent=2))
        return
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pi05-export-", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"  # LeRobot.create requires a new root.
        _convert(args.train.expanduser().resolve(), args.validation.expanduser().resolve(), staging)
        summary = check_dataset(staging)
        os.rename(staging, output)
    summary["root"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
