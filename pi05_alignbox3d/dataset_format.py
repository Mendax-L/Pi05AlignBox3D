"""Small read-only checker for the ReposeEnv LeRobot v3 datasets used here."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODE_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
BASE_FEATURES = {"observation.state", "action", "timestamp", "frame_index",
                 "episode_index", "index", "task_index"}


def validate_dataset(root: str | Path) -> dict:
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0" or not BASE_FEATURES <= set(info["features"]):
        raise ValueError(f"{root}: missing LeRobot v3 base features.")
    data_files = sorted((root / "data").rglob("*.parquet"))
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not data_files or not episode_files:
        raise ValueError(f"{root}: missing data or episode files.")
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    required = {key for key, feature in info["features"].items() if feature["dtype"] != "video"}
    frame_count = 0
    frame_tasks = []
    for path in data_files:
        table = pq.read_table(path)
        if not required <= set(table.column_names):
            raise ValueError(f"{path}: missing numeric features {sorted(required - set(table.column_names))}.")
        indices = table["index"].to_numpy()
        if len(indices) and (indices[0] != frame_count or not np.array_equal(
                indices, np.arange(frame_count, frame_count + len(indices)))):
            raise ValueError(f"{path}: global frame indices are not contiguous.")
        frame_count += len(table)
        frame_tasks.extend(table["task_index"].to_pylist())
    episodes = [row for path in episode_files for row in pq.read_table(path).to_pylist()]
    if (frame_count != info["total_frames"] or len(episodes) != info["total_episodes"]
            or not episodes or episodes[0]["dataset_from_index"] != 0
            or episodes[-1]["dataset_to_index"] != frame_count):
        raise ValueError(f"{root}: data and episode totals disagree.")
    for episode_number, row in enumerate(episodes):
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        if (row["episode_index"] != episode_number or stop - start != row["length"]
                or (episode_number and start != episodes[episode_number - 1]["dataset_to_index"])):
            raise ValueError(f"{root}: episode {episode_number} has invalid bounds.")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if (len(tasks) != info["total_tasks"] or list(tasks.columns) != ["task_index"]
            or tasks["task_index"].tolist() != list(range(len(tasks)))
            or not tasks.index.is_unique):
        raise ValueError(f"{root}: task metadata is invalid.")
    if any(type(index) is not int or not 0 <= index < len(tasks) for index in frame_tasks):
        raise ValueError(f"{root}: frame task index is invalid.")
    for row in episodes:
        expected = {int(tasks.loc[task, "task_index"]) for task in row["tasks"]}
        if set(frame_tasks[row["dataset_from_index"]:row["dataset_to_index"]]) != expected:
            raise ValueError(f"{root}: episode task labels disagree with frames.")
    video_ends = {}
    for row in episodes:
        for key in video_keys:
            path = root / info["video_path"].format(
                video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{root}: missing or empty video {path}.")
            start, stop = (row[f"videos/{key}/{side}_timestamp"] for side in ("from", "to"))
            if (not np.isfinite([start, stop]).all() or start < 0
                    or not np.isclose(stop - start, row["length"] / info["fps"], atol=1e-4)
                    or not np.isclose(start, video_ends.get((key, path), 0.0), atol=1e-4)):
                raise ValueError(f"{root}: video timestamps disagree with episode length.")
            video_ends[(key, path)] = stop
    for (key, path), stop in video_ends.items():
        with imageio.get_reader(path) as reader:
            metadata = reader.get_meta_data()
            height, width, _ = info["features"][key]["shape"]
            if (metadata["size"] != (width, height)
                    or not np.isclose(metadata["fps"], info["fps"], atol=1e-3)
                    or reader.count_frames() != round(stop * info["fps"])):
                raise ValueError(f"{root}: video dimensions, FPS or frame count differ: {path}.")
    return {"root": str(root.resolve()), "frames": frame_count, "episodes": len(episodes),
            "tasks": len(tasks), "data_files": len(data_files), "video_files": len(video_ends)}
