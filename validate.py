"""Run PI05 closed-loop in new AlignBox3D validation scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import threading

from data import check_dataset
from environment import Bridge
from model import Pi05Model
from record import Recorder


def _new_validation_seed(dataset_root: Path, count: int, config: dict) -> int:
    manifest = json.loads((dataset_root / "meta/pi05_export.json").read_text())
    seed = manifest["validation_last_attempt_seed"] + 1
    start, stop = config["splits"]["validation"]["seed_range"]
    if seed < start or seed + count > stop:
        raise ValueError("Not enough untouched validation seeds for this rollout")
    return seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reposeenv", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-steps", type=int, default=8,
                        help="Actions executed before requesting a new observation (1..chunk_size).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=Path, help="Directory for one camera2 MP4 per seed.")
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.action_steps) < 1:
        parser.error("episodes, max-steps and action-steps must be positive")
    dataset_root = args.dataset.expanduser().resolve()
    reposeenv = args.reposeenv.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    trace_path = output.with_suffix(".trace.jsonl")
    if trace_path.exists():
        raise FileExistsError(f"Validation trace already exists: {trace_path}")
    if not (reposeenv / "policy_interface.py").is_file():
        raise FileNotFoundError(reposeenv / "policy_interface.py")
    data_config = reposeenv / "config/alignbox3d_data.yml"
    config = json.loads(data_config.read_text())
    collected = json.loads((dataset_root / "meta/data_split.json").read_text())
    if config != collected["validation"]["config"]:
        raise ValueError("Current validation scene config differs from the dataset's source")
    check_dataset(dataset_root)
    seed = _new_validation_seed(dataset_root, args.episodes, config)
    model = Pi05Model(checkpoint, dataset_root, args.device, args.action_steps)
    recorder = Recorder(args.videos.expanduser().resolve(), model.contract["fps"]) if args.videos else None
    bridge = Bridge(model, recorder)
    thread = threading.Thread(target=bridge.serve, name="pi05-environment-bridge", daemon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    thread.start()
    command = ["conda", "run", "--no-capture-output", "-n", "ReposeEnv", "python",
               str(reposeenv / "policy_interface.py"), "--env", "alignbox3d",
               "--obs", "image", "--split", "validation",
               "--data-config", str(data_config), "--episodes", str(args.episodes),
               "--seed", str(seed), "--max-steps", str(args.max_steps),
               "--host", "127.0.0.1", "--port", str(bridge.port),
               "--socket-timeout", "300", "--output", str(output)]
    command.extend(["--trace", str(trace_path)])
    try:
        subprocess.run(command, cwd=reposeenv, check=True)
    finally:
        bridge.close()
        thread.join(timeout=5)
        if recorder is not None:
            recorder.close()
    if bridge.error is not None:
        raise RuntimeError("Environment bridge failed") from bridge.error
    report = json.loads(output.read_text())
    report.update({"task": model.task, "checkpoint": str(checkpoint),
                   "dataset": str(dataset_root), "new_validation_seed": seed,
                   "action_steps": model.action_steps, "model_chunk_size": model.chunk_size,
                   "videos": str(args.videos.resolve()) if args.videos else None})
    report["trace"] = str(trace_path)
    output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"success_rate": report["success_rate"], "seed": seed,
                      "episodes": args.episodes, "report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
