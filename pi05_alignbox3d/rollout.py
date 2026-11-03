"""Evaluate a LeRobot pi05 checkpoint in the isolated AlignBox3D simulator.

The model process runs in the LeRobot environment.  ReposeEnv's existing
policy_interface.py remains the simulator client and reports success rates.
The task prompt comes from the training dataset, not the expert's stage labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import torch

from .wire import PROTOCOL_VERSION, PolicyServer, decode_observation, validate_contract


SUPPORTED_LEROBOT_VERSION = "0.6.1"
CAMERA_KEY = "observation.images.camera2"


def load_dataset_contract(dataset_root: Path) -> tuple[dict, str, dict]:
    """Read the exact public environment contract and the single rollout task."""
    import pandas as pd

    metadata = dataset_root / "meta"
    contract_path = metadata / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Missing {contract_path}; regenerate the pi05 dataset with its exact environment contract."
        )
    contract = validate_contract(json.loads(contract_path.read_text(encoding="utf-8")))
    info = json.loads((metadata / "info.json").read_text(encoding="utf-8"))
    if info.get("robot_type") != "alignbox3d":
        raise ValueError("The dataset must describe alignbox3d.")
    tasks = pd.read_parquet(metadata / "tasks.parquet")
    if len(tasks) != 1 or tasks["task_index"].tolist() != [0]:
        raise ValueError("Pi05 rollout requires exactly one high-level task at task_index 0.")
    task = tasks.index[0]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("The dataset's high-level task must be nonempty text.")
    if info.get("codebase_version") != "v3.0" or info.get("total_tasks") != 1:
        raise ValueError("Expected a LeRobot v3 dataset with one high-level task.")
    features = info.get("features", {})
    for key, dimension in (("observation.state", contract["obs_dim"]), ("action", contract["action_dim"])):
        feature = features.get(key, {})
        if feature.get("shape") != [dimension] or feature.get("names") != contract.get(
            "observation_names" if key == "observation.state" else "action_names"
        ):
            raise ValueError(f"Dataset and environment contract disagree at {key}.")
    if CAMERA_KEY not in features or features[CAMERA_KEY].get("dtype") != "video":
        raise ValueError(f"Dataset must contain the {CAMERA_KEY} RGB video.")
    return contract, task, features


class Pi05Session:
    """Adapt the existing policy_interface.py protocol to LeRobot pi05."""

    def __init__(self, checkpoint: Path, dataset_root: Path, device: str):
        import lerobot
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        if lerobot.__version__ != SUPPORTED_LEROBOT_VERSION:
            raise RuntimeError(
                f"This adapter targets lerobot {SUPPORTED_LEROBOT_VERSION}; found {lerobot.__version__}."
            )
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        self.contract, self.task, features = load_dataset_contract(dataset_root)
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if cfg.type != "pi05":
            raise ValueError(f"Checkpoint policy type is {cfg.type!r}, expected 'pi05'.")
        if cfg.use_relative_actions:
            raise ValueError("AlignBox3D delta EE actions require use_relative_actions=false.")
        expected_inputs = {"observation.state", CAMERA_KEY}
        if set(cfg.input_features) != expected_inputs or set(cfg.output_features) != {"action"}:
            raise ValueError("Checkpoint features must be exactly state, camera2, and action.")
        state_shape = tuple(cfg.input_features["observation.state"].shape)
        image_shape_chw = tuple(cfg.input_features[CAMERA_KEY].shape)
        action_shape = tuple(cfg.output_features["action"].shape)
        if (state_shape != (self.contract["obs_dim"],)
                or action_shape != (self.contract["action_dim"],)
                or len(image_shape_chw) != 3 or image_shape_chw[0] != 3):
            raise ValueError("Checkpoint state, image, or action shape differs from the dataset contract.")
        self.image_shape = (image_shape_chw[1], image_shape_chw[2], 3)
        if list(self.image_shape) != features[CAMERA_KEY]["shape"]:
            raise ValueError("Checkpoint image shape differs from the LeRobot dataset.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pi05 rollout needs a CUDA device.")
        cfg.device = device
        # One action per fresh image makes the rollout genuinely closed-loop.
        cfg.n_action_steps = 1
        self.policy = PI05Policy.from_pretrained(checkpoint, config=cfg)
        self.preprocess, self.postprocess = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.policy.reset()
        self.checkpoint = checkpoint
        self.action_low = np.asarray(self.contract["action_low"], dtype=np.float32)
        self.action_high = np.asarray(self.contract["action_high"], dtype=np.float32)

    def __call__(self, request: dict) -> dict:
        operation = request.get("op")
        if operation == "health":
            if request.get("protocol", PROTOCOL_VERSION) != PROTOCOL_VERSION:
                raise ValueError("IPC protocol mismatch.")
            return {
                "ok": True, "protocol": PROTOCOL_VERSION, "runtime": "model",
                "env": Path(sys.prefix).name, "environment": "alignbox3d",
                "policy_name": "pi05", "input_mode": "multimodal",
                "checkpoint": str(self.checkpoint), "contract": self.contract,
                "obs_shape": [1, self.contract["obs_dim"]],
                "action_shape": [1, self.contract["action_dim"]],
                "action_horizon": self.policy.config.chunk_size,
                "image_shape": list(self.image_shape), "image_key": CAMERA_KEY,
                "requires_instruction": True,
            }
        if operation == "reset":
            seed = request.get("seed")
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("seed must be an integer in [0, 2**63).")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.policy.reset()
            return {"ok": True}
        if operation == "predict":
            state, image = decode_observation(request["obs"], self.contract, self.image_shape)
            if state.shape != (1, self.contract["obs_dim"]) or not np.isfinite(state).all():
                raise ValueError("Pi05 observation state must be finite with shape [1, obs_dim].")
            if image.shape != (1, *self.image_shape) or image.dtype != np.uint8:
                raise ValueError("Pi05 image history must be uint8 [1, H, W, 3].")
            # The client reports the environment's generic wording. Use the exact
            # high-level prompt that every training frame carried instead.
            observation = {
                "observation.state": torch.from_numpy(state.copy()),
                CAMERA_KEY: torch.from_numpy(np.ascontiguousarray(image.transpose(0, 3, 1, 2))).float() / 255.0,
                "task": [self.task],
            }
            processed = self.preprocess(observation)
            with torch.inference_mode():
                action = self.policy.select_action(processed)
            action = self.postprocess(action)
            values = np.asarray(action.detach().cpu().float().numpy(), dtype=np.float32)
            if values.shape != (1, self.contract["action_dim"]) or not np.isfinite(values).all():
                raise RuntimeError("Pi05 returned an invalid action.")
            values = np.clip(values, self.action_low, self.action_high)
            return {"ok": True, "action": values.tolist()}
        if operation == "close":
            return {"ok": True}
        raise ValueError(f"Unsupported operation: {operation!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="LeRobot checkpoint's pretrained_model directory.")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Local high-level pi05 dataset containing meta/contract.json and meta/tasks.parquet.")
    parser.add_argument("--reposeenv-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "ReposeEnv")
    parser.add_argument("--reposeenv-python", type=Path,
                        help="Python executable from the ReposeEnv environment; default: conda run -n ReposeEnv python.")
    parser.add_argument("--data-config", type=Path,
                        help="AlignBox3D train/validation seed config; default: ReposeEnv/config/alignbox3d_data.yml.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--socket-timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path,
                        help="Metrics JSON path; default: <checkpoint>/../../inference/pi05_metrics.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.episodes, args.max_steps) < 1 or not 0 <= args.seed < 2**63 - args.episodes:
        raise ValueError("episodes/max-steps must be positive and seeds must fit int64.")
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    reposeenv_root = args.reposeenv_root.expanduser().resolve()
    data_config = (args.data_config or reposeenv_root / "config/alignbox3d_data.yml").expanduser().resolve()
    output = (args.output or checkpoint.parent.parent / "inference/pi05_metrics.json").expanduser().resolve()
    if not (reposeenv_root / "policy_interface.py").is_file():
        raise FileNotFoundError(reposeenv_root / "policy_interface.py")
    if not data_config.is_file():
        raise FileNotFoundError(data_config)
    if args.reposeenv_python is not None and not args.reposeenv_python.is_file():
        raise FileNotFoundError(args.reposeenv_python)
    session = Pi05Session(checkpoint, dataset_root, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PolicyServer(session, args.host, args.port, args.socket_timeout) as server:
        errors: list[BaseException] = []

        def serve() -> None:
            try:
                server.serve()
            except BaseException as error:
                errors.append(error)
                server.close()

        thread = threading.Thread(target=serve, name="pi05-policy-server", daemon=True)
        thread.start()
        client_python = ([str(args.reposeenv_python.resolve())] if args.reposeenv_python
                         else ["conda", "run", "--no-capture-output", "-n", "ReposeEnv", "python"])
        host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        command = [*client_python, str(reposeenv_root / "policy_interface.py"),
                   "--env", "alignbox3d", "--obs", "image", "--split", "validation",
                   "--data-config", str(data_config), "--episodes", str(args.episodes),
                   "--seed", str(args.seed), "--max-steps", str(args.max_steps),
                   "--host", host, "--port", str(server.port),
                   "--socket-timeout", str(args.socket_timeout), "--output", str(output)]
        try:
            subprocess.run(command, cwd=reposeenv_root, check=True)
        finally:
            server.close()
            thread.join(timeout=5)
        if errors:
            raise RuntimeError("Pi05 policy server failed.") from errors[0]
    report = json.loads(output.read_text(encoding="utf-8"))
    report["checkpoint"] = str(checkpoint)
    report["dataset_root"] = str(dataset_root)
    report["task"] = session.task
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pi05 validation report: {output}", flush=True)


if __name__ == "__main__":
    main()
