"""LeRobot PI05 checkpoint adapter for AlignBox3D closed-loop evaluation."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path

import lerobot
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import numpy as np
import torch

from data import check_dataset


CAMERA = "observation.images.camera2"


def load_checked(loader, *args, **kwargs):
    """Fail if LeRobot silently keeps random PI05 weights after a load error."""
    messages = StringIO()
    try:
        with redirect_stdout(messages):
            policy = loader(*args, **kwargs)
    finally:
        print(messages.getvalue(), end="", flush=True)
    if "All keys loaded successfully!" not in messages.getvalue():
        raise RuntimeError("PI05 weights did not load completely; see loader messages above.")
    return policy


class Pi05Model:
    """Use the dataset's one high-level task for every policy prediction."""

    def __init__(self, checkpoint: Path, dataset_root: Path, device: str = "cuda",
                 action_steps: int = 8) -> None:
        if lerobot.__version__ != "0.6.1":
            raise RuntimeError(f"Expected LeRobot 0.6.1, found {lerobot.__version__!r}.")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        weights = self.checkpoint / "model.safetensors"
        if not weights.is_file() or weights.stat().st_size < 1_000_000:
            raise FileNotFoundError(f"Missing or incomplete PI05 checkpoint weights: {weights}")
        checked = check_dataset(self.dataset_root)
        self.task = checked["task"]
        self.contract = json.loads((self.dataset_root / "meta/contract.json").read_text(encoding="utf-8"))
        info = json.loads((self.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
        features = info["features"]
        if self.contract.get("obs_dim") != 20 or self.contract.get("action_dim") != 14:
            raise ValueError("AlignBox3D requires a 20-D state and 14-D action.")
        if (features["observation.state"]["shape"] != [20]
                or features["action"]["shape"] != [14]
                or features[CAMERA]["dtype"] != "video"):
            raise ValueError("Dataset policy features disagree with AlignBox3D.")
        camera_shape = tuple(features[CAMERA]["shape"])
        if len(camera_shape) != 3 or camera_shape[0] != 3:
            raise ValueError("LeRobot camera metadata must use CHW RGB shape.")
        self.image_shape = (camera_shape[1], camera_shape[2], 3)
        self.action_low = np.asarray(self.contract["action_low"], dtype=np.float32)
        self.action_high = np.asarray(self.contract["action_high"], dtype=np.float32)
        if self.action_low.shape != (14,) or self.action_high.shape != (14,):
            raise ValueError("Dataset action bounds must have 14 values.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for PI05 inference.")

        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        if config.type != "pi05" or config.use_relative_actions:
            raise ValueError("Checkpoint must be PI05 with use_relative_actions=False.")
        if (set(config.input_features) != {"observation.state", CAMERA}
                or set(config.output_features) != {"action"}
                or tuple(config.input_features["observation.state"].shape) != (20,)
                or tuple(config.input_features[CAMERA].shape) != camera_shape
                or tuple(config.output_features["action"].shape) != (14,)):
            raise ValueError("Checkpoint policy features disagree with the exported dataset.")
        if not 1 <= action_steps <= config.chunk_size:
            raise ValueError(f"action_steps must be in [1, {config.chunk_size}].")
        config.device = device
        config.n_action_steps = action_steps
        self.config = config
        self.chunk_size = int(config.chunk_size)
        self.action_steps = action_steps
        self.policy = load_checked(PI05Policy.from_pretrained, self.checkpoint, config=config)
        self.policy.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    def reset(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.policy.reset()

    def predict(self, state: np.ndarray, image: np.ndarray, instruction: str) -> np.ndarray:
        if instruction != self.task:
            raise ValueError(
                "Environment instruction differs from the training task: "
                f"{instruction!r} != {self.task!r}."
            )
        state = np.asarray(state, dtype=np.float32)
        image = np.asarray(image)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("PI05 observation.state must be finite with shape (20,).")
        if image.shape != self.image_shape or image.dtype != np.uint8:
            raise ValueError(f"PI05 camera must be uint8 HWC {self.image_shape}.")
        observation = {
            "observation.state": torch.from_numpy(np.ascontiguousarray(state[None])),
            CAMERA: torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)[None])).float() / 255.0,
            "task": [self.task],
        }
        processed = self.preprocess(observation)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(processed)[:, :self.action_steps]
        actions = self.postprocess(actions)
        values = np.asarray(actions.detach().cpu().float().numpy(), dtype=np.float32)
        if values.shape != (1, self.action_steps, 14) or not np.isfinite(values).all():
            raise RuntimeError("PI05 returned an invalid action chunk.")
        return np.clip(values[0], self.action_low, self.action_high)
