"""Check and fine-tune LeRobot 0.6.1 PI05 on the fixed-task AlignBox3D dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import lerobot
from lerobot.configs import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.scripts.lerobot_train import train
import torch
from transformers import AutoTokenizer

from data import REPO_ID, check_dataset
from model import load_checked


TOKENIZER_NAME = "google/paligemma-3b-pt-224"
TOKENIZER_MAX_LENGTH = 200


def check_language(task: str) -> dict:
    """Probe PI05's exact task/state prompt before its 200-token truncation.

    PI05 normalizes the 20-D state and discretizes each element to one of the
    integer values -1..255. Probe all 257 values repeated in every position;
    a long high-level task fails before expensive policy initialization.
    """
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    if not cleaned:
        raise ValueError("The exported high-level task is empty.")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    longest = 0
    longest_bin = None
    for start in range(-1, 256, 64):
        values = range(start, min(start + 64, 256))
        prompts = [
            f"Task: {cleaned}, State: {' '.join([str(value)] * 20)};\nAction: "
            for value in values
        ]
        lengths = [
            len(ids)
            for ids in tokenizer(prompts, truncation=False, padding=False,
                                 add_special_tokens=True)["input_ids"]
        ]
        maximum = max(lengths)
        if maximum > longest:
            longest = maximum
            longest_bin = start + lengths.index(maximum)
    if longest > TOKENIZER_MAX_LENGTH:
        raise ValueError(
            f"PI05 task/state prompt reaches {longest} tokens with state bin "
            f"{longest_bin}, exceeding the {TOKENIZER_MAX_LENGTH}-token limit. "
            "Shorten the high-level task and re-export the dataset."
        )
    return {"tokenizer": TOKENIZER_NAME, "prompt_template":
            "Task: {task}, State: {20 discretized state bins};\\nAction: ",
            "max_repeated_bin_tokens": longest, "max_repeated_bin": longest_bin,
            "tokenizer_max_length": TOKENIZER_MAX_LENGTH}


def check_base_weights(model: str) -> str:
    local = Path(model).expanduser()
    if local.is_dir():
        weights = local / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Missing PI05 base weights: {weights}")
    else:
        weights = Path(hf_hub_download(repo_id=model, filename="model.safetensors"))
    if weights.stat().st_size < 1_000_000:
        raise ValueError(f"PI05 base weights are empty or incomplete: {weights}")
    config = PreTrainedConfig.from_pretrained(model)
    if config.type != "pi05":
        raise ValueError(f"Base checkpoint must be a PI05 policy, found {config.type!r}.")
    return str(weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Dataset created by data.py export.")
    parser.add_argument("--output", required=True, type=Path,
                        help="New LeRobot training output directory.")
    parser.add_argument("--pretrained", default="lerobot/pi05_base",
                        help="PI05 base model ID or local model directory.")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=2_000)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--check-only", action="store_true",
                        help="Check dataset, language and base weights without training.")
    args = parser.parse_args()

    if lerobot.__version__ != "0.6.1":
        raise RuntimeError(f"Expected LeRobot 0.6.1, found {lerobot.__version__!r}.")
    if min(args.steps, args.batch_size, args.eval_steps, args.save_freq) < 1 or args.num_workers < 0:
        parser.error("steps, batch-size, eval-steps and save-freq must be positive; "
                     "num-workers must be nonnegative.")
    if args.device.startswith("cuda") and not torch.cuda.is_available() and not args.check_only:
        raise RuntimeError("CUDA is unavailable for PI05 training.")
    if (args.dtype == "bfloat16" and args.device.startswith("cuda")
            and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
            and not args.check_only):
        raise RuntimeError("This GPU lacks bfloat16; choose --dtype float32.")

    dataset_root = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.check_only:
        raise FileExistsError(f"Training output already exists: {output}")
    dataset_check = check_dataset(dataset_root, sample_videos=args.check_only)
    language_check = check_language(dataset_check["task"])
    weights = check_base_weights(args.pretrained)
    print(json.dumps({"lerobot": lerobot.__version__, "policy": "pi05",
                      "pretrained": args.pretrained, "weights": weights,
                      "dataset": dataset_check, "language_check": language_check,
                      "output": str(output), "steps": args.steps,
                      "batch_size": args.batch_size, "device": args.device,
                      "dtype": args.dtype}, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return

    dataset = DatasetConfig(repo_id=REPO_ID, root=str(dataset_root),
                            eval_split=dataset_check["eval_split"],
                            use_imagenet_stats=False, video_backend="pyav")
    policy = PI05Config(pretrained_path=Path(args.pretrained), device=args.device,
                        dtype=args.dtype, n_action_steps=8,
                        tokenizer_max_length=TOKENIZER_MAX_LENGTH,
                        use_relative_actions=False, train_expert_only=True,
                        gradient_checkpointing=True, push_to_hub=False)
    cfg = TrainPipelineConfig(dataset=dataset, policy=policy, output_dir=output,
                              job_name="alignbox3d_pi05", steps=args.steps,
                              batch_size=args.batch_size, num_workers=args.num_workers,
                              eval_steps=args.eval_steps, save_freq=args.save_freq,
                              env_eval_freq=0, seed=args.seed)
    cfg.wandb.enable = False
    original_method = PI05Policy.__dict__["from_pretrained"]
    original_loader = PI05Policy.from_pretrained

    @classmethod
    def checked_loader(cls, *loader_args, **loader_kwargs):
        return load_checked(original_loader, *loader_args, **loader_kwargs)

    PI05Policy.from_pretrained = checked_loader
    try:
        train(cfg)
    finally:
        PI05Policy.from_pretrained = original_method


if __name__ == "__main__":
    main()
