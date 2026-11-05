# AlignBox3D × LeRobot π₀.₅

独立于 ActionExpert 的六脚本项目：`environment.py` 接 ReposeEnv，`data.py` 接示范数据，`model.py` 跑 π₀.₅，`train.py` 调用 LeRobot 官方训练入口，`validate.py` 做闭环验证，`record.py` 录制模型看到的 camera2 视频。**服务器的部署、原始训练/验证数据生成、导出、训练与验证 recipe 全部在本文件。**

**语言约定：**ReposeEnv 原始示范的 `task` 是专家逐帧阶段句子，不能直接用于标准 LeRobot π₀.₅ 的固定高层目标推理。`data.py` 从采集时保存的环境源码提取 `task_description`，核对原始每帧阶段句子与阶段 ID，再导出所有帧都使用该高层目标的新数据集。原始阶段标签留在原始目录。训练、离线验证和闭环演示都使用导出数据中的同一句 `task`；服务端发现环境传来不同句子会报错。这个项目训练的是**高层目标 → 动作**，不会预测或输出子任务文本。

**动作约定：**`action[14]` 已经包含末端 XYZ/RPY 增量和夹爪绝对开度。保持 `use_relative_actions=False`；LeRobot 的可选相对动作变换会再做 `action - state`，而这里的 `state[20]` 与 `action[14]` 不同维、不同语义。π₀.₅ 使用当前一帧观察，预测 50 步，闭环默认执行前 8 步后再观察。

## 1. 部署两个项目，设置路径

在服务器上使用 Bash，**按顺序在同一个会话执行**。先把本次更新的 `Pi05AlignBox3D-server-bundle-20261105.tar.gz` 上传到 `/srv/`；旧包没有四种横向抵箱错误和一致姿态抓取。新包含 Pi05 的七个文件及更新后的 ReposeEnv 源码、机器人资源和 `config/alignbox3d_data.yml`。把 `INSTALL_ROOT` 改为服务器上尚不存在的目录；`RUN_ROOT` 默认放在 Pi05 下，若 `/srv` 空间不足，执行前改成大容量磁盘上的新目录。已有原始数据时，再把 `TRAIN_RAW`、`VAL_RAW` 改成原始数据目录。**不要把当前正在训练的 ReposeEnv 目录覆盖掉。**

```bash
set -euo pipefail
BUNDLE=/srv/Pi05AlignBox3D-server-bundle-20261105.tar.gz
INSTALL_ROOT=/srv/pi05_alignbox3d_run01
test -f "$BUNDLE"
test ! -e "$INSTALL_ROOT"
mkdir -p "$INSTALL_ROOT"
tar -xzf "$BUNDLE" -C "$INSTALL_ROOT"
REPOSE_ROOT="$INSTALL_ROOT/ReposeEnvPi05"
PROJECT_ROOT="$INSTALL_ROOT/Pi05AlignBox3D"
TRAIN_N=200
VAL_N=24
CONTACT_ERROR_FRACTION=0.20
GRASP_READY_FRACTION=0.20
EVAL_N=10
RUN_ROOT="$PROJECT_ROOT/runs/$(date +%Y%m%d-%H%M%S)"
TRAIN_RAW="$RUN_ROOT/raw_train"
VAL_RAW="$RUN_ROOT/raw_validation"
PI05_DATA="$RUN_ROOT/lerobot_pi05"
TRAIN_OUTPUT="$RUN_ROOT/pi05_output"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
for file in README.md environment.py data.py model.py train.py validate.py record.py; do
  test -f "$PROJECT_ROOT/$file"
done
test -f "$REPOSE_ROOT/environment.yml"
test -f "$REPOSE_ROOT/alignbox3d/collect.py"
test -f "$REPOSE_ROOT/policy_interface.py"
test -f "$REPOSE_ROOT/config/alignbox3d.yml"
test -f "$REPOSE_ROOT/config/alignbox3d_data.yml"
test -f "$REPOSE_ROOT/alignbox3d/native/assets/arx_l5/arx_l5.xml"
source "$(conda info --base)/etc/profile.d/conda.sh"
python -m json.tool "$REPOSE_ROOT/config/alignbox3d.yml" > /dev/null
python -m json.tool "$REPOSE_ROOT/config/alignbox3d_data.yml" > /dev/null
export MUJOCO_GL=egl
df -h "$RUN_ROOT"
printf 'PROJECT_ROOT=%s\nREPOSE_ROOT=%s\nRUN_ROOT=%s\nTRAIN_RAW=%s\nVAL_RAW=%s\nPI05_DATA=%s\nTRAIN_OUTPUT=%s\nEVAL_N=%s\n' \
  "$PROJECT_ROOT" "$REPOSE_ROOT" "$RUN_ROOT" "$TRAIN_RAW" "$VAL_RAW" "$PI05_DATA" "$TRAIN_OUTPUT" "$EVAL_N" \
  | tee "$RUN_ROOT/paths.txt"
```

ReposeEnv 的 `.yml` 配置文件沿用旧后缀，内容实际采用 JSON 语法，程序用 `json.load` 读取。请保持 JSON 写法；普通 YAML 缩进语法不能直接使用。`alignbox3d.yml` 定义观察、动作和频率，`alignbox3d_data.yml` 定义场景与训练/验证划分。采集时会将配置写入原始数据的源码快照，并将数据划分配置写入 `meta/data_split.json`；`validate.py` 会比对当前场景配置与导出数据保存的配置。交付包已包含本地尚未被 Git 跟踪的 `alignbox3d_data.yml`。

## 2. 安装并预检 ReposeEnv

若 Conda 环境 `ReposeEnv` 不存在，按交付源码的 `environment.yml` 创建；存在则沿用。渲染预检应输出 `state (20,)`、`camera (154, 256, 3)`；它只重置一个验证场景，不写数据集。若服务器 EGL 不可用，先解决 MuJoCo 离屏渲染，再采集数据。

```bash
if ! conda env list --json | python -c 'import json,sys; sys.exit(0 if any(p.rstrip("/").split("/")[-1] == "ReposeEnv" for p in json.load(sys.stdin)["envs"]) else 1)'; then
  conda env create -f "$REPOSE_ROOT/environment.yml"
fi
conda activate ReposeEnv
cd "$REPOSE_ROOT"
python -c 'import mujoco, numpy, cv2, pyarrow; print("ReposeEnv dependencies OK")'
python -c 'import importlib.util; assert importlib.util.find_spec("torch") is None, "ReposeEnv must not contain PyTorch"'
python - <<'PY'
from environment import make_environment
env = make_environment("alignbox3d", render_size=256, split="validation")
try:
    state = env.reset(900000)
    image = env.lerobot_observation()["observation.images.camera2"]
    assert state.shape == (20,) and image.shape == (154, 256, 3)
    print("state", state.shape, "camera", image.shape, image.dtype)
finally:
    env.close()
PY
```

## 3. 生成训练和验证原始示范

默认生成 200 条训练成功示范和 24 条验证成功示范。在同一个原始目录中混合完整正常抓取、错误抵箱恢复和抓取就绪轨迹，尝试概率为 60%/20%/20%；由于只保存成功轨迹，最终比例以采集报告为准。恢复轨迹让两只夹爪各自偏到箱壁外侧或内侧，覆盖外/外、内/内、外/内、内/外四种组合；从已测得接触、且箱壁不在任何一只夹爪的两指之间的状态开始。抓取就绪轨迹从箱沿已在两指之间、下一步直接闭合的状态开始，近垂直和倾斜夹持方向各占约一半。预定位和制造错误的动作不进入训练。已有原始目录时，在第 1 步将 `TRAIN_RAW`、`VAL_RAW` 指向它们，下面的条件判断会跳过对应采集。原始目录应有 `meta/collection.json`、`meta/data_split.json`、`meta/language.json` 和 `meta/source_snapshot/alignbox3d/env.py`。旧数据缺少新类型，不能复用；若目录存在但不完整，直接停止并报告，**不要删除或覆盖**。

```bash
cd "$REPOSE_ROOT"
if test -e "$TRAIN_RAW"; then
  test -f "$TRAIN_RAW/meta/collection.json"
else
  python -m alignbox3d.collect --split train --episodes "$TRAIN_N" \
    --max-attempts "$((5 * TRAIN_N))" --workers 4 --render-size 256 \
    --recovery-fraction 0 --contact-error-fraction "$CONTACT_ERROR_FRACTION" --grasp-ready-fraction "$GRASP_READY_FRACTION" \
    --data-config "$REPOSE_ROOT/config/alignbox3d_data.yml" --output "$TRAIN_RAW"
fi
if test -e "$VAL_RAW"; then
  test -f "$VAL_RAW/meta/collection.json"
else
  python -m alignbox3d.collect --split validation --episodes "$VAL_N" \
    --max-attempts "$((5 * VAL_N))" --workers 4 --render-size 256 \
    --recovery-fraction 0 --contact-error-fraction "$CONTACT_ERROR_FRACTION" --grasp-ready-fraction "$GRASP_READY_FRACTION" \
    --data-config "$REPOSE_ROOT/config/alignbox3d_data.yml" --output "$VAL_RAW"
fi
python - "$TRAIN_RAW" "$VAL_RAW" <<'PY'
import json, sys
from pathlib import Path
for path_text, expected in zip(sys.argv[1:], ("train", "validation")):
    path = Path(path_text)
    report = json.loads((path / "meta/collection.json").read_text())
    split = json.loads((path / "meta/data_split.json").read_text())
    assert report["complete"] is True and report["saved_episodes"] == report["requested_episodes"]
    assert report["split"] == split["split"] == expected
    modes = report["saved_by_recovery_stage"]
    assert "grasp_ready" in modes and "saved_by_contact_pattern" in report, f"Old collector output; regenerate {path}"
    patterns = report["saved_by_contact_pattern"]
    assert modes["none"] > 0 and modes["contact_error"] > 0 and modes["grasp_ready"] > 0, (path, modes)
    if expected == "train":
        assert all(patterns.values()), (path, patterns)
    print(expected, "saved", report["saved_episodes"], "attempts", report["attempts"], "modes", modes, "patterns", patterns, path)
PY
```

采集时只保存成功轨迹；失败尝试及原因仍留在 `meta/collection.json`。训练和验证使用配置中互不重叠的 seed 范围。两份原始数据保留逐帧阶段文字和阶段 ID，不直接交给 π₀.₅ 训练。`data.py` 导出时核对新增的第 7 阶段“重新抓取”，然后将所有阶段句子统一替换为固定的高层任务指令。完整正常、错误恢复和抓取就绪轨迹必须一起训练。

## 4. 安装 LeRobot，导出并检查数据

LeRobot 0.6.1 单独使用 Python 3.12。先在 [PaliGemma 页面](https://huggingface.co/google/paligemma-3b-pt-224)接受 tokenizer 许可，且服务器已登录有权访问的 Hugging Face 账号。若 `hf auth whoami` 或后续权重预检失败，停止并让账号持有人处理授权。Pi05 环境不得与 ReposeEnv 环境混用。

```bash
if ! conda env list --json | python -c 'import json,sys; sys.exit(0 if any(p.rstrip("/").split("/")[-1] == "lerobot-pi05" for p in json.load(sys.stdin)["envs"]) else 1)'; then
  conda create -y -n lerobot-pi05 python=3.12
fi
conda activate lerobot-pi05
conda install -y -c conda-forge ffmpeg
python -m pip install 'lerobot[pi,training]==0.6.1'
hf auth whoami
python -c 'import lerobot, torch, av; print("LeRobot:", lerobot.__version__, "CUDA:", torch.cuda.is_available(), "bf16:", torch.cuda.is_bf16_supported()); assert lerobot.__version__ == "0.6.1"'
cd "$PROJECT_ROOT"
if test -e "$PI05_DATA"; then
  test -f "$PI05_DATA/meta/pi05_export.json"
else
  python data.py export --train "$TRAIN_RAW" --validation "$VAL_RAW" --output "$PI05_DATA"
fi
python data.py check --dataset "$PI05_DATA"
```

`data.py` 用 LeRobot 官方 `LeRobotDataset` 读写，修正相机 metadata 为 LeRobot 使用的 CHW 形状，按训练在前、验证在后合并 episode，并只用训练帧写入归一化统计量。`check` 应显示训练/验证 episode 数与原始采集一致、一个高层任务、全部帧 `task_index=0`，且 state/action 均有 `q01/q99`。LeRobot v0.6.1 按任务将**最后**的 episode 留作离线验证，和这里的顺序一致。视频由 LeRobot 重新编码，RGB 像素可能略有差别；数值动作与任务时序保持原义。

## 5. 预检并训练 π₀.₅

默认从 `lerobot/pi05_base` 后训练。`train.py` 在 Python 里构造 LeRobot 的 `DatasetConfig`、`PI05Config`、`TrainPipelineConfig` 并调用官方 `train(cfg)`。预检下载/核对基础权重文件与配置，并用 π₀.₅ 的 PaliGemma tokenizer 检查完整任务提示词未被 200 token 截断；**完整权重键核对发生在正式训练启动时**。先前 3000 步只够验证训练流程，闭环出现长期停在接近阶段；下面改为 20000 步、每 2000 步评估并每 5000 步保存 checkpoint。默认需要 CUDA 与 bfloat16；若 GPU 不支持 bfloat16，可在两个命令都加 `--dtype float32`，并先确认显存。`TRAIN_OUTPUT` 必须尚不存在；已有完整训练 checkpoint 时，跳过训练并在第 6 步设置 `CHECKPOINT`。

```bash
df -h "$RUN_ROOT"
python -c 'import torch; assert torch.cuda.is_available(), "Pi05 training needs a visible CUDA GPU"'
python -c 'import torch; print("GPU:", torch.cuda.get_device_name(0), "bf16:", torch.cuda.is_bf16_supported())'
python train.py --dataset "$PI05_DATA" --output "$TRAIN_OUTPUT" --check-only
test ! -e "$TRAIN_OUTPUT"
python train.py --dataset "$PI05_DATA" --output "$TRAIN_OUTPUT" \
  --steps 20000 --batch-size 1 --num-workers 2 \
  --eval-steps 2000 --save-freq 5000 \
  2>&1 | tee "$RUN_ROOT/train.log"
grep -E 'Train/eval split:|All keys loaded successfully!|eval_loss=' "$RUN_ROOT/train.log"
```

日志应显示 `Train/eval split: 200 train, 24 eval`（或你设置的数量）、`All keys loaded successfully!` 及每 2000 步的 `eval_loss=`。若权重键未完整加载，代码会退出。`eval_loss` 是离线动作损失，不是任务成功率。`use_relative_actions=False` 已固定在 [train.py](train.py)；不要照搬官方相对动作示例的 `true`。

## 6. 闭环验证并录制视频

验证脚本自动从原始验证采集的最后一次**尝试** seed 之后选新 seed，避免重复采集时的初始场景；仍属于相同的验证外观组。π₀.₅ 每次预测 50 步动作块，默认把前 8 步作为 `[8, 14]` 发给环境；环境逐步执行，随后用新观察再次预测。可用 `--action-steps` 选择 1 到 50 步。每条录像是模型每次预测时收到的 camera2 帧，录像帧率不代表环境控制帧率。

```bash
CHECKPOINT="$TRAIN_OUTPUT/checkpoints/last/pretrained_model"
test -s "$CHECKPOINT/model.safetensors"
test -f "$CHECKPOINT/policy_preprocessor.json"
test -f "$CHECKPOINT/policy_postprocessor.json"
python validate.py --checkpoint "$CHECKPOINT" --dataset "$PI05_DATA" \
  --reposeenv "$REPOSE_ROOT" --episodes "$EVAL_N" --max-steps 3000 \
  --action-steps 8 --output "$RUN_ROOT/validation.json" --videos "$RUN_ROOT/videos"
cat "$RUN_ROOT/validation.json"
```

查看 `success_rate`、每条 seed/success/steps、`videos/seed-*.mp4`，以及自动生成的 `validation.trace.jsonl`。轨迹逐行刷新：`prediction_request` 保存模型请求前的 20 维状态和高层指令，`prediction` 保存返回的 `[8,14]` 动作块，`action` 保存每步实际动作、下一状态、奖励和成功标志，`episode_end` 保存结果。若模型在预测时报错，最后一条请求仍留在文件中。用 `head -n 3 "$RUN_ROOT/validation.trace.jsonl"` 查看前几条，用 `tail -n 20` 查看故障前后。录像只记录模型每次收到的 camera2 帧，不是每个执行步的完整视频。10 条只是快速检查；正式结论需要更多未采集的验证场景。单一固定任务的结果只能证明这个任务的模仿与场景泛化，不能证明语言指令切换或阶段预测能力。

```bash
test -s "$RUN_ROOT/validation.trace.jsonl"
python - "$RUN_ROOT/validation.json" "$EVAL_N" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["episodes"] == len(report["results"]) == int(sys.argv[2])
print("success_rate", report["success_rate"], "action_steps", report["action_steps"])
for row in report["results"]:
    print(row["seed"], "success", row["success"], "steps", row["steps"], "scene", row["scene"]["visual_case"])
PY
```

## 发回给我的结果

请提供 `data.py check`、`train.py --check-only` 的完整输出；LeRobot 版本、GPU 型号和剩余磁盘空间；训练日志里的验证划分、权重加载信息、全部 `eval_loss=` 与最后 30 行；`validation.json` 全文、失败 seed 对应的录像及轨迹 JSONL 中该 seed 的记录。任一步出错时，发执行命令和首个完整 traceback，并停止后续步骤。训练输出目录创建后不能用同一路径从头重跑；先报告错误，再选新输出目录或制定续训方案。

参考：[LeRobot 数据集 API](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/lerobot_dataset.py)、[π₀.₅ 训练入口](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_train.py)、[π₀.₅ 语言处理](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/processor_pi05.py)。
