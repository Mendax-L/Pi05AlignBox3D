# AlignBox3D → LeRobot π₀.₅：服务器执行 recipe

本文针对 **LeRobot 0.6.1 标准 `pi05` 动作微调**、Linux 服务器和 ReposeEnv 的 AlignBox3D 自动示范。每帧给模型的 `task` 都是同一句高层指令；接近、夹紧、抬起、搬运、放下、松开等阶段由观测和动作学习，不把采集专家的阶段标签当成推理时可用的输入。这是单任务模仿学习设置，不会单独训练或输出子任务文本。它与[原论文图 3](https://www.physicalintelligence.company/download/pi05.pdf)所示的“先预测子任务文本、再预测动作”的完整两级流程不同。**原始采集数据中的逐帧阶段文字、`language.stage_id` 和专家阶段元数据完全保留**，可供以后训练高层预测器；只有本 recipe 的派生模型数据集将它们排除出输入。导出只改变数据呈现，不改变仿真、专家控制器或动作的物理语义。

## 0. 先确认路径和环境

以下命令在 Bash 中执行。将前两行改成服务器上 `ReposeEnv` 和本项目的真实绝对路径。`TRAIN_N=100`、`VAL_N=12` 是一次试跑规模，后续可以增加，但两者必须都大于零。训练集和验证集使用同一份 `config/alignbox3d_data.yml`；采集器会从互不重叠的 seed 范围生成场景，并排除训练集中的验证保留外观组合。

```bash
set -euo pipefail
REPOSE_ROOT=/srv/ReposeEnv
PROJECT_ROOT=/srv/Pi05AlignBox3D
TRAIN_N=100
VAL_N=12
WORKERS=4
RUN_ID=$(date +%Y%m%d-%H%M%S)
RUN_ROOT="$REPOSE_ROOT/training_data/pi05_runs/$RUN_ID"
TRAIN_RAW="$RUN_ROOT/raw_train"
VAL_RAW="$RUN_ROOT/raw_validation"
PI05_DATA="$RUN_ROOT/lerobot_pi05"
mkdir -p "$RUN_ROOT"
cd "$REPOSE_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ReposeEnv
```

`raw_train`、`raw_validation` 和 `lerobot_pi05` 必须是尚不存在的目录。每次使用新的 `RUN_ID`，避免把旧数据混进新数据。不要修改采集的阶段或速度参数；下面调用仓库默认专家。

## 1. 采集原始成功示范

```bash
python -m alignbox3d.collect \
  --split train --episodes "$TRAIN_N" --max-attempts "$((5 * TRAIN_N))" \
  --workers "$WORKERS" --data-config config/alignbox3d_data.yml \
  --output "$TRAIN_RAW"

python -m alignbox3d.collect \
  --split validation --episodes "$VAL_N" --max-attempts "$((5 * VAL_N))" \
  --workers "$WORKERS" --data-config config/alignbox3d_data.yml \
  --output "$VAL_RAW"
```

两条命令必须都正常退出。`--episodes` 指**保存成功轨迹的条数**；失败尝试记录在各自的 `meta/collection.json`，不会作为训练示范。若 `max-attempts` 用完导致命令失败，先检查该文件，不要继续导出不完整数据。

## 2. 生成 π₀.₅ 数据集并检查

导出器按参数顺序写入 **训练轨迹在前、验证轨迹在后** 的一个 LeRobot 数据集。它把原始每帧阶段指令换成固定的高层任务文字，保留原始 `observation.state`、`observation.images.camera2` 视频帧、`action` 和 episode 边界；其他相机及采集元数据不进入这个模型数据集。

```bash
PYTHONPATH="$PROJECT_ROOT" python -m pi05_alignbox3d.export \
  --train "$TRAIN_RAW" --validation "$VAL_RAW" \
  --output "$PI05_DATA" \
  --task '请抓取上箱，抬起并与下箱对齐，然后平稳放下、松开夹爪并撤离。'

PYTHONPATH="$PROJECT_ROOT" python -m pi05_alignbox3d.export --check "$PI05_DATA"
```

检查输出中的 `episodes` 必须为 `TRAIN_N + VAL_N`，`validation_episodes` 必须为 `VAL_N`，两者相减必须为 `TRAIN_N`；每帧任务文字必须相同，数值必须有限，视频文件、尺寸、帧数、时间戳与 episode 边界必须一致。导出器另存 `meta/contract.json`，供后续接入闭环推理时核对状态和动作语义。下面的 LeRobot 读取会实际解码训练集首帧和验证集末帧。保留原始目录供排错，后续训练只读取 `PI05_DATA`。

## 3. 安装与验证 LeRobot 运行环境

`ReposeEnv` 使用 Python 3.10；LeRobot 0.6.1 使用独立的 Python 3.12 环境。不要在正在训练的环境里升级包。π₀.₅ 需要 Hugging Face 上的 `google/paligemma-3b-pt-224` tokenizer 授权；在浏览器接受其许可后运行 `hf auth login`。

```bash
conda create -y -n lerobot-pi05 python=3.12
conda activate lerobot-pi05
conda install -y -c conda-forge ffmpeg
python -m pip install 'lerobot[training,pi]==0.6.1'
python -m pip show lerobot
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("bfloat16 supported:", torch.cuda.is_bf16_supported())'
hf auth login
```

`pip show` 应显示 `Version: 0.6.1`，CUDA 和 bfloat16 支持均应显示 `True`。若不支持 bfloat16，先根据服务器 GPU 调整 `--policy.dtype`，不要原样启动下面的训练命令。先用 LeRobot **自己的读取器**解码训练和验证来源的各一帧，避免等到长训练的第一批才发现格式问题：

```bash
python - "$PI05_DATA" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset('local/alignbox3d_pi05', root=sys.argv[1])
assert len(ds) > 0
first = ds[0]
last = ds[len(ds) - 1]
for row in (first, last):
    assert row['observation.state'].shape[-1] == 20
    assert row['action'].shape[-1] == 14
    assert isinstance(row['task'], str) and row['task']
    assert 'observation.images.camera2' in row
assert first['task'] == last['task']
print('episodes:', ds.meta.total_episodes, 'frames:', len(ds))
print('cameras:', ds.meta.camera_keys)
print('task:', first['task'])
print('state:', tuple(first['observation.state'].shape), 'action:', tuple(first['action'].shape))
print('first/last camera shapes:', tuple(first['observation.images.camera2'].shape), tuple(last['observation.images.camera2'].shape))
PY
```

若这一段失败，停止训练并修复导出数据。不要通过改任务逻辑或删相机字段来绕过格式错误。

## 4. π₀.₅ 后训练及离线验证

LeRobot 0.6.1 的 `dataset.eval_split` 会按任务把**末尾** `ceil(总 episode 数 × eval_split)` 条轨迹留作验证。由于导出器写入训练在前、验证在后，且所有 episode 使用同一句高层任务文字，下面的比例正好选中独立生成的验证集。`eval_steps` 定期计算验证损失，`env_eval_freq=0` 关闭尚未接入的在线仿真评估。

```bash
EVAL_FRACTION=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_split"])' "$PI05_DATA/meta/pi05_export.json")
TRAIN_OUTPUT="$RUN_ROOT/pi05_output"

lerobot-train \
  --dataset.repo_id=local/alignbox3d_pi05 \
  --dataset.root="$PI05_DATA" \
  --dataset.eval_split="$EVAL_FRACTION" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.use_relative_actions=false \
  --policy.n_action_steps=8 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --output_dir="$TRAIN_OUTPUT" \
  --job_name=alignbox3d_pi05 \
  --batch_size=1 --num_workers=2 \
  --steps=20000 --save_freq=2000 --eval_steps=2000 \
  --env_eval_freq=0 --seed=1000 \
  2>&1 | tee "$RUN_ROOT/train.log"
```

`pretrained_path` 加载基础权重，输入输出特征名取自新数据集。导出器会**只用训练 episode** 为 state 和 action 写入全局统计量，包括 `q01`、`q99`，供 π₀.₅ 默认的 `QUANTILES` 归一化使用，避免验证数据进入归一化拟合。当前 14 维动作本身是末端增量加绝对夹爪命令，保持 `use_relative_actions=false`，不要再对它应用 π₀.₅ 的“相对当前 state”变换。`n_action_steps=8` 是日后每次推理执行的块长度；π₀.₅ 默认训练预测块仍为 50 步。`train_expert_only=true` 冻结 VLM，仅训练动作专家及投影层，减少显存占用。

训练日志必须出现 `Train/eval split: 100 train, 12 eval`（若改过 `TRAIN_N`、`VAL_N` 则替换为实际数字），并在 2000、4000、…、20000 步出现 `eval_loss=`。比较这些验证损失与训练损失；持续升高通常提示过拟合。**验证损失不是闭环任务成功率。** 当前 AlignBox3D 未注册为 LeRobot 的 `--env.type`，不能直接使用通用 `lerobot-eval`；下一步使用本项目的 π₀.₅ 适配脚本进行闭环验证。

## 5. 在验证场景闭环测试

保持 `lerobot-pi05` 环境激活。`PROJECT_ROOT` 是这个独立项目在服务器上的路径，整个流程不调用 ActionExpert。LeRobot 保存的 `checkpoints/last` 通常指向最后一个 checkpoint；先检查 `pretrained_model` 确实存在。若不存在，用 `find` 列出的实际目录替换 `CHECKPOINT`，不要填训练目录本身。

```bash
CHECKPOINT="$TRAIN_OUTPUT/checkpoints/last/pretrained_model"
if [ ! -d "$CHECKPOINT" ]; then
  find "$TRAIN_OUTPUT/checkpoints" -type d -name pretrained_model -print
  echo '请将 CHECKPOINT 改成上面实际存在的 pretrained_model 路径后重试。' >&2
  exit 1
fi

PYTHONPATH="$PROJECT_ROOT" python -m pi05_alignbox3d.rollout "$CHECKPOINT" \
  --dataset-root "$PI05_DATA" \
  --reposeenv-root "$REPOSE_ROOT" \
  --data-config "$REPOSE_ROOT/config/alignbox3d_data.yml" \
  --episodes 10 --seed 0 --max-steps 3000 \
  --output "$RUN_ROOT/pi05_validation_metrics.json"
```

脚本在 LeRobot 环境加载 checkpoint、导出数据的 `meta/contract.json` 和固定高层 `task`，调用官方 π₀.₅ 的预处理、`select_action` 和后处理；它通过现有 ReposeEnv TCP 客户端运行 validation split，每次新观测只执行一步动作。检查 `pi05_validation_metrics.json` 的 `success_rate`、每条 `results[].success` 和失败时的 seed。10 条只是快速验证；正式结论应使用更多未见过的验证场景。闭环成功率与上面的 `eval_loss` 分开报告。

## 6. 完成前核对

1. 导出检查与 LeRobot 单帧读取均通过。
2. 导出检查和 LeRobot 单帧读取显示固定高层任务句子；训练日志显示正确的 train/eval episode 数和每轮 `eval_loss`。
3. 保存 `RUN_ROOT`、两份原始数据、导出数据、训练日志、checkpoint 和 `pi05_validation_metrics.json`。后续闭环评估继续使用 `config/alignbox3d_data.yml` 的 validation split。
4. 将来推理时持续给模型 `meta/pi05_export.json` 中的同一句 `task`；不要从仿真专家读取阶段标签，也不要在执行中自动换成阶段句子。

## 7. 执行后发回给我什么

请把下面几项**原样回传**，不要只说“训练成功”或“效果不好”。这些信息足够判断数据格式、验证划分、收敛情况和真实任务成功率：

1. `PYTHONPATH="$PROJECT_ROOT" python -m pi05_alignbox3d.export --check "$PI05_DATA"` 的完整输出，以及第 3 步 LeRobot 首帧/末帧读取的输出。
2. `python -m pip show lerobot` 中的 `Version`，训练实际使用的完整 `lerobot-train` 命令、`TRAIN_N`、`VAL_N`、GPU 型号。
3. 训练日志中 `Train/eval split` 那一行、所有 `eval_loss=` 行、最后 30 行，以及实际使用的 `pretrained_model` checkpoint 路径。可用：`grep -E 'Train/eval split|eval_loss=' "$RUN_ROOT/train.log" | tail -n 30` 和 `tail -n 30 "$RUN_ROOT/train.log"`。
4. `cat "$RUN_ROOT/pi05_validation_metrics.json"` 的输出，尤其是 `success_rate`、每条 `results[].seed`、`results[].success`、`results[].steps`。如有失败，请给一个失败 seed 对应的录像或截图（若有保存）。

任何步骤报错时就在该步停止，发回**执行的命令、首个完整 traceback 和前一步的检查输出**；不要跳过错误进入训练或闭环验证。

官方依据：[LeRobot 0.6.1 安装](https://huggingface.co/docs/lerobot/v0.6.1/installation)、[π₀.₅ 训练与归一化](https://huggingface.co/docs/lerobot/v0.6.1/pi05)、[LeRobot 0.6.1 episode 留出逻辑](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/factory.py)、[训练验证损失实现](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_train.py)。
