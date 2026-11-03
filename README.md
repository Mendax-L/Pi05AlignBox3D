# Pi05AlignBox3D

独立于 ActionExpert 的 LeRobot v0.6.1 `pi05` 训练数据与验证工具。

- `pi05_alignbox3d.export`：把 ReposeEnv 原始 train/validation 示范合并成单一高层 `task` 的 LeRobot v3 数据集；原始阶段标签保持在原始目录。
- `pi05_alignbox3d.rollout`：加载训练后的 π0.5 checkpoint，在 ReposeEnv validation 场景闭环运行，输出成功率。
- [RECIPE.md](RECIPE.md)：服务器上从采集到训练、离线验证、闭环验证和反馈回传的完整命令。

本项目使用标准 LeRobot `pi05` 动作微调路径。[π0.5 论文](https://www.physicalintelligence.company/download/pi05.pdf)的显式子任务文本预测需要另行实现；这里不会把原始专家阶段标签当作推理输入。
