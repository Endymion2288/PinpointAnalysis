# S4 冻结与解冻编译修复通过真实 CUDA 回归

## 阶段目标

确认修复恢复 encoder 前端更新，并在相同权重下核查 FP32/BF16 执行误差，决定是否启动完整 LP 候选重训。

## 工作内容

正式 Kubeflow Job `pinpoint-s4-transfer-recovery-v4` 于 03:41:01 UTC 创建，使用 H100 NVL MIG 3g.47gb，运行源码 `6c3dce5929a726340d38b809e224f7b0cf7405d5`。gate 使用八个原 train 事件，在 FP32/BF16 下分别执行旧编译、eager、修复编译路径，每条路径两步 LP、一次 LP eval、三步解冻更新。数值检查使用相同最终权重，标准已在阶段159测量前冻结。

## 详细数据

gate 在 03:42:46 UTC 完成，耗时 88.564584 秒，13 项检查全部通过。[精简实测](../roadmap/results/s4_transfer_compile_gate_v4.json)保留检查、事件键、真实梯度和参数更新。完整记录位于输出根 `experiments/s4_transfer_compile_gate_v4_s42/transfer_compilation.json`。

| Precision/path | Input maximum parameter change | FT input/local gradients |
| --- | ---: | --- |
| FP32 legacy compiled | 0 | Absent |
| FP32 eager | 0.000090386719 | Positive on all three steps |
| FP32 repaired compiled | 0.000090386719 | Positive on all three steps |
| BF16 legacy compiled | 0 | Absent |
| BF16 eager | 0.000090364367 | Positive on all three steps |
| BF16 repaired compiled | 0.000090468675 | Positive on all three steps |

| Final weights from | FP32 compiled max logit error | Eager BF16 RMS to FP32 | Compiled BF16 RMS to FP32 | Predeclared RMS limit |
| --- | ---: | ---: | ---: | ---: |
| FP32 training | 8.940697e-7 | 0.005080304 | 0.005543683 | 0.010260608 |
| BF16 training | 9.238720e-7 | 0.004559353 | 0.004345136 | 0.009218707 |

FP32 独立训练轨迹最大 logit 差为 9.089708e-7；BF16 为 0.019856453。相同权重 eager/compiled BF16 最大概率差分别为 0.001589537 和 0.001564562。这些是八个事件上的实现回归测量，不代表完整事件分布的精度系统效应。阶段159旧 gate 保持 failed，不以本轮通过改写历史。

gate 通过后，同一冻结 campaign 自动启动六并发的 1k 三折 Scratch/MAE3 LP 候选，随后运行六项 10k 候选。六个 1k 候选实际训练记录均已观测到解冻后的 input/local 正梯度，首条 FT input 梯度平方范数为 1.241240–158.738647，local 为 0.020495–120.309036；当时已保存的采样均 healthy。完整轮次、真实参数改变及 OOF 选择尚未完成，不能以首条梯度代替最终验收。

另外八项正常 non-LP/frozen 恢复已通过 HTCondor 正式提交到 bigbird21，cluster 2205000；与已有恢复作业共享 claims。调度器在提交前比较实时队列和 EOS 兼容性，选择依据保存在 `logs/submission_pinpoint-s4-readout-resume-compact.json`。所有日志和运行产物写入 EOS，AFS 只保存小型提交文件。原 60 device-hours 累计不重置。

## 阶段性结论

事实：新回归同时复现故障、恢复真实前端更新，并通过预设相同权重数值检查；六项实际 1k 训练的解冻前端梯度已恢复。旧 LP 候选、旧共同选择和相应受影响重拟合仍失效。

待测：12 项完整 LP 候选、重新共同选择的 recipe/epoch、三 seed refit、完整 10k validation 与物理分块统计。gate 通过只允许继续训练，S4 仍进行中。

## 下一步计划

每个预算六项 LP 完成后，与36项正常 non-LP候选共同重做原 train-inner OOF 选择；固定末轮进行三 seed refit，检查真实训练前端梯度及参数改变，再执行 checkpoint/readback、独立 FP32 和完整物理评价。继续有效的 frozen readability、matched-compute 和 100k 目标对照，最终合并资源账本与终态。
