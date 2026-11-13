# S4 终态成本查询修复与独立监督 seeds 进展

## 阶段目标

补全按设备区分的实际 allocation 成本，读取新完成监督模型，继续计算量匹配控制。

## 工作内容

成本脚本原先对 Condor history 使用 `-since 'ClusterId < 当前ID'`。History 按完成时间排列，cluster ID 不决定完成先后，该停止条件会提前截断查询。改为精确 cluster constraint 和预期 proc 数量的 `-match`；已取得全部终态 proc 的 cache 可直接复用，仍在运行或证据缺失的 allocation 继续查询。20 秒超时保留已有 ads，不抹去记录。

设备名称读取增加 Condor job ad 中实际匹配得到的 `MATCH_GPUs_DeviceName` / `MATCH_EXP_GPUsDeviceName`；不能从声明的多设备 pool 推断实际获配硬件。额外记录 AssignedGPUs，以区分 MIG 与整卡。

配置 `s4_validation_job.json` 推进至 v5；已保存的 v4 输出和提交快照保留。新 v5 将包含新增 Scratch seeds 和 frozen random probes，仍为 progress evaluation，最终 strict 验收另行执行。

## 详细数据与验证

2026-11-13 约21:00 UTC，去除错误停止条件的直接 history 查询取得：

| Cluster | 终态 | Remote wall time | Exit code |
| --- | --- | ---: | ---: |
| 2198166 | completed | 72 s | 0 |
| 2198297 | completed | 64 s | 0 |
| 1574091 | removed | 0 s | 未执行 |

前两项为 CPU 作业，补充 core-hours，不增加 GPU device-hours。实际匹配字段确认 1574116 为 H100 MIG 2g.24gb、1574593 为 H100 MIG 1g.12gb，失败的 2198172 为 V100-PCIE-32GB，2198296 为 H100 MIG 2g.24gb。失败的 allocation 时长继续计入。

LCG108/Python3.12 执行 `python -m unittest tests.test_flavour_s4_accounting`，1 项通过，0.005 秒，验证 exact-cluster query 不使用 ID 停止条件、匹配设备名称、成本求和和 terminal cache 复用。

21:00 UTC 的正式作业进展：2198680 已获配 `b9pgpun207`，同时启动等计算监督 1k/10k；CPU 分支诊断 2200411 正在运行。Scratch1k seed2026 完成全部 epochs，正式 best 指标和内部健康由 v5 统一核对；不能用末 epoch 的高过拟合 NLL 替代 best。随机 encoder 的 100k-label seed2026 frozen probe 也已完成。

上一完整成本快照为20:54:49 UTC：5.46265712 device-hours，98.99213149 CPU core-hours；本次修复后的完整刷新执行中，本报告不把该旧快照当作当前最终成本。

## 阶段性结论

终态查询缺口来自调度记录查询条件，已有模型及预测不受影响。新增 compact MIG 已分担计算量匹配实验；现有训练正常继续。

## 下一步计划

完成 v5 的统一 readback 与统计，检查完整分支诊断，刷新所有 allocation 成本，清理已核实终态的 AFS 小提交文件。
