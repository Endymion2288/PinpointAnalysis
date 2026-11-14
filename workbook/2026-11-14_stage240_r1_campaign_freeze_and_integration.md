# R1：筛选合同冻结与完整训练入口验证

**阶段目标**

在首次新 Tune score 前冻结 R1 比较规则，并验证训练、保存、推理与逐事件曝光记录的完整流程。完整 Train100k/Tune40k 的泛化比较尚未完成，R1 保持进行中。

本阶段产物根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。未读取 Cal-A、Cut-B、Confirm payload。

**工作内容与详细数据**

1. `campaign.json` 已在新 Tune score 读取前生成；仓库副本为 `roadmap/v2/configs/r1_campaign.json`。每族 T3/T4 各两学习率，同族四项在同一 allocation 顺序运行，每项 300 训练秒。B1/B4 候选为 0.0003/0.001，B2/B3 为 0.0001/0.0003。固定 Tune family-hash 1k/class 做筛选，按共同非 τ BA3、再按 NLL3 选择。完整训练使用四 passes，最强两族与 B4 扩 seeds17/42/2026。混合 GPU/MIG 的小时按型号分别报告。
2. `fit/all_health_k10` 在 Kubeflow H100 MIG 3g.47gb 完成八项真实拟合，使用实际 local chunks：B2=512、B3=1024、B4=2048；B2 eager，B3/B4 compiled。所有项通过 NLL、accuracy、非常数概率、分支梯度、参数更新和有限 activation 检查。T3 NLL 分别为 B1=0.0000380021、B2=0.00277960、B3=0.00373878、B4=0.0000875347；T4 分别为 0.000162465、0.00632769、0.0776550、0.000329833。B3 T4 accuracy=96.875%，其余100%。本结果只验证学习路径。
3. `smoke/a1` 只取固定32个真实 Train 事件，其中24个作为训练、8个作为模拟 validation。八项 B1–B4×T3/T4 都完成一 pass、checkpoint读回、预测表和曝光记录。`integrity.json` 明确记录 `actual_Tune_used=false`；这些模拟 validation 日志不作为泛化结果。
4. 另以相同 Train-only 数据执行总 operation wall-budget 流程：目标20.000秒，实际20.227874秒，偏差+1.1394%；130 updates、782次非τ曝光、18个unique训练事件。完整模拟 validation 推理0.047429秒。证据为 `smoke/a1/experiments/smoke_b1_t3_wall/manifest.json`。正式同计算控制同时报告目标、实际耗时与5%容差，包含 loader、training 和 validation；不是仅对齐 CUDA kernel 时间。
5. `profile_b2_a6` 在 A10040GB 完成 chunk512 的256 updates/2048 exposures，总训练80.663803秒、operation102.976231秒、峰值1995.041 MiB。第二个128-update窗口为19.555227秒，但它与第一窗口的事件不同，不能单独归因为某项kernel改进。
6. `audit/diagnose_B2_sentinel_a1.json` 的真实 GPU 检查表明，改为 BF16 精确表示的 sentinel 仍产生 compiled BF16 反向 NaN。上一报告中的舍入假设未解决故障，现撤销该共享 S2 实验修改，保留 B2 eager 稳定执行路径。故障尚未定位到具体 kernel，不据此评价 ParticleNet 的物理能力。
7. 训练记录增加 branch activation RMS、最大绝对值、finite 与调用次数；hook 设在 compiled 子图外。stress catalog 按各扰动后的实际通道数写出。提交入口支持独立配置 Condor job flavour 与 Kubeflow deadline，以容纳真实 profile 估计的长任务。

验证命令：

```bash
python -m unittest tests.test_flavour_v2_r1 tests.test_flavour_spatial tests.test_flavour_s3_repair
python -m py_compile src/analysis/flavour/v2/r1/*.py
python -m analysis.flavour.v2.r1.fit --root R1_OUT --r0 R0_OUT \
  --attempt all_health_k10 --families B1 B2 B3 B4 --bf16 --compile-local
python -m analysis.flavour.v2.r1.smoke --root R1_OUT --r0 R0_OUT --attempt a1
```

28项相关测试通过。全部56个转换分片的最终合并、真实 Tune 指标、完整四轮训练、三seed、资源总量仍未测量完成。

**阶段性结论**

已测事实：四个架构在正式精度/执行分块下均有可用学习路径；端到端训练入口与总耗时预算入口可以运行。sentinel 修改不是 B2 compiler 故障的有效修复。实际架构优劣、T3/T4取舍和模态净收益尚无新泛化证据。

**下一步计划**

完成140k事件的身份、标签、PE与数量验收后，立即并行运行 B0 CPU 与四族 GPU screen。随后完成四pass、同计算控制、前两族与共同参考三seed、无tracker重训、固定noise/combined推理、400次paired物理块bootstrap与英文图表；按完整交付检查更新R1状态。
