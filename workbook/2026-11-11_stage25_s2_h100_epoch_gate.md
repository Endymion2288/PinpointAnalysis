# 阶段 25：冻结 H100 完整 epoch 的资源选择标准

H100 的 batch1 资源测量支持进一步验证整轮耗时。已建立独立候选目录，并在 H100 分类训练开始前冻结原有全部 35 项科学配置及纯执行/耗时选择标准；先提交两个 native-joint 模型。当前正式 V100 任务继续运行和排队。

## 阶段目标

测量相同数据与 batch4 流水线的首轮程序耗时，决定后续完整 S2 矩阵采用的统一执行环境。最终交付仍为 35 项科学配置、41 项单 seed 配对、4 项种子均值及既有物理/资源诊断。

## 工作内容

[候选合同](../roadmap/configs/s2_h100_v2.json)从既有 spatial/controls/confirmation/cross-plane 四份配置逐项复制 `experiments`，全部模型参数、4 epochs、batch4、每轮 20,016 attempted steps、种子、选择规则和表示保持原值。35 个名称唯一，两个先行 task 为 0（ParT native joint）和 6（ParticleNet native joint）。

执行环境为 NVIDIA H100 NVL、LCG_108、Torch2.7.1、BF16，JobFlavour=workday。候选根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100`。通过 symlink 共享既有 `plan.json`、90k-event manifest、catalog、32 个完整 shards，以及 4 个已完成 CPU references；模型、checkpoint 和任务记录单独写入候选目录。具体来源记录为 `h100/source_links.json`。沿用用户接受的 samples、source-family split 和 detector-proxy。

## 详细数据与预先固定标准

资源比较为阶段 24 实测同事件 batch1 compute 的约 1.69–2.33 倍；整轮耗时尚未测量。先行任务使用完整 80k train/10k validation，按原计划运行四轮；在两者首轮均完成后评价资源标准：

| Requirement | ParT native joint | ParticleNet native joint |
| --- | --- | --- |
| V100 first-epoch program wall [s] | 4,859.18128 | 2,937.91183 |
| H100 maximum first-epoch program wall [s] | 3,887.34503 | 2,350.32946 |
| Maximum H100/V100 ratio | 0.80 | 0.80 |
| Minimum applied/attempted optimizer steps | 0.99 | 0.99 |

同时要求两项均保持完整输入和冻结配置，train loss、validation 概率与 checkpoint 参数有限，恢复/优化器状态可读。程序内 epoch wall 包含数据准备、训练和 validation；Condor allocation 与启动开销另行核算。validation AP、accuracy 和 NLL 的数值不用于硬件选择；四轮 checkpoint selection 仍按既有 validation NLL。

两者均通过后，完整 35 项矩阵采用统一 H100/LCG108/BF16，移除已被替代且仍 idle 的 V100 任务，已运行的 V100 任务继续完成并单列为辅助结果。任一未通过，则保留 V100 正式合同，候选不扩展。所有原 V100、资源测量、候选与后续任务均计入 S2 的 **200 GPU-hours** 总组织上限；完整环境中的 100 GPU-hours 初筛与 100 GPU-hours 控制/确认安排继续检查实际成本。

候选任务的完整首轮、optimizer counters、NLL/AP 和最终成本均未测量。准备工作确认完整合同引用的 experiment names 均存在于 35 个科学配置或 4 个 references 中，原有 41+4 比较文件可直接用于独立候选目录。

```bash
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --tasks 0 6 \
  --lcg-view LCG_108_cuda --gpu-device 'NVIDIA H100 NVL' \
  --training-flavour workday
```

## 阶段性结论

完整的候选矩阵与选择标准已在 H100 分类训练结果产生前固定。当前证据支持同事件 batch1 执行较快，尚未证明完整 epoch 的收益。正式比较仍须在单一选定执行环境完成全部指定任务。

## 下一步计划

核实两个先行任务实际 GPU、precision、数据数量与程序执行，取得两份完整首轮 checkpoint 后按上述标准作出资源决定。随后完成选定环境的整个科学矩阵及严格评价，保持 calibration/test 冻结至 S6。
