# S4 数据扩展与预训练协议

## 阶段目标

将 S3 的稳定编码器用于 S4，冻结数据、目标和比较方式，回答 global 分支弱利用与 ντ 增益不足是否可由预训练改善。本阶段验收为协议、100k train manifest、新 bar 软标签定义和小样本验证；训练性能由后续实测报告给出。

## 工作内容

新增 [S4 合同](../roadmap/11_s4_pretraining.md)、`analysis.flavour.s4.prepare/submit`。既有 80k train、10k validation 保持原样；每类另选 8 个 disjoint chunks、5k train，新增 20k。只读取已接受 inventory，不重新认证样本或校准 detector-proxy。

联网读取 FASERCal 以外的 Sonata/NOMAE CVPR 2025、VICReg ICLR 2022 原文，补读 CERN ML flavor/training 文档。NOMAE 的 visible-neighborhood 查询和 Sonata 的 geometric shortcut 提醒决定本轮的无泄漏和 feature-variance 检查。输出资料存 `results/flavour/s4_2026-11-13/literature`。

## 详细数据

从 `Hits/scintHits` 读取 16 个真实事件验证新增 target codec。硬件 channel IDs 全部一致；bar 数量从 0 到 2,542，最大能量相对差为 **5.957364216627158e−8**，simplex 和最大差为 **1.1102230246251565e−16**。所有正能量 bar 的 contribution fractions 有效。证据：`results/flavour/s4_2026-11-13/rel_smoke.json`。

标签定义：按每轨迹 `edep` 对同一硬件 bar 累加，类别 EM(11/22)、muon(13)、hadron、other。`ScintSD.cc` 的 `barEnergyMap` 保留 track 与 PDG，符合此定义。Sidecar 与 observations 分开，仅用于 simulation-supervised partial Rel loss。Pixel 首贡献 flags 不适用精确软标签，本轮不使用。

执行 manifest 命令：

```bash
python -m analysis.flavour.s4.prepare plan \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 \
  --base /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100
```

转换通过 HTCondor 正式提交 16 tasks，其中 4 tasks 生成新 observation shards，16 tasks 生成 train-only Rel sidecars。即时 schedd 选择依据与任务 ID 由 `submission_*.json` 保存。批量转换及训练结果本报告时未测量。

## 阶段性结论

现有 bar 数据能够构造精确的 per-track energy contribution target，无须重新模拟。此结论只涉及新标签的定义和上述 16-event 实测；100k 目标的 join 和能量一致性由批量产物汇总验证。S3 global 弱利用和 ντ 增益不足仍是待解问题，不能由目标可用性推出预训练有效。

## 下一步计划

完成 100k train 数据与 sidecar 汇总；实现 visible-only masking、含空通道的 query decoder、EMA/variance 目标及 train-only branch probes；通过真实事件拟合和占用压力验证后，再提交 0.5/0.75 mask pilots 和正式对照。所有 S4 allocations 合计上限 60 device-hours，优先完成同 backbone、同标签和 matched-compute 比较。
