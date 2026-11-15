# Stage293：R3资源选择与训练健康读回口径

## 阶段目标

按实际吞吐决定是否将三个原生训练迁移到较大设备，并使R3最终验收覆盖已冻结的完整模型矩阵。保留真实记录的健康粒度，不由新代码补造历史未测分支。

## 工作内容

提交入口增加可选`--allow-backfill`，只设置站点广告支持的`AllowBackfillSlot=true`，不改变accounting group或设备型号条件。探针1583012.0获得A100-PCIE-40GB，固定相同Train10k，对control、relations、去global各执行128updates并测三进程并行。实际获配的是普通`slot1_1`；不能将成功启动归因为backfill选项。

资源选择保留当前三个分别运行的1g MIG。新探针没有建立缩短整体完成时间的证据，既有作业持续构建输入，未中断或重启。较大设备的型号与显存本身不构成迁移理由。部分公开空闲H100的`SCond_vo_filter`排除当前accounting group，包含backslot；继续遵守站点准入。

新增`r3.health_review`，从真实optimizer健康记录汇总分支梯度RMS、代表参数更新及有限激活。梯度RMS以该分支全部参数数目归一化；runner的update RMS仅来自代表tensor，明确区分二者。M0历史只记录image/tracker/head三组，新G1记录stem/stages等八组，读回按原测量范围处理。冻结特征probe改用本地mmap以降低完整Tune准备峰值。

最终accept新增实际campaign完整模型集合检查，并要求紧凑图像决策条目；不能以满足“至少五模型”漏掉已冻结的C1。后续必要扩展加入campaign后也须进入完整验收。

## 详细数据

输出根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。

| Probe / review | Actual result |
| --- | --- |
| A100 control operation | 17.291973 events/s |
| A100 relations operation | 19.327646 events/s |
| A100去global operation | 20.418895 events/s |
| A100三进程并行operation | 37.717382 events/s |
| A100配置 | 128updates、同10k输入，BF16；显存40GB；申请8CPU/24GB RAM；未限定型号 |
| 迁移决策 | 不迁移；保留1583008/1583009/1583010，各自独立1g MIG |
| 正式新组批后1g训练速度 | 本报告形成时未测量；旧同形状资源profile各约17 events/s，合计是估计而非已完成对照 |
| M0真实健康读回 | image/tracker/head各152个windows，全部有非零梯度及代表参数更新，已观测激活全部finite |
| 新G1健康范围 | stem/readout、4个image stages、tracker、head；终态汇总待完成 |
| C1当前进度 | 37,540/50,051 updates，约300k/400k exposures，尚未形成终点结论 |
| 正式G2当前进度 | allocation正常、4个构建worker持续计算；完整100k缓存与正式优化结果未完成 |

证据：`audit/profile_g2_backfill_c10/results.json`、`audit/native_migration_decision_c10.json`、`health/m0_c2/results.json`及各`groups/*.json`。健康读回工具首次将M0按新G1细分名称解析时遇到KeyError，尚未产生结果；修正为读取实际历史分组后，`m0_c2`完成。首次空输出目录仅作诊断，不计为科学测量。

主要验证：

```bash
python -m unittest tests.test_flavour_v2_r3 tests.test_flavour_v2_r2 tests.test_flavour_v2_r1 -q
python -m analysis.flavour.v2.r3.health_review --root "$R3_ROOT" \
  --experiments "$R3_ROOT/experiments/b1_m0_t4_s42" --attempt m0_c2
```

40项tests通过；M0实际健康汇总完成。完整终态GPU/CPU成本继续由allocation账本计算，profile与失败/恢复均纳入。

## 阶段性结论

事实：本次A100并行吞吐不足以支持迁移，当前使用多个独立MIG继续更合适。M0负结果建立在实际学习和更新存在的基础上；历史没有测量更细image stages，不补填它们。

尚未测量：正式G2完整新路径吞吐、泛化收益、训练seed波动及最终R3总成本。本次资源决策不改变任何模型配方、曝光、评价合同或科学晋级门槛。

## 下一步计划

优先读回C1完整终点并完成冻结8项配对，随后运行其local/regional/global/tracker probes。继续G1与G2，在完整端点形成后决定必要seed/分辨率扩展。所有正式模型、分支、完整配对和allocation终态进入R3最终验收。
