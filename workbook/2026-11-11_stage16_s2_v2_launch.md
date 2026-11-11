# 阶段 16：v2 GPU 验证完成并重启空间矩阵

显式保留当前表示总量的 v2 已通过两族 V100 smoke 和完整 576-row resource profile。两 smoke 各完成 15 个 optimizer steps，AMP skipped steps 均为零。重启同一 80k train/10k validation 的 12 项空间对照；S2 科学验收仍未完成。

## 阶段目标

用修正后的汇聚进行原生/粗化/投影对照，同时保留 reference、真实 allocation 和分阶段完成状态。

## 工作内容

1573744 两 smoke tasks 均完成，1573745 profile 完成。模型版本 `d22bd0a`，V100/LCG_107/FP16。ParT dim32 smoke 总时长 29.103 s、peak 132.941 MiB；ParticleNet dim32 为 13.472 s。smoke 使用 29 train/3 validation，缺少两个 validation classes，只作执行验证，不作科学分类结果。

v2 profile 独立写入 `resource_profile_v2/`，不覆盖 v1；相同 24 个占用秩事件、六种表示、两家族、inference/training 各一次。全部 loss 有限，query tiling 保留每个真实块。初次 CUDA warmup 仍包含在第一事件最大耗时中。

冻结 [18 个空间/参考配对](../roadmap/configs/s2_spatial_pairs_v2.json)：两家族各七项粒度/关联效应、native/1mm 的家族对照、两个 native full-response 与 B0 summary 的对照。后两项同时改变输入表示和 estimator，不解释成纯架构收益。primary 仍是 τ AP；同时给 τ ROC AUC、固定 FPR efficiency、fixed-common-support weighted AP 的配对区间。

评价可在多个 run 尚未完成时保存已有结果与明确的 pending comparisons；最终验收使用默认完整模式。新诊断图入口生成四类 confusion/ROC、τ score 与 learning-curve PDF，以及 truth E/vertex matching 前后的密度图。reference 写 `evaluation/`，空间写 `evaluation_spatial/`，原生 displays 可按 tag 分开输出。训练进度显式显示 applied/AMP skipped counters。

## 详细数据

v2 batch-1 median training time 为 forward+backward，未含 optimizer/loader；最大显存来自该 24-event 压力抽样。

| Family / representation | Inference median [s] | Training median [s] | Maximum training GPU [MiB] |
| --- | ---: | ---: | ---: |
| ParT native joint | 0.02080 | 0.06907 | 204.23 |
| ParT native projection | 0.02121 | 0.07651 | 160.89 |
| ParT 100μm joint | 0.01882 | 0.06703 | 186.13 |
| ParT 200μm joint | 0.01770 | 0.06542 | 172.44 |
| ParT 1mm joint | 0.01806 | 0.06324 | 161.01 |
| ParT 1mm projection | 0.01746 | 0.06370 | 160.85 |
| ParticleNet native joint | 0.01781 | 0.04412 | 549.66 |
| ParticleNet native projection | 0.01850 | 0.04535 | 545.58 |
| ParticleNet 100μm joint | 0.01750 | 0.04405 | 549.62 |
| ParticleNet 200μm joint | 0.01764 | 0.04110 | 548.48 |
| ParticleNet 1mm joint | 0.01747 | 0.04083 | 547.54 |
| ParticleNet 1mm projection | 0.01671 | 0.04162 | 545.08 |

更新 reference evaluation 1573746 完成。既有 accepted family annotations 在本 validation 子集中形成 128 个 bootstrap blocks、每类 32；与原 chunk 划分一致，因此阶段 12 的 AP 区间数值不变。新增固定 common-support weights 条件下 ΔAP 为：

| A → B | Weighted Δ τ AP | 95% interval |
| --- | ---: | --- |
| Signal → full B0 | +0.00327 | [−0.00958, +0.01595] |
| Shape → full B0 | +0.01354 | [+0.00565, +0.02216] |
| Full B0 → truth E/vertex diagnostic | +0.05629 | [+0.03259, +0.07848] |

这些区间以固定 common-support weights 为条件，不包含权重重新估计的不确定度，不对应 flux/exposure。B0 shape 相对 signal 的额外收益仍未在 primary AP 上明确；不能以 AUC 上的差别替代 primary gate。0.1% FPR 的实际背景仍仅 7 个，保持尾部统计不足标记。

v1 终止时实际已运行 802+802+503=2,107 s，即 0.58528 GPU-hours；其余 v1 jobs 为 idle 后移除、GPU allocation 为零。加上更早 smoke/profile，当次 accounting 总计 0.74778 GPU-hours。之后的 v2 成本另随 terminal history 合并，未隐藏已停止实验。

```bash
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS" \
  --campaign "$PWD/roadmap/configs/s2_spatial_screen_v2.json" \
  --lcg-view LCG_107_cuda --gpu-device 'Tesla V100-PCIE-32GB'
python -m analysis.flavour.s2.submit evaluate --root "$S2_RESULTS" \
  --evaluation-tag evaluation_spatial --comparisons "$PWD/roadmap/configs/s2_spatial_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
```

## 阶段性结论

事实：v2 的总量路径在真实 V100 优化和各表示压力输入中可执行，资源与 v1 同量级。reference 的 conditional AP 区间仍支持“先验与信号统计必须控制”的研究重点。

未测量：v2 科学 validation 增益、factorial interaction、三种子 spread 和完整 GPU-hours；尚不能判断 native 是否优于 coarse。

## 下一步计划

监测重启矩阵的完整 epochs、loss/AMP counters 和实际 allocation；完成 18 个主配对与两家族 interaction，随后运行 OR/shift/modal/graph/pair/shuffle/domain 和三种子确认，逐项完成 S2 科学清单。
