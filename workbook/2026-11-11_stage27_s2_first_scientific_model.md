# 阶段 27：首个四轮 ParticleNet 模型与 B0 配对结果

ParticleNet 原生联合位置模型完成四轮，τ AP=0.35784。相对 B0 的 AP 差为 +0.01313，95% 配对区间为 [−0.00097, +0.02446]；固定 E/vertex common-support 权重后的差为 +0.01862，区间为 [+0.00686, +0.02971]。完整 S2 合同仍有44项比较待完成。

## 阶段目标

将首个完成的深度模型纳入预先冻结的科学评价，回答其在现有 validation benchmark 上相对 B0 summaries 的表现，并记录成本、错误类型和结论范围。

## 工作内容

`pn_native_joint_s42_e4_v2` 使用 accepted 80k train/10k validation，各类分别20k/2.5k，pixel+bar 原生完整观测，无事件截断。模型为当前层级 ParticleNet adaptation，161,156 parameters；seed42、batch4、四轮、每轮20,016 attempted steps，按 validation NLL 选择 checkpoint。calibration/test 未使用。

原 `1573747.6` 正常退出，Condor history 确认 JobStatus=4、ExitCode=0。更新 terminal allocation 后，以完整45项比较合同运行 `evaluate --allow-incomplete`；仅纳入完成模型，保留其余 pending。评价包含4个既有 references 与1个完成科学模型、133行分域结果、15个历史三类诊断和1个完成配对。绘制的 paired CI 与 resolution 图已实际查看；后者仅有1个原生点，其余粒度空缺。

资源统计使用新合并后端核算，评价时间匹配元数据字段改为 `actual_recorded_backend_allocation_used`。相关评价、Condor task 映射和 Kubeflow lifecycle 共16项测试通过，耗时1.101 s。该字段更名不改变指标或 bootstrap 计算。

## 详细数据

| Metric | B0 all summaries | ParticleNet native joint |
| --- | ---: | ---: |
| Tau AP | 0.34471068 | 0.35783732 |
| Tau ROC AUC | 0.62567088 | 0.63023309 |
| Four-class balanced accuracy | 0.402900 | 0.397600 |
| NLL | 1.24504237 | 1.24514857 |
| Brier | 0.67530617 | 0.67668165 |
| Tau selected at 1% FPR | 65/2,500 | 84/2,500 |
| Background selected at that point | 75/7,500 | 75/7,500 |
| Common-support weighted tau AP | 0.30845016 | 0.32706642 |

主表采用各类25%的 benchmark prior。1% FPR 为 validation empirical curve 上的阈值，不是最终 test 工作点。0.1% FPR 仅保留7个背景，ParticleNet选中14个τ，尾部统计不足的标记保持。

400次 paired、class-stratified、accepted-family-connected chunk bootstrap，128个物理块，每类32块：

| ParticleNet minus B0 | Point estimate | 95% interval |
| --- | ---: | --- |
| Tau AP | +0.01312665 | [−0.00097300, +0.02445526] |
| Tau ROC AUC | +0.00456221 | [−0.00650053, +0.01381586] |
| Tau efficiency at 1% FPR | +0.00760000 | [−0.00320388, +0.01618409] |
| Fixed common-support tau AP | +0.01861626 | [+0.00686071, +0.02970525] |

Common support 为原先固定的8,559 events、Neff=7,959.41、103个 E/x/y/z cells；区间以既有 validation weights 和训练模型为条件。当前原始 AP 区间既包含0，也包含预先关注的 +0.01 gain，不能据此给出等效结论。B0 为 CPU summary estimator，这个配对同时改变表示和估计器，不是等 GPU-time 或纯空间粒度对照。

ParticleNet confusion counts，行是真实类别、列是 argmax 预测：

| True / predicted | CC nu_e | CC nu_mu | CC nu_tau | NC |
| --- | ---: | ---: | ---: | ---: |
| CC nu_e | 1,225 | 102 | 837 | 336 |
| CC nu_mu | 268 | 701 | 976 | 555 |
| CC nu_tau | 394 | 345 | 1,321 | 440 |
| NC | 299 | 445 | 1,027 | 729 |

τ 分支在同一个 inclusive 1% FPR 阈值0.443035下，electron为11/445=2.4719%，muon为6/395=1.5190%，hadronic为67/1,660=4.0361%。相应分支对共同7,500背景的 AUC 分别为0.59666、0.55828、0.65636；各分支的 prior 不同，AP 不能直接横向解释成分支识别能力的排序。

pixel-centered 域为885 events、176τ，τ AP=0.30579；large-scintillator 域为4,494 events、1,053τ，AP=0.39528，域 prior 分别19.89%/23.43%，不同于全样本25%。这些是完成的 all-domain 模型在各域上的评价；训练域对照仍待完成。

本模型输入为空的440 events 包含148τ，其 AUC=0.5、AP=148/440，符合相同空输入只能输出固定分数的行为。tracker-only 的158 events也属于该模型的空输入，不能当作已利用 tracker 后得到的结果。

| Execution item | Measured value |
| --- | ---: |
| GPU / software | Tesla V100-PCIE-32GB / LCG107 / Torch2.5.1 / FP16 |
| Selected epoch | 4 |
| Program wall time | 11,332.78029 s |
| Condor terminal allocation | 11,379 s = 3.16083 GPU-hours |
| Allocation outside recorded program time | 46.21971 s |
| Applied / attempted optimizer steps | 80,030 / 80,064 = 99.9575% |
| AMP skips | 34 |
| Train exposures | 80,040 per epoch；320,160 total |
| Peak allocated GPU memory | 1,546.40576 MiB |

Exposures 包含匹配步数的少量重复，独立 train events 仍为80k。22:06:37 UTC S2 累计16.78668 GPU-hours，missing history records=0，包含旧失败、重启和 H100 profile；Kubeflow GPU 当时仍未分配。

复现命令：

```bash
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS" \
  --output "$S2_RESULTS/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.display --root "$S2_RESULTS" \
  --evaluation-tag evaluation_final --experiment pn_native_joint_s42_e4_v2
```

结果根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/`。`first_scientific_model.json` 固定保存本报告对应的模型、配对、分域、history 和 terminal allocation；`evaluation_final/` 随后续完成模型更新。对应评价代码 commit 为 `96f457e`，本阶段提交包含上述元数据字段更名。

原生事件显示已正常完成：`evaluation_final/event_displays/pn_native_joint_s42_e4_v2/native_error_groups.pdf` 共20页。已查看首页及第10/12/15页；`inspected_pages.json` 固定页码、event key 与对应 PNG。第10页τ→νμ事件（10083/103/758）呈稀疏近直线 pixel 响应，pτ=0.08236；第12页τ→NC事件（10083/234/347）没有 active pixel，仅有 bar 响应和图示单个 tracker hit，pτ=0.15741；第15页NC→τ事件（10084/811/763）有分散 pixel 与 bar 活动，pτ=0.54663。tracker 面板为诊断展示，此模型没有使用它。以上是错误组实例的可见响应，错误机制的总体占比和可恢复程度仍需冻结模态/结构对照。

## 阶段性结论

已测得首个深度模型的 τ AP 点估计高于 B0，原始配对区间仍跨0；固定 common-support 条件下的 AP 增益区间为正。四类 balanced accuracy、NLL 和 Brier 没有同时改善，τ AP 的变化不能当作整体分类性能均已提升。

目前尚未测量原生对1mm、joint对projection、结构消融、tracker增益或三种子均值，不能将 B0 配对差解释成其中任何一项的物理贡献。τ 与 CC nu_mu/NC 的混淆仍多，保留既定分支、模态和结构对照以检查具体原因。

## 下一步计划

继续其余34项科学配置和 Kubeflow GPU 探测。待相应模型完成后更新预定配对与资源比较；全部41项单seed比较、4项种子均值和两族原生错误显示齐全后，执行严格 S2 验收。
