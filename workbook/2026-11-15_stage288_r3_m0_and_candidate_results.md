# R3 全程重混结论与原生候选图实测

## 阶段目标

用实际固定endpoint选择G1/G2的采样配方，交付候选关系覆盖与recorded-label recall，并以profile优化执行方式。R3正式结构比较仍在运行。

## 工作内容

M0 Kubeflow正式训练完成，使用同300k unique、1.2M曝光、150117 updates与原D2相同逐update batch大小/LR。CPU Condor `1583001.0`完成四个视图、八个配对、每视图11域、400次physical-block bootstrap、1368个独立点值检查及116个核心区间检查。

候选审计对同1024个Train事件（256/class）比较all-visible与modality-visible。正式G1根据M0质量约束保留spatial sampler，fresh stride2已在Kubeflow `pinpoint-r3-g1-k1`运行；既有R2 D2 stride4为匹配参考。原生本族正式配置尚待完整validation/profile再冻结。

## 详细数据

输出根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。M0最终训练2878.5459 s、validation200.2710 s、data wait51.4954 s；各类曝光300000，`sum(lr)=23.589806175917317`，与预定差小于3e-14。最终BA3/NLL3为0.575100/0.8457252，BA4/NLL4为0.460275/1.1635314，τ efficiency@1% FPR为0.060300。

| M0 minus R2 D2 raw endpoint | Difference | Paired 95% interval |
| --- | ---: | --- |
| Common BA3 | +0.383333 pp | [+0.035338,+0.735033] pp |
| Common NLL3 | −0.00330528 | [−0.00497473,−0.00130250] |
| Common e recall | +6.370000 pp | [+5.726488,+7.028339] pp |
| Common μ recall | −4.080000 pp | [−4.711377,−3.466899] pp |
| Common NC recall | −1.140000 pp | [−1.746481,−0.562904] pp |
| τ efficiency@1% FPR | −0.020000 pp | [−0.454780,+0.405539] pp |
| NLL4 | −0.00270176 | [−0.00445404,−0.00097694] |
| Sparse μ recall | −14.816137 pp | [−16.323035,−13.373717] pp |
| Sparse macro AUROC3 | +0.00422546 | [−0.00114017,+0.00918363] |

Sparse仍按原accepted总channels<100，未因模型输入改变分组。完整结果在`evaluation_m0/`；endpoint为主，best-screen另外报告。单seed的event区间不估计training seed总体波动。

**采样决定：不采用全程重混作为下一批默认。** 小幅BA/NLL收益伴随明确μ质量损失，未达到2 pp BA晋级和逐类约束；τ无增益。保留完整效应，不补M0 T3，也不把该结果解释成所有reshuffling无效。

| Candidate audit on same Train1024 | All visible | Modality visible |
| --- | ---: | ---: |
| Node-weighted any-edge coverage | 0.980699 | 0.990138 |
| Node-weighted cross-plane coverage | 0.980635 | 0.990102 |
| Maximum out-degree | 12 | 14 |
| Edges/event p50 | 1392.5 | 1919.0 |
| Edges/event p95 | 25456.75 | 25765.75 |
| Edges/event p99 | 50648.17 | 50836.98 |
| Maximum edges/event | 100091 | 110466 |
| Recorded-label recall, pixel gap1 | 34730/135493 = 0.256323 | 37297/135493 = 0.275269 |
| Recorded-label recall, pixel gap2 | 5430/63196 = 0.085923 | 6412/63196 = 0.101462 |
| Recorded-label recall, pixel gap3 | 0/4628 = 0 | 809/4628 = 0.174806 |
| Mean largest weak-component fraction | 0.694731 | 0.821553 |

Recall分母是source pixel node × target visible pixel plane，要求目标plane存在同一个正recorded trackID；任一候选连到该ID即成功。PixelSD只保留首贡献者；没有完整ancestry claim，也不是reconstructed track efficiency。无重复recorded pixel channel。较低recall提示bounded KNN仍丢失许多记录关联，留R4候选扩展/多候选学习，不阻止保留全部raw节点的分类单因素检验。weak component仅表示多次传播可能范围；当前只有一层raw-node消息，不能称整component已进入同一receptive field。

原生profile修订为allocation-local exact输入只构建一次，control与relations共享同一缓存；真实fixture 24个optimizer Train事件全collated tensors精确相同。Train10k cache实际1,404,362,752 bytes、四worker构建199.2349 s，32个scattered输入精确检查通过；candidate degree≤14，real blocks 2088960，有external/cross-plane context的block分别2081365/2081125，node数11029940。该较大Train10k的edge max379360，说明1024-event的max不是全样本上界。完整Tune cache/profile仍在执行。

旧`1582998.0`四项学习gate保留；为避免每个profile phase重复单worker构图，在尚无正式分类模型时将其替换为共享四worker profile `1583000.0`，全部已消耗allocation成本保留。全局stack移除的真实T4拟合由`1582999.0`完成：32 events、160 steps、100% accuracy、NLL0.0256318，所有学习健康检查通过。

核对stage285的smoke措辞：源为32个真实Train事件，但fixture实际划为24个optimizer Train和8个synthetic-validation。三passes的actual coverage为72次曝光、9updates；continuous/recovered参数差0与完整表相等不变。新增`audit/m0_fixture_readback.json`明确分母，避免将32个源事件全部称为optimizer Train。

英文图已生成并目视：`r3_phase_sensitivity.*`、`r3_stride_cost_profile.*`；M0 endpoint和配对图、candidate coverage图由`analysis.flavour.v2.r3.plots`生成。主要命令：

```bash
python -m analysis.flavour.v2.r3.evaluate --root "$R3_ROOT" \
  --contract "$R3_ROOT/contracts/m0_evaluation.json" --output evaluation_m0 \
  --replicates 400 --workers 8
python -m analysis.flavour.v2.r3.campaign G1 --root "$R3_ROOT" \
  --mixing spatial --cache-workers 4 --reason 'M0 mu quality guard fails; retain matched D2 sampler'
```

## 阶段性结论

M0没有修复共同任务逐类取舍，因此保留原采样。新候选图在固定Train审计中扩大传播和recorded-label recall，仍不足以作为完整对象关联方案。原生关系及global-stack ablation均能真实学习，正式收益未测量。

## 下一步计划

完成G1的固定曝光模型与G2完整profile，冻结本族control/relations/retrained-ablation；完整Tune配对评价、独立readability和训练梯度共同形成available/used/useful结论。只按已冻结质量门槛触发三seed/分辨率和anchor/fusion扩展；最终交付R3科学决策、全部终态成本和验收。
