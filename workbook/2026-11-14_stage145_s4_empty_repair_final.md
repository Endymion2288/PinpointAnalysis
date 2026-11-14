# S4空响应修复、18项重新训练与完整验证结果

## 阶段目标

验证空响应占位的架构修复能否消除异常反向梯度，并在相同分类合同下重新测量Scratch/MAE3的标签效率。18项训练、完整10k validation、4模型分支诊断及21项技术检查全部完成；61个本S4 allocation均有终态，累计账本为24.325865/60 device-hours。旧异常模型只保留作诊断，以下方法比较全部来自新矩阵。

## 工作内容

### 架构与训练修复

S3已采用Peri-LN、QK normalization、LayerScale、分支与Head归一化、FP32 SiLU Head及线性readout，并增加全部local blocks的直接路径。本轮进一步定位到空事件被表示为mask=true的零占位节点：连续零方差归一化放大反向梯度，全局clipping随之压低其他分支的更新。相同seed与首批train事件的损失分解，确认两项旧初始化的preclip norm分别为6.4265×10¹²、1.8055×10¹²；H100 BF16也复现。

新`empty_response_bypass=true`排除虚假local节点、屏蔽detector type3的regional token，并跳过全空source的attention更新。空事件继续参与分类，学习共同的空响应类别概率；它对硬件input参数的梯度精确为0。非空CPU输出与旧路由逐位相同，参数命名和897,992个参数保持兼容。完整分解、论文依据及新gate见[阶段137](2026-11-14_stage137_s4_empty_response_gradient.md)、[阶段138](2026-11-14_stage138_s4_gpu_empty_replay.md)、[阶段139](2026-11-14_stage139_s4_empty_gate_and_launch.md)。

[Xiong等，ICML 2020](https://proceedings.mlr.press/v119/xiong20b.html)与[Peri-LN，ICML 2025](https://proceedings.mlr.press/v267/kim25u.html)支持检查归一化位置、方差与梯度传播；具体空占位病因由本项目的同初始化损失分解测得。修复后BF16重放的两个norm为156.1000和59.7312，真实响应的学习路径保留。

### 样本与冻结合同

沿用已接受的samples、family split和detector-proxy。Run 10081/10082/10083/10084对应CC νe/CC νμ/CC ντ/NC，输入为原生pixel、bar信号及tracker响应代理。Train共100,000事件，validation共10,000事件；各类分别25,000和2,500。监督标签预算为嵌套、平衡的1k/10k/100k，seeds为17/42/2026。

矩阵为Scratch/MAE3×3预算×3 seeds，共18项，配置为[s4_empty_repair.json](../roadmap/configs/s4_empty_repair.json)，配对为[s4_empty_pairs.json](../roadmap/configs/s4_empty_pairs.json)。1k使用8×512次更新，10k为8×1,256，100k为4×6,256，总计235,008次；按既定最低validation NLL选checkpoint。所有更新都检查preclip norm，初始32次密集记录内部激活，此后每100次采样。

Scratch重新初始化；MAE3复用此前已通过重建学习与readback的3套预训练encoder，再重新fine-tune。**本轮没有重新训练SSL encoder本身。** 两组同时采用相同新路由、标签子集、sampler及更新预算。原EMA/Rel、compute-matched与frozen-probe结果未在本修复矩阵中重测，不用于扩展本报告的比较结论。

统计使用同一完整validation的9项配对、3项三seed均值对照，400次、128个物理chunk/family块bootstrap，并报告raw及既定common-support重权结果。区间为条件于已训练seeds、固定权重和checkpoint选择的pointwise事件区间；训练seed SD另存，均值不是ensemble。此次修复在观察旧S4后设计，validation被重复使用。Calibration/test继续冻结，本报告不是最终独立test结论。

## 详细数据

输出根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。精简数值为[s4_empty_repair_summary.json](../roadmap/results/s4_empty_repair_summary.json)，完整结果为输出根`evaluation/empty_repair/summary.json`。

### 数值健康与信息路径

| Measurement | Result |
| --- | ---: |
| Real 32-event BF16 memorization | Accuracy 1.0; NLL 1.899898×10⁻⁷; 768 updates |
| Empty / median / maximum occupancy stress | 6/6 forward/backward rows finite |
| Largest train event | 36,549 native nodes → 14,599 local blocks → 481 regions / 962 regional tokens |
| Local / global readout | All local blocks retained; 64 global latents |
| Every-update gradient checks | 235,008/235,008; maximum preclip norm 188.390472 |
| Minimum recorded clip scale | 0.005308124; AMP skipped steps 0 |
| Scheduled internal-health samples | 2,916/2,916 passed |
| Head SiLU sampled RMS | 0.110746–0.805425; maximum exact-zero fraction 0 |
| Minimum sampled informative input gradient squared norm | 2.722356×10⁻⁹ |
| Checkpoint checks | 18 models × 14 checks = 252/252 passed |
| Independent best-checkpoint CPU FP32 backward | 18/18 passed |
| Complete-validation CUDA FP32 Head replay | 4/4 maximum logit error exactly 0 |

最大事件的每个原生节点都参与局部编码；分类同时读取local、regional与global路径，不再只依赖固定24个latent。池化仍会压缩信息，本结果不称无损表示，也不将预训练收益归因于本轮未新增对照的native/rebin差异。压力测试原表为`stress/s4_empty_backbone_health_s42/measurements.parquet`。

21项验收均通过，包括全部配置、checkpoint、独立backward、每步梯度、非恒定预测、四类工作点计数、τ子道、完整分支干预统计、图表、资源终态和预算。新增分支验收直接检查5组干预、3个映射、400个有效replicates及有限区间。执行结果为`evaluation/empty_repair/acceptance.json`；技术通过与科学收益分别判断。

### 四分类结果

以下为3个独立训练seed的均值，AP、NLL、F1均为无量纲。

| Labels | Initialization | NLL | Macro AP | Macro F1 | Balanced accuracy | CC ντ AP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | Scratch | 1.564575 | 0.288856 | 0.263101 | 0.285033 | 0.261082 |
| 1,000 | MAE3 | 1.447382 | 0.345980 | 0.312576 | 0.333800 | 0.286128 |
| 10,000 | Scratch | 1.276154 | 0.425891 | 0.397094 | 0.402200 | 0.323674 |
| 10,000 | MAE3 | 1.252464 | 0.440384 | 0.402908 | 0.408433 | 0.331719 |
| 100,000 | Scratch | 1.214263 | 0.469263 | 0.430289 | 0.429833 | 0.351484 |
| 100,000 | MAE3 | 1.207805 | 0.473555 | 0.430321 | 0.429933 | 0.357508 |

MAE3−Scratch的配对差值，方括号为95%物理块区间：

| Labels | Δmacro AP | ΔNLL | ΔCC ντ AP |
| ---: | --- | --- | --- |
| 1,000 | +0.057124 [0.052411, 0.062256] | −0.117193 [−0.128214, −0.106363] | +0.025046 [0.018301, 0.033329] |
| 10,000 | +0.014493 [0.010374, 0.018721] | −0.023690 [−0.028177, −0.018498] | +0.008045 [0.002172, 0.014508] |
| 100,000 | +0.004292 [0.002020, 0.006590] | −0.006458 [−0.008634, −0.004276] | +0.006024 [0.001484, 0.010809] |

三个预算的每个seed均有正macro AP差值，其训练seed SD分别为0.010558、0.002741、0.000923。1k的MAE3平均NLL仍高于均匀先验ln4=1.386294，所有1k模型均选第一轮；不能因相对Scratch改善而称其概率质量已足够。1k和10k的τ argmax F1平均差值分别为−0.032139、−0.011317，对应物理块区间均为负，且训练seed SD很大（0.140789、0.083556）。100k的macro F1和balanced accuracy差值区间跨零。

100k逐类AP均值按CC νe/CC νμ/CC ντ/NC为：Scratch 0.609913/0.466050/0.351484/0.449605，MAE3 0.612880/0.469578/0.357508/0.454252。排序改善存在，但本轮不称所有分类与决策指标全面改善。

### τ子道、条件域与低背景工作点

Common-support下的总体τ AP差值在1k为+0.010941，[0.004116,0.018413]；10k为+0.005751，[−0.000183,0.011574]；100k为+0.003501，[−0.000353,0.007578]。100k common-support macro AP差值+0.002484的区间也跨零。因此raw总体收益不能直接解释为对能谱/位置先验独立的τ识别改善。

Hadronic τ子道与全部非τ背景比较的AP差值为：

| Labels | Raw ΔAP [CI95] | Common-support ΔAP [CI95] |
| ---: | --- | --- |
| 1,000 | +0.027774 [0.019130, 0.036820] | +0.015837 [0.007430, 0.025051] |
| 10,000 | +0.013564 [0.006446, 0.020635] | +0.009678 [0.003077, 0.015904] |
| 100,000 | +0.007824 [0.002506, 0.013679] | +0.004792 [0.000528, 0.009375] |

Electron、muon τ子道的对应AP区间均跨零。这里排除其他τ子道，common-support使用既定energy/vertex重权；不是完整拓扑或因果分离。

所有预算的τ效率在1%及0.1% FPR处，MAE3−Scratch的三seed差值区间均跨零。MAE3 seed42的实际validation计数为：

| Labels | Signal selected at 1% FPR | Background selected | Signal selected at ≤0.1% FPR | Background selected |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 33 / 2,500 | 75 / 7,500 | 1 / 2,500 | 7 / 7,500 |
| 10,000 | 69 / 2,500 | 75 / 7,500 | 6 / 2,500 | 7 / 7,500 |
| 100,000 | 89 / 2,500 | 75 / 7,500 | 14 / 2,500 | 7 / 7,500 |

7/7,500为0.09333%背景接受率；该尾部统计标记为不足。100k seed42在此处的τ效率为0.56%，描述性Clopper–Pearson 95%区间为[0.3065%,0.9378%]，背景接受率区间为[0.03753%,0.19221%]。所有四类的两档工作点均保留实际计数和区间，见精简JSON。阈值来自validation经验ROC；这些计数区间不提供独立工作点覆盖率或最终校准结论。

### 修复后分支的实际作用

正式Condor 1578703在H100 12GB MIG上完成4模型、完整10k validation的FP32诊断，耗时570.156 s。三个固定映射seeds为2026/17/42，保持模态presence；关闭TF32、compile，不更新权重。Head重放保留原batch形状及事件位置，4模型误差均为0。精简结果为[s4_empty_functional_paths_seed42.json](../roadmap/results/s4_empty_functional_paths_seed42.json)。

| Model / labels | Global permutation ΔNLL | CI95 |
| --- | ---: | --- |
| Scratch / 1k | 0.003667 | [0.002256, 0.005098] |
| MAE3 / 1k | 0.055214 | [0.048247, 0.062468] |
| Scratch / 10k | 0.00002515 | [0.00001470, 0.00003603] |
| MAE3 / 10k | 0.025673 | [0.021844, 0.030132] |

MAE3在两个预算下有可测global路径作用，regional/local路径也影响输出。Scratch 10k的global效应虽区间为正，但实际很小，其readout更依赖response summary。固定权重置换会破坏分支相关性，是拟合路径依赖诊断，不等于重训ablation、独有信息量或物理因果重要性，也不能将各分支增量相加。

三模态全空事件在train为2,790/100,000，在validation为282/10,000；validation类别计数为56/62/90/74。对于获得相同空输入的确定性分类器，其经验条件entropy为1.369617 nats，对全validation NLL的经验下界贡献为0.038623；即使全部非空事件完全正确，对应accuracy经验上界仍为0.9808。这是已接受catalog的描述统计，不是总体Bayes error测量。2.82%的空输入不能解释总体误差。

### 资源、终态与验证

| Backend / GPU | Cumulative device-hours |
| --- | ---: |
| Kubeflow / H100 47GB MIG | 8.076976 |
| HTCondor / H100 24GB MIG | 12.206944 |
| HTCondor / H100 12GB MIG | 3.970000 |
| HTCondor / V100 32GB historical attempt | 0.071944 |
| Total | 24.325865 / 60 |

这是S4全部历史与本轮allocation账本，包含失败、初始化、共享GPU和保守上界项；MIG按分区单列，不换算整卡或FLOPs。CPU allocation为282.757649 core-hours，按申请核数累计。正式分类峰值为5个MIG allocation上的13进程：47GB×1跑6进程，24GB×3各跑2进程，12GB×1跑1进程；主Pod内存峰值52,935,700,480 bytes，限额约63GB。MIG百分比GPU利用率未测量，保留实际并发、吞吐与内存记录。

新增A100 Kubeflow候选被namespace总GPU单位上限1拒绝，未分配设备。T4可参与的正式FP32诊断池通过3项入口拒绝检查和submission dry-run；该候选尚未获配时，12GB MIG已领取诊断，随后移除idle候选。所有训练由正式batch入口执行，claim避免重复writer。

Kubeflow主Job确认11个子进程exit0，Job于01:23:58 UTC成功完成，Pod已被平台清理。账本保存Job终态，将completion time作为Pod结束的保守上界，计4,657 s；随后完成Job清理并确认无残留Pod。新增accounting回归用例覆盖Pod消失但Job仍active、成功Job终态及后续缓存读回，相关4项测试通过。全部61个allocation终态均归档；本次最终清理18个与EOS存档完全一致的AFS提交文件，共21,521 bytes。

产物与证据：

- `evaluation/empty_repair/{summary.json,acceptance.json,bootstrap_statistics.npz}`；`experiments/s4_empty_validation_final/`。
- `evaluation/empty_repair/{label_efficiency.pdf,four_class_diagnostics.pdf,classification_training_curves.pdf,classification_internal_health.pdf,functional_paths_NLL.pdf,functional_paths_macro_AP.pdf}`。已人工查看标签效率、内部健康和分支NLL图，均为英文。
- `experiments/s4_empty_functional_paths_seed42/{functional_paths.json,manifest.json}`。
- `campaign_accounting.json`、`audit/backends/`、`audit/kubeflow_empty_main_release.json`、`audit/afs_cleanup_empty_final.json`。
- 实际提交脚本与schedd负载/权限依据保存在`logs/submission_*.json`，删除AFS副本不影响复核。

LCG108 CUDA环境、仓库src加入PYTHONPATH后，最终命令包括：

```bash
python -m unittest tests.test_flavour_s4_accounting
python -m analysis.flavour.s4.accept_empty \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 \
  --campaign roadmap/configs/s4_empty_repair.json \
  --comparisons roadmap/configs/s4_empty_pairs.json \
  --evaluation /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/evaluation/empty_repair/summary.json \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/evaluation/empty_repair/acceptance.json
```

两条命令均exit0，最终21项检查全部为true。架构修复的既有169项相关测试及真实小样本gate见前述阶段报告；本阶段新增accounting测试和最终真实产物验收补充资源与统计完整性证据。

## 阶段性结论

事实：修复后18项训练未重现空占位梯度异常，Head保持非零、真实input梯度保留；完整validation支持三个预算下的raw排序及NLL相对Scratch改善，低标签收益更大。MAE3的global路径在seed42的1k/10k检查中实际影响分类。1k概率质量、τ硬决策和低背景效率仍有明确不足；100k的common-support总体AP和τ AP收益未确认。

假设：1k多轮重复曝光、首次checkpoint过晚及readout/encoder共同更新可能造成过拟合和决策不稳。训练loss下降而validation NLL上升支持过拟合判断；更早checkpoint、降低encoder学习率、增强正则或LP-FT能否改善，尚未测量，不能归为已证实的架构病因。

## 下一步计划

本轮不扩至1M。先在原1k/10k预算内固定分层train-inner folds，对相同Scratch/MAE3比较更早checkpoint、较少重复曝光、学习率/正则和LP-FT；选择过程使用的标签计入原预算，再对全预算重拟合。LP-FT依据及适配限制见[S4路线图](../roadmap/archive/2026-11-14_v1/11_s4_pretraining.md)；不在当前validation拟合温度或class bias。验收同时要求NLL相对均匀先验、τ argmax及低FPR计数，不只看AP。新的方案记录为已观察本轮validation后设计；最终calibration/test和系统效应仍按后续冻结阶段执行。
