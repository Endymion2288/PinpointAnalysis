# 阶段55：S2科学总报告与最终验收

S2已完成：35项科学训练、490项CPU读回检查、41项单seed配对和4项三seed均值全部齐备，严格评价与`--require-complete`均exit0。组织实际成本92.73055556 GPUh，所有149条后端记录终止、缺失历史为0。

科学结论：当前共享压缩、四轮预算下，native相对1 mm的三seed τ AP平均收益未确认；两架构τ AP平均差也未确认。Bar在pixel之上的增量有正区间支持；训练域覆盖会明显影响全域评价。Tracker与ParT在部分四分类指标上的点估计较好，τ单指标不足以决定最终架构或模态。结论适用于本次detector-proxy和训练条件。

## 阶段目标

回答原生pixel、投影关联、探测器模态与跨层消息在统一比较条件下贡献了什么；交付可复核的四分类基线、统计区间、资源/错误诊断和S2 gate decision。S2的验收目标是形成有数据证据的结论，未要求所有新方法都获得正收益。

## 工作内容与冻结合同

- 输入run映射：10081→CC nu e，10082→CC nu mu，10083→CC nu tau，10084→NC，含对应反中微子；主类别顺序固定。
- 80,000 train、10,000 validation，各类20,000/2,500；accepted-family分组split和现有detector-proxy沿用用户决定。主模型只用硬件观测，E/vertex、target和τ分支仅作诊断。
- 原生binary pixel与聚合bar信号为主输入，accepted tracker单独消融。两族都用同平面4 mm、每块至多32 nodes的learned compression，处理所有全局块；没有按事件占用截断。
- ParT dim48、151,228参数，ParticleNet dim64、161,156参数；这比较共享压缩后的实现，不代表未经压缩全事件ParT或原始硬件信息的性能上限。
- H100 NVL、LCG108、Python3.12.11、Torch2.7.1、BF16，batch4，四轮，每轮20,016 attempted steps。35项均80,064 applied updates、0 AMP skips；各项实际exposures从自己的history读取。
- 主checkpoint按full-validation NLL选择。条件域指标沿用同一主checkpoint；自身域posthoc选点单列，不替代主结果。

## 详细数据

### 完整性与统计口径

| Evidence | Measured value |
| --- | ---: |
| configurations | 35 |
| manifest_status | {'complete': 35} |
| output_complete | 35 |
| paired_comparisons | 41 |
| seed_mean_comparisons | 4 |
| pending_comparisons | 0 |
| evaluated_results | 39 |
| strata | 1051 |
| historical_three_class_tables | 117 |
| factorial_interactions | 2 |
| readback_checks_passed | 490 |
| classwise_metrics | 156 |
| tau_branch_rows | 195 |
| model_target_groups | 273 |
| terminal_backend_records | 149 |
| optimizer_steps_applied | 2802240 |
| AMP_skipped_steps | 0 |
| actual_training_exposures | 11204668 |

评价代码commit：`d7fed400d327158e8871332613f99b53bc6a31fe`；最终accounting时刻：`2026-11-12T13:27:05.917183+00:00`。配置来自`roadmap/configs/s2_h100_v2.json`，完整比较来自`s2_complete_pairs_v2.json`，均未更改科学合同。

400次accepted-family/chunk bootstrap，128 blocks，每类32。所有CI均为validation上条件于已拟合模型的pointwise事件区间，未作多重比较校正。三seed使用相同物理事件重采样后平均每seed AP差，不拼接为30k事件，也不构造ensemble；训练sample SD另列，不能用事件CI代替未来训练seed的总体区间。

固定粗E/x/y/z联合支持保留8,559事件、Neff=7,959.41365，未匹配target Z或同位素；这不是去掉全部物理先验。Inclusive AP使用25% τ benchmark prior，条件域和分支有各自prior。工作点为validation经验曲线，不是最终测试阈值。0.1% FPR最多7个背景，低于预定25事件尾部标准。

### 十二项空间配置

| Family / representation | Tau AP | Fixed-support AP | BA | Tau efficiency at 1% FPR | Allocation [GPUh] |
| --- | ---: | ---: | ---: | ---: | ---: |
| part_100um_joint_s42_e4_v2 | 0.35025881 | 0.31715126 | 0.4135 | 0.033600 | 1.92166667 |
| part_1mm_joint_s42_e4_v2 | 0.35502857 | 0.32086106 | 0.4033 | 0.035200 | 2.02166667 |
| part_1mm_projection_s42_e4_v2 | 0.35421688 | 0.31995100 | 0.4137 | 0.036800 | 1.80083333 |
| part_200um_joint_s42_e4_v2 | 0.35513011 | 0.31829366 | 0.4101 | 0.031600 | 1.71472222 |
| part_native_joint_s42_e4_v2 | 0.35858391 | 0.32383619 | 0.4184 | 0.037200 | 1.94250000 |
| part_native_projection_s42_e4_v2 | 0.35733812 | 0.32248272 | 0.4150 | 0.032800 | 2.78250000 |
| pn_100um_joint_s42_e4_v2 | 0.35548005 | 0.32148225 | 0.4040 | 0.030800 | 1.54805556 |
| pn_1mm_joint_s42_e4_v2 | 0.35303603 | 0.32318232 | 0.3977 | 0.035600 | 1.43638889 |
| pn_1mm_projection_s42_e4_v2 | 0.35100848 | 0.32166112 | 0.4048 | 0.032000 | 1.09916667 |
| pn_200um_joint_s42_e4_v2 | 0.36010319 | 0.33309115 | 0.4053 | 0.038000 | 1.27888889 |
| pn_native_joint_s42_e4_v2 | 0.35510474 | 0.32202830 | 0.3974 | 0.033200 | 1.75277778 |
| pn_native_projection_s42_e4_v2 | 0.34797906 | 0.31945295 | 0.3962 | 0.028000 | 1.02305556 |

分辨率曲线不单调；seed42的native−100 μm与三seed的native−1 mm回答不同问题。不能从单个正配对推断所有粗粒度都稳定劣于native。两个native/1 mm×joint/projection的交互如下，均跨0：

| Family | Interaction: CP−CJ−NP+NJ [95% CI] |
| --- | --- |
| part | +0.00043410 [-0.00790547,+0.00875624] |
| particlenet | +0.00509813 [-0.00510744,+0.01683727] |

### 信息、模态、结构与训练域控制

| B−A contrast | Raw tau AP difference [95% CI] | Fixed E/vertex difference [95% CI] |
| --- | --- | --- |
| ParT native−100 um | +0.00832510 [+0.00282045,+0.01378043] | +0.00668493 [+0.00166274,+0.01158621] |
| PN joint−projection (native) | +0.00712568 [+0.00003031,+0.01448372] | +0.00257536 [-0.00352278,+0.00843062] |
| ParT main−local shuffle | +0.00847852 [+0.00281484,+0.01296334] | +0.00794723 [+0.00257296,+0.01269657] |
| PN main−local shuffle | +0.00396180 [-0.00300075,+0.01075346] | +0.00060939 [-0.00618280,+0.00742795] |
| ParT pair bias−no pair | +0.00242736 [-0.00350744,+0.00865689] | +0.00327901 [-0.00220454,+0.00858433] |
| PN dynamic−fixed graph | +0.00075113 [-0.00567937,+0.00685530] | -0.00128660 [-0.00712364,+0.00463922] |
| PN pixel+bar−pixel | +0.06890803 [+0.05663332,+0.08443211] | +0.04851078 [+0.03602058,+0.06154046] |
| PN pixel+bar−bar | +0.01073932 [+0.00463872,+0.01687213] | +0.00180202 [-0.00491768,+0.00812752] |
| PN +tracker−pixel+bar | -0.00424866 [-0.01307753,+0.00369210] | -0.00282447 [-0.01101520,+0.00480946] |
| PN full−center training | +0.05449639 [+0.04020247,+0.06874907] | +0.03210627 [+0.01756088,+0.04732816] |
| PN full−large training | +0.03136247 [+0.02176243,+0.04088759] | +0.01862325 [+0.00830622,+0.02916431] |
| ParT cross-plane−plane-only | +0.00803127 [-0.00049395,+0.01671763] | +0.00574564 [-0.00196088,+0.01402978] |
| PN cross-plane−plane-only | +0.00574514 [-0.00169270,+0.01314704] | +0.00521402 [-0.00106270,+0.01159757] |

原生同平面shuffle在ParT上的AP正区间同时出现在raw/common支持中；shuffle也改变后续局部块组织，不能隔离为单一物理关联因素。Pair-bias、fixed/dynamic、OR/count和半格原点控制的区间跨0均不构成一般信息等效。完整41项配对、工作点区间和两项跨层结论在版本化数值表及[阶段54](2026-11-12_stage54_s2_cross_plane_controls.md)中保留。

Bar在pixel之上有明确正区间，1% FPR下τ为31/2,500→83/2,500，效率差CI=[+0.01034517,+0.03158505]。Pixel在bar之上的raw AP为正，固定支持区间跨0；中心/large条件结果见阶段43–44。Tracker令输入全空440→282，seed42 BA 0.3974→0.4367、NLL 1.23930563→1.20936304，nu_mu AP 0.41459288→0.47679774、NC AP 0.36957463→0.43971042；这些非τ/综合差值CI未测量。

Full−center训练的固定支持增量也为正，但中心条件域AP差区间跨0；full−large在large域也为正。Center训练7,429 unique/319,233 exposures，large训练35,936 unique/320,039 exposures，使用各自域inverse-frequency类别权重。训练域比较同时改变unique数量、重复与覆盖，不能只归因于空间位置。各域NLL学习曲线和posthoc自身域选点见阶段46/49。

### 四组三seed确认

| Prespecified contrast | Raw mean delta [95% event CI] | Fixed-support mean delta [95% CI] | Raw delta training sample SD |
| --- | --- | --- | ---: |
| part_native_benefit_three_seed_mean | +0.00079004 [-0.00282494,+0.00445758] | -0.00201089 [-0.00526091,+0.00111996] | 0.00553825 |
| pn_native_benefit_three_seed_mean | -0.00013880 [-0.00454523,+0.00421196] | -0.00134447 [-0.00522708,+0.00286726] | 0.00519741 |
| part_benefit_native_three_seed_mean | +0.00219679 [-0.00401226,+0.00795043] | -0.00244385 [-0.00832807,+0.00308100] | 0.00518276 |
| part_benefit_1mm_three_seed_mean | +0.00126795 [-0.00485443,+0.00824882] | -0.00177744 [-0.00654021,+0.00456009] | 0.00480552 |

四组raw/common区间均跨0，上界均低于+0.01；可排除当前已拟合条件下该平均收益，不能宣称未来seed、更多轮数或更强编码器下普遍等效。逐seed方向和训练spread见阶段50–53及`three_seed_four_class_summary.json`。

| Three-seed configuration | e AP | mu AP | tau AP | NC AP | BA | BA sample SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| part native | 0.60310732 | 0.45086405 | 0.35457262 | 0.41744087 | 0.41353333 | 0.00444110 |
| part 1mm | 0.60414105 | 0.45033550 | 0.35378258 | 0.41797796 | 0.41656667 | 0.01150014 |
| pn native | 0.59654370 | 0.41101658 | 0.35237583 | 0.36720646 | 0.39480000 | 0.00301164 |
| pn 1mm | 0.59737415 | 0.41503406 | 0.35251462 | 0.37771866 | 0.40000000 | 0.00347707 |

ParT的nu_mu/NC和综合指标描述均值较高，相关差值CI未测量；不以τ的未确认差异替代四分类判断，也不以描述均值宣称已确认架构优胜。

### B0、τ分支、接受度与空响应

| Model | Tau AP | BA | NLL |
| --- | ---: | ---: | ---: |
| b0_all_s42 | 0.34471068 | 0.40290000 | 1.24504237 |
| part_native_joint_s42_e4_v2 | 0.35858391 | 0.41840000 | 1.22107879 |
| pn_native_joint_s42_e4_v2 | 0.35510474 | 0.39740000 | 1.23930563 |
| pn_native_tracker_s42_e4_v2 | 0.35085607 | 0.43670000 | 1.20936304 |
| truth_energy_vertex_s42 | 0.44106095 | 0.34990000 | 1.32108745 |

Truth E/vertex模型仅作先验诊断，不能用作硬件分类器。B0与深度模型的全曲线/配对区间保存在主评价，不用单个argmax accuracy替代PR或工作点。

| Model | Tau branch | Signal count | Branch AP | Efficiency at inclusive 1% FPR |
| --- | --- | ---: | ---: | ---: |
| part_native_joint_s42_e4_v2 | electron | 445 | 0.08132916 | 0.01573034 |
| part_native_joint_s42_e4_v2 | muon | 395 | 0.06071660 | 0.01012658 |
| part_native_joint_s42_e4_v2 | hadronic | 1660 | 0.29800798 | 0.04939759 |
| pn_native_joint_s42_e4_v2 | electron | 445 | 0.07676873 | 0.01573034 |
| pn_native_joint_s42_e4_v2 | muon | 395 | 0.05989884 | 0.01012658 |
| pn_native_joint_s42_e4_v2 | hadronic | 1660 | 0.29638621 | 0.04337349 |
| pn_native_tracker_s42_e4_v2 | electron | 445 | 0.07600838 | 0.01797753 |
| pn_native_tracker_s42_e4_v2 | muon | 395 | 0.05657664 | 0.01012658 |
| pn_native_tracker_s42_e4_v2 | hadronic | 1660 | 0.29306329 | 0.03433735 |

各分支使用该分支τ与全部7,500非τ背景，不将其他τ分支当背景；分支AP的prior不同，不直接横比为inclusive收益。Unknown/ambiguous也保留在全部195行分支诊断中。中心885/large4,494事件、能量/vertex/占用及tracker有无等1051行条件表和117张历史三类条件表均已生成。历史三类表来自四类模型去除真τ后重归一化，不是重训旧模型或复现旧研究。

Validation proxy全空282/10k（90τ），pixel+bar输入全空440（148τ），只有accepted tracker响应158（58τ）。ParT native42在输入全空域AUC=0.5、AP=148/440=0.33636364，在全空域AP=90/282=0.31914894；输入active的9,560事件τ AP=0.35691643、AUC=0.63309985，接近全域AP=0.35858391。因此少量全空事件不能单独解释τ困难；secondary可见性、shower重叠和编码器压缩等解释尚未隔离。

Target诊断覆盖90k事件，14个target PDG、8个Z；validation有7个Z，W占92.76%。最终39结果×7Z=273组，与主评价模型集合完全一致。它们是truth组成/条件点估计，不是额外匹配或硬件输入；小target子样本差值CI未测量。

### 资源、信息toy与错误显示

组织终态成本92.73055556 GPUh、373.02637509 CPU core-hours；H100正式结果目录范围56.49222222 GPUh。组织成本包含旧V100先导、被替代/中断/恢复、probe及两后端分配；GPUh不是同型号性能换算。35模型实际allocation与program wall、额外分配分别保留，未用program wall填补未知成本。

| Equal-allocation comparison (seed42) | Eligible epochs A/B | Delta tau AP B−A | Budget [s] |
| --- | --- | ---: | ---: |
| part_1mm_joint_s42_e4_v2 / part_native_joint_s42_e4_v2 | 3/4 | +0.00447383 | 6992.77439 |
| pn_1mm_joint_s42_e4_v2 / pn_native_joint_s42_e4_v2 | 4/3 | -0.00585053 | 5170.45471 |
| pn_native_joint_s42_e4_v2 / part_native_joint_s42_e4_v2 | 4/3 | +0.00074505 | 6309.67797 |
| pn_1mm_joint_s42_e4_v2 / part_1mm_joint_s42_e4_v2 | 4/2 | -0.02844170 | 5170.45471 |

同allocation仅在已保存整轮checkpoint上按预算内最低NLL选点，报告unused seconds和开销；上表是点估计、CI未测量，不能替代主四轮更新匹配。节点和运行时段不同，不把wall差直接解释为纯架构原因。PN fixed graph实际allocation较dynamic低24.06%，质量区间与其资源优势一起保留。

完整资源压力表576行，每架构/表示24个占用秩事件，batch1 BF16，包括最大占用；第一事件warmup，compute统计排除它。Native/1 mm最大nodes=36,547/8,417，blocks=2,641/1,830。训练forward+backward最大ParT=0.31984009/0.23518436 s，PN=0.16380406/0.11723026 s；训练峰值ParT=241.13574/206.72363 MiB，PN=597.21387/595.29395 MiB。这不是总体均值或端到端batch4吞吐，profile未执行optimizer fitting。

Toy共420行，用真实pitch 0.0208/0.0228 mm和plane z，展示双轨support、kink、shower overlap和projection pairing碰撞。Toy为binary/no-added-noise，未含材料散射、未拟合轨迹，只证明表示层面的信息保留；不能据此声称真实kink分辨或τ探测效率。

两族native seed42各20页固定错误组/随机参照显示已查看并保留review；全部39模型四分类/ROC/score/学习曲线PDF已生成，新增plane-only页与跨层图的实际审阅见阶段54。Truth图注仅作解释，不进入模型。

Kubeflow已按CERN实际flavor落地：整卡`h100-94gb`为`NVIDIA-H100-NVL`与`nvidia.com/gpu: 1`，CPU/RAM由平台注入46/127G；`h100-47gb`是`nvidia.com/mig-3g.47gb`，不混入本次整卡科学矩阵。GPU Pod server dry-run实际通过；v9始终无node，全部剩余科学任务在Condor运行后主动缩短deadline释放配额，实际0 GPUh。CPU容器EOS/CVMFS/LCG路径已实测，Kubeflow GPU runtime未测量；没有把dry-run或Pending当作GPU运行成功。完整依据与路径见[08_kubeflow](../roadmap/08_kubeflow.md)。

### 复现命令与产物

父目录`S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11`，正式根为`$S2_RESULTS/h100`。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json"
python -m analysis.flavour.s2.target_diagnostics --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/target_diagnostics"
python -m analysis.flavour.s2.output_inventory --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --require-complete
```

上述四步顺序真实exit0，完整命令、起止UTC和日志在`strict_final_pipeline.json`；日志为`*_strict_final.log`。最后训练14项读回见`readback_pn_planeonly.log`；490检查逐项来自各模型`final_checkpoint_checks.json`。本阶段只更新报告和完成元数据，已通过的核心回归未重复运行。

- [版本化S2数值表](../roadmap/results/s2_summary.json)：39结果四分类/τ指标、成本、41配对/4均值/2交互与严格计数。
- `evaluation_final/summary.json`及`metrics.csv`：完整条件表、主配对、bootstrap与预算选点。
- `s2_scientific_tables.md`、`three_seed_four_class_summary.json`：可读总表与四分类seed描述统计。
- `s2_final_acceptance.json`、`s2_requirement_audit.json`：严格门槛与16项研究要求证据。
- `resource_profile_h100_v2/final_stress_resource_table.json`、`toys/`位于父目录；图与大产物均留在results。

## 阶段性结论与gate decision

**验收：S2通过。**完整科学合同、终态、CPU读回、统计评价、资源和中文解释全部落实；用户接受的sample/proxy决定已执行。Calibration/test未用，S3–S6未执行。

**方法晋级：当前证据不支持仅以τ AP为由宣布native或ParT相对1 mm/PN全面胜出。**三seed平均差的事件区间排除当前条件下+0.01收益，训练spread和更多预算下结论仍独立待测。Bar贡献与完整训练域覆盖有明确正区间支持；保留这些信息。Tracker的四分类改善值得在后续同预算多指标确认，不能按τ AP单项丢弃。

事实与假设分开：本次可复核地测到了读出表示、模态和当前编码器行为；没有测量真实硬件标定、生成器迁移、系统包络、最终测试表现或预期物理产额。τ拓扑可能需要更适合的局部编码和辅助任务，这仍是下一阶段假设。

## 下一步计划

S3工单沿用本次冻结输入、预算记录和评价口径，以PN/ParT的native及1 mm三seed作为基线；优先检验plane-aware局部编码、module-aware context/latent fusion和多模态四分类，先区分局部压缩与跨层消息的作用。保留bar，并为tracker同时报告四类收益、覆盖与成本。晋级仍需同预算配对区间、逐类指标和资源证据；本次不启动S3训练。

S4的FASERCal式MAE/Rel保持同编码器/标签与总算力对照；S5再隔离τ分支、charm、secondary可见性和重建覆盖；S6在模型/阈值/校准冻结后执行独立test与系统效应。用户接受的proxy值继续使用。
