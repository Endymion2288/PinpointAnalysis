# Stage284：R2完整结果、顺序效应与下一批科学决策

**R2已完成，198项验收全部通过。当前配方中，300k独立事件优于重复100k事件；增强正则化和末轮重混均能恢复部分性能，但尚未建立收敛或最优性。参考保留D2的真实三类与四类固定终点，下一批先检验全程重混，再单独检验局部几何信息路径。**

## 阶段目标

回答更多unique事件、更多重复曝光、正则化和训练顺序各自能解释多少性能变化，完成三类/四类、稀疏域、训练健康、收敛和成本审查，并决定是否触发更长训练、1M数据或加宽。交付八模型完整矩阵、必要补充控制、明确参考模型和下一批可执行工单。

R2结果是seed42的Tune开发比较。独立Confirm、Cal-A和Cut-B保持封存；本阶段结束表示已完成上述决策批次。

## 工作内容与比较合同

使用R0冻结的accepted Train100k/300k与Tune40k。Tune原四类各10k，共同三类使用相同30k非τ事件；T3训练排除τ，T4的条件三类概率在e/μ/NC之间重归一化。各域按域内均匀类别先验评价，保留原始支持数。τ效率在Tune非τ背景的1% FPR处读回，尚非Cut-B冻结阈值后的独立确认。

所有模型为fresh B1 ResNet，width16、batch8、seed42、LR=.0003、BF16 encoder/FP32 head；T4实际793,674参数。各自完整updates的前5% warm-up，随后cosine至5% LR floor。D1WD只将AdamW weight decay从.01改为.1。

| Design | Four-class pool | Passes | T3 exposures / updates | T4 exposures / updates | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 | 100,000 | 4 | 300,000 / 37,550 | 400,000 / 50,051 | .01 |
| D1 | 100,000 | 12 | 900,000 / 112,645 | 1,200,000 / 150,153 | .01 |
| D2 | 300,000 | 4 | 900,000 / 112,630 | 1,200,000 / 150,117 | .01 |
| D1WD | 100,000 | 12 | 900,000 / 112,645 | 1,200,000 / 150,153 | .1 |

T3实际unique监督行数为表中pool的3/4。D1/D2匹配曝光，mixed-shard尾批带来小量update差；D2−D0同时改变数据和曝光。完整LR轨迹、batch大小、`sum(lr)`和逐行曝光已经保存并核对。

每75k非τ曝光评价固定4k Tune screen。固定预算`exposure.pt`为主比较；按own-task最低screen NLL选择的`best.pt`为次比较，两者均评价完整40k Tune，共16视图、128配对、11域。D1有更多screen选点机会，报告保留两种策略，不将best-screen差异直接解释成固定预算收益。

配对统计使用400次family-connected physical-chunk bootstrap，seed20261116；本次6,845个physical chunks形成6,845个blocks。以下95%区间描述已拟合模型在评价事件上的变化，未估计训练seed或Train pool抽样变化。所有八个endpoint另用相同physical-chunk五折做两个global bias和正温度readout，每折原四类各2k。

## 详细数据

### 完整40k Tune原始固定终点

| Model | BA3 (%) | NLL3 | BA4 (%) | NLL4 | Tau eff @ 1% FPR (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 T3 | 56.6500 | .871321793 | — | — | — |
| D1 T3 | 53.1700 | 1.143738363 | — | — | — |
| D2 T3 | 57.0967 | .855866803 | — | — | — |
| D1WD T3 | 54.6133 | .962710252 | — | — | — |
| D0 T4 | 56.5333 | .864586107 | 44.7850 | 1.185567841 | 5.11 |
| D1 T4 | 53.7300 | 1.015106150 | 42.3700 | 1.337161873 | 4.38 |
| D2 T4 | 57.1267 | .849030484 | 45.7650 | 1.166233178 | 6.05 |
| D1WD T4 | 54.9767 | .915119484 | 43.6075 | 1.226066265 | 5.29 |

T4的BA3/NLL3均为共同非τ任务。NLL使用自然对数。完整逐类、工作点、域、best-screen及资源计数见[最终结果JSON](../roadmap/v2/r2_results.json)。

| Endpoint contrast | Difference | 95% interval |
| --- | ---: | --- |
| D2−D1 T3 BA3 | +3.926667 pp | [3.380711, 4.438761] |
| D2−D0 T3 BA3 | +.446667 pp | [.077522, .815348] |
| D2−D0 T4 conditional BA3 | +.593333 pp | [.181383, .981433] |
| D2−D0 T4 BA4 | +.980000 pp | [.622120, 1.375480] |
| D2−D0 tau eff @ 1% FPR | +.940000 pp | [.419277, 1.566227] |
| D2 T3−conditional T4 BA3 | −.030000 pp | [−.392504, .340599] |
| D2 T3−conditional T4 NLL3 | +.006836319 | [.004669564, .009040244] |
| D1WD−D1 T3 BA3 | +1.443333 pp | [.994768, 1.942180] |
| D1WD−D1 T4 conditional BA3 | +1.246667 pp | [.752100, 1.720469] |

同曝光下更多unique事件的收益在当前配方中成立。D2相对短D0的BA3增量低于预设2 pp实用门槛；T3的e/μ/NC recall差为−2.25/+5.28/−1.69 pp。D2 T3 recalls为60.45%/55.44%/55.40%，conditional T4为60.50%/54.33%/56.55%；三类化仍有μ/NC取舍。

τ正增量成立，但.940000 pp点值低于1 pp晋级门槛。D2 T3/conditional T4的BA差区间跨零，T4 NLL更好，未建立所有逐类非劣。因此保留真实T3交付及T4研究，进一步细分留给R5条件辅助头。

### 稀疏域、gap与readout

稀疏域定义为accepted pixel+bar+tracker channels<100，包含空响应；e/μ/τ/NC支持数为2,971/2,801/3,371/3,010。

| Endpoint | Sparse BA3 (%) | Sparse mu AUROC |
| --- | ---: | ---: |
| D0 T3 | 39.4008 | .545741946 |
| D2 T3 | 38.6875 | .548284293 |
| D2 T4 conditional | 38.5634 | .551839475 |

D2−D0 T3稀疏BA3为−.713293 pp [−1.605710, +.083226]，没有建立净改善。μ recall上升伴随e/NC损失，排序仍弱；空响应仅为域内一部分，当前证据不能将整个稀疏域判为物理不可分。

共同Train诊断每类1k，使用eval mode。D1 T3 Train/Tune NLL为.626982597/1.143738363，gap=.516755766；D1WD为.780739158/.962710252，gap=.181971094。D1WD T4条件三类/四类gap为.071002716/.070346988，D2 T3为.034921150，D2 T4为.019469321/.017612684。增强weight decay减轻长重复配方的泛化差距，但两个固定终点仍低于D0；一个系数不足以建立最佳正则化。

OOF global bias没有改善D0/D2总体BA。D1WD T3 raw→bias的BA3为54.6133%→55.3967%，NLL为.962710252→.917961395；D1WD T4为54.9767%→56.2233%，NLL为.915119484→.869241001。偏移可恢复量随配方变化。正温度保持argmax并降低部分NLL；这是Tune开发诊断，R7仍单独拟合最终校准。

### 固定prefix末轮顺序控制

触发证据是D1/D1WD两个任务在第12轮末端共同偏向e、损失μ，原末100 batch的e/μ/τ/NC为289/97/213/199。冻结`b1_d1wd_t4_s42/best.pt`第11轮末prefix：position=(10,12513)、137,640 updates、1.1M曝光、每行11次。两分支复用模型、AdamW和RNG状态，追加相同100k行、相同逐update batch大小和12,513 updates，沿原150,153总updates的LR轨迹，tail `sum(lr)=.21017710552966024`。

原顺序与label-independent全池随机排列（seed20261118）只改变末轮次序；重混末100 batch为207/209/189/193。原顺序对保存endpoint的全部概率max/mean绝对差均为0，argmax agreement=1。两分支各14个health windows均通过，每行最终均12次曝光。

| Replay | Common BA3 (%) | NLL3 | BA4 (%) | NLL4 | Tau eff @ 1% FPR (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 54.9767 | .915119484 | 43.6075 | 1.226066265 | 5.29 |
| Global random | 56.2533 | .865668386 | 44.8900 | 1.185202789 | 5.40 |

| Remixed−original | Difference | 95% interval |
| --- | ---: | --- |
| Common BA3 | +1.276667 pp | [.810168, 1.803464] |
| NLL3 | −.049451098 | [−.051917887, −.047035086] |
| Macro AP3 | +.003140572 | [.001719936, .004475272] |
| Macro AUROC3 | +.002604028 | [.001509131, .003793788] |
| BA4 | +1.282500 pp | [.903090, 1.678340] |
| Tau eff @ 1% FPR | +.110000 pp | [−.316426, .424990] |
| Sparse BA3 | +.422271 pp | [−.530633, 1.407653] |
| Sparse macro AUROC3 | +.000143546 | [−.005449805, .006529967] |

e/μ/NC recall差为−22.68/+25.56/+.95 pp。顺序影响在这个prefix上成立，微小排序增益与大量决策再分配同时存在；稀疏排序和τ工作点改善尚未建立。重混后仍未超过原D2参考。

该prefix原先由Tune screen选出，控制只回答条件末轮问题；fresh全程重混效果未测量。本控制单列，不改写原八模型矩阵，也不能将全部重复训练退化归于过拟合或全部归于顺序。文献依据为[Random Reshuffling，M9](../roadmap/v2/05_literature.md#m9)，官方摘要已联网读取；其gradient-descent理论不直接给出本项目AdamW的收益预测。

缓存使用同allocation-local transformed arrays，100k事件3,482,748,960 bytes，构建46.037197 s，32个分散事件缓存前后模型输入完全相同。aggregate RSS峰值10,157.223 MiB，整个重放operation wall637.740836 s。Kubeflow `pinpoint-r2-sampling-k5`于13:23:49 UTC Complete；CPU正式评价Condor1582992.0在bigbird20终态。临时数组未持久保存至EOS。

### 收敛、容量与条件扩展决定

八模型均未满足“连续三个screen NLL变化绝对值<.001”的工程plateau。D1/D1WD后段恶化，D2仍有波动/改善。matching BF16的width16三/四类32-event真实拟合均为100%，encoder/tracker/head梯度与参数更新有效，中途恢复最大参数差为0。小样本拟合证明优化路径有效，未证明大样本容量充分。

| Extension | R2 decision | Basis |
| --- | --- | --- |
| D1WD | Completed | 单因素改善重复训练，但固定终点仍落后D0 |
| Same-prefix final-pass replay | Completed | 末轮偏移与局部类组成构成触发证据，原顺序精确复现 |
| D3: 300k×12 | Deferred | 原长重复路径退化，D2未证明plateau，三倍曝光价值未建立 |
| D4: 1M | Deferred | 相对D0的共同BA与τ点效应未达实用门槛，需新证据和资源计划 |
| Width32 | Deferred | 稳定Train/Tune双侧停滞尚未建立，容量问题保持开放 |
| Second native family | R3 controlled study | 使用本族关系对照，先profile并验证真实学习 |

## 验证、产物与资源

主输出根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2`。训练及checkpoint在`experiments/`，正式评价在`evaluation/`，OOF在`readout/`，补充控制在`sampling/`，终态账本与决策分别为`resources.json`和`decision.json`。仓库只保存代码、合同、[紧凑最终证据](../roadmap/v2/r2_results.json)与报告。

| Validation | Result |
| --- | --- |
| Final R2 acceptance | 198 passed; failed=[] |
| Main matrix | 8 models; 16 views; 128 pairs; 11 domains |
| Independent main points / sklearn rankings | 3,976 passed |
| Independent main core intervals | 1,808 passed |
| Sampling replay | 3 views including parent endpoint; 8 pairs; acceptance complete |
| Independent replay points / core intervals | 1,026 / 115 passed |
| R0/R2/readout/sampling unit tests | 24 passed |
| Cal-A / Cut-B / Confirm payload reads | None |

以下六组正式PDF/PNG已检查英文标注及可读性。较早的`*_partial`为历史快照；最终图为下列无partial后缀文件。

| Figure | Artifact |
| --- | --- |
| Learning curves | [PDF](../../results/flavour/v2_2026-11-14/R2/figures/r2_learning_curves.pdf) |
| Scaling endpoints | [PDF](../../results/flavour/v2_2026-11-14/R2/figures/r2_scaling_endpoints.pdf) |
| Low occupancy | [PDF](../../results/flavour/v2_2026-11-14/R2/figures/r2_low_occupancy.pdf) |
| Readout diagnostic | [PDF](../../results/flavour/v2_2026-11-14/R2/figures/r2_readout_diagnostic.pdf) |
| Train–Tune gap | [PDF](../../results/flavour/v2_2026-11-14/R2/figures/r2_train_tune_gap.pdf) |
| Sampling replay | [PDF](../../results/flavour/v2_2026-11-14/R2/figures/r2_sampling_replay.pdf) |

资源账本时点为2026-11-15 13:30:44 UTC，39条allocation全部终态，`active_or_unresolved=[]`。总计7.475833333异构device-hours、116.65 allocated CPU core-hours，包含失败、探针和恢复。同卡多个独立模型共享一次allocation，记账一次。

| Device / partition | Allocated device-hours |
| --- | ---: |
| H100L MIG 2g.24gb | 4.026111111 |
| H100L MIG 1g.12gb | .173888889 |
| A100-PCIE-40GB | .061666667 |
| H100 NVL MIG 3g.47gb | 3.214166667 |

另有一次EOS路径策略提交拒绝，未产生allocation。R2交互operation下界.177657235 h另列，完整交互allocation开销未测量。R0上界+R1/R2 batch累计54.269576868 h，原80 h规划名义余量25.730423132 h；该余量尚未扣除完整R1/R2交互。历史SSL单列，异构device-hours不折算成相同FLOPs。

继续同时使用HTCondor和Kubeflow，按预计完成时间选择兼容GPU、共享设备或MIG，不要求同型号或整卡。R2实测H100 MIG3g.47gb四进程总operation吞吐311.40 events/s，单进程190.04 events/s；后续并行度以真实输入、RAM/VRAM和完整validation峰值为准，各模型独立optimizer/checkpoint。批量处理与统计继续使用HTCondor，提交前检查schedd兼容性和负载。

复核命令在仓库根目录执行，运行前使用匹配的CVMFS环境：

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
R2_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2
python -m unittest tests.test_flavour_v2_r0 tests.test_flavour_v2_r2 \
  tests.test_flavour_v2_r2_readout tests.test_flavour_v2_r2_sampling
python -m analysis.flavour.v2.r2.accounting --root "$R2_ROOT"
python -m analysis.flavour.v2.r2.accept --root "$R2_ROOT"
python roadmap/tools/export_r2_final_review.py --root "$R2_ROOT" \
  --output roadmap/v2/r2_results.json
```

24项tests已通过，最终accounting、accept及export均实际执行成功。独立概率、排序和区间检查使用已完成评价产物；最终交付复核未另启训练。

交付前另核对3份JSON解析、2份修改Python的语法、388处本地文档链接及入口状态/成本一致性，全部通过；本报告八模型主表与accepted endpoint逐项一致，`git diff --check`通过。

## 阶段性结论

事实：更多unique优于原重复配方；增强正则化减少gap；固定prefix的顺序效应由精确重放建立。原始D2 endpoint为本批参考：T3 BA3=57.0967%，T4 conditional BA3=57.1267%、BA4=45.7650%、τ在1% FPR效率6.05%。稀疏净分离改善、训练收敛和容量上限均未建立。

待检验假设：全程事件重混能否稳定改善fresh训练；B1投影降采样或原生压缩前关系是否丢失了可用局部信息。当前结果不能确定哪一项是主要瓶颈。后续各项需独立对照，保留逐类与工作点约束。

未测量：fresh全程重混、D3/1M/width32效果、新几何/对象/τ辅助/定向SSL收益、新Confirm上的泛化、真实硬件系统包络、物理通量和τ产额、完整交互allocation成本。

## 下一步计划

固定参考为`b1_d2_t3_s42/exposure.pt`与`b1_d2_t4_s42/exposure.pt`，best-screen单列。R3尚未执行，下一批依次为：

1. **M0：全程采样控制。** fresh D2 T4，300k×4、width16、WD=.01、batch8，仅改每pass事件混合；保持原逐update batch大小、U=150117及LR轨迹。先做缓存输入等价、吞吐和完整validation峰值，随后与原D2比较all/sparse、逐类、τ及成本。实际有效再采用新配方并补必要T3。
2. **G0/G1：B1局部保留。** 先以守恒padding检验采样相位，再仅将首层stride4→2、累计stride32→16，其他配方固定。32-event真实拟合、分支梯度和峰值通过后，做同unique/曝光比较，采样与结构增量分别报告。
3. **G2/G3：原生关系与复现。** 使用本族匹配控制检验压缩前跨窗口/跨plane边，报告候选覆盖与边数尾部；仅最有价值改变补seeds17/2026，再做native/100μm/1mm配对。沿用2 pp共同BA或1 pp τ门槛及相应质量约束。
4. **R4–R7。** 对象质量闭环→条件τ辅助/specialist→胜出encoder定向训练→封存独立确认。τ更细标签先验证物理定义和可见支持，unknown子道只mask辅助loss。后续原42 h完整矩阵按实际小批次重排，保留R7资源。

完整研究理由、物理两条路径及各阶段退出条件见[研究决策](../roadmap/v2/09_decisions_and_handoff.md)、[执行路线](../roadmap/v2/04_execution_plan.md)和[资源安排](../roadmap/v2/07_resources_and_operations.md)。旧v1路线已归档于[archive/2026-11-14_v1](../roadmap/archive/2026-11-14_v1/README.md)，历史结果保持原形成时点。
