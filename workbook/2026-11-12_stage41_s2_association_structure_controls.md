# 阶段41：S2 平面内关联与 ParT pair bias 控制

ParT原生主模型相对同平面y坐标置换模型的τ AP提高0.00847852，pointwise 95%事件区间=[0.00281484,0.01296334]；固定E/vertex common support后提高0.00794723，区间=[0.00257296,0.01269657]。当前seed42/四轮结果支持ParT可利用平面内关联。ParticleNet的同项AP区间跨0，ParT显式pair bias的增益区间也跨0；物理拓扑和跨种子稳定性仍未确定。三项四轮输出均完成，14项CPU读回全部通过。

## 阶段目标

分开回答两个问题：保留平面内x–y关联是否改善分类；在已经输入坐标时，显式geometry pair bias是否提供额外收益。给出四类表现、τ工作点、同allocation预算和输出证据。

## 工作内容

三个Job均在`bigbird20.cern.ch`正常退出，JobStatus=4、ExitCode=0、NumJobStarts=1：ParT shuffle=`1573782.12`，PN shuffle=`1573782.15`，ParT no-pair=`1573782.16`。CPU读回配置、checkpoint/optimizer、NLL选择、10k最佳预测及40k逐轮预测，全部14项pass。沿用用户已接受的samples、冻结split和detector-proxy。

`shuffle_local=True`实际在**每个pixel plane内置换y坐标**，保留x/y边缘分布、每层节点数、pixel计数与bar响应；随后重新计算4mm局部块。它不是只在4mm窗口内shuffle，也不是可实现的二值pixel读出。它改变平面内关联及后续邻域，不能直接把AP差归因于某种τ衰变拓扑。所有模型均在各自表示上重训，结果也不是同一个模型的逐事件信息开关。

No-pair移除2个局部和4个全局attention块的显式pair网络，原坐标、局部残差、计数/信号和仍输入。参数由151,228降至148,708，减少2,520（1.666%），因此同时涉及结构和容量；不能将它解释为移除所有几何信息。两族仍共享同平面4mm、canonical chunks≤32节点的局部learned compression，并处理全部全局块。

## 详细数据

80k train/10k validation，各类20k/2,500；seed42、batch4、四轮、20,016 attempted steps/epoch；H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。三项四轮exposures均为80,040/80,044/80,040/80,040，总320,164；80,064 applied updates、0 AMP skips，均按NLL选epoch4。calibration/test未使用。

| Model | Parameters | NLL | Tau AP | Fixed common-support AP | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ParT main | 151,228 | 1.22107879 | 0.35858391 | 0.32383619 | 0.41840 | 0.41844267 |
| ParT shuffle | 151,228 | 1.23209203 | 0.35010539 | 0.31588896 | 0.40720 | 0.40135810 |
| ParT no-pair | 148,708 | 1.24520246 | 0.35615655 | 0.32055718 | 0.40070 | 0.38965690 |
| ParticleNet main | 161,156 | 1.23930563 | 0.35510474 | 0.32202830 | 0.39740 | 0.39204149 |
| ParticleNet shuffle | 161,156 | 1.24327650 | 0.35114293 | 0.32141892 | 0.40470 | 0.39939055 |

| Control | CC nu_e AP | CC nu_mu AP | CC nu_tau AP | NC AP | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| ParT shuffle | 0.60180959 | 0.43737154 | 0.35010539 | 0.40249657 | 0.66963890 |
| ParT no-pair | 0.59569078 | 0.39215450 | 0.35615655 | 0.37939936 | 0.67759824 |
| ParticleNet shuffle | 0.59595209 | 0.41463315 | 0.35114293 | 0.37830292 | 0.67575561 |

四类AP以各25%的benchmark prior为条件。ParT no-pair的NC argmax recall为418/2,500=16.72%，主模型为744/2,500=29.76%；即使τ AP接近，四类决策仍有差别。非τ配对CI未测量，不据此宣称四类总体优劣。Shuffle也不能仅用τ argmax recall衡量，其增加的τ预测会同时吸收更多背景。

### 配对事件区间

方向均为主模型−控制。400次class-stratified accepted-family/chunk bootstrap，128 blocks、每类32；common support为8,559 events、Neff=7,959.41365，固定权重条件区间。以下均为pointwise区间，不含训练种子波动或多重比较同时覆盖。

| Contrast | Raw tau ΔAP [95% CI] | Fixed common-support tau ΔAP [95% CI] |
| --- | --- | --- |
| ParT main−shuffle | +0.00847852 [+0.00281484,+0.01296334] | +0.00794723 [+0.00257296,+0.01269657] |
| PN main−shuffle | +0.00396180 [−0.00300075,+0.01075346] | +0.00060939 [−0.00618280,+0.00742795] |
| ParT main−no-pair | +0.00242736 [−0.00350744,+0.00865689] | +0.00327901 [−0.00220454,+0.00858433] |

ParT关联增益点估计低于预定+0.01，但区间仍允许超过+0.01；PN raw区间也未排除该收益。No-pair的raw/common上界均低于+0.01，约束当前两次训练的τ AP差异，不能扩展为pair bias普遍无用。PN的τ ROC AUC差为+0.00657952、区间=[+0.00063924,+0.01249562]，与AP/尾部的结论不同，须按预定指标分别解释。ParT shuffle的AUC差区间跨0。

| Contrast | Tau efficiency difference at 1% FPR [95% CI] | Main/control tau selected | Background selected each |
| --- | --- | ---: | ---: |
| ParT main−shuffle | +0.00800000 [−0.00121421,+0.01413594] | 93 / 73 | 75 |
| PN main−shuffle | −0.00160000 [−0.01063354,+0.00667392] | 83 / 87 | 75 |
| ParT main−no-pair | +0.00600000 [−0.00405408,+0.01308916] | 93 / 78 | 75 |

1%工作点差的区间全部跨0，ParT的AP正区间没有同时确立尾部效率收益。控制模型的validation阈值依次为0.44733781/0.42646077/0.44702312；τ分母各2,500，背景7,500。0.1%下三个控制均仅7个背景，通过τ分别13/13/10，尾部不足；最终工作点留至S6。

### 学习曲线与成本

| Epoch | ParT shuffle NLL / tau AP | PN shuffle NLL / tau AP | ParT no-pair NLL / tau AP |
| --- | --- | --- | --- |
| 1 | 1.34635145 / 0.28960646 | 1.35229649 / 0.28941583 | 1.28270732 / 0.34445060 |
| 2 | 1.28898010 / 0.32211440 | 1.28835445 / 0.31077934 | 1.26777383 / 0.34853997 |
| 3 | 1.25891381 / 0.34968785 | 1.26766421 / 0.34246213 | 1.26185233 / 0.35101321 |
| 4 | 1.23209203 / 0.35010539 | 1.24327650 / 0.35114293 | 1.24520246 / 0.35615655 |

| Control | Program wall [s] | All-attempt allocation [s / GPUh] | Extra allocation [s] | Peak allocated memory [MiB] |
| --- | ---: | ---: | ---: | ---: |
| ParT shuffle | 7,177.40684 | 7,190 / 1.99722222 | 12.59316 | 1,474.46045 |
| PN shuffle | 5,719.61928 | 5,745 / 1.59583333 | 25.38072 | 1,479.78857 |
| ParT no-pair | 4,595.50185 | 4,603 / 1.27861111 | 7.49815 | 509.70068 |

| Main−control at matched allocation | Common budget [s] | Control / main epoch | Tau ΔAP |
| --- | ---: | ---: | ---: |
| ParT main−shuffle | 6,992.77439 | 3 / 4 | +0.00889606 |
| PN main−shuffle | 5,744.21304 | 4 / 3 | −0.00395743 |
| ParT main−no-pair | 4,602.25093 | 4 / 2 | −0.01860726 |

这些是计入setup/全部尝试成本后的整轮分辨率点估计，CI未测量。No-pair的显存与wall明显降低，在较小allocation预算下AP点估计优于只能完成两轮的主模型；这个工程取舍与固定四轮的比较不同，后续研究应保留。它不意味着在相同参数量或所有硬件上均成立。

### 产物与复现

正式根目录为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。独立快照`association_structure_comparison.json`包含三项训练/读回、两项主模型、81行新增分域指标、完整配对和成本。配对图第3页`association_structure_paired_review.png`及三个新增`*_diagnostic_review.png`已实际查看。分支/域指标保存在快照，分支及域内差值CI未测量。

合并评价正常退出，commit=`013038e`：23 results（4 references+19科学模型）、619行分域、69个历史三类诊断、25 paired/20 pending、2 factorial、0 seed means；`requested_comparisons_complete=false`。08:17:51 UTC组织累计70.96387692 GPUh，H100视图34.72554358 GPUh，missing history records=0；运行中任务成本是该时刻的实际allocation估计。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" \
  --experiments part_native_shuffle_s42_e4_v2 pn_native_shuffle_s42_e4_v2 part_native_nopair_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.output_inventory --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json"
```

新增`output_inventory`只汇总现有配置、14项读回、评价和带时刻的backend台账，不重新认证样本或proxy。实际未完成状态下`--require-complete`退出1，正确保留pending，语法与`git diff --check`通过。最终先执行无`--allow-incomplete`评价，再以`--require-complete`检查35项完整输出；科学结论另由中文阶段报告验收。

## 阶段性结论

事实：ParT的平面内关联在raw与固定粗E/vertex支持下均有τ AP正事件区间；两族当前控制没有确定的1% FPR效率收益。显式pair bias的τ AP收益小于+0.01的事件区间上界，同时其移除带来明显资源下降和不同四类混淆。

待检验解释：关联可能编码shower/track的平面形态或局部聚集；当前干预会同时改变局部分块，不能区分这些来源。Common support只控制既定粗分箱，并未消除全部运动学先验。PN与ParT的差异可能来自优化和关系编码，跨种子/模型族差异的差值CI未测量。

## 下一步计划

完成fixed-graph、pixel/bar/+tracker、训练域和额外种子、plane-only控制，逐项读回并填满41单seed配对+4三seed均值。新完成的PN fixed-graph进入下一批评价；其结果不混入本次冻结快照。完整输出、资源、物理解释及无缺项评价齐备后验收S2；calibration/test继续冻结至S6，不开始S3–S6。
