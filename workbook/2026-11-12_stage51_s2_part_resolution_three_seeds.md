# 阶段51：ParT native−1 mm 的三seed确认

在冻结的四轮、full-validation NLL选点及共同局部压缩下，ParT native−1 mm三seed平均τ AP差为+0.00079004，pointwise 95%事件区间为[−0.00282494,+0.00445758]。该区间条件于这三个已拟合模型，可排除此条件下预定+0.01的平均收益；训练差值sample SD=0.00553825，不能将该区间推广为训练seed总体不确定度或一般信息等效结论。

## 阶段目标

完成第一组预定三seed配对，分开报告物理事件统计、训练波动、综合四分类、工作点和实际成本。其余三组均值与两个plane-only控制继续执行，不启动S3–S6。

## 工作内容

新增`part_1mm_joint_s2026_e4_v2`对应`1573782.26`，真实JobStatus=4、ExitCode=0、NumJobStarts=1，节点`slot1_4@b9pgpun003.cern.ch`。四轮完成、全部14项CPU读回pass，按full-validation NLL选epoch4，80,064 applied updates、0 AMP skips。三seed六个模型全部通过同一14项检查。

共用80,000 train/10,000 validation、每类20,000/2,500、pixel+bar、ParT dim48/151,228参数、H100 NVL/LCG108/Torch2.7.1/Python3.12.11/BF16、batch4、每轮20,016 attempted steps。共同局部编码为同平面4 mm/每块最多32 nodes，处理全部全局块；当前结果属于这一压缩模型。沿用已接受的样本、split与detector-proxy。

## 详细数据

### 完整三seed差值

| Seed | 1 mm tau AP | Native tau AP | Native−1 mm [95% event CI] | Fixed-support difference [95% CI] |
| --- | ---: | ---: | --- | --- |
| 17 | 0.35142978 | 0.35583088 | +0.00440110 [−0.00207130,+0.01145529] | −0.00138996 [−0.00768450,+0.00504632] |
| 42 | 0.35502857 | 0.35858391 | +0.00355534 [−0.00186106,+0.00868839] | +0.00297513 [−0.00192093,+0.00846325] |
| 2026 | 0.35488937 | 0.34930306 | −0.00558631 [−0.01149809,+0.00010084] | −0.00761783 [−0.01260776,−0.00286495] |
| Mean of per-seed differences | — | — | +0.00079004 [−0.00282494,+0.00445758] | −0.00201089 [−0.00526091,+0.00111996] |

400次共享物理事件块重采样，128个accepted-family/chunk blocks、每类32；每次分别计算三seed的AP差值后取平均。三次拟合仍只有相同10,000个物理validation事件，不构成30,000个独立事件。结果不是ensemble AP，也不是先平均概率再计算AP。

固定粗E/vertex支持为8,559 events、Neff=7,959.41365，未控制target Z/同位素。Raw/common均值区间都跨0、上界都低于+0.01；只对既定拟合、预算、表示和权重作这一判断。逐seed差值方向不一致，raw差值sample SD=0.00553825、common差值sample SD=0.00532371；这两个SD描述三次训练差值的波动，不是事件区间。

| Representation | Raw AP mean | Raw sample SD | Fixed-support AP mean | Fixed-support sample SD |
| --- | ---: | ---: | ---: | ---: |
| 1 mm | 0.35378258 | 0.00203877 | 0.32107277 | 0.00291798 |
| Native | 0.35457262 | 0.00476665 | 0.31906188 | 0.00413959 |

三次训练不足以判断哪种表示的seed方差总体更小。没有挑选最佳seed，也没有通过新增超参搜索改变本比较。

### 四分类、工作点与分支

| Seed / representation | BA | Macro F1 | NLL | Tau argmax recall | Tau selected at 1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 / 1 mm | 0.4237 | 0.42196263 | 1.22338293 | 0.4700 | 74 / 2,500 |
| 17 / native | 0.4125 | 0.41212061 | 1.23418634 | 0.4448 | 82 / 2,500 |
| 42 / 1 mm | 0.4033 | 0.39880034 | 1.24293540 | 0.5136 | 88 / 2,500 |
| 42 / native | 0.4184 | 0.41844267 | 1.22107879 | 0.5488 | 93 / 2,500 |
| 2026 / 1 mm | 0.4227 | 0.41895101 | 1.21753388 | 0.5560 | 82 / 2,500 |
| 2026 / native | 0.4097 | 0.40044304 | 1.22823669 | 0.6672 | 87 / 2,500 |

1%工作点每个模型均通过75/7,500背景。Native各seed多通过8/5/5个τ，但对应单seed效率差值区间均跨0；三seed效率均值CI未测量。不能以此宣称稳定工作点收益。0.1%每项仅7背景，尾部统计不足。

新增1 mm seed2026四类AP=0.60899846/0.46540404/0.35488937/0.42679547，Brier=0.65993255，τ ROC AUC=0.63376840。对应native的τ AUC=0.62746589，差值−0.00630251、CI=[−0.01129605,−0.00154388]；其raw AP差值区间仍跨0。不同指标的区间结论分别表述。非τ和综合指标配对CI未测量。

所有六模型的τ→e/μ/had、中心/large、能量/vertex和active/empty条件表保存在主summary及本阶段快照中；分支AP有自身prior，不直接与inclusive AP比较。Calibration/test未使用，阈值仅为validation经验工作点。

### 实际成本和训练曝光

| Seed | Exposures, both representations | Native allocation [s] | 1 mm allocation [s] | Native peak [MiB] | 1 mm peak [MiB] |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | 80040 / 80036 / 80036 / 80040 = 320152 | 6075 | 8536 | 1583.52490 | 1541.58447 |
| 42 | 80040 / 80044 / 80040 / 80040 = 320164 | 6993 | 7278 | 1589.18994 | 1545.35400 |
| 2026 | 80044 / 80048 / 80040 / 80040 = 320172 | 6082 | 5902 | 1584.63574 | 1542.09863 |

新增1 mm seed2026 program wall=5,895.22494 s、allocation=5,902 s=1.63944444 GPUh、额外6.77506 s。三seednative总allocation=19,150 s=5.31944444 GPUh，1 mm=21,716 s=6.03222222 GPUh；节点、运行时段和seed42早期门槛经历不同，不将成本差全部归因于表示。全四轮成本均入账。

Seed2026共同allocation预算5,898.76109 s下，1 mm可选epoch4，native可用NLL最优epoch3，AP差native−1 mm=−0.01133363；这是整轮分辨率的点估计，没有CI。Seed17共同时间点估计方向相反，见阶段50；本阶段主三seed比较仍按冻结四轮步数合同。

11:34:36 UTC组织累计86.79951360 GPUh，H100视图50.56118026 GPUh，missing history records=0。组织成本包括失败、恢复和旧尝试，未按best checkpoint删去已执行训练。

### 产物、图示与验证

结果根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`part_resolution_three_seed_comparison.json`保存六模型metrics/manifest/history/readback、三个单seed配对、预定均值、条件表、终态和成本；`stage51_execution_snapshot.json`记录11:45 UTC实际5 running/1 idle及Kubeflow Pending/无node。`part_1mm_s2026_terminal.json`与`readback_part_1mm_s2026.log`保存新增终态与读回。

`part_resolution_three_seeds/`中的逐seed/均值配对图、`evaluation_final/training_seed_spread.png`及`part_1mm_s2026_diagnostic_review.png`四张PNG均已实际查看；英文PDF可独立导出。图中训练SD和事件CI分开标注，第6页新增分类诊断显示四轮NLL下降，没有更换主选点。

Accounting/evaluate/target/inventory顺序进程真实exit0，评价commit=`979ed40`。主表33 results=4 references+29科学配置、889 strata、99历史三类表、33单seed配对、1三seed均值、11 pending、2 factorial；target诊断231 model×Z组。Inventory 29 output_complete、5 running、1 not_started，strict gate=false。日志后缀统一`part_resolution_seeds.log`。复用既有计算入口，新增绘图读取已有区间，未改核心计算，未重跑已通过的完整回归套件。

```bash
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments part_1mm_joint_s2026_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.target_diagnostics --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/target_diagnostics"
python -m analysis.flavour.s2.output_inventory --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json"
```

Kubeflow H10094GB flavor备用配置已生成但未提交，最初候选task33随后已在Condor运行，故不用该配置启动副本；真实GPU probe v9继续排队。只有仍idle目标可在runtime通过后迁移。

## 阶段性结论

事实：首个预定三seed差值完成，当前ParT压缩模型native相对1 mm的平均τ AP收益很小；固定已训练seed的事件区间排除+0.01平均收益。训练波动和条件支持下的方向变化已量化，未确认一般原生分辨率优越性。

解释范围：不把四轮小模型的结果当作原始硬件信息的可用上限，不以未确认收益判定细节无用；后续模型与任务的效果仍需对应实验。其余三组均值与跨层控制未完成，S2仍未整体验收。

## 下一步计划

继续既有ParticleNet seeds和plane-only训练，补齐41单seed+4三seed合同。全部35项输出完整后执行不带allow-incomplete的严格评价、最终target刷新、require-complete门槛及中文科学总报告；S3–S6保持未执行。
