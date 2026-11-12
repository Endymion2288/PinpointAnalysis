# 阶段43：双向模态增量与靶核条件诊断

ParticleNet的pixel/bar/pixel+bar三项完整对照已齐备。Pixel在bar之上的raw τ AP增量为+0.01073932，pointwise 95%事件区间=[+0.00463872,+0.01687213]；固定粗E/vertex common support后增量为+0.00180202，区间=[−0.00491768,+0.00812752]。其nominal收益与固定支持下的结果不同，1% FPR效率差也尚未获得正区间支持。Bar在pixel之上的增量则在raw/common两种评价中均有正区间，见阶段42。正式S2现有22/35科学模型完整、28项配对完成。

## 阶段目标

补齐pixel在bar之上的互补性测量，分开评价排序、尾部效率和计算成本；补充现有接受样本的truth target组成及条件指标，明确E/vertex匹配的实际范围。

## 工作内容

Bar-only `1573782.19`在`bigbird20.cern.ch`正常结束，JobStatus=4、ExitCode=0、NumJobStarts=1，原节点`b9pgpun501.cern.ch`。14项CPU输出读回全部pass。仅输入bar的区间、edep/PE detector-proxy、panel和相应加性汇总，pixel与tracker响应均不进入模型；名义参数仍为161,156。

新增只读入口`analysis.flavour.s2.target_diagnostics`：读取现有90k-event provenance，保存同位素/PDG计数，并按靶核Z评价已有validation概率。Nuclear PDG按`10LZZZAAAI`解码Z，proton归Z=1，其他非核靶保留Z=0。Target只作truth诊断，不进入模型、重分split或改变事件选择；主41单seed配对+4三seed均值合同不变。

## 详细数据

共同80k train/10k validation，各类20k/2,500，seed42、batch4、4 epochs、20,016 attempted steps/epoch，H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。Bar-only按NLL选epoch4；80,064 applied、0 AMP skips，四轮exposures=80,040/80,044/80,040/80,040，总320,164。calibration/test未使用。

| Model | Raw tau AP | Fixed common-support AP | Tau AUC | NLL | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pixel-only | 0.28619671 | 0.27351752 | 0.56983725 | 1.34070449 | 0.32930 | 0.31120173 |
| Bar-only | 0.34436541 | 0.32022628 | 0.62393899 | 1.24988971 | 0.39200 | 0.39182795 |
| Pixel+bar | 0.35510474 | 0.32202830 | 0.63150771 | 1.23930563 | 0.39740 | 0.39204149 |

| Bar-only metric | CC nu_e | CC nu_mu | CC nu_tau | NC |
| --- | ---: | ---: | ---: | ---: |
| AP | 0.59018423 | 0.40911816 | 0.34436541 | 0.35107004 |
| Argmax recall | 0.4640 | 0.2420 | 0.4968 | 0.3652 |

Bar-only Brier=0.67939687。Pixel+bar的四类AP点估计均高于bar-only，但NC argmax recall为0.2844，低于bar-only的0.3652；分类排序与未校准argmax决策分别报告。非τ差值CI未测量。

### 配对与工作点

400次、128个accepted-family/chunk blocks（每类32）的pointwise事件bootstrap；固定common support为8,559 events、Neff=7,959.41365。区间以已有训练和固定权重为条件，不包含跨seed波动或同时覆盖保证。

| Contrast | Raw tau ΔAP [95% CI] | Fixed common-support tau ΔAP [95% CI] |
| --- | --- | --- |
| Pixel+bar−bar | +0.01073932 [+0.00463872,+0.01687213] | +0.00180202 [−0.00491768,+0.00812752] |
| Pixel+bar−pixel | +0.06890803 [+0.05663332,+0.08443211] | +0.04851078 [+0.03602058,+0.06154046] |

Pixel增量的raw区间下界小于+0.01，尚未确认至少+0.01的收益；固定支持上界已低于+0.01且跨0。Raw与weighted AP使用不同有效分母/权重，不能按差值缩小的比例分解“多少来自先验”。此消融同时增加pixel占用、计数和几何，原生pitch相对1mm的贡献仍由独立空间配对回答。

Pixel+bar−bar τ AUC差=+0.00756872，区间=[+0.00215637,+0.01322917]。1% FPR下bar阈值=0.44480830，通过74/2,500个τ和75/7,500个背景；主模型通过83/75。效率差=+0.0036，区间=[−0.00461223,+0.01168827]，不能由AP正区间推定尾部效率也提高。0.1%下bar通过7个τ/7背景，主模型3/7；尾部统计不足。

Bar-only的electron/muon/hadronic τ在inclusive 1%阈值下通过8/4/62个，共同分母445/395/1,660；主模型7/4/72。分支AP分别为0.07548534/0.05805350/0.28311634。分支差值CI未测量。

### 接受度与历史三类诊断

Bar-only有响应9,517 events，输入空483（159 τ）；空输入AUC=0.5，AP=159/483=0.32919255。Pixel+bar输入空440，故新增pixel还改变了43个事件的响应状态；不能把全部AP收益都解释为精细空间信息。

| Domain | Events / tau | Bar-only tau AP | Pixel+bar tau AP |
| --- | ---: | ---: | ---: |
| Full | 10,000 / 2,500 | 0.34436541 | 0.35510474 |
| Pixel-centered | 885 / 176 | 0.33126632 | 0.31582532 |
| Large scintillator | 4,494 / 1,053 | 0.37212781 | 0.38642610 |

中心域τ AP点估计方向与full/large不同，域内差值CI未测量；完整训练域控制仍在运行。为对应历史三类报告，另取中心域709个真e/μ/NC，使用同一四类模型的三类条件归一化分数：

| Conditional three-class metric | Bar-only | Pixel+bar |
| --- | ---: | ---: |
| CC nu_e AUC | 0.91995954 | 0.92395320 |
| CC nu_mu AUC | 0.76775370 | 0.76939856 |
| NC AUC | 0.76624564 | 0.78712048 |
| Balanced accuracy | 0.62142320 | 0.61904398 |

三类AUC点估计可提高而四类τ AP下降；因此历史三类与当前四类的结论不能直接互换。此表没有重新训练旧三类模型，三类差值CI未测量。独立快照为`historical_center_modality_comparison.json`。

### 靶核组成与条件结果

90k冻结子集包含14种target PDG、8种Z；validation实际有7种Z。计数表完整保留同位素。当前26个完成结果（22科学模型+4参照）共输出182个model×Z条件组。

| Validation target | Events | e / mu / tau / NC | Tau prior |
| --- | ---: | --- | ---: |
| W, Z=74 | 9,276 | 2,326 / 2,331 / 2,310 / 2,309 | 0.24902975 |
| C, Z=6 | 363 | 89 / 87 / 92 / 95 | 0.25344353 |
| Al, Z=13 | 292 | 65 / 68 / 80 / 79 | 0.27397260 |
| Proton, Z=1 | 36 | 11 / 5 / 8 / 12 | 0.22222222 |
| N, Z=7 | 27 | 9 / 6 / 8 / 4 | 0.29629630 |
| O, Z=8 | 5 | 0 / 2 / 2 / 1 | 0.40000000 |
| Ar, Z=18 | 1 | 0 / 1 / 0 / 0 | 0.00000000 |

| Target | Pixel-only tau AP | Bar-only tau AP | Pixel+bar tau AP |
| --- | ---: | ---: | ---: |
| W | 0.28602460 | 0.34466121 | 0.35606566 |
| C | 0.29540100 | 0.34438761 | 0.34222396 |
| Al | 0.28628886 | 0.37959213 | 0.38398118 |

W占validation的92.76%，W内仍有两种模态增量的正AP点估计；nominal差异并未只出现在非W事件。C中pixel+bar相对bar的点估计略降。各Z仍含能谱/顶点差异，Z内差值CI未测量，其余69个事件保留计数和点估计。主common-support只匹配既定粗E/vertex分箱，未额外匹配Z或同位素；这些条件表补充说明其范围。

### 学习曲线与实际成本

| Epoch | Bar-only NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.27881198 | 0.33339995 | 806.41201 |
| 2 | 1.28230270 | 0.31639691 | 1,586.62691 |
| 3 | 1.26746564 | 0.33697237 | 2,378.15790 |
| 4 | 1.24988971 | 0.34436541 | 3,193.20154 |

Bar程序wall=3,193.89696 s，allocation=3,210 s=0.89166667 GPUh，额外16.10304 s，peak allocated memory=594.48633 MiB。主模型allocation=6,310 s，bar-only为其50.87%。等allocation预算3,209.30457 s时bar选epoch4、主模型选epoch2，pixel+bar−bar ΔAP=−0.01664876；这是整轮分辨率点估计，CI未测量。固定四轮的正增量与较低计算预算下的排序不同，后续工程选择保留bar-only参照。

08:49:23 UTC组织累计74.11387479 GPUh，H100视图37.87554145 GPUh，missing history records=0。Target条件表只在CPU读取小型既有目录/预测，未申请GPU，CPU用时未单独测量。

### 产物、命令与验证

正式根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`modality_comparison.json`保存三种模态、两项增量、bar训练/读回和实际成本；`modality_paired_review.png`与bar四分类诊断页已实际查看。`target_diagnostics/summary.json`和`target_counts.csv`保存靶核条件结果。新只读模块实际运行完成，计数总和为80k train/10k validation，Z计数与模型条件分母对应；语法与`git diff --check`通过。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments pn_native_bar_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.target_diagnostics --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/target_diagnostics"
python -m analysis.flavour.s2.output_inventory --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json"
```

本批评价commit=`1d6fb98`、正常退出：26 results、700行分域、78个历史三类诊断、28 paired/17 pending、2 factorial、0 seed means。日志为`readback_pn_native_bar.log`、`accounting_bar_control.log`、`evaluation_bar_control.log`、`target_diagnostics.log`与`output_inventory.log`。新增target模块及本报告随本阶段提交，不改变既有训练代码。

## 阶段性结论

事实：两个方向的raw AP增量均为正区间；bar增量在固定E/vertex支持后仍为正，pixel增量在该条件下跨0且上界小于+0.01。Pixel的1% FPR效率收益尚未确立，bar-only在较低allocation预算下仍是重要参照。目标域/历史三类指标会改变比较的点估计方向。

待检验解释：pixel可能提供部分能量/顶点相关接受度及空间结构；当前不能将nominal增量直接归因于τ细拓扑。粗匹配后的变化也可能涉及分母、权重和训练分布，不能以一次重加权证明唯一机制。后续跨层、域训练和种子结果用于收紧这些解释。

## 下一步计划

完成余下13个科学配置：+tracker、两项训练域、8项额外种子与两项plane-only。继续使用现有6个running/7个idle Condor任务；Kubeflow v7仍Pending、未分配node。严格评价及35项输出gate仍待全部结果；calibration/test冻结至S6，不开始S3–S6。
