# 阶段45：Tracker proxy增量与四类/τ指标的不同取舍

ParticleNet加入现有tracker proxy后，四分类balanced accuracy从0.3974提高至0.4367、NLL从1.23931降至1.20936；这些是单seed点估计，差值区间未测量。τ AP差值为−0.00424866，pointwise 95%事件区间=[−0.01307753,+0.00369210]，固定E/vertex支持区间也跨0且上界低于+0.01。当前四轮预算下没有τ AP正收益支持，不能由四分类指标改善推定τ识别改善。正式S2现有23/35科学模型完整、29项配对完成。

## 阶段目标

完成pixel/bar/pixel+bar/+tracker模态对照，量化接受的tracker proxy对四类分类、τ排序/尾部、响应覆盖与实际成本的增量，并区分full、中心和large区域。

## 工作内容

`pn_native_tracker_s42_e4_v2`对应`bigbird20.cern.ch`的`1573782.20`，真实JobStatus=4、ExitCode=0、NumJobStarts=1，原节点`b9pgpun010.cern.ch`。14项CPU输出读回全部pass。资源核算、部分评价、输出inventory均正常退出；另使用阶段44的条件配对入口评价相同中心/large事件。

输入沿用用户接受的tracker hypothesis：测得的`xy_mm`、`sigma_xy_mm`、station标识与相应通道计数；没有truth momentum。当前表示将station索引编码为纵向特征（`station*1000`后统一缩放），不把它解释为新增实测crossing-z。现有proxy与参数直接使用，不进行硬件标定或样本来源重认证。主模型与控制分别重训，故增量包括表示、覆盖与有限模型重新优化的共同影响。

## 详细数据

共同80k train/10k validation，各类20k/2,500，seed42、batch4、4 epochs、20,016 attempted steps/epoch，H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。两项均161,156 parameters，full-validation NLL选epoch4。+Tracker的80,064次更新全部应用、0 AMP skips；各轮exposures=80,040/80,044/80,040/80,040，总320,164。calibration/test未使用。

| Metric | Pixel+bar | Pixel+bar+tracker |
| --- | ---: | ---: |
| Tau AP, raw | 0.35510474 | 0.35085607 |
| Tau AP, fixed E/vertex support | 0.32202830 | 0.31920384 |
| Tau ROC AUC | 0.63150771 | 0.63094824 |
| Four-class NLL | 1.23930563 | 1.20936304 |
| Balanced accuracy | 0.39740000 | 0.43670000 |
| Macro F1 | 0.39204149 | 0.43802815 |
| Brier | 0.67434646 | 0.65590187 |

| Class | Pixel+bar AP | +Tracker AP | Pixel+bar argmax recall | +Tracker argmax recall |
| --- | ---: | ---: | ---: | ---: |
| CC nu_e | 0.59882218 | 0.61614705 | 0.4920 | 0.5388 |
| CC nu_mu | 0.41459288 | 0.47679774 | 0.2468 | 0.3764 |
| CC nu_tau | 0.35510474 | 0.35085607 | 0.5664 | 0.4452 |
| NC | 0.36957463 | 0.43971042 | 0.2844 | 0.3864 |

e/μ/NC的AP与argmax recall点估计均提高，τ argmax recall下降0.1212；argmax与固定背景接受率是不同决策。非τ AP、四分类指标和recall差值CI未测量，不将点估计表述为跨seed稳定收益。

### 预定配对与尾部

400次、128个accepted-family/chunk blocks（每类32）的配对bootstrap。区间为pointwise事件区间，以现有拟合模型为条件。固定common support为8,559 events、Neff=7,959.41365，控制既定粗E/vertex分箱，未匹配靶核Z或同位素。

| +Tracker minus pixel+bar | Difference [95% CI] |
| --- | --- |
| Raw tau AP | −0.00424866 [−0.01307753,+0.00369210] |
| Fixed-support tau AP | −0.00282447 [−0.01101520,+0.00480946] |
| Tau ROC AUC | −0.00055947 [−0.00716032,+0.00637854] |
| Tau efficiency at 1% FPR | −0.00560000 [−0.01114052,+0.00538917] |
| Tau efficiency at 0.1% FPR | +0.00280000 [−0.00358118,+0.00539464] |

Raw/common AP的上界均低于预定+0.01工程收益，但下界允许超过0.01的损失，不能据此判为近等效。1% FPR下+tracker阈值0.44111499，通过69/2,500个τ与75/7,500个背景；pixel+bar为83/75。0.1%下+tracker通过10个τ/7背景，pixel+bar为3/7；背景计数不足25，效率差区间也跨0，不宣称稳定尾部改善。

在inclusive 1%阈值下，electron/muon/hadronic τ通过8/4/57个，分母445/395/1,660；pixel+bar通过7/4/72。少14个τ的点估计主要对应hadronic分支通过数减少15，分支差值CI未测量，不能把它视为已确定的物理机制。

### 响应覆盖与条件域

既有validation目录的tracker计数保存为`tracker_response_counts.json`，只描述接受的proxy响应。

| True class | Tracker active / 2,500 | Tracker-only response | Mean tracker channels |
| --- | ---: | ---: | ---: |
| CC nu_e | 1,359 | 35 | 9.2152 |
| CC nu_mu | 1,646 | 28 | 5.2364 |
| CC nu_tau | 1,497 | 58 | 5.7160 |
| NC | 1,128 | 37 | 4.8336 |

共有5,630个tracker-active事件。加入tracker后输入空事件由440（148 τ）降至282（90 τ），新增158个tracker-only响应事件，其中58 τ；+tracker模型在该158-event域的AP=0.41096208、AUC=0.52586207，相比pixel+bar先验型AP=58/158=0.36708861、AUC=0.5。域内差值CI未测量，1% FPR仅允许1个背景通过，未选中τ。剩余282个全空输入仍为AUC=0.5、AP=90/282=0.31914894。

| Fixed conditional domain | Events / tau | Pixel+bar tau AP | +Tracker tau AP |
| --- | ---: | ---: | ---: |
| Tracker active | 5,630 / 1,497 | 0.36618102 | 0.37135908 |
| Tracker empty | 4,370 / 1,003 | 0.34018923 | 0.32129296 |
| Pixel centered | 885 / 176 | 0.31582532 | 0.35497267 |
| Large scintillator | 4,494 / 1,053 | 0.38642610 | 0.37618916 |

两模型分别训练，即使在tracker-empty事件上，权重也会不同。不能把该域AP下降直接归因于缺失tracker的事件级信息损失。各域类别比例不同，不直接比较跨域AP绝对值。

中心/large使用相同full-validation NLL选点，条件配对同为400次/128 blocks，未重加权E/vertex：

| Domain | Tau ΔAP [95% CI] | Tau efficiency difference at domain-specific 1% FPR [95% CI] |
| --- | --- | --- |
| Centered | +0.03914735 [−0.01543292,+0.09356065] | +0.03977273 [−0.02666615,+0.09682320] |
| Large | −0.01023694 [−0.02268400,+0.00225976] | −0.00854701 [−0.01926866,+0.00608063] |

两域AP和效率差区间均跨0；large的AP上界小于+0.01，中心域尚不能排除该收益。条件1%工作点，中心+tracker选8τ/7背景、主模型1/7，large分别41/34和50/34；中心尾部统计不足。中心四分类balanced accuracy从0.51078713升至0.59396594、NLL从1.06776883降至0.96164001，large分别0.48970113→0.54274507、1.12794971→1.07213827，均为点估计。

中心709个真e/μ/NC事件的历史三类条件诊断中，e/μ/NC AUC分别从0.92395320/0.76939856/0.78712048升至0.93274103/0.88896338/0.85646261，balanced accuracy从0.61904398升至0.73440988。这是同一四类模型去τ并对三类分数归一化的点估计，没有重训历史模型或测差值CI；三类改善仍不能推定四类τ改善。

### 学习曲线与实际成本

| Epoch | +Tracker NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.26823976 | 0.29705650 | 1,357.98140 |
| 2 | 1.23302940 | 0.33440809 | 2,686.91937 |
| 3 | 1.21783511 | 0.34955030 | 4,015.18306 |
| 4 | 1.20936304 | 0.35085607 | 5,343.06718 |

+Tracker完整程序wall=5,343.75757 s，allocation=5,353 s=1.48694444 GPUh，额外9.24243 s，peak allocated memory=1,623.12842 MiB。Pixel+bar为6,310 s、1,593.10254 MiB。此次实际allocation为主模型84.83%；不同执行时段/节点与有限运行有影响，不据单次wall比较宣称添加模态本身提速。

等allocation预算5,352.30960 s下，+tracker可用epoch4，主模型可用epoch3，τ AP差为+0.00367057；主模型剩余582.82945 s不足容纳下一完整轮。该整轮分辨率点估计没有CI，排序与固定四轮不同。09:34:40 UTC组织累计78.59287089 GPUh，H100视图42.35453756 GPUh，missing history records=0。

### 产物与验证

正式根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`tracker_comparison.json`保存模型、配对/条件表、manifest/history/readback、真实Condor终态和成本快照；`tracker_terminal.json`与`tracker_response_counts.json`保留独立记录。`conditional_tracker_increment.json/.png/.pdf`、`tracker_diagnostics/paired_comparisons.pdf`及其首页PNG、`evaluation_final/pn_native_tracker_diagnostic_review.png`均已生成并实际查看。

评价commit=`c43df0d`，正常退出：27 results（4 references+23科学模型）、727行分域、81张历史三类条件表、29 paired/16 pending、2 factorial、0 seed means。23项科学输出均已读回、核算并进入评价；严格output gate仍为false。更新target诊断后覆盖27个结果的189个Z条件组，主匹配合同不变。日志为`readback_pn_native_tracker.log`、`accounting_tracker_control.log`、`evaluation_tracker_control.log`、`conditional_tracker_increment.log`和`output_inventory.log`。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments pn_native_tracker_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.conditional_pairs --root "$S2_RESULTS/h100" \
  --a pn_native_joint_s42_e4_v2 --b pn_native_tracker_s42_e4_v2 \
  --output "$S2_RESULTS/h100/conditional_tracker_increment.json"
python -m analysis.flavour.s2.target_diagnostics --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/target_diagnostics"
python -m analysis.flavour.s2.output_inventory --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json"
```

本阶段未改训练、配置或评价代码，复用阶段44入口；真实输出读回、条件分析和主评价正常退出，`git diff --check`通过。53项既有回归测试的最近完整通过记录见阶段44，本阶段未重复执行。

09:45:49 UTC实际状态为23 complete、5 running、7 idle；`stage45_execution_snapshot.json`保留各配置进度。09:46–09:47 UTC查询Kubeflow v7时，CERN OIDC登录需要交互式续期，两次非交互查询均失败，当前Job/Pod状态不可观测；这是客户端登录状态，未据此结束或替换既有Job。已提醒用户按所提供CERN文档重新登录，Condor继续；精简证据为`kubeflow/client_auth_expired.json`。

## 阶段性结论

事实：接受的tracker proxy增加158个事件的输入响应，并改善e/μ/NC AP与四分类综合指标点估计；τ AP、固定背景效率及两个条件域均没有正事件区间支持。当前模型下τ AP至少+0.01的收益可排除，等效或跨seed稳定性未确定。

待检验解释：新增tracker可能主要帮助μ/NC等类别区分，有限共享模型和NLL目标也可能重新分配分类能力。Tracker-active与empty的τ点估计方向不同，不能由此识别唯一原因。后续融合设计应分别保留四类目标与τ目标的评价，而不是只按单一AP删去模态。

## 下一步计划

继续两项训练域、8项额外种子与两项plane-only，共12个既有任务。训练域完成后在full/center/large交叉评价，三种子完成后计算既定四项均值与训练spread；全部结果齐备后执行无`--allow-incomplete`的严格评价、35项output gate和最终S2科学验收。S3–S6不开始。
