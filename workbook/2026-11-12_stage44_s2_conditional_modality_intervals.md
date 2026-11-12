# 阶段44：固定区域内的双向模态配对区间

在相同large-scintillator validation事件上，pixel+bar相对bar-only的τ AP增量为+0.01429830，pointwise 95%事件区间=[+0.00374257,+0.02424773]；中心区域为−0.01544100，区间=[−0.05592653,+0.02131973]。中心点估计下降尚不能确定为负收益，也不能排除+0.01收益。Bar在pixel之上的AP增量在两个区域均有正区间。这些是已有full-domain训练模型的条件评价，训练域控制与三种子确认继续运行。

## 阶段目标

补充阶段43中心/large模态点估计的区间，判断它们分别支持何种结论；将区域内的排序收益与低背景工作点分开，推进历史fiducial差异的解释。

## 工作内容

新增`analysis.flavour.s2.conditional_pairs`，读取已完成且输出读回pass的两模型validation概率，按既有`pixel_centered`或`large_scintillator`标记取相同事件。调用已有accepted-family/chunk配对bootstrap，保留各域四类计数、四分类指标、τ工作点与历史三类条件指标，并生成英文图。没有新训练或重新选择checkpoint；均沿用full-validation NLL选择的best。

这是结果后的条件诊断，不增加主41单seed配对+4三seed均值合同。各域使用实际类别比例；本诊断未进行E/vertex重加权。区域之间的AP绝对值含不同prior，不能由这些表推断区域交互显著性；中心与large也不是独立统计样本。

## 详细数据

原训练共享80k train/10k validation，各类20k/2,500，seed42、4 epochs、batch4、20,016 attempted steps/epoch，H100 NVL/LCG108/Torch2.7.1/BF16。Pixel-only、bar-only、pixel+bar三项的14项CPU输出读回均已pass。仅使用冻结validation，calibration/test未使用。

| Evaluation domain | Events | e / mu / tau / NC | Tau prior |
| --- | ---: | --- | ---: |
| Pixel-centered | 885 | 245 / 237 / 176 / 227 | 0.19887006 |
| Large scintillator | 4,494 | 1,131 / 1,183 / 1,053 / 1,127 | 0.23431242 |

| Domain | Model | Tau AP | Tau AUC | Four-class NLL |
| --- | --- | ---: | ---: | ---: |
| Centered | Pixel-only | 0.21925256 | 0.55167329 | 1.26085029 |
| Centered | Bar-only | 0.33126632 | 0.69231632 | 1.07862769 |
| Centered | Pixel+bar | 0.31582532 | 0.67987883 | 1.06776883 |
| Large | Pixel-only | 0.28051924 | 0.58820952 | 1.29558794 |
| Large | Bar-only | 0.37212781 | 0.67372749 | 1.14098095 |
| Large | Pixel+bar | 0.38642610 | 0.68070524 | 1.12794971 |

### 条件配对

400次bootstrap、seed2026，两域均有128个accepted-family/chunk blocks，每类32；两模型共享每次抽样。区间以现有拟合模型为条件，没有跨训练种子或同时覆盖含义。

| Increment | Domain | Tau ΔAP [95% CI] | Tau ΔAUC [95% CI] |
| --- | --- | --- | --- |
| Pixel+bar−bar | Centered | −0.01544100 [−0.05592653,+0.02131973] | −0.01243749 [−0.02744776,+0.00120147] |
| Pixel+bar−bar | Large | +0.01429830 [+0.00374257,+0.02424773] | +0.00697775 [−0.00013454,+0.01451110] |
| Pixel+bar−pixel | Centered | +0.09657276 [+0.05550730,+0.14421228] | +0.12820554 [+0.09431597,+0.16775625] |
| Pixel+bar−pixel | Large | +0.10590686 [+0.08568412,+0.12833682] | +0.09249572 [+0.07595863,+0.10835598] |

Large域的pixel增量AP区间为正，但下界小于+0.01，尚不能确认至少+0.01的收益。AUC区间跨0，故不能将AP结果概括为所有排序指标均提高。中心域pixel增量的AP/AUC区间均跨0；其误差较大，不能用中心负点估计解释历史研究分歧。

### 每个区域自身的1%背景接受率

以下阈值分别在当前条件域的背景ROC上选择，与阶段43的inclusive阈值不同。Centered有709个背景，最多7个通过；large有3,441个背景，最多34个通过。

| Domain | Model | Threshold | Tau selected / denominator | Background selected / denominator |
| --- | --- | ---: | --- | --- |
| Centered | Pixel-only | 0.30004716 | 4 / 176 | 7 / 709 |
| Centered | Bar-only | 0.39634508 | 9 / 176 | 7 / 709 |
| Centered | Pixel+bar | 0.39302287 | 1 / 176 | 7 / 709 |
| Large | Pixel-only | 0.30117267 | 13 / 1,053 | 34 / 3,441 |
| Large | Bar-only | 0.46394026 | 31 / 1,053 | 34 / 3,441 |
| Large | Pixel+bar | 0.46707004 | 50 / 1,053 | 34 / 3,441 |

| Increment | Domain | Tau efficiency difference at 1% FPR [95% CI] |
| --- | --- | --- |
| Pixel+bar−bar | Centered | −0.04545455 [−0.07772331,+0.02474252] |
| Pixel+bar−bar | Large | +0.01804368 [+0.00483296,+0.02751320] |
| Pixel+bar−pixel | Centered | −0.01704545 [−0.04294643,+0.03957088] |
| Pixel+bar−pixel | Large | +0.03513770 [+0.01576298,+0.05395728] |

Large域两种模态增量的1%效率差均有正事件区间；中心域均跨0，且7个背景少于预定25个尾部统计参考。0.1%点的两域背景最多0/3个，统计不足，不作稳定rejection结论。完整点估计和配对区间保存在JSON。

### 验证与产物

正式根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。输出为`conditional_pixel_increment.json/.png/.pdf`与`conditional_bar_increment.json/.png/.pdf`，日志同名`.log`。两项配对各自两域的全部τ AP点估计与主`evaluation_final/summary.json`逐项一致；两张英文图实际查看。新入口的真实两对×两域运行正常退出，未申请GPU；CPU用时未单独测量。新模块语法及`git diff --check`通过。

现有回归套件在实现新条件诊断之前执行，53项测试、65.935秒、exit0，日志`s2_regression_suite_055a8f0.log`；其范围是原有表示/训练恢复/配对/种子/资源代码，新模块另由上述实际数据运行验证。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_H100=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100
python -m analysis.flavour.s2.conditional_pairs --root "$S2_H100" \
  --a pn_native_bar_s42_e4_v2 --b pn_native_joint_s42_e4_v2 \
  --output "$S2_H100/conditional_pixel_increment.json"
python -m analysis.flavour.s2.conditional_pairs --root "$S2_H100" \
  --a pn_native_pixel_s42_e4_v2 --b pn_native_joint_s42_e4_v2 \
  --output "$S2_H100/conditional_bar_increment.json"
```

## 阶段性结论

事实：同域配对支持large域的pixel AP及1%效率收益，中心域未确定方向；bar增量的AP收益在两个区域均成立，但中心域尾部收益未确定。中心域的pixel+bar NLL更低而τ AP点估计更低，说明四分类NLL选择与单类排序有不同取舍。

待检验解释：fiducial、响应覆盖与训练分布可能影响模态互补性。条件域内仍含能谱/顶点差异；本结果没有控制这些变量，也没有单独识别pitch收益。历史三类、当前四类与不同训练域仍需分别解释。

## 下一步计划

继续现有+tracker、训练域及额外种子任务。训练域模型完成后，复用本入口在相同中心/large事件上交叉评价，保留full-validation NLL选点规则，分开说明重复小域、unique事件数与matched steps的影响。最后补齐跨层与三seed结果，执行严格完整评价和35项输出gate，再完成S2报告；S3–S6不开始。
