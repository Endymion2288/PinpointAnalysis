# 阶段 29：S2 首个正式 H100 模型与原生错误事件

统一H100矩阵的首个科学模型ParticleNet native joint已完成四轮，τ AP=0.35510。相对B0的原样本AP差值区间跨0，固定能量/顶点common-support上的差值为+0.01358，95%区间[+0.00324,+0.02458]。20页原生事件图已生成，首页与4个固定混淆页已实际查看。其余正式配置仍在训练或排队，S2整体验收尚未完成。

## 阶段目标

完成统一H100合同中的首个完整模型，核对四轮checkpoint、概率、NLL选模与实际allocation，运行冻结的validation比较并查看原生错误响应。沿用用户接受的samples、source-family split和detector-proxy；calibration/test保留至S6。

## 工作内容

`1573760.1`使用`roadmap/configs/s2_h100_v2.json`的ParticleNet native joint配置：seed42、batch4、4 epochs、每轮20,016 attempted steps，161,156 parameters。训练代码commit=`6fad82e`；软件为LCG108/Python3.12.11/Torch2.7.1/PyArrow20、H100 NVL/BF16。正式环境依据阶段25的双家族完整首轮执行门槛选定。

训练池80k、validation10k，各类20k/2.5k。输入为原生pixel+bar，使用既定的同平面局部learned compression（每块至多32 nodes）与全局块处理，全部事件响应按既有表示进入模型。tracker作为单独冻结消融；事件图的tracker面板提供探测器响应背景。

真实Condor终态为JobStatus=4、ExitCode=0、NumJobStarts=1。CPU读回`last.pt`、`best.pt`和概率文件的12项检查通过，覆盖完整四轮、冻结配置、optimizer计数、NLL选模、参数与optimizer有限性、恢复所需状态、唯一validation keys、概率范围/归一化及其与checkpoint的一致性、四轮概率覆盖和exposures。证据为`experiments/pn_native_joint_s42_e4_v2/final_checkpoint_checks.json`。

随后以`--allow-incomplete`运行正式H100目录的冻结比较，正常退出，评价commit=`85b259c`。当前有5个结果（4 references和1个正式模型）、133行分域指标、15个历史三类诊断；1项配对完成、44项pending，`requested_comparisons_complete=false`。truth E/vertex reference仍单列为先验诊断。

## 详细数据

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] | Train exposures |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.29708937 | 0.30611956 | 1,604.22022 | 80,040 |
| 2 | 1.28099697 | 0.32771665 | 3,154.69846 | 80,044 |
| 3 | 1.25283637 | 0.34718550 | 4,752.73711 | 80,040 |
| 4 | 1.23930563 | 0.35510474 | 6,292.93493 | 80,040 |

按NLL选择epoch4。80,064次optimizer更新全部应用，AMP skips=0；exposures总计320,164，独立训练事件仍为80k。最终程序wall=6,293.25696 s，Condor allocation=6,310 s=1.75278 GPU-hours，额外allocation=16.74304 s。peak allocated GPU memory=1,593.10254 MiB，parent peak RSS=2,265.09375 MiB。

四类balanced accuracy=0.39740、macro F1=0.39204、NLL=1.23930563、Brier=0.67434646。下表为25%每类的validation mixture，recall使用四类argmax。

| Class | AP | ROC AUC | Recall |
| --- | ---: | ---: | ---: |
| CC nu_e | 0.59882218 | 0.78161211 | 0.4920 |
| CC nu_mu | 0.41459288 | 0.66153117 | 0.2468 |
| CC nu_tau | 0.35510474 | 0.63150771 | 0.5664 |
| NC | 0.36957463 | 0.64676405 | 0.2844 |

| Prespecified ParticleNet − B0 | AP difference | Pointwise 95% CI |
| --- | ---: | --- |
| Original validation mixture | +0.01039406 | [−0.00250340,+0.02294209] |
| Fixed E/vertex common support | +0.01357814 | [+0.00323872,+0.02457823] |

400次accepted-family/chunk配对bootstrap使用128个物理blocks，每类32个。common support保留8,559 events、Neff=7,959.41365，权重固定。原样本增量仍不确定；固定支持结果支持正增量，是否达到预定+0.01仍不确定。该对照同时改变表示与估计器，B0为CPU reference，GPU时间匹配未测量。

validation empirical 1% FPR阈值为0.45402154，保留83/2,500个τ和75/7,500个背景，τ效率3.32%。相对B0的效率差为+0.0072，CI=[−0.00278142,+0.01495826]。0.1% FPR只保留3个τ和7个背景，尾部统计不足。工作点属于S2 validation比较，最终阈值留至后续阶段。

τ分支在同一7,500个non-τ背景上评价，下表计数使用inclusive 1% FPR阈值。分支效率差值区间未测量。

| Tau branch | Signal events | Selected signal | Efficiency | Branch AP |
| --- | ---: | ---: | ---: | ---: |
| Electron | 445 | 7 | 0.01573034 | 0.07676873 |
| Muon | 395 | 4 | 0.01012658 | 0.05989884 |
| Hadronic | 1,660 | 72 | 0.04337349 | 0.29638621 |

分支AP分别以各分支自身的signal prior为条件。以下分域也各自使用对应样本组成。

| Validation domain | Events / tau events | Tau AP | Tau ROC AUC |
| --- | --- | ---: | ---: |
| Full | 10,000 / 2,500 | 0.35510474 | 0.63150771 |
| Pixel centered | 885 / 176 | 0.31582532 | 0.67987883 |
| Large scintillator | 4,494 / 1,053 | 0.38642610 | 0.68070524 |
| Pixel+bar input empty | 440 / 148 | 0.33636364 | 0.50000000 |
| Tracker-only response | 158 / 58 | 0.36708861 | 0.50000000 |

全空输入域的AP等于该域τ比例，表现为先验型输出；158个tracker-only事件在当前pixel+bar模型中同样没有区分信息。已冻结的+tracker控制将测量新增信息。pixel-centered域仅占validation的8.85%，其单独1% FPR工作点只有7个背景通过，统计不足；large域有34个背景、50个τ通过。

### 原生事件显示

`1573788.0`在提交器检查可用schedd负载、EOS兼容性和WRITE权限后选择`bigbird20.cern.ch`，2 CPU/6 GB，ExitCode=0，实际allocation=114 s。输出20页PDF、事件keys及首屏PNG；首屏和第3/7/10/15页均已实际查看，正式resolution与paired-effects图也已查看。

| Page | Key (run/chunk/row) | Truth → prediction | Pixel / bar / tracker channels | p(tau) |
| --- | --- | --- | --- | ---: |
| 3 | 10081/674/317 | CC nu_e → CC nu_tau | 72 / 542 / 22 | 0.52296543 |
| 7 | 10082/1177/798 | CC nu_mu → CC nu_tau | 107 / 532 / 19 | 0.53889525 |
| 10 | 10083/438/64 | CC nu_tau → CC nu_mu | 7 / 559 / 3 | 0.06792136 |
| 15 | 10084/551/845 | NC → CC nu_tau | 43 / 3,031 / 5 | 0.53235644 |

这些页按固定confusion组的最高置信度例子选取。三个假τ均有稀疏pixel和非空bar响应；第10页为已注释的muon分支真τ，pixel仅7个通道，tracker三站各一个hit。四例都在既有pixel-centered域之外。图像与计数提示后续检查coverage、模态和轨迹关联；它们与整体错误率的关系及机制贡献仍需冻结对照给出统计证据。E/vertex和τ分支只用于显示与诊断。

### 产物与复现

正式根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。

- `first_scientific_model.json`：指标、完整history、真实终态/allocation、12项checkpoint检查、配对、τ分支与事件图复核。
- `evaluation_final/summary.json`、`metrics.csv`、`resolution_comparison.png/pdf`、`paired_comparisons.pdf`：正式部分评价。
- `evaluation_final/event_displays/pn_native_joint_s42_e4_v2/`：20页PDF、`examples.csv`、复核PNG和`review.json`。
- `evaluation_first_model.log`、`submission_display_012318.json`与`logs/display.1573788.*`：执行日志和调度证据。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.submit display --root "$S2_RESULTS/h100" \
  --evaluation-tag evaluation_final --experiment pn_native_joint_s42_e4_v2
```

01:30:56 UTC累计37.71104 GPU-hours，其中正式H100目录4.80427 GPU-hours，missing history records=0，组织上限200 GPU-hours。释放的H100槽位已由原队列ParT 1mm projection接续；正式矩阵为1 complete、4 running、30 idle。Kubeflow v6仍Pending、无node。

## 阶段性结论

首个正式模型完成了训练、概率/checkpoint读回、预定配对与原生事件图复核。固定能谱/顶点支持上的AP增量有条件事件区间支持，原样本及1% FPR效率增量仍不确定；跨种子稳定性与架构/粒度贡献尚待测量。输入全空与tracker-only域的先验型输出是当前模型的实测信息边界。

## 下一步计划

继续其余34个正式配置，按完成结果补齐41项单seed和4项三种子均值比较。正式ParT native joint完成后生成并查看其原生错误事件。全部配置、分域/分支、交互效应、资源及错误图齐全后，去掉`--allow-incomplete`执行严格评价，再完成S2科学验收。
