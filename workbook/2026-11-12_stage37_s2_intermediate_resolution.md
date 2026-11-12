# 阶段 37：S2 正式中间粒度结果

ParticleNet 100μm joint四轮完成，τ AP=0.35548005。native−100μm 的τ ΔAP=−0.00037532，pointwise 95% CI=[−0.00552033,+0.00501993]；固定common-support ΔAP=+0.00054605，CI=[−0.00476111,+0.00600712]。当前seed42、四轮和既定局部压缩下，没有native正收益的区间支持，可以排除这一比较至少+0.01的收益；跨种子、表示机制及完整粒度曲线仍待完成。

## 阶段目标

补充正式100/200μm尺度曲线，区分native到中间粒度的收益与1mm压缩效应。本次完成PN100μm；其余中间粒度任务继续原队列，尚未测量的结果保留缺项。

## 工作内容

`1573782.8`真实终态JobStatus=4、ExitCode=0，实际allocation=5,573 s。05:02:57 UTC，14项CPU checkpoint/optimizer/预测读回全部pass，包括四轮完整性、冻结配置和环境、validation keys、40k逐轮概率、逐轮NLL/AP重算和选模一致性。资源核算及部分评价正常退出，评价commit=`577204e`；当前13 results（4 references+9正式模型）、349行分域指标、39个历史三类诊断、13项预定配对、32项pending、2组factorial。resolution图与paired-effects首页已实际查看。

模型为ParticleNet dim64、161,156 parameters、pixel+bar、seed42、batch4、4 epochs、20,016 attempted steps/epoch，H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。100μm count表示不保留原生cell残差；同平面4mm分块、canonical chunks≤32 nodes局部learned compression后处理全部全局块。结果以这一压缩及提取方法为条件。

## 详细数据

沿用已接受detector-proxy与source-family split：独立train80k/validation10k、每类20k/2,500。各轮exposures=80,040/80,044/80,040/80,040，总320,164；80,064 applied updates、0 AMP skips。validation NLL选中epoch4，calibration/test未使用。

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.29155009 | 0.32051407 | 1,410.62940 |
| 2 | 1.27136835 | 0.32616570 | 2,783.13069 |
| 3 | 1.25358302 | 0.35157427 | 4,173.64345 |
| 4 | 1.23904515 | 0.35548005 | 5,559.01690 |

程序wall=5,559.87420 s，实际allocation=5,573 s=1.54805556 GPUh，额外allocation=13.12580 s，peak allocated memory=1,589.68506 MiB。Balanced accuracy=0.40400、macro F1=0.40106373、Brier=0.67395955；各类AP依次为e/μ/τ/NC=0.59923933/0.41802202/0.35548005/0.37119544，argmax recall=0.5016/0.2736/0.5460/0.2948。AP使用各类25%的validation mixture。

### 粒度比较与实际成本

| PN joint representation | Tau AP | Fixed common-support AP | All-attempt GPUh |
| --- | ---: | ---: | ---: |
| Native | 0.35510474 | 0.32202830 | 1.75278 |
| 100μm | 0.35548005 | 0.32148225 | 1.54806 |
| 200μm | 未测量 | 未测量 | 运行中 |
| 1mm | 0.35303603 | 0.32318232 | 1.43639 |

区间采用400次、128个accepted-family/chunk blocks（每类32）的配对bootstrap；固定common support为8,559 events、Neff=7,959.41365。native−100μm的raw/common-support区间见开头，二者均跨0、各自上界低于+0.01。这是逐项、以已训练模型及固定支持权重为条件的事件区间，不构成一般等效结论。

共同allocation预算=5,572.14270 s，100μm选epoch4，native只能选epoch3（AP=0.34718550），native−100μm ΔAP=−0.00829455，native剩余802.66254 s。全部准备和额外allocation已计入；该结果为整轮分辨率点估计，CI未测量，不能据此确定一般算力优势。

### 工作点、分支与域

100μm的validation empirical 1% FPR阈值=0.46034747，通过77/2,500个τ与75/7,500个背景，效率=0.0308；native通过83个τ，效率差=+0.0024，CI=[−0.00644214,+0.00701903]。0.1% FPR通过13个τ、7个背景，尾部统计不足，最终阈值留待S6。

τ electron/muon/hadronic信号数445/395/1,660，inclusive 1% FPR分别通过7/4/66个；分支AP=0.08356863/0.05911594/0.29432017，使用各分支信号与同一7,500个non-τ背景。相对native通过数变化为0/0/−6，分支差值CI未测量，物理机制尚未确定。

| PN 100μm validation domain | Events / tau | Tau AP |
| --- | --- | ---: |
| Pixel centered | 885 / 176 | 0.32753228 |
| Large scintillator | 4,494 / 1,053 | 0.38745947 |
| Pixel+bar input empty | 440 / 148 | 0.33636364 |
| Tracker-only response | 158 / 58 | 0.36708861 |

空输入和tracker-only域AUC=0.5，AP等于域prior。不同域AP包含样本组成变化，域内配对CI未测量。

### 产物与复现

正式根目录 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`：`pn_100um_comparison.json`保存manifest/history、14项readback、模型与配对、分域及核算快照；`readback_pn_100um.log`、`accounting_pn_100um.log`、`evaluation_pn_100um.log`为执行日志。评价表和图位于`evaluation_final/`。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments pn_100um_joint_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
```

05:03:43 UTC组织累计55.82911496 GPUh、missing history records=0。05:06:29 UTC正式9 complete、3 running、23 idle、0 held；ParT100/200μm在最后一轮、PN200μm在第三轮。Kubeflow GPU probe v6仍Pending、无node。

## 阶段性结论

首个正式中间粒度模型及证据完整。PN native和100μm在当前设置下的AP差值小、区间跨0，并排除至少+0.01的native收益。固定四轮与同成本点估计需共同保留；完整曲线和机制判断继续等待其余粒度及信息控制。

## 下一步计划

完成另外三项中间粒度训练与readback，补齐两族尺度曲线及预定相邻粒度配对；继续23项控制/额外种子。全35项、41项单seed+4项均值、成本和物理解释齐备后执行严格最终评价，S2当前保持active。
