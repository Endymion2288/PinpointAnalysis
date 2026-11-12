# 阶段 36：S2 两族正式四格与成本比较

正式 H100 的 ParT、ParticleNet native/1 mm × joint/projection 八个模型已完整。ParticleNet native−1mm joint 的 τ ΔAP=+0.00206870，pointwise 95% CI=[−0.00485307,+0.00996035]；1mm pairing ΔAP=+0.00202755，CI=[−0.00647042,+0.00934805]。两者均未得到正区间支持；ParticleNet 交互效应同样未确定。当前结果以 seed42、四轮、既定局部压缩和已接受 detector-proxy 为条件。

## 阶段目标

完成第二组正式四格，回答 ParticleNet 的粒度、pairing 及交互问题，并补齐1mm架构和实际成本比较。样本沿用已接受的 source-family split，calibration/test 保留至 S6。

## 工作内容

ParticleNet 1mm joint（`1573782.6`）真实终态 JobStatus=4、ExitCode=0，allocation=5,171 s。04:43:32 UTC，14项 CPU checkpoint/optimizer/预测读回全部pass，覆盖冻结配置、环境、NLL选模、10k validation keys、40k逐轮概率、逐轮重算指标、恢复状态及exposures。资源核算和部分评价均正常退出；评价 commit=`072ab88`，12 results（4 references+8正式模型）、322行分域指标、36个历史三类诊断、12项配对、33项pending、2组factorial。resolution图及paired-effects首页已实际查看。

新模型为ParticleNet dim64、161,156 parameters、pixel+bar、seed42、batch4、4 epochs，每轮20,016 attempted steps；H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。同平面4mm分块、canonical chunks≤32 nodes 的局部 learned compression 后处理全部全局块。这里的两族方法均以这一表示为条件，未经压缩的全事件模型尚未测量。

## 详细数据

独立 train 80k、validation 10k，各类20k/2,500；每轮exposures=80,040/80,044/80,040/80,040，总320,164。80,064 applied updates、0 AMP skips；NLL选中epoch4。

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.35941875 | 0.27630291 | 1,325.81062 |
| 2 | 1.34223826 | 0.29606821 | 2,593.62395 |
| 3 | 1.27005382 | 0.34874996 | 3,870.14674 |
| 4 | 1.24970777 | 0.35303603 | 5,144.51787 |

程序wall=5,145.06316 s，实际allocation=5,171 s=1.43638889 GPUh，额外allocation=25.93684 s；peak allocated memory=1,552.50732 MiB。相同四轮下，1mm joint比1mm projection多分配1,214 s，但性能差需结合配对区间理解。

### 两族四格

| Model | Tau AP | NLL | Balanced accuracy | Fixed common-support AP | All-attempt allocation GPUh |
| --- | ---: | ---: | ---: | ---: | ---: |
| ParT native joint | 0.35858391 | 1.22107879 | 0.41840 | 0.32383619 | 1.94250 |
| ParT native projection | 0.35733812 | 1.22337225 | 0.41500 | 0.32248272 | 2.78250 |
| ParT 1mm joint | 0.35502857 | 1.24293540 | 0.40330 | 0.32086106 | 2.02167 |
| ParT 1mm projection | 0.35421688 | 1.22571671 | 0.41370 | 0.31995100 | 1.80083 |
| ParticleNet native joint | 0.35510474 | 1.23930563 | 0.39740 | 0.32202830 | 1.75278 |
| ParticleNet native projection | 0.34797906 | 1.24334514 | 0.39620 | 0.31945295 | 1.02306 |
| ParticleNet 1mm joint | 0.35303603 | 1.24970777 | 0.39770 | 0.32318232 | 1.43639 |
| ParticleNet 1mm projection | 0.35100848 | 1.23785700 | 0.40480 | 0.32166112 | 1.09917 |

ParT native projection的成本包含原不完整尝试和完整替换，阶段35已说明。新PN1mm joint的macro F1=0.39400330、Brier=0.67899814。各类AP/ROC AUC/argmax recall分别为：e 0.58784082/0.77331109/0.5088，μ 0.40652015/0.65573568/0.2888，τ 0.35303603/0.62838139/0.5268，NC 0.37074632/0.64007987/0.2664。AP以各类25%的validation mixture为条件。

### 新增预定配对与交互

400次配对 accepted-family/chunk bootstrap，128 blocks、每类32；固定common support为8,559 events、Neff=7,959.41365。区间为逐项pointwise区间，固定支持区间以既定权重为条件。

| Prespecified B−A | Raw tau ΔAP [95% CI] | Fixed common-support ΔAP [95% CI] |
| --- | --- | --- |
| PN native joint − 1mm joint | +0.00206870 [−0.00485307,+0.00996035] | −0.00115402 [−0.00722797,+0.00565627] |
| PN 1mm joint − projection | +0.00202755 [−0.00647042,+0.00934805] | +0.00152121 [−0.00829112,+0.00956812] |
| ParT − PN, 1mm joint | +0.00199254 [−0.00584165,+0.01104786] | −0.00232126 [−0.01033698,+0.00601845] |

新配对的raw/common-support区间均跨0。前两项数值上界低于预定+0.01；native−1mm raw上界仅低0.00003965，且bootstrap为400次，不能把这一边界位置解释为稳定的排除结论。1mm架构比较raw上界高于+0.01，当前既未确定架构优劣，也未排除该量级的ParT收益。三种子仍待完成。

ParticleNet交互 `AP_NJ−AP_NP−AP_CJ+AP_CP`=+0.00509813，CI=[−0.00510744,+0.01683727]；ParT为+0.00043410，CI=[−0.00790547,+0.00875624]。两者均无粒度依赖的区间支持。PN native pairing原样本ΔAP=+0.00712568，CI=[+0.00003031,+0.01448372]，固定common-support区间跨0；这一事件区间结论与整体交互未确定同时成立。

### 同实际allocation预算

所有额外allocation均在首个可用checkpoint之前计入；按预算内validation NLL选整轮checkpoint，下列为点估计，CI未测量。

| B−A | Shared budget [s] | A / B selected epoch | Tau ΔAP |
| --- | ---: | --- | ---: |
| PN native − 1mm joint | 5,170.45471 | 4 / 3 | −0.00585053 |
| PN 1mm joint − projection | 3,955.74365 | 4 / 3 | −0.00225852 |
| ParT − PN, 1mm joint | 5,170.45471 | 4 / 2 | −0.02844170 |

三项B剩余预算分别400.97456、59.66007、1,474.81756 s。最后一项ParT epoch3放不进预算，只能选epoch2。固定四轮的正点估计在同成本整轮比较中转负，说明目前训练吞吐和checkpoint分辨率会影响排序；这些单seed点估计尚不能确定一般算力优势。

### 工作点、分支与域

PN1mm joint的validation empirical 1% FPR阈值=0.44543493，通过89/2,500个τ与75/7,500个背景，τ效率=0.0356；对应projection为80个τ，native joint为83个τ。1mm pairing效率差=+0.0036，CI=[−0.00637392,+0.01321110]；native−1mm joint为−0.0024，CI=[−0.01234629,+0.00489018]。0.1% FPR通过9个τ、7个背景，尾部统计不足，最终阈值留待S6。

| Tau branch | Signal events | 1mm joint selected | 1mm projection selected | Joint branch AP |
| --- | ---: | ---: | ---: | ---: |
| Electron | 445 | 10 | 8 | 0.07666659 |
| Muon | 395 | 8 | 4 | 0.06247147 |
| Hadronic | 1,660 | 71 | 68 | 0.29097515 |

通过数使用各自inclusive 1% FPR阈值，增加2/4/3个，合计9个；分支差值CI未测量，尚未确定物理机制。各分支AP使用该分支信号及同一7,500个non-τ背景。

| PN 1mm joint domain | Events / tau | Tau AP |
| --- | --- | ---: |
| Pixel centered | 885 / 176 | 0.31333254 |
| Large scintillator | 4,494 / 1,053 | 0.38287618 |
| Pixel+bar input empty | 440 / 148 | 0.33636364 |
| Tracker-only response | 158 / 58 | 0.36708861 |

空输入与tracker-only域的AUC=0.5、AP等于域prior。分域AP受组成变化影响，域内配对CI未测量；训练域及tracker消融继续按冻结矩阵运行。

### 产物与复现

正式根目录 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`：`second_complete_factorial.json`保留本次manifest/history、14项readback、四格、配对、交互、分域与核算快照；`readback_pn_1mm_joint.log`、`accounting_pn_1mm_joint.log`、`evaluation_second_factorial.log`为实际执行日志。评价位于`evaluation_final/`。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments pn_1mm_joint_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
```

04:44:19 UTC组织累计54.57364299 GPUh、missing history records=0。正式8 complete、4 running、23 idle、0 held。Kubeflow GPU probe仍Pending且无node，平台报告GPU不足及调度限制；该等待未产生GPU allocation。原Condor任务继续运行，没有迁移或重复writer。

## 阶段性结论

两族正式四格及输出证据齐备。新增粒度、pairing、1mm架构和交互的事件区间均跨0；当前结果支持继续分解表示信息、提取能力与训练成本，尚不支持一般等效或架构优劣结论。同成本点估计与固定四轮排序不同，后续同时保留两类比较。

## 下一步计划

继续27项未完整正式配置，优先获得100/200μm曲线，补齐信息、模态、训练域和plane-only控制及三种子。全部35项和41项单seed+4项均值严格评价完成后验收S2；当前goal保持active，不进入S3–S6。
