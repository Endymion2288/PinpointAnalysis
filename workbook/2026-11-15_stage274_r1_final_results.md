# R1：完整基线、三类与四类比较及最终验收

**R1完成：41/41模型、133项配对、21项三seed平均配对、11个评价域和55条压力路径已交付，452项验收全部通过。B1 ResNet在本轮配方下表现最好，但尚未建立训练收敛或全局最优性。保留四类研究，同时交付真实三类基线；下一步优先做训练配方与数据规模的分离实验。R2尚未启动。**

## 阶段目标

完成新版路线图R1及W04/W05/W06/W08：在相同accepted observations和冻结事件上比较B0–B4的T3/T4，交付合作组三类域表、τ污染、同曝光与同allocation等时预算对照、训练健康、压力结果和完整资源账本。按预定研究门槛判断实际改进，不以完成训练代替性能验收。

## 工作内容

最终训练与评价源码快照来自Git `4567ab8`。16项神经网络LR screen、seed42四pass排名、B1/B2及共同参考B4的三seed扩展、B0三seed、MAE初始化参考、普通CNN和重训无tracker控制均完成。最后两个B2 compute完成后，唯一finish控制器依次提交完整nominal评价、`review --require-full`和英文图表。统计及图表通过HTCondor运行；本阶段只执行终态账本、accept、summary及文档收尾。

最终结果导出为[roadmap/v2/r1_results.json](../roadmap/v2/r1_results.json)，研究与执行入口为[新版路线图](../roadmap/README.md)。旧S0–S6独立test结论保留在原报告；本次是新Tune上的开发比较。

## 详细数据

### 样本、统计和比较口径

- Train为嵌套池中的100k四类均衡事件。神经网络主T3实际监督75k非τ unique events，主T4监督100k；共同75k非τ事件均完整曝光4次，分别为300k和400k总训练曝光。HGB使用相同unique事件，boosting预算按CPU复杂度单列。
- Tune为40k、每类10k；共同三类指标使用完全相同的30k非τ事件。T4先取`p(e,mu,NC)`并重归一化。10k真τ另输入T3统计强制归类与污染，不重标为NC。
- 全域和几何域均按任务等类benchmark加权，同时保留raw人数、raw/weighted confusion、Neff及工作点coverage。11域包括all、pixel-centered、large、active/empty、tracker active/empty和四档occupancy。
- Nominal使用6,845个冻结physical blocks、400次配对bootstrap，RNG seed为20261116。区间是已拟合模型及实际runtime条件下的pointwise事件区间；三seed sample SD单列，不是训练seed总体置信区间。未执行最终Confirm/Holm检验。
- B0/B1/B2/B4主任务为seeds17/42/2026；B3及MAE参考只有seed42。B1 seed42为Tesla T4/FP32，另两seed为H100 1g MIG/BF16 encoder+FP32 head；B2为H100 NVL/BF16，B3为A100/BF16。seed波动同时包含实际执行配方差异。
- B2 T3/T4分别由screen选择LR `0.0001/0.0003`；类别数比较包含各任务已选择的配方，不能把全部差异归因于类别数。CNN成本控制同样包含其实际数值配方，不是仅替换backbone的严格消融。
- Cal-A、Cut-B、Confirm payload始终封存。工作点在Tune raw scores上描述性选取，尚未成为独立冻结部署阈值。

### 主任务与同计算结果

下表为单模型指标的seed均值，不是概率ensemble。BA以百分数表示，NLL使用自然对数；括号内为BA的sample SD，单位pp。单seed的SD记为“—”。

| Family | Seeds | T3 BA3 (%) | T4 conditional BA3 (%) | Compute T3 BA3 (%) | T3 NLL3 | T4 conditional NLL3 | Compute NLL3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 HGB | 3 | 54.5900 (0.0000) | 54.5300 (0.0000) | — | .897449 | .897246 | — |
| B1 ResNet | 3 | 56.5633 (.2401) | 56.3044 (.1721) | 56.3689 (.2100) | .870377 | .867105 | .875549 |
| B2 ParticleNet adaptation | 3 | 55.0689 (.0619) | 55.3911 (.2550) | 55.0778 (.2411) | .899966 | .892534 | .899183 |
| B3 Geometry ParT | 1 | 55.0600 (—) | 55.2900 (—) | 55.3700 (—) | .897301 | .893037 | .889884 |
| B4 revision3 Scratch | 3 | 54.3922 (.0674) | 54.4867 (.2150) | 54.6744 (.0486) | .902022 | .904369 | .899337 |
| B4-MAE reference | 1 | — | 54.7200 (—) | — | — | .890817 | — |

冻结排名仍按seed42、T3/T4共同BA3均值排序：B1/B2/B3/B4为56.5567/55.3950/55.1750/54.5133%。B2−B3为+0.220 pp，CI95[−0.0468,+0.4944]，点排序不等于已确认的架构优势。三seed实际配方下，B2相对B1的T3 BA低1.4944 pp [−1.8101,−1.1625]，T4 conditional低0.9133 pp [−1.2021,−0.6423]。这些是本轮预算下的比较，未覆盖各族收敛后的上限。

| T4 family | BA4 (%) | NLL4 | Tau efficiency at 1% FPR (%) | Tau efficiency at 0.1% FPR (%) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 43.1025 | 1.214964 | 2.8300 | .2600 |
| B1 | 44.5733 | 1.187673 | 5.0200 | 1.1667 |
| B2 | 43.3658 | 1.214989 | 2.9933 | .4400 |
| B3, seed42 | 43.2775 | 1.214732 | 2.7000 | .2600 |
| B4 | 42.3217 | 1.225322 | 2.5967 | .3300 |
| B4-MAE, seed42 | 43.0325 | 1.212520 | 2.4100 | .3200 |

各模型全域τ工作点的实际FPR分别为1%和0.1%，对应本Tune的300/30个非τ背景通过；完整ties和raw selections保存在结果中。B1 seed42在1%点选中509/10,000个τ，背景e/μ/NC为48/132/120。在固定分数、固定cut及类条件响应的情景下，其均衡prior purity为62.917%，τ prior=.001时为.5069%；这不是物理flux或产额测量。绝对效率仍不足以支持高纯稀有τ分析。

### 预定研究门槛的实际判定

最终summary对28项逐seed及8项三seed平均比较给出36条`research_criteria`。以下为三seed结论；所有差值为candidate−reference。

| Contrast | Main measured effect | Research criteria | Reason |
| --- | --- | --- | --- |
| B1 T3 − HGB T3 | ΔBA3 +1.973333 pp [1.513219,2.378174] | Fail | 精确点值低于预设2 pp，不四舍五入判通过；其余约束通过 |
| B1 T3 − B4 T3 | ΔBA3 +2.171111 pp [1.807777,2.519771] | Fail | μ recall −2.410 pp [−3.208548,−1.690460]，未满足1 pp逐类非劣 |
| B2 T3 − HGB T3 | ΔBA3 +.478889 pp [.104135,.896437] | Fail | 实用BA增益及e recall非劣未满足 |
| B2 T3 − B4 T3 | ΔBA3 +.676667 pp [.382897,1.010675] | Fail | 实用BA增益未满足；μ recall −3.773333 pp |
| B1 T4 − HGB T4 | Δτ efficiency +2.190 pp [1.663966,2.827275] | Pass | τ增益、共同BA3、NLL4及实际FPR记录满足预定条件 |
| B1 T4 − B4 T4 | Δτ efficiency +2.423333 pp [1.940786,2.977398] | Pass | ΔBA3 +1.817778 pp，ΔNLL4 −.037648562；仍须报告μ recall代价 |
| B2 T4 − HGB T4 | Δτ efficiency +.163333 pp [−.316053,.589963] | Fail | 未达1 pp实用值，区间跨零 |
| B2 T4 − B4 T4 | Δτ efficiency +.396667 pp [.049216,.685945] | Fail | 正的小效应低于1 pp实用值 |

B3单seed相对B0/B4的T3和τ条件均未全部满足。B1 T4通过的是既定τ研究资源门槛，其合同约束共同BA3而非逐类recall；这不能掩盖其相对B4的conditional μ recall −1.8533 pp [−2.6109,−1.0634]，也不构成最终独立确认。

### 真实三类、四类与预算的关系

BA差单位pp，NLL差为原单位；B1/B2/B4为三seed均值，B3为seed42。

| Contrast | Delta BA3 [95% interval] | Delta NLL3 [95% interval] |
| --- | ---: | ---: |
| B1 main T3 − T4 conditional | +.258889 [.041753,.478975] | +.003272 [.001661,.004801] |
| B1 compute T3 − T4 conditional | +.064444 [−.178610,.302260] | +.008444 [.006379,.010497] |
| B1 compute T3 − main T3 | −.194444 [−.427739,.034225] | +.005171 [.003806,.006575] |
| B2 main T3 − T4 conditional | −.322222 [−.510707,−.103931] | +.007432 [.005675,.008952] |
| B2 compute T3 − T4 conditional | −.313333 [−.513827,−.112856] | +.006649 [.005003,.008099] |
| B2 compute T3 − main T3 | +.008889 [−.177952,.204718] | −.000783 [−.001761,.000248] |
| B3 compute T3 − T4 conditional | +.080000 [−.258352,.437231] | −.003153 [−.005268,−.000719] |
| B3 compute T3 − main T3 | +.310000 [−.081957,.641937] | −.007417 [−.009402,−.005469] |
| B4 compute T3 − T4 conditional | +.187778 [−.071898,.412885] | −.005032 [−.006562,−.003441] |
| B4 compute T3 − main T3 | +.282222 [.088207,.468171] | −.002685 [−.003671,−.001616] |

十项compute控制均满足±5%操作wall预算，实际相对误差范围[−.280963%,+.113357%]。预算在本族、本seed的同一allocation内匹配fresh T4，包含loader、训练、验证与最终推理；不同族使用不同T4时长，并非跨族统一计算量。wall-based schedule与重复曝光同时改变，不能将其解释为单独的曝光效应。

B1 compute−T4的e/μ/NC recall差为−.626667/+6.583333/−5.763333 pp；B2为−1.10/+2.74/−2.58 pp。三类化没有稳定提升共同任务，增加预算也未消除逐类取舍。

| Last B2 compute | Operation wall (s) | Target wall (s) | Exposures | Updates | BA3 | NLL3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed17 | 4983.629454 | 4983.576084 | 382172 | 47830 | .548000 | .907466 |
| seed2026 | 4183.928768 | 4183.187681 | 379402 | 47481 | .552333 | .890066 |

### 合作组三类、τ污染与局部困难

[合作组对照](../roadmap/v2/08_r1_collaboration.md)交付18行/126值的all、pixel-centered、large表，已与最终evaluation逐项核对。seed42 B1 T3 BA3为56.620/73.817/68.891%；相应T4 conditional为56.493/74.185/69.038%。B1 FP32无tracker重训的全域BA3为52.693%，比有tracker低3.927 pp；这是实际输入增量，不等同于固定权重移除压力。

T3无法显式识别τ。下表是同10k真τ的seed42强制归类人数，分母保持10k，不是所选样本的τ纯度；完整污染矩阵及先验口径在最终JSON。

| T3 model, seed42 | Assigned e | Assigned mu | Assigned NC |
| --- | ---: | ---: | ---: |
| HGB | 2957 | 3094 | 3949 |
| B1 | 3049 | 2939 | 4012 |
| B2 | 2656 | 3182 | 4162 |
| B3 | 2878 | 3013 | 4109 |
| B4 | 2426 | 3982 | 3592 |

R1的投影、tracker set、网络尺寸、4 passes和新事件划分与Wu/Mahlstedt原研究不同；其原事件身份、12维tracker、权重及完整cuts尚未取得。当前为明确描述的同类方法对照，不能量化宣布优于合作组原模型。

三seed的低occupancy μ损失已独立复核：[stage266](2026-11-15_stage266_r1_occupancy_three_seed_readback.md)中channel<100的2,801个μ，B1−B4 T3 recall差为−11.46/−24.24/−12.03 pp；全域贡献均值−4.45667 pp。241个全空μ的正确数差均为0。另一方面，[stage262](2026-11-15_stage262_r1_muon_working_points_and_time_handoff.md)中B1相对B4的μ 1% FPR效率高4.850 pp [4.057,5.823]，90% benchmark purity效率高8.503 pp。argmax取舍、分数排序和信息损失需要分别检验，不能仅凭某项recall诊断不可逆表示缺陷。

### 学习曲线、压力与系统性不足

seed42八个四pass主模型的末轮Tune-screen NLL3都继续下降，B1 T3/T4为−.000704/−.012936，B2为−.007568/−.011267，B3为−.016398/−.016055，B4为−.007754/−.018221。四类模型末轮NLL4下降.033923–.050080。未建立收敛，不能把本轮排序称为各架构最佳效果。

55条固定权重压力路径已完成400 draws，包含11个seed42模型×nominal/noise/combined/remove_pixel/remove_tracker；压力子集为1,600个事件，非τ三类1,200个。例：B1 T3 nominal/noise/combined BA3为57.583/57.417/56.833%；B4-MAE T4为55.333/52.667/52.917%。这些子集值不能与全域40k结果直接作差。固定权重remove_pixel不回答从scratch去掉pixel后的可替代信息量；完整压力配对和固定nominal cut结果保存在`evaluation/stress_results.json`。

**事实：**本轮补齐了同输入基线和真实T3；原生复杂backbone未在当前预算下胜过B1。三seed非劣失败、四pass仍改善、低occupancy损失及固定权重压力说明训练、决策和表示都需要继续研究。**待验证假设：**投影stride/采样相位、分层压缩、缺少显式跨层连续性、分支优化和局部类别比例可能影响表现；目前证据尚不能选定其中某一项为原因。

[stage271](2026-11-15_stage271_r1_sampling_metadata_review.md)的24组metadata重放显示single-class batch均少于.3%，不支持大段按类训练；局部窗口仍有类比例自相关，其分类影响未测量。R2保留滚动类频率/logit诊断，出现模型证据后才加重混控制。

### 验收、图表和终态资源

| Validation | Passed | Failed | Scope |
| --- | ---: | ---: | --- |
| Final accept | 452 | 0 | 完整矩阵、身份/曝光、预算、工作点、图表、终态 |
| Independent nominal review | 2557 | 0 | 概率、人数、raw/weighted confusion、工作点、曝光 |
| Independent core intervals | 2512 | 0 | 79个核心model-domain、133配对、21平均配对、seed SD |
| Independent stress review | 5666 | 0 | 55路径的类均值、固定cut、原始selection和点差 |
| Final document readback | 735 | 0 | 18行合作组域表、主任务/四类表、10项预算配对、阈值合同及385个本地链接 |

核心区间复核直接对逐事件正确判定和log-loss独立加权求和，沿用相同冻结block draws；最大点/端点/seed SD误差分别9.215e−15/7.161e−15/4.696e−15。Nominal评价操作192.629970 s，核心区间独立操作28.547352 s。新增模块接入后的27项R1与summary tests已在[stage273](2026-11-15_stage273_r1_core_interval_readback.md)通过，本阶段无代码变化，未重复运行。

文档读回使用最终summary逐项核对显示精度，并从逐模型指标独立求三seed平均BA/NLL差；原T3/τ数值门槛与HEAD一致，所有本地链接存在。证据为`audit/final_document_readback_stage274.json`；这项核对不读取新的事件payload，也不替代上表的预测和区间审查。

11套英文PDF/PNG全部生成并目视：共同三类、T3 ROC/PR、T4 conditional ROC/PR、完整四类ROC/PR、confusion、τ ROC/PR、raw scores、resource Pareto、learning curves、stress、matched compute。contact sheet和三项成本/压力/预算图已分别查看，未发现裁剪或标签缺失。图表留在`R1/figures/`。

2026-11-15 08:22:02 UTC终态账本共122条allocation记录，`active_or_unresolved=[]`。最后B2的Condor1582928/1582929均status4、Exit0，完整wall14,182/11,914 s、各5 CPU；最终CPU评价、复核和图表也均终态。

| Device / partition | Allocated device-hours |
| --- | ---: |
| H100 NVL | 14.335000 |
| A100-PCIE-40GB | 14.829444 |
| H100 1g.12gb MIG | 7.073333 |
| H100 2g.24gb MIG | 2.327500 |
| H100 NVL 3g.47gb MIG | 2.685833 |
| T4 | 4.298611 |
| V100 failed compatibility attempts | .063333 |
| h100-47gb flavor, early attempt without runtime model | .095833 |
| Total heterogeneous allocations | **45.708889** |

Allocated CPU为**307.181111 core-hours**。总device-hours为原R1 18 h规划的2.539倍，差额27.708889 h；包含编译、失败、取消前已运行部分、迁移和探针，不是41个最终模型wall之和。跨GPU/MIG小时不作FLOP等效量。

复用MAE3历史SSL操作为2,287.156994 s、.635321 device-hours（98k训练+2k内部reconstruction probe），单列初始化成本，不算新R1开支。已记录交互操作下界1,171.919903 s=.325533 h，启动/未记录失败开销未测量，不能称总allocation成本。Kubeflow凭据约00:08 UTC失效，已保存全部历史终态Pod/Job；最后账本使用`--cached-kubeflow`，没有以登录不可用代替资源终态证据。

## 执行命令与产物

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export R1_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT" --cached-kubeflow
python -m analysis.flavour.v2.r1.reference_costs --root "$R1_ROOT"
python -m analysis.flavour.v2.r1.accept --root "$R1_ROOT"
python -m analysis.flavour.v2.r1.summarize --root "$R1_ROOT" \
  --plan roadmap/v2/experiment_plan.json --output roadmap/v2/r1_results.json
python3 "$R1_ROOT/audit/final_document_readback_stage274.py"
git diff --check
```

完整nominal/review/plots由Condor1582934/1582935/1582936执行；配置及实际payload位于`R1/runtime/`，不可变源码由对应submission的`source_archive`定位。最终证据为`acceptance.json`、`resources.json`、`reference_costs.json`、`evaluation/{results,independent_review,independent_core_intervals,independent_stress_review}.json`、`evaluation/exposure_and_compute.csv`及`figures/`。模型、预测和日志留results；仓库只提交代码、规格、摘要与报告，用户`docs/`保持原未跟踪状态。

## 阶段性结论

**已测量：**B1在本轮三类BA、NLL和τ工作点上最强，τ相对B4/HGB有满足预设研究值的增益；三类重训没有稳定胜过四类条件结果，四类主线值得保留。B1三类相对HGB仍差.026667 pp才达到2 pp实用门槛，对B4存在μ非劣失败。全部交付验收通过与这些科学门槛失败同时成立。

**尚未建立：**各模型的收敛上限、跨层/压缩缺陷的因果解释、严格合作组原研究复现、300k/1M规模收益、真实低背景τ产额及最终独立确认。R1结果不替代旧S6 test，也没有读取新的Confirm。

## 下一步计划

1. **R2先选B1一族。**冻结fresh D0=100k×4、D1=100k×12、D2=300k×4及相同实际数值配方，先profile再分批提交。D1/D2的T4曝光均1.2M、T3均.9M；保留任务各自的模型/输入/LR/batch/loss和5%warm-up、update-normalized cosine、5%floor。壁钟只管成本/deadline。
2. **提交前实算sampler。**从冻结池生成updates、尾批分布、LR轨迹/`sum(lr)`、滚动类频率；同曝光不保证同updates。新D0匹配精度与schedule，固定预算和best-screen结果分别报告。逐窗口检查NLL、AP/ROC、e/μ/NC recall及低occupancy logit；仅有模型证据时追加顺序重混或交叉拟合readout诊断。
3. **按成本决定扩展。**B1 D1+D2四项线性估算10.087 h，B2另一族13.406 h，未计新D0、转换和失败；先一族回答数据/曝光问题，再按实际余量决定B2/B3或容量对照。R1已超规划，不能照搬原80 h总额度下余下60 h的完整矩阵。
4. **R3做表示单因素。**先stride/相位与高分辨局部残差、窗口边界通信和显式cross-plane geometry，保留低occupancy分层与内部健康；R4再以candidate recall、fake/efficiency/coverage推进对象。
5. **R5–R7按证据推进。**保留flat4，比较3+τ及条件decay头/τhad specialist；分支有效标签只用于训练与事后分层，不作truth推理路由。强监督之后才扩大定向SSL；最后锁模型、Cal-A、Cut-B、Confirm及主检验。文献及适用边界见[05](../roadmap/v2/05_literature.md)，本阶段没有新增R2训练。

GPU继续采用尽快获得可用资源的原则，HTCondor与Kubeflow并用，接受不同型号、共享设备与MIG；同allocation内配对、记录真实精度/吞吐，避免为统一型号或整卡等待。后续具体工单和验收见[04](../roadmap/v2/04_execution_plan.md)。
