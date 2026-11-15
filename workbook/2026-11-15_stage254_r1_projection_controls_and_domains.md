# 阶段254：投影模型、tracker重训与合作组条件域

**B1三类模型在seed42的全域BA3为0.5662，相对HGB提高2.03个百分点，配对CI95为[1.57,2.52] pp；相对B4提高2.27 pp，但μ recall低6.91 pp。**相同非τ曝光下，B1的T3与T4条件三类BA差接近零，T4的NLL更好。模型是否更强需要同时指定参照模型、类别与报告域。

## 阶段目标

评价完整四pass的B1三类输出、FP32对齐的无tracker重训和普通CNN成本参照；给出全域、pixel-centered、large三个条件域的同事件比较及τ污染。完整架构排名和必要seed仍在执行，本阶段不作最终晋级判断。

## 工作内容

使用7个已完成模型，计算18项同事件配对、400次accepted-family连接的物理块重采样；另由独立数组/人数/曝光读回核对统计输入和点估计。所有结果使用工作点浮点边界修订`r1-tie-aware-float-boundary-2`。制作英文图，分别展示域依赖、平均BA差、逐类代价和μ域变化。

同时在不提前生成完整排名的条件下，证明B1已经必然位于冻结规则的前两名，并提交两个必要seed，减少后续等待。

## 详细数据

### 样本与统计口径

Train100k为四类各25k；T3实际使用相同pool中的75k非τ，四pass共300k曝光，逐事件恰好4次；T4共400k曝光，非τ也是逐事件4次。Tune40k为四类各10k，共同三类30k。重采样seed20261116、6,845个物理块、0个跨chunk块；同一draw用于配对模型，按class signature分层。

下述区间条件于已训练的seed42模型，描述开发Tune事件差异；不包含重新训练、架构选择或新的独立确认不确定度。工作点在Tune描述性选取，概率尚未校准。Cal-A/Cut-B/Confirm未读取。

### 全域共同三类结果

| Model | BA3 | NLL3 | Electron recall | Muon recall | NC recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 ResNet T3 | 0.566200 | 0.871967 | 0.6301 | 0.4998 | 0.5687 |
| B1 ResNet T4 conditional | 0.564933 | 0.866249 | 0.6306 | 0.4694 | 0.5948 |
| B4 native T3 | 0.543500 | 0.901241 | 0.5446 | 0.5689 | 0.5170 |
| HGB T3 | 0.545900 | 0.897449 | 0.5812 | 0.4881 | 0.5684 |
| B1 T3 without tracker, FP32 | 0.526933 | 0.898594 | 0.7094 | 0.4898 | 0.3816 |
| B1 plain CNN T3 | 0.547500 | 0.910778 | 0.6196 | 0.4791 | 0.5438 |
| B1 plain CNN T4 conditional | 0.544133 | 0.906572 | 0.6116 | 0.4647 | 0.5561 |

| Paired contrast, a−b | ΔBA3, pp [CI95] | ΔNLL3 [CI95] |
| --- | --- | --- |
| B1 T3 − HGB T3 | +2.030 [+1.567,+2.517] | −0.025482 [−0.029935,−0.020781] |
| B1 T3 − B4 T3 | +2.270 [+1.827,+2.740] | −0.029274 [−0.033395,−0.025241] |
| B1 T3 − B1 T4 conditional | +0.127 [−0.177,+0.491] | +0.005718 [+0.003526,+0.007861] |
| B1 full − without tracker | +3.927 [+3.386,+4.431] | −0.026627 [−0.029996,−0.023026] |
| CNN T3 − ResNet T3 | −1.870 [−2.320,−1.453] | +0.038811 [+0.034209,+0.043182] |
| CNN T4 − ResNet T4 | −2.080 [−2.523,−1.665] | +0.040323 [+0.036761,+0.044318] |

B1−HGB逐类recall差为e +4.89 pp [+4.15,+5.67]、μ +1.17 pp [+0.30,+2.02]、NC +0.03 pp [−0.68,+0.81]。对HGB这一参照，本seed达到预定2 pp实际BA增益、正下界、NLL和逐类退化约束；最终结论还需必要seed与同计算交付。

B1−B4的μ差为−6.91 pp [−7.74,−6.04]，不满足对B4的逐类非劣要求。B1 T3−T4的μ/NC差为+3.04/−2.61 pp，BA接近掩盖了类别之间的取舍。B1的T3在本seed并未全面优于T4条件三类。

### Tracker贡献与成本参照的解释

B1 full−无tracker的e/μ/NC recall差为−7.93/+1.00/+18.71 pp；对应区间为[−8.83,−7.09]/[+0.20,+1.81]/[+17.67,+19.72]。NC是本次平均BA增益的主要来源。该重训同时移除了tracker presence、hit数量和空间信息，不能据此单独归因于空间分辨率。后续presence/count-only控制可进一步区分信息来源。

主B1在Tesla T4、FP32执行；无tracker对齐控制在H100 MIG3g.47gb、FP32执行；CNN在H100整卡、BF16 encoder/FP32 head执行。训练和推理均允许matmul/cuDNN TF32；不同硬件支持与具体kernel使用不同，未测量逐kernel执行精度。因此这些是实际配方的结果，不是隔离硬件或精度因素的因果效应。

同T4/FP32的固定32事件microprofile见[r1_projection_cost.json](../roadmap/v2/r1_projection_cost.json)：ResNet/CNN分别793,674/179,370参数，median step为0.08410/0.06992 s。预加载microprofile不包含读入和validation，不能代替完整训练吞吐。CNN使用B1筛出的配方，是成本参照，尚未建立独立充分调参后的CNN上限。

### 合作组条件域

所有模型在全Train域训练；下列truth vertex mask仅用于报告，不作为模型输入或推理路由。

| Domain | Raw e / μ / τ / NC | Non-tau events | B1 T3 BA3 | NLL3 | Macro AUROC3 |
| --- | --- | ---: | ---: | ---: | ---: |
| All | 10000 / 10000 / 10000 / 10000 | 30000 | 0.566200 | 0.871967 | 0.769480 |
| Pixel-centered | 930 / 1050 / 757 / 1028 | 3008 | 0.738171 | 0.621752 | 0.897889 |
| Large scintillator | 4424 / 4764 / 4136 / 4678 | 13866 | 0.688915 | 0.706104 | 0.863680 |

以上指标在每个域内使用共同平衡三类prior；原始人数与observed-mixture结果保存在完整JSON。B1逐类AUROC为：全域0.815108/0.759404/0.733928，pixel-centered 0.920341/0.908932/0.864394，large 0.910648/0.859327/0.821065。

实现由`analysis.flavour.s2.prepare.regions`和`R1/data/plan.json`确定：target z范围[−2993,−1804.88] mm；pixel-centered在pixel sensor横向边界内缩20 mm、target起点后前500 mm；large在scintillator横向边界内缩15 mm、排除target末端200 mm，采用保存坐标的闭区间。原合作组权重、同事件清单及exact cuts不可得，目前完成的是定义明确的同构对照。

B1−B4的BA差在pixel-centered为+0.873 pp [−0.193,+2.044]，large为+2.734 pp [+2.189,+3.315]；μ recall差分别为+1.810 pp [−0.181,+3.839]和+5.877 pp [+4.713,+6.903]，与全域的负差不同。B1 full−无tracker的μ差也从全域+1.00 pp变成pixel-centered −2.857 pp、large −3.631 pp。域内高分和全域逐类质量回答不同问题，不能用较好fiducial域代替全域交付。

### 实际三类模型的τ混入

B1 T3对10k真τ强制分配至e/μ/NC为3,049/2,939/4,012。它没有τ拒绝输出。

在非τ共同三类1% FPR处，三项选择的raw通过人数如下；各类选择单独选cut，并非互斥路由。

| Selection | Selected e / μ / τ / NC | τ acceptance | τ fraction, balanced four-class input | τ fraction, prior πτ=0.001 |
| --- | --- | ---: | ---: | ---: |
| Electron | 2103 / 59 / 195 / 141 | 1.95% | 7.806% | 0.02542% |
| Muon | 40 / 1774 / 380 / 160 | 3.80% | 16.143% | 0.05778% |
| NC | 80 / 120 / 273 / 1343 | 2.73% | 15.033% | 0.05310% |

稀有prior情景将其余三类平均分配，固定scores、cuts与各类条件响应，只重新混合；它不是实际flux估计，也没有改变模型概率。0.1%工作点、1% τ prior及三个域的完整计数保存在产物中。三类输出可先交付，但真实开放输入的τ污染仍需明确计算。

### 调度、验证与产物

B1冻结排名分数为(0.566200+0.5649333)/2=0.5655667；已完成B2为0.553950、B4为0.5451333，唯一未完成族为B3。因此无论B3最终分数为何，B1最差第二名。此证明写入`runtime/b1_guaranteed_top_two.json`，没有生成或更改最终排名。

01:24 UTC提交B1 seed17/2026，Condor1582904/1582905；两项均在提交前自动查询schedd队列、EOS策略及WRITE权限，选择兼容且最低idle的bigbird20.cern.ch。记录名与控制器预期完全一致，以避免重复提交。GPU最低SM75、不指定型号或整卡；三个必要任务T4/T3/fresh同计算控制在同一allocation依次执行。其他族的扩展仍等待完整排名。

B2 seed42同计算控制1582877已终止，ExitCode0、allocation12,574 s；fresh T3总操作4,488.164 s，相对T4预算误差−0.03398%，通过±5%门槛。381,170次非τ曝光，BA3=0.552000、NLL3=0.900017；完整配对结论待最终统计。组织性GPU总成本仍在增长，最终值未测量。

本阶段183项独立检查通过；BA/NLL/工作点最大绝对差分别9.33×10⁻¹⁵、4.22×10⁻¹⁵、2.05×10⁻¹⁴，prior purity误差0。统计1582902使用8 CPU/16GB，进程77.138 s、allocation88 s；独立审计1582903使用1 CPU/3GB，allocation13 s；均终态4、ExitCode0。

完整统计及审计：`R1/interim/projection_controls/evaluation/{results,independent_review}.json`。紧凑结果为[r1_projection_controls.json](../roadmap/v2/r1_projection_controls.json)。英文图位于`R1/interim/projection_controls/figures/r1_projection_controls.{pdf,png}`，已目视核查各数据点及图例。

```bash
python -m analysis.flavour.v2.r1.evaluate \
  --root "$R1_ROOT/interim/projection_controls" \
  --config "$R1_ROOT/interim/projection_controls/config.json" \
  --replicates 400 --workers 8
python -m analysis.flavour.v2.r1.review \
  --root "$R1_ROOT/interim/projection_controls" \
  --config "$R1_ROOT/interim/projection_controls/config.json"
python roadmap/tools/plot_r1_projection_controls.py --root "$R1_ROOT"
```

环境为LCG108/Python3.12、PYTHONPATH=`src`、BLAS/OpenMP线程1。图表脚本只读取已完成统计，不重新扫描全事件。

## 阶段性结论

**事实：**投影ResNet在本seed全域优于HGB的平均三类质量，对B4则存在显著μ代价；tracker重训增益主要来自NC；三类重训未建立相对四类的全面优势。高质量fiducial域与全域表现有明显差异。

**研究判断：**继续保留B1作为真实三类交付候选，同时保留B4和强native参照。类别数、native细节和tracker空间精度的价值须分别评价；当前结果尚不支持按平均BA单独决定整体替换模型。

## 下一步计划

完成B3四pass T3并生成冻结完整排名；完成已确定B1的seed和其余必要扩展、所有同allocation计算控制、55条条件路径。随后完成最终400次统计、全矩阵独立审计、英文图、所有allocation终态记账与R1验收。R2训练尚未启动。
