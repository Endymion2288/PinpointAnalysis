# S5 完整验证、架构健康与方向读出结果

后续[原始范围审查](2026-11-14_stage205_s5_scope_completion_audit.md)补齐exact-target联合匹配和τ几何诊断：匹配域3,179事件/Neff2,886.875，six-way τ AP差区间跨零；首面pixel投影域的高纯度τ效率与占用/位置混淆单列。以下保留完整域主矩阵和当时资源快照；最终累计25条后端记录、CPU61.355000 core-hours及补充图见阶段205和更新后的精简JSON。

## 阶段目标

完成FASERCal式六类/物理辅助任务、观测对象及坐标方案的同预算比较；出现回归学习不足后修复并重新训练、验证。旧S3失活结果继续只作故障诊断，本报告全部分类数值来自已修复、实际更新的模型。

## 工作内容

沿用用户确认的samples、split、几何及detector-proxy，不重新追查生产认证或要求真实硬件标定。Run 10081/10082/10083/10084对应CC νe/CC νμ/CC ντ/NC。既有110k响应缓存含100k train、10k validation，本阶段监督训练固定从train选择10k标签；不将它写成100k监督训练或全7.24M事件结果。

输入是原生pixel/bar/tracker观测及可测量派生对象；truth独立连接，只用于监督、matching和条件评价。未使用true vertex/四动量、τ/charm标签或event ID作为分类输入。当前backbone为multiscale原生局部编码、64 latents及独立local读出，保留FP32 LayerNorm/SiLU、分支归一化、LayerScale、QK normalization和空响应bypass；不把局部汇聚宣称为无损压缩。

完整矩阵为15项四pass screen、2项额外四pass回归head修复、3种结构×3seed的9项八pass确认，共26个神经网络；另有3个固定HGB基线。每个神经网络四/八pass分别2,540/5,080 updates；全部累计88,900 updates及1,291次内部健康采样。十五项screen按预定validation NLL规则选择vertex+kinematics，six-way是预定独立假设，direct4为控制。确认从各seed的MAE3重新初始化，不从四pass终点接续。两项head修复是额外探索对照，没有反向改写此前selection。

论文与适配见[研究合同](../roadmap/archive/2026-11-14_v1/12_s5_physics.md)、[head修复](2026-11-14_stage196_s5_component_head_repair.md)及[方向诊断](2026-11-14_stage202_s5_direction_readout_results.md)。FASERCal用于six-way、vertex/semantics/charm/visible-system辅助任务；GradNorm/FAMO只提供任务尺度审查依据，未声称实现完整算法。固定encoder的FP64 ridge诊断使用train内family folds选择正则化，没有把已见过全部train标签的encoder称为独立CV。

## 详细数据

### 评价与统计口径

最终统一评价CPU job1582676完成29结果、33项同事件配对，实际180.511165 s。每结果使用相同10k validation，每类2,500；400次accepted physical-block bootstrap，共128 blocks、每类32。CI95条件于已训练权重、seed及反复使用的validation，不包含训练seed总体不确定度，也没有独立test覆盖解释。Training seed sample SD另列；三seed均值是指标平均，不是ensemble预测。

Common-support复用既定能量/顶点条件权重，只作条件诊断，不等于真实flux或产额权重。1%背景工作点使用7,500背景，0.1%仅约7个可接受背景事件，尾部统计不足。Validation选择阈值的Clopper–Pearson计数区间与物理块区间分别保留。Calibration/test未打开，最终冻结test属于S6。

### 三seed八pass确认

| Variant | NLL | Macro AP | Tau AP | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| direct4 | 1.237649 ± 0.002359 | 0.450829 ± 0.001577 | 0.337941 ± 0.002948 | 0.416800 ± 0.002042 |
| vertex_kinematics | 1.235693 ± 0.002768 | 0.451739 ± 0.002521 | 0.338488 ± 0.001878 | 0.416433 ± 0.001172 |
| six | 1.239364 ± 0.004950 | 0.449213 ± 0.003275 | 0.333177 ± 0.005147 | 0.415233 ± 0.003564 |

差值为variant减同seed、同八pass direct4；下表为raw总体的三seed平均差值。

| Variant | Metric | Mean delta | Event-block CI95 | Seed delta SD |
| --- | --- | ---: | --- | ---: |
| six | NLL | +0.001715 | [-0.000529, +0.003645] | 0.002675 |
| six | macro_AP | -0.001616 | [-0.003674, +0.000521] | 0.002092 |
| six | class_2_AP | -0.004764 | [-0.008558, -0.001192] | 0.005169 |
| vertex_kinematics | NLL | -0.001956 | [-0.003363, -0.000725] | 0.000827 |
| vertex_kinematics | macro_AP | +0.000910 | [-0.000381, +0.002352] | 0.002180 |
| vertex_kinematics | class_2_AP | +0.000547 | [-0.001915, +0.003037] | 0.001450 |

Vertex+kinematics的三个seed NLL差值均为负，均值−0.001956的区间支持小幅改善；macro AP与τ AP的区间均跨零。Six-way的总体NLL与macro AP没有显示可靠改善，τ AP差−0.004764的区间为负；本合同下不晋级six-way为更优分类方法。不能将本次相对direct4的微小变化解释为普遍的FASERCal方法优劣。

### 架构与真实学习

26/26均通过完整训练、实际encoder参数变化及selected-checkpoint重算；最大概率读回差9.727478e-5，低于既定0.003容差。1,291次健康采样全部通过，非空输入的SiLU零比例最大0。Local-pool RMS范围0.996118–1.019853，Head输入归一化0.999080–1.058107，隐藏LayerNorm 0.988113–1.009646，SiLU激活RMS 0.198322–0.620309。旧模型的尺度放大、整层归零与上游不更新故障没有再次出现。

真实均衡六类fit为每子类4个事件，共24个，384步后six accuracy24/24，合并四类NLL0.000152291。它证明全部输出通道能够学习这些事件，不作为泛化性能。正式six-way validation的有效子类分母9,999，另1个unknown τ保留mask；τ→e/μ先验约0.0444/0.0395。三个seed的τ→μ argmax均为0，τ→e只有2/1/0个；其概率std约0.022–0.027，AP分别约0.068–0.073/0.061–0.071，均高于相应先验。稀有通道仍弱；低先验及与prompt轻子混淆是需要继续检验的原因，没有将零argmax自动归为整层失活，也没有因此把分类负结果忽略。

### 回归修复与仍弱的方向

旧Linear+Huber的encoded px/py MSE为0.033531/0.031631，差于train-mean reference 0.028805/0.028573；在实际Huber目标上也更差。审查后改为train-only逐分量std预条件的MSE、均值初始化、非负pz/E link，并比较Linear与normalized residual MLP。两项均完成四pass新训练；16-event真实fit同时要求四分量误差下降，已通过。

MLP validation的四分量MSE为[0.028464897,0.028672704,0.029460884,0.029446014]；px约改善1.18%，pz/E约36.4%，py区间跨零。Wrapped Δφ absolute median1.385622 rad、68% quantile1.936310 rad，pT>0.1 GeV分母9,595，仍不能称可靠方位角重建。新head没有建立额外分类收益。

继续完成最佳MLP的完整10k train及10k validation固定encoder诊断。Train上px/py MSE0.027708222/0.027236771，均低于train常数0.028285827/0.027677612；validation py bias为0.010963514。FP64 ridge读出使用1,440维features，正则化仅由train内选择，py bias降至0.002822821，validation px/py MSE为0.028505082/0.028475051。相对constant的py差−0.000098174，CI95[−0.000178426,−0.000018241]，约0.34%改善。Encoder保留了少量signed方向信息，不能把原head不足解释成硬件没有信息；方向整体仍弱，后续vector/probabilistic head另立新合同。

### τ困难通道与观测对象

τ分母2,500中，electron444、muon395、hadronic1,660、unknown1；无daughter response873，有response1,627。八passvertex+kinematics seed42在1%背景阈值接受66个τ：τ→e为10/444、τ→μ为5/395、τ→had为51/1,660；无response为1/873，有response为65/1,627。各分层阈值共享7,500背景，不将子类purity与inclusive purity混淆。

同模型的τ→e AP0.074043，block CI95[0.064351,0.088073]；τ→μ AP0.057081，[0.049561,0.069488]；hadronic AP0.278881，[0.261815,0.298143]。不同域先验不同，不能直接用这些AP大小比较固有可辨性。无response域在1%背景处的效率0.001145，block CI[0,0.003509]，有response域0.039951，[0.023412,0.053843]；这是当前模型/proxy的条件描述，不是硬件信息上界。

近邻pixel segment候选在各类的purity只有约2.0%–5.4%，当前算法有严重组合fake；对象分类只能评价这份观测候选构造，不能否定对象路线。Primary双轨eligible806个，现有候选中29个匹配xy<1 mm且|dz|<10 mm，效率3.598%、block CI[2.509%,4.871%]；τ双直接prong eligible31个仅2个匹配，6.452%、CI[0,16%]，不包括单prong kink。

同容量absolute/relative坐标seed42四pass对照中，relative的macro AP差+0.004691，CI[0.002509,0.007006]，τ AP差+0.005726，CI[0.000478,0.011490]；加入kinematics后不呈现一致改善。该结果限于单seed及对应任务，未升级为普遍坐标收益。

### 全部结果

HGB使用固定160 boosting iterations、15 leaves，与神经网络passes分列。下表保留全部29结果，精确工作点及条件比较见EOS完整summary；修复和确认两个cohort按其控制分别评价。

| Experiment | Passes / kind | NLL | Macro AP | Tau AP | Tau efficiency at 1% background |
| --- | --- | ---: | ---: | ---: | ---: |
| s5_direct4_n10000_s42 | 4 | 1.240712 | 0.447629 | 0.336007 | 0.0240 |
| s5_six_n10000_s42 | 4 | 1.237479 | 0.450024 | 0.334041 | 0.0288 |
| s5_vertex_n10000_s42 | 4 | 1.240485 | 0.447286 | 0.335054 | 0.0288 |
| s5_semantics_n10000_s42 | 4 | 1.237786 | 0.451111 | 0.336566 | 0.0304 |
| s5_charm_n10000_s42_mem1 | 4 | 1.242678 | 0.446428 | 0.333165 | 0.0228 |
| s5_kinematics_n10000_s42_mem1 | 4 | 1.240855 | 0.448339 | 0.338146 | 0.0268 |
| s5_combined_n10000_s42_mem1 | 4 | 1.238886 | 0.447774 | 0.334439 | 0.0296 |
| s5_uncertainty_n10000_s42_mem1 | 4 | 1.239353 | 0.449313 | 0.336473 | 0.0316 |
| s5_objects_n10000_s42_mem1 | 4 | 1.239773 | 0.445544 | 0.333556 | 0.0224 |
| s5_vertex_kinematics_n10000_s42_mem1 | 4 | 1.236937 | 0.451093 | 0.335515 | 0.0276 |
| s5_relative_v2_n10000_s42_mem1 | 4 | 1.238426 | 0.451146 | 0.340151 | 0.0272 |
| s5_relative_objects_v2_n10000_s42_mem1 | 4 | 1.240626 | 0.447465 | 0.338019 | 0.0244 |
| s5_kinematics_relative_v2_n10000_s42_mem1 | 4 | 1.241505 | 0.447914 | 0.338098 | 0.0268 |
| s5_absolute_coordinates_v2_n10000_s42_mem1 | 4 | 1.242917 | 0.446455 | 0.334425 | 0.0252 |
| s5_kinematics_absolute_v2_n10000_s42_mem1 | 4 | 1.241389 | 0.446537 | 0.334569 | 0.0228 |
| s5_kinematics_standard_linear_n10000_s42 | 4 | 1.237917 | 0.450042 | 0.336602 | 0.0264 |
| s5_kinematics_standard_mlp_n10000_s42 | 4 | 1.239434 | 0.449262 | 0.339917 | 0.0240 |
| s5_confirm8_direct4_n10000_s17 | 8 | 1.238230 | 0.449487 | 0.334844 | 0.0300 |
| s5_confirm8_direct4_n10000_s42 | 8 | 1.235054 | 0.452566 | 0.338268 | 0.0228 |
| s5_confirm8_direct4_n10000_s2026 | 8 | 1.239664 | 0.450434 | 0.340713 | 0.0340 |
| s5_confirm8_vertex_kinematics_n10000_s17 | 8 | 1.235478 | 0.452323 | 0.336320 | 0.0276 |
| s5_confirm8_vertex_kinematics_n10000_s42 | 8 | 1.233038 | 0.453917 | 0.339555 | 0.0264 |
| s5_confirm8_vertex_kinematics_n10000_s2026 | 8 | 1.238563 | 0.448977 | 0.339589 | 0.0312 |
| s5_confirm8_six_n10000_s17 | 8 | 1.239470 | 0.448153 | 0.329300 | 0.0276 |
| s5_confirm8_six_n10000_s42 | 8 | 1.234363 | 0.452887 | 0.339017 | 0.0280 |
| s5_confirm8_six_n10000_s2026 | 8 | 1.244260 | 0.446600 | 0.331214 | 0.0312 |
| s5_hgb_moments_n10000_s42 | HGB | 1.253049 | 0.436265 | 0.323559 | 0.0264 |
| s5_hgb_moments_objects_n10000_s42 | HGB | 1.247476 | 0.440891 | 0.325977 | 0.0216 |
| s5_hgb_object_only_n10000_s42 | HGB | 1.338693 | 0.347610 | 0.275755 | 0.0124 |

### 产物、资源与复现

产物根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14`。`evaluation/final/`包含13份英文PDF、相应PNG、完整summary、metrics/kinematic/native-example CSV；`experiments/*/best.pt`、完整validation predictions、health/history/readback、HGB模型、方向ridge系数及所有后端证据保留。人工检查三seed图、分量图、方向train/readout图；方向图刻度重叠已单独修正，没有重算指标。

全部24条后端allocation/提交记录已终止，无遗留训练或评价作业。Remote device allocation为2.673889 device-hours，CPU allocation为61.256111 core-hours；四次交互T4真实fit另计232.280006 s，未记录交互allocation边界，不能把fit时长当完整allocation。MIG47GB、MIG24GB及不同设备时长分别保留，device-hours不是等价整卡吞吐或货币费用。Kubeflow六并发的主存实测峰值46.542295 GB、无OOM；MIG利用率接口为N/A，未捏造GPU利用率百分比。

CPU清理job1582679合并或删除471份重复/中间文件，释放1,134,394,836 bytes（1.134 GB），EOS本阶段最终约237 MiB。AFS清理38个小型提交文件、32,256 bytes，`~/condor/Pinpoint/flavour_s5/`为空；准确脚本已保存在EOS提交证据。26个selected checkpoints、3个HGB模型、完整预测及13份最终PDF全部保留。`acceptance.json`的scientific_complete与stage_complete均为true；仓库精简结果为[JSON](../roadmap/results/s5_summary.json)与[CSV](../roadmap/results/s5_metrics.csv)。

可复现环境为LCG108_cuda/Torch2.7.1，输入全部沿用已接受缓存。S5源码快照及准确提交命令在EOS `code/`与`runtime/*.submission.json`；HTCondor提交按可用schedd的idle/held/running和EOS/WRITE兼容性即时选择。本轮GPU使用Kubeflow H100 MIG47GB与HTCondor H100 MIG24GB，按claims分工；整卡H100未获调度的尝试取消，没有计作训练速度。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 在CPU批作业中执行统一评价；模型必须已完成。
python -m analysis.flavour.s5.evaluate --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14 \
  --campaign roadmap/configs/s5_final_matrix.json --tag final --bootstrap 400 --workers 4
# 读取已有完整证据，不启动新训练或打开test。
python -m analysis.flavour.s5.finalize --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14 \
  --campaign roadmap/configs/s5_final_matrix.json --require-closure
```

本轮新增/相关unittest10项全部通过（3.821 s），包含真实缺失标签权重分母、已知正负方向外推、head梯度/符号、masked坐标、六类合并及truth/object核心约束。正式29结果及真实checkpoint复算是主要数据验证，不能以单元测试替代。

## 阶段性结论

修复后的架构保持有效学习，旧S3故障结果未回流为科学结论。FASERCal式vertex+kinematics辅助带来小幅NLL改善，未建立总体AP或τ AP净收益；six-way在本预算下的τ AP下降，保留direct4作为直接分类控制。修复后的px/pz/E及收敛py读出分别有程度不同的constant-relative改善，但方位角仍弱。对象假关联与不可见τ响应是需要继续分析的具体问题，没有硬件不可辨性的结论。

## 下一步计划

S6先冻结候选、概率校准与阈值规则，再使用既有独立test；报告总体/τ子道/空响应/接受度及预定proxy扰动下的效率与概率质量。未测量：独立test、generator变体包络、真实flux加权产额及真实硬件方向分辨率。继续使用用户接受的detector-proxy基值，当前完成性不依赖真实硬件标定。若另做新方向head或更好的轨迹实例关联，应作为新实验合同，保留当前负结果和同预算控制。
