# S3 架构修复、20 模型重训与完整评价

## 阶段目标

确认旧 S3 的内部学习故障，修复尺度控制和过早汇聚，重新训练并评价四分类。旧 25 模型的方法结论撤回，仅保留故障和执行证据。本轮 20 模型、80 epochs、280 项 checkpoint 读回、8,020 次计划健康采样、14 个单 seed 配对、9 个三 seed 配对及 7 个条件域全部完成；`final_acceptance.json` 的 18 项检查全部通过。

修复解决了旧 Head 全零和上游零梯度。Multiscale+tracker 的三 seed macro AP 为 0.46338±0.00189，相对不加 tracker 增加 0.02496，配对物理块 CI95=[0.01984,0.02999]。同时，global latent 路径的依赖仍弱，ντ 收益没有得到区间支持。整体有效学习不代表每个分支均被充分利用。

## 工作内容

### 已确认故障与修复依据

旧主模型 best checkpoint 的 8 个非空 validation 事件 CPU FP32 诊断：local pool RMS 约 101–129，Head 第一层 Linear 输出范围 [−79.890541,−13.976823]，GELU 输出零比例 1.0，input/local/module/latent 分类梯度平方范数全为 0。参数 finite、零 AMP skips 和完整训练均未排除故障。GELU 数学上并非在所有 x<−5 处恒为零；这里是具体 checkpoint、数值范围和实现精度下的实测。

逐层审查发现 local projection、embedding 相加、残差、context 重复注入及最终 concat 缺少完整尺度约束；缺失模态也接收同一 context。旧结构还存在固定 module queries 和 latent 数量的压缩边界。旧模型同时有 module pool 路径，因此不能把它描述成唯一的 24-token 路径。另检查过采样器：20,016 个旧 batch 中单类占 1.9634%，最长主导类别连续 9 batch，未证实长时间单类流是故障原因。

修复实现于 [models_repair.py](../src/analysis/flavour/s3/models_repair.py)，包括：

- local projection、各模态/尺度分支、concat 和 Head 隐藏层分别 LayerNorm；缺失分支归零并显式 presence，移除重复 context 注入。
- attention/FF 使用 Peri-LN、LayerScale 初值 0.1、每 head 无 affine 的 Q/K LayerNorm、std 0.02 初始化和 SwiGLU，约束残差增量与 attention 投影尺度。
- Head 为 FP32 Linear→LayerNorm→SiLU，另有独立 linear logits 路径；encoder 使用 BF16。修复效果不能归于单独替换 GELU。
- 全部节点先形成每块至多 32 节点的 local blocks，按占用增长的 regions 产生区域 tokens；64 global latents 跨 region 读取。全部 local blocks 另有独立的各模态 mean/max 读出，减少对固定少量 queries 的依赖。

联网检索及正式链接记录在 [修复路线图](../roadmap/10_s3_repair.md)：Peri-LN（ICML 2025）、LayerScale（ICCV 2021）、QK normalization（ICML 2023）、PTv3（CVPR 2024），结合 FASERCal/Perceiver 的 module-aware 融合。补读 ICML 2021 的 rank 分析、ICML 2023 的 attention entropy collapse、NeurIPS 2021 的 Gradient Starvation 和 CVPR 2022/TPAMI 2024 的 OPM/OGM，用于检查分支同质化与优化竞争。当前矩阵未运行 branch dropout、gradient modulation 或额外预训练；论文提供可检验依据，不直接构成本模型的病因诊断。

### 输入、冻结训练和实际执行

沿用用户接受的 samples、split、geometry 和 detector-proxy。输入来自 S2 原生观测缓存；原始 run 为 10081/10082/10083/10084，类别顺序为 CC νe、CC νμ、CC ντ、NC。每模型使用 80,000 train、10,000 validation（每类 2,500），包含空响应。Calibration/test 未使用，S4–S6 未执行。

冻结配置为 [8 个 screen](../roadmap/configs/s3_repair_screen.json) 和 [12 个确认训练](../roadmap/configs/s3_repair_confirmation.json)。Fixed native、multiscale native/1 mm、tracker、region8 和 fresh ParT 各有 seeds 17/42/2026；fixed 1 mm 与 no-direct 为 seed42 控制。候选在完整 validation 结果产生前冻结。

所有模型均为 4 epochs、batch8、每轮 10,016 updates，共 40,064 applied updates；AdamW lr=0.0003、betas=(0.9,0.95)、eps=1e−8，norm/bias 不做 weight decay，500-step warmup 后逐 update cosine 至 1e−5，clip=1。Best 按完整 validation NLL 选取，各条件域复用同一 checkpoint。19 个模型 best epoch4，ParT seed17 为 epoch3；全部仍完成四轮。实际成功训练曝光分别为 seed42 的 320,288、seed17 的 320,304、seed2026 的 320,320 events/model，未将采样器末尾曝光误写成正好 320,000。

正式训练前，8 个配置分别通过 32 个真实 train 事件、每类 8 个、768 updates 的拟合：accuracy 全为 1.0、32 组不同分数，末次 probe NLL 范围 0–0.000282117457。每配置另通过包含空/中/最大占用的完整前向和反向压力测试。它们证明可拟合这些事件，不是泛化性能。

统一采用 LCG108/Torch2.7.1、local execution version2、`torch.compile(dynamic=True)` 和 chunk2048；chunk 仅分批执行 local blocks，不改变物理窗口、region 或节点覆盖。健康 hooks 位于编译区外，采样与普通更新使用同一计算路径。编译先导 version1 中的执行路径切换已修复，失败先导单独归档计费，未用于正式结果。

实际训练同时使用 Kubeflow H100 NVL MIG 3g.47gb、Condor 整张 H100 和 A10040GB，以原子 claim 保证唯一 writer，同卡并发。Seed17 多在 H100、seed42 多在 MIG、seed2026 多在 A100，部分任务跨设备恢复。因此下述 training seed sample SD 同时包含这些已记录的执行差异，未单独识别硬件效应。AFS 仅放提交脚本；本阶段结束时 `~/condor/Pinpoint/flavour_s3/` 已清空，产物均在 EOS。

14:17–14:31 UTC 的 SSH 接续未持续运行，主动停止原 screen child 又导致旧 Kubeflow campaign 退出并中断两个确认任务。这是调度失误，详见 [阶段86](2026-11-13_stage86_s3_incremental_completed_model_checks.md)。后续改用正式 batch 入口，从已提交 epoch 恢复模型/optimizer/scheduler/RNG，重算未提交 updates；全部尝试的费用保留。原 Pod 已被清理，保存实际 Job Failed 时间作为 allocation 上界，未观测的 Pod phase/exit code 保持 null。最终 20 模型均完成，不能把中途启动或中断写成成功终态。

可复现入口如下。训练命令由已获配 GPU 的 batch wrapper 执行；全矩阵使用相同入口和冻结 task 索引。运行环境的 scratch Triton/compiler 准备见 `runtime_repair.sh`，无需 Poetry 重新安装与实际训练不匹配的旧 Torch 声明。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S3_REPAIR_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13
# 单 task 入口示例；完整矩阵的领取与并发由 parallel/campaign_repair 管理。
python -m analysis.flavour.s3.train_repair --root "$S3_REPAIR_ROOT" \
  --campaign roadmap/configs/s3_repair_screen.json --task 2
python -m analysis.flavour.s3.finalize_repair --root "$S3_REPAIR_ROOT" \
  --campaign roadmap/configs/s3_repair_screen.json \
  --confirmation roadmap/configs/s3_repair_confirmation.json \
  --comparisons roadmap/configs/s3_repair_confirmation_pairs.json --wait
python -m analysis.flavour.s3.summarize_repair --root "$S3_REPAIR_ROOT" \
  --output roadmap/results/s3_repair_summary.json
```

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13`。可机读完整数据在 [s3_repair_summary.json](../roadmap/results/s3_repair_summary.json)，逐模型、逐配对及工作点表在 [measured_tables_zh.md](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13/evaluation_final/measured_tables_zh.md)。下面数值均为实测。

### 统计口径

四类使用非插值 one-vs-rest AP 并保留精确 ties；NLL 为 FP64 归一化概率的交叉熵，Brier 为四类平方误差之和，precision/recall/F1/混淆矩阵来自 argmax。所有预定配对均有 34 项统计，使用 400 次、128 个 accepted family/chunk 物理块的相同重采样。本文 CI 为 pointwise percentile 区间，条件于已拟合 seeds 和 validation checkpoint 选择；未做多重比较修正，也未把 training seed SD 合入物理事件区间。三 seed 平均是三个指标的均值，不是 ensemble 预测。

Common-support 复用 accepted S2 粗 E/vertex 权重：8,559 个正权重事件、103 个支持 cells、每类权重和 1,586、Neff=7,959.4136465；未匹配 target Z/同位素，不是 flux/产额权重。全空域每类计数 [56,62,90,74]，不能以不同 prior 下的 AP 解释纯拓扑能力。

### 三 seed 四分类与逐类 AP

数值为均值±training seed sample SD。

| Model | Macro AP | NLL | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Stable fixed native | 0.42636±0.00074 | 1.25755±0.00197 | 0.39008±0.00102 | 0.39180±0.00114 |
| Multiscale 1 mm | 0.43984±0.00052 | 1.24386±0.00170 | 0.39902±0.00142 | 0.40133±0.00163 |
| Multiscale native | 0.43843±0.00123 | 1.24382±0.00131 | 0.39866±0.00209 | 0.40020±0.00180 |
| Multiscale region8 | 0.43722±0.00177 | 1.24434±0.00253 | 0.39638±0.00256 | 0.39827±0.00337 |
| Multiscale + tracker | 0.46338±0.00189 | 1.22356±0.00394 | 0.42882±0.00277 | 0.42830±0.00276 |
| Fresh ParT | 0.45434±0.00398 | 1.22594±0.00427 | 0.41161±0.00712 | 0.41320±0.00666 |

| Model | AP CC νe | AP CC νμ | AP CC ντ | AP NC |
|---|---:|---:|---:|---:|
| Stable fixed native | 0.57088±0.00219 | 0.41083±0.00168 | 0.33573±0.00309 | 0.38802±0.00262 |
| Multiscale 1 mm | 0.59366±0.00119 | 0.42379±0.00066 | 0.34726±0.00178 | 0.39464±0.00061 |
| Multiscale native | 0.59122±0.00222 | 0.42206±0.00116 | 0.34436±0.00269 | 0.39607±0.00119 |
| Multiscale region8 | 0.58958±0.00126 | 0.41922±0.00214 | 0.34356±0.00245 | 0.39651±0.00307 |
| Multiscale + tracker | 0.60567±0.00384 | 0.46024±0.00079 | 0.34810±0.00112 | 0.43951±0.00253 |
| Fresh ParT | 0.60542±0.00196 | 0.44443±0.01007 | 0.35136±0.00214 | 0.41614±0.00678 |

修复模型 dim96；fixed 为 967,888 参数，multiscale/region8/tracker 为 897,992，no-direct 为 782,792。ParT dim64、264,284 参数，使用独立 4 mm local blocks 和全 block pair attention。相同训练配方/update 数没有消除容量、窗口和计算量差异。

### 预定配对结果

下表差值均为 B−A，区间为物理块 CI95。每项三 seed 比较同时保留 raw 和 common-support。

| B minus A | Domain | Δ macro AP [CI95] | Δ NLL [CI95] |
|---|---|---:|---:|
| Fixed native − ParT | raw | −0.02797 [−0.03320,−0.02291] | +0.03161 [+0.02645,+0.03756] |
| Fixed native − ParT | common | −0.02686 [−0.03188,−0.02155] | +0.02860 [+0.02246,+0.03472] |
| Multiscale native − ParT | raw | −0.01591 [−0.02002,−0.01143] | +0.01788 [+0.01377,+0.02236] |
| Multiscale native − ParT | common | −0.01610 [−0.02031,−0.01131] | +0.01722 [+0.01251,+0.02215] |
| Multiscale 1 mm − ParT | raw | −0.01450 [−0.01915,−0.00988] | +0.01792 [+0.01393,+0.02214] |
| Multiscale 1 mm − ParT | common | −0.01497 [−0.01993,−0.01051] | +0.01815 [+0.01348,+0.02315] |
| Multiscale tracker − ParT | raw | +0.00904 [+0.00415,+0.01424] | −0.00238 [−0.00804,+0.00327] |
| Multiscale tracker − ParT | common | +0.01465 [+0.00819,+0.02091] | −0.00484 [−0.01140,+0.00173] |
| Region8 − ParT | raw | −0.01712 [−0.02118,−0.01255] | +0.01840 [+0.01441,+0.02287] |
| Region8 − ParT | common | −0.01637 [−0.02111,−0.01178] | +0.01588 [+0.01103,+0.02079] |
| Multiscale native − 1 mm | raw | −0.00141 [−0.00329,+0.00024] | −0.00004 [−0.00185,+0.00174] |
| Multiscale native − 1 mm | common | −0.00113 [−0.00299,+0.00102] | −0.00093 [−0.00284,+0.00093] |
| Multiscale native − fixed native | raw | +0.01206 [+0.00814,+0.01571] | −0.01373 [−0.01855,−0.00904] |
| Multiscale native − fixed native | common | +0.01075 [+0.00628,+0.01448] | −0.01137 [−0.01614,−0.00622] |
| Tracker − no tracker | raw | +0.02496 [+0.01984,+0.02999] | −0.02027 [−0.02533,−0.01568] |
| Tracker − no tracker | common | +0.03076 [+0.02477,+0.03769] | −0.02207 [−0.02800,−0.01648] |
| Region8 − region32 | raw | −0.00121 [−0.00293,+0.00044] | +0.00052 [−0.00158,+0.00274] |
| Region8 − region32 | common | −0.00026 [−0.00220,+0.00162] | −0.00135 [−0.00374,+0.00119] |

Seed42 的 native direct−no-direct 控制：Δ macro AP=+0.01366 [0.00957,0.01828]，Δ NLL=−0.01476 [−0.01964,−0.01028]；common-support 分别为 +0.01117 [0.00650,0.01584] 和 −0.01328 [−0.01794,−0.00869]。这支持该配置下独立 local 读出的价值，尚无三 seed 确认。Fixed→multiscale 同时改变 region/queries、direct 路径和 context 计算，不能把其净收益归于单一组件。

### Tracker 的物理分层与工作点

Tracker−no-tracker 三 seed raw 逐类 AP 增量：νe +0.01446 [0.00857,0.02008]；νμ +0.03818 [0.02640,0.05098]；ντ +0.00374 [−0.00277,0.00987]；NC +0.04344 [0.03457,0.05178]。νμ/NC 是主要收益，尚不能声称提高了 ντ 判别。

| Domain | Events | Δ macro AP [CI95] | Δ NLL [CI95] |
|---|---:|---:|---:|
| Pixel-centered | 885 | +0.03311 [+0.01490,+0.04996] | −0.05683 [−0.08007,−0.03571] |
| Large scintillator | 4,494 | +0.02789 [+0.01976,+0.03583] | −0.03104 [−0.04078,−0.02233] |
| Detector active | 9,718 | +0.02517 [+0.01994,+0.03012] | −0.02075 [−0.02595,−0.01588] |
| All empty | 282 | 0 [0,0] | −0.00360 [−0.00894,+0.00222] |
| Pixel ≥1,000 | 1,201 | +0.01667 [−0.00036,+0.03426] | −0.01911 [−0.03596,−0.00368] |
| Tracker active | 5,630 | +0.02383 [+0.01753,+0.03079] | −0.02159 [−0.02840,−0.01463] |
| Tracker empty | 4,370 | +0.00637 [+0.00048,+0.01211] | −0.01856 [−0.02539,−0.01150] |

Tracker-empty 域没有新的 tracker hit 内容，两个独立训练的差异可来自 presence、全局拟合及其他分支；不解释为该域测到了 tracker 空间信息。另有 158 个 tracker-only response 事件的描述性分层。全空输入的常数分数符合可观测信息范围，不按学习失败处理。

全 validation 在 1% FPR 的三 seed 效率（均值±sample SD）如下，每类分母固定为 2,500 signal / 7,500 background：

| Model | CC νe | CC νμ | CC ντ | NC |
|---|---:|---:|---:|---:|
| Multiscale native | 0.17547±0.00428 | 0.07173±0.00335 | 0.02920±0.00212 | 0.06507±0.00401 |
| Multiscale + tracker | 0.18813±0.00284 | 0.08987±0.00266 | 0.03053±0.00522 | 0.06960±0.00694 |
| Fresh ParT | 0.18653±0.00284 | 0.08507±0.00987 | 0.03160±0.00080 | 0.07800±0.00904 |

工作点使用完整 ties 的经验阈值，未插值出未被样本支持的尾部。0.1% FPR 最多接受 7 个背景事件，未达到 25 个背景的尾部统计标记；ντ seed42 的 80%/90% purity 点均只选到 1 signal、0 background，不能作为可靠高纯度性能。所有阈值来自 validation；purity 基于 benchmark 先验，未换算物理 flux/产额。冻结 test 效率尚未测量。

### 激活、梯度和 global 路径

20/20 模型各有 401 次健康采样，共 8,020 次。全部计划记录存在、激活 finite，全部 informative 样本的聚合 encoder 梯度非零；每模型另有 best checkpoint 的 8 个非空 validation 事件独立 CPU FP32 backward。20 个 NLL−constant-prior 的物理块 CI95 上界均小于 0。17 个修复结构的 Head SiLU 采样零比例最大值均为 0；ParT 有独立 Head 定义，不能按相同层号解释为 SiLU。

Native seed42 的 local pool RMS 为 0.99982–1.16338，Head 输入归一化 RMS 为 0.86143–0.99999，第一 Linear RMS 为 0.75968–6.26229，随后 LN 为 0.99883–1.09343。第一 Linear 仍有较大负值，实际最小 −39.27676；隐藏层 LN 和激活后的梯度维持连通。因此结论是修复了整层失活，不是所有层、所有分量都单位尺度。

17 个模型另执行 64-event occupancy-quantile 非空 validation 路径诊断。只在相同 presence 模式内置换，Head 缓存重建一致；不是总体随机抽样、重训消融或因果信息量。下面展示 seed42：

| Model | Δ probe NLL: regional | local | global | summary | Global between-event RMS |
|---|---:|---:|---:|---:|---:|
| Fixed native | +0.15065 | absent | 0.00000 | +0.59933 | 0.000113 |
| Multiscale native | +0.19285 | +0.12918 | +0.00457 | +0.09448 | 0.137548 |
| Multiscale + tracker | +0.03942 | +0.13554 | −0.00160 | +0.15121 | 0.010755 |

Native seed42 的 latent-read/global-block 梯度平方范数中位数为 0.0285945/0.00131088；tracker seed42 为 0.0000894599/0.00000570837，informative 样本中两路径精确零计数均为 0。梯度非零，但 tracker 对 global 置换的 prediction RMS 仅 0.002083、argmax 变化为 0，显示这条路径使用较弱。

探针记录组的 uncentered squared-spectrum entropy rank：native seed42 的 regional/latent/global-norm 为 1.03539/2.23141/1.05535，tracker 为 1.00490/1.01581/1.00112；对应每组 queries 为 2/64/64。这不是整个事件表示的 rank，也不说明不同 regions 相同；它与低事件变化及弱置换影响共同提示 query 同质化和 global 弱利用。没有把这些路径宣布为已充分学习，也没有据低 rank 单独宣布整个模型失败。

待区分的解释是 query 对称性/同质化、local/summary 的优化竞争、以及 global 信息冗余。当前未测量独立 global probe、无 global 重训或 OPM 对照，不能确定是哪一种原因。

### Token 覆盖与压缩

同一个最大占用 train 事件，全硬件共 36,549 节点，其中 pixel+bar 为 36,547：

| Representation | Active nodes | Local blocks | Regional tokens | Global tokens | Direct local blocks |
|---|---:|---:|---:|---:|---:|
| Fixed native | 36,547 | 14,597 | absent | 24 | absent |
| Multiscale native | 36,547 | 14,597 | 958 | 64 | 14,597 |
| Multiscale 1 mm | 8,417 | 6,169 | 430 | 64 | 6,169 |
| Multiscale + tracker | 36,549 | 14,599 | 962 | 64 | 14,599 |
| Multiscale region8 | 36,547 | 14,597 | 3,690 | 64 | 14,597 |
| Fresh ParT | 36,547 | 2,641 | absent | 2,641 | absent |

Fixed 的 31 个实际 module/type groups 产生 124 个 module tokens。Multiscale 的 regional tokens 随占用增长，local 直接读出覆盖全部 blocks，避免将所有分类信息强制通过 24 tokens。池化仍非可逆，不能称无损。固定 32 槽 padding 的 native 有效占用仅 7.8242%，1 mm 为 4.2638%，ParT 为 43.2447%；改善稀疏执行有计算空间。Census 为相同 min/median/max train occupancy 事件×20 配置，共 60 行，不是全体 transformed token 分位分布。

### 图表、验收和费用

英文产物位于 `evaluation_final/`：20 页 `four_class_diagnostics.pdf`、20 页 `internal_training_health.pdf`、17 页 `fitted_path_diagnostics.pdf`，以及模型总览、raw/common-support 配对和逐类 AP 区间、资源图、条件曲线和错误表。原生事件显示为 native 19 页、tracker 20 页；按每个 confusion pair 的首个高置信事件和四个固定随机参考自动选择，未手选有利事件。Truth E/vertex 仅作图注，不是模型输入。

人工复核了最终模型总览、配对/逐类图，fixed/native/1 mm/no-direct/region8/tracker/ParT 的四分类页，native2026/1mm17/ParT2026 健康页、native42/2026 路径页和两套事件显示首页。总览第三面板长标题原有裁切，已缩短并重新渲染。数值证据与人工复核范围分别保存在 summary 和 `audit/final_visual_review.json`，未声称逐页检查所有 PDF。

S3 targeted suite：`python -m unittest discover -s tests -p 'test_flavour_s3*.py'`，49 tests、7.671 s、通过。完整真实输出 `accept_repair --require-complete` 通过 18 项门槛，包含冻结矩阵、内部学习、全部 34 项统计/配对、条件域、终态和预算。`compileall` 与 `git diff --check` 通过。最终状态计数依据 20 个 complete manifests 校正，避免运行中的旧 finalizer 留下 19/20 的旧计数。

全部 allocation 已结束，missing history=0。新 Kubeflow 恢复 Job/Pod 为 Succeeded，原 H100 Condor 作业正常 exit0；A100 在自身全部 writers 结束后主动释放，相关 Condor 队列为空。失败/迁移/编译/空转费用均保留：

| Scope | Device-hours upper bound |
|---|---:|
| Repair campaign | 10.845833333 |
| Legacy S3 | 31.238268678 |
| Cumulative S3 | 42.084102011 |
| Remaining under 120 | ≥77.915897989 |

整卡与 MIG device-hours 是 allocation 计费口径，不能当吞吐等效量。13:50 UTC 的 21 秒 NVML 样本中，共享三 fit 的 H100 利用率均值 80%、A100 42.14%，设备显存分别 16,889/18,596 MiB；MIG 利用率未测量，不外推全程 busy time。混合设备、并发和恢复费用未用于模型本体效率晋级。

## 阶段性结论

**事实**：旧 Head/encoder 故障已复现并修复，完整重训显示有效学习。独立 local 读出和 multiscale 配置组合的收益有对应控制证据；native 相对 1 mm、region8 相对 region32 在本轮三 seed 平均中未显示明确收益，不能据此判定原生细节无效或已经达到信息上限。

**事实**：multiscale+tracker 相对 ParT 的 raw macro F1 增加 0.01721 [0.01052,0.02439]、balanced accuracy 增加 0.01510 [0.00847,0.02222]；NLL 增量上界 0.00327，小于冻结非劣界 0.01，各类 AP 点退化均未超过 0.01。它通过本轮预定质量晋级条件。Raw macro AP 增量 0.00904 的点值本身未达到 0.01，晋级依据是 F1/BA，而不是单看 AP。Common-support 方向一致。

**解释范围**：tracker 候选比 ParT 多用 tracker 输入，参数和窗口也不同；晋级说明该额外硬件输入配置在本预算/validation 上有效，不能证明分层 backbone 在相同输入下优于 ParT。Pixel+bar-only 条件下 fresh ParT 仍更强；建议保留它作为同输入基线，同时保留 multiscale+tracker 作为多探测器候选。

**尚未证实**：global latent 仍弱利用，ντ AP 与固定低背景工作点没有明确改善，未建立全局 attention 的独立收益。三个 seeds 的结果含设备执行差异。Calibration、冻结 test、generator/detector 系统效应及物理产额均未测量；本轮结论是有限训练预算下的 validation 结果。

## 下一步计划

1. 先在 train 内划分 probe-fit/probe-validation，比较冻结 local/regional/global/summary 的独立线性和小 MLP probes；固定 probe 容量、曝光与归一化，量化各路径可读信息和完整模型实际梯度的差异。不得用 probe-validation 结果声称最终 test 性能。
2. 冻结无 global 重训和现有模型的相同输入/预算/三 seed 对照，再分别检验 query 几何/多样性与 branch dropout/gradient modulation。只改变一个设计因素；如果弱分支可读却不被完整模型使用，才进一步检验优化竞争。OPM/OGM 来自模态间控制，共享 encoder 的尺度路径需另定义梯度作用范围和训练缺失模式。
3. 对 ντ 继续按可观测衰变/接受度代理与混淆类别分析，保留全部事件分母；先证明收益来自可读拓扑，再决定预训练或对象辅助任务。当前尚未运行这些新实验。
4. Ragged/sparse local 执行只作为等价计算优化，先验证节点/窗口覆盖、输出和梯度，再实测同设备吞吐与显存。正式 test 按原 S6 协议在模型、校准和阈值冻结后执行。

后续任一模型出现常数预测、激活饱和、上游零/极小梯度或明显学习失败，重新审查架构、尺度、采样、精度和训练设置，保存故障证据并修复重训；完成 epoch 或 checkpoint 读回不能代替有效学习验证。
