# R1：共同任务接口、强基线与真实优化路径检查

**阶段目标**

把 R1 的 B0–B4 与 T3/T4 比较落实为可执行代码，先回答“各模型是否能学习真实观测、缺失模态是否有正确数值路径”。R1 的泛化比较仍在进行，本报告不宣布 R1 完成，不给出新架构排名或三类/四类的物理取舍。

本阶段输出目录为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`，下文路径均相对此目录。沿用 R0 frozen Train100k/Tune40k；未读取 Cal-A、Cut-B、Confirm payload。

**工作内容**

新增 `analysis.flavour.v2.r1`：统一任务和 loss、完整数据转换、共同 observations dataset、tracker set late fusion、1 mm 双投影 CNN、训练/推理、HGB、paired evaluation、资源记账与 campaign 入口。训练与评价 runner 的完整大样本集成验收尚未完成，后续报告记录其实际运行与结果。

T3 原标签 `[0,1,3]` 映射为三个输出，训练时显式排除真 τ；Tune 推理保留四类物理事件，从而可以量化 τ 被强制分配到 e/μ/NC 的污染。T3T 使用一个 sigmoid gate 和 conditional softmax；BCE 对全部事件、conditional CE 对非 τ 事件求和，再统一除以事件数。T4D 的未知 decay 只 mask 分支 loss，保留 CC τ 主监督。

所有 encoder 只读 accepted observation whitelist。truth vertex/energy、τ branch、family 与 run/chunk/row 进入 provenance、loss 或事后诊断，不进入输入 tensor。B0 在既有 pixel/bar summaries 上增加 tracker 的 presence、数量、station 数量与 xy moments；B1–B3 使用同一 station/xy/sigma set branch，缺失 mask 显式传入。B2 采用既有 fixed geometry ParticleNet adaptation，B3 采用既有 geometry ParT adaptation，pixel/bar 保留原生通道并使用既有 4 mm local grouping；B4 复用 revision 3 multiscale encoder。

B1 为合作组式同构对照，不是原研究权重的严格复现。pixel 保留两个投影的 count；bar 在可测的 XZ 或 YZ 投影放置 count、edep 与 PE，不把未测坐标定位成点。稀疏 index/weight 在 GPU 上 scatter 为图像，避免长期保存大量 dense image。CNN 使用共享双视图 residual encoder、GroupNorm 与 tracker late fusion。

数据准备在 frozen 100k/40k 内按完整 selected ROOT chunk 分 shard，避免同一个 partition 内为了不同 shard 重复打开同一源文件。Train 与 Tune 可能需要分别访问同一 ROOT 文件，不声称跨 partition 只读一次。Tune source 文件稀疏，转换仍受 ROOT/EOS I/O 限制。

用户本轮明确要求优先减少排队、使用不同型号及 GPU 分区，调度已据此更新：Condor 不再限定整卡 H100 或同一型号；老卡自动使用 FP32，支持的卡使用 BF16。Kubeflow `ml-prod/xcheng` 登录有效，复用既有 Alma9/CVMFS/EOS 环境，已实际执行 H100 MIG 3g.47gb 任务。旧 S2 的整卡合同保留为归档历史，不应用于新 R1。每族的 T3/T4 与学习率候选尽量放入同一 allocation；不同设备的小时分别记账，不当作 FLOP 等效计算量。

**详细数据**

准备计划实际生成 **140,000 events / 56 shards**：Train100k 40 shards，Tune40k 16 shards。源文件数为 Train 170、Tune 6,845；完整转换与最终身份/PE/计数验收将在后续报告给出，本报告不把已提交任务视为转换成功。

B1 的实际图像原点为 `(-133,-222,-2996) mm`，纵向 1,322 bins、横向 420 bins，pitch=1 mm。两视图各四个通道，使用物理 z 距离，没有以 plane ordinal 替代毫米距离。模型 stem/下采样属于网络内部计算，输入像素仍为 1 mm。

初始 FP32 拟合使用 R0 的固定真实 Train 32-event 清单，各类八个；T3 使用其中 24 个非 τ 事件。随机种子 42，AdamW LR=0.0005、weight decay=0、batch8、clip norm1，设备 Tesla T4。下表是实际最终读回，不是泛化估计。

| Model | Task | Parameters | Updates | NLL | Accuracy | Peak allocated MiB | Fit seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 projection CNN + tracker | T3 | 793,545 | 64 | 0.0000406544 | 100% | 1,212.215 | 14.353 |
| B1 | T4 | 793,674 | 64 | 0.000140413 | 100% | 1,213.157 | 5.901 |
| B2 fixed ParticleNet + tracker | T3 | 199,753 | 96 | 0.00214786 | 100% | 1,687.668 | 85.875 |
| B2 | T4 | 199,882 | 128 | 0.0104853 | 100% | 1,627.424 | 116.806 |
| B3 geometry ParT + tracker | T3 | 302,881 | 160 | 0.00112939 | 100% | 1,992.947 | 338.202 |
| B3 | T4 | 303,010 | 160 | 0.146635 | 96.875% | 2,044.654 | 286.353 |
| B4 revision3 Scratch | T3 | 896,358 | 64 | 0.000119821 | 100% | 220.755 | 30.893 |
| B4 | T4 | 897,992 | 64 | 0.000476328 | 100% | 260.225 | 32.453 |

八项拟合均通过既定的 accuracy≥93.75%、NLL<0.25、概率非常数、分支梯度与参数更新检查。B1–B3 的 tracker 分支和主表示分支分别记录；B2/B3 另记录 local/global 参数组。实际输入梯度包含 pixel、bar 以及 tracker set。原生 point 分支不含 tracker 时，其对应 native tracker 输入梯度为零符合配置，不能据此判为 tracker set 未使用。

完整证据为 `fit/interactive_a1/*.json`、`summary.json`；每个 JSON 保留物理事件键、各 checkpoint 的 NLL/accuracy/probability std、分支 gradient squared norm、参数更新 RMS。英文图为 `figures/r1_real_fit.pdf`，同名 PNG 已检查；数据为 `figures/r1_real_fit.csv`。拟合时间包含当前脚本内构建和评价，不能据此给出架构吞吐排名。

较大样本 profile 发现两个先前小样本没有覆盖的问题，已按程序原因处理：

1. **B1 空投影梯度放大。**同一真实八事件 batch，在 H100 MIG 3g.47gb 上，修复前 FP32/BF16 的 loss 均有限，参数逐元素梯度也有限，但总体 L2 norm 为 Infinity。空图经过多层 GroupNorm，有限前向不能保证可用反向。将无响应投影在 fusion 前置零后，前向 loss 逐点相同，FP32 两步 norm 为 **5.75952 / 5.91262**，BF16 为 **5.72545 / 5.88647**，无 NaN 参数。证据为 `audit/diagnose_B1_k5.json` 与 `audit/diagnose_B1_empty_k6.json`。本修复不删除任何事件，presence 仍进入分类头。
2. **B2 compiled/BF16 局部反向。**固定同一八事件 batch、1,111 local blocks、32 slots，FP32 eager、FP32 compiled、BF16 eager 都通过两步 loss/gradient/parameter 检查；只有 BF16 compiled 在第一步产生 local/input NaN 梯度。不能把它解释为 ParticleNet 架构或原生 pixel 无效。R1 默认先使用已验证的 eager/BF16 路径。另将 masked max 的 sentinel 改为 BF16 精确可表示的 -16,384，核查 fused backward 比较的舍入假设；该假设的专门 GPU 验证仍在排队，不能称其已证明根因。证据为 `audit/diagnose_B2_k3.json`。

空投影修复及 B2 eager 路径随后在 Kubeflow H100 MIG 上进行了真实 BF16 小样本重新训练：

| Model | Task | Final NLL | Accuracy |
| --- | --- | ---: | ---: |
| B1 | T3 | 0.0000380021 | 100% |
| B1 | T4 | 0.000162465 | 100% |
| B2 eager | T3 | 0.000504440 | 100% |
| B2 eager | T4 | 0.00345028 | 100% |

内部 health 和原始曲线位于 `fit/bf16_k7/`。这些值仍只回答可优化性，不用于挑选泛化最优架构。

截至本报告，**28 项相关测试通过**，包括任务概率闭合、分解 NLL、未知 decay、完整 source-shard 覆盖、投影信号守恒与方向、tracker set permutation/empty mask、空投影反向、tie-aware 加权指标与 sklearn 对照、原 S2/S3-repair 回归。实际 Parquet dotted projection 会把 pixel/bar 的两个 leaf 都命名为 `channel_id`，已改为保留其列位置区别并加入磁盘读回测试。

profile 的工程失败另外保留原始日志：初版错误要求 R0 catalog 含 `all_empty`，改为从实际通道数推导；LCG 的 PyArrow 没有 zstd codec，所有新输出改为复用项目已有的可用 codec 选择。未通过最终写出的 profile 不记作完整成功。失败与取消、Condor/Kubeflow allocation 都纳入 `resources.json`，该文件随在运行任务持续更新；R1 最终 device-hours/CPU-hours **尚未测量完成**。

交互拟合只能检查少量真实事件；完整转换、流水 profile 与正式训练均使用 batch backend。提交记录、现场 schedd 负载与兼容性、准确源码归档位于 `runtime/*.submission.json` 和 `code/`。本轮最空闲且符合 EOS/WRITE 策略的 schedd 为 `bigbird20.cern.ch`；其他节点空闲不能代表本账号有相应 GPU 权限。

主要命令示例：

```bash
python -m analysis.flavour.v2.r1.prepare plan --root R1_OUT --r0 R0_OUT
python -m analysis.flavour.v2.r1.prepare convert --root R1_OUT \
  --task-start 0 --task-count 7 --processes 4 --threads 2
python -m analysis.flavour.v2.r1.prepare finalize --root R1_OUT
python -m analysis.flavour.v2.r1.fit --root R1_OUT --r0 R0_OUT --attempt UNIQUE
python -m analysis.flavour.v2.r1.fit --root R1_OUT --r0 R0_OUT \
  --attempt UNIQUE_BF16 --families B1 B2 --bf16
python -m analysis.flavour.v2.submit --stage R1 --root R1_OUT --name UNIQUE \
  --gpu --min-capability 7 analysis.flavour.v2.r1.profile \
  --root R1_OUT --r0 R0_OUT --family B1 --attempt UNIQUE
python -m analysis.flavour.v2.submit --stage R1 --backend kubeflow \
  --flavor h100-47gb --root R1_OUT --name UNIQUE --gpu \
  analysis.flavour.v2.r1.profile --root R1_OUT --r0 R0_OUT \
  --family B4 --attempt UNIQUE --compile-local
python -m unittest tests.test_flavour_v2_r1 tests.test_flavour_spatial \
  tests.test_flavour_s3_repair
```

新输入 profile 使用可调整的 local execution chunk，数学分组与通道覆盖不变。R0 的短 fit chunk=256 不应机械成为正式吞吐设置；B4 复用原 reference 的 2,048，B2/B3 分别检查 512/1,024。各设置的最终计时与显存验收在后续报告中给出，不能混用不同卡和不同冷启动状态宣称纯 kernel 加速。

文献再次在线核对了 [Group Normalization](https://arxiv.org/abs/1803.08494)、[PointNet](https://arxiv.org/abs/1612.00593)、[ParticleNet](https://arxiv.org/abs/1902.08570)、[Particle Transformer](https://arxiv.org/abs/2202.03772) 和 [Mixed Precision Training](https://arxiv.org/abs/1710.03740)。前两者支持小 batch normalization/set pooling 的选择，后两者是现有 detector adaptation 的方法来源；mixed precision 文献支持独立验证数值路径与保留稳定精度，不证明本次某个 compiler kernel 的具体故障。Torch troubleshooting 网页本次返回 HTTP 403，未作为已读证据。来源访问记录保存在 `audit/literature.json`。

**阶段性结论**

已测事实：三类/四类任务接口与标签对应明确；四个神经网络均存在有效真实小样本学习路径；空模态会暴露被信息丰富拟合样本掩盖的梯度故障；B1 修复与 B2 稳定执行路径已通过真实 BF16 重训。GPU 分区可以运行这些模型，实际内存与耗时必须由 profile 确定。

待验证假设：B2 fused masked-max 的数值舍入是否是 compiled/BF16 故障根因；新模型在完整 Tune 上是否改善现有 B4；三类训练是否减少任务干扰；tracker 或原生 pixel 的分类净收益。目前均没有泛化数据支持结论。

**下一步计划**

1. 完成全部 56 shards 与精确身份、标签、总 PE、观测数量验收，生成共同 B0 特征与固定 Tune screen/stress 子集。
2. 在任何新 Tune score 读取前冻结 campaign、两学习率 screen 额度、选择规则、四 pass 扩展、seed17/42/2026、bootstrap 与工作点定义。不同 GPU 型号按用户要求并行使用，同族任务/学习率在同 allocation 配对，设备成本单列。
3. 实际运行 B0–B4 的 T3/T4 与 B4-MAE T4，延长最优配方，并对最强两族与 B4 共同基线扩 seeds；执行共同曝光与同设备计算控制。
4. 给出完整 Tune 的 all/pixel-centered/large 三类表、四类条件三类、τ forced assignment/contamination、tracker/pixel 控制与 fixed noise/combined stress。按 accepted-family-connected source block 做 400 次 paired bootstrap，训练 seed 波动另列。
5. 用真实结果完成 W04–W06/W08 和 R1 整体验收、资源清理与报告；在此之前保持 R1 为进行中。
