# R0：新分区、实际结构与数据流水线

**R0完成。**新分区与历史使用记录冻结，真实结构和输入白名单导出，真实32-event拟合通过，256/10k事件的输入等价与完整数据流水线profile完成；最终73项读回检查和81项相关自动化测试通过。本轮建立后续模型比较的运行基础，新增模型泛化质量尚未测量。

## 阶段目标

落实 [R0 合同](../roadmap/v2/04_execution_plan.md#2-r0使后续比较具体可执行)：建立可追溯的历史研究使用记录，冻结与历史开发及初始化语料隔离的新分区；自动导出真实 transform/model 输入与结构；用真实 Train 事件验证缓存等价性、学习健康和资源效率，为 R1 的三类/四类同输入基线提供运行基础。

既有 accepted samples、`family_hi/family_lo` 和 detector-proxy 继续复用。新 Confirm 只公开 family 身份，row/label 位于独立 sealed 路径；本轮不转换 Confirm observations、不进行 Confirm 推理。

## 工作内容

- 新增 `analysis.flavour.v2`：分区与历史来源核对、按需转换、实际结构导出、native/packed 数据读取、micro/stream profile、32-event fit、验收、资源关闭及绘图。
- 用现成 LCG108_cuda Python 3.12.11、Torch 2.7.1、PyArrow 20.0.0；未安装本项目或重建模拟软件。编译对照复用已验证的 S3 repair 资产，在 worker scratch 设置 Triton 3.3.1 与 compiler headers/linker files。大样本元数据归约、10k 转换和 10k GPU profile 经 HTCondor，交互 T4 只执行 256-event microbenchmark 与 32-event fit。
- 修正未来 S2/S3/S4/S5 训练 manifest 的 representation metadata。S2/S3 的既有数值常量接入 transform 与 exporter 的共同定义，数值不变；已有模型和历史 manifest 不改写。
- 缓存为 allocation scratch 上的临时资产；geometry/readout/representation revision、窗口、模态、响应和 region 配置参与 cache key。最终格式另存事件身份索引，加载时核对实际物理键；feature arrays 不包含 labels、truth 或 provenance features。

运行根目录统一为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14`，下文路径相对该目录。代码与配置在仓库，运行快照在 `code/`，执行时的配置副本在 `configs/`。

## 详细数据：历史使用与新分区

accepted inventory 有 **7,237,572 行、7,237,572 个 family**。本轮按研究实际语料取完整集合，另保守排除整个旧 calibration/test，包括 usable-only catalog 未收录的 222 行。历史使用清单共 **1,560,122 个 family**。

最终追溯覆盖 20 份去重后的 accepted source tables，另有 2 个旧分区全量过滤规则。S3/S3-repair 的 shard 链接解析到 S2，S5 解析到 S4；S4 MAE/Rel、probe、方向/对象与辅助任务按完整语料排除，包含 prepared reserves。S0 显示、smoke 和人工诊断也纳入。`usage_mask` 表示 fit/SSL、validation/probe、calibration、published test、pilot/manual、prepared reserve 的合并角色；这是语料使用的保守并集，不是每个 event 的精确 optimizer exposure。

S6 的 **3,512** 个外部 generator events 用 `(tune, source_path, source_row)` 单列；`run=20001/20002` 与 accepted inventory 的 run namespace 不同。其 vertex donor 来自旧 calibration，已完整排除。独立 hard-scatter 属性沿用 S6 accepted contract，不伪造 accepted-family key。

排除历史研究使用后的可用量，在分配本轮预留集之前为：

| Class | Unused rows / families |
| --- | ---: |
| e CC | 669,597 |
| mu CC | 3,298,100 |
| tau CC | 493,480 |
| NC | 1,215,348 |

跨标签的 unused ambiguous family 为 0。新分区全部平衡，冻结 seed=`20261114`：

| Partition | Events | Per class | Role |
| --- | ---: | ---: | --- |
| Train100k | 100,000 | 25,000 | R1 common pool |
| Train300k | 300,000 | 75,000 | Nested scaling |
| Train1M | 1,000,000 | 250,000 | Conditional scaling |
| Tune | 40,000 | 10,000 | Development / checkpoint selection |
| Cal-A | 40,000 | 10,000 | Calibration after freeze |
| Cut-B | 160,000 | 40,000 | Threshold selection after freeze |
| Confirm | 200,000 | 50,000 | R7 independent confirmation |

Train100k⊂Train300k⊂Train1M；训练只从旧 train 中选择。训练先打乱 source chunks、再随机 chunk 内行，以提高后续 ROOT 读取密度；预留分区按事件随机排列选取。family 不拆分，所有新分区均排除整个历史使用 family，因此与复用 MAE 初始化的原训练/SSL 语料也不交叉。初次分区构造 **33 项检查全部通过**。

可复核产物：`partitions.json`、`manifests/usage_ledger.parquet`、`usage_sources.parquet`、`external_generator_usage.parquet`、`usage_closure.json` 及各分区 manifest。`usage_sources.parquet` 用 `ledger_row` 与 frozen ledger 对齐，`source_mask` 的 bit 对应 closure 中 `source_id`。

Confirm 的公共 manifest 只有 `confirm_index/family_hi/family_lo`；row/label 保存在 `sealed/confirm_rows.parquet`，目录/文件权限为 700/600。这是工作流盲化；R0 构造时使用标签进行平衡抽样，之后的 profile/fit/readback 不打开 sealed row/label。

## 详细数据：真实表示与输入

参考结构为 `s4_empty_scratch_n100000_s42` 的 revision 3 multiscale encoder，本轮从 config 实例化结构，不载入历史权重。

| Quantity | Measured value |
| --- | --- |
| Parameters | 897,992 |
| Node features | 18 observed/derived columns |
| Pixel pitch x / y | 0.0208 / 0.0228 mm |
| Pixel window | 16 × 16 pixels = 0.3328 × 0.3648 mm |
| Channels per local block | At most 32; no event cap |
| Blocks per region | 32 in canonical module/type order |
| Queries per region | 2 |
| Global latents | 64 |
| Local attention layers | 2 |
| Explicit geometry context layers | 0 |
| Global latent attention layers | 2 |
| Direct local readout / empty bypass | Enabled / enabled |
| Tracker z code | Station ordinal × 1000 mm proxy |

模型具备通过 latent cross-attention 汇合不同 plane 的路径；当前并无显式跨 plane 的几何关系层。region geometry 使用 mean scaled z、detector one-hot 与 response，xy geometry 为零。pixel local residual 使用实际 pixel window；bar/tracker residual 仍使用 4 mm 窗口，tracker grouping 使用 16 mm xy cells。tracker z 为序号编码，不能解释为物理飞行距离。

输入白名单为 pixel 的 `channel_id/xyz_mm`，bar 的 `bounds_mm/horizontal/edep_proxy_MeV/pe_proxy/panel/layer`，tracker hypothesis 的 `station/xy_mm/sigma_xy_mm`，以及固定 geometry 字段。`representation_metadata.json` 同时列出 18 个 feature 名称、坐标除数、所有模型读取的 batch tensors 和可选 object 输入。label 仅进入 loss，index 仅用于事件对齐；truth、family、run/chunk、τ 分支和能量标签不进入分类 encoder。

旧 `same_plane_4mm` 与 `cross_plane_message_passing=true` 文本不再覆盖未来真实结构信息。修正适用于新运行，未修改旧 scientific artifacts。

## 详细数据：真实样本与等价性

只转换 Train100k 中每类前 2,500 个 frozen class rank，共 **10,000** 事件；读取 **19** 个 source ROOT 文件，生成四个各 2,500-event 的混类 shard。转换为四进程×两线程，作业申请 8 CPUs。source shard 转换耗时为 97.863 / 128.180 / 123.714 / 123.472 s，分别记录 ROOT read、aggregation、write；最大单 converter RSS=1,556.91 MiB。总 PE 逐 shard 精确守恒。

10k native channel occupancy 的 min/median/P90/P99/max 为 **0 / 565.5 / 2,866.1 / 6,826.03 / 29,340**。固定 micro sample 为每类 64 个事件，在各类 occupancy 排序上等距抽取，共 256 个；低/中/高 bands 各类 22/21/21。它用于压力与效率核对，不代表全 inventory 的无偏占用分布。

10k 事件的 native 与 packed cache 所有 collated tensors，包括事件 index、label、通道、group/module mapping 和 masks，逐位一致。固定 256 事件另与启用原 S2 grouping 的 canonical frontend 逐数组比较；全部通过。bar signal 的 float32 log 特征反变换最大相对误差为 edep **2.6372×10⁻⁷**、PE **4.2307×10⁻⁷**。A100/H100的两事件GPU logits readback完全一致；T4最大绝对差为 **1.1921×10⁻⁷**，通过profile原定的`rtol=3e-5, atol=3e-6`检查。

转换身份另外按每一整数列单独比较，避免 signed event key 与 uint64 family 混成 NumPy 浮点数组时丢失低位。10k 行逐整数列完全一致；低位差异回归测试已加入。

最终 cache format revision 2 在 256 个真实事件上再次通过事件键和全 tensor 等价检查；数据目录或事件顺序错误、表示配置不匹配及 tensor 损坏均有针对性测试。

## 详细数据：真实学习检查

从固定 micro 样本中选取 pixel/bar 均有响应、总 native channels<5,000 的真实事件，在各类 occupancy 上取 8 个，共 32 个。按 `(run,chunk,row)` 映射实际 dataset index，并核对载入后每类确为 8 个。

Scratch seed42、完整 897,992 参数结构、FP32、AdamW LR=0.0005、weight decay=0、batch8、clip norm1；T4 上将 `local_chunk_blocks` 设为256以控制执行内存，不改变图与样本。64 updates=512 event exposures，耗时 **36.094 s**。

| Fit measurement | Initial | Step 32 | Step 64 |
| --- | ---: | ---: | ---: |
| NLL | 1.329527 | 0.024746 | 0.000418 |
| Accuracy | 40.625% | 100% | 100% |
| Distinct probability vectors | 32 | 32 | 32 |

平衡常数分类器 NLL=ln4=1.386294。step1/4/16/32/64 的内部 health 均通过；input/local/latent_read/global_blocks 梯度非零，所监控的七组参数全部有实际更新。step64 的 input、local、latent_read 梯度平方范数分别为 `6.9765e-4 / 2.2367e-4 / 1.4969e-5`；global_blocks 为 `6.5706e-6`。最终各类概率标准差约0.4327，最大 GPU allocated memory=230.12 MiB。

证据位于 `fit/fit32_a2.json` 与对应32-event键表。本检查回答训练路径能否学到真实样本；泛化质量、新三类/四类比较、τ工作点及饱和仍**未测量**。

## 详细数据：效率、资源与验收

完整流水测试采用与参考一致的结构，BF16 encoder/FP32 classifier、batch8；同一 A100 allocation 内比较 eager 的 native、native+bucket、packed+bucket，各使用2/4 workers，每种两遍，另测 inference。另一 H100 allocation 比较 compiled local encoder 的 native 与 packed+bucket，均为2 workers。profile 执行 forward/CE/backward/clip/AdamW，LR=0；inference 也只读 Train，属于运行计时。

每个 case 的首遍包含 worker 启动；EOS→allocation-local staging 单列。data-wait 是主进程等待下一个 batch 的阻塞时间，不是全部 CPU 预处理耗时。steady 口径排除首 batch；首遍和完整遍的口径包含 iterator startup。旧 eager 快照的 `data_wait_fraction` 未加 startup，最终验收按分段原始数值重算，不修改已运行快照。

| Eager / A100 scenario | First-pass events/s | Second-pass events/s | Steady wait | Aggregate RSS GiB | GPU allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| native, 2 workers | 55.236 | 69.551 | 6.683% | 6.515 | 1.014 |
| native, 4 workers | 45.120 | 70.301 | 1.158% | 9.820 | 1.014 |
| native+bucket, 2 workers | 47.264 | 56.863 | 17.294% | 6.788 | 1.091 |
| native+bucket, 4 workers | 42.137 | 60.284 | 4.196% | 10.182 | 1.091 |
| packed+bucket, 2 workers | 48.066 | 60.955 | 4.768% | 5.089 | 1.091 |
| packed+bucket, 4 workers | 42.664 | 63.537 | 0.658% | 6.561 | 1.091 |

最终 RSS 以 `acceptance.json` 为准；aggregate 是同一时刻 parent+workers RSS 的采样峰值，可能重复计入共享映射。native 的4 workers仅比2 workers提高约1.08%热态吞吐，worker启动约74 s对35 s，内存增加约51%；短训练采用2 workers更合适。同为bucket的配置中，packed cache移除了重复tokenization，但仍未超过普通native的总体吞吐。

| Compiled / H100 scenario | First-pass events/s | Second-pass events/s | Steady wait | Aggregate RSS GiB | GPU allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| native, 2 workers | 73.817 | 173.892 | 20.880% | 7.155 | 0.786 |
| packed+bucket, 2 workers | 150.312 | 188.144 | 10.801% | 5.872 | 0.862 |

H100上第二条流水线热态吞吐提高8.20%，是cache与sampler的组合对照；不能单独归因于cache。按顺序运行的case共用compiler disk cache：native单事件warm-up/compile为42.858 s，后续packed为0.037 s；native首遍还包含形状相关编译。因此首遍差异不能解释为独立冷启动编译收益。A100与H100的对照分别成立，未测量同设备eager→compiled的因果加速比。

两种sampler的channel padding utilization同为 **8.833%**；event block padding utilization从native的 **22.301%** 提高到bucket的 **48.826%**。更少event padding并未保证eager更快。局部32-slot padding仍占较大比例，变长local kernel或窗口规则值得在后续独立测量，不能由本轮效率结果推断细粒度信息没有物理作用。

H100的10k packed cache格式2为 **976,458,096 bytes**，构建 **53.964 s**，native observations+targets staging为196,287,484 bytes、0.884 s；A100旧格式为976,377,584 bytes、构建78.794 s、staging1.020 s。格式差异为新增事件身份索引。H100每10k热态pass只节省4.356 s，构建成本约需 **12.39个热态pass** 才摊平；这是假定吞吐和CPU成本保持的规划推算，尚未实测独立重启的总成本。R1首轮4-pass默认采用 **native、2 workers、共享本地staging、bounded 2-shard decoded LRU**；compiled沿用已验证修复环境，packed在同表示多次训练可复用时再按总成本选择。

默认compiled native仍有20.880%的稳态阻塞等待；该遍tokenization为47.534 worker-s、collate为12.392 worker-s。R1在更大pool和各架构上重测这些CPU阶段，不将“已完成R0”解释为所有场景都达到20%建议值。

只对最快训练配置各测一遍inference：A100/native4为160.881 s、62.158 events/s、steady wait22.889%；H100/packed+bucket2为35.209 s、284.018 events/s、steady wait35.843%。其启动分别为71.778/10.298 s；H100 collate为13.149 worker-s，GPU compute为15.080 s。inference的batch/workers尚未独立优化，R1应保留持久validation loader，并单独检查collate和较大batch的内存/吞吐。

microbenchmark 在T4/FP32上完成：native2 workers第二遍23.3136 events/s，稳态wait=1.2742%；4 workers为23.1247 events/s；packed+bucket2/4 workers为21.5691/21.6152 events/s。256小样本下packed没有提高总体训练吞吐。

81项相关自动化测试全部通过，日志为 `logs/unit_tests_delivery.log`；覆盖新R0、原空间/plane frontend、修复模型、S4、S5对象输入、empty response与缓存。最终验收 **73/73** 项通过：新分区计数/交集/嵌套、历史来源、精确uint64身份、输入白名单、真实拟合、三个profile的完整事件覆盖与等价性、临时cache清理均通过。验收CPU任务耗时5.921 s、peak RSS800.207 MiB；其代码只读Confirm公共family表及封存文件的权限元数据。

全部实际提交的任务在`bigbird20.cern.ch`终止，每条仅启动一次；下表为schedd history的实际allocation wall time与CPU申请值，包含失败尝试。选择时的全schedd负载、EOS策略与WRITE检查保存在各`runtime/*.submission.json`。

| Task | Cluster | Exit | Allocation seconds | CPUs |
| --- | ---: | ---: | ---: | ---: |
| Partitions a1 | 1582822 | 0 | 191 | 6 |
| Prepare a1 | 1582823 | 0 | 145 | 8 |
| Stream eager a1 | 1582824 | 0 | 2,688 | 8 |
| Usage closure a2 | 1582825 | 1 | 25 | 4 |
| Usage closure a3 | 1582826 | 0 | 36 | 4 |
| Stream compiled a1 | 1582827 | 0 | 585 | 6 |
| Acceptance a1 | 1582828 | 1 | 15 | 4 |
| Acceptance a2 | 1582829 | 0 | 12 | 4 |

| Resource | Measured cost |
| --- | ---: |
| A100-PCIE-40GB allocation | 0.746667 device-hours |
| H100 NVL allocation | 0.162500 device-hours |
| Interactive T4, valid plus failed bounds | ≤0.175688 device-hours |
| Total, counts GPU hours separately from FLOP equivalence | ≤1.084855 device-hours |
| R0 planning envelope | 2 device-hours |
| HTCondor allocated CPU | 7.686667 core-hours |
| HTCondor measured process CPU | 1.544444 CPU hours |
| Interactive CPU | 未测量 |

GPU allocation含setup、编译、完整cache验证与profile。T4成功micro/fit为327.768/36.094 s；失败fit按记录start到terminal traceback log mtime加2 s计上界28.615 s；失败micro按观察到的PID生命周期计上界240 s，精确区间未测量。站点拒绝的closure a1没有allocation。完整记账为`resources.json`；不同GPU型号的小时不等效于相同计算量。

所有八个cluster已从live queue消失，交互R0 PID为0；五份成功fit/cache/profile产物均确认scratch删除，交互`/tmp/pinpoint-r0-*`无残留。九个AFS attempt目录删除，释放14,351 bytes，wrapper/submit文本及源码快照已保存在EOS。运行目录保留约312.3 MB的compact manifests、10k/256真实Train复核输入、日志和图表；packed cache、checkpoint与xAOD均未长期保留。清理证据为`cleanup.json`。

英文图表与数据：[Loader timing](../../results/flavour/v2_2026-11-14/figures/r0_loader_profile.pdf)、[Memory and occupancy](../../results/flavour/v2_2026-11-14/figures/r0_memory_and_occupancy.pdf)、[Compiled pipeline](../../results/flavour/v2_2026-11-14/figures/r0_compiled_pipeline.pdf)、[Profile CSV](../../results/flavour/v2_2026-11-14/figures/r0_stream_profile.csv)。对应PNG已逐图检查；每batch原始分段计时在`profiles/*_batches.parquet`。

用于R1排期的线性估计：reference的100k×4-pass热态训练，A100/eager/native2约1.60 h，H100/compiled/native2约0.64 h；H100/packed+bucket约0.59 h，加按事件数外推的cache构建约0.15 h。以上不含首次编译、读入与validation，100k实测吞吐及新架构成本仍未测量。

主要执行命令如下；第一次构造分区和数据使用新的运行目录，每次提交使用独立attempt name，实际提交的完整命令、源码快照和schedd依据在 `runtime/*.submission.json`。

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R0_OUT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14
R0_CANDIDATES="$R0_OUT/configs/s6_reference.json"

python -m analysis.flavour.v2.submit --root "$R0_OUT" \
  --name r0_partitions_a1 --cpus 4 --memory 12GB \
  analysis.flavour.v2.partitions --root "$R0_OUT" \
  --config "$PWD/roadmap/configs/v2_r0.json" --cpus 4
python -m analysis.flavour.v2.submit --root "$R0_OUT" \
  --name r0_prepare_a1 --cpus 8 --memory 24GB \
  analysis.flavour.v2.prepare --root "$R0_OUT"
python -m analysis.flavour.v2.metadata --root "$R0_OUT" --candidates "$R0_CANDIDATES"
python -m analysis.flavour.v2.profile --root "$R0_OUT" --candidates "$R0_CANDIDATES" \
  --mode micro --attempt a2 --batch-size 4 --passes 2
python -m analysis.flavour.v2.submit --root "$R0_OUT" \
  --name r0_stream_a1 --cpus 8 --memory 24GB --gpu \
  analysis.flavour.v2.profile --root "$R0_OUT" --candidates "$R0_CANDIDATES" \
  --mode stream --attempt a1 --batch-size 8 --passes 2
python -m analysis.flavour.v2.submit --root "$R0_OUT" \
  --name r0_stream_compiled_a1 --cpus 6 --memory 16GB --gpu \
  analysis.flavour.v2.profile --root "$R0_OUT" --candidates "$R0_CANDIDATES" \
  --mode stream --attempt compiled_a1 --batch-size 8 --passes 2 \
  --compile-local --variants native packed_bucket --workers 2
python -m analysis.flavour.v2.fit --root "$R0_OUT" --candidates "$R0_CANDIDATES" --attempt a2
python -m analysis.flavour.v2.data --root "$R0_OUT" --candidates "$R0_CANDIDATES"
python -m analysis.flavour.v2.submit --root "$R0_OUT" \
  --name r0_usage_closure_a3 --cpus 4 --memory 12GB \
  analysis.flavour.v2.usage_closure --root "$R0_OUT"
python -m analysis.flavour.v2.submit --root "$R0_OUT" \
  --name r0_accept_a2 --cpus 4 --memory 12GB \
  analysis.flavour.v2.accept --root "$R0_OUT"
python -m analysis.flavour.v2.resources --root "$R0_OUT"
python -m analysis.flavour.v2.plots --root "$R0_OUT"
python -m unittest tests.test_flavour_v2_r0 tests.test_flavour_spatial \
  tests.test_flavour_s3 tests.test_flavour_s3_repair tests.test_flavour_s4 \
  tests.test_flavour_s5_model_objects tests.test_flavour_empty_response tests.test_flavour_cache
```

本轮已修复的实现问题与失败成本分别记录：首次micro监控调用psutil全局进程扫描，触发lxplus `/proc/1` 访问限制，改为本任务后代PID遍历；首次fit误用micro清单序号作为shard读取序号，改为物理键映射并添加回归测试，另修复最终JSON的NumPy bool序列化；历史来源closure首轮漏把全量旧cal/test过滤规则列作来源，补齐后通过。schedd选择曾误把负向EOS限制表达式识别为兼容，bigbird21提交遭站点拒绝且未产生allocation；修复匹配后在bigbird20提交。

首次最终验收额外把GPU logits逐位相等列为必要条件，因T4的1.1921×10⁻⁷舍入差异失败；原profile的数值容差检查和输入逐位一致检查均已通过。最终验收将输入逐位等价与GPU数值读回分别检查，后者采用最大绝对误差≤3×10⁻⁶；首次结果保存在`audit/r0_accept_a1.json`，不重跑GPU profile。所有失败计入资源，只作工程诊断。

资源汇总首次遇到schedd以空stdout表示空队列，修正为兼容空stdout和`[]`后完成读回。分区构造代码已固化Confirm目录700、payload600权限；本轮sealed状态通过最终stat检查，未重建分区。

## 阶段性结论

**已测事实：**现有inventory足以建立所计划的平衡嵌套训练集和440k非训练预留集；历史使用记录与新分区可追溯；当前结构的实际窗口和显式几何层与旧文字记录不同；真实32事件拟合和内部更新正常；本轮数据表示与缓存保持同事件输入等价。

**运行决策：**R1短screen先用native、2 workers、共享local staging和bounded decoded LRU；compiled沿用已验证环境，缓存复用成本单列。A100的4 workers与H100的packed+bucket分别有最高热态吞吐，但短训练需计入启动、RAM和cache构建成本。inference仍受CPU阶段和启动影响，需在R1单独优化。

**假设与后续验证：**局部padding和缺少显式跨plane几何层可能限制效率与拓扑利用；R0只确证前者的槽位利用率和后者的结构事实，没有测量它们对分类质量的因果影响。旧31%–47%的data-wait与本轮运行在设备、编译和数据规模上不同，不把两者直接相减称为纯loader改进。

## 下一步计划

1. R1实现T3/T4/T3T统一loss/evaluator和B1/B2/B3的tracker late fusion，使用新Train100k/Tune40k，Confirm保持封存。验收为同物理事件、类别映射、缺失模态mask及真实小样本学习正确。
2. 在胜出loader设置上先做每族两学习率的同计算短screen，再延长最强配方。B4 Scratch为共同参考，MAE初始化单列原SSL成本；实际三类重训不能用删去四类中的真τ代替。
3. 用R0测量限定资源，R1仍单独验证各新架构和compiled执行的吞吐；R2再按数据/曝光曲线扩展300k/1M。窗口或跨层关系改变进入独立模型对照，以共同三类和τ固定FPR结果判断收益。
