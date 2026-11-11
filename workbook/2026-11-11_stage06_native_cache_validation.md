# 阶段 06：S1 本地 shard 缓存、原生传输与验收补充

已补齐 S1 计划中的 worker/local scratch 缓存、可复现窗口采样、标签状态传递和实测。15 项测试通过；HTCondor 完成四类共 4,000-event、三个配置各两个 epoch 的 I/O 检查，事件与通道完整保留，临时缓存全部清理。64-event 原生 GPU tensor 往返完全一致。**S1 detector-proxy 工程交付已补齐；S0 生产 source-family 映射仍待实际生产记录，整个 S0/S1 正式验收尚未关闭。**

## 阶段目标

回答原生 catalog 能否通过有界本地缓存与多 worker 流式读取，以及增加 worker 是否真正提高吞吐。上一份 S1 报告测的是单 shard 已在内存中的 collation；这次将 staging、decode、worker startup 和 IPC 纳入 wall time，并保留类别混合条件。

## 工作内容

- `CachedNativeDataset` 在初始化时只读 observation/target Parquet footer，以累计行边界定位 shard。每个 PID 建立私有 scratch 目录，按需复制 observation/target 两个文件，用 LRU 限制驻留 shard 数；只解码 `label/label_status` 两个 target 列。provenance、ideal tracker 与 B0 features 不进入缓存。
- scratch 按显式路径、Condor scratch、系统 tmp 的优先级选择，拒绝 EOS/AFS。fork 子进程不继承父进程目录的所有权；spawn 不序列化父缓存。显式 `close()`、context manager 和正常 worker exit 清理各自目录。
- `ShardBatchSampler` 用 seed+epoch 打乱 shard 窗口及窗口内事件，覆盖每个索引一次，不截断通道。窗口不能超过缓存容量。当前四个 shard 各为单一类别，单 shard 窗口只作为 I/O 对照；训练需混类 shard 或覆盖四类的窗口，并按冻结 manifest 选择 split。
- `collate_native` 保留 `targets.label_status`，调用者可以明确隔离标签例外，状态不进入 observations。新增 profiler 和 Condor 提交入口；原 B0 训练结果未重跑。
- 修正路线图首页中全库事件数/训练仍未测量的旧措辞，并将 diagnostic 的存储外推状态指向独立 inventory+pilot 计算。

## 详细数据

### 输入、环境与命令

输入复用阶段 05 的 `shards/{10081,10082,10083,10084}`，每类首个完整 chunk 共 1,000 行。全部产物位于 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11`。CPU 测量在 `b9p19p2243.cern.ch`，使用 LCG_108_cuda 的 Python 3.12.11、PyArrow 20.0.0、PyTorch 2.7.1。代码基于 `6e6f52d`，本阶段实现和本报告一起提交；`validation.json` 登记最终 implementation commit。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
TASK_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11
python -m analysis.flavour.submit loader_profile --root "$TASK_RESULTS"
python -m analysis.flavour.profile_loader --input "$TASK_RESULTS/shards" \
  --output "$TASK_RESULTS/native_transfer.json" --transfer-only
```

提交前检查了 schedd 队列、直接 EOS policy 与 WRITE 授权；bigbird20 当时 running/idle/held 均为 0，且允许当前路径合同。Cluster `1573720.0` 申请 3 CPUs、6 GB，站点调整为 3 CPUs、9,000 MB，运行一次、exit 0；scheduler wall time 66 s、user/system CPU 56/17 s。证据为 `submission_loader_profile_154925.json`、`condor_loader_resources.json` 和对应日志。

### 完整性与吞吐

共同参数：batch size 16、seed 42；两个 epoch 的种子分别由 epoch 0/1 派生。两个 worker 使用 spawn、persistent workers、prefetch=1，每 worker 的 Arrow/BLAS/Torch 各 1 线程。每个 epoch 均读到 **4,000 行、4,263,296 个 pixel/bar/tracker tokens、每类 1,000 行**；标签状态保持 3,997 consistent 和 3 ν–e elastic candidates。这个扫描只检查 I/O，不训练、不计算 calibration/test 分类性能。

| Workers | Cache / window (shards per worker) | First epoch (s; events/s) | Second epoch (s; events/s) | Mixed-class batches / epoch |
| --- | --- | --- | --- | ---: |
| 0 | 1 / 1 | 3.375; 1,185.16 | 2.347; 1,704.34 | 0 / 252 |
| 2 | 1 / 1 | 11.908; 335.91 | 5.009; 798.56 | 0 / 252 |
| 2 | 4 / 4 | 12.297; 325.29 | 3.437; 1,163.71 | 250 / 250 |

首 epoch 包含 worker 启动；OS/EOS cache 状态未控制，两个 epoch 不能给出吞吐置信区间。单进程和双 worker 的 1-shard 配置使用相同采样规则，增加 worker 在本次小型任务中没有提高吞吐。4-shard 配置同时改变缓存容量和 batch 混合，不能只凭它与 1-shard 的吞吐差值归因缓存，但其第二个 epoch 的累计 stage/decode counters 完全不变，直接证明复用了本地已解码数据。未测量同一 mixed-window 的串行对照，也未据此选择 S2 训练的最优 worker 数。

### 缓存容量、开销与清理

| Configuration | Two-epoch misses / hits (all workers) | Total staged bytes | Peak local bytes per worker | Process peak RSS (MiB) |
| --- | --- | ---: | ---: | --- |
| 0 workers, 1 shard | 8 / 7,992 | 152,847,162 | 23,549,125 | 737.28 |
| 2 workers, 1 shard | 16 / 7,984 | 305,694,324 | 23,549,125 | 741.27 / 742.32 |
| 2 workers, 4 shards | 8 / 7,992 | 152,847,162 | 76,423,581 | 1,138.80 / 1,149.46 |

单进程的累计 stage/decode 为 `0.828/3.289 s`；双 worker、1-shard 各为 `0.277/3.158 s` 与 `0.273/3.410 s`；双 worker、4-shard 各为 `0.149/1.856 s` 与 `0.152/1.986 s`。这些是各进程累计时间，不能跨 worker 直接相加解释 wall time。collation 的 worker 时间和分 epoch counters 见 `loader_profile.json`。

所有五个 CPU 配置进程的私有 scratch 目录都在正常结束后检查为不存在；源 shard 保留。LRU 容量约束的是 cache 的本地文件及 shard 对象数；当前 batch 引用、prefetch、Arrow allocator 和 Python/Torch 基础开销都计入 RSS，因此不能用压缩文件 bytes 代替内存申请。Condor 的最终 ResidentSetSize 约 100,000 KB，明显低于进程内 `getrusage` 的实测峰值，不能将该 scheduler 快照当成完整进程组内存峰值，也没有将各进程的非同时峰值相加称作实测总峰值。

### 原生 GPU 传输

在 lxplus901 的 Tesla T4 上取每类前 16 行，共 64 行。只传输 observations 的全部 nested tensors，包括 ID、位置、bar signal、offsets、测量假设和 masks；targets 留在 CPU。

| Measurement | Value |
| --- | --- |
| Tensor payload | 3,761,808 bytes |
| CPU stage + decode + collate | 2.00684 s |
| Synchronized CPU→GPU transfer | 0.025381 s |
| Incremental peak allocated GPU memory | 3.59229 MiB |
| Exact CPU→GPU→CPU equality | passed, all tensors |
| Cache misses / hits | 4 / 60 |

这是单次未 pin memory 的传输检查，未测量原生深度模型的 forward/backward、持续 GPU feed rate 或总体 GPU 利用率。B0 的 forward/backward 结果继续以阶段 05 为准。

### 自动化与阶段验收

最终 `python -m unittest discover -s tests -v`：15 tests，6.976 s，全部通过，见 `loader_unit_tests.log`。新增测试覆盖真实微型 Parquet 的边界、LRU、保留旧 batch views、物理文件清理、索引错误、行数错配、拒绝共享 scratch、全覆盖与种子、label_status 隔离、pickle，以及 fork/spawn 的父子目录独立与正常退出清理。真实数据的所有属性保留另由既有 `loader_validation.json` 和本次 tensor 往返验证。

| Requirement | Status and evidence |
| --- | --- |
| S0 full inventory, event joins, geometry/schema registry | passed: 7,240 files / 7,237,572 events, stage 04 |
| S0 explicit labels and exceptions | passed: 1,150 exceptions retained/quarantined, stage 04 |
| S0 source-family identity and formal split | pending: actual production generator / macro / source-build records unavailable |
| S0 measured readout vs proxy boundary | recorded: electronics/field calibration unknown=null; current conclusions limited to declared proxy |
| S0 historical study conditions | known values recorded in `05_local_studies.md`; missing run IDs/score/region conditions remain pending for controlled historical comparison |
| S1 native joint geometry, aggregation, conservation, masks, empty events | passed at detector-proxy level, stage 05 and tests |
| S1 dataset, scratch caching, reproducible batching, cleanup | passed on four pilot shards, this report |
| S1 occupancy tails, displays, storage extrapolation | measured: 16 displays, p99/p99.9 and 131.402 GB observation estimate, stages 04–05 |
| S1 small train/validation and resource breakdown | passed: B0 smoke, native transfer, conversion/loader timings; stages 05–06 |
| Formal physics performance / test campaign | not performed; S0 source-family gate remains open |

## 阶段性结论

**事实：**原生 catalog 可以按 worker 私有、有界 shard cache 加载，空事件、所有通道和异常标签状态在本次流程中保留。4-shard 配置的第二个 epoch 不再复制/解码，正常结束后无缓存目录残留。该小型任务的多 worker startup/IPC 开销大于其并行收益；不能默认更多 worker 更快。

**假设与待测：**更大、混类生产 shard 和 S2 网络计算可能改变最优 cache/worker/prefetch 配置；在相同采样和模型预算下再测。当前 bar 和 tracker 仍为明确标注的 detector proxy。生产来源映射没有新增证据，不将精确签名无重复升级为真实 source-family 无交叉证明。

## 下一步计划

1. 取得 10081–10084 实际使用的 generator ROOT/source-entry、`run.mac`、generator/转换/Geant4 版本与构建记录；建立真实家族 key，对全库重核交叉与 split，并说明几何版本差异。验收为每条可用 observation 有可核对的来源、重复模拟家族不跨 split。
2. 在匹配生产的硬件资料中核对 readout/field/clustering；未知项继续保留 null 与 proxy 标签，不把旧汇报 resolution 当标定。此项决定后续硬件性能适用范围。
3. S0 gate 关闭后进入 S2：先冻结 train/validation、native/coarse 与 joint/projection 对照，准备混类 shard；在相同 mixed-batch 条件下测 worker/cache 配置，再训练 ParticleNet/ParT。原生模型性能和全量资源目前未测量。
