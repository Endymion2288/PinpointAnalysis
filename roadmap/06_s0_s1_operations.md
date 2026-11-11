# S0/S1 实现与运行入口

独立实现位于 `src/analysis/flavour/`。本阶段转换原生 pixel/channel 与模拟 detector proxy，B0 用 pixel 和 bar 的 20 个汇总特征。没有调用旧的 good-event selection、run 取模标签或 highest-true-energy tracker hit。运行合同见 [s0_s1.yaml](configs/s0_s1.yaml)，结果与验收状态以对应 workbook 和 results 的实测 JSON 为准。

## 环境与目录

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

使用 CVMFS 的 Python 3.12.11，实际包版本记录在输出。这个环境没有与 Poetry lock 完全匹配；没有安装或升级全局包。LCG setup 依赖其 `PYTHONPATH`，追加项目路径时必须保留它。setup 完成前不要开启 shell nounset。Parquet 根据编译支持选择 zstd/snappy/gzip，当前环境使用 snappy。

结果目录统一为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11`。后文命令中的 `TASK_RESULTS` 指此目录，设置它不会修改系统环境目录。提交文件放在 `~/condor/Pinpoint/flavour_s0_s1/`，日志与数据均写入 results。

## 全库身份盘点

```bash
TASK_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11
python -m analysis.flavour.inventory plan --pilot --files-per-task 1 --output "$TASK_RESULTS/pilot_plan.json"
python -m analysis.flavour.inventory plan --output "$TASK_RESULTS/plan.json"
python -m analysis.flavour.submit pilot --root "$TASK_RESULTS"
# 四个 pilot audit 完成并验证后：
python -m analysis.flavour.submit pilot_merge --root "$TASK_RESULTS"
python -m analysis.flavour.submit inventory --root "$TASK_RESULTS"
# 全部 scan audit 完成且无文件错误后：
python -m analysis.flavour.submit merge --root "$TASK_RESULTS"
```

每次 `submit` 自动读取 schedd 当前负载，查询 `SUBMIT_REQUIREMENT_NoEos`，筛选允许直接 EOS 日志/工作目录且 WRITE 授权的标准 bigbird 队列，优先选择 idle 最少者，记录完整快照、策略、资源、wrapper 与 ClusterId。本次 bigbird20 接受此路径合同，bigbird21 的普通提交拒绝 EOS 路径。全库默认每任务最多 320 个文件，共 24 个任务，限制同时读取 EOS 的进程数；提交器拒绝超过 24 个任务的计划。用 `condor_q -name <recorded-schedd> <ClusterId>` 查看状态，不因轮询超时重复提交。任务失败后按 audit 定位文件/错误，再仅重跑原 task；merge 必须等待完整成功。

盘点读取 scalar event 元数据、最新 geometry/schema、detector IDs 和每模态一个计数向量。输出 event manifest、geometry registry、schema registry、文件审计、标签异常表、重复签名表和范围图。每个文件先验证所有事件 ID 与非空关联，再允许修复空 pixel 的陈旧 ID；非空错配不会自动修复。

最终检查结果保存在 `validation.json`。汇总与校验成功后，逐任务 Parquet/audit 已合并清理；保留 `inventory/`、`pilot_inventory/`、`inventory_task_statistics.json` 与任务日志。`compaction.json` 记录清理范围。复现 merge 前需重新生成 scan parts；只核对已完成产物无需重跑全库任务。

**事件签名与真实源身份分开。**当前缺少生产 source-event 映射，使用高维 generator 元数据的精确签名保守合并重复记录；签名排除重新随机生成的 vertex。源事件在不同 generator/转换版本下改变这些元数据时，精确签名不能建立家族关系。split 标为 `provisional_signature_grouped_development_only`，所有表示共用落盘的 `70/10/10/10` manifest；只有生产映射核对后才能转为正式物理评价合同。签名无 split 交叉不等于真实源家族映射已经证明。

`python -m analysis.flavour.submit source_match --root "$TASK_RESULTS"` 可对已找到的旧 `2026-05-08-genie` 四个 gFaser 文件进行全条目 scalar-signature 匹配，输出 `source_match/source_<run>.json`。该检查不只比较首条记录；保持旧源只读，不读取 particle-vector branches，也不凭匹配候选直接认定其为生产 macro 所用版本。

## 原生转换、数据加载和 B0

```bash
python -m unittest discover -s tests -v
python -m analysis.flavour.create_data \
  --input /eos/project-f/fasersim-bonn/public/pinpoint/data/10081/10081_000.root \
  --events 40 --output "$TASK_RESULTS/smoke/10081"
# 小样本验证通过后，通过 Condor 转换每类一个完整 chunk：
python -m analysis.flavour.submit convert --root "$TASK_RESULTS"
python -m analysis.flavour.submit diagnostics --root "$TASK_RESULTS"
# B0 是 4,000-event 级的小规模验证，可在交互 GPU 执行：
python -m analysis.flavour.train --input "$TASK_RESULTS/shards" \
  --inventory "$TASK_RESULTS/inventory/events.parquet" \
  --output "$TASK_RESULTS/b0" --device cuda --epochs 60
```

`NativeShard` 和 `collate_native` 提供变长 offsets、所有原生通道属性、空模态 mask 与独立 target 字典。`label_status` 保留在 targets 中供调用者隔离异常。模型只接收 observations；provenance 单独保存，B0 显式读取特征白名单，train-only normalization。S1 不打开 calibration/test 来调参。

多 shard 输入用 `CachedNativeDataset`：构造时只读取 Parquet footer，worker 按需将 `observations.parquet` 与 `targets.parquet` 复制到自己的本地临时目录。target 默认只解码 `label/label_status`；不复制 provenance、ideal tracker 和 B0 features。scratch 优先级为显式 `cache_dir`、`_CONDOR_SCRATCH_DIR`、系统 tmp，拒绝 EOS/AFS。LRU 限制每 worker 的本地文件与已加载 shard 数，正常退出/显式 `close()` 清理自身目录；当前 batch 的 NumPy views 和 prefetch queue 可使 RAM 超过 LRU 本身，容量依据 decoded RSS 与 scratch 一同配置。

`ShardBatchSampler` 在 seed+epoch 决定的 shard 窗口内 shuffle，覆盖全部事件且不截断 token。窗口不超过 cache 容量。当前四个 pilot shard 各自只有一个类别，单 shard 窗口用于 I/O profiler；训练需混类 shard，或覆盖四类的窗口，并由冻结 manifest 选择 train/validation。这个 profiler 不做 split selection，扫描 4,000 行只用于数据流检查，不输出 calibration/test 性能。

```bash
# 全部 pilot 的多 worker I/O 测量通过 Condor：
python -m analysis.flavour.submit loader_profile --root "$TASK_RESULTS"
# 最多 64-event 原生 tensor 传输检查，可在交互 GPU 运行：
python -m analysis.flavour.profile_loader --input "$TASK_RESULTS/shards" \
  --output "$TASK_RESULTS/native_transfer.json" --transfer-only
```

`loader_profile.json` 比较单进程/1-shard cache、2-worker/1-shard cache、2-worker/4-shard mixed window，各跑两个 epoch；记录 staging、decode、collate、wall time、cache hits/misses、临时文件容量、per-worker RSS、类别混合和清理。2-worker 使用 spawn、每个 worker 1 个 Arrow/BLAS/Torch 线程、prefetch=1。`native_transfer.json` 仅测量全部原生 observation tensors 的传输与精确往返，不运行新分类网络。

本次第二个 epoch 分别达到 1,704/799/1,164 events/s，多 worker 没有提高小型 pilot 的吞吐；4-shard window 的第二个 epoch 没有新增 staging/decode。完整条件与 S0/S1 逐项验收见[阶段 06 报告](../workbook/2026-11-11_stage06_native_cache_validation.md)。缓存实现验收不改变 source-family split 的 provisional 状态。

pixel ID 可逆到 per-file geometry 中的 `(plane,row,col)` 与中心，保留原生联合位置和 binary occupancy；bar 以 `(layer,panel,orientation,bar)` 合并，保存实测方向与另一方向的有限区间。edep 与 PE 是无饱和/无积分窗的线性模拟 proxy，不声明已标定。时间不进入输入。输出能量 float32 的总和另用 `rtol=2e-6, atol=1e-6 MeV` 检查；PE 使用整数精确合并。

tracker 的连续 G4 crossing 单独存为 ideal proxy；另用 station 内固定 0.1 mm 网格、L3 的 `σx=0.58 mm, σy=0.017 mm, ε=0.99` 形成无 truth 关联的测量假设。网格中心随后加 Gaussian smear；`sigma_xy_mm` 记录该 Gaussian 分量，0.1 mm 网格另有每坐标最多 0.05 mm 的量化偏差，不将这一字段称为总测量协方差。此方案尚未验证当前真实 strip readout、field、noise 与 clustering，也不等同于旧报告的距离式 step merge。未做动量拟合，不将无响应归为出接受度；未知状态为 `-1`。B0 暂不使用 tracker，以免将这套假设的效果混入通道管线验收。

`evaluate` 每类用 pixel occupancy 的最低、25%、50%、95% 四个位置选图，共 16 张；另给 occupancy/signal/cutflow、p99/p99.9、shard bytes/event、loader 时间和 RSS。小样本 shard 允许低于 256 MiB，扩大转换前依据实测 bytes/event 合并到目标大小。资源外推注明每类只转换一个 chunk，不能将其尾部分位数当作全库保证。
