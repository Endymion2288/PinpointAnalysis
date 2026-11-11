# 阶段 05：S1 原生通道管线与 B0 开发验证

每类一个完整 ROOT chunk 已通过 HTCondor 转换，共 4,000 个事件；保留原生 pixel 联合位置、合并 bar 通道并保留空事件。10 项自动化检查、真实数据往返检查、16 张分层事件图和一次 GPU 训练已完成。原生 loader 的 batch collation 达到约 1,791–2,196 events/s。当前交付层级为 **detector proxy 开发验证**；生产 source-event 映射和电子学标定尚未确认，正式 S0 split 与硬件物理性能验收仍未关闭。

## 阶段目标

验证不额外粗粒化的通道转换、变长数据加载和四类训练链路。验收包括原生位置、channel aggregation、量纲、守恒、空事件、mask、shuffle invariance、数据与 target 隔离及资源实测。S1 不训练 ParticleNet/ParT，也不进行正式 test campaign。

## 工作内容

- 新增 `analysis.flavour` 独立命名空间，包含 inventory、readout、create_data、datasets、features、train、evaluate 和 submit。没有使用旧 good-event selection、取模标签或 truth-energy tracker hit 选择。
- pixel 用可逆的原生 channel ID 与 float32 中心坐标，二值占用，同 pixel OR，不把 edep 当测量特征。bar 按 `(layer,panel,orientation,bar)` 汇总逐 track 记录，保存有限 transverse 区间、panel z、模拟 edep 和整数 PE。
- tracker 原始 G4 crossing 单独保存；另形成 truth-free 的 0.1 mm station grid + L3 Gaussian resolution/efficiency 假设。其 Gaussian `sigma_xy_mm` 不包含网格最大 0.05 mm 的量化偏差，尚不是总测量协方差。B0 不使用 tracker。
- observations、targets、provenance 分文件。变长 batch 保留 offsets、所有通道属性、response/unknown-state masks，无 token 截断。Arrow buffers 一次转为连续 NumPy arrays，避免逐事件构造大量 Python 对象。
- B0 为 `20→48→24→4` MLP，ReLU，AdamW，learning rate `10⁻³`、weight decay `10⁻²`、seed 42、60 epochs，inverse train-frequency class weights。normalization 仅由 train 拟合，checkpoint 按 validation NLL 选择；calibration/test 未用于训练或选择。

## 详细数据

输入均为 `/eos/project-f/fasersim-bonn/public/pinpoint/data/<run>/<run>_000.root`。运行目录为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11`，以下路径相对此目录。实际环境是 CVMFS LCG_108_cuda、Python 3.12.11、NumPy 2.1.3、Awkward 2.8.1、uproot 5.6.0、PyArrow 20.0.0、PyTorch 2.7.1、CUDA 12.5；没有安装项目或更改 lock。该现成环境与 Poetry lock 的版本并不完全一致。

### 关联、空事件与守恒

| Run | Events | Metadata-consistent | Empty pixel / repaired ID | Empty bar | Empty raw FASER | All-empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10081 | 1,000 | 997 | 194 | 38 | 472 | 21 |
| 10082 | 1,000 | 1,000 | 194 | 46 | 332 | 22 |
| 10083 | 1,000 | 1,000 | 246 | 64 | 404 | 33 |
| 10084 | 1,000 | 1,000 | 249 | 46 | 561 | 30 |

所有非空 pixel 和全部 bar/FASER ID 对齐，event ID 唯一且四个 tree entry 数相同，才允许为上述空 pixel 使用对应 event row。全空 106 个事件仍保留；未施加 response、true fiducial、minimum-node 或触发 selection。三个标签异常作为 ν–e elastic candidates 保留在 targets/异常表中，B0 排除它们的训练标签。原因及源版本限制见 S0 报告与 `provenance_review.json`。

4,000 个事件的 bar PE 合并均精确守恒；float64 合并前后 event edep 最大绝对差为 `5.821×10⁻¹¹ MeV`。输出 float32 channel energy 另按 `rtol=2×10⁻⁶, atol=10⁻⁶ MeV` 检查通过。守恒对象是原有线性模拟响应，没有模拟真实 threshold/window/saturation 的信号损失。

### 转换与存储实测

| Run | ROOT read/decompression (s) | Aggregation (s) | Write (s) | Total incl. inventory (s) | Peak RSS (MiB) | Observation bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10081 | 16.334 | 15.762 | 4.319 | 42.672 | 735.45 | 23,416,692 |
| 10082 | 15.087 | 12.376 | 2.854 | 37.354 | 618.21 | 17,420,777 |
| 10083 | 14.507 | 13.109 | 2.879 | 34.157 | 667.80 | 17,701,315 |
| 10084 | 14.878 | 12.570 | 2.710 | 34.733 | 611.84 | 17,356,458 |

每任务申请 2 CPUs、8 GB memory，在 bigbird20 的 cluster `1573713` 运行。Parquet 当前使用 snappy；LCG 的 PyArrow 未编译 zstd。observation 约 17.4–23.4 kB/event；小型 pilot shard 不足 256 MiB，扩大生产时按实测 event size 合并到 256 MiB–1 GiB。全量体积估算需以完成后的 S0 事件数按类加权，见 S0 总结。

### 原生 occupancy 尾部

分位数由每类 1,000 个完整事件计算，包含空事件；是本次 chunk 的经验分位数。

| Run | Pixel p99 / p99.9 / max | Merged bar p99 / p99.9 / max | Digitized tracker p99 / p99.9 / max |
| --- | --- | --- | --- |
| 10081 | 7121.17 / 14850.414 / 16263 | 2948.30 / 3105.256 / 3361 | 123.01 / 268.016 / 284 |
| 10082 | 3783.32 / 6455.701 / 8155 | 2917.55 / 3409.147 / 3556 | 95.03 / 159.031 / 190 |
| 10083 | 2733.40 / 3972.695 / 4667 | 2835.08 / 3162.241 / 3403 | 52.26 / 188.023 / 211 |
| 10084 | 4468.02 / 6757.385 / 8141 | 2950.14 / 3381.089 / 3470 | 74.02 / 136.284 / 420 |

pixel 最大值已高于原先每类 32 事件抽样的上限，支持在模型阶段使用完整变长事件及测量驱动的 token budget。全库 raw record 尾部另由 S0 盘点给出；不能将这里的 bar/track proxy 尾部外推为全库保证。

### Loader、图与通道映射

| Run | Shard load/buffer preparation (s) | Collate 1,000 events (s) | Collate events/s | Total native/proxy tokens |
| --- | ---: | ---: | ---: | ---: |
| 10081 | 0.720 | 0.455 | 2195.59 | 1,380,221 |
| 10082 | 0.567 | 0.556 | 1797.27 | 959,079 |
| 10083 | 0.460 | 0.558 | 1791.30 | 969,056 |
| 10084 | 0.676 | 0.502 | 1993.32 | 954,940 |

以上是 CPU collation，batch size 16、0 data workers，不包含深度模型 forward。旧逐行 Python 列表实现测得 42.4–68.5 events/s；改为连续 buffer 后本轮达到上表范围，worker/机器与缓存状态可能不同，不能仅用两轮比值当严格硬件加速实验。诊断任务峰值 RSS 1,632.36 MiB，包含 event display 的 Python 副本；常规 loader 不构造这些显示副本。

`diagnostics/` 有每类 occupancy 最低、25%、50%、95% 的四张图，共 16 张，并保留 `occupancy_signal.png`、`cutflow.png`、`summary.json` 和 `channel_maps.json`。四类均观察到全部 96 个 bar panels 的 layer/panel/orientation 关联；这是数据内的经验 map，未替代生产几何 crossing 验证。原生 pixel 维度为 `[15,8596,12788]`，只保存激活通道，未展开 dense tensor。

### B0 GPU 验证

| Measurement | Value |
| --- | --- |
| Training events by class | 709 / 678 / 684 / 721，合计 2,792 |
| Validation events by class | 100 / 94 / 101 / 97，合计 392 |
| Balanced accuracy / macro F1 | 0.38723 / 0.38743 |
| NLL / multiclass Brier | 1.31301 / 0.70894 |
| ROC-AUC, νe / νμ / ντ / NC | 0.72670 / 0.62247 / 0.57383 / 0.65338 |
| PR-AUC (average precision), νe / νμ / ντ / NC | 0.50859 / 0.34186 / 0.33103 / 0.36590 |
| Feature load / device transfer | 0.925 s / 0.462 s |
| Training / accumulated forward / backward | 6.926 s / 2.280 s / 3.038 s |
| GPU / PyTorch allocated peak | Tesla T4 / 17.687 MiB |
| CPU peak RSS | 984.34 MiB |

指标按 validation 的未加权样本混合计算，包含空事件，属于开发 smoke baseline。392 个 validation 事件不能支撑 `10⁻³` 背景接受率的工作点结论。正式源身份未验证，也没有 detector/generator systematic 或物理 priors/exposure；不由这些数值推导最终 τ rejection、purity 或产额。

`b0/gpu_usage.csv` 以 100 ms 采样，184 个 device-global samples 的 median/max utilization 均为 100%，device-global peak memory 13,575 MiB；它们包含同卡其他活动和初始化，不能归因于 B0。模型自身 allocator peak 使用上表 PyTorch 测量。计时启用了 CUDA synchronization，适用于本次 profile，不代表异步优化后的极限吞吐。

### 命令、验证和产物

复现命令见 [S0/S1 运行入口](../roadmap/06_s0_s1_operations.md)。主要产物为 `shards/<run>/` 的 observations、targets、provenance、ideal tracker、B0 feature Parquet 与 summary；`b0/` 中有 checkpoint、validation predictions、metrics 与 GPU sampling。没有将大型结果加入 Git。

`python -m unittest discover -s tests -v` 的 10 项检查涵盖像素 joint geometry/逆映射、无效 index、bar 能量/PE/区间/顺序、空事件、tracker 无 truth 合并与顺序、空 pixel join、源签名、分组比例与种子、ragged offsets/dtypes/target isolation。`loader_validation.json` 另记录真实 rows 0、1、12、39 的所有通道属性往返、整数 ID 与 mask shape 检查。

## 阶段性结论

**事实：**原生联合位置和硬件 bar 级响应可以压缩到紧凑的变长格式；已见的空 pixel ID 问题可在逐文件完整检查后修复。bar 逐 track 信息无需作为输入，合并后的信号在指定容差内守恒。连续 buffer loader 已消除本轮明显的 Python 对象转换瓶颈，四类训练链路可运行。

**假设及待验收项：**bar PE/energy 仍是线性模拟 proxy，tracker 为未匹配生产的简化测量假设；正式读出、field、聚类、fake associations 和总协方差未测量。源签名分组仅支持开发，生产 source-family 合同未确认。因此本报告完成的是 S1 开发层级交付，不将 S0/S1 的正式物理验收标为全部完成。

## 下一步计划

1. 用 S0 全库结果补齐实际事件总数、跨文件签名重复、NC flavour 覆盖和全量存储估算；确认 shard 与最终 manifest 的 split 一致。
2. 获取生产 generator/run.mac 与硬件读出条件，将 provisional source-signature split 转成可核对的 source-family split；按对应版本验证 geometry/readout。
3. 完成 S0 gate 后再进入 S2：固定 split、读出和模态，比较 native/1 mm × joint/projections，训练 ParticleNet/ParT，保留全空事件与相同 denominator。
