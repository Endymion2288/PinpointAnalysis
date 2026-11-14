# 阶段 04：S0 全库盘点与开发数据合同

当前状态更新：用户已确认 samples 满足验收要求并接受现有 detector-proxy 参数，S0/S1 已完成，来源认证和硬件标定不再阻塞。详见[阶段 07](2026-11-11_stage07_s0_s1_acceptance.md)。下文保留本阶段当时的测量与判断。

四个 run 的 **7,240 个 ROOT、7,237,572 个事件**已全部盘点，输入逻辑大小 2,386,015,012,744 bytes；全部文件的 event/hit entry 关联通过，相关 schema 与 geometry 各只有一个版本。修复空 pixel 陈旧 ID 1,566,540 条；保留全空事件 189,009 个。全库发现 1,150 条标签例外，已隔离；NC 中包含 6,207 个 ντ/反ντ 事件。开发 split 已落盘且无重复签名交叉，**真实生产 source-family 映射尚未取得，S0 正式验收仍有一项来源 gate 未关闭**。

## 阶段目标

逐文件确认数据规模、标签、事件身份、空响应、schema/geometry、原生 occupancy 与输入白名单；冻结所有后续表示共用的开发 manifest，明确其与正式物理合同的差别。源版本、硬件条件、历史配置和无法确认的标签问题分别保留状态，正常事件继续处理。

## 工作内容

1. `analysis.flavour.inventory` 生成只含既有文件的计划。先用 HTCondor 处理每类一个完整 chunk，全部关联通过后扩展到 24 个任务、每任务至多 320 个文件；只读 scalar event metadata、geometry/schema、事件 ID 与每探测器一个计数向量，不读取完整 truth hit payload。
2. 每文件检查 metadata event ID 唯一、tree entry 数相等、全部非空 pixel ID 及全部 bar/FASER ID 对齐。只有空 pixel 允许按 event row 关联。没有将 hit tree 的陈旧 ID 用作 join key。
3. 核对显式 run→class、CC 入射 flavour/direct lepton/符号与 NC 出射中微子身份，保留 label exceptions。用 generator 元数据的 128-bit 精确签名做保守重复检查，并按签名加固定 seed 分配 `70/10/10/10`。
4. 追溯本地/共享模拟源、旧 generator 与宏线索。完整扫描已找到的四个旧 gFaser 文件共 4,229,300 条 scalar entries，与当前 4,000 个 pilot 事件做全字段签名匹配；未找到匹配。未将旧文件的 source IDs/exposure 当作当前生产信息。
5. 冻结开发读出合同和运行说明，完成原生通道 pilot；历史 PDF 的已知条件和缺失配置按 [本地汇报核对](../roadmap/archive/2026-11-14_v1/05_local_studies.md)记录。详细 S1 结果见 [阶段 05](2026-11-11_stage05_native_channel_pipeline.md)。

## 详细数据

输入根目录为 `/eos/project-f/fasersim-bonn/public/pinpoint/data`。所有运行结果位于 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11`，下文文件路径均相对此目录。

### 全库计数与关联

| Run / class | Files | Actual events | Metadata-consistent | Label exceptions | Empty pixel | Empty bar | Empty raw FASER | All-empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10081 / CC νe | 874 | 873,065 | 872,092 | 973 | 178,911 | 41,404 | 409,785 | 23,120 |
| 10082 / CC νμ | 4,158 | 4,157,413 | 4,157,236 | 177 | 870,213 | 178,320 | 1,375,893 | 99,865 |
| 10083 / CC ντ | 653 | 652,231 | 652,231 | 0 | 164,956 | 39,251 | 268,463 | 22,964 |
| 10084 / NC | 1,555 | 1,554,863 | 1,554,863 | 0 | 352,460 | 67,037 | 849,067 | 43,060 |
| Total | 7,240 | 7,237,572 | 7,236,422 | 1,150 | 1,566,540 | 326,012 | 2,903,208 | 189,009 |

7,237,572 条 metadata rows 与逐文件唯一 event keys 数相等，无文件内重复 ID；每个 run 内 evtID 也无重复。所有文件的 evtID 为连续区间，起点等于 `chunk×1000`，区间长度等于实际 entry 数。尾文件分别为 65/413/231/863 事件。NC 的 38 个缺失 chunk 造成 38,000 个编号缺口，完整编号见 `inventory/summary.json`；缺失原因未确认。其他 run 没有 chunk 编号缺口。

所有非空 pixel、全部 bar 与 FASER 的 ID 对齐；恰好全部空 pixel ID 都是陈旧值。逐文件修复条件全部满足，修复不会生成假 hit。全空/缺失模态仍在样本分母中，不施加 hit 数、触发或 true-fiducial selection。

相关 `event` 和三个 detector trees 的 schema ID 为 `5f250ecf6150db74fc3826f8`，ROOT geometry ID 为 `60c5428955f2890e1642ab91`，覆盖全部 7,240 文件。geometry 是 ROOT 参数内容的短标识，不是生产代码版本或大文件 checksum。辅助 particle/decay trees 不做 event-row join，本轮没有将其全库转换为对象标签。

### 标签例外与明确处理

| Directory | Machine status | Numeric metadata | Count | Primary training decision |
| --- | --- | --- | ---: | --- |
| 10081 | `nu_e_elastic_candidate` | `intType=0, scatteringType=7, initPDG=±12, fslPDG=11` | 498 | 保留原始元数据，排除主 CC νe 标签 |
| 10081 | `nu_e_elastic_candidate` | `intType=3, scatteringType=7, initPDG=±14 or −16, fslPDG=11` | 475 | 保留例外，未静默改为 NC |
| 10082 | `unresolved_metadata` | `intType=2, scatteringType=14, initPDG=−12, fslPDG=13` | 177 | IMD-annihilation candidate，排除主 CC νμ 标签 |

当前 converter 的 `ScatteringType` 枚举将 7 映射为 `NuElectronElastic`、14 为 `IMDAnnihilation`。νe chunk 000 的 evt 121/609/706 的 primaries 均包含电子、同 flavour 出射中微子及钨核，支持 ν–e scattering 解释。新检查的 `10082_010.root` evt 10631 primaries 为 `[1000741860,13,−14]`，入射 `−12`，支持 IMD-annihilation 解释。证据在 `provenance_review.json` 和 `imd_example.json`。

这些是四类标签定义的例外，不能统称为损坏文件。177 条记录保持机器状态名 `unresolved_metadata`，其候选物理解释和处理决定由本表补充；未根据目录强行授予 CC νμ 标签。10082 另有 3,305 条 `scatteringType=8`，在当前 converter 对应 inverse muon decay，仍满足现有 νμ/direct-μ 标签协议；核子与电子靶过程的最终任务边界需结合真实生产输入明确。

所检查的 `GFaserGenerator::EncodeProcessName` 将 DIS/νELASTIC 编为 4/8，而 converter 对应 3/7，因此不以字符串 `processName` 作为过程的权威解码。生产 converter/宏尚未确认，候选解释均保存数值证据。旧 lepton-PDG filter 未显式要求 CC，可以解释此类混入机制，但尚不能证明就是当前生产所用脚本。

全部 event weights 均为 1，没有 exposure/flux/cross-section normalization。`W` 有 1,296 条非有限值，分 run 为 145/756/117/278；其余所检查 scalar fields 无非有限值。W 不进入 B0 observation；未因这一 truth-only 数值问题剔除四类事件。后续若回归 W，必须显式使用 finite/physical-validity target mask；成因待源 generator 核对。

### Flavour 与反中微子覆盖

| Class | Positive incoming PDG | Negative incoming PDG | Notes |
| --- | ---: | ---: | --- |
| Consistent CC νe | νe 576,312 | anti-νe 295,780 | 不包含上述 973 条例外 |
| Consistent CC νμ | νμ 3,286,266 | anti-νμ 870,970 | 不包含 177 条 anti-νe candidates |
| Consistent CC ντ | ντ 481,882 | anti-ντ 170,349 | 全部 metadata 一致 |
| NC, e flavour | νe 162,439 | anti-νe 102,926 | 两符号均覆盖 |
| NC, μ flavour | νμ 970,477 | anti-νμ 312,814 | 两符号均覆盖 |
| NC, τ flavour | ντ 4,267 | anti-ντ 1,940 | 合计 **6,207**，首 chunk 抽样未观察到 |

当前 NC 包含全部三种 flavour；应使用全库结论替代首 1,000 事件中未见 ντ 的抽样结果。各 flavour 数量不是物理 priors，τ NC 的分层指标仍需按实际有限样本给出区间。

### 全库 occupancy 与范围

以下为 raw detector records，尚未把 scintillator 的逐 track 记录合并成 bar；实际 channel 分位数见 S1 报告。

| Run | Pixel p99 / p99.9 / max | Scint raw records p99 / p99.9 / max | FASER raw steps p99 / p99.9 / max |
| --- | --- | --- | --- |
| 10081 | 7653 / 16054.872 / 46154 | 145168.8 / 210777.872 / 291322 | 164 / 573.936 / 2040 |
| 10082 | 3831 / 8468 / 40638 | 103911 / 179719.292 / 336236 | 70 / 222 / 1428 |
| 10083 | 3280 / 6508.310 / 19483 | 87640.8 / 114120.620 / 151671 | 81 / 250.770 / 1289 |
| 10084 | 3606 / 8191 / 44314 | 97821.52 / 173049.484 / 332418 | 65 / 208 / 1479 |

pixel 最大 46,154，明显高于最初 32-event 抽样的 6,056；不能据小样本设置会丢 hit 的硬 token cap。ντ 的 incident energy 上限为 2.3118 TeV，其余三个目录为 5.0412 TeV；x/y 分布也存在样本间差异。能谱/位置先验与可见拓扑的贡献需在 S2/S5 用共同 support 的匹配诊断区分，不能由当前分布直接宣称分类增益来自衰变结构。

`inventory/data_ranges.png` 使用各类共用 bin edges，给出 energy、true vertex 与 raw occupancy 分布；坐标和图注均为英文。完整 min/p50/p90/p95/p99/p99.9/max 在 `inventory/summary.json`，event cutflow 在 `inventory/cutflow.csv`。

### 开发 split 与来源边界

| Partition | Consistent-label events |
| --- | ---: |
| Train | 5,066,296 |
| Validation | 723,413 |
| Calibration | 722,272 |
| Test | 724,441 |

签名使用 `[initPDG,initPx,initPy,initPz,initE,tgtPDG,hitnucPDG,Q2,xBj,y,W,scatteringType,fslPDG]`，排除重新随机化的 vertex 和 run/file/row。精确的高维签名未发现重复；同签名跨 run/跨 split 数均为 0。固定 seed `20261111`，BLAKE2b-128 + SplitMix64 后按比例分组，类内比例由稳定 hash 得到。每条 observation 可通过 `(本 inventory,run,chunk,row,evtID)` 追溯到 ROOT；7,240 个文件的路径/大小/schema/geometry/关联结果在 `inventory/files.json`。

**没有把这项检查等同于真实 generator 家族已验证。**当前尚缺实际 generator 文件、原始 entry/iev、生产 macro/commit 与重模拟关系。元数据在不同 generator/转换版本变化时，精确签名不能替代 source IDs；非有限 W 的原始位模式也属于该代理签名。manifest 标为 `provisional_signature_grouped_development_only`，正式 test 使用保持关闭。

全库 manifest 与 pilot 的 4,000 条 identity、signature、label status 和 split 已逐条一致性检查通过；S1 的开发训练分组可以追溯到本次完整 manifest。校准/test 事件没有用于 B0 训练、选择或预训练。

### 模拟、硬件与历史配置

- 已检查本地 source HEAD `dbbdb5155a66f5329fd27ea5285d037bf8fb2f03`；共享旧 source HEAD `fd8c6d3` 且工作树有改动。均未证明为这四个 run 的生产版本，未重新编译或修改模拟源码。
- 每个 ROOT 的 pixel geometry 为 `266×196 mm²`、15 planes，pitch `20.8×22.8 μm²`；当前 source 默认 `250×200 mm²`，因此本次从 ROOT 读 geometry。silicon thickness 数值 10 的单位是 μm；位置使用 mm。
- bar 有 96 panels、每方向 40 bars；S1 四类数据内观察到的每 panel layer/orientation 关联唯一且一致。保留 bar extent，不虚构精确的未测坐标。该经验 map 未替代生产 crossing 测试。
- binary pixel、线性 bar edep/PE proxy、ideal/digitized-hypothesis tracker 分级独立。真实积分窗、threshold、SiPM saturation、timing、field map、tracker reconstruction/covariance 的已标定数值仍为空；这些未确认项不进入“已验证硬件”主结果。
- L1/L2/L3 的 run IDs、L2 small/large 绝对边界、精确 score cuts/class weights 及旧 FASER 与当前生产 field 的对应仍待查。已知 PDF 条件在 `roadmap/05_local_studies.md`，当前全库正常事件不因历史配置缺失而停流。

### 命令、调度、资源与产物

复现入口为 [S0/S1 运行文档](../roadmap/archive/2026-11-14_v1/06_s0_s1_operations.md)。首轮盘点 cluster `1573712`，全库 `1573714`（24 tasks），汇总 `1573719`；原生转换 `1573713`，最终 diagnostics `1573717`，旧 source 匹配 `1573718`，均使用 bigbird20。每轮 submission JSON 保存当时 schedd 负载、WRITE、EOS 路径策略、submit/wrapper 和 ClusterId。bigbird21 的普通队列拒绝直接 EOS 路径，已按实际策略从兼容集合剔除；没有将低 idle 单独当作兼容证据。

全库 scan 每任务申请 2 CPUs，站点 memory floor 为 6,000 MB；任务 wall-time min/median/max 为 47.961/1079.272/1599.088 s，累计 25,858.598 task-seconds，scan 峰值 RSS 最大 590.24 MiB。累计 task time 不是端到端 elapsed。汇总实际处理阶段 71.994 s，峰值 RSS 6,293.00 MiB，申请 8 CPUs、24,576 MB；全过程没有为盘点使用 GPU。输入 2.386 TB 是文件逻辑总大小，实际选列读取字节数未测量。

`condor_s0_resources.json` 确认 24 个 scan 作业均 exit 0、各启动一次；累计 scheduler user/system CPU time 为 4,253/745 s，scheduler wall time 为 26,114 s（包含 wrapper 等额外时间）。最终按 `validation.json` 检查行数、例外、关联、source signature、NC τ、白名单、split、图和文档，通过后合并清理 58 份临时/重复文件，减少 874,683,944 bytes；完整清理记录在 `compaction.json`。

早期 wrapper 曾因 CVMFS setup 的 nounset 与 PYTHONPATH 覆盖失败，PyArrow 也未编译 zstd；已修正并记录到运行说明，最终任务使用 snappy。保留成功任务证据和失败原因，没有将初次失败混作科学结果。所有源码/配置在仓库中；日志、图、manifest 和模型仅在 results。

主要输出为 `inventory/events.parquet`（约 763 MiB）、`files.json`、geometry/schema registry、`label_exceptions.parquet`、`summary.json`、cutflow 与范围图。`inventory_task_statistics.json` 保存任务级计数/资源；`source_match/`、`provenance_review.json`、`imd_example.json` 保存来源核对。重复的逐任务审计/Parquet 在最终完整性检查后合并清理。

## 阶段性结论

**事实：**全库身份与空 pixel 问题已经按逐文件条件解决；明确存在非 canonical-label 过程，不能只用目录赋标签。NC 覆盖 τ flavour，小样本“未见”不是缺失证据。原生 occupancy 尾部比最初抽样更高，变长/局部编码预算必须以全库计数为依据。所有权重为 1，不能由样本数得到物理产额。

**已完成：**全库 inventory、相关 schema/geometry registry、读出白名单与 proxy 分级、标签异常表、开发分组 split、cutflow/范围图以及 S1 开发接口。

**尚未完成的正式验收：**真实 generator source-family 与生产版本核对。当前数据只支持开发协议，不能声明跨 generator/重模拟家族的独立性已经证明。硬件中未标定的响应保持 proxy 级别；没有用假定标定填补条件表。

## 下一步计划

1. 取得 10081–10084 生产实际使用的 `run.mac`、generator ROOT、原始 entry/iev 与生产 source/build 配置，核对每个 run 的 source mapping 与特殊 leptonic processes，冻结正式 family split。
2. 结合硬件 readout/field/标定资料或匹配版本单粒子样本，验证 detector-proxy 条件；未取得条件前维持本报告的分级，不以任意数字代表标定。
3. 正式 S0 gate 通过后进入 S2 的共同数据子集；保留 native/1 mm × joint/projections、完整分母、能量/位置 support 匹配及 τ/NC-flavour 分层。
