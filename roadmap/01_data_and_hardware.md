# 数据证据、硬件响应与输入合同

当前合同：S0/S1 已由用户确认通过；现有 samples、冻结 split 与 detector-proxy 数值直接接受，不再要求来源认证或真实硬件标定。见[阶段 07](../workbook/2026-11-11_stage07_s0_s1_acceptance.md)。以下保留原始数据审计和设计说明，其追溯条款已由该决定覆盖。

本文件区分三类信息：2026-11-11 的实际抽样结果、当前模拟源码的行为、待验证的硬件响应假设。抽样只用于确定研究起点，不估计全样本分类能力。

**S0 全量更新：**随后已通过 HTCondor 读取全部 7,240 文件的相关 metadata、geometry/schema、事件 ID 与 detector count vectors，确认 7,237,572 个事件、1,566,540 个空 pixel ID 修复、189,009 个全空事件。1,150 条标签例外已列入异常表；NC 实际包含 4,267 个 ντ 与 1,940 个 anti-ντ。所有权重为 1，单一相关 schema/geometry，无重复精确元数据签名；真实 source-family mapping 仍未验证。详细分 run 表、非有限 W、异常过程、开发 split 和验收状态见 [S0 报告](../workbook/2026-11-11_stage04_flavour_inventory.md)。下文首轮抽样数字保留原有统计范围，不代表最终全库结论。

## 1. 实测数据范围

输入根目录：`/eos/project-f/fasersim-bonn/public/pinpoint/data`。遍历四个 run 目录的文件名和文件大小；按 chunk 数值排序，每类打开首/中/末三个 ROOT 的 header 与 geometry。每类 chunk 000 读取 1,000 条 scalar event 元数据、前 32 条 detector hit entries。未批量转换数据。

| Run | Nominal label | Files | Size (bytes) | Size (GiB) | Chunk range | Last-file event entries |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 10081 | CC νe | 874 | 539,664,893,970 | 502.602 | 000–873 | 65 |
| 10082 | CC νμ | 4,158 | 1,202,843,268,358 | 1,120.235 | 000–4157 | 413 |
| 10083 | CC ντ | 653 | 228,950,146,446 | 213.226 | 000–652 | 231 |
| 10084 | NC | 1,555 | 414,556,703,970 | 386.086 | 000–1592 | 863 |

合计 7,240 文件、2,386,015,012,744 bytes，即约 2.386 TB / 2.170 TiB。10084 有 38 个缺失 chunk 编号；原因未测量。其他三个 run 在上述编号区间内无缺口。总事件数未测量，不能假设每个文件均为 1,000 事件。

抽样的 12 个 ROOT 中，`event`、pixel、Scintillator、FASER tree entry 数分别相等。每个 run 的首/中/末三份 geometry 完全相同；这不替代全量 geometry/schema 盘点。`Hits/scintHits`、`Hits/scintillatorPixelHits` 存在多个 TTree cycle，例如 `10081_000.root` 中 `scintHits;12` 为 1,000 entries、`;11` 为 968 entries，应明确使用最新 cycle，不能累加所有 cycle。

## 2. 样本中已确认的事件与标签问题

### 2.1 空 pixel 事件的 ID

| Run | Hit entries sampled | Empty pixel entries | Pixel ID disagreements | Empty scintillator entries | Empty FASER entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10081 | 32 | 6 | 6 | 3 | 15 |
| 10082 | 32 | 7 | 7 | 1 | 9 |
| 10083 | 32 | 11 | 11 | 6 | 16 |
| 10084 | 32 | 8 | 8 | 1 | 18 |

全部已见 pixel ID 不一致发生在空 pixel entry；Scintillator 和 FASER 的这 128 条 entry 没有 ID 不一致。例如 `10081_000` 的 event 1 是空 pixel，但 pixel tree 中 ID 为 0；events 22–24 的空 pixel ID 均停留在 21。

当前源码 `Pinpoint/src/AnalysisManager.cc::FillHitsOutput` 在 hit 循环内赋值 `fPixelEventID = evtID`，空 collection 仍 `Fill()`；这是与观测一致的程序原因。当前源码并不等同于生产版本，因此转换器须先在每文件验证 event entries 唯一性、entry 数、所有非空 pixel 的 ID 对齐，再允许按对应 row 为**空 entry**建立事件键；不将这个修复泛化给非空错配。后续模拟应在进入 hit 循环前写事件 ID，并用一个带空事件的小样本验证。

主事件键为 `production_id + source_file/chunk + evtID`；`vtxID` 仅作为该事件内 vertex target 的子键。若一个事件含多个 vertex，detector 响应属于完整事件，不能复制成多个独立训练事件。raw source entry ordinal 也应保留，以复核空事件关联。

### 2.2 标签冲突

每类 chunk 000 的 1,000 条 metadata 中，10082 的 `intType=2, |fslPDG|=13`、10083 的 `intType=2, |fslPDG|=15` 与名义类别一致；10084 的 `intType=3` 且 `fslPDG=initPDG`。10081 有以下例外：

| File | evtID | initPDG | fslPDG | intType | processName |
| --- | ---: | ---: | ---: | ---: | --- |
| 10081_000.root | 121 | 12 | 11 | 0 | empty |
| 10081_000.root | 609 | 14 | 11 | 3 | WeakNC |
| 10081_000.root | 706 | 12 | 11 | 0 | empty |

这三条的原因尚未确定。S0 要追溯 generator 输入、`isCC/isNC`、反应类型编码及末态列表，分别记录 metadata 错误、目录混入或非目标过程；先隔离其标签决策，正常数据流继续。主结果和使用目录标签的敏感性结果都记录这些事件的处理及分母，不静默改标签。

在上述 4,000 条 scalar 元数据中 `weight=1`。当前 `GFaserGenerator.cc` 也写死 `metadata.weight=1.0`。这不能给出 luminosity 或真实 flavour priors。10084 抽样的入射 flavour 为 `±12, ±14`，是否包含 `ντ NC` 尚需全量盘点。

## 3. 原生占用与规模证据

以下分位数包含空事件，每类仅取 chunk 000 的首 32 事件。Scintillator channel 是 `(layerID,panelID,isHorizontal,rowID,colID)` 的唯一组合；这是用于估计通道合并收益的定义，正式通道键仍需几何映射核查。

| Run | Pixel hits median / max | Scint records median / max | Unique bars median / max | Records / bars (pooled) |
| --- | --- | --- | --- | ---: |
| 10081 | 303.5 / 4,960 | 10,076 / 174,535 | 588.5 / 3,008 | 34.45 |
| 10082 | 16.5 / 619 | 657.5 / 43,857 | 264.5 / 2,972 | 9.26 |
| 10083 | 9 / 1,947 | 5,284.5 / 100,232 | 253 / 2,542 | 26.33 |
| 10084 | 26 / 6,056 | 2,662 / 111,021 | 623.5 / 2,576 | 18.53 |

样本中的 `nPE` 有零值、无负值；这不表示所有记录都是有效触发。`hitTime` 最大值分别为约 105,864 / 26,347 / 106,229 / 119,931 ns。晚到信号须经过真实电子学窗口和 threshold 处理，不能假设全部落在实验读出窗口内。pixel `edep` 最小值约 `3.63×10⁻⁴ MeV`，当前源码有约 100 electron-hole pairs（360 eV）的总像素阈值；生产阈值仍需确认。

S0/S1 要扩展为覆盖各 chunk 区间、能量、vertex 区域与 τ 衰变道的分层抽样，再由批处理统计 p50/p90/p95/p99/p99.9、空事件、active fraction、各模态缺失率。所有 sampler/token budget 的选择依据这些测量，不能依据本表外推高占用尾部。

## 4. 几何与版本

已检查的 geometry：pixel 活性尺寸 `266×196 mm²`，`12,788×8,596` transverse pixels，pitch 由中心差得到 `20.8×22.8 μm²`，silicon thickness `10 μm`。15 个 pixel z planes 范围为 `−2981.995` 至 `−1674.855 mm`；`nLayers=15`，但 `layer_is_pixel` 长度为 22，这是混合模块序列，不能作为同一个索引空间。

Scintillator 尺寸 `420×420 mm²`，bar 活性宽度 `10 mm`、中心 pitch `10.5 mm`，各方向 40 bars；`scint_Zpos` 有 96 项。pixel 的 y 中心范围约 `−109.989` 至 `85.977 mm`，Scintillator transverse offset 也不为零。pixel plane、mixed module、scint layer、panel 是不同层级，必须输出显式的 `(detector, module, plane/panel, channel) → physical geometry` 表。

实际可用模拟目录是 `/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4`；本次该 Git 工作树干净，HEAD `dbbdb5155a66f5329fd27ea5285d037bf8fb2f03`。用户提到的 `PinpointAnalysis/Pinpoint_G4` 不存在。当前源码默认 pixel 尺寸 `250×200 mm²`，与样本不同，且 HEAD 包括 Scintillator bar 间材料改动。因此默认 macro、HEAD 或旧 `geo.pkl` 都不能替代生产 ROOT 的 geometry。当前 pixel 源码还有 world→sensor-local 索引修正，Scintillator 使用专门 panel/layer lookup；是否已用于生产样本尚未追溯。S0 用相匹配版本的单粒子 crossing/hit map 核对坐标和通道映射，再决定是否需要处理旧响应差异，不静默平移现存 hits。

S0 需补齐：生产 commit、宏、field map/强度、generator 文件标识与版本、physics list、run/chunk 与 generator entry 的关系、随机种子。新模拟复用匹配配置的现成构建，仅源码变化时编译。为几何版本分配人工可读 ID，并保存结构化参数；不逐个大文件做 SHA 扫描。

## 5. ROOT 字段到物理输入的分级

| Source | Candidate inference inputs | Truth/target only | Required handling |
| --- | --- | --- | --- |
| `Hits/pixelHits` | `rowID,colID,layerID`；binary occupancy；仅在真实读出允许时使用 charge/ToT 代理 | `pdg,trackID,parentID,fromPrimaryLepton,fromPrimaryEMShower,fromCharmedHadron,fromTau` | 用 geometry 映射到中心；同 pixel 合并；`edep` 是模拟沉积，不自动等于测量 charge |
| `Hits/scintHits` | bar/panel/orientation；经 channel 合并与电子学处理的 PE/ADC、时间、valid mask | `trackID,parentID,pdg,fromMuon,fromPrimary* ,fromTau`；per-track 分解条数 | 按硬件通道和真实积分窗口合并；零 PE、不触发和坏道分别编码 |
| `Hits/scintillatorPixelHits` | 本身不增加独立硬件测量 | track 贡献与可能的 truth 监督 | 当前源码未设置细分 `scintPixelID`，不能假定它是真实三维空间点或第二份独立探测器数据 |
| `Hits/faserHits` | 经分辨率/效率模型转换的 station hit/cluster 位置；由这些测量拟合出的轨迹、协方差、χ²、可重建时的 `q/p` | G4 `px,py,pz,energy,charge,pdg,trackID` | 连续位置是理想 G4 step crossing；需重建测量，不直接使用真动量或 per-track hit 去重 |
| `event` | 无真值字段直接进入推理 | `initPDG,fslPDG,intType,initE,initX/Y/Z,Q2,xBj,y,W,tgt*,weight` | 标签核查、权重、分层评价和辅助 target |
| `primaries,tau,charm,trajectories` | 仅其可从 detector hits 独立重建的替代量 | 末态/衰变/生产顶点、动量和 ancestry | 生成 target、分支与可见性分析；不作为特征 |
| `geometry` + conditions | detector type、层位置、边界、分辨率、材料厚度/辐射长度、已知 field/校准 | 无 event truth | 只能来自探测器描述与 conditions，不能用真实 interaction vertex 对齐输入 |

`run/chunk/file/evtID`、数据顺序、文件大小、padding 的 class 编码均不作为模型特征。真实模态缺失与空 hit 是测量状态，应保留，不能通过只训练有 pixel 的事件改变任务。

## 6. 通道重建与响应升级

**Pixel：**当前 `PixelSD.cc::AddHit` 用 `layer,row,col` 聚合沉积，已处于原生 pixel 粒度；细到像素内部的 G4 step 坐标不是硬件测量。首先保留全部原生像素中心和 binary occupancy；若硬件只提供 binary，则以此作为主结果。`edep→electron-hole charge→threshold/noise/dead channel/saturation/charge sharing` 只在有依据的 digitizer 中使用，模拟 analog 作为独立信息上限。不能把保存的 `edep` 直接命名为实验 measured energy。

**Scintillator：**当前 `ScintSD.cc` 按 `(layer,panel,orientation,bar,track)` 累积。将 track 拆分合并成实际 bar/pulse 再输入模型，禁止输入同 bar 的 G4 track 数。代码采用 `10,000 photons/MeV`、衰减长度 `380 cm`、transport efficiency `0.10`、PDE `0.40`、group velocity `c/1.58`、decay constant `2.1 ns` 的快速解析模型；transport/PDE 的标定在代码中仍有 TODO。独立 Poisson PE 在理想线性无饱和条件下可相加；每 track 的“最早时间+一次指数延时”不等于真实 channel waveform/leading-edge time，不能简单取最小值就宣称完成电子学模拟。

S1 先做 bar-energy 与无 timing 的 detector proxy；主硬件版本要验证 channel-level PE/ADC、阈值、积分窗、SiPM 饱和、cross-talk/afterpulse、噪声、time walk、time resolution 和 event time reference。若缺少生成 pulse 所需 step 信息，以专用小样本新输出校准，不从 per-track 时标反推不存在的波形。双端读出、time-derived longitudinal position 只有在硬件方案确认后加入；现有源码是一端读出，不假设两端已存在。

**FASER：**先把 charged-step crossing 转为 station/strip 测量，验证多 hit 合并、finite resolution、inefficiency、acceptance、alignment 和场内轨迹拟合。轨迹关联必须来自测量，`trackID` 只用于事后评估。三层理想 crossing 并不保证能获得可校准动量。无法可靠重建时，先用 station hit sets 与有效性 mask，另列 ideal-position 结果；不要挑选“真能量最高 hit”代表每层。

更新的 2026-10-21 汇报提供一个简化重建参照：`σx=0.58 mm、σy=17 μm`、99% hit efficiency、无 noise、0.1 mm step 合并、3 mm 直线相容判断及 2.4 TeV momentum cap。其 field/坐标与当前生产匹配关系尚未确认，先作为待验证假设；详细条件见 [L3 核对](05_local_studies.md)。新转换必须分别记录出接受度、无测量、拟合失败和高动量不确定/截顶状态，不把 capped p 当作精确测量。

**Bar view 歧义：**一条 bar 只测一个 transverse 坐标、有限的另一方向范围与所在平面，保留 orientation、bar extent 和测量协方差；不随意填一个精确的二维点。可选 H/V compatibility graph 或 learned cross-view association，保留 multiple hypotheses 和 fake rate；只有显式构造空间点时才定义与 FASERCal ghost 类似的匹配标签。

## 7. 数据格式、split 与完整性合同

转换输出分三部分：`observations`（推理白名单）、`targets`（event/soft semantic/kinematic labels）、`provenance`（事件键、schema/geometry/conditions/split）。模型 forward 只接收第一部分；训练 loss 才可读取第二部分。pixel 用 compact channel IDs、float32 坐标或 geometry lookup；Scintillator 按 channel/pulse 输出；变长事件用 offsets，另有 validity 与 detector-presence mask。训练集统计量拟合 normalization，验证和测试只复用。

建议 tensor 合同为 pixel coordinates `[Np,3]`（mm）、pixel features `[Np,Fp]`、bar features `[Ns,Fs]` 与区间/协方差、tracker observations `[Nt,Ft]`，各自配长度 `[B+1]` offsets 与 `[B,3]` modality-state mask；输出 calibrated probabilities `[B,4]`。全空模态以明确的 no-response token/mask 处理，不能对全 masked attention 行产生 NaN；label `[B]`、semantic fractions 与 kinematics 均留在独立 target 结构。

先按 generator source event/重模拟家族分组，再在各类与 production 条件内做固定 `70/10/10/10` 的 train/validation/calibration/test 分配；同一物理事件的噪声变体、不同 geometry/digitization、τ enrichment 复制必须同组。chunk 可能只是切片，不能证明事件独立；缺少 source ID 时先追溯生产映射。随机种子 `17,42,2026` 用于训练重复，split 种子固定为 `20261111`，落盘 manifest 后所有表示和模型共用。energy/vertex matching 只改变诊断权重或训练子集，不能重新随机划分测试集。

转换验收包括：每个有效 source event 恰好一条 observation；空事件保留；pixel/channel ID 有效且量纲正确；按通道合并前后 `edep` 守恒（float 容差随累加尺度声明）；理想线性合并 PE 守恒；窗口/threshold 造成的信号损失单列。高占用事件不按类别截断。所有 cutflow 包括标签未决、无响应、重建失败、触发与 fiducial selection；truth fiducial 子样本只能作为明确的条件效率。

另为 historical-study 诊断定义 pixel-centered 与 large-scintillator volume，记录三维边界、坐标原点、target/material 与 source config。2026-09-16 汇报的 2 cm/前 50 cm 和 1.5 cm/下游 20 cm 是待映射到当前几何的选择参照，2026-10-07 的 small/large 不能未经配置核对就视为同一选择。主 split 仍覆盖完整声明目标域，这些 truth-domain 标记仅用于条件评价；不作为推理输入或默认 selection。

## 8. 本次证据的复现

```bash
python3 roadmap/tools/inspect_inputs.py \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_2026-11-11/input_audit.json
```

实测环境为 Python 3.9.25、NumPy 1.23.5、uproot 5.6.9、awkward 2.8.12，使用现成工具做只读盘点，未安装项目。生产开发仍按项目 Python 3.12 合同选匹配 CVMFS 环境。本次未使用 GPU、未提交 HTCondor，也未运行旧预处理器。

完整小型审计 JSON、文献抓取的最终摘要存于 `/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_2026-11-11/`。图、模型指标、全量事件数、production commit 和真实电子学标定均为未测量/待追溯项目。
