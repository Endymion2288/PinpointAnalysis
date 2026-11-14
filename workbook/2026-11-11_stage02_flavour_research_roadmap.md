# 阶段 02：基于硬件响应的中微子反应分类路线图

已完成 [研究路线图](../roadmap/README.md)、研究配置合同与只读审计工具，共 7 个文件。建议主线为原生 binary pixel、聚合 bar 与 digitized tracker 的分层局部编码和多探测器融合，再在同一编码器上验证 FASERCal 式预训练。本阶段完成的是研究设计、文献核查和有界数据审计，分类性能尚未测量。

## 阶段目标

为 `10081=CC νe`、`10082=CC νμ`、`10083=CC ντ`、`10084=NC` 制定可执行的研究计划，回答原生像素细节、其他探测器信息、模型关系结构和预训练各自能提供什么增量。验收要求为：有本地资料与联网论文依据，区分硬件观测与模拟 truth，定义对照实验、数据划分、统计评价、系统效应、计算预算和阶段交付，不以已有失败方法作为架构前提。

## 工作内容

- 阅读可用的 FASERCal arXiv v2 全文源码及 ICHEP 汇报，在线核对 Nature Machine Intelligence 正式发表信息、作者代码与相关方法；比较 ParticleNet、Particle Transformer、Perceiver IO 及 2024–2026 点云方法。LitePT、NOMAE 另抽读 Methods；文献及迁移判断见 [文献依据](../roadmap/archive/2026-11-14_v1/02_literature.md)。
- 检查 ROOT schema、geometry、事件与 hit 对应、标签一致性、原生占用、Scintillator 通道合并收益；对照 Geant4 响应源码。旧程序仅参考文件和绘图接口，未复用其网络、truth selection 或标签规则。
- 在 [数据与硬件合同](../roadmap/archive/2026-11-14_v1/01_data_and_hardware.md) 中定义 observation/target/provenance 隔离、原生通道表示、digitization、空事件、generator-event family split 与完整性检查。
- 在 [模型与实验设计](../roadmap/archive/2026-11-14_v1/03_models_and_experiments.md) 中规划 ParticleNet/ParT 独立基线、局部/module/latent 主模型、原生与粗网格对照、多模态消融、MAE/自蒸馏/Rel、τ/charm 与对象分支。明确 coarse OR 与 active-pixel count 的区别，避免将信号归约变化混入分辨率结论。
- 在 [执行工单](../roadmap/archive/2026-11-14_v1/04_execution.md) 中安排 S0–S6、建议 12–16 周主线、资源测量与 HTCondor 规则；[study.yaml](../roadmap/configs/study.yaml) 是待实现合同，现有训练器尚不读取该配置。

## 详细数据

### 输入规模与抽样边界

输入根目录为 `/eos/project-f/fasersim-bonn/public/pinpoint/data`。文件规模来自四个目录的名称和文件大小盘点；总事件数未逐文件测量。

| Run | Class | ROOT files | Size (bytes) | Last chunk entries |
| --- | --- | ---: | ---: | ---: |
| 10081 | CC νe | 874 | 539,664,893,970 | 65 |
| 10082 | CC νμ | 4,158 | 1,202,843,268,358 | 413 |
| 10083 | CC ντ | 653 | 228,950,146,446 | 231 |
| 10084 | NC | 1,555 | 414,556,703,970 | 863 |
| Total | Four classes | 7,240 | 2,386,015,012,744 | — |

合计约 2.386 TB / 2.170 TiB。NC 在 chunk 000–1592 范围缺 38 个编号，原因未确定；不能用文件数乘 1,000 推算事件数。

每类按 chunk 数值排序，检查首/中/末三个 ROOT 的 header/geometry，共 12 文件；每类 chunk 000 读取 1,000 条 scalar metadata，共 4,000 条；每类 chunk 000 读取前 32 条 pixel、Scintillator、FASER entries，每个 hit tree 共 128 条。12 个抽样文件中四种 tree entries 相等，每类三份 geometry 一致；未据此推断全库一致。

### 会改变执行顺序的实际发现

| Observation | Measured result | Planned action |
| --- | --- | --- |
| Empty-pixel event IDs | 四类首 32 entries 分别有 6/7/11/8 条 pixel ID 不一致，均为空 pixel；Scintillator/FASER 同范围无不一致 | 每文件验证非空 ID 与 row 对齐后，才为对应空 entry 建正确事件键 |
| Label metadata | `10081_000.root` 的 evtID 121、609、706 共 3 条与名义类别存在元数据冲突 | 追溯 generator 与反应编码，保留未决状态和处理分母；正常事件继续 |
| Bar aggregation | per-track records 合并为候选硬件 bar 通道，抽样 pooled 记录数减少 9.26–34.45 倍 | 按通道与真实读出窗口聚合，不以 track 记录条数作为特征 |
| Pixel occupancy | 含空事件的四类 median/max 为 303.5/4960、16.5/619、9/1947、26/6056 hits/event | 扩展分层抽样并测 p99/p99.9，再设 token budget |
| Timing | Scintillator 抽样最大 hitTime 约 119,931 ns | 验证电子学积分窗和 channel timing，不能直接使用全部晚到记录 |
| Event weights | 抽样 4,000 条均为 1，当前 generator 源码也写死 1 | exposure/真实 priors 未获得前，先报告明确权重场景的效率与纯度 |

当前源码 `AnalysisManager.cc::FillHitsOutput` 在 hit 循环内更新 pixel event ID，与空 collection ID 停留的观测一致；生产版本仍需追溯。`PixelSD` 的 ancestry flags 只保留首次创建像素时的贡献者，不能由现有 flags 还原 FASERCal 式精确 soft semantic mixtures。FASER 分支中的动量、charge、trackID 是 G4 truth，主推理输入必须改为 digitized observations 或由其拟合的量。

实际模拟路径为 `/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4`，检查时工作树干净，HEAD 为 `dbbdb5155a66f5329fd27ea5285d037bf8fb2f03`。仓库内 `PinpointAnalysis/Pinpoint_G4` 不存在。ROOT pixel 活性尺寸为 `266×196 mm²`、pitch 为 `20.8×22.8 μm`、15 个 pixel planes；当前源码默认尺寸为 `250×200 mm²`，且存在材料和坐标索引修正。当前 HEAD 不能视为这批数据的生产配置。

### 本地资料与引用范围

有效资料为 `docs/FASERCal/arXiv-2604.07037v2.tar.gz` 和 19 页的 `ichep-sam-26-07-30.pdf`。以下四个 `.pdf` 实为 CERN HTML 登录页，未获得其研究内容：

- `docs/FORTVNE_10212026.pdf`
- `docs/FORTVNE_Update_kinematic_variables_2026-10-07.pdf`
- `docs/FORTVNE_pixel_vs_scintonly_meeting_260916.pdf`
- `docs/FASERCal/FASERCal_wshopRun4.pdf`

FASERCal 正式论文 DOI 为 `10.1038/s42256-026-01309-6`，在线日期 2026-09-30。路线图的精确 AUC、样本与成本数值引用本地 arXiv v2，并标明属于 FASERCal；例如其 1,011 GH200 GPU-hours 预训练成本用于设计同算力对照，不作为 PINPOINT 预算测量。联网筛选覆盖 CVPR 2024/2025/2026、ICCV 2025 和 NeurIPS 2025 的相关公开论文与作者资料，2026 候选包含 LitePT、PointINS、PointCSP；未将任务不匹配的 GGPT 纳入主线。

### 命令、环境和验证

审计执行命令：

```bash
python3 roadmap/tools/inspect_inputs.py \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_2026-11-11/input_audit.json
```

使用现成 Python 3.9.25、NumPy 1.23.5、uproot 5.6.9、awkward 2.8.12 完成只读审计；未安装项目或重新构建模拟。生产开发仍按项目 Python 3.12 合同选择匹配环境。本次未运行旧预处理、模型训练、GPU 测试或 HTCondor 作业。

| Validation | Result |
| --- | --- |
| Roadmap artifacts | 7 files，93,715 bytes，907 lines；含 5 个 Markdown |
| Local roadmap links / explicit anchors | 14 个检查通过 |
| Whitespace / final newline / code fences | 检查通过 |
| Python script syntax | `ast.parse` 通过 |
| YAML contract | `yaml.safe_load`、四类映射、split 比例、train-only pretraining/normalization、种子、event/vertex 子键检查通过 |
| New-model AUC / efficiency / calibration | 未测量 |
| Full-library event count / tail occupancy | 未测量 |
| Hardware calibration / generator systematics / physical yields | 未测量 |
| GPU throughput / VRAM / training GPU-hours / schedd load | 未测量 |

上述路线图统计不含本阶段报告本身。提交前执行 `git diff --cached --check`。本地 `docs/` 未改动，未加入本次提交。

代码与计划在 `roadmap/`，本报告在 `workbook/`。紧凑运行证据位于 `/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_2026-11-11/`：`input_audit.json`、`source_summary.json`、`cvpr2026_day{1,2,3}_selected.json` 与 `validation.json`。临时网页正文未保留在最终结果目录，原始 ROOT 未修改。

## 阶段性结论

**事实：**现有库具备原生 pixel、bar 与 tracker 模拟信息，但存在事件关联、标签元数据和生产配置待追溯问题。pixel 原生读出已聚合到真实像素粒度，Scintillator 则需消除 per-track 记账拆分。保存的理想位置、沉积能、时间和真动量具有不同测量含义，不能统一当作已标定硬件输入。

**研究假设：**保留原生局部结构、按 plane/module 组织信息、再做小 latent 多模态融合，可能改善 τ 困难通道及资源效率；MAE、关系监督和对象任务是否有额外收益，尚需同 split、同读出、同标签及同算力的对照。本阶段没有证明任何新模型优于旧模型。

## 下一步计划

| Task | Required input | Acceptance criterion |
| --- | --- | --- |
| S0：全量轻量 inventory 与异常追溯 | 四类 ROOT、generator source mapping、生产 commit/macros | 实际事件计数、标签状态、空事件对应、schema/geometry 范围和 source-family split 可复核 |
| S0：硬件响应合同 | pixel readout、bar/SiPM 与 tracker 规格或标定 | 推理白名单、单位、window/threshold、响应层级和未知项有明确记录 |
| S1：native catalog 与 B0 | 冻结数据合同、每类一个 chunk 的 pilot | 通道合并与守恒、空/缺失 mask、吞吐/占用测量和四类基线通过 |
| S2：原生/粗表示与 ParticleNet/ParT | 共享 manifest、固定预算与评价器 | 原生细节净贡献有配对统计证据，或给出当前统计下的可分辨收益区间 |

后续大样本处理通过 HTCondor，提交前按兼容性与实时负载选择 schedd，并将选择依据写入任务 manifest。S3–S6 依上一阶段证据逐步扩展，不在本次规划阶段启动批量训练。
