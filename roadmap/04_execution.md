# 分阶段工单、资源与交付

S0/S1/S2均已完成。用户已确认samples满足要求并接受现有detector-proxy；来源认证和真实硬件标定不再作为前置条件。S2完整35项科学配置、490项读回、41单seed配对和4三seed均值通过严格评价，实际组织成本92.73055556 GPUh。[最终报告](../workbook/2026-11-12_stage55_s2_final_acceptance.md)给出科学结论和gate decision，[运行入口](07_s2_operations.md)保留可复现命令。S3–S6尚未执行；以下工单和建议预算保留为研究路线图。

以下 S0/S1 工单保留原始研究设计；其中来源追溯和硬件标定条款已由上述用户决定覆盖，不再执行。S2–S6 沿用接受的 detector-proxy 和冻结 manifest，后续模型评价按其独立实验计划执行。

## 1. S0：可观测量、事件身份和样本协议

**要回答的问题：每条 observation 对应什么真实事件，哪些字段在硬件上可测？**

先使用 [inspect_inputs.py](tools/inspect_inputs.py) 复核本次证据。随后建立 `inventory/manifest` 生成器，通过 HTCondor 分块读取所有文件的轻量 header、geometry/label schema、scalar metadata 和 event IDs；不为盘点读取完整大 hit branches。首轮只处理每 run 一个 chunk，小规模 spot check 后再扩展。记录 NC 编号缺口和不完整尾文件；总事件数以实际 metadata 唯一事件计数为准。

追溯生产配置，显式核对 `10081→CC νe`、`10082→CC νμ`、`10083→CC ντ`、`10084→NC`，同时核对 generator 的反应类型与 direct CC lepton。为异常 labels 保留 unresolved 列表和处理决策，正常事件继续；不按 run 取模。验证空 pixel 的 entry 顺序后给出 join 修复方案。获取原始 generator source/entry 关系，防止重新模拟或相邻 chunk 中重复同一物理事件。

与硬件设计/标定资料核对 binary/analog pixel、bar 单/双端 readout、可用时间信息、SiPM 动态范围、实际 window/threshold、tracker 分辨率/拟合能力。已知范围写入 conditions；未知项保持 null，并只报告相应 detector-proxy/ideal response 层级，不把未标定项加入“已验证硬件”主结果。

更新四份本地汇报后，S0 增加 historical-study 对照表：L1–L3 对应的 run IDs、模型/representation、fiducial 原点与边界、score selections、class priors/weights、FASER field/reconstruction 假设。2026-10-07 的 small/large 与 2026-09-16 的定义分别核对；资料未给出的项保持待查，当前数据审计继续。具体证据见 [05_local_studies.md](05_local_studies.md)。

交付：版本化 data contract、完整 event inventory、source-family split manifest、readout/geometry map、异常表、cutflow 和数据范围图。验收：每个可用事件可追溯、所有 split 无家族交叉、schema 白名单与模型输入隔离、所有标签例外有可复核状态；总事件数、source overlap、全空事件和 ν/anti-ν/NC-flavour 覆盖有测量。

## 2. S1：原生通道数据管线与基线

**要回答的问题：能否以适合 GPU 的速度无额外损失地读取原生硬件观测？**

新增独立命名空间建议为 `src/analysis/flavour/`，按 `inventory.py / readout.py / create_data.py / datasets.py / features.py / models/ / pretrain.py / train.py / evaluate.py` 拆分。旧 `utils` 的图形、单位和 ROOT 读取方式可参考，逐个适配真实 branch 名；不直接调用旧 `get_good_event_ids`、modulo labels 或旧 FASER highest-true-energy hit 选法。

先交互式读取 10–100 events 验证索引/通道聚合，再用 HTCondor 跑每类一个 chunk。pixel 保留原始索引/中心，bar 合并并验证 response，tracker 分别输出 ideal proxy 与可行的 digitized measurements。使用 uproot/Awkward 分批读指定列、NumPy/vectorized reductions 或 C++ 聚合；依据 profiler 决定是否使用 ROOT RDataFrame，不先用大型 pandas `groupby`/逐 event 重复扫描。

计时至少区分 ROOT read/decompression、channel aggregation、shard write、GPU load/forward/backward。shard 初始目标为约 256 MiB–1 GiB，按测量调整，row groups/offsets 支持变长 batch。训练在 worker/local scratch 分块缓存，结束时只保留需要的紧凑数据、manifest 和合并统计；不长期保留重复 feature copies。

交付：通道转换器、dataset/loader、B0 baseline、10–20 个分层 event displays、signal/occupancy/cutflow、资源报告。验收：能量/PE 的适用守恒检查、空/缺失 mask、量纲、inverse geometry、shuffle invariance 和一次小规模训练/验证通过；主指标计算不依赖 truth selection；记录 p99/p99.9 和预计全量数据体积。

## 3. S2：原生分辨率、ParticleNet 与 ParT

**要回答的问题：粗 bin 丢失了哪些对反应分类有用的结构？**

在固定训练子集上完成 E01–E03 的最小版。建议 pilot 为每类最多 20k train events（实际数量取有效样本与预算允许的较小值），筛选用种子 42；最终对最强两个架构与至少一个粗表示重复三种子。共享 validation/calibration/test manifest；不能按 class 选择不同的 occupancy cut。

先固定其他模态，在 pixel 分支比较 native/1 mm × joint positions/xz–yz projections 四种处理，再补 100/200 μm 分辨率曲线。统一 OR/count/signal reducer 并记录处理顺序，量化关联丢失与粗粒度的交互。所有模型在完整目标域及经 S0 固定的中心/large 区域交叉评价，另做训练域控制；两份既有报告的差异只有在条件统一后才能归因。

同时做可控 toy topology：单/双轨像素间距、kink angles、shower overlap，在真实 plane/pitch/噪声范围内检查原生与粗表示的分辨能力。toy study 只证明信息保留/实现行为，不替代 neutrino sample 指标。量化 layer 内局部结构、跨层连通和能量/位置先验各自贡献。

交付：B1/B2、native/rebin 对照、各类/τ 分支统计、memory/latency 曲线和错误事件组。验收：有统一输入与预算下的可复核结论；若增益不显著，报告区间与是否有统计能力排除预定最小收益，不用“未显著”直接判定等效。

## 4. S3：分层编码与多探测器融合

**要回答的问题：局部几何和跨探测器 context 如何分工？**

实现 plane-aware local encoder + module context + latent fusion，先用无复杂 decoder 的 direct-supervised 模型。依次比较 late pooling、latent、joint attention；关闭 module/pair geometry；关闭 pixel/bar/tracker/time。不同读出层级分开建表，time 和 charge 需分别证明标定/validity。

记录 all-event 与 active 条件效率，以及高占用、出接受度、缺失子系统情况。按 validation 的 τ 工作点、macro metrics 和实际 GPU-hours 选择两个 Pareto 候选，再用三种子确认。

FASER 的增量同时报相同 purity/background acceptance 下的效率、完整曲线和拟合覆盖率；digitized hit、reconstructed track 与简化 12-feature 参照分开。检查 `q/p` uncertainty、fake associations 与动量截顶，不把高纯低覆盖分支当作全样本重建。

交付：推荐 backbone、资源 Pareto 图、modal ablation 和解释图。验收：达到 README 的预注册晋级门槛，或在置信区间支持的近等效质量下明显减小资源；若未达标，继续最强简单模型，分层结构保留为解释性结果。

## 5. S4：预训练与资源收益

**要回答的问题：预训练改善了什么，是否比继续监督学习更划算？**

只用 train partition。先用 100k events、mask `0.5/0.75` 做短 pilot，验证 decoder 不泄漏 active support，reconstruction 不塌缩，再扩至 1M。Scratch、MAE、self-distillation、MAE+Rel 用同一个下游 backbone；标签 budget `1k/10k/100k`，可行时扩展 1M；验证/测试保持完整且不变。

Rel 需要的 soft contribution truth 若不完整，先做少量匹配模拟，仅保存紧凑 accumulators。新增 output/响应逻辑时才重编译对应源码，复用现成工具链；不为 whole sample 输出 step-level 巨量文件。区分弱 pixel flags、精确 soft labels 和纯自监督目标。

交付：learning curves、matched-compute 表、linear probe 和 full fine-tuning、Rel 标签质量与新增成本。验收：在 τ 工作点、低标签适应或总资源上有证据支持的收益；证据不足时不扩大至 FASERCal 约千 GPU-hour 的预训练规模。

## 6. S5：τ、charm 与可重建对象

**要回答的问题：τ 类别的增益来自可见衰变结构还是样本谱形？**

完成 direct four-way 与 six-subclass sum 的对照，分开评价 τ→e/μ/had、τ decay location、visible prongs、charm presence、containment。逐项加入 vertex/semantic/charm auxiliaries；验证辅助标签定义、target masks 和负迁移。objects 分支先报告 track/cluster association、fake rate、vertex efficiency/resolution，再接分类。

抽取在 detector geometry 内可见与不可见的 secondary signature 子样本，画正确/错误/随机事件的 raw response。以能谱、真实 vertex/target matched 诊断控制先验，并检查 model score 与 reconstructed observable 的联系。对程序来源（ID/labels、真值混入、截断）和物理来源（接受度、重叠、不可见衰变）分别验证。

按更新的运动学汇报，增加预测 vertex-relative 与 absolute/raw 双路径、component direction 和 θ/φ 分开评价。每个 quality/score cut 对应 resolution–coverage 表与能量/位置图。τhad 先形成稳定诊断，但最终保留 τ→e/μ；分别写 τhad 与 inclusive τ 的分母，不用 2025 workshop 的 11% τhad efficiency 代表全部 τ。

交付：τ efficiency/purity 与 decay-visibility 图、charm 假 τ 分析、辅助任务/objects 净贡献。验收：至少形成一个有计数、区间和物理解释的困难通道结论，并明确仍待查的混淆。

## 7. S6：系统效应与最终物理结果

**要回答的问题：在什么 detector/generator 条件下，四类概率和选择效率可信？**

验证 calibration 与 weighted priors。对 response nuisances 做 paired resimulation/digitizer variations，幅度由 hardware measurements 决定；先比较 nominal、单项扰动和组合包络，不把任意 ±10% 称作真实 systematic。保留未参与 augmentation 的条件作 held-out domain test。

generator variations 需兼容当前能区与 target；尽量匹配 incoming neutrino/vertex/target，分别比较 flavour 和 charm，必要时补高统计 rare-background sample并保存其生成权重。冻结候选、超参、阈值、校准、selection 后只运行一次正式 test campaign；保存每个 event 的四类分数，系统变体按同一物理事件对应。

交付：完整 confusion/ROC/PR、τ operating points、uncertainty/calibration、systematics、资源、适用域和 score migration。若具备 exposure，交付 profile-likelihood sensitivity/forward-folding 示例；否则交付按明确 priors 的 efficiency/purity 扫描，luminosity 产额与 significance 仍标“未测量”。

最终方法与原生分辨率/模态/pretraining 的每项贡献均能用同 split 的表格追溯；阶段报告说明事实、支持范围、假设和下一项可测问题。

## 8. 计算与调度合同

| Resource item | Initial proposal | Measurement / decision |
| --- | --- | --- |
| Interactive verification | 10–100 events；单 GPU short step 如需要 | 验证后停止交互式批量处理，长作业进入 HTCondor |
| CPU conversion | 每任务 4–8 CPUs；内存从 8–16 GiB pilot 起 | 根据最大原始 records 和 chunk read size 测 RSS/吞吐再调整 |
| GPU job | 1 GPU、8 CPUs；batch 依实际 hit budget | 记录 GPU 型号/VRAM、显存峰值、events/s、GPU utilization、CPU wait |
| Screening budget | S2–S3 合计先封顶约 100 GPU-hours | 按验证结果分轮使用，不预排所有 Cartesian products |
| Expanded supervised study | 再分配约 100–200 GPU-hours | 仅保留 2 个候选与关键三种子 ablations |
| Pretraining pilot | MAE 约 25–50 GPU-hours，self-distill 约 25–50 | 决策后再申请至多约 200 GPU-hours continuation；不自动耗尽预算 |
| Final systematics | 根据独立事件统计和 nuisance 样本估算 | 与训练成本分列；GPU/CPU 型号不同不可直接比较 hours |

上表预算是研究组织建议，并非排期承诺、实际测量或本次作业授权。训练用 mixed precision（支持时 bfloat16）、gradient accumulation、packed/bucketed variable-size batches；attention backend 根据实际 CUDA/PyTorch/GPU 支持选择，始终有可核查的 PyTorch fallback。按 profiler 决定 checkpointing/compilation/更大模型；不要为了论文使用自定义 kernel 而重建整个环境。

多进程/多线程以申请资源为上限：例如 8-CPU GPU job 分配 4 个 data workers、worker 内 BLAS/decompression/PyTorch 单线程，主进程 1–2 compute threads，其余用于 I/O；测量后调整。避免每个 worker 再启动 8 个 OpenMP threads。CPU conversion 的解压/聚合线程也须共享同一 CPU 预算。

每次批量提交前读取当前 schedd 负载，而不是沿用规划日期的快照。可先用以下只读命令获取候选 scheduler 摘要：

```bash
condor_status -schedd -af Name TotalRunningJobs TotalIdleJobs TotalHeldJobs MaxJobsRunning
```

再对兼容 GPU/CPU 类型、作业时长、账号权限和站点 storage/container 要求的候选检查队列与可用资源；在兼容集合中优先选择排队/负载较低者。将时间、候选、idle/running、GPU/时长约束、选择依据与最终 `ClusterId` 写入 task manifest。只有 queue length 不足以代表等待时间，需记录可比较的约束。必要时限制 `max_materialize`/并发数，减少同时读 EOS 的小请求。

S0/S1 已按此合同运行 HTCondor；每次提交的负载与兼容性记录在 results 的 `submission_*.json`，不将规划时的队列状态当作下一次提交依据。若后续改用 Kubeflow 而会话未登录，再提醒登录。

## 9. 路径、产物与复现

| Purpose | Path / policy |
| --- | --- |
| Source input | `/eos/project-f/fasersim-bonn/public/pinpoint/data/{10081,10082,10083,10084}/`；只读 |
| Simulation | `/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4`；固定相匹配 production/conditions |
| Current planning evidence | `/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_2026-11-11/` |
| Updated-PDF review evidence | `/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_pdf_refresh_2026-11-11/` |
| Future run output | `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/<stage>/<experiment_id>/` |
| Submit files | `~/condor/Pinpoint/flavour/<stage>/`；只放小 wrappers/config/submit 文件 |
| Logs / checkpoints / figures | `results/flavour/.../logs,checkpoints,figures`；作业 stdout/stderr/event log 同样写 results |
| Versioned artifacts | 仓库代码、`roadmap/configs`、workbook、精简汇总表 |

数据/日志与训练产物不写 home，不提交大型 notebook 输出。新 path/config 层明确接入上面的 ROOT 根目录；现有 `get_data_path/get_root_path` 的默认 `../data/root` 不匹配当前 public data 目录，不能靠工作目录猜测。训练读取紧凑 native catalog，在 worker scratch 缓存所需 shard；成功验证后删除本任务的临时中间副本，原始 public ROOT 不改动。

每次 experiment manifest 至少包括 code commit、dataset/split/geometry/readout ID、生成家族规则、whitelist、类别顺序、cutflow/label unresolved policy、normalization、训练种子、超参、GPU/CPU/软件版本、checkpoint rule、命令、schedd/ClusterId、资源测量、validation/calibration/test 角色。每个 seed 只保留 best 与可恢复的 last checkpoint；批量 jobs 的 JSON 统计合并成按 experiment/event-key 去重的 Parquet/JSON，不在 EOS 留几千份相同 summary。

最终图标题、坐标、legend 与图内 annotations 用 English，例如 `Tau efficiency at fixed background acceptance`、`Native pixel pitch`、`Expected yield`；报告解释用中文。图注明 sample/split、选择、权重、误差、读出/系统版本。绘图输出 standalone PDF/PNG，避免把大量图片嵌入 Git notebook。

## 10. 下一次开发的最小起点

S2已通过验收；下一阶段S3以本次PN/ParT native及1 mm三seed结果为基线，检验plane-aware局部编码、module-aware context与latent fusion。保留bar，tracker同时评价四类收益/覆盖/成本；不把native或某架构的τ AP未确认增量理解为硬件信息无用。具体结论见[总报告](../workbook/2026-11-12_stage55_s2_final_acceptance.md)，本次未启动S3。共享输入、冻结proxy与calibration/test角色继续沿用。
