# PINPOINT 下一轮模型优化路线图（v2）

日期：2026-11-14。审查基线：Git `37dd763`，截至 workbook stage236 的 S0–S6。本版是下一轮研究设计，新增模型效果尚未测量。

**结论：已经取得可复核的小幅改进，尚无证据表明现有方法达到最佳效果。下一步先建立可与合作组对齐的三类强基线，并行保留四类主线；优先解决同输入架构比较、跨层几何关系和训练规模问题，再决定扩大预训练或采用更多类别。**

独立 test 上，MAE3 ensemble 的四类 balanced accuracy 为 **43.5226%**，相对同为 100k 监督标签的 Scratch 增加 **0.3354 个百分点**；τ 在冻结 validation 1% 背景阈值下效率 **3.5556%**，实际 test FPR **1.2080%**。这些结果支持预训练收益，绝对性能仍不足以支持高纯 τ 分析。两者监督协议匹配，MAE 另有预训练成本。当前所有 accepted samples、split 和 detector-proxy 继续作为有效研究基础。

本次重新读取六个最终 revision 3 模型的训练记录和已有 validation 预测，发现：

- 六模型均以第 4/4 轮为 best，末轮 NLL 继续下降 **0.00203–0.00779**，训练饱和未建立。
- 100k 监督标签、100k SSL 语料和约 0.898M 参数构成当前主模型规模；没有完成 300k/1M 监督训练的饱和研究。
- 从固定四类 ensemble 删除真 τ 并重归一化，三类 validation BA 为 **54.6933%**；pixel-centered 为 **74.4517%**。这是条件诊断，三类实际重训效果**未测量**。
- 记录中的 data-wait/总耗时为 **31.16%–46.64%**。先提升数据供给，可把更多计算用于长训练和有效模型比较。

数值、统计口径和代码证据见[独立评估](v2/01_evidence_review.md)及[复核 JSON](v2/current_results_review.json)。

## 阅读顺序

| Document | Purpose |
| --- | --- |
| [01：结果、资料与系统性不足](v2/01_evidence_review.md) | 回答改进是否成立、哪些比较尚不公平、哪些故障已修复、哪些瓶颈仍是假设 |
| [02：类别选择与评价合同](v2/02_tasks_and_evaluation.md) | 三类、四类、条件六类的并行方式；新独立确认集、尾部统计与晋级规则 |
| [03：模型与物理设计](v2/03_models_and_physics.md) | 跨层稀疏关系、raw/object 双路径、异构融合、τ/charm 与方向任务 |
| [04：执行路线与实验工单](v2/04_execution_plan.md) | R0–R7 依赖、最小实验矩阵、资源安排、首批具体工作和交付标准 |
| [05：文献依据](v2/05_literature.md) | 本地资料及本次联网核对的论文、采用理由、适用边界 |
| [06：完整工作记录索引](v2/06_workbook_index.md) | 236 份既有 workbook 的阶段归类与结果入口 |
| [实验规格](v2/experiment_plan.json) | 可解析的设计合同；需实现对应 runner 后才能提交，不能直接传给旧 CLI |

## 研究路线与决策

```mermaid
flowchart TD
    A[Reuse accepted readout and data] --> B[New development and confirmation manifests]
    B --> C[Three-class and four-class matched baselines]
    C --> D[Training and data scaling]
    D --> E[Cross-plane geometry and sparse local relations]
    E --> F[Raw and reconstructed object fusion]
    F --> G[Tau specialists and conditional decay heads]
    G --> H[Targeted pretraining and robustness]
    H --> I[Locked independent confirmation]
    C --> J[Three-class collaboration result]
    G --> K[Keep four classes if useful tau gain is confirmed]
```

三类输出为 `CC νe / CC νμ / NC`；四类输出为 `CC νe / CC νμ / CC ντ / NC`。三类模型训练时排除 τ，部署时仍须评价 τ 污染。四类模型必须在同一非 τ 样本上给出条件三类结果；两者由同样的输入、样本和计算预算比较。τ 子通道优先作为条件辅助头或 specialist，不直接把全部训练改成六类均衡 softmax。

| Stage | Main question | Deliverable / decision |
| --- | --- | --- |
| R0 | 哪些事件仍可用于新的独立确认？现有训练瓶颈在哪里？ | 使用记录清单、新 manifests、实际 token/geometry 元数据和 loader profile |
| R1 | 三类任务做到什么程度？四类是否损害共同三类任务？ | 同输入强基线、合作组条件对照、τ 混入矩阵 |
| R2 | 数据、曝光、容量中哪个限制性能？ | 100k→300k→1M 条件扩展；4→12→收敛的学习曲线 |
| R3 | 显式跨层关系能否把细粒度转为有效收益？ | 几何稀疏图、边界通信、原生/粗粒度配对和三 seed 确认 |
| R4 | 能否重建有用对象，而不是引入大量假关联？ | Track/vertex/association 的效率–fake–coverage 与 raw/object 分类对照 |
| R5 | τ 是否值得保留为主类别，细分是否有用？ | Flat4、3+τ、条件 τ decay 头；固定 FPR、分支及 charm 对照 |
| R6 | 预训练和鲁棒训练能否超过强监督基线？ | 定向 MAE/局部监督、同总计算监督控制、held-out 条件 |
| R7 | 候选是否在新事件和系统变化下保持收益？ | 冻结独立确认、概率与工作点、物理条件和完整报告 |

R1 和 R2 是首要交付；R3 是优先架构改进；R4–R6 由前序证据触发。预计主线 6–10 周，属排期建议，实际 GPU/CPU 时间由 R0 微基准确定。首轮建议分阶段预算 80 device-hours，具体分配与停止条件见[执行计划](v2/04_execution_plan.md)；本次只完成研究设计和轻量复核。

## 历史计划与结果

旧路线的 **14 份完整文档**归入 [archive/2026-11-14_v1](archive/2026-11-14_v1/README.md)，归档说明见[archive 索引](archive/README.md)。根目录原 01–13 页面保留历史链接入口；原 `configs/`、`results/`、`tools/` 路径继续服务已有脚本和报告。旧路线中“下一阶段”“尚未测量”等文字反映各记录形成时点，S0–S6 的最终状态以 stage236 及各阶段最终报告为准。

关键证据：[S2 stage55](../workbook/2026-11-12_stage55_s2_final_acceptance.md)、[修复 S3 stage87](../workbook/2026-11-13_stage87_s3_repair_final_results.md)、[S4 stage181](../workbook/2026-11-14_stage181_s4_final_results.md)、[S5 stage203](../workbook/2026-11-14_stage203_s5_final_results.md)、[S5 补充 stage205](../workbook/2026-11-14_stage205_s5_scope_completion_audit.md)、[S6 stage236](../workbook/2026-11-14_stage236_s6_final_results.md)、[本次 stage237](../workbook/2026-11-14_stage237_independent_review_and_roadmap_v2.md)。
