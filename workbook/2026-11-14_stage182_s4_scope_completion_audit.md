# S4 原合同全范围完成性审查

S4六项执行要求及遮蔽、Rel、故障修复、调度交付均有对应实证，本次审查未发现尚未执行的必需实验。保留修正后MAE3作为有限标签基线，本轮不扩1M。阶段181的20项检查覆盖最后五初始化矩阵；本报告另核对早期pilot、分支probe、几何控制、三seed和matched-compute，形成[完整证据映射](../roadmap/results/s4_scope_completion_audit.json)。

## 阶段目标

逐项回答原S4研究合同是否已完成，区分实验交付、方法收益、失败配方和后续研究。依据为[执行路线图第5节](../roadmap/04_execution.md)和[S4合同第3至5节](../roadmap/11_s4_pretraining.md)。保留原合同，按其“证据不足不扩量、无收益也完成对照”的条件判断完成性。

## 工作内容

读取55个直接证据路径及选定原始JSON字段，核对早期train-only诊断与修复后结果的适用范围；未新增训练或GPU allocation。样本、split与detector-proxy沿用用户接受状态。最后矩阵的checkpoint、独立backward和磁盘回放复用已保存的实测记录，本次未重新加载模型或重新生成已清理的feature缓存。

修正`04_execution.md`的“S4未执行”与后续开发入口；将`11_s4_pretraining.md`中尚用未来时描述的30项readout回放改为已完成；Rel语义表的分类pending状态更新为指向独立的最终分类评价。历史模型配置、原始评价和失效清单保持各自的实验身份。

## 原要求与直接证据

下文`ROOT`为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。表中的完成表示预定研究与对照已交付，方法质量和物理收益在结论中分别判断。

| 原要求 | 直接证据与实测 | 审查结果 |
| --- | --- | --- |
| 第1项：100k语料、Rel sidecars、98k/2k train-inner、嵌套标签预算 | `ROOT/plan.json`、`data_summary.json`和SSL manifests；100k train四类各25k，原validation四类各2,500；SSL 98k fit与2k probe | 完成；标签预算1k/10k/100k，calibration/test未参与 |
| 第2项：S3各分支linear/MLP probe | `ROOT/experiments/s4_s3tracker_branchprobes_s42/{manifest,metrics}.json`；8k fit、2k probe，regional/global/local/summary/full各两种读出，共10项 | 完成；上游S3 encoder见过train标签，是可读性诊断 |
| 第3项：真实拟合、精度/占用压力、无泄漏、EMA/方差、恢复和0.5/0.75 pilot | `ROOT/ssl_gate/{s4p2_mae_m50_s42,s4p2_mae_m75_s42,s4p4_distill_global_b64_s42,s4p6_barrel_channels_s42}/gate.json`及对应`ssl_readback.json`；4项gate均通过，最大36,549原始通道在FP32/BF16下有非零input梯度；无泄漏与恢复检查见`tests/test_flavour_s4.py`和阶段94/106记录 | 完成；稀疏native target修正后重新pilot，未沿用失败目标权重 |
| 第4项：同backbone多目标、continuation、matched-supervised与关键三seed | [空响应修复三seed](../roadmap/results/s4_empty_repair_summary.json)、[有效低标签三seed](../roadmap/results/s4_transfer4_low_label_summary.json)、[最终五初始化](../roadmap/results/s4_fp64_objectives_summary.json)、[matched预算](../roadmap/results/s4_empty_matched_budgets.json) | 完成；关键MAE3/Scratch用17/42/2026确认，最后五目标筛查仅seed42 |
| 第5项：每预算linear/full、完整10k评价、学习曲线、健康、成本和分支诊断 | [45预测/66配对](../roadmap/results/s4_fp64_objectives_summary.json)、[20项技术检查](../roadmap/results/s4_fp64_objectives_acceptance.json)、[完整路径诊断](../roadmap/results/s4_transfer4_functional_paths.json)；30个full/global读出均驻点并回放 | 完成；fixed-C低标签概率失败及τ尾部无确定增益均已报告 |
| 第6项：依据重建、下游和成本决定1M | [终态决定](../roadmap/results/s4_final_closure.json)、[最终结果](2026-11-14_stage181_s4_final_results.md) | 条件判断完成：不扩1M |
| 遮蔽与重建附加合同：visible-only输入、正负/全几何查询、geometry/permutation控制及coverage | `s4/dataset.py`、对应测试、三套MAE3 `ssl_readback.json`及`ROOT/experiments/s4_geometry_query_baseline_s42/{cache_summary,reconstruction}.json` | 完成；结论只对应声明的混合查询分布 |
| Rel标签质量与新增成本 | `ROOT/data_summary.json`、`rel_smoke.json`、两种continuation manifests、[语义控制](../roadmap/results/s4_rel_semantics.json)和最终分类表 | 完成；精确bar soft contributions是额外模拟语义监督，未声称纯自监督 |
| 异常审查、修复重训和失效结果隔离 | [S3修复](../roadmap/10_s3_repair.md)、[LP失效清单](../roadmap/results/s4_transfer_compilation_invalidation.json)、阶段145/167、[概率诊断](2026-11-14_stage177_s4_probability_diagnostic.md)及最后健康记录 | 完成；失活、空响应和LP断梯度结果未进入有效结论 |
| 调度、终态、费用与清理 | `ROOT/campaign_accounting.json`、`audit/fp64_recovery_and_superseded_cpu_stop.json`、feature/AFS清理清单及[收尾摘要](../roadmap/results/s4_final_closure.json) | 完成；114 allocations终止，费用含历史失败、恢复与重复执行 |

## 详细数据与证据范围

### 重建、几何与分支

S3 train-only global linear/MLP的macro AP为0.354040/0.327226，full为0.393321/0.417233。这里的2k probe与encoder上游训练不独立，后续预训练收益依据完整下游validation另测。

主mask选择0.75依据是修正后的512-step pilot实际学习与吞吐，不能解释为0.75在全部质量指标优于0.5。三套完整MAE3的fine-patch AP（seed17/42/2026）为0.487495/0.454979/0.516933，global有效秩为8.918626/7.532382/9.236249，event-source置换均增加三尺度BCE。

| Reconstruction target | Geometry query-only AP | Full MAE3 seed42 AP |
| --- | ---: | ---: |
| Fine native patch | 0.028193 | 0.454979 |
| 4-mm pixel cell | 0.111532 | 0.780899 |
| Scintillator bar | 0.445807 | 0.898600 |

Geometry控制fit为2,048事件/1,024 steps，与full MAE不等曝光或计算量。同一2k probe的fine-patch uniform查询只有2/67,180个正例；可证明非零远区覆盖，不能支持全几何细粒度重建精度。输入仍是native通道，patch仅定义decoder target。

有效低标签路径检查覆盖4个checkpoint、各10k validation和三套固定presence-preserving donor mappings；Head重放logit误差均为0。10k MAE3的global置换ΔNLL为0.060310，common-support为0.056311。它说明该checkpoint的拟合路径依赖；global分支重训净收益仍未测量。

### Rel标签与新增成本

40个紧凑sidecars覆盖100k train、69,948,793个bar channels，其中34,294,631个为混合贡献；全部匹配对应硬件通道，按相同float32存储比较的最大能量相对差为0。Other贡献能量占比1.7010621×10⁻⁵。构造耗时逐shard相加为1,755.746777秒，峰值RSS 1,608.765625 MiB；sidecars合计1,151,062,225 bytes。前者是任务内耗时之和，CPU allocation core-hours另计，二者不能互换。复用现有ROOT累积贡献，无新增模拟或重编译。

完整channel Rel在原512事件/37,567有效bar通道上的CE为0.642756；同probe逐硬件位置均值先验为0.662919，差0.020163的CI95为[0.011855,0.029564]。事件source置换CE为0.797911。这支持语义学习超越所测几何先验，分类效果仍以完整validation结果判断。

MAE5与Channel Rel5均从MAE3继续12,288次更新、mask epochs3/4，并额外使用98k fit与2k train-inner probe；Rel另外读取模拟语义目标。MAE continuation实际task elapsed为2,388.688287秒（Kubeflow 47GB MIG），channel Rel为2,178.035451秒（Condor 24GB MIG）。设备和共享负载不同，不能由这两个耗时推出Rel更快，也不能把并发task elapsed求和当allocation成本。

### 训练、资源与有效性

修正后的48项train-inner共同配方比较完成137,088更新、2,112,000次曝光和1,732次健康采样，informative前端梯度均为正。两预算重新选择后，1k采用LP-FT/8 passes，10k采用early/6 passes；1k重新训练，10k复用实际训练设置完全一致的六项正常non-LP候选，来源保留在[复用记录](../roadmap/results/s4_transfer4_10k_candidate_reuse.json)。

Matched-supervised另完成12预测/12配对；声明的Linear/attention forward/backward/recompute矩阵操作匹配误差为0.030651%/0.069562%/0.008342%。原parent和continuation计算均计入。该proxy排除elementwise、normalization和compiler fusion；硬件实测等效FLOPs未测量。

最后15项分类训练共有147,130更新、1,710次计划健康采样、210项checkpoint检查和15次独立selected-checkpoint FP32 backward；30个驻点full/global磁盘读出在同一完整10k特征上回放，最大概率差为0。技术记录支持修复后学习路径连通，不替代泛化评价。

114 allocations全部终止，组织成本35.399029353/60 device-hours及423.043923533 CPU allocation core-hours。不同MIG profile分别计时。回放后清理3,643,405,382 bytes临时features/gate缓存和9个AFS终态目录；保留readouts、预测及来源配置，复跑回放需先重建features。本次审查未新增GPU成本。

### 本次验证

直接读取JSON并核对：55个证据路径存在；四项pilot gate、三套MAE3学习readback通过；48项inner及4项完整路径证据齐全；matched评价12预测/12配对且pending为空；最后评价45预测/66配对且pending为0；20项技术检查为true；资源和不扩1M的决定与终态摘要一致。证据表与选定数值保存在`roadmap/results/s4_scope_completion_audit.json`。

本次只改文档与证据索引。两份修改/新增JSON解析通过，4份Markdown的103个本地链接全部存在，`git diff --check`实际exit0。模型测试及实际训练/backward/回放沿用各阶段已运行结果，未为文档修改重复训练。

## 阶段性结论

事实：原S4研究与技术交付已完成。修正后MAE3在有限标签下改善概率和排序，并提供global可读类别信息；额外EMA、MAE continuation及Rel未建立跨预算一致的总体收益。Hadronic τ有条件AP收益，electron/muon子道和高纯度效率改善尚未确认。

固定C=1的1k full读出是已完成诊断的失败概率配方：MAE3 fit NLL0.174783、validation NLL3.246709，2,606个错误预测confidence>0.9，最高1% loss事件仅贡献总损失5.08%。目前假设为高维少标签与正则化设置导致的泛化问题，唯一病因未独立证明。它没有作为可用概率模型晋级，也没有被用来宣布encoder无信息。原合同并未要求所有固定控制配方都取得正收益；后续调C应建立单独优化对照。

本轮无需扩1M。最后五目标筛查为单seed，三seed结果在独立表中；validation已重复用于研究，所有配对区间条件于既定权重与设计。冻结test、校准、系统包络和物理产额未测量，属于S6交付。

## 下一步计划

1. 保留修正后的MAE3及共同Scratch。在各原标签预算内按物理family分组选择readout的C，再用完整预算refit；与fixed-C对照同时报告fit/probe/validation NLL、置信度和排序，验收要求包括原预算约束、驻点与readback，不能只看收敛。
2. S5优先研究τ轻子子道、charm混淆与可见拓扑，先给对象重建efficiency/resolution/coverage，再评价分类净贡献。
3. 若研究global独立贡献，补相同协议下的重训消融；在方法、阈值与校准冻结后按S6执行正式test及可获得的系统变化评价。
