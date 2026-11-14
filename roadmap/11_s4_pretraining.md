# S4：可验证的预训练与标签效率

**最终状态：S4本轮已完成，见[阶段181](../workbook/2026-11-14_stage181_s4_final_results.md)。15项seed42分类训练、30个驻点full/global读出、45套完整10k预测和66项配对完成，20项检查通过；[精简数值](results/s4_fp64_objectives_summary.json)及[终态清理](results/s4_final_closure.json)已保存。114条allocation全部终止，35.399029/60 device-hours、423.043924 CPU allocation core-hours。MAE3保留为后续基线；低标签固定C=1的概率失败另列诊断，本轮不扩1M。**

原六项执行要求、遮蔽/重建/Rel合同及资源交付已逐项审查，见[阶段182](../workbook/2026-11-14_stage182_s4_scope_completion_audit.md)和[证据映射](results/s4_scope_completion_audit.json)。该审查包含早期pilot、train-only分支probe、几何控制、三seed与matched-compute，覆盖最后20项技术检查以外的研究交付。

**2026-11-14 纠错与有效结果：阶段155发现LP解冻后input/local编译断梯度，旧12项LP inner及旧1k优化模型作废，见[失效清单](results/s4_transfer_compilation_invalidation.json)。修复后真实CUDA gate通过，48项有效inner完成并重做两预算选择：1k LP-FT/8 pass、10k early/6 pass。新的三seed完整10k评价和实际参数更新均通过，见[阶段167](../workbook/2026-11-14_stage167_s4_valid_low_label_results.md)：1k MAE3 NLL 1.447382→1.358905、macro AP 0.345980→0.330614；10k MAE3 NLL 1.252464→1.238574、macro AP 0.440384→0.448223。概率与排序权衡分开解释，旧故障模型未进入以上结果。100k frozen full读出达到5,000次L-BFGS上限，旧完整readability评价已因上游需审查而停止；同目标Newton-CG、真实驻点验收及GPU/CPU拆分恢复见[阶段166](../workbook/2026-11-14_stage166_s4_convex_readout_recovery.md)，同目标v2真实回归四项通过；最终100k设备接续已完成，见第7节，中间执行见[阶段174](../workbook/2026-11-14_stage174_s4_stationary_gate_passed.md)。阶段145有效基线不受LP故障影响。**

阶段145结果：空响应修复矩阵已完成18项分类重训、完整10k validation及4模型分支诊断，21项技术检查全部通过。235,008次逐步梯度检查、2,916次激活采样、252项checkpoint检查和18次独立best FP32 backward通过；最大preclip norm为188.390472，未重现旧空占位约10¹²的异常。两组同时使用`empty_response_bypass=true`；新配置为[configs/s4_empty_repair.json](configs/s4_empty_repair.json)。MAE3复用此前已读回的3套encoder重新fine-tune，本轮没有重训SSL encoder本身。旧故障结果保留作诊断，不作为新结论。完整结果见[阶段145](../workbook/2026-11-14_stage145_s4_empty_repair_final.md)、[精简数值](results/s4_empty_repair_summary.json)和[分支诊断](results/s4_empty_functional_paths_seed42.json)。输出根为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`；阶段145时原61个allocation已终止，后续修复与对照继续累计同一账本。03:49:08 UTC历史快照为30.386366/60 device-hours，当时仍有训练与评价运行；最终费用见页首。本轮不扩1M，优先解决低标签过拟合和τ决策问题。

## 1. 研究问题与固定条件

1. S3 的 global 分支虽有非零梯度，tracker 模型中置换 global 输出的 probe NLL 增量为 −0.001597；预训练能否学到可线性读出的跨探测器表示，并在实际分类中得到利用？
2. tracker 对 νμ、NC 有增益，ντ AP 增量的区间跨零。预训练是否改善 ντ 及其 e/μ/had 子道的混淆，还是主要学习 shower occupancy 和能量先验？
3. 低标签收益能否覆盖新增资源成本？同样预算继续监督或 MAE 是否已能取得相同结果？

沿用已接受的 detector-proxy、geometry 和 family split。既有 80k train、10k validation 原样保留，另从八个新 chunk/类选择 5k train/类，组成 100k train。所有 SSL 输入、重建验证和 probe-fit 均来自 train；独立 validation 只用于下游选择/评价。Calibration/test 不打开。

原始S4合同的共同backbone为S3 revision 2 multiscale native + tracker：Peri-LN、QK-LN、LayerScale、FP32 normalized SiLU Head、全部local blocks直接读出。本次18项分类同时采用revision 3的`empty_response_bypass=true`，由Scratch控制隔离预训练初始化的差别；原冻结配置用于复现历史。导出features的接口与checkpoint参数命名兼容，预训练decoder/teacher在fine-tuning时丢弃；分类heads对所有初始化策略使用相同种子重新初始化。

## 2. 文献依据与适配

| 来源 | 采用的方法 | PINPOINT 的适配 |
| --- | --- | --- |
| [FASERCal, arXiv:2604.07037v2](https://arxiv.org/abs/2604.07037) | MAE、Rel、同下游 Scratch 对照、标签效率 | binary pixel 不预测 analog charge；Rel 仅使用可精确累加的 bar 贡献，明确记为 partial Rel |
| [NOMAE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Abdelsamad_Multi-Scale_Neighborhood_Occupancy_Masked_Autoencoder_for_Self-Supervised_Learning_in_LiDAR_CVPR_2025_paper.html) | visible-neighborhood occupancy、正负候选、多尺度 | 查询位置独立于 hidden support；额外加入跨层和全几何 uniform 候选，保留孤立短轨的非零覆盖概率 |
| [Sonata, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.html) | self-distillation、geometric shortcut 诊断、linear probe | teacher 仅看完整硬件 observation；不任意旋转或缩放探测器，不将 τ daughter 删除后的视图当作等价完整分类事件 |
| [VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906) | feature variance/covariance 约束和坍缩监测 | 与 EMA 对齐共同使用，检查各分支跨事件方差、有效秩和梯度，不能仅凭 loss 下降验收 |

2026-11-13 已联网获取上述 CVPR 原文、VICReg 原文及 CERN ML training/flavor 文档，来源与时间保存在输出根 `literature/sources.json`。FASERCal 全文与 S3 稳定性文献复用既有缓存。

## 3. 遮蔽、重建与 Rel 合同

遮蔽先作用于 observation，再重新建立 local/region tokens、presence 和全部 summary。遮蔽规则由固定几何 cell ID 和 epoch seed 决定，对空/有响应通道一致；不把 hidden active 坐标、总数、模态总信号或原始 group layout 提交 encoder。先比较 0.5/0.75，增加整 plane/panel 和 detector 遮蔽；不以 truth 选 mask。

Decoder queries 仅由几何和 visible 邻域产生，混合邻近、跨层与 uniform 候选；候选集合先生成，再查询 reconstruction targets。必须报告每种候选的正例比例、loss、PR-AUC、reconstruction coverage，声明优化的是所定义混合查询分布，不能称为均匀全探测器 risk。bar signal 只用接受的能量/PE proxy。负候选不能按 hidden support 筛除。直接依赖 visible geometry 的 trivial baseline 与 encoder-permutation baseline 同时报告。

首轮 512-step pilot 中，256 个 train-inner probe events 的逐 native pixel 查询仅有 2–3/16,384 正例，不能用其 loss 下降支持细结构学习。修正后的目标为 16×16 native-channel patch occupancy、binary channel count、patch 内 native x/y 位置矩，另保留 4-mm occupancy/count；**encoder 输入仍为逐通道原生观测**。64-event 同查询/同 masks 对照的 native/8×8/16×16 正例数为 0/28/83。新的监督尺度需重新通过真实事件拟合与 corpus pilot，不沿用原目标的训练权重。逐 native pixel target 的统计不足作为明确的实现/物理分辨率约束保留。

EMA 单独增加 teacher-multiscale 对照：完整 observation 的 local/regional/global pooled features 等权形成 teacher target，student 仍从 global latent 预测，检验 global-only EMA 是否只重复 occupancy 主轴。与原 global-only teacher 分开运行；不将多个目标改变合并归因。

`ScintSD.cc` 的 `barEnergyMap` 以 channel/track/PDG 为 key 累加；ROOT `Hits/scintHits` 保存相应 `pdg/edep`。额外 sidecar 按硬件 channel 累加 EM(11/22)、muon(13)、hadron、other 的能量分数，正能量有效通道才产生目标。质量检查仅针对新增标签的 channel join、能量一致和 simplex，不重复样本认证。Pixel 首贡献 flags、ghost/hierarchy 不冒充精确 soft truth；本轮不用 pixel Rel。原 global-only Rel pilot 的语义 cross-entropy 为 0.73062，未超过常量 probe prior 的 0.72388，已停止扩量；新 Rel route 允许读取全部 kept bar local blocks，必须先通过独立语义学习 readback（见 stage96）。MAE reconstruction 仍只读取 global latents。模拟贡献只进 loss，不进入 input、mask 或 query 位置。

MAE/EMA仅从observation构造目标；partial bar Rel另使用98k pretrain-fit事件的模拟channel语义，及2k原train内probe的语义评价。Rel衡量事件类别标签效率与额外模拟语义监督，不能称为仅用1k/10k总标注的纯自监督比较。相同continuation steps/views不等同于相同FLOPs，附加decoder和语义目标的计算量另列。

## 4. 分阶段实验与预算

S4 本轮组织上限 **60 device-hours**，包括所有后端的 pilot、编译、失败、训练、probe 和评价；不同设备、MIG与整卡分别列出，时长不换算成等价FLOPs。最终35.399029 device-hours、423.043924 CPU allocation core-hours，包含历史尝试、初始化、失败与保守结束上界。基线、有效低标签重选、matched-compute及完整目标/readout比较均完成。

1. 完成 100k corpus、独立 Rel sidecars；train 内嵌套 98k pretrain-fit/2k reconstruction-probe。监督标签预算为嵌套的 1k/10k/100k，四类平衡，τ 子道按原样本比例分层。
2. 导出 S3 encoder 的 local/regional/global/summary features，在 train 内 fit/probe-validation 上分别拟合 linear/MLP，测可读信息与实际梯度的差别。不能用 64-event 置换取代分类 probe。
3. 真实 32-event 拟合、FP32/BF16、最大 occupancy forward/backward、无泄漏、负候选、EMA/variance、checkpoint/resume 通过后，正式 pilot 比较 mask 0.5/0.75。重建内部验证及分布/coverage 决定主 mask；全零预测不作通过结果。
4. 同 backbone 比较 Scratch / MAE / EMA self-distillation / MAE+barRel；另保留 MAE continuation 和继续监督的 matched-compute 控制。相同 fine-tuning 设置与独立初始化 heads。先 seed 42，关键候选与 Scratch 用 17/42/2026 确认。
5. 每个标签预算同时交付 frozen linear probe 和 full fine-tuning、learning curves、checkpoint 读回、内部健康和总成本表。完整 validation 为 10k，四分类指标、各类 1%/0.1% FPR 效率、τ 子道及 common-support 条件指标齐全。0.1% FPR 的低通过背景数按实测计数报告。
   补充固定 checkpoint 的完整 10k validation 分支置换：三个预先固定的 modality-preserving donor mappings，逐项 global/regional/local/summary，并以同精度 baseline 和 paired chunk bootstrap 比较。用于检验实际拟合路径依赖，结合 frozen readability 解释；不当作重训 ablation 或物理因果重要性。
6. 只有重建健康且下游收益与资源证据支持时才扩至 1M；无收益也完成本轮对照与统计结论，不盲目扩量。1M 是否执行及原因写入最终验收。

## 5. 调度与验收

批量转换只用 HTCondor；选择 schedd 时保存即时 load、EOS policy、WRITE 结果。训练可由 HTCondor/Kubeflow 竞争领取互斥 task claim；使用平台规定的 GPU product/resource flavor，不在 GPU Pod 中自填 CPU/RAM。并发数以真实 CPU/data wait、显存和 GPU utilization 调整。正式长训练不得从 SSH 临时启动或暂停 PID 1 进行迁移。

验收除结果文件完整外，要求 informative encoder 梯度、非饱和 head、非恒定预测、SSL 的 feature 方差和实际拟合证据；若出现明显异常，先定位架构/尺度/采样/优化并修复重训。最终记录所有 allocations 终态和资源成本；阶段报告与代码共同提交。

当前配对冻结于[configs/s4_empty_pairs.json](configs/s4_empty_pairs.json)，全部9项配对和3项三seed均值已完成。MAE3−Scratch的raw结果如下，区间条件于已训练seeds和固定validation；训练seed SD在精简JSON中单列，均值不是ensemble。

| Labels | Macro AP: Scratch → MAE3 | Δmacro AP [CI95] | NLL: Scratch → MAE3 |
| ---: | --- | --- | --- |
| 1,000 | 0.288856 → 0.345980 | +0.057124 [0.052411, 0.062256] | 1.564575 → 1.447382 |
| 10,000 | 0.425891 → 0.440384 | +0.014493 [0.010374, 0.018721] | 1.276154 → 1.252464 |
| 100,000 | 0.469263 → 0.473555 | +0.004292 [0.002020, 0.006590] | 1.214263 → 1.207805 |

1k平均NLL仍高于ln4=1.386294；1k/10k的τ argmax F1差值分别为−0.032139/−0.011317，物理块区间为负，且seed波动较大。三个预算的τ低FPR效率差值区间均跨零。100k raw τ AP差值+0.006024，[0.001484,0.010809]；common-support总体τ AP及macro AP收益未确认。Hadronic τ子道在common-support的AP差值为+0.015837/+0.009678/+0.004792，区间均为正，electron/muon子道未确认改善。排序收益不能替代概率、硬决策和高纯度工作点质量。

新MAE3 seed42在1k/10k完整validation中，global置换的NLL增量为0.055214/0.025673，区间均为正；Head重放误差均为0。Scratch 10k对应效应只有0.00002515，实际很小。Local/regional与summary诊断保留在完整表中。这证明特定checkpoint的拟合路径依赖，不是重训ablation或物理因果重要性。最大train事件36,549个原生节点、14,599个local blocks、962个regional tokens全部参与，另有64个global latents；池化仍是压缩，不称无损。

FASERCal式partial Rel语义学习的前期诊断见[stage119](../workbook/2026-11-13_stage119_s4_full_rel_event_information.md)，与本次分类比较分开。旧矩阵及其解释只保留于标明状态的阶段记录，不复用为新路由的收益证据。Calibration/test继续冻结。

## 6. 修复后优化结果与剩余对照

有效的[48项train-inner矩阵](configs/s4_low_label_inner_v4.json)采用四种共同配方：early、低encoder LR、强正则、LP-FT，各预算/初始化三折。修复后的12项LP与36项正常non-LP共同完成137,088次更新、2,112,000次曝光和1,732次健康采样，所有informative FT的input/local梯度为正，见[健康汇总](results/s4_transfer4_inner_health.json)。旧LP矩阵的同名选择与checkpoint继续作废。

按原train-inner共同OOF NLL规则，1k选择LP-FT/8 pass，Scratch/MAE3 OOF NLL为1.448791/1.389765；10k选择early/6 pass，为1.302972/1.277887。两者均在全部原nested-label预算内选择，不增加事件类别标签。1k六项修复refit已重新训练；10k六项正常候选与新选择的实际训练设置完全一致，保留原配置与成本复用，见[复用证据](results/s4_transfer4_10k_candidate_reuse.json)。

两预算三seed完整评价、实际参数更新和独立FP32均通过，见[阶段167](../workbook/2026-11-14_stage167_s4_valid_low_label_results.md)与[精简数值](results/s4_transfer4_low_label_summary.json)：

| Labels / initialization | Baseline → optimized macro AP | Baseline → optimized NLL | Interpretation |
| --- | --- | --- | --- |
| 1k Scratch | 0.288856 → 0.307071 | 1.564575 → 1.407840 | Probability/ranking improve; NLL remains above ln4 |
| 1k MAE3 | 0.345980 → 0.330614 | 1.447382 → 1.358905 | Probability improves; ranking decreases |
| 10k Scratch | 0.425891 → 0.423526 | 1.276154 → 1.272265 | Small changes, paired intervals cross zero |
| 10k MAE3 | 0.440384 → 0.448223 | 1.252464 → 1.238574 | Both improve in all three seeds |

优化MAE3相对共同Scratch的τ F1在三个seed均有正增量，但0.1% FPR仅7个背景、1k MAE3只保留0/1/1个τ，10k为8/6/4，仍不能作稳定高纯度结论。Common-support的τ AP相对基线增量区间仍跨零。完整raw/common-support、τ子道和逐类计数保存于冻结评价；seed均值不等同ensemble。

新有效seed42分支检查完成，见[分支实测](results/s4_transfer4_functional_paths.json)：10k global置换的NLL增量Scratch为0.000085，MAE3为0.060310，后者common-support为0.056311；local/regional也影响MAE3预测。完整10k、三套presence-preserving映射和400次物理块bootstrap支持当前checkpoint确实使用这些分支；它们是拟合路径依赖，不能解释为重训ablation或物理因果重要性。

修复后的监督matched-compute控制已完成12预测/12配对，见[阶段163](../workbook/2026-11-14_stage163_s4_compute_control_results.md)。声明矩阵操作匹配误差为0.030651%/0.069562%/0.008342%，所有parent计算计入。1k/10k continuation过拟合，100k相对Scratch的macro AP变化+0.000157、区间跨零；三个监督envelope均保留原parent，同时计入全部continuation成本。该结论只覆盖固定seed42及所测监督配方，不代表最优监督调参或硬件实测FLOPs等效。

EMA3、MAE5、partial channel Rel5的三个预算下游训练均已完成；10k正常候选按新共同选择复用。所有来源SSL已通过健康/学习readback；channel Rel使用额外模拟语义监督，不能称纯自监督标签效率。最终目标比较冻结于[30配置/45预测](configs/s4_fp64_objectives_evaluation.json)及[66配对](configs/s4_fp64_objectives_pairs.json)。

Frozen readout的L-BFGS max_iter从500提高到5,000后，1k/10k可收敛，100k full Scratch/MAE3仍触及上限，旧完整readability评价已停止。首轮同目标回归中Newton-CG全部满足1e-6梯度要求，但L-BFGS参考梯度超过原1e-8要求。[同目标v2回归](configs/s4_newton_solver_gate_v2.json)为参考增加`ftol=0`的continuation，保持原阈值和目标；四项真实回归已全部通过，参考梯度9.87e-10至4.88e-9、最大概率差4.93e-5，见[阶段174](../workbook/2026-11-14_stage174_s4_stationary_gate_passed.md)。[五项GPU暂存features](configs/s4_newton_v2_feature_banks.json)已在同一H10047GB MIG五并发，各提取一次100k train/10k validation；[15项CPU readouts](configs/s4_newton_v2_readouts.json)按精确nested-label事件键分别拟合；GPU不等待CPU求解。最终100k采用已过真实gate的设备FP64执行，独立CPU驻点全部通过；1k/10k复用原CPU fits。暂存数组已在完整验收后删除，保留小型readouts、预测和provenance；未收敛旧模型仅作诊断。

本轮不扩至1M。目标对照、驻点readout、完整条件统计、真实allocation终态和总成本已完成。最终100k采用真实8项CPU/GPU gate通过的FP64设备Hessian products，五组full/global独立CPU梯度均≤1e-6，保持原C/scaler/凸目标。原耗时CPU任务在替代产物通过后正式终止，费用保留，旧配置不改写。

完整评价已重新加载30个full/global磁盘readouts，在同一10k validation特征上按事件键回放概率，最大绝对差为0，满足≤1e-10的要求；回放合同见[阶段173](../workbook/2026-11-14_stage173_s4_saved_readout_replay.md)，实际结果见阶段181。回放与真实梯度驻点分别保存，临时feature数组已在通过后清理；重新执行回放需按冻结配置重建缓存。本步骤没有重新拟合或调整模型。

固定C=1的1k MAE3 full已确认概率泛化失败：fit NLL0.174783、validation NLL3.246709，10k中2,606个错误预测confidence>0.9；top1%事件仅贡献5.08%总损失，不能归为少数离群点。见[概率诊断](../workbook/2026-11-14_stage177_s4_probability_diagnostic.md)和[精简数值](results/s4_frozen_probability_diagnostic.json)。这是数值收敛后的固定读出设置问题，不能作为encoder无信息的结论。该配方保留作诊断，后续readout优化只在各原标签预算内按物理分组train-inner选择C，并独立记录与固定配方的差别；本轮不在观察validation后静默修改目标。

优化设计已观察前期S4 validation，后续区间条件于该设计、同一validation与既定权重；不能描述成新的独立泛化证据。Calibration/test及最终class bias、温度与工作点选择继续冻结。

2026-11-13 23:18 UTC重新联网读取[Kumar等，ICLR 2022 LP-FT](https://arxiv.org/abs/2202.10054)及[Wortsman等，CVPR 2022 WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)，源文件与URL记录位于输出根`literature/low_label_transfer_sources.json`。LP-FT给出随机Head与encoder同时更新可能扰动预训练特征的可检验假设；其论文主要研究OOD，当前数据尚未证明这种病因。WiSE-FT依赖可用的zero-shot分类模型，当前没有对应分类Head，暂不直接采用权重插值。前述CVPR 2025 Sonata/NOMAE和FASERCal仍为主要预训练依据。

## 7. 最终目标与读出结论

最后的五初始化比较仅为seed42，三seed低标签证据在第6节另列。MAE3−Scratch的raw macro AP增量在1k/10k/100k为+0.033039/+0.025899/+0.005324，CI95均为正；NLL增量为−0.055747/−0.036485/−0.007624，均为负。100k common-support macro AP区间跨零，macro F1和balanced accuracy也未确认提高。Hadronic τ子道在common-support的AP增加，electron/muon子道及低FPR效率收益未确认。

EMA3相对MAE3在三个预算的macro AP/NLL均退化，raw/common-support区间支持该差别。MAE5相对MAE3未建立raw总体排序收益；Channel Rel5在1k提高macro AP但降低τ AP，100k只确认raw NLL小幅降低。Rel的global线性可读性在10k/100k提高，完整fine-tuning没有相同一致收益。保留MAE3作后续有限标签基线，不根据最低单点NLL扩量。

100k frozen global的MAE3 macro AP为0.404062，Scratch为0.340248，差+0.063814 [0.056307,0.072226]。这是固定线性配方的可读信息证据，结合完整10k路径置换解释具体checkpoint；global重训净收益仍未测量。固定C=1的1k full广泛过度自信保留为失败诊断，后续在原标签预算内分组train-inner选择正则化。

15项分类的147,130次更新、1,710次计划健康采样、210项checkpoint读回和15次独立best FP32 backward通过；30个磁盘readouts完整10k回放误差为0。20项[收尾检查](results/s4_fp64_objectives_acceptance.json)全部通过后，清理3,643,405,382 bytes临时feature/gate缓存和9个终态AFS目录。小型readouts、预测与来源manifest保留；重新回放须按冻结配置重建缓存。完整图、计数区间、资源分项及复现命令见阶段181。
