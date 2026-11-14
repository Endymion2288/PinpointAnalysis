# S4 修复后10k共同重选与候选复用

## 阶段目标

完成12项新LP候选，重新共同选择10k recipe/epoch，以有效选择决定重拟合和完整评价。

## 工作内容

Kubeflow `pinpoint-s4-transfer-recovery-v4` 已成功终止，Job completionTime为04:05:46 UTC，全部13项任务exit0，包括CUDA gate和12项LP训练。Job/Pod真实成功终态已保存到输出根 `audit/backends/` 后删除Job，释放GPU配额。

新10k六项LP均完成12个complete passes，并通过resolver对配置、实际曝光和input/local训练梯度的检查；与36项正常non-LP及新1k六项LP组成完整有效48项内折矩阵。重新选择不读取独立validation。

## 详细数据

10k仍选中 **early、6 complete passes**，mean OOF NLL为 **1.290429473**，Scratch **1.302972206**、MAE3 **1.277886741**；见[完整候选表](../roadmap/results/s4_transfer4_selection_10k.json)。原已完成的六项10k non-LP refit与此次新选择的全部实际训练设置逐项相同，只有任务名、选择文件路径和选择记录属于元数据差别。因此保留其原name/config/manifest/code_commit与成本，作为有效选择后复用的候选，不重复训练，见[逐项证据](../roadmap/results/s4_transfer4_10k_candidate_reuse.json)。

每项候选原先已完成6×635=3,810次更新、60,000次完整曝光，固定第6轮；还须通过新的参数差分与完整评价。旧故障1k模型及旧LP选择仍作废。新拟议名称仅存EOS审计，实际[refit合同](../roadmap/configs/s4_low_label_refit_v4_10k.json)保存被复用的原冻结配置。

冻结[12模型评价](../roadmap/configs/s4_transfer4_10k_evaluation.json)、[配对与seed均值](../roadmap/configs/s4_transfer4_10k_pairs.json)、[CPU任务](../roadmap/configs/s4_transfer4_10k_validation.json)。与1k相同，重复使用原validation的推断保持条件性，calibration/test冻结。

准备[17项后续dispatch](../roadmap/configs/s4_transfer4_refit_and_readout_dispatch.json)：九项有效1k refits/目标对照及八项正常100k/readout恢复，六并发，用claims自动跳过Condor已领取或完成的任务。100k EMA3/MAE5已在Condor运行，保留其正式allocation。

新增CPU参数检查stage同时登记到资源账本的CPU列表，避免把CPU等待误计为GPU分配。4项accounting回归测试通过，覆盖CPU-only计费、完整Condor终态恢复和Pod清理后的真实终态上界。此时新10k完整分类统计尚未测量。

## 阶段性结论

事实：两预算均在修复后的有效竞争矩阵上重新共同选择，1k为LP-FT/8 pass，10k为early/6 pass。10k六项已完成候选可以按完全相同的实际训练协议复用，选择确认与训练成功分别保留证据。

## 下一步计划

正式启动后续Kubeflow六并发和10k CPU完整评价，补齐1k三seed、目标对照、frozen readout及参数验收。完成物理统计、成本与终态后才结束S4。
