# S3重新打开：架构故障审查与修复合同

原S3的方法结论撤回，旧25模型只保留为架构/训练故障证据。用户要求修复内部故障后重新训练、测试；本阶段先定位问题、检索论文并实现revision 2，科学训练结果尚未测量。修复方案见[roadmap](../roadmap/archive/2026-11-14_v1/10_s3_repair.md)。

## 阶段目标

恢复有效分类学习，分离尺度稳定性与信息压缩影响。正式重训前先用真实train小样本证明拟合能力与encoder梯度连接；发现明显异常必须修复，不再以完成epoch或finite检查代替科学有效性。

## 工作内容

重读local/module/context/latent/head及训练采样器；核对旧FP32逐层诊断。联网读取Peri-LN（ICML2025）、LayerScale（ICCV2021）、PTv3（CVPR2024）、PTv2、NormFormer、QK normalization、DyT（CVPR2025）、LitePT与FASERCal2026 v2。选用Peri-LN、LayerScale、QK-LN和显式Head尺度控制；扩展为随占用增加的regions，并保留全部local block的独立读出。

旧checkpoint加载代码保持，新模型单独版本。训练增加warmup/逐step cosine、norm/bias no-decay、激活/encoder梯度/完整validation多样性监测和错误终止；保留原accepted samples/proxy与冻结calibration/test。Pilot使用32个train事件，不用validation调整修复。

## 详细数据

旧native主模型8事件FP32：local pool RMS101–129，Head Linear min/max−79.890541/−13.976823，GELU零比例1.0，input/local/module/latent梯度平方范数0。证据为旧结果根`diagnostic_factorial_f16_native_best.json`。这是确定的软件/优化故障，不构成硬件信息或模型类别无效的证据。

额外审查发现同一latent context被加到三模态池，包含缺失模态；module embedding及concat边界缺少统一尺度。旧模型已有module pools旁路，不能把36,547→24当作唯一信息路径。确认固定4 module queries的占用瓶颈，采用region及direct-local控制进一步测试。

采样器疑点已实测排除：epoch0 20,016 batch，单类393（1.9634%），最长主导类别连续9 batch，四类齐全1,786；曝光计数[20011,20010,20011,20008]。证据在新结果根`audit/legacy_sampler.json`。本轮不把未证实的class-streak问题写成原因。

新根`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13`只读链接accepted plan/shards/catalog；旧输出保留。新模型单元检查覆盖极端projection尺度下Head规范化与非零encoder梯度、区域覆盖随占用增长、全空/missing与batch isolation、所有local blocks梯度通路、BF16 optimizer。修复真实训练及性能：未测量，等待GPU pilot。

## 阶段性结论

事实：旧主模型在有限精度下GELU全零且encoder断开，之前用它们判断新方法不晋级不合适。Revision 2针对已确认机制建立稳定化控制与压缩控制；性能提升尚未测量。

假设：尺度统一、残差约束及更丰富局部信息路径可恢复学习，需由memorization、全量训练和配对评价验证。不能仅凭论文或单元检查认定修复成功。

## 下一步计划

完成GPU32-event memorization gate与最大占用压力测试；不通过则继续修复。通过后冻结科学矩阵、在GPU并行重训并执行全四类/条件域/内部健康评价，更新最终科学结论。S4–S6未执行。
