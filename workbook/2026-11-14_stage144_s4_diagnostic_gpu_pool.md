# 固定权重诊断的FP32资源池

## 阶段目标

在保持18项BF16修复训练和完整validation协议的前提下，减少四模型分支诊断的GPU排队时间。

## 工作内容

为正式Condor campaign增加`cuda-fp32-diagnostic`池，只接受单并发、`program=functional_probe`且`device=cuda`的任务。它可匹配当前LCG108/PyTorch支持的T4、A100、H100整卡或MIG、H200，至少10GB显存；V100/sm70继续排除。该任务原定FP32、关闭autocast/TF32和compile，扩池不更改精度或统计合同；BF16训练不能进入此池。

## 详细数据

00:57 UTC的Condor资源快照包含T4 15个、A100 40GB 87个、H100整卡描述84个、H100 12GB MIG 15个、24GB MIG 17个、H200 48个及V100/V100S描述89个；这些是slot广告中的GPU描述计数，不代表独立可用物理卡。未发现处于Unclaimed的可用候选，不能据此保证获配。00:53 UTC的compact候选匹配分析显示0个愿意运行该任务的slot。证据：输出根`audit/gpu_resource_properties_20261114T0057.json`、`audit/compact_queue_analysis_20261114T0053.txt`。

Kubeflow新增A100 dry-run被namespace总GPU单位上限1拒绝，当前47GB MIG已占用该单位；已保存拒绝原因，实际分配时长为0。现有6并发Pod及两组2并发Condor训练继续运行。00:54:49 UTC总账为22.070602/60 device-hours，261.764854 CPU core-hours。

同一已接受catalog的描述统计：train 100,000事件中2,790个三模态全空；validation 10,000事件中282个全空，类别计数为56/62/90/74。若所有空输入对确定性模型相同，其经验条件entropy为1.369617 nats，对全validation NLL的经验下界贡献为0.038623；即使全部非空事件分类完全正确，空输入造成的经验accuracy上界仍为0.9808。这不是总体Bayes error测量，未改变样本选择或训练，完整口径见`audit/empty_response_observation_stats.json`。

最大train事件压力测试的36,549个原生节点全部保留，经14,599个local blocks得到481个regions、962个regional tokens，全部local blocks直接参与readout，另有64个global latents。该表示仍有池化压缩，不称无损。空事件占位硬件分支的梯度为0，事件本身保留用于学习空响应类别概率。

验证采用源码编译和实际submission dry-run检查；正式新池的任务执行与获配尚未测量。当前18项分类的完整最终评价仍未测量。

## 阶段性结论

固定权重FP32诊断可以扩大到T4而不改变BF16训练合同。全空事件只占validation 2.82%，不足以解释总体泛化误差；过拟合与架构数值健康分别报告。

## 下一步计划

在正式GPU allocation上完成4个checkpoint的完整10k分支诊断，与18项训练的严格评价一并验收；保存终态与实际资源成本，随后清理已无用候选。
