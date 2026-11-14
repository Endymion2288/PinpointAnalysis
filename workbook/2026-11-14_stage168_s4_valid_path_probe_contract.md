# S4 有效低标签模型的分支依赖检查

## 阶段目标

检查有效重拟合是否实际使用local、regional、global和summary分支，补充概率和排序权衡的解释。

## 工作内容

冻结[两项正式配置](../roadmap/configs/s4_transfer4_functional.json)，分别检查1k/10k seed42的Scratch与MAE3，共四个已通过readback/FP32的有效checkpoint。沿用阶段145实现，完整10k validation上做三套presence-preserving donor mappings，置换指定分支，在同一FP32 baseline下求差；400次128物理块bootstrap，报告raw/common-support。

## 详细数据

FP32禁用TF32和compile，保持原ragged batch形状、recipient顺序与Head replay数值一致。不给模型重新拟合，不用置换输出重新选择模型或阈值。检查all local、all regional、global latent、response summary、all spatial等现有分支。四个模型的本轮分支差尚未测量，阶段145的旧有效基线分支结果保留作独立参考。

为避免CPU solver gate等待期间GPU闲置，可用Kubeflow正式两并发执行；若先获得Condor的兼容FP32 GPU，使用shared claims跳过重复任务。全部allocation继续计入原60 device-hours。

## 阶段性结论

该检查衡量固定checkpoint对各分支的依赖，跨分支相关性会被打破，不能解释为重训ablation收益或因果物理重要性。结果未测量。

## 下一步计划

执行两项正式检查，保存真实终态后释放资源。CPU solver gate通过后转入新的feature bank和CPU驻点求解；合并完整有效结果。
