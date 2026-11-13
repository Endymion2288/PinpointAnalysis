# S4：continuation 使用新 mask views，Scratch seed42 完成

## 阶段目标

确保MAE continuation的额外训练使用新views，固定内层评价不变；完成seed42三档Scratch参考。

## 工作内容与详细数据

两条MAE continuation共同设置`mask_epoch_offset=3`，以新view epochs3/4训练；模型、optimizer重置和等steps规则不变。SSL inner reconstruction-probe仍固定epoch0，不受offset影响。实际train event0的pixel/bar/tracker hidden counts由20/320/3改为22/313/6，queries与encoder nodes改变；validation event0的754/2194/2及全部query/nodes逐值不变。记录于`audit/continuation_mask_offset.json`。这些检查针对mask调度，不重新认证sample/geometry/proxy。

seed42全部Scratch已完成：

| Labels | Macro AP | Tau AP | NLL |
| --- | ---: | ---: | ---: |
| 1,000 | 0.29720868 | 0.25953440 | 1.51618843 |
| 10,000 | 0.42710017 | 0.32129605 | 1.27706020 |
| 100,000 | 0.46976869 | 0.35133806 | 1.21380233 |

均为固定10k validation、相等class benchmark prior；不是物理flux加权结果。1k/10k checkpoint读回各14项通过；100k读回将在本阶段提交后执行。附加pretraining/Rel收益尚未由这些数值测量。

## 阶段性结论与下一步

Scratch reference已完成首个seed；continuation视图调度已修正并验证。立即推进下一正式Kubeflow wave和HTCondor竞争任务，完成预训练label curves、global/full probes与配对seeds。
