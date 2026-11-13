# S4 Scratch 标签预算对照

## 阶段目标

在 SSL pilots 完成时并行推进同 backbone 的直接监督 learning curve，形成后续所有初始化的相同训练设置与完整验证基准。

## 工作内容

固定 `s4_scratch.json` 三项 seed 42 配置。Backbone 和 observation 与已通过真实拟合 gate 的 S3 revision 2 multiscale+tracker 相同；train label budgets 嵌套、类别平衡，τ 子道按比例分层。完整 10k validation 不变，checkpoint 按 validation NLL 选择。

## 详细数据

| Label budget | Epochs | Optimizer steps/epoch | Batch size | Warmup steps |
| --- | --- | --- | --- | --- |
| 1,000 | 8 | 512 | 16 | 250 |
| 10,000 | 8 | 1,256 | 16 | 250 |
| 100,000 | 4 | 6,256 | 16 | 250 |

小标签组在同一 label subset 内重复 sampling，不将 exposures 当作额外独立事件。所有初始化策略在相同 budget 采用本表；AdamW lr 3e−4、weight decay 0.01、gradient norm clip 1.0、warmup/cosine、BF16 encoder/FP32 head。现有 32-event 分类 gate accuracy 1、NLL 2.4087e−5；上述正式训练的指标尚未测量。

## 阶段性结论

该配置冻结下游设置，后续预训练比较不再按单个方法修改 fine-tuning 步数或选择规则。不同 label budgets 的 exposures 有意不同，资源对比必须同时报告实际成本和重复次数。

## 下一步计划

正式 batch 提交 Scratch 三项和 train-only branch probes，利用 HTCondor/Kubeflow 独立 allocation。SSL 通过后使用同样设置进行初始化对照；再追加 matched-compute continuation 与关键结论三种子验证。
