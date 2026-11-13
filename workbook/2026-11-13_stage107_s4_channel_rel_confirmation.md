# S4：pooling前语义路线与完整 frozen seed 对照

## 阶段目标

检验 bar channel 局部信息是否优于块级 pooling；使三 seed frozen representation 比较具备相同 seed 的 random encoder 控制。

## 工作内容与配置

`s4_rel_channel_wave.json` 新增13项任务：从相同 MAE3 seed42 checkpoint 继续训练2 epochs 的 per-channel bar Rel、三标签预算 fine-tuning 与 frozen probes，以及 random frozen seed17/2026 各三预算。主 wave 的 local-block Rel 保留为单变量压缩对照，既有冻结配置不修改。MAE/Rel 两个 continuation 使用相同新 mask epochs3/4、fresh optimizer 与一致 schedule；分类 backbone 不变，decoder丢弃。

首轮 local/channel pilots 都通过，channel CE0.71789485、local0.72199172，常量0.72388333。仅据训练内语义读回设定这项对照，不据未完成的分类结果选模型。`s4_pairs.json` 扩为78项配对、9项三 seed 均值比较，新增的六项均值为三预算 × full/global frozen；独立SSL seeds与同seed random encoder一一匹配。

标签效率图按 full fine-tuning、frozen full、frozen global 分页，颜色对应初始化策略；保留 mean/seed SD，不将三类 readout 画成同一条线。

## 详细数据与验证

完整训练尚未开始，分类收益未测量。上阶段21项测试通过；本阶段增加冻结配置、比较声明、绘图分页，并用同一小型 exact shape table 补算原 Scratch 完整预算，避免为重复分类 epochs 重读原始响应；后续正式规划和统计作业将验证产物。HTCondor紧凑资源确认备份已提交 bigbird21、cluster2198175：2并发/6CPU/12GB，支持原生BF16的20GB以上候选（含24GB H100 MIG）。与whole-GPU确认使用相同互斥claims，先进入运行者获取任务；另一份保持排队时清理。

## 阶段性结论与下一步

目前仅有 Rel 语义和表示谱的训练内证据。完成补充路线、等计算监督控制和完整validation后，报告无收益或收益的具体通道；不扩1M。
