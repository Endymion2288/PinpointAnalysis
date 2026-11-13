# S4：分支可读性、EMA 特征谱和独立种子对照

## 阶段目标

明确 S3 global 分支的信息是否可读；核对增大 variance/covariance 约束后的 EMA 状态；将关键 Scratch 配对种子预先冻结，并增加完整评价管线检查。

## 工作内容与实际数据

产物根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。本阶段没有重新认证样本、split、geometry 或 detector-proxy。

`experiments/s4_s3tracker_branchprobes_s42/`：固定 10k train 中 8k readout-fit、2k readout-probe。上游 S3 supervised encoder 已见部分标签，因此这些结果只表示可读性，不能当独立泛化结果。

| Branch | Linear macro AP | MLP macro AP | Linear NLL | Centered effective rank (2k probe) |
| --- | ---: | ---: | ---: | ---: |
| regional | 0.392433 | 0.380326 | 1.319382 | 3.45013 |
| global | 0.354040 | 0.327226 | 1.347126 | 2.78739 |
| local | 0.386094 | 0.387713 | 1.350848 | 5.33439 |
| summary | 0.412501 | 0.413171 | 1.271696 | 4.19564 |
| full | 0.393321 | 0.417233 | 1.424337 | 5.69736 |

标准尺度约束（gamma=1、variance weight=25、covariance weight=1、reconstruction weight=1）下，1024-step、batch16 的两个 EMA pilot 均已完成。512 个原 train 内 reconstruction-probe 事件的实测如下：

| Teacher target | Fine-patch AP | 4-mm AP | Bar AP | Student global rank | Teacher projection rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| global | 0.064245 | 0.225790 | 0.522825 | 4.36484 | 2.43087 |
| multiscale | 0.075001 | 0.228308 | 0.518382 | 4.11434 | 1.79175 |

旧弱约束 global teacher 的 rank 为 1.08056。增加约束后 global target 已改善，但 96-dimensional projection 仍明显各向异性。multiscale teacher 没有解决这一问题，不能将其作为优胜方案扩量。本阶段保留 global-only target，并新增 batch64 的真实 pilot，检验更可靠的事件间 covariance 统计。该对照同时增加事件曝光量，报告将明确列出 steps、events 与资源，不据此单独归因 batch size。

`pretrain.feature_statistics` 增加最大方差特征值占比、前八项谱及解释 90% 方差的维度数；完整 evaluation 同时统计 student projection。已有冻结任务保持原代码快照。完整 SSL 任务结束后必须进行 checkpoint/learning readback，通过后才启动下游。

`roadmap/configs/s4_seed_wave.json` 提前固定 Scratch 的 17/2026 两个额外种子、三个标签预算的随机冻结 encoder linear probes，以及上述 batch64 EMA pilot。当前 core wave 继续运行；未完成的四分类训练不写成最终结论。

## 文献依据和验证

已联网读取 [VICReg 原文](https://arxiv.org/abs/2105.04906)、[官方实现](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py)、[Sonata 官方实现](https://github.com/facebookresearch/sonata)，并获取 [LeJEPA](https://arxiv.org/abs/2511.08544) 全文，缓存于 `literature/repair_sources.json`。VICReg 原文 Eq.(6) 对两条可求导分支分别约束；当前 EMA + cosine + reconstruction 的适配与原始 VICReg 不同。原文也指出 stop-gradient 与 covariance 联合优化存在困难。LeJEPA 对 complete/dimensional collapse 的区分支持增加谱诊断；本阶段未实现或声称复现 SIGReg。

执行 `python -m unittest tests.test_flavour_s4`：12 项通过。新增检查确认 rank-one 输出即使每维方差非零也会被谱统计识别；完整评价测试覆盖 SSL 过滤、相同事件配对、common support 和 paired zero contrast。S4 真实输出与配置均保存在 EOS，仓库不收录 checkpoints。

## 阶段性结论

事实：global 分支存在可读分类信息；EMA 加强尺度约束后谱有所改善，multiscale teacher 未显示优势。假设：小批量 covariance、EMA 滞后及低维 occupancy 捷径可能共同限制表征，batch64 pilot 和后续 frozen probes 将继续检验。下游预训练收益、1M 扩量决定目前均未测量。

## 下一步计划

完成完整 MAE 和下游 label curves；从相同 MAE checkpoint 分叉 continuation/partial bar Rel；根据谱与重建 readback 选择 EMA 后续；完成全量 validation、配对多 seed、matched compute 与成本分析。
