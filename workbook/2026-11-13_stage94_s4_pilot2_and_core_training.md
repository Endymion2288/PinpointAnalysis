# S4 第二轮重建结果与正式训练

## 阶段目标

选定具有实测重建信息和成本依据的主 MAE 目标，推进完整预训练和标签曲线；识别 EMA 的维度坍缩后先修复再扩量。

## 工作内容

第二轮五项 512-step corpus pilots 已完成。32-event 优化、最大 occupancy stress、sampled gradients 及 checkpoint 读回通过。正式 MAE 采用 75% mask、16×16 patch/count/native-centroid + 4-mm/count + bar occupancy/signal，从新初始化训练 3 epochs × 6,144 steps，不复用 pilots 权重。50% 保留诊断，不声称 75% 在质量上优于 50%；选择 75% 基于有学习证据且 measured training throughput 更高。

EMA 的 teacher projection effective rank 接近 1，global-only 与 multiscale teacher 都未解决。即使各维方差非零，仍可存在维度相关坍缩。本轮不据此扩到 full EMA，先将方差/协方差设置改为 VICReg 原文尺度：γ=1、variance coefficient 25、covariance coefficient 1，同时 reconstruction anchor weight 1；两种 teacher 分开跑 1,024-step 修复 pilot。此项为根据内部结果作出的架构/优化目标检查，旧 EMA 不作为最终推荐。

## 详细数据

512-event train-inner probe，每类 query 32,768：

| Strategy | Fine-patch AP | 4-mm AP | Bar AP | Global rank | Train events/s |
| --- | --- | --- | --- | --- | --- |
| MAE 50% | 0.0822185692 | 0.2613733214 | 0.6906344512 | 2.57477 | 102.4701 |
| MAE 75% | 0.0852784814 | 0.2771708992 | 0.6685027452 | 2.48486 | 122.3365 |
| MAE+barRel 75% | 0.0937725806 | 0.2977782335 | 0.6684070734 | 2.99956 | 117.6035 |
| Global EMA | 0.0786863027 | 0.2255844533 | 0.4981987377 | 2.09005 | 55.7758 |
| Multiscale EMA | 0.0731981236 | 0.2165266160 | 0.5033864737 | 1.85504 | 55.0411 |

75% 三类查询 positive priors 分别 0.01666260、0.08050537、0.33221436；fine-patch positive 数 546。MAE 75% 的 latent permutation ΔBCE 分别 +0.00567571、+0.09149081、+0.29917338。所有目标的 event features 非常量，但 rank 仍低，不能把方差非零当成语义充分。

Global EMA teacher projection rank **1.0805645**，multiscale teacher **1.1542912**；跨事件 RMS 分别 0.15566、0.12224。这是需要修复的维度相关结构，不以 finite/checkpoint complete 作为最终有效学习验收。

截至资源刷新时，累计实际 allocation **0.3452716442 device-hours**、CPU allocation **3.608888889 CPU-hours**，含数据等待/失败；该值是当时快照，后续以 `campaign_accounting.json` 更新为准。

## 阶段性结论

调整 patch 查询后，细粒度目标具备数百正例和高于先验的 AP，并依赖 event embedding；encoder 输入继续保持所有原生通道。该结果只支持扩展 MAE 训练，不能替代下游 ντ/四分类对照。EMA 的低维 teacher target 明确触发进一步修复。

## 下一步计划

正式 core wave 同时运行 Scratch 1k/10k/100k、train-only S3 branch probes、MAE 3-epoch 预训练和两项 EMA 修复 pilots。采用 dependency-aware batch scheduler，只启动依赖已完成的任务；同名 task claims 跨后端互斥。随后完成 MAE continuation vs partial Rel，以及相同下游训练、linear probes 和 compute controls。
