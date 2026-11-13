# S4 首轮 corpus pilot 与稀疏查询修正

## 阶段目标

根据 train-inner 重建证据判断预训练是否有效学习，并修复明显欠约束的目标后重训。此阶段不评价完整四分类泛化。

## 工作内容

四项 512-step pilots 已完成，每项以 98k train 中的有限 exposures 拟合，在原 train partition 的固定 256-event 子集做 reconstruction probe。最大 occupancy stress 和真实拟合已通过，所有 informative 内部健康采样通过。该批原生 pixel query 保留为采样失败诊断，不作为 native 结构重建已解决的证据。

## 详细数据

| Strategy | Fine positives / 16,384 queries | 4-mm pixel AP | Bar AP | Global between-event RMS | Global centered effective rank |
| --- | --- | --- | --- | --- | --- |
| MAE 50% | 3 | 0.2866746092 | 0.7195245083 | 0.7430161 | 3.233272 |
| MAE 75% | 2 | 0.3102200895 | 0.7174446448 | 0.8060573 | 3.521838 |
| EMA 75% | 2 | 0.2556106706 | 0.5338247400 | 0.6696520 | 2.001111 |
| MAE+barRel 75% | 2 | 0.3142675919 | 0.7230914428 | 0.7319017 | 4.166963 |

4-mm positive priors 分别为 0.06958008（50%）和 0.08312988（75%），bar priors 约 0.351。原生 pixel 正例数仅 2–3，不评价其 AP 精度。MAE 75% 的 event-latent permutation ΔBCE 为 4-mm **+0.12939013**、bar **+0.35173363**；其余策略同样在 coarse/bar 上为正。说明 global representation 对重建具有事件条件信息，但低秩结构和下游四类可读信息仍待检查。

64 个固定 train-inner events、同样 masks/queries 的 target granularity 对照：

| Pixel target | Positive / 4,096 queries |
| --- | --- |
| One native channel | 0 |
| 8×8 native channels | 28 |
| 16×16 native channels | 83 |

因此第二轮将 fine reconstruction 改为 16×16 occupancy/count/native-centroid；encoder 继续读取所有逐通道观测，未对分类输入 rebin。查询仍先于 hidden targets 生成。另增加 global-only EMA 与 local/regional/global teacher 的单独对照，检查原 EMA rank 低是否源于 teacher 目标。

`s4_pilot_v2.json` 冻结五项 512-step experiments：MAE 50/75%、global EMA、multiscale EMA、MAE+barRel；原目标权重不继续使用。新增 patch-count/native-centroid 和 multiscale teacher 的梯度测试，S4 共 **10 项**通过。

## 阶段性结论

首轮预训练的 coarse/bar 目标有有效学习与事件依赖，global 不是常量；不能据此宣称 ντ 或低标签分类改善。逐 native pixel 查询的正例采样不足，需要修改目标再训练。不同 mask 的 AP 包含不同查询先验，未以原始 AP 直接裁定 mask 优劣。

## 下一步计划

完成第二轮拟合/重建与 frozen probes，结合等先验、相同查询统计与实际吞吐选定主 mask/teacher。随后运行完整预训练和三标签预算对照；保留继续 MAE/继续监督的资源控制。已提交 Scratch label curves 和 train-only branch probes，当前 HTCondor 仍排队。
