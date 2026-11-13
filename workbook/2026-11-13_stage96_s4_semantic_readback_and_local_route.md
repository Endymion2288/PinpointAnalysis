# S4：Rel 语义读回和保留局部信息的修复

## 阶段目标

检验 partial bar Rel 是否学习逐 bar 粒子贡献，而不只是总体先验；补齐 query-geometry shortcut 对照，冻结 MAE continuation 和下游矩阵。

## 工作内容与详细数据

输出根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。

对原 `s4p2_mae_barrel_m75_s42` checkpoint 追加 FP32 CPU 语义读回，保持原 BF16 reconstruction/history 和 checkpoint selection 不变。使用相同 512 个 train-inner probe events；37,567 个有效 visible-bar targets，15,921 个 mixed channels。

| Statistic | Initialization | Trained checkpoint |
| --- | ---: | ---: |
| Soft-target cross-entropy | 1.39281535 | 0.73061657 |
| Fraction RMSE | 0.39438951 | 0.30260852 |
| Constant probe-prior cross-entropy | 0.72388333 | 0.72388333 |
| Target entropy | 0.19901407 | 0.19901407 |

目标均值为 EM 0.42160356、muon 0.00810717、hadron 0.57024807、other 0.00004089。模型虽然显著降低初始 loss，仍未超过常量先验。因此原 Rel pilot 的 occupancy 学习不能作为 Rel 语义成功证据；更新后 `ssl_readback.json` 的 measured-learning 状态为 false，旧版 global-only Rel 不扩量。该常量使用 probe 自身均值，是描述性的 oracle constant 下界，没有冒充独立拟合 predictor。

修复 `rel_source=global_and_bar_local`：Rel decoder 同时读取 global latents 与全部 visible bar local blocks。按 observation 的 event/detector 索引打包，无 truth routing 或截断；内部 source normalization 与 QK normalization 保留。MAE reconstruction 继续仅从 global latents 重建。下游 classifier 的参数、输入和架构不变。此设计参考 FASERCal kept-voxel semantic objectives，并针对 PINPOINT 的 native readout 进行适配。`roadmap/configs/s4_rel_repair.json` 固定新的 1024-step real-corpus pilot；必须先通过语义读回，才允许 full MAE+barRel。

`roadmap/configs/s4_mae_downstream.json` 固定同一 MAE3 checkpoint 的两条 2×6144-step continuation 分支，分别为 MAE-only 和 MAE+barRel，两条均重置 optimizer/schedule。三个标签预算的 fine-tuning 和 frozen probes 使用同 backbone、同 head 初始化协议。下游依赖还必须等待实际 `ssl_readback.json`；一个策略失败只阻止它的依赖，独立任务继续运行。

新增 `query_baseline.py`：从 98k fit 中按固定随机索引取 2048 个事件，保留与主实验相同的 geometry/visible-neighborhood queries；训练同一 decoder，但所有事件共享常量 source tokens，完全不读取 event encoder。固定 2k train-inner probe 用于比较 AP、loss 和 permutation；该 baseline 独立由正式 CPU HTCondor 运行。`roadmap/configs/s4_query_baseline.json` 为冻结配置，结果目前未测量。

## 调度与验证

额外 whole-H100 的 server dry-run 被 CERN `namespace-gpu-quota/check-total-gpu-units` 拒绝：现有 MIG 也计入总 GPU 配额。未分配资源，记录于 `logs/whole_gpu_admission_rejection.json`。保留当前 MIG，额外 GPU 工作转入 HTCondor。seed wave 已提交 bigbird20 cluster 1574094；schedd 依据即时 idle/held/running 与兼容性选择。当前队列仍在等待匹配。

17:49:43 UTC accounting 实测为 0.76501119 device-hours，含运行中的 allocation；CPU batch 为 3.60888889 core-hours。该值是当时快照，最终成本以结束后的 accounting 为准。完整 MAE 第一轮 2k reconstruction-probe AP 为 fine 0.30736260、4-mm 0.68925729、bar 0.88391206，global effective rank 7.22234；不是下游分类结果。

新增检查覆盖 Rel loss 单独到达 global/local encoder、detach global 后新增 bar-local route 仍有梯度、预训练 encoder 导入与相同随机 heads、query-only baseline 的 event-input independence。执行命令：`python -m unittest tests.test_flavour_s4`：16 项全部通过。

## 阶段性结论

事实：原 global-only Rel 主要学到总体贡献先验；原占据重建验收不足以证明语义学习，现已分别验收。局部 route 的语义增益仍是待实测假设。MAE 的完整重建已有学习证据，分类收益尚待下游验证。

## 下一步计划

运行局部 Rel pilot 和 query-only CPU baseline；完成 MAE3 readback；按 dependency 顺序训练 fine-tuning、continuation 和 probes；保留失败策略的实际证据，完成 paired seeds 与 matched-compute 分析。
