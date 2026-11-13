# S4：完整 MAE、预算对照和源代码归档

## 阶段目标

读回完整 MAE；将进一步分类收益与额外训练成本分开比较；使正式 worker 使用稳定的本地源代码归档。

## 工作内容与详细数据

输出根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。

`s4_mae_m75_e3_s42` 已完成 3×6144 = 18,432 optimizer steps，未使用 pilot 权重。98k train-fit、固定 2k train-inner reconstruction-probe。10 项 checkpoint/internal checks 与 4 项 learning checks 全部通过。

| Target | Positive / queries | Prior | AP | AP / prior | Event-permutation ΔBCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16×16 native pixel patch | 2,257 / 128,000 | 0.01763281 | 0.45497851 | 25.80295 | +0.07611045 |
| 4-mm pixel cell | 10,144 / 128,000 | 0.07925000 | 0.78089867 | 9.85361 | +0.40820169 |
| scintillator bar | 42,643 / 128,000 | 0.33314844 | 0.89860048 | 2.69730 | +0.99856112 |

Global features 的 event RMS 为 0.65376973、centered effective rank 为 7.53238。该结果支持 decoder 使用事件观测，不能替代独立四分类 validation 的收益。

query-only CPU 对照已正式运行：bigbird21 cluster 2198169；MAE downstream/Rel repair mixed campaign 为 cluster 2198170，2 并发、6 CPU，等待 GPU。seed campaign cluster 1574094 原先 15-CPU 请求只有 1 个 idle GPU slot 可匹配，已在 JobStatus==1 的约束下调整为 2 并发、6 CPU、18 GB，科学配置保持原冻结值，变更记录于 `logs/seed_resource_adjustment.json`。

新增 `compute_profile.py`：在各实际 train sampler 中预先随机选择 12 个 batch，分别计数 MAE、local bar Rel 与 1k/10k/100k supervised training 的 Linear/attention matrix operations、backward 和 eager checkpoint recomputation。该量用于预先确定 matched-budget supervised continuation 的 steps；其定义不包括 normalization、indexing、elementwise operations 或 compiler fusion，必须同时报告真实 allocation 时长。不能将此 surrogate 写成完整硬件 FLOPs 或 MIG/整卡等价算力。实际 profile 数值目前未测量。

## 源代码和验证

新提交归档时，发现 EOS 上 `accounting.py` 的 loose Git blob 为空。用保持不变的 working source 计算 Git object ID，确认与 commit 中引用的 `33aa79ef6e81aaa51952213dbbdebf005d6498d4` 完全一致后恢复该对象；commit history 未重写。`git fsck --full` 通过，审计保存在 `audit/git_recovery/repair.json`。仓库 `core.fsync=committed`。

后续正式 worker 从已提交的单个 source tar 复制并解包至 worker local scratch，`PYTHONPATH` 使用该本地源码；EOS 保留可审阅的冻结版本与 archive，写 archive 复用既有 atomic full-write/retry。这样减少大量 Python 小文件对 EOS 的依赖。

在已通过的 S4 16 项检查之外，新增 matrix-operation unit test：无 input gradient 的 Linear forward+weight backward 为 96 operations，需要 input gradient 时为 144，实测通过。采样 profile 的实际运行仍须单独验收。

## 阶段性结论

完整 MAE 已学到多尺度重建，并通过内部数值和 checkpoint 验证。下游标签效率、ντ 子道和 matched-compute 收益尚待实测；现阶段不扩至 1M。

## 下一步计划

完成 query-only control、局部 Rel/EMA pilot、完整 fine-tuning 和 frozen probes；依据真实 profile 冻结额外 supervised controls，按 17/42/2026 确认关键结果并报告完整成本。
