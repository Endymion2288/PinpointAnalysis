# 阶段 12：S2 低维参照的实际分类结果

固定 80k train / 10k validation 上，完整 B0 的 τ AP 为 **0.34471**，1% background acceptance 下 τ efficiency 为 **2.60%（65/2,500）**。truth energy/vertex 诊断的 τ AP 为 0.44106；共同 energy/vertex 粗分箱支持后仍为 0.36474，说明这些粗匹配没有完全移除样本先验。空间网络的真实比较尚未完成。

## 阶段目标

测量全局响应、空间 summary 和样本能量/位置先验的分类能力，为 native/coarse 对照提供相同物理事件上的基准。

## 工作内容

HTCondor 1573735 的四个 CPU reference tasks 均正常完成；1573739 统一评价正常 exit 0。所有模型复用既有 S2 80,000 train 和 10,000 validation，后者各类 2,500，τ prevalence=25%。按 validation NLL 从 10–200 iterations 的 checkpoints 选择，模型设定见阶段 10。以下均为 validation 比较，未打开 calibration/test，不对应物理 flux/exposure。

评价输出位于 `results/flavour/s2_2026-11-11/evaluation/`，包括 `summary.json`、`metrics.csv`、`tau_PR.png` 和 `error_event_groups.parquet`。新增 display 工具根据这些错误组生成多页原生响应 PDF：每 confusion pair 取固定最高置信例子，再加入固定随机参照，附事件 keys 和仅用于解释的 truth annotations。

为下一轮 GPU 训练添加 256-event 窗口内的 occupancy bucketing 配置入口，使用同一原生通道数给所有表示分 batch；不删除或截断事件、steps 保持相同。每 250 steps 输出进度，分别记录 data wait、transfer、forward/backward 和显存。新的 sampler 测试确认覆盖不变且 padding 成本不增加，恢复一致性测试继续通过。

## 详细数据

| Reference | Balanced accuracy | Macro F1 | NLL | τ AP | τ ROC AUC | τ eff @ 1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 all | 0.40290 | 0.40458 | 1.24504 | 0.34471 | 0.62567 | 0.02600 |
| B0 signal | 0.35600 | 0.35287 | 1.29971 | 0.33345 | 0.61090 | 0.02960 |
| B0 shape | 0.35670 | 0.35566 | 1.31118 | 0.32451 | 0.60147 | 0.02160 |
| Truth energy/vertex diagnostic | 0.34990 | 0.32658 | 1.32109 | 0.44106 | 0.69145 | 0.06800 |

训练总耗时（含读取与反复 validation）：B0 all 64.398 s，signal 46.907 s，shape 60.858 s，truth diagnostic 54.535 s；均 CPU。四项的 1% 工作点各通过 75/7,500 个背景。0.1% 工作点仅通过 7 个背景，未达到预先设置的 25-background 稳定尾部标准。

400 次 class-stratified paired chunk bootstrap，Δ 为 B−A；这些是事件统计区间，尚非训练种子 spread：

| A → B | Δ τ AP | 95% interval |
| --- | ---: | --- |
| Signal → full B0 | +0.01126 | [−0.00261, +0.02618] |
| Shape → full B0 | +0.02020 | [+0.01282, +0.02958] |
| Full B0 → truth diagnostic | +0.09635 | [+0.07026, +0.12153] |

完整 B0 相对 signal 的区间包含零，也不能排除 +0.01 收益，不作等效结论。相对 shape 的结果支持增加 signal summary 有贡献。truth 模型仅量化样本先验的可区分性，不算硬件性能增益。

共同支持采用阶段 09 固定的 E/x/y/z 分箱，每 cell 各类至少 5 个事件，权重=min(class count)/class count≤1。保留 103/234 populated cells、8,559 events、2,236 τ，四类权重和均为 1,586，Neff=7,959.41，weighted τ prevalence=25%。

| Reference | Common-support weighted τ AP | Weighted τ ROC AUC |
| --- | ---: | ---: |
| B0 all | 0.30845 | 0.58945 |
| B0 signal | 0.30518 | 0.58005 |
| B0 shape | 0.29491 | 0.56699 |
| Truth energy/vertex diagnostic | 0.36474 | 0.62386 |

以上只平衡了给定粗 bins 的联合密度；bin 内连续谱形仍可能不同。没有把残余性能解释成已去除所有 priors 的 topology-only 结果。

完整 B0 条件诊断：pixel-centered 885 events/176 τ，τ AP=0.33926；large-scintillator 4,494 events/1,053 τ，τ AP=0.37093。两区域 τ prevalence 分别为 19.887% / 23.431%，AP 不直接横向比较。282 个全空事件中 90 个 τ，ROC AUC=0.5、AP=90/282=0.31915，符合没有事件内可见区分信息的同分数输出。

同一 inclusive 1% FPR 阈值下，τ→e 为 7/445=1.573%，τ→μ 为 8/395=2.025%，τ→had 为 50/1,660=3.012%；inclusive 为 65/2,500，不用 hadronic 分母替代全部 τ。

兼容 GPU smoke 1573737 两任务 exit 0：ParT 在 V100S、ParticleNet 在 V100 各完成 29 train/3 validation、1 epoch。dim32 的 train throughput 为 3.114 / 1.229 events/s，峰值 allocated memory 为 132.93 / 285.77 MiB。该小样本不平衡、首次 CUDA 开销和 occupancy 都不代表科学训练，未用其分类指标推断性能。150k/160k 参数规模的完整压力测量 1573738 正在运行。

新增 sampler/recovery 测试 3 项在 LCG_107 通过，5.012 s。复现入口：

```bash
python -m analysis.flavour.s2.submit reference --root "$S2_RESULTS"
python -m analysis.flavour.s2.submit evaluate --root "$S2_RESULTS" \
  --comparisons "$PWD/roadmap/configs/s2_reference_pairs.json"
python -m analysis.flavour.s2.submit display --root "$S2_RESULTS" --experiment b0_all_s42
```

## 阶段性结论

事实：低维硬件 summaries 已有高于随机 benchmark 的四类/τ 排序能力，严格背景接受率下的 τ efficiency 仍低。全局信号解释了较大部分 B0 τ AP；可观测形状的额外贡献在该指标下尚未明确。样本 E/vertex priors 可明显区分 τ，粗匹配后仍有残余。

假设：native 局部结构可能提供 B0 summaries 未表达的信息；必须通过共享事件、相同 steps 与实际 GPU-time 的 native/coarse 和 local-shuffle 对照判断。

## 下一步计划

完成 resource profile 并冻结科学 campaign 的 batch/epochs；启动两族 native/1mm×joint/projection、100/200 μm 对照，随后做 modal、OR、shift、graph/pair、shuffle/domain 与三种子确认。检查错误组原生显示，保留待查事件而不据少数例子关闭研究分支。
