# 阶段 20：首个完整科学 epoch 与耗时实测

ParT 1mm 双投影完成首个 80k-train pass 和 10k-validation：τ AP=0.30649、balanced accuracy=0.35040、NLL=1.32943。该 run 仍在按既定四轮合同继续训练，不能用这一中间结果代替最终 native/coarse 或架构结论。

## 阶段目标

确认完整规模的训练、validation、checkpoint 与成本记录可连续执行，并用完整 epoch 更新实际计算安排。科学验收仍需全部指定对照和三种子结果。

## 工作内容

跟进 `1573747.3`，配置 `part_1mm_projection_s42_e4_v2`，固定 Tesla V100-PCIE-32GB、LCG_107_cuda、FP16、ParT dim48。通过当前 Condor 状态与进度确认作业持续运行，随后读取首个完整 `history.json`。

验证产物在 results 下该 experiment 目录：`history.json`、`validation_predictions.parquet`、`epoch_predictions.parquet`、`last.pt`、`best.pt` 均已写出。`manifest.json` 状态仍为 `running`，正式 evaluator 因此继续把它列为待完成项，不混入已完成 references。

## 详细数据

首个 epoch 的 validation 每类 2,500 个事件；训练共享 frozen 80k train，每个事件完整覆盖后有少量确定性重复。实际 80,040 exposures 不是 80,040 个独立训练事件。

| Quantity | Measured value |
| --- | ---: |
| Attempted optimizer steps | 20,016 |
| Applied optimizer steps | 20,007 |
| AMP skipped steps | 9 (0.045%) |
| Train exposures | 80,040 |
| Mean train loss | 1.36528 |
| Train time | 2,883.115 s |
| Validation time | 171.833 s |
| Total epoch wall time including setup | 3,063.325 s (0.85092 h) |
| Train throughput | 27.7616 events/s |
| Validation events | 10,000 |
| Validation balanced accuracy | 0.35040 |
| Validation macro F1 | 0.31158 |
| Validation NLL | 1.32943 |
| Validation Brier score | 0.71827 |

| Class | ROC AUC | AP |
| --- | ---: | ---: |
| CC νe | 0.68329 | 0.48610 |
| CC νμ | 0.60869 | 0.32251 |
| CC ντ | 0.58591 | 0.30649 |
| NC | 0.57898 | 0.29338 |

计时中的 forward/backward 为 2,782.619 s，data wait=76.350 s，transfer=5.213 s。forward/backward 占 train time 约 96.5%；在该 epoch 中增加 loader CPU 不是主要提速方向。保留既定 4 CPUs/2 workers，未改变实验配置。

若后续 epoch 与首次相同，该配置四轮约需 3.404 GPU-hours；这是单一配置的线性估计，含首次启动，不能当作其他表示/家族或总 campaign 的实测成本。实际 allocation 另核算，仍按初筛 100、追加 100、总 200 GPU-hours 的组织上限跟进。

19:23:20 UTC 的 `campaign_accounting.json` 累计实测 4.99215 GPU-hours，missing history records=0，包含旧尝试、smoke/profile 与当前运行 allocation；尚不是最终成本。当前 12 GB RAM 申请保持，已观测 Condor `MemoryUsage` 为 3,663–3,907 MiB，但 native/tracker 完整 train+validation 峰值仍未全部测得，不依据这一低成本配置调整全部资源申请。

复现与核对入口：

```bash
condor_q -name bigbird20.cern.ch 1573747.3 -af JobStatus NumJobStarts MemoryUsage
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
```

`$S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11`。指标来自 `experiments/part_1mm_projection_s42_e4_v2/history.json` 的 epoch1；正式完整评价入口仍为 [S2 operations](../roadmap/07_s2_operations.md)。

## 阶段性结论

事实：第一项科学 run 已完成全规模训练/validation 和可恢复 checkpoint，资源处于既定单 V100 申请内。其首轮指标低于已经完成的 B0 nominal reference，但两者当前训练阶段/估计器不同，不以该点宣称深度模型最终较差或改变训练合同。

后续首轮记录现已补齐如下。它们都继续按原配置训练，不作为最终 NLL-selected 模型的完成结果。

| Configuration | Tau AP (epoch1) | NLL (epoch1) | Balanced accuracy | Epoch wall [s] | Four-epoch linear estimate [GPU h] | Applied / skipped steps |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ParT native joint | 0.34117 | 1.27921 | 0.37390 | 4,859.181 | 5.39909 | 20,005 / 11 |
| ParT native projection | 0.34389 | 1.28103 | 0.37730 | 3,555.224 | 3.95025 | 20,004 / 12 |
| ParT 1mm joint | 0.28580 | 1.33085 | 0.33600 | 4,293.019 | 4.77002 | 20,008 / 8 |
| ParT 1mm projection | 0.30649 | 1.32943 | 0.35040 | 3,063.325 | 3.40369 | 20,007 / 9 |
| ParticleNet native joint | 0.29390 | 1.32811 | 0.34140 | 2,937.912 | 3.26435 | 20,006 / 10 |

五项均为同一 80k/10k、seed42、V100/LCG_107/FP16、20,016 attempted steps。四轮估计只把各自第一轮 wall time 乘四，不是测得的最终成本；两族/表示实际速度不同，最终同时报告等步数和共同 GPU-time 比较。第一轮 native/projection 数值只描述学习进程，不用于停止对照或判定等效/晋级。

19:52:51 UTC 更新的 accounting 累计 7.45134 GPU-hours、missing history records=0。尚未完成任何一项四轮科学 run，因此正式完整评价继续保留 pending 状态。

未测量：四轮后的 NLL-selected 指标、native/coarse 配对区间、其他尚未启动配置的完整 epochs、三种子 spread 与最终 allocation。

## 下一步计划

保持原有 33 项训练，结合阶段 21 的两项跨层控制共 35 项科学配置继续执行。任务完成后运行严格的 41 个单 seed 配对、4 个三种子均值、factorial interaction、响应域/τ 分支与错误事件显示，再形成 S2 科学验收结论。
