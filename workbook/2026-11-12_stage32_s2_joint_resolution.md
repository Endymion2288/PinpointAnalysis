# 阶段 32：S2 正式 joint 输入的 native−1 mm 比较

正式 H100 ParT 1 mm joint 已完成四轮并通过全部 14 项 CPU 读回，τ AP=0.35502857。native−1 mm joint 的 ΔAP=+0.00355534，pointwise 95% CI=[−0.00186106,+0.00868839]；固定 E/vertex 支持上的区间也跨 0 且上界低于 +0.01。当前 seed42、四轮和既定局部压缩下，尚无 native 粒度正增量的区间支持，可排除该条件下至少 +0.01 的 AP 收益。多种子和信息控制仍待完成。

## 阶段目标

完成首个正式 native/1 mm joint 粒度配对，核对新模型的完整输出、物理分域、τ 分支和实际 allocation。沿用用户接受的样本与 detector-proxy，calibration/test 保留至 S6。

## 工作内容

`1573782.1` 执行冻结 `s2_h100_v2.json` 的 task2：ParT dim48、151,228 parameters、seed42、batch4、4 epochs、20,016 attempted steps/epoch。输入为 pixel+bar；pixel 在 1 mm 网格保留 count 聚合，再经同平面局部 learned compression 与完整全局块处理。bar 与 native 配置相同，比较以该表示和训练预算为条件。

训练代码 commit=`9546871`，环境为 H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。真实 history 为 JobStatus=4、ExitCode=0、NumJobStarts=1。14 项 CPU 读回覆盖四轮、配置与环境、80,064 次更新、NLL 选模、参数/optimizer 有限性、恢复状态、10k validation keys、概率与 best checkpoint 一致性、40k epoch 概率及其逐轮 NLL/AP、exposures，全部通过，读回 commit=`4f94a47`。

该进程使用旧写入函数，训练期间已通过原作业的 `condor_ssh_to_job` 在写入节点同步并保存最后一轮表。02:52:41 UTC 副本含 40k 行，当时 history 尚为 epoch3；随后真实训练提交 epoch4 并结束。02:55:33 UTC 在当前节点比较源表与同步副本，逐值相同，最终 history=4、完整读回=pass。副本未用于提前宣布训练完成。

正式部分评价在核算与读回完成后运行，commit=`4f94a47`，正常退出。当前为 7 个有效结果（4 references+3 完整模型）、187 行分域指标、21 个历史三类诊断，4 项配对完成、41 项 pending，完整标记为 false。原 native projection 尝试已归档，等待冻结配置替换 `1573791.0`；本次有效主表不读取其历史预测。resolution 与 paired-effects 首页已实际查看。

## 详细数据

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.28264235 | 0.33554724 | 1,900.66383 |
| 2 | 1.28141170 | 0.32459433 | 3,681.18338 |
| 3 | 1.25879018 | 0.35411008 | 5,461.77554 |
| 4 | 1.24293540 | 0.35502857 | 7,263.09506 |

NLL 选择 epoch4。独立 train 80k、validation 10k，每类分别 20k/2,500；train exposures 共 320,164，applied steps=80,064、AMP skips=0。程序 wall=7,263.54623 s；实际 allocation=7,278 s=2.02166667 GPU-hours，额外 allocation=14.45377 s。peak allocated GPU memory=1,545.35400 MiB。

### 四类结果与粒度配对

1 mm joint 的 balanced accuracy=0.40330、macro F1=0.39880034、NLL=1.24293540、Brier=0.67581568。各类 AP 使用每类 25% 的 validation mixture，recall 使用四类 argmax。

| Class | AP | ROC AUC | Recall |
| --- | ---: | ---: | ---: |
| CC nu_e | 0.59928946 | 0.78159768 | 0.4932 |
| CC nu_mu | 0.41981120 | 0.66200560 | 0.3920 |
| CC nu_tau | 0.35502857 | 0.63183757 | 0.5136 |
| NC | 0.39628130 | 0.63774131 | 0.2144 |

native joint 的 NLL=1.22107879、balanced accuracy=0.41840；两项点估计更好，四类差值区间未测量。以下预定配对以 τ AP 为主指标。

| Prespecified native − 1mm joint | Difference | Pointwise 95% CI |
| --- | ---: | --- |
| Original validation mixture tau AP | +0.00355534 | [−0.00186106,+0.00868839] |
| Fixed E/vertex common-support tau AP | +0.00297513 | [−0.00192093,+0.00846325] |
| Tau efficiency at validation 1% FPR | +0.00200000 | [−0.00519545,+0.00869765] |

区间使用 400 次配对 accepted-family/chunk bootstrap、128 个物理 blocks（每类 32 个）。固定支持保留 8,559 events、Neff=7,959.41365，权重固定；1 mm 的 common-support AP=0.32086106，native 为 0.32383619。两个 AP 区间均跨 0，且上界低于预定 +0.01，说明当前事件区间可排除该设置下至少 +0.01 的收益。种子17/2026 尚未完成，不能把这个条件结果推广为原生信息的一般上限，也不能推断等效。

同实际 allocation 预算 6,992.77439 s 内，NLL 选择 native epoch4 与 1 mm epoch3，τ AP 差值=+0.00447383。1 mm 对应已计开销时间=5,476.22932 s，剩余 1,516.54507 s 不足以容纳下一完整轮。这是 epoch-resolution 点估计，CI 未测量；原 native 的调度恢复开销已计入。

### 工作点、τ 分支与响应域

1 mm joint 的 validation empirical 1% FPR 阈值=0.44188008，选中 88/2,500 个 τ、75/7,500 个背景，τ 效率=0.0352。0.1% FPR 只有 10 个 τ、7 个背景，尾部统计不足。最终阈值尚未确定。

| Tau branch | Signal events | 1mm selected at inclusive 1% FPR | Native selected | 1mm branch AP |
| --- | ---: | ---: | ---: | ---: |
| Electron | 445 | 12 | 7 | 0.08133046 |
| Muon | 395 | 5 | 4 | 0.06177801 |
| Hadronic | 1,660 | 71 | 82 | 0.29193590 |

分支 AP 使用各自分支 prior 与同一 7,500 个 non-τ 背景。native 相对 1 mm 的通过数变化为 electron −5、muon −1、hadronic +11，合计 +5；分支差值区间未测量，尚不能确定通道机制。

| 1mm validation domain | Events / tau events | Tau AP | Tau ROC AUC |
| --- | --- | ---: | ---: |
| Full | 10,000 / 2,500 | 0.35502857 | 0.63183757 |
| Pixel centered | 885 / 176 | 0.32895379 | 0.67745063 |
| Large scintillator | 4,494 / 1,053 | 0.38681431 | 0.68324376 |
| Pixel+bar input empty | 440 / 148 | 0.33636364 | 0.50000000 |
| Tracker-only response | 158 / 58 | 0.36708861 | 0.50000000 |

全空输入与 tracker-only 域的 AP 等于各域 prior，AUC=0.5。pixel-centered 的单独 1% FPR 只有 5 个 τ、7 个背景通过，统计不足；large 域为 43 个 τ、34 个背景。各域 AP 包含样本组成差异，分域粒度差值区间未测量。模态与训练域控制继续按冻结矩阵执行。

### 产物与复现

正式根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。

- `first_joint_resolution_comparison.json`：本次 manifest/history、真实终态与 allocation、14 项读回、粒度配对、分支与分域、同步副本最终核对。
- `experiments/part_1mm_joint_s42_e4_v2/final_checkpoint_checks.json`、`readback_part_1mm_joint.log`：完整输出证据。
- `recovery_epoch_tables/part_1mm_joint_s42_e4_v2/final_verification.json`：40k 行源表与原节点同步副本的逐值一致性。
- `evaluation_final/summary.json`、`metrics.csv`、`resolution_comparison.png/pdf`、`paired_comparisons.pdf`：当前有效部分评价；`evaluation_first_joint_resolution.log` 为执行日志。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments part_1mm_joint_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
```

02:54:25 UTC 组织累计 45.83821 GPU-hours，missing history records=0，包含归档 projection 的 5,618 s。正式配置为 3 complete、2 running、30 idle，0 held；V100 另有 1 项辅助训练运行。替换 `1573791.0` 正在排队，其他两项旧写入进程的原节点同步辅助进程仍跟进最后一轮。Kubeflow v6 仍待 GPU 分配。

## 阶段性结论

首个正式 joint 粒度比较及完整输出读回完成，native 的四类指标点估计较好，τ AP 的最小 +0.01 收益在当前条件事件区间内被排除；跨种子稳定性、配对与粒度交互、信息机制仍待控制。原节点同步与跨节点核对已在该模型成功完成，后续完整输出保留相同核对要求。

## 下一步计划

继续其余 32 个正式配置，包括已提交的 projection 替换任务，跟进原生 ParticleNet projection 和 ParT 1 mm projection 的最终输出。补齐两族四格、100/200 μm、控制与三种子，再对全部 41+4 比较执行严格评价。S2 goal 保持 active。
