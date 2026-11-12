# 阶段 30：S2 正式 H100 原生架构比较

正式 ParT native joint 已完成四轮，τ AP=0.35858。相对 B0 的原样本 ΔAP=+0.01387，pointwise 95% CI=[+0.00152,+0.02523]；validation 1% FPR 的效率差也有正区间支持。ParT−ParticleNet 的 ΔAP=+0.00348，CI=[−0.00369,+0.01036]，架构优劣尚未确定。两族正式 native 错误事件图均已生成并实际查看。S2 当前仅完成 35 个正式模型中的 2 个。

## 阶段目标

核对第二个正式模型的训练、checkpoint、概率与实际 allocation，完成冻结的 native 架构/B0 配对和 ParT 原生错误事件审阅。沿用用户接受的 samples、source-family split 与 detector-proxy；calibration/test 保留至 S6。

## 工作内容

`1573760.0` 执行 `roadmap/configs/s2_h100_v2.json` 的 ParT native joint：seed42、dim48、151,228 parameters、batch4、4 epochs，每轮 20,016 attempted optimizer steps。训练池 80k、validation 10k，各类分别 20k/2,500。输入为 native pixel+bar；同平面 4 mm 分块后按 canonical chunks≤32 nodes 做局部 learned compression，再进入完整全局块。这一模型的结论以既定压缩表示为条件。

环境为 H100 NVL、LCG108/Python3.12.11/Torch2.7.1/BF16。Condor history 为 JobStatus=4、ExitCode=0、NumJobStarts=2，包含前期执行门槛调度后的恢复；原任务从完整 epoch1 恢复，恢复证据已在前阶段记录。最终 CPU checkpoint/概率读回 12 项全部通过：四轮与配置、optimizer 计数、NLL 选模、权重/optimizer 有限性、恢复状态、10k 唯一 keys、概率归一化及 checkpoint 一致性、40k epoch 概率、exposures。

本次正式评价 commit=`62acc91`，命令显式使用 `--allow-incomplete`。输出包含 6 个结果（4 references+2 正式模型）、160 行分域指标、18 个历史三类诊断；3 项配对完成、42 项 pending，`requested_comparisons_complete=false`。静态合同覆盖检查的 8 项通过只说明 35 个配置均被比较覆盖。

## 详细数据

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] | Train exposures |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.29925662 | 0.33367609 | 2,223.38095 | 80,040 |
| 2 | 1.26560853 | 0.33754929 | 3,809.78355 | 80,044 |
| 3 | 1.25214773 | 0.35584978 | 5,362.36683 | 80,040 |
| 4 | 1.22107879 | 0.35858391 | 6,903.16975 | 80,040 |

NLL 选择 epoch4。80,064 次更新全部应用，AMP skips=0；320,164 exposures 对应独立训练事件 80k。程序 wall=6,903.39536 s，真实 allocation=6,993 s=1.94250 GPU-hours，额外 allocation=89.60464 s；peak allocated GPU memory=1,589.18994 MiB。

四类 balanced accuracy=0.41840、macro F1=0.41844267、NLL=1.22107879、Brier=0.66252223。以下指标使用每类 25% 的 validation mixture。

| Class | AP | ROC AUC | Recall |
| --- | ---: | ---: | ---: |
| CC nu_e | 0.60478117 | 0.78360797 | 0.4944 |
| CC nu_mu | 0.46065084 | 0.68530133 | 0.3328 |
| CC nu_tau | 0.35858391 | 0.63245197 | 0.5488 |
| NC | 0.42513003 | 0.66562965 | 0.2976 |

混淆矩阵行是真类、列是预测类，顺序为 e、mu、tau、NC：

```text
1236  149   925  190
 236  832   986  446
 391  362  1372  375
 282  414  1060  744
```

### 预定配对与时间匹配

400 次 accepted-family/chunk 配对 bootstrap 使用 128 个物理 blocks，每类 32 个。以下是 pointwise 事件区间，训练种子均为 42。

| Contrast | Original mixture ΔAP [95% CI] | Fixed common-support ΔAP [95% CI] |
| --- | --- | --- |
| ParT − B0 | +0.01387323 [+0.00151857,+0.02522555] | +0.01538603 [+0.00635680,+0.02482101] |
| ParT − ParticleNet | +0.00347918 [−0.00368918,+0.01035692] | +0.00180789 [−0.00453733,+0.00875252] |

固定 E/vertex 支持保留 8,559 events，Neff=7,959.41365；权重固定，ParT 的 common-support AP=0.32383619。ParT 相对 B0 的正增量得到当前事件区间支持；区间下界仍低于预定 +0.01，尚未确认至少 +0.01 的收益。B0 是 CPU reference，比较同时改变表示和估计器，未测 GPU 时间匹配。

ParT−ParticleNet 的原样本区间覆盖 0 和 +0.01。固定支持区间上界低于 +0.01，但这是该固定权重条件下的结果；架构跨种子稳定性待测。不能由区间跨 0 推断等效。

实际 allocation 时间匹配预算=6,309.67797 s，按预算内最小 NLL 选择 ParticleNet epoch4、ParT epoch3。ParT 对应已计入开销的时间=5,451.97147 s，剩余 857.70651 s；τ AP 差值=+0.00074505。这是 epoch-resolution 点估计，CI 未测量。

### 工作点、τ 分支与响应域

ParT 的 validation empirical 1% FPR 阈值=0.43040636，选中 93/2,500 个 τ、75/7,500 个背景，τ 效率=0.0372。相对 B0 的效率差=+0.0112，CI=[+0.00116809,+0.02016826]；相对 ParticleNet 为 +0.0040，CI=[−0.00273842,+0.01186143]。0.1% FPR 仅选中 17 个 τ、7 个背景，尾部统计不足。最终阈值尚未确定。

| Tau branch | Signal events | Selected at inclusive 1% FPR | Efficiency | Branch AP |
| --- | ---: | ---: | ---: | ---: |
| Electron | 445 | 7 | 0.01573034 | 0.08132916 |
| Muon | 395 | 4 | 0.01012658 | 0.06071660 |
| Hadronic | 1,660 | 82 | 0.04939759 | 0.29800798 |

分支 AP 使用该分支自身 prior 与同一 7,500 个 non-τ 背景。ParticleNet 对应通过数为 7/4/72；两者 inclusive 工作点的净差 10 个全在 hadronic 分支。分支差值 CI 未测量，该计数尚不能确认增益机制。

| Validation domain | Events / tau events | Tau AP | Tau ROC AUC |
| --- | --- | ---: | ---: |
| Full | 10,000 / 2,500 | 0.35858391 | 0.63245197 |
| Pixel centered | 885 / 176 | 0.33532613 | 0.68636203 |
| Large scintillator | 4,494 / 1,053 | 0.38651796 | 0.68199934 |
| Pixel+bar input empty | 440 / 148 | 0.33636364 | 0.50000000 |
| Tracker-only response | 158 / 58 | 0.36708861 | 0.50000000 |

输入全空域和 tracker-only 域都表现为先验型输出；tracker 信息增量留给冻结的 +tracker 消融。pixel-centered 单独 1% FPR 只有 4 个 τ、7 个背景通过，统计不足；large 域为 36 个 τ、34 个背景。分域 AP 含各域样本组成差异。

### 原生错误事件与实际查看

`1573789.0`（2 CPU/6 GB）JobStatus=4、ExitCode=0、allocation=80 s。20 页 PDF 的首页与第 3/7/10/15 页均已实际查看；resolution 与 paired-comparisons 图也已查看。

| Page | Key (run/chunk/row) | Truth → prediction | Pixel / bar / tracker channels | p(tau) |
| --- | --- | --- | --- | ---: |
| 3 | 10081/728/81 | CC nu_e → CC nu_tau | 54 / 1,424 / 20 | 0.51954484 |
| 7 | 10082/3130/159 | CC nu_mu → CC nu_tau | 76 / 2,619 / 6 | 0.49539548 |
| 10 | 10083/103/758 | CC nu_tau → CC nu_mu | 10 / 166 / 1 | 0.04064186 |
| 15 | 10084/1144/775 | NC → CC nu_tau | 59 / 2,017 / 6 | 0.52607060 |

页面按固定 confusion 组的最高置信度确定。三个假 τ 的 pixel 稀疏，bar 跨多个平面展开，均在既有 pixel-centered 域之外。第 10 页为已注释的 muon 分支真 τ，十个 pixel 在 x−z、y−z 中呈近直线，tracker 仅显示一站一个 hit。图中的 tracker 是响应背景，当前模型输入仅 pixel+bar。E、vertex、τ 分支为图示诊断。这些例子支持后续检查 coverage、模态及轨迹关联；其总体贡献由冻结对照测量。

### 产物与复现

正式根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。

- `second_scientific_model.json`：本次完整 manifest/history、真实终态/allocation、12 项读回、配对、分支、响应域及事件图复核。
- `experiments/part_native_joint_s42_e4_v2/final_checkpoint_checks.json`：最终读回检查。
- `evaluation_final/summary.json`、`metrics.csv`、`resolution_comparison.png/pdf`、`paired_comparisons.pdf`：部分正式评价；`evaluation_second_model.log` 为执行日志。
- `evaluation_final/event_displays/part_native_joint_s42_e4_v2/`：20 页 PDF、`examples.csv`、实际查看 PNG 与 `review.json`。
- `submission_display_020615.json`、`logs/display.1573789.*`：显示提交与终态日志。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.submit display --root "$S2_RESULTS/h100" \
  --evaluation-tag evaluation_final --experiment part_native_joint_s42_e4_v2
```

02:08 UTC 正式矩阵为 2 complete、4 running、29 idle，0 held；释放槽位已由原队列 ParticleNet native projection 接续。02:03:39 UTC 组织累计 allocation=42.00897 GPU-hours，missing history records=0，上限 200 GPU-hours。Kubeflow v6 仍 Pending、无 node。V100 新完成结果在阶段 28 辅助表中另记。

## 阶段性结论

两个正式 native 模型完成四轮、checkpoint/概率读回及错误事件图审阅。ParT 相对 B0 的原样本 AP、固定支持 AP 和 1% FPR 效率增量有当前 seed 的事件区间支持；最小 +0.01 AP 收益与跨种子稳定性尚未确认。架构优劣尚未确定，粒度、联合坐标、模态和跨层贡献仍待对照。

## 下一步计划

继续其余 33 个正式 H100 配置，补齐 41 项单 seed 与 4 项三种子均值比较、两族 factorial、粒度曲线及冻结控制。全部训练、条件域/分支、实际时间匹配、资源和图示齐全后，以不含 `--allow-incomplete` 的命令严格评价，再完成 S2 科学验收。
