# 阶段 33：S2 正式投影模型与配对信息比较

正式 H100 ParticleNet native projection 与 ParT 1 mm projection 均完成四轮并通过 14 项 CPU 输出读回。ParticleNet native joint−projection 的 τ ΔAP=+0.00712568，pointwise 95% CI=[+0.00003031,+0.01448372]；固定 E/vertex common-support 区间跨 0。ParT 1 mm 的配对增量区间跨 0，且上界低于 +0.01。两项结论均以 seed42、四轮及既定局部压缩为条件；完整四格和跨种子仍待完成。

## 阶段目标

完成两个正式投影模型的完整输出核对、预定 joint−projection 比较、分域/τ 分支解释与实际成本匹配。沿用用户接受的样本和 detector-proxy，calibration/test 保留至 S6。

## 工作内容

`1573782.5` 对应冻结 `s2_h100_v2.json` task7，ParticleNet dim64、161,156 parameters；`1573782.2` 对应 task3，ParT dim48、151,228 parameters。两者使用 pixel+bar、seed42、batch4、4 epochs、每轮 20,016 attempted steps，按 validation NLL 选 best。投影移除 x–y pairing，其他冻结输入合同保持一致；模型经同平面 4 mm、canonical chunks≤32 nodes 的局部 learned compression 后处理完整全局块。

环境统一 H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。两者 Condor history 均为 JobStatus=4、ExitCode=0、NumJobStarts=1。03:04:30/03:09:01 UTC 分别通过 14 项读回，覆盖完整四轮、冻结配置、环境、optimizer、恢复字段、NLL 选模、validation keys、best checkpoint 与概率、40k 四轮概率及逐轮重算指标、exposures。

两个训练进程均已加载旧写入函数，训练期间通过各自原作业的 `condor_ssh_to_job` 在写入节点同步并保存四轮表。03:09:44 UTC 最终核对源表与同步副本逐值相同、history=4、完整读回=pass。两族 native joint 的早期 12 项记录也已补齐当前 14 项输出检查，全部通过；样本与 detector-proxy 无新增认证。

部分评价 commit=`41f3a6d`，正常退出；9 个有效结果（4 references+5 正式模型）、241 行分域指标、27 个历史三类诊断，6 项配对完成、39 项 pending。原生 ParT projection 的归档尝试未进入当前评价，替换 `1573791.0` 正在真实运行。更新后的 resolution 图和 paired-effects 首页已实际查看。

## 详细数据

独立 train=80k、validation=10k，每类分别 20k/2,500；两模型每轮 exposures 为 80,040/80,044/80,040/80,040，合计 320,164。80,064 applied updates、0 AMP skips；均按 NLL 选择 epoch4。

| Model | Epoch | Validation NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: | ---: |
| PN native projection | 1 | 1.29485182 | 0.30810273 | 937.94194 |
| PN native projection | 2 | 1.29935861 | 0.31047490 | 1,849.59053 |
| PN native projection | 3 | 1.27277118 | 0.34312047 | 2,761.23461 |
| PN native projection | 4 | 1.24334514 | 0.34797906 | 3,671.93675 |
| ParT 1mm projection | 1 | 1.29637371 | 0.32759710 | 1,616.09726 |
| ParT 1mm projection | 2 | 1.30541959 | 0.31079450 | 3,216.33942 |
| ParT 1mm projection | 3 | 1.25525027 | 0.34887113 | 4,860.50225 |
| ParT 1mm projection | 4 | 1.22571671 | 0.35421688 | 6,465.01821 |

| Model | Program wall [s] | Allocation [s / GPUh] | Extra allocation [s] | Peak GPU [MiB] |
| --- | ---: | --- | ---: | ---: |
| PN native projection | 3,672.16069 | 3,683 / 1.02305556 | 10.83931 | 603.29736 |
| ParT 1mm projection | 6,465.38416 | 6,483 / 1.80083333 | 17.61584 | 374.43408 |

### 四类指标与预定配对

| Model | Balanced accuracy | Macro F1 | NLL | Brier | Tau ROC AUC | Common-support tau AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PN native projection | 0.39620 | 0.39223621 | 1.24334514 | 0.67665858 | 0.62644947 | 0.31945295 |
| ParT 1mm projection | 0.41370 | 0.41055262 | 1.22571671 | 0.66620162 | 0.63214616 | 0.31995100 |

| Class AP | PN native projection | ParT 1mm projection |
| --- | ---: | ---: |
| CC nu_e | 0.59616339 | 0.60150397 |
| CC nu_mu | 0.40759333 | 0.44241032 |
| CC nu_tau | 0.34797906 | 0.35421688 |
| NC | 0.36615513 | 0.41156264 |

AP 使用每类 25% 的 validation mixture，各类 recall 与完整 confusion counts 保存于快照。以下为冻结配对的 B−A，400 次配对 accepted-family/chunk bootstrap、128 blocks（每类 32 个）。固定 common-support 保留 8,559 events、Neff=7,959.41365，权重不重拟合。

| Joint − projection | Raw tau ΔAP [pointwise 95% CI] | Fixed common-support ΔAP [pointwise 95% CI] |
| --- | --- | --- |
| PN native | +0.00712568 [+0.00003031,+0.01448372] | +0.00257536 [−0.00352278,+0.00843062] |
| ParT 1mm | +0.00081170 [−0.00433001,+0.00629580] | +0.00091006 [−0.00469003,+0.00678403] |

PN raw AP 区间下界接近 0，有当前事件区间的正增量支持；固定支持区间跨 0，且上界低于 +0.01。两者条件不同，尚不能据此确定 x–y pairing 的物理机制或跨训练稳定性。ParT 1 mm 两个 AP 区间跨 0，上界均低于预定 +0.01，可排除当前设置下至少 +0.01 的增量，不能推断一般等效。PN 的 1% FPR τ 效率差=+0.0052，CI=[−0.00275095,+0.01227093]；ParT 1 mm 为 −0.0016，CI=[−0.00828100,+0.00653526]。其余完整四格成员与控制尚未齐全，当前不判定配对收益的粒度依赖。

### 实际成本匹配

| Joint − projection | Shared allocation budget [s] | NLL-selected joint / projection epochs | Tau ΔAP | Joint unused time [s] |
| --- | ---: | --- | ---: | ---: |
| PN native | 3,682.77606 | 2 / 4 | −0.02026240 | 511.33456 |
| ParT 1mm | 6,482.63405 | 3 / 4 | −0.00010680 | 1,006.40473 |

全部实际 allocation 开销已计入。这是整轮 checkpoint 分辨率的点估计，时间匹配 CI 未测量；joint 剩余时间不足下一完整轮。固定四轮的原样本 AP 增量不能直接解释为同实际算力优势。

### 工作点、τ 分支与响应域

PN native projection 的 validation empirical 1% FPR 阈值=0.44520605，通过 70/2,500 个 τ、75/7,500 个背景，效率=0.0280；ParT 1 mm projection 阈值=0.43893462，通过 92 个 τ、75 个背景，效率=0.0368。0.1% FPR 分别为 11/16 个 τ、均为 7 个背景，尾部统计不足。最终阈值尚未确定。

| Tau branch | Events | PN native projection selected / joint | ParT 1mm projection selected / joint |
| --- | ---: | --- | --- |
| Electron | 445 | 6 / 7 | 11 / 12 |
| Muon | 395 | 5 / 4 | 6 / 5 |
| Hadronic | 1,660 | 59 / 72 | 75 / 71 |

通过数使用各模型 inclusive validation 1% FPR 阈值；分支差值 CI 未测量。ParT 1 mm joint 相对 projection 合计少 4 个 τ，其中 electron +1、muon −1、hadronic −4。PN 合计多 13 个 τ，其中 electron +1、muon −1、hadronic +13。分支变化尚不能确定机制。

| Domain | Events / tau | PN native projection AP | ParT 1mm projection AP |
| --- | --- | ---: | ---: |
| Pixel centered | 885 / 176 | 0.30319232 | 0.31816317 |
| Large scintillator | 4,494 / 1,053 | 0.37298959 | 0.38406857 |
| Pixel+bar input empty | 440 / 148 | 0.33636364 | 0.33636364 |
| Tracker-only response | 158 / 58 | 0.36708861 | 0.36708861 |

空输入和 tracker-only 域均为 prior AP、AUC=0.5；当前输入不含 tracker。分域 AP 包含样本组成影响，域内配对差值区间未测量。模态与训练域控制继续沿用冻结配置。

### 产物与复现

正式目录 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`：

- `first_projection_comparisons.json`：两模型完整证据、终态、成本、配对、分支与域。
- `experiments/{pn_native_projection,part_1mm_projection}_s42_e4_v2/final_checkpoint_checks.json`：14 项读回；对应 `recovery_epoch_tables/` 下 `final_verification.json` 为最终源表/副本核对。
- `readback_native_current.log`：两族 native joint 当前 14 项读回。
- `evaluation_final/summary.json` 与图表、`evaluation_first_projection_comparisons.log`：部分评价。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" \
  --experiments pn_native_projection_s42_e4_v2 part_1mm_projection_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
```

03:09:45 UTC 组织累计 46.45342 GPU-hours，missing history records=0，包含所有归档/失败 allocation。正式 5 complete、1 running、29 idle、0 held；V100 九项辅助全部结束。Kubeflow v6 仍 Pending、未分配 node。

## ParticleNet 1 mm projection 与双投影粒度比较补充

04:23:52 UTC，`1573782.7` 正常结束，JobStatus=4、ExitCode=0、NumJobStarts=1。冻结 task9 为 ParticleNet dim64、161,156 parameters，pixel+bar、seed42、batch4、四轮、每轮20,016 attempted steps，环境与本报告既有正式模型一致。训练 commit=`cc19ad8`。04:25:40 UTC 的14项 CPU读回全部通过，读回与评价 commit=`bba5f42`。独立train80k/validation10k、exposures320,164、80,064 applied、0 AMP skips，按NLL选中epoch4。

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.33807288 | 0.29232667 | 1,023.96858 |
| 2 | 1.26839655 | 0.33176457 | 1,997.10702 |
| 3 | 1.25682543 | 0.34310336 | 2,970.52898 |
| 4 | 1.23785700 | 0.35100848 | 3,944.75469 |

程序wall=3,946.01105 s，实际allocation=3,957 s=1.09916667 GPUh，额外开销=10.98895 s，peak=601.75781 MiB。四类balanced accuracy=0.40480、macro F1=0.40550990、Brier=0.67280623；CC nu_e/CC nu_mu/CC nu_tau/NC的AP分别为0.59907056/0.42474727/0.35100848/0.38679720。τ ROC AUC=0.63021832，固定common-support AP=0.32166112。每类25%的benchmark prior与本报告既有模型一致。

预定native projection−1mm projection的τ ΔAP=−0.00302942，pointwise 95% CI=[−0.01277175,+0.00658906]；固定E/vertex common-support ΔAP=−0.00220817，CI=[−0.01266791,+0.00716999]。区间沿用400次、128个accepted-family/chunk blocks，固定支持为8,559 events、Neff=7,959.41365。两个AP区间均跨0、上界均低于预定+0.01，当前seed42、四轮及既定局部压缩下没有原生双投影粒度的正收益区间支持，可排除至少+0.01的AP收益，不能判定一般等效。1% FPR τ效率差=−0.0040，CI=[−0.01277477,+0.00318884]。

共同实际allocation预算3,682.77606 s内，按NLL分别选native epoch4与1mm epoch3，native−1mm τ ΔAP=+0.00487570；1mm已计开销2,981.51793 s，剩余701.25812 s不足下一整轮。这是整轮分辨率点估计，时间匹配CI未测量。

1mm projection的validation empirical 1% FPR阈值=0.41372472，通过80/2,500个τ、75/7,500个背景，效率=0.0320；electron/muon/hadronic分别为8/4/68，对应native projection为6/5/59。native相对1mm通过数变化为−2/+1/−9，合计−10，分支差值CI未测量。0.1% FPR为9个τ、7个背景，尾部统计不足。pixel-centered AP=0.32782319、large域AP=0.37687616；空pixel+bar输入与tracker-only域仍为prior AP、AUC=0.5。分域配对CI未测量，不能从分域AP点值单独确定粒度机制。

部分评价正常退出，当前11个有效结果（4 references+7正式模型）、295行分域指标、33个历史三类诊断、9项配对完成/36项pending。ParT正式四格与交互效应已完成（阶段35），ParticleNet仍待1mm joint补齐完整四格。新的resolution图与paired-effects首页已实际查看。

正式根目录新增 `pn_projection_resolution_comparison.json`，固定保存manifest/history、14项读回、真实终态、成本、配对、分支和域；日志为`readback_pn_1mm_projection.log`、`evaluation_pn_projection_resolution.log`。复现命令与上文一致，readback选择`pn_1mm_projection_s42_e4_v2`。04:26:27 UTC组织累计53.11281 GPU-hours，missing history records=0；Kubeflow v6仍Pending，无GPU allocation。

## 阶段性结论

PN native pairing 的原样本 τ AP 有接近零下界的正事件区间支持，固定支持条件与同成本点估计尚未支持相同结论。ParT 1 mm 配对与PN双投影原生粒度未见正区间支持，当前各自可排除至少 +0.01 的 AP 增量。上述结果继续由完整四格、信息控制与三种子检验。

## 下一步计划

继续剩余正式配置，补齐ParticleNet四格、中间粒度、控制、训练域与三种子。全部 41 单seed+4 种子均值比较和最终科学报告齐备后才验收 S2；当前 goal 保持 active。
