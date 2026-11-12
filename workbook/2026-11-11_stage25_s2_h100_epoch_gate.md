# 阶段 25：H100 完整 epoch 的冻结标准、实测与执行选择

两族H100完整首轮均通过预先冻结的执行/耗时门槛：ParT为V100 wall的45.76%，ParticleNet为54.60%，两者均0 AMP skips。2026-11-12 00:04 UTC已按原规则选择H100/LCG108/BF16统一执行35项科学配置，全部已提交；尚未完成S2科学验收。原5项正在运行的V100任务继续作为辅助结果。

## 阶段目标

测量相同数据与 batch4 流水线的首轮程序耗时，决定后续完整 S2 矩阵采用的统一执行环境。最终交付仍为 35 项科学配置、41 项单 seed 配对、4 项种子均值及既有物理/资源诊断。

## 工作内容

[候选合同](../roadmap/configs/s2_h100_v2.json)从既有 spatial/controls/confirmation/cross-plane 四份配置逐项复制 `experiments`，全部模型参数、4 epochs、batch4、每轮 20,016 attempted steps、种子、选择规则和表示保持原值。35 个名称唯一，两个先行 task 为 0（ParT native joint）和 6（ParticleNet native joint）。

执行环境为 NVIDIA H100 NVL、LCG_108、Torch2.7.1、BF16，JobFlavour=workday。候选根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100`。通过 symlink 共享既有 `plan.json`、90k-event manifest、catalog、32 个完整 shards，以及 4 个已完成 CPU references；模型、checkpoint 和任务记录单独写入候选目录。具体来源记录为 `h100/source_links.json`。沿用用户接受的 samples、source-family split 和 detector-proxy。

## 详细数据与预先固定标准

提交时的资源依据为阶段24实测同事件batch1 compute约1.69–2.33倍；当时整轮耗时尚未测量，后续实际首轮见下文。先行任务使用完整 80k train/10k validation，按原计划运行四轮；在两者首轮均完成后评价资源标准：

| Requirement | ParT native joint | ParticleNet native joint |
| --- | --- | --- |
| V100 first-epoch program wall [s] | 4,859.18128 | 2,937.91183 |
| H100 maximum first-epoch program wall [s] | 3,887.34503 | 2,350.32946 |
| Maximum H100/V100 ratio | 0.80 | 0.80 |
| Minimum applied/attempted optimizer steps | 0.99 | 0.99 |

同时要求两项均保持完整输入和冻结配置，train loss、validation 概率与 checkpoint 参数有限，恢复/优化器状态可读。程序内 epoch wall 包含数据准备、训练和 validation；Condor allocation 与启动开销另行核算。validation AP、accuracy 和 NLL 的数值不用于硬件选择；四轮 checkpoint selection 仍按既有 validation NLL。

两者均通过后，完整 35 项矩阵采用统一 H100/LCG108/BF16，移除已被替代且仍 idle 的 V100 任务，已运行的 V100 任务继续完成并单列为辅助结果。任一未通过，则保留 V100 正式合同，候选不扩展。所有原 V100、资源测量、候选与后续任务均计入 S2 的 **200 GPU-hours** 总组织上限；完整环境中的 100 GPU-hours 初筛与 100 GPU-hours 控制/确认安排继续检查实际成本。

提交时完整首轮和最终成本尚未测量；后续首轮执行证据见下文，分类质量数值未用于环境选择。完整四轮的终态成本仍待测量。准备工作确认完整合同引用的 experiment names 均存在于35个科学配置或4个references中，原有41+4比较文件可直接用于独立候选目录。

commit `1654db1` 后，21:14:20 UTC 通过当前 schedd/EOS/WRITE 检查提交为 **`1573760.0/1`**，分别对应配置 tasks0/6。选用 `bigbird20.cern.ch`，记录为 `h100/submission_train_211421.json`。21:19 UTC 两项均 idle、NumJobStarts=0，尚无训练 manifest。

21:20 UTC 的 `condor_q -better-analyze` 显示当前 0 个愿意运行任务的 slot；具体检查 `b9pgpun014` 时，其两个 parent slots 均为 Start=false。任务请求为 4 CPUs、12 GiB memory、12 GiB disk、一个 H100 NVL；继续等待符合条件的 allocation。

```bash
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --tasks 0 6 \
  --lcg-view LCG_108_cuda --gpu-device 'NVIDIA H100 NVL' \
  --training-flavour workday
```

22:56:49 UTC，ParT候选 `1573760.0` 已在 `slot1_2@b9pgpun501.cern.ch` 开始实际执行。manifest确认NVIDIA H100 NVL、Torch2.7.1、Python3.12.11、BF16、80k train/10k validation、151,228 parameters及冻结配置；输入和split继续沿用既有accepted决定。22:58:35记录为750 applied steps、0 AMP skips、loss有限，随后推进到2,000步；这些是运行证据，不能代替完整首轮。另一族 `1573760.1` 仍idle。原始记录为 `h100/candidate_execution_start.json`，训练代码commit=`f5b8bab`。

新增可执行验收入口 `analysis.flavour.s2.hardware_gate`。完整首轮提交后，它读取该轮wall/counters/probabilities及实际checkpoint，检查冻结配置、GPU/软件/precision、训练记录数量、有限性和恢复字段，再使用上表冻结限值。只缓存已测得的首轮执行证据；缓存同时保存完整验收标准、模型配置和环境，标准变化时拒绝复用旧判断。两族均pass才输出总体pass，一族未提交完整首轮则保持pending；该工具不会把首轮验收视为S2完成，也不使用AP/accuracy/NLL数值选择硬件。

```bash
python -m unittest tests.test_flavour_s2_hardware_gate
python -m analysis.flavour.s2.hardware_gate --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json"
```

5项测试于0.002 s内通过，覆盖80%耗时/99% applied边界、快速但数值检查失败、双家族共同完成、质量指标不改变选择、修改标准不能复用旧pass。23:01:09 UTC首次真实入口正常退出，输出 `h100/execution_gate.json` 为pending，两族原因均为 `no_committed_first_epoch`；完整checkpoint验收分支仍待真实首轮。

23:34:17 UTC首个完整真实epoch验收完成：ParT wall=2,223.38095 s，为V100的0.45756287，低于冻结限值3,887.34503 s；20,016 attempted/applied、0 AMP skips，23项执行检查全部通过，包括完整10k validation概率、checkpoint参数、optimizer与恢复字段。`h100/execution_gate.json` 为ParT pass/ParticleNet pending，总体仍pending；没有读取AP/accuracy/NLL数值作硬件选择。此前未测量的真实checkpoint分支现已成功执行。

另一族原Job仍idle且无可运行匹配。23:27已将其JobPrio从0提高到20，证据 `h100/candidate_priority.json`。23:34:42 UTC在ParT首轮checkpoint读回通过后，将原 `1573760.0` 暂时hold以释放H100，优先让 `1573760.1` 取得完整首轮资源。原JobId和四轮配置保持，所有额外allocation继续计费。已启动有界观察：ParticleNet实际running后release原ParT任务；若10分钟仍没有调度收益则也release恢复原排队。原始状态/原因/后续release写入 `h100/candidate_checkpoint_scheduling.json`。23:35:19 UTC观察到ParticleNet已在刚释放的 `slot1_2@b9pgpun501.cern.ch` 实际running，随即release原ParT任务，查询确认为idle；调度暂停约37 s后达成目的，没有重提或双writer。这是首轮执行证据的调度安排，总体pass之前不扩展35项H100矩阵。

ParticleNet实际manifest已确认H100 NVL、Python3.12.11、Torch2.7.1、BF16、161,156参数和80k/10k完整子集。首轮已推进、0 AMP skips；`h100/second_candidate_execution_start.json` 保存配置与执行快照，不能代替完整首轮。23:38:38 UTC组织总成本24.62590 GPU-hours，其中H100候选合计0.70501（含ParT暂时hold前的2303 s allocation），missing=0；两项候选均无held状态，ParT原任务idle、ParticleNet running。

## 2026-11-12 完整门槛通过与矩阵切换

00:01:53 UTC，ParticleNet首轮完成后，`s2.hardware_gate` 正常exit0，总体pass。两族均按同一冻结配置执行80k train/10k validation，每类20k/2.5k；实际完整checkpoint、optimizer、RNG恢复字段及10k validation概率已读取检查，两个record各23项全部通过。没有提取AP/accuracy/NLL数值作选择。

| Family | H100 first-epoch program wall [s] | H100 / V100 | Applied / attempted | AMP skips |
| --- | ---: | ---: | ---: | ---: |
| ParT native joint | 2,223.38095 | 0.45756287 | 20,016 / 20,016 | 0 |
| ParticleNet native joint | 1,604.22022 | 0.54604097 | 20,016 / 20,016 | 0 |

按预先固定on-pass规则，00:03:16 UTC提交余下33项到 `bigbird20.cern.ch`，新Cluster=`1573782`；提交器保存当前schedd负载、EOS/WRITE与GPU资源证据，原两个候选JobId复用。配置和35个experiment names逐项核对为原spatial/controls/confirmation/cross-plane合同的完整集合，没有新增重复科学任务。

| Jobs | H100 task indices | Role |
| --- | --- | --- |
| `1573760.0/1` | 0 / 6 | ParT / ParticleNet native joint候选继续四轮 |
| `1573782.0–9` | 1–5 / 7–11 | 余下10项空间配置 |
| `1573782.10–22` | 12–24 | 13项控制 |
| `1573782.23–30` | 25–32 | 8项额外种子 |
| `1573782.31/32` | 33 / 34 | 两族plane-only控制 |

原ParT候选恢复队列优先级20；新任务中的6项主factorial优先级10，中间粒度4项为5，controls/seeds为0。排序依据冻结实验角色，没有使用分类分数。`h100/activation_plan.json` 在总体pass前保存计划，`h100/activation.json` 保存完整门槛、映射和实际动作。

确认33项提交成功及完整35个唯一配置后，00:04:13 UTC才执行服务器端 `JobStatus == 1` 条件，移除26个被替代的V100 idle任务，5个running任务继续。随后真实队列与accounting均确认原35项为4 complete/5 running/26 removed，26个被移除任务的总allocation=0；所有旧失败、profile、中断和已运行成本保留。

正式科学结果根为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100`，35项都在H100/LCG108/BF16环境完成后，统一运行原41项单seed与4项种子均值合同。父目录V100部分结果作为辅助，已完成4个模型及3项配对不表示H100矩阵已完成。00:06队列为H100 1 running/34 idle，尚无四轮H100科学模型complete；Kubeflow v6仍Pending未分配node。

00:05:52 UTC组织累计27.34940 GPU-hours，其中H100目录1.15893，missing history records均为0。35项包含15 ParT/20 ParticleNet；仅以两族native首轮线性外推四轮、忽略配置/域差异与新中断，程序时间规划量约72.70569 GPU-hours。这是未验证的预算估计，不是实测完整矩阵成本；实际allocation继续按200 GPU-hours组织上限核算。

配置验证：LCG108下PyYAML成功读取更新后的 `roadmap/configs/s2.yaml`，选定执行合同指向H100；35个experiment dictionaries与原四份冻结campaign逐项完全相同。`git diff --check`通过。阶段26的10项accounting/Kubeflow测试已通过；本次切换未修改训练、模型或评价实现。

## 阶段性结论

完整矩阵与选择标准已在H100分类训练结果产生前固定。两族完整首轮的执行和wall门槛现均通过，因此选择统一H100环境，全部35项已提交。该结论支持当前数据流的资源选择，正式分类结论仍待完整四轮模型、41+4比较与原生错误显示。

## 下一步计划

继续H100完整35项和5项V100辅助任务的真实队列/进程观察、checkpoint恢复与预算核算。H100两族native joint完成后生成原生错误显示；完整矩阵之后运行严格41+4评价，保持calibration/test冻结至S6。
