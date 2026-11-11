# 阶段 25：冻结 H100 完整 epoch 的资源选择标准

H100 的 batch1 资源测量支持进一步验证整轮耗时。已建立独立候选目录，并在 H100 分类训练开始前冻结原有全部 35 项科学配置及纯执行/耗时选择标准；先提交两个 native-joint 模型。当前正式 V100 任务继续运行和排队。

## 阶段目标

测量相同数据与 batch4 流水线的首轮程序耗时，决定后续完整 S2 矩阵采用的统一执行环境。最终交付仍为 35 项科学配置、41 项单 seed 配对、4 项种子均值及既有物理/资源诊断。

## 工作内容

[候选合同](../roadmap/configs/s2_h100_v2.json)从既有 spatial/controls/confirmation/cross-plane 四份配置逐项复制 `experiments`，全部模型参数、4 epochs、batch4、每轮 20,016 attempted steps、种子、选择规则和表示保持原值。35 个名称唯一，两个先行 task 为 0（ParT native joint）和 6（ParticleNet native joint）。

执行环境为 NVIDIA H100 NVL、LCG_108、Torch2.7.1、BF16，JobFlavour=workday。候选根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100`。通过 symlink 共享既有 `plan.json`、90k-event manifest、catalog、32 个完整 shards，以及 4 个已完成 CPU references；模型、checkpoint 和任务记录单独写入候选目录。具体来源记录为 `h100/source_links.json`。沿用用户接受的 samples、source-family split 和 detector-proxy。

## 详细数据与预先固定标准

资源比较为阶段 24 实测同事件 batch1 compute 的约 1.69–2.33 倍；整轮耗时尚未测量。先行任务使用完整 80k train/10k validation，按原计划运行四轮；在两者首轮均完成后评价资源标准：

| Requirement | ParT native joint | ParticleNet native joint |
| --- | --- | --- |
| V100 first-epoch program wall [s] | 4,859.18128 | 2,937.91183 |
| H100 maximum first-epoch program wall [s] | 3,887.34503 | 2,350.32946 |
| Maximum H100/V100 ratio | 0.80 | 0.80 |
| Minimum applied/attempted optimizer steps | 0.99 | 0.99 |

同时要求两项均保持完整输入和冻结配置，train loss、validation 概率与 checkpoint 参数有限，恢复/优化器状态可读。程序内 epoch wall 包含数据准备、训练和 validation；Condor allocation 与启动开销另行核算。validation AP、accuracy 和 NLL 的数值不用于硬件选择；四轮 checkpoint selection 仍按既有 validation NLL。

两者均通过后，完整 35 项矩阵采用统一 H100/LCG108/BF16，移除已被替代且仍 idle 的 V100 任务，已运行的 V100 任务继续完成并单列为辅助结果。任一未通过，则保留 V100 正式合同，候选不扩展。所有原 V100、资源测量、候选与后续任务均计入 S2 的 **200 GPU-hours** 总组织上限；完整环境中的 100 GPU-hours 初筛与 100 GPU-hours 控制/确认安排继续检查实际成本。

候选任务的完整首轮、NLL/AP和最终成本尚未测量；已开始执行的初期counters见下文。准备工作确认完整合同引用的 experiment names 均存在于35个科学配置或4个references中，原有41+4比较文件可直接用于独立候选目录。

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

## 阶段性结论

完整的候选矩阵与选择标准已在H100分类训练结果产生前固定。ParT候选已通过完整首轮执行/耗时门槛；ParticleNet完整首轮尚待资源，总体切换门槛仍未通过。正式比较仍须在单一选定执行环境完成全部指定任务。

## 下一步计划

核实两个先行任务实际 GPU、precision、数据数量与程序执行，取得两份完整首轮 checkpoint 后按上述标准作出资源决定。随后完成选定环境的整个科学矩阵及严格评价，保持 calibration/test 冻结至 S6。
