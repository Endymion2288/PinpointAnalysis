# 阶段 31：S2 输出完整性与 native 配对比较

正式 H100 ParT native projection 原尝试四轮正常结束，best checkpoint 与 10k validation predictions 一致，τ AP=0.35733812。其 40k 行 epoch 概率表在 EOS 两个副本上写回不完整，完整读回尚未通过。已实现同步后原子发布输出，四次 EOS 替换及直接 XRootD 读回验证通过。原节点恢复无法匹配资源，已归档原尝试并提交冻结配置的完整替换重跑。

## 阶段目标

将完成模型的 CPU checkpoint/预测读回固化为可复现入口，定位新增 projection 输出问题并修复后续写入流程，保留已测的 native 配对结果及其适用范围。

## 工作内容

新增 `analysis.flavour.s2.readback`，使用冻结 campaign 和已接受 catalog，逐模型核对 13 项：完整轮次、冻结配置、执行环境、optimizer 计数、NLL 选模、模型与 optimizer 有限性、恢复字段、validation keys、概率归一化、best checkpoint/预测一致性、全部 epoch 概率和 exposures。检查在 CPU 上执行，不加载原始探测器响应。输出 `final_checkpoint_checks.json`，失败时返回非零。

`train.save_checkpoint` 和新增的 `save_parquet` 使用临时文件，在写入、flush、fsync 完成后原子替换目标文件。epoch 表和 best validation predictions 的三处写入调用已接入该流程，恢复时的预测重建也使用相同入口。模型、训练配置、数据、步数和选模规则沿用冻结设置。已经运行的进程继续使用启动时加载的写入函数；后续新进程使用此次修复。

## 详细数据

### 正式 projection 训练与已验证部分

`1573782.0` 的 Condor history 为 JobStatus=4、ExitCode=0、NumJobStarts=1，实际 allocation=5,618 s=1.56055556 GPU-hours。程序 wall=5,602.46798 s，额外 allocation=15.53202 s。H100 NVL/BF16、151,228 parameters、80k train/10k validation、seed42、batch4；80,064 applied steps、0 skips、320,164 exposures；peak allocated GPU memory=443.35596 MiB。

| Epoch | NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.28522510 | 0.33062624 | 1,499.56133 |
| 2 | 1.26784029 | 0.32789119 | 2,870.73063 |
| 3 | 1.24075242 | 0.34899397 | 4,227.47262 |
| 4 | 1.22337225 | 0.35733812 | 5,541.51083 |

按 NLL 选择 epoch4，balanced accuracy=0.41500、macro F1=0.41465904、Brier=0.66463187。validation empirical 1% FPR 阈值=0.46394488，选中 82/2,500 个 τ、75/7,500 个背景，τ 效率=0.0328；0.1% FPR 为 14 个 τ、7 个背景，尾部统计不足。

02:31:29 UTC 的实际 CPU 读回中 12 项通过，唯一失败项为 `all_epoch_probabilities_present`。已通过项确认 `last.pt`/`best.pt` 可读且参数与 optimizer 有限，四轮 history、配置、环境与更新计数一致，NLL 选择正确，10k keys 与已接受 validation 相同，概率有限且归一化，best 模型权重及 best probabilities 与保存结果一致。完整状态保持 `fail`，待 epoch 表恢复。

### native 联合坐标相对双投影

使用已验证 best predictions 的正式部分评价正常退出，commit=`851ffc0`，7 个结果、4 项配对完成、41 项 pending。新增 ParT native joint−projection 的 τ AP 差值=+0.00124580，400 次 accepted-family/chunk 配对 bootstrap 的 pointwise 95% CI=[−0.00501470,+0.00830354]；固定 E/vertex common-support 差值=+0.00135347，CI=[−0.00498468,+0.00777324]。1% FPR 效率差=+0.0044，CI=[−0.00365371,+0.01089213]。

本 seed、四轮和既定局部 learned compression 下，两个 AP 区间上界都低于预定 +0.01，尚无 native 配对正增量的区间支持；该结果不确定其他表示或训练规模的收益。多种子、shuffle/pair/plane-only 对照仍待完成。resolution 与 paired-effects 图已实际查看。该次评价启动时 allocation 快照尚未收录新终态，时间匹配标记为未测；epoch 表恢复且核算更新后重新计算。

### EOS 失败证据与恢复

两次 CPU 读取均报 `Parquet magic bytes not found in footer`。直接 XRootD 读取也仅得到 221,184 bytes。EOS namespace 登记该文件为 1,023,199 bytes、状态 `fuse::needsflush`；`eos file check` 显示两个副本的物理大小都为 221,184 bytes，均报 `SIZE`。这些事实确认 epoch 表的磁盘副本不完整；EOS 写回未完成的具体原因仍待确认。

原 worker 为 `b9pgpun009.cern.ch`。已在检查 schedd 负载、EOS 与 WRITE 兼容性后选择 `bigbird20.cern.ch`，提交原节点恢复作业 `1573790.0`。恢复程序尝试从原节点 FUSE 视图读取 40k 行表，另写恢复文件并 fsync，保留原文件。默认 CPU 池不含该节点；定位后又确认原节点 START 要求 `RequestGPUs>0`，故将该原作业调整为 4 CPU/3 GB/1 GPU、JobPrio=30 的短恢复任务。02:31 UTC 仍 idle，尚无恢复结果。恢复实际 GPU allocation 会纳入组织总成本。

### 新写入流程验证

02:26:13 UTC，在 EOS 上连续四轮替换概率表和 checkpoint，再通过 XRootD 直接复制到临时本地目录读回，三项均通过：40k 行表完全相同、checkpoint 轮次为 4、checkpoint 概率逐值相同。该检查使用已有 10k best probabilities 构造 I/O 负载，验证写入与读回；未运行新训练。

新增 CPU 读回入口也在 `smoke/readback_module/` 通过全部 13 项，使用已有完整 ParticleNet 模型的只读文件链接；原模型验收记录保持原样。projection 的实际损坏文件则使该入口明确返回非零，并保留其余 12 项独立证据。`git diff --check` 通过。

### 原尝试归档与冻结配置替换

02:37 UTC，原节点 GPU 已被另一任务占用；`1573790.0` 仍为 idle、NumJobStarts=0、allocation=0，无法读取该节点的本地 FUSE 视图。已撤下这项恢复作业，history 确认为 JobStatus=3、0 starts、0 s。后续采用完整替换重跑，以补齐不可读的四轮概率表。

原 `experiments/part_native_projection_s42_e4_v2/` 整个目录保留于 `recovery_projection/original_training_attempt/`，包含 best/last checkpoint、有效 best predictions、history、损坏 epoch 表及失败读回记录。正式同名目录标记 `awaiting_complete_reexecution`，原有已完成模型数相应回到 2；先前部分评价的 projection 指标对应归档原尝试。当前正式有效结果待替换任务完成后更新。

替换 `1573791.0` 使用 `s2_h100_v2.json` 的同一个 task1、H100 NVL/LCG108/BF16、seed42、batch4、80k/10k、4 epochs、20,016 steps/epoch及 NLL 选模，代码包含同步与原子发布修复 `d9a5d77`。提交器重新检查 schedd 负载和兼容性，选择 `bigbird20.cern.ch`；作业在启动前 held 以完成原目录归档，确认 NumJobStarts=0 后设置 JobPrio=30 并释放。没有训练副本同时写入正式目录。

替换决定依据是已确认的文件不完整，在读取替换质量指标之前固定。原 5,618 s 与替换尝试的实际 allocation 均按同一实验计入成本。最终科学主表使用替换的完整结果，原尝试保留为历史证据；同耗时比较继续计入失败尝试开销。决定与精确 job 映射为 `recovery_projection/reexecution_decision.json`，提交记录为 `submission_train_023730.json`。

```bash
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --tasks 1 \
  --lcg-view LCG_108_cuda --gpu-device 'NVIDIA H100 NVL' --training-flavour workday
```

### 产物与命令

根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。

- `third_scientific_model_pending_epoch_recovery.json`：本次训练、配对与部分读回证据。
- `recovery_projection/initial_eos_evidence.json`、`readback_before_recovery.json/log`：两副本状态、文件大小与实际失败。
- `recovery_projection/cpu_pool_adjustment.json`、`submission_recover_022450.json`、`logs/recover.1573790.*`：原节点恢复提交与资源调整。
- `smoke/atomic_output_io/summary.json`：四次替换的 EOS/XRootD 集成验证；`smoke/readback_module/experiments/pn_native_joint_s42_e4_v2/final_checkpoint_checks.json`：完整模型上的 13 项验证。
- `evaluation_third_model.log`、`evaluation_final/summary.json`：当前部分正式评价。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" \
  --experiments part_native_projection_s42_e4_v2
```

## 阶段性结论

新写入流程的 EOS/XRootD 集成验证通过，可复现读回入口能区分完整模型和当前 epoch 表缺损。projection 原尝试的分类结果有 best checkpoint 与预测一致性证据，完整输出由冻结配置替换任务重新生成。原尝试的 native 配对在当前设置下未显示至少 +0.01 AP 的收益；正式结论将在完整替换结果与其余冻结矩阵上更新。

## 下一步计划

跟进替换 `1573791.0`，完成后执行 CPU 完整读回并更新正式配对；全部旧尝试成本保留。继续其余正式训练，按新增终态更新实际成本和配对。全部 35 个正式配置与 41+4 比较齐全后执行严格评价，S2 goal 保持 active。
