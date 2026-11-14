# GPU、Kubeflow 与执行安排

**优先使用能够尽快启动、满足实际软件与显存要求的设备。**HTCondor、Kubeflow、不同GPU型号、平台提供的共享设备和MIG均可使用。科学比较通过相同事件、曝光记录和同allocation内配对实现。旧S2的固定整卡H100合同属于历史阶段，不是v2的提交要求。

## 1. 软件兼容性与精度

当前环境固定为CVMFS `LCG_108_cuda/x86_64-el9-gcc13-opt`，Python3.12、Torch2.7.1/CUDA12.5。2026-11-14实测`torch.cuda.get_arch_list()`为：

```text
sm_75, sm_80, sm_86, sm_89, sm_90, compute_90
```

| Device / resource | Current execution policy | Evidence / limitation |
| --- | --- | --- |
| T4 / SM75 | FP32、eager；按实测时长选择deadline | R1已测CNN和ParT profile；CUDA零张量/归约启动通过 |
| A100 / SM80 | 已验收的BF16 encoder + FP32 head；编译按各族健康选择 | 多项R0/R1真实拟合、profile和训练 |
| H100 / SM90, full GPU or MIG | 同上；MIG按独立allocation调度 | R1已有3g.47gb实际训练 |
| V100 / SM70 | 当前构建不能执行；不提交本环境训练 | 两项CNN扩展出现`no kernel image is available` |
| Other devices | 先做CUDA和真实小样本forward/backward probe | 不能仅根据更新的Capability数值认定当前二进制可用 |

V100本身支持FP32；本次失败的原因是当前Torch二进制没有SM70内核，改用FP32不能补上内核。若未来有现成兼容环境，可独立验证V100，不需要为本轮先重新编译Torch。B2另有compiled BF16局部反传故障，当前采用已通过真实拟合的eager路径；这是算子/执行配方限制。

`analysis.flavour.v2.submit`默认最低Capability为7.5，不限定DeviceName。R1训练在创建实验目录、复制全样本之前执行一个实际CUDA算子；SM75运行时切换FP32/eager，并在manifest分别保存submitted recipe、实际精度、GPU、Capability和Torch内核架构。更高设备仍须由实际probe验证。

## 2. 调度与比较

1. 大样本转换、统计和重模拟使用HTCondor；训练同时考虑Condor与Kubeflow。先检查兼容schedd的队列、WRITE和EOS策略，再选择负载最低者，保留submission记录。
2. 在平台实际提供的设备中，选择预计完成时间较短的组合：等待时间、启动/编译、实测吞吐、验证与剩余曝光都计入。不能只按卡峰值算力或显存大小排序。
3. 一个族的T3/T4、两个LR screen及同计算控制尽量在同一已获得的allocation中顺序执行；不同族和seed可同时使用不同卡。已有running训练继续完成，不为统一设备重新排队。
4. Pending/Idle任务可迁移到空闲后端；先确认原任务没有运行进程或存活writer，再结束原排队副本并启动新尝试。已经获得资源的任务需按可恢复进度判断迁移收益。
5. 使用平台提供的MIG或共享GPU资源；`request_gpus=1`指一个平台公布的逻辑设备，具体是否为分区按runtime记录。共享比例、显存和隔离由平台能力决定，不能自行假设任意小数GPU请求可被调度。

**曝光匹配**回答相同非τ事件训练次数下的任务差异。**同计算匹配**在同一allocation上从fresh初始化执行T3/T4预算控制，记录实际wall、训练、验证、updates及±5%预算误差。跨GPU的device-hours仅作资源记账；不同型号、MIG和精度不解释为相同FLOPs。平台并发负载会影响wall比较，结果中保留实际吞吐。

## 3. Kubeflow

沿用[归档操作指南](../archive/2026-11-14_v1/08_kubeflow.md)的CERN flavor/resource、挂载与准入细节。context=`ml-prod`、namespace=`xcheng`；已有登录可用。token过期时提醒用户登录，继续推进不依赖登录的分析。

H100整卡和MIG使用相同`nvidia.com/gpu.product`，但resource分别为`nvidia.com/gpu`与`nvidia.com/mig-3g.47gb`。CERN按flavor注入CPU/RAM，GPU Job不能额外手填CPU/RAM。先对真实Pod模板做server dry-run；Job对象dry-run不足以证明Pod准入。namespace当前配额与scheduler events决定可用性，归档的历史配额不是实时空闲量。

CLI保留显式`--flavor`，因为平台资源键不同。选择flavor时检查当前可观察到的等待原因及已有吞吐，必要时迁移尚未开始的任务；不把CLI默认`h100-47gb`解释为必须等待该设备。集群node列表不可访问时，使用本namespace的Pod、配额和事件，不推断全平台空闲卡。

2026-11-14 22:17 UTC，第二个MIG Pod的真实server dry-run被`namespace-gpu-quota/check-total-gpu-units`拒绝：`Total GPU units in namespace xcheng cannot exceed the quota of 1 units.`当前有效限制是所有GPU资源合计1单位，MIG也计入。标准ResourceQuota中的`nvidia.com/gpu` used=0不能据此推断还能增加MIG；证据为`R1/audit/k13_quota_rejection_stage245.json`。该次Job未创建、allocation为0。现有MIG任务完成后才迁移合适的排队任务，其余训练继续用Condor。

以下是已实现的提交形式，name必须对应新的唯一尝试；同一family/seed仅允许一个writer。示例用于展示接口，不应为已经运行的配置另建副本：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R1_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1
python -m analysis.flavour.v2.submit --root "$R1_ROOT" --stage R1 \
  --name UNIQUE_ATTEMPT --backend kubeflow --flavor h100-47gb --gpu \
  analysis.flavour.v2.r1.extend --root "$R1_ROOT" --family B4 --seed 42 --compile-local
```

Condor提交使用相同入口，替换backend并按profile指定CPU/RAM/deadline，通常不传`--gpu-model`。`--min-capability 7.5`允许当前环境的T4及更新兼容设备。B3等较慢模型应按可获设备的剩余曝光估时，选择足够的JobFlavour，避免在默认8小时之前中断。

## 4. 已执行的恢复与资源记录

R1原CNN任务1582878及无tracker控制1582879在V100启动失败，均已到终态；失败目录和日志保留，恢复记录为`R1/audit/v100_kernel_recovery_a1.json`。重试1582881、1582882在Idle且零次启动时移除，尝试Kubeflow迁移；上述实时quota阻止迁移后，重新提交1582883、1582884。均使用SM75下限且不限定型号；Idle移除与quota拒绝均为零allocation。迁移证据为`R1/audit/cnn_queue_migration_stage245.json`。22:33 UTC读回1582883已在T4执行，1582884仍在排队。不得将启动失败或未分配作业计作模型学习或性能证据。

主运行目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。提交快照在`runtime/*.submission.json`，各族配置在`configs/`，训练记录在`experiments/*/manifest.json`。Kubeflow日志同步到EOS；已清理Pod使用已保存的真实终态快照记账。失败attempt也用于关联runtime GPU名称，避免其成本丢失。

```bash
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT"
```

`resources.json`按GPU型号/分区分别列出allocation和CPU core-hours，包含启动、编译、失败和恢复。记录仍运行的任务时明确为截至该时点成本，不标为最终成本。MIG device-hour不折成整卡hour。18 R1/80总device-hours是规划额度，按实测预测调整候选数量与后续矩阵，不因预算数字而宣称未完成实验已有结论。

R1全部screen完成；完整四pass、真实seed重复、同计算配对、压力对照和终态验收仍在执行，完成后以实际报告更新。R2–R7的研究工作按[执行路线](04_execution_plan.md)推进。
