# GPU、Kubeflow 与执行安排

**优先使用能够尽快启动、满足实际软件与显存要求的设备。**HTCondor、Kubeflow、不同GPU型号、平台提供的共享设备和MIG均可使用。科学比较通过相同事件、曝光记录和同allocation内配对实现。旧S2的固定整卡H100合同属于历史阶段，不是v2的提交要求。

2026-11-15 R1最终状态：122条allocation全部终态，`active_or_unresolved=[]`；batch成本45.708889异构device-hours和307.181111 allocated CPU core-hours，452项验收通过。失败、探针、迁移前已运行部分均保留成本；历史SSL .635321 h单列，交互操作另有.325533 h已记录下界。[stage274](../../workbook/2026-11-15_stage274_r1_final_results.md)给出完整账本口径；下文迁移记录说明当时的调度选择。

R2最终状态（2026-11-15 13:30:44 UTC）：39条allocation全部终态，`active_or_unresolved=[]`，7.475833异构device-hours、116.65 allocated CPU core-hours，198项验收通过；另一次提交拒绝未产生allocation。交互operation下界.177657 h另列，完整交互allocation开销未测量。R0上界+R1/R2 batch累计54.269577 h，原80 h规划名义余量25.730423 h；后续按实测小批次重排并保留R7。[stage284](../../workbook/2026-11-15_stage284_r2_final_results.md)和[r2_results.json](r2_results.json)为最终口径；下文R1/R2运行实例保留为历史操作证据。

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
| V100 / SM70 | LCG108不能执行；已有LCG107兼容构建，R1尚未复验 | 两项LCG108扩展失败；历史S2的LCG107/Torch2.5.1已执行V100 |
| Other devices | 先做CUDA和真实小样本forward/backward probe | 不能仅根据更新的Capability数值认定当前二进制可用 |

V100本身支持FP32；本次失败的原因是LCG108的Torch二进制没有SM70内核，改用FP32不能补上内核。历史[stage11](../../workbook/2026-11-11_stage11_s2_gpu_compatibility.md)已确认LCG107/Torch2.5.1/Python3.11.9支持SM70，S2也已实际执行。若后续profile显示它能缩短完成时间，复用该构建并重新验证当前模型的真实forward/backward、数据读回和数值配方，记录软件版本；当前R1训练仍使用已验证的LCG108。B2另有compiled BF16局部反传故障，当前采用已通过真实拟合的eager路径。

`analysis.flavour.v2.submit`默认最低Capability为7.5，不限定DeviceName。R1训练在创建实验目录、复制全样本之前执行一个实际CUDA算子；SM75运行时切换FP32/eager，并在manifest分别保存submitted recipe、实际精度、GPU、Capability和Torch内核架构。更高设备仍须由实际probe验证。

2026-11-15实际广告补充：Condor H100 1g.12gb/2g.24gb MIG省略Capability；提交条件对这些明确的SM90型号补充识别，保留实际CUDA probe。完整B2已测aggregate RSS最高13,441.5 MiB，GPU默认请求15GiB，经站点变换为5 CPU/15,360 MB，可进入17GB的1g节点；旧16GiB会变成6 CPU/18GB而排除它们。三个尚未开始的作业修正后已在1g MIG启动，见[stage255](../../workbook/2026-11-15_stage255_r1_stress_audit_and_mig_matching.md)。

## 2. 调度与比较

1. 大样本转换、统计和重模拟使用HTCondor；训练同时考虑Condor与Kubeflow。先检查兼容schedd的队列、WRITE和EOS策略，再选择负载最低者，保留submission记录。
2. 在平台实际提供的设备中，选择预计完成时间较短的组合：等待时间、启动/编译、实测吞吐、验证与剩余曝光都计入。不能只按卡峰值算力或显存大小排序。
3. 一个族的T3/T4、两个LR screen及同计算控制尽量在同一已获得的allocation中顺序执行；不同族和seed可同时使用不同卡。已有running训练继续完成，不为统一设备重新排队。
4. Pending/Idle任务可迁移到空闲后端；先确认原任务没有运行进程或存活writer，再结束原排队副本并启动新尝试。已经获得资源的任务需按可恢复进度判断迁移收益。
5. 使用平台提供的MIG或共享GPU资源；`request_gpus=1`指一个平台公布的逻辑设备，具体是否为分区按runtime记录。共享比例、显存和隔离由平台能力决定，不能自行假设任意小数GPU请求可被调度。

**曝光匹配**回答相同非τ事件训练次数下的任务差异。**同计算匹配**在同一allocation上从fresh初始化执行T3/T4预算控制，记录实际wall、训练、验证、updates及±5%预算误差。跨GPU的device-hours仅作资源记账；不同型号、MIG和精度不解释为相同FLOPs。平台并发负载会影响wall比较，结果中保留实际吞吐。

[stage259](../../workbook/2026-11-15_stage259_r1_three_seed_task_evidence_and_scheduling.md)实际比较B2：1g MIG当前约22 events/s，2g Train-only probe为34.62/37.41 events/s；单seed三项操作的训练时间投影可缩短约4–5 h，尚不是最终wall测量。替代allocation先取得资源并通过profile，再确认旧作业eviction/abort与终态、归档未完成目录，最后以包含新allocation ID的release文件启动fresh T4/T3/compute。两项旧1g运行的2,616/2,929 s全部计费。初始40 events/s门槛拒绝了35–37的profile，按剩余完成时间证据改为32，并提交新的不可变attempt；此处调整资源启发式，模型/排名合同不变。

`submit --min-gpu-memory-mb`默认10,000，继续接受1g。上述替代探针临时使用20,000以寻找不同容量分区，不限定型号或要求整卡。B1继续在1g有效训练；接收任意卡与按实测为某架构选择更合适的卡同时适用。

[stage261](../../workbook/2026-11-15_stage261_r1_collaboration_and_count_audit.md)进一步测得两个H100 NVL Train-only profile为64.63/61.12 events/s；固定65门槛拒绝了仍可能节省约2 h的候选。后续探针使用可执行性下限，release前按实际旧进度、两边吞吐、fresh全部操作及600 s余量，要求至少节省1 h；不以速率是否超过某个整数代替剩余完成时间判断。30,000 MiB只用于这一组更大资源的候选，现有2g训练在新profile/余量条件满足前继续运行。探针、旧进度和失败attempt均记账。

[stage262](../../workbook/2026-11-15_stage262_r1_muon_working_points_and_time_handoff.md)已完成seed17的下一次迁移：1582928 profile为64.545 events/s，按当时余量预计节省1.993 h；旧1582921确认eviction/abort及history状态3、2,829 s全部计费，归档后才release新H100 NVL writer。seed2026继续2g有效训练，新候选尚在排队；不因seed17获得较大设备而让另一seed停止工作。

[stage265](../../workbook/2026-11-15_stage265_r1_b2_remaining_time_handoff.md)中seed2026候选1582929也已获配H100 NVL，实测82.757 events/s；以旧195,813次曝光的当时进度计算，预计节省2.336 h。旧1582922确认终态与eviction/abort、5,245 s全部计费后归档，04:53 UTC才release新writer。最终1582928/1582929均status4、Exit0，完整allocation分别14,182/11,914 s，各5 CPU，所含T4/T3/compute均交付。未运行旧allocation剩余反事实，不能把预计节省写成实测迁移收益。

## 3. Kubeflow

沿用[归档操作指南](../archive/2026-11-14_v1/08_kubeflow.md)的CERN flavor/resource、挂载与准入细节。context=`ml-prod`、namespace=`xcheng`。R1已实际使用该服务；2026-11-15约00:08 UTC凭据曾过期，R1最终账本使用已保存的终态证据。R2开始时访问已恢复，`pinpoint-r2-d01-k3`已在H100 NVL MIG 3g.47gb完成最多四个独立训练进程，后续正则化与顺序控制也已完成；最终账本采用live查询。不将历史过期状态当成当前不可用，后续凭据实际失效时再提醒用户登录。

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

已有Pod/Job终态均已保存且登录暂不可用时，可追加`--cached-kubeflow`避免重复认证；它只读取已有证据，缺失或未终止的记录仍保留unresolved。

`resources.json`按GPU型号/分区分别列出allocation和CPU core-hours，包含启动、编译、失败和恢复。记录仍运行的任务时明确为截至该时点成本，不标为最终成本。MIG device-hour不折成整卡hour。18 R1/80总device-hours是规划额度，按实测预测调整候选数量与后续矩阵，不因预算数字而宣称未完成实验已有结论。

R1全部screen、四pass、必要三seed、10项同计算控制、压力对照及终态验收已完成。45.708889 h为原18 h规划的2.539倍，不能把原80 h总额减去阶段计划值后照常执行剩余矩阵。R2已完成一族D0/D1/D2及必要控制；R3从M0全程重混控制开始，后续工作与成本重排见[执行路线](04_execution_plan.md)。

R2已实际测试单allocation内1/2/4/1并发：H100 MIG 3g.47gb的aggregate operation吞吐分别190.04/257.97/311.40/190.88 events/s，四并行比单进程约高63%，见[stage275](../../workbook/2026-11-15_stage275_r2_execution_and_parallel_gates.md)。这是多个独立模型共享一次staging的实测，并非单模型data parallel。CPU、RAM、worker和完整validation峰值决定可用并行度；已有任务完成时不为了维持四进程再增加没有科学问题的新任务。Condor另一MIG上的D2双并行也已完成。所有allocation各计一次，异构device-hours只作资源记账。

R3当前的实测选择见[stage298](../../workbook/2026-11-15_stage298_r3_partial_gpu_handoff_and_g1_readback.md)：3g.47gb三并行的各worker仅22.61/22.10/23.77 events/s，没有比三个独立1g更快；双并行为32.40/31.43 events/s。故只迁移control/relations，消融保留原1g，先确认旧writer终态再恢复100k checkpoint。计入准备/验证/丢失工作和余量后预计提前41.9min，10%吞吐惩罚下仍预计提前22.2min；这些是完成时间预测，不是已测反事实收益。两项新卡实际恢复均参数差0，全部正式分类终点尚未齐备。
