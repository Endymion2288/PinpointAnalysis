# 阶段245：R1成本参照、交付依赖与实时资源约束

**补齐普通CNN成本参照和完整四类图表，R1正式训练继续执行。**统计程序的串行/4-worker输出逐字段一致，21项相关测试通过。B4无tracker的T3已完成，BA3=0.504533；其与完整输入重训的差值及区间尚未测量。完整排名、三seed统计和最终验收仍待训练依赖。

## 阶段目标

逐项对照R1合同，补齐路线图中“CNN作为成本参照”与完整四类ROC/PR和confusion交付；让依赖调度等待真实完整结果再推进统计。保持Cal-A/Cut-B/Confirm封存，继续优先利用兼容的整卡、共享GPU和MIG。

## 工作内容

1. `r1.models.ProjectionEncoder`新增`variant=base`，四层3×3 stride2卷积、通道24/48/72/96，采用GroupNorm/SiLU；使用同一物理投影、空视图处理和tracker late fusion。原ResNet为默认，已运行任务使用各自不可变源码快照。
2. 新增`r1.cnn_control`，先在真实Train事件上完成T3/T4学习检查，再各运行四pass。沿用B1选定的学习率，不新增HPO；标记为`cnn_cost_control`，独立于冻结的Scratch架构排名。`deliver`、成本图和验收同步纳入该项。
3. `r1.evaluate --workers`并行计算不同model/domain的bootstrap；所有比较复用同一组400次物理块权重、相同事件顺序。只改变执行并行度。
4. 新增完整四类one-versus-rest ROC/PR与T3/T4 row-normalized confusion图。原`r1_t4_roc_pr`继续表示T4训练后的条件三类，避免将其误作完整四类曲线。
5. 验收新增计算控制与T4参考的allocation/GPU、目标预算核对，以及排名所需seed/task清单检查。`r1.finish`替代临时依赖脚本，等待普通CNN等完整产物后才生成正式evaluation config，以8CPU/8 workers提交统计，随后按stress和图表依赖推进。完成提交不等于科学验收。

## 详细数据

### 当前完成的正式结果

输入均为冻结Train100k及Tune40k，T3 Train为其中75k非τ行，T4 Train为100k；非计算控制的神经网络每行四次曝光。以下为22:33 UTC以前已完成的单模型描述性结果，尚无最终paired区间。

| Model | Train exposures | Common BA3 | Operation wall (s) | Device |
| --- | ---: | ---: | ---: | --- |
| B4 T4 seed42 | 400,000 | 0.546767 | 2,024.190 | H100 NVL MIG3g.47gb |
| B4 T4 seed2026 | 400,000 | 0.545300 | 2,088.722 | H100 NVL |
| B4 T3 without tracker seed42 | 300,000 | 0.504533 | 2,122.267 | H100 NVL |

产物位于`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1/experiments/`对应`final_*`目录。B4无tracker的实际完整输入对照、普通CNN结果、各族四pass排名、tau污染与stress净效应均**未测量**。其他任务正在训练，manifest在pass边界更新；stdout中的updates/exposures用于核对实时进度，不把旧manifest计数误作停滞。

### 验证

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p 'test_flavour_v2*.py'
python -m analysis.flavour.v2.r1.evaluate --root "$R1_ROOT/smoke/graphics_a1" \
  --config "$R1_ROOT/smoke/graphics_a1/evaluation_config.json" --replicates 400 --workers 4
python -m analysis.flavour.v2.r1.plots --root "$R1_ROOT/smoke/graphics_a1" --results
```

21项测试通过（0.854 s），空投影有限反传检查同时覆盖ResNet和普通CNN。统计集成样例含8个model条目、3项pair、400次bootstrap；workers1和4去除执行时间/worker元数据后整个JSON相同。11套PDF/PNG生成成功，新增四类ROC/PR与confusion已目视检查英文标签和布局。此样例仅检验代码，使用的Train-only smoke分数不进入正式R1结果，不代表8次独立模型训练。`git diff --check`通过；新增依赖模块和验收修改通过Python语法核对。

### 调度与资源

Condor1582881/1582882在Idle且NumJobStarts=0时移除，history确认JobStatus3、allocated wall=0。尝试第二个Kubeflow MIG Pod时，server dry-run被实时策略拒绝：

```text
namespace-gpu-quota/check-total-gpu-units:
Total GPU units in namespace xcheng cannot exceed the quota of 1 units.
```

因此`pinpoint-r1-extend-b1-s42-k13` Job未创建、allocation为0。该限制来自集群准入，标准`nvidia.com/gpu`计数不能单独代表MIG可用配额。随后提交Condor1582883/1582884；22:33 UTC读回1582883已在T4训练，1582884仍排队。普通CNN成本参照已提交1582885。所有任务保留SM75下限，不限定具体GPU型号；CERN按站点规则调整最终CPU/RAM请求，实际申请以job ad为准。

证据为`R1/audit/cnn_queue_migration_stage245.json`、`k13_quota_rejection_stage245.json`和`runtime/*.submission.json`。原V100失败成本仍为0.063333 device-hours，继续计入总成本。各族完整矩阵的总成本尚未测量；18 device-hours为原规划，不能通过省去必要对照把未完成矩阵标为完成。

依赖控制入口：

```bash
python -m analysis.flavour.v2.r1.finish --root "$R1_ROOT"
```

临时`launch_remaining_a1.py`控制进程已终止，改用上述唯一控制进程；训练writer没有因控制器替换而重启。具体GPU操作与大规模统计仍全部提交批处理。

## 阶段性结论

**事实：**普通CNN成本对照的实现/提交、完整四类图表和统计并行验证已经补齐。当前Kubeflow namespace有效上限为1 GPU单位；Condor已接续排队的CNN训练。

**待验证判断：**无tracker点估计较低提示tracker值得继续保留，但须与同配方完整输入训练做配对统计后才能量化增益。当前数据尚不能确定最终强基线、三类与四类训练的优劣或最佳架构。

## 下一步计划

完成B1–B4的seed42 T3/T4及同allocation计算控制，再按冻结排名扩展前两名与B4参考到seeds17/42/2026。完成MAE初始化、无tracker重训和普通CNN对照，以及55条模型/条件诊断后执行正式统计、合作组域对照和英文图表。最终逐项核对训练覆盖、健康、paired区间、成本与后端终态，写完整R1报告后更新路线状态。
