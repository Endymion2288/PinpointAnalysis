# 阶段 34：S2 汇总表区分程序时间与实际 allocation

正式评价的 `summary.json` 与 `metrics.csv` 现同时给出程序 GPU-hours、所有尝试的实际 allocation GPU-hours 及额外开销。已有配对统计保持一致；五个完整正式模型、六项比较和 39 项 pending 未改变。

## 阶段目标

让最终资源主表直接包含冻结合同要求的实际成本，避免将训练 manifest 的程序计时用于替代实际 allocation。训练、选模与科学比较合同保持一致。

## 工作内容

原 `gpu_hours` 从 manifest 读取程序计时，`matched_allocated_time` 已独立使用真实 backend 核算。现在保留旧列兼容已有读取，新增 `program_gpu_hours`、`allocated_gpu_hours`、`allocated_seconds`、`allocation_overhead_seconds`、`allocation_complete`；summary 同时保存字段定义与 accounting 时间。

实际成本读取本 root 的 `campaign_accounting.json`，包含该实验的所有失败/替换/恢复尝试。只有 allocation 完整、全部尝试终态且模型完整时填入实际列；缺记录或尚运行时为 null，CSV 对应空值。CPU references 没有训练 GPU allocation 记录，其实际列也为空，不拿程序零值代填。正式与 V100 同名实验仍按各自 root 分开。

新增一项统计测试覆盖程序一小时而全部尝试实际 1.5 小时、缺记录、allocation 不完整、backend 非终态与模型不完整。现有配对统计测试合计 11 项通过（0.963 s）。随后用真实正式目录执行部分评价，正常退出；21 项集成核对通过，覆盖九行结果的新列、实际成本与 accounting 逐值一致、两项新增投影的完整配对对象与阶段33快照逐值相同、pending 仍为39。

## 详细数据

本次使用 03:09:45 UTC 的真实 accounting；以下是完整正式模型中的三个实例。

| Model | Program GPUh | All-attempt allocation GPUh | Allocation [s] | Extra allocation [s] |
| --- | ---: | ---: | ---: | ---: |
| ParT native joint | 1.91760982 | 1.94250000 | 6,993 | 89.60464 |
| PN native projection | 1.02004464 | 1.02305556 | 3,683 | 10.83931 |
| ParT 1mm projection | 1.79594004 | 1.80083333 | 6,483 | 17.61584 |

新列只是显式呈现既有核算，未把这次 CPU 评价开销计为 GPU 训练成本。先前 ParT native projection 的归档 allocation 与正在运行的替换共同记账，替换完整结束前不填该模型的最终实际列。当前正式评价为九个结果（四 references+五科学模型）、241 行分域指标、六项配对/39项 pending；完整标记仍为 false。

结果根目录 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`：

- `evaluation_final/summary.json`、`metrics.csv`：实际刷新后的资源列与字段解释。
- `explicit_resource_columns_checks.json`：21 项集成核对和实际数值。
- `evaluation_explicit_resource_times.log`：真实部分评价日志。
- `resource_time_columns.patch`：本次集成执行时相对 `cc19ad8` 的源码与测试差异；随后随本报告提交。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s2_evaluation
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
```

## 阶段性结论

最终表格现在可直接区分程序时间与实际资源成本，失败与恢复开销有独立列。科学统计未改变，当前未新增完整模型或训练结论。

## 下一步计划

继续五项完整、六项运行、24项排队的正式矩阵。新模型终态后更新 accounting、完成14项读回，再刷新预定比较。最终41+4严格评价与科学解释齐备后验收S2，calibration/test继续冻结。
