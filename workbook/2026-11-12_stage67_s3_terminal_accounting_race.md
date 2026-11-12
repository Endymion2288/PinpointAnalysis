# 阶段67：S3终态查询竞态与六模型读回

## 阶段目标

修复作业在history与queue两次查询之间退出时的账本缺口，继续记录完整四分类screen。冻结训练和接受的数据保持原值。

## 工作内容

原账本先查history、再查当前queue。`1573835.2`恰在两次查询之间结束，第一次查询时尚无history，第二次查询时已离开queue，出现一条`not_in_current_queue_or_retained_history`。单独查询真实history确认它已正常终态；并非实际缺失历史或模型失败。

共享accounting改为先取queue快照，再按已记录submission中缺席的精确ClusterId/ProcId查history，并用缺席数量限制匹配数。若一个运行作业在查询之间结束，当前快照保留其明确标记的运行估算，下一次刷新读取终态。仍然找不到的记录继续显式标missing，不补造零成本。全部作业真实终态才可最终验收。

该查询也减少为仍在运行或idle的任务扫描历史。S3通过同一入口复用修复；只重写S3自己的账本，不改S2结果。模型、optimizer、配置和训练计算未改。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

| Item | Actual observation |
| --- | --- |
| Affected model | `s3_f16_joint_s42_e4` |
| Actual terminal Job | `1573835.2`, bigbird20, JobStatus4/ExitCode0 |
| Actual allocation | 6,798 s = 1.888333333 GPUh |
| Inconsistent snapshot | 16:15:48 UTC, missing1, 7.837456065 GPUh；未作为完整成本使用 |
| Refreshed snapshot | 16:17:57 UTC, missing0, 9.833573274 GPUh |
| Organization bound after refresh | 10.014715888 GPUh，含首次交互profile保守上界 |
| Targeted checks | 5 tests pass, 0.008 s |

检查覆盖结束发生在两次查询之间时保留live快照、全live队列不扫描history、历史仍缺失时保持missing，以及既有部分提交ID与多root/后端成本隔离。日志`unit_accounting_race.log`、`accounting_progress_1615.log`和`accounting_progress_race_fixed.log`。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s2_accounting
python -m analysis.flavour.s3.accounting --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12
```

另有三项真实完成，至此6/17 screen模型均四轮、80,064 updates、零AMP skips、14项CPU读回通过，共84项。

| Newly completed model | Macro AP | NLL from training history | Macro F1 | Balanced accuracy | Allocated GPUh |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-pitch joint | 0.346749611 | 1.340082256 | 0.327626581 | 0.3370 | 1.888333333 |
| Primary native | 0.250000000 | 1.386298356 | 0.100000000 | 0.2500 | 1.562500000 |
| Primary 1 mm | 0.250000000 | 1.386298203 | 0.100000000 | 0.2500 | 1.106666667 |

主native的Condor `1573837.0`分配5,625 s，1 mm的`1573835.5`分配3,984 s，均JobStatus4/ExitCode0。三个新增模型均选择第四轮。Joint四类AP为[0.430235422, 0.343227919, 0.288928786, 0.324606315]；两个主模型四类AP均0.25。

最终主native与1 mm各自在完整10,000 validation事件上只有1组不同概率向量。两者macro AP差为0，400次paired事件区间[0,0]；这来源于常数预测，不支持探测器表示的物理等效。阶段65已用CPU FP32梯度诊断定位主native第二轮的分类头全零GELU；最终常数预测与该机制一致，首次触发step和所有事件的逐层梯度未测量。

已有6个S3和16个只读S2模型的非strict四分类评价可执行，22模型、8项已可计算冻结配对、26项仍pending。原先生成的资源项使用了有缺口的账本，因此账本修复后重新运行`analysis.flavour.s3.evaluate`并覆盖`evaluation_progress/`；不能将旧的joint allocation不完整记录用于Pareto选择。图标题、坐标和图例使用英文。

## 阶段性结论

事实：账本缺口由查询竞态引起，现已通过针对性检查及真实终态回读修复。进行中的成本为快照，最终必须所有后端终态且missing0。六个完成模型的训练与输出完整，不代表六者均具备有效分类能力。

主native/1 mm的预测塌缩限制了本seed对分辨率假设的解释；不能以极窄零差值区间宣称native与coarse输入物理等效。S3筛选仍须全部17项和冻结确认矩阵，不提前选择候选。

## 下一步计划

继续剩余11项screen、原失败任务的第四轮和最终读回。新获配的4 mm context-latent及含tracker配置已在HTCondor运行；Kubeflow仍Pending，继续等待真实ready再唯一交接。完整screen后冻结并执行确认，完成所有四类、三seed、条件域、错误显示、资源图和strict gate。
