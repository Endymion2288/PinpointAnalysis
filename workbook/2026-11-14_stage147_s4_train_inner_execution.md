# S4 train-inner正式执行与修复后frozen probes

## 阶段目标

执行[阶段146](2026-11-14_stage146_s4_train_inner_optimization_contract.md)冻结的48项低标签优化，并补齐revision 3下的frozen label-efficiency参照。此报告记录运行中的实测状态，不作尚未齐全的配方选择或独立validation结论。

## 工作内容

正式训练使用提交`d9bd2f6`的本地scratch源码快照。Kubeflow的H100 47GB MIG已获配，六进程运行；HTCondor同时提交可接受整卡/A100/24GB MIG的两进程候选，以及12GB MIG单进程候选。三个allocation共享互斥claims，不重复写同一实验。

Condor选择`bigbird21.cern.ch`：提交时该schedd为0 idle、0 held，通过WRITE及AFS小提交目录/EOS worker日志策略检查。即时schedd完整列表和选择依据保存在各`logs/submission_pinpoint-s4-inner-*.json`。提交目录为AFS下`~/condor/Pinpoint/flavour_s4/`，训练数据、日志、模型和图均在EOS输出根。

另冻结[6项frozen probes](../roadmap/configs/s4_empty_frozen_probes.json)：Scratch/MAE3×1k/10k/100k，seed42，均使用修复后的空响应路由。分别读取full和global表示，拟合只使用原标签子集的StandardScaler及multinomial logistic regression，C固定为1，不以validation选择超参数。这回答新路由下可线性读出的信息，与固定checkpoint分支置换及完整fine-tuning分别比较；当前6项尚未执行。100k的probe只改变拟合标签数，不增加SSL corpus或扩至1M。

## 详细数据

输出根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。02:12:27 UTC的实时队列及训练快照见`audit/s4_train_inner_launch.json`：

| Measurement | Result |
| --- | ---: |
| Frozen train-inner matrix | 48 fits |
| Completed / running fits | 18 / 6 |
| Updates in committed epoch histories | 11,244 |
| Event exposures in committed histories | 145,977 |
| Internal-health samples | 284; all passed |
| Intentional LP health samples | 43 |
| Maximum sampled preclip norm | 227.387479 |
| Kubeflow cgroup memory / limit | 25,464,647,680 / 62,999,998,464 bytes |
| Observed Kubeflow cgroup peak | 37,217,476,608 bytes |

其中历史计数只包含已提交的完整epoch，正在进行的更新不计入该行。LP解冻后的第一个更新已触发完整健康检查；最大梯度仍低于冻结的10,000 review阈值，未重现约10¹²异常。

| Allocation | Live state at snapshot | Concurrency |
| --- | --- | ---: |
| Kubeflow `pinpoint-s4-inner-main-pdwqm` | Running, `ml-004-h100-001-uusckqrqfxt3-node-0` | 6 |
| Condor 2204846 / `bigbird21.cern.ch` | Idle | 2 |
| Condor 2204847 / `bigbird21.cern.ch` | Idle | 1 |

Kubeflow于02:03:26 UTC启动子进程；同allocation共用2,144,919,451 bytes的本地压缩输入拷贝，staging耗时1.055618 s。已完成首批1k Scratch三折，独立MAE配对及完整选择矩阵尚未齐全；单折或各折独立最小值不替代预先声明的共同OOF选择。

MIG的`nvidia-smi`对memory返回Insufficient Permissions，对utilization返回N/A，metrics API返回Forbidden。因此GPU百分比未测量；实际并发、cgroup内存、训练吞吐和各阶段时长保留。该权限状态不影响训练，当前CERN登录及Pod状态查询正常。

02:10:56 UTC累计S4账本为24.461239/60 device-hours，CPU allocation为285.735880 core-hours，包含正在运行的allocation；剩余预算35.538761 device-hours。旧阶段145的24.325865是其终态时点，本表继续累计同一上限，不重置预算。

本阶段没有修改训练源码，复用阶段146的41项测试通过结果。正式命令形态：

```bash
python -m analysis.flavour.s4.submit \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 \
  --backend kubeflow --stage campaign \
  --campaign roadmap/configs/s4_low_label_inner.json \
  --name pinpoint-s4-inner-main --concurrency 6 \
  --flavor h100-47gb --deadline 14400 --tasks {0..47} --apply
```

三份提交均包含0–47全部任务，精确arguments、admission及source snapshot见EOS submission记录。Calibration/test未使用，train-inner拟合及选择均未读取独立validation。

## 阶段性结论

事实：已完成18项真实train-inner训练，运行的LP/full-FT健康检查通过，47GB MIG实际六进程稳定运行。Scratch的低标签NLL仍存在不足，完整MAE配对尚待完成，不能提前给出最优配方。

假设：更早checkpoint或LP-FT能否改善重拟合后的概率质量和τ决策，仍待共同OOF选择及完整独立validation评价。冻结probe将补充表示可读性证据；其结果不能代替full fine-tuning收益。

## 下一步计划

完成48项冻结矩阵及真实历史/健康审查，按固定OOF规则生成12项重拟合；保持固定末轮选择，并与既有修复基线作完整10k事件的配对评价。随后执行6项新路由frozen probes、按同设置复核其他预训练目标及compute-matched控制。所有allocation结束后记录终态、更新账本并清理终态AFS提交副本；运行中的脚本继续保留。
