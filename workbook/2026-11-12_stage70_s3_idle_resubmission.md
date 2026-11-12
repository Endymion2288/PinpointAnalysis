# 阶段70：S3剩余 idle 任务的 schedd 复核与安全重提交

## 阶段目标

减少剩余 S3 screen 的排队时间，同时保证已经运行或已经发布的模型不被迁移，不产生重复 writer。

## 工作内容

实时状态显示 12/17 模型已完成并通过 readback；task 11、12、13、15、16 对应的旧 Condor 作业 `1573835.10/.11/.12/.14/.15` 均为 `JobStatus=1` idle，五个对应模型目录均为 `not_published`。这些作业未启动、没有 checkpoint writer，因此按 roadmap 允许迁移。

检查了 `bigbird21.cern.ch`：WRITE 为 `ALLOW xcheng@cern.ch`，但 `SUBMIT_REQUIREMENT_NoEos` 返回完整表达式而不是仓库选择器要求的 literal `True`。仓库的 `analysis.flavour.s3.condor` 因此按保守规则跳过该 schedd；没有绕过 direct-EOS 策略强行提交。自动选择结果仍为 `bigbird20.cern.ch`。

先逐个执行 `condor_rm -name bigbird20.cern.ch 1573835.{10,11,12,14,15}`，随后 `condor_q` 不再返回旧 cluster 行。使用同一冻结 campaign 和 tasks `[11,12,13,15,16]` 提交新 cluster `1573917`，实际 schedd 为 `bigbird20.cern.ch`；五项均 idle、尚未获得 H100。随后设置 JobPrio `[20,16,15,12,10]`，并以 `condor_q` 回读确认。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

| Item | Actual value |
| --- | --- |
| Completed/readback before migration | 12/17, 12/17 |
| Old cluster | `1573835`, bigbird20 |
| Removed idle processes | `.10`, `.11`, `.12`, `.14`, `.15` |
| New cluster | `1573917`, bigbird20 |
| New processes/tasks | `.0/.1/.2/.3/.4` → tasks `11/12/13/15/16` |
| Confirmed JobPrio | 20/16/15/12/10 |
| New jobs status | all idle; no RemoteHost |
| Kubeflow | Pending; no node |
| Cost after resubmission | 15.903611111 GPUh; organization upper bound 16.084753725 GPUh; missing history 0 |

原始记录为 `migration_idle_controls_190736.json`、`submission_screen_condor_bigbird21_v1_190736.json`、`campaign_accounting.json`。旧作业删除和新作业提交是顺序操作；不重复计算已完成的 12 项。

## 阶段性结论

五个旧 idle 副本已安全替换为同一冻结配置的新提交，未改变训练输入、seed、任务或 writer 合同。bigbird21 因 direct-EOS 策略检查未通过而没有使用；这避免了将完整 EOS 输出写入未经仓库策略确认的 schedd。新作业目前仍等待 H100，迁移尚未产生 GPU allocation。

## 下一步计划

继续由 monitor 观察 17 项的模型发布和 CPU readback；新作业一旦获得 GPU，继续执行完整四分类评价、400 次配对 bootstrap、严格账本和 Pareto 选择。Kubeflow dispatch Job 保持原队列位置；若真实 Scheduled/Running，按唯一 handoff 规则交接一个仍 idle 的任务。
