# 阶段 23：Startd 关闭后的 checkpoint 与排队恢复

两项运行中的 ParT 任务因 `Startd was shutdown` 返回 idle，随后均以原 JobId 在 V100 上从已提交 epoch 恢复。两份 atomic checkpoint 读取成功，optimizer/AMP counters 与新进度一致。其余科学任务继续运行。

## 阶段目标

确定执行中断原因、检查已有恢复状态，并将中断成本纳入 S2。同一冻结配置继续完成原定四轮和完整比较合同。

## 工作内容

20:35 UTC 启动只读观察，每分钟查询具体 Condor JobId、状态与 checkpoint/progress。20:44–20:45 UTC 的运行数从 5 降为 3，随即读取 Job ClassAds。

两项 `NumVacatesByReason` 均为 `StartdShutdown: 1`，`VacateReasonCode=1013`。JobStatus 从 2 转为 1，仍存在于队列。原任务会在重新匹配资源后运行既有训练入口，由 `last.pt` 恢复 optimizer、scheduler、scaler 和 RNG。

对两份 checkpoint 进行 CPU readback，检查冻结配置一致、epoch/history 一致、参数有限及恢复状态字段。随后仅对这两个 idle JobId 设置 JobPrio=20，优先恢复已有训练；其余模型和输入配置不变。

## 详细数据

| Job | Experiment | Vacate UTC | Previous allocation [s] | Last committed epoch | Checkpoint elapsed [s] | Allocation outside checkpoint [s] |
| --- | --- | --- | --- | --- | --- | --- |
| `1573747.0` | `part_native_joint_s42_e4_v2` | 20:45:19 | 8,239 | 1 | 4,859.44940 | 3,379.55060 |
| `1573747.3` | `part_1mm_projection_s42_e4_v2` | 20:43:59 | 8,039 | 2 | 5,989.93211 | 2,049.06789 |

两项均为第 1 次 allocation 后返回 idle，NumVacates=1。表中最后一列合计 **1.50795 GPU-hours**，包括程序准备和未提交 epoch 的 allocation，不能直接当作纯训练秒数；最终等 GPU-time 比较会通过实际 allocation 核算包含这些成本。

Checkpoint readback 使用 LCG_107/Torch2.5.1、CPU，正常退出。原生联合位置为 20,005 applied steps/11 AMP skips；1 mm 投影为 40,014/18；两者参数全部有限、配置等于冻结 campaign、epoch 数等于保存 history 长度，optimizer/scheduler/scaler/RNG/best-model 状态存在。实际重新开始执行仍待调度。

20:47:52 UTC 状态快照：两项原 JobId 仍 idle、NumJobStarts=1、JobPrio=20；总科学任务 35 项，其中 3 running、32 idle。未新增训练提交。

20:48:43 UTC accounting 更新为 **12.20294 GPU-hours**，missing history records=0。其中这两项原 JobId 分别累计 8,239/8,039 s，当前 JobStatus=1，已用分配时间完整保留。

证据均位于 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/`：

- `startd_shutdown_observation.json`：原始相关 Job ClassAds。
- `startd_shutdown_checkpoint_readback.json`：实际 checkpoint 检查。
- `startd_shutdown_resume_plan.json`：优先级核实、恢复位置和 allocation 差额。
- `training_observation.json`：滚动最新只读快照；运行状态以其中 Condor JobStatus 为准，训练 manifest 的 `running` 可早于重排事件。
- `campaign_accounting.json`：累计实际 allocation，含中断任务的已用时间。

执行的调度操作：

```bash
condor_qedit -name bigbird20.cern.ch \
  -constraint 'ClusterId == 1573747 && (ProcId == 0 || ProcId == 3) && JobStatus == 1' JobPrio 20
condor_q -name bigbird20.cern.ch 1573747 -af ProcId JobStatus JobPrio RemoteHost
```

`condor_qedit` 返回 2 matching jobs；后续查询确认这两个 JobId 均为 priority20。

## 阶段性结论

补充实际执行证据：`.0` 在 `slot1_1@b9g47n2007.cern.ch` 恢复，manifest 时间为 21:44:20 UTC，`resumed_from_epoch=1`。21:47:27 的 epoch2/step500 对应 applied steps=20,505（原20,005+500）、AMP skips=11、loss=1.27536。

`.3` 在 `slot1_1@b9g47n2008.cern.ch` 恢复，manifest 时间为 21:50:04 UTC，`resumed_from_epoch=2`。21:58:19 记录的 epoch3/step3,250 对应 applied steps=43,262、AMP skips=20，累计 attempted=43,282，与原40,014 applied/18 skips衔接一致；loss=1.26417。两项 `NumShadowStarts=2`，`NumJobStarts` 仍为1。原始 manifest/progress 与调度事实保存在 `startd_shutdown_resume_execution.json`。

22:40:04 UTC 新证据：`.3` 已提交恢复后的完整epoch3 checkpoint，随后进入epoch4。LCG108/Torch2.7.1使用CPU执行 `torch.load(last.pt, map_location='cpu', weights_only=False)`：106个model-state tensors、318个optimizer-state tensors全部有限，best probabilities有限；checkpoint/history均为3轮、配置等于运行manifest，scheduler/scaler/四组RNG恢复字段存在。累计applied=60,022、AMP skips=26，合计60,048=3×20,016 attempted；checkpoint elapsed=8,927.58764 s。8项检查均通过，原始结果为 `startd_shutdown_post_resume_checkpoint.json`。这是恢复后新完整epoch的实际读回，四轮任务尚未终态。

已确认的中断原因是执行侧 Startd 服务关闭；原任务均已恢复执行。可恢复的进度为已提交 epoch，未提交 epoch 的部分训练重新执行，实际成本保留。

中断与恢复证据不支持新增架构或粒度结论。其他科学模型的完成结果另见阶段27/28；S2仍进行中，calibration/test保持冻结。

## 下一步计划

`.3`恢复后的完整checkpoint已经读回通过，继续等待四轮终态；`.0`继续跟踪恢复后的完整checkpoint。全部35项完成后按原41项单seed配对、4项种子均值与科学交付清单验收。
