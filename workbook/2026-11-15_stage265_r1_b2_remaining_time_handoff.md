# R1：B2 seed2026 剩余完成时间迁移

## 阶段目标

只有在替代allocation已获得资源、完成真实Train吞吐探针且有明确剩余时间收益时，才迁移B2 seed2026。确认旧writer终态后归档并释放新writer，继续完成R1，不启动R2。

## 工作内容

已排队的候选1582929在`slot1_3@b9pgpun003.cern.ch`获得H100 NVL；没有追加新候选或限定整卡型号。其冻结Train前512 batches、zero-LR探针实际完成4,096次曝光、49.494359 s训练，测得**82.756905 events/s**，activations finite。探针不评价Tune，也不是学习效果证据。

04:52:52 UTC重新读取旧1582922的实际tail与新旧queue ads，按原至少节省1 h、fresh加600 s余量的规则计算收益。满足后移除旧writer；队列无旧任务、history状态3及event log eviction/abort均确认后，才归档旧未完成目录并写入绑定新allocation的release。

## 详细数据

| Quantity | Measured value / resource projection |
| --- | ---: |
| Old T4 training exposures | 195,813 |
| Old training time | 4,830.022667 s |
| Old validation time | 51.751847 s |
| New Train-only profile rate | 82.756905 events/s |
| Projected total exposures, three operations | 1,100,000 |
| Keep old: estimated remaining time | 22,303.134649 s |
| Fresh new: estimated remaining time, including 600 s margin | 13,891.942218 s |
| Estimated saving | **8,411.192430 s = 2.3364 h** |
| Required saving | 3,600 s |

曝光总数与时间是线性资源投影，不是最终wall实测；实际compute控制曝光会受验证/输入时间影响。新probe较快不代表模型质量改变。

旧1582922已确认JobStatus3、NumJobStarts1，allocation **5,245 s**；event log记录`condor_rm`导致eviction与abort。旧目录归档为`R1/failed_attempts/final_b2_t4_s2026_mig2g_stage265`，保留其中训练进度及健康记录。此前1g、2g、被拒probe和当前候选的全部分配时间均纳入成本，不把已丢弃曝光计入最终模型。

04:53:41 UTC写入`R1/runtime/b2_s2026_remaining_time_stage261.release.json`，allocation_id为1582929.0、previous_writer_terminal为true。新`final_b2_t4_s2026` manifest已经实际Running，GPU为H100 NVL、allocation为1582929.0。归档前该目录仍绑定旧1582922.0；新T3/compute目录当时不存在，没有两个writer同时写最终模型目录。新训练的最终指标尚未测量。

04:55:32 UTC的实时tail已显示新训练完成1,000次updates、7,995次曝光、86.364356 s训练，窗口NLL为1.364540、实际LR为.000119777；它已经进入实际优化，不仅是manifest创建。

完整决策与终态证据位于`R1/audit/b2_s2026_handoff_stage265.json`。`R1`为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。

04:54:09 UTC刷新记账：**38.920168 allocation device-hours**、273.093099 allocated CPU core-hours；当前3项Running、无Idle候选，分别为B3 compute与B2两seed。历史Kubeflow均使用已保存终态证据，没有重新触发认证。成本仍未关闭，异构device-hours不是FLOP等效。

## 验证命令与产物

```bash
condor_q -name bigbird20.cern.ch 1582922 -json \
  -attributes ClusterId,JobStatus,RemoteHost,NumJobStarts
condor_history -name bigbird20.cern.ch 1582922 -limit 1 -json \
  -attributes ClusterId,JobStatus,ExitCode,RemoteWallClockTime,NumJobStarts
condor_tail -name bigbird20.cern.ch -maxbytes 16000 1582929.0
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT" --cached-kubeflow
git diff --check
```

决策算术、同一allocation release、旧终态、目录归档和新manifest绑定逐项核对通过。当前34/41模型complete，完整R1评价/验收尚未运行。唯一finish控制器PID2052979经实时`ps`确认存活，没有启动第二个控制器。Cal-A/Cut-B/Confirm payload保持封存。

## 阶段性结论

**事实：**新资源已获配且实测明显更快；旧writer终态后完成归档及新writer启动。所有旧成本保留。

**待检验：**实际全程wall节省量及两个B2扩展seed的最终分类表现。不能把资源投影当成已实现时间收益，也不因资源迁移改变R1模型排名合同。

## 下一步计划

跟踪两项B2训练和B3 compute。完整41模型与全部压力路径完成后，由唯一控制器执行最终统计、独立复核和英文图表；终态资源关闭后执行accept、结果导出与最终报告。R2继续只设计。
