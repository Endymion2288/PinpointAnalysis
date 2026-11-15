# R3：机器计划同步与完整交付范围核对

## 阶段目标

同步机器可读计划与真实执行状态，逐项核对当前完成范围和未完成依赖。完整R3范围继续保留，不以已完成模型代替整个阶段。

## 工作内容

发现`roadmap/v2/experiment_plan.json`仍记录G1训练中、G2尚未正式启动、C1尚未实现。将其同步至M0/G1/C1完整评价不晋级、三个G2正式训练中；补充revision2交付矩阵、CPU依赖观察进程、实际结果引用和注明时间的资源快照。原始阶段work、任务、分区、公平控制、baseline、seed、指标、统计、晋级门槛和学习要求逐项保持相等。

对冻结completion/campaign及两份geometry评价合同做source与live解析比较，四项均相等；历史`delivery_stage290.json`保留其形成时状态，不修改为当前结果。

## 详细数据

运行根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。

2026-11-15 18:24:58 UTC逐项核对manifest、预测表是否存在、评价与独立复核状态及24项证据metadata，产物为`audit/completion_scope_stage301.json`：

| Frozen requirement | Current evidence | Remaining |
| --- | --- | --- |
| Six formal models | M0/G1/C1 complete; all have both prediction tables | Three G2 fits still running; no complete prediction tables |
| M0 evaluation | 4 views, 8 pairs; results and both independent reviews complete | Final synthesis and figures review |
| Geometry evaluation | Contract: 12 views, 48 pairs | Complete G2 endpoints, full statistics and two independent reviews |
| C1 evaluation | 4 views, 8 pairs; results and both independent reviews complete | Final synthesis and figures review |
| Global-ablation / strong projection | Contract: 4 views, 8 pairs | Complete no-global endpoint, statistics and independent reviews |
| Required evidence | 19 of 24 paths at expected metadata status | Full formal health, native probes/readback, final scientific review, conditional decisions, final visual review |
| Matched optimizer controls | All five factor definitions retained | Verify final exact exposures, updates and LR sums |
| Final decision and cost | Decision files absent; two GPU allocations active | Evidence-based extensions, reviewed decisions, terminal accounting and acceptance |

上述“19/24”只是已声明证据路径的metadata核对，未替代内容复核或最终accept。三G2的100k保存prefix也不代表当前真实进度，最终仍需各400k有效曝光。M0/G1各150,117updates/1.2M曝光，C1为50,051updates/400k曝光；原要求不缩减。

18:23 UTC实际Pod状态为Running、主容器状态running；Condor1583010.0为JobStatus2，RemoteHost`slot1_1@b9pgpun106.cern.ch`，累计user/sys CPU14,177/143 s。CPU observer通过真实tool session39030轮询确认仍在运行；当前缺终点，未提交正式geometry评价。未据观察超时重启任何训练。

验证：JSON解析、四份冻结合同source/live相等、九组研究合同字段修改前后相等、原R3 work相等、四套评价列表相等；没有训练代码变化，不重复运行刚通过的45项模型测试。没有读取Cal-A/Cut-B/Confirm payload。

## 阶段性结论

事实：机器可读路线与文档执行状态已一致，原六模型和全部评价/证据要求保留。三个G2仍真实运行，完整R3尚未完成。

待测：正式G2分类收益、最终特征可读性与重训消融结论。已有负结果只约束已测配方，不支持探测器原生信息无效或模型达到最佳的结论。

## 下一步计划

等待当前真实训练完成，并核对按依赖自动提交的完整CPU评价、健康与科学review。检查G2自动最终probes后，用完整效应、质量约束和成本决定必要扩展；完成全部条件决定、英文图审查、终态账本、严格accept及最终路线图回填。
