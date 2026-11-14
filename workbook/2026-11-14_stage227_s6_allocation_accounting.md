# S6：allocation 记账与后端关闭合同

## 阶段目标

按实际后端记录计入失败、初始化和共享worker，明确排队、factory未materialize行与GPU设备类型。

## 工作内容

Kubeflow从实际PodScheduled到主容器结束计算allocation，包含image pull与初始化；另列主容器时间。MIG instance和整卡保持各自resource记录，一个MIG小时不换称整张H100小时。共享exec worker属于已有Pod，不重复记GPU费用。

HTCondor同时保存CPU allocation与RemoteUserCpu+RemoteSysCpu实际CPU时间，按QDate/JobStartDate给初始排队时长。已取消factory中没有backend ad的未materialize行不虚构运行时间；取消前状态和完整query状态一并保存。最终audit要求完整后端查询、无active任务、无未解释缺行、cleanup与独立backend closure证据。

## 详细数据

Transport共享GPU推理实际351.800秒，包含于原test Pod allocation。Generator错误schedd的48个materialized任务全部Idle、无JobStartDate，取消后220个任务由1582756接续。已保存pre-removal JSON，未分配任何虚构GPU或CPU运行时间。

约14:46 UTC：test推理203/227，transport truth302/368、generator105/220；完整allocation累计值尚未测量，待所有任务终态后汇总。GPU utilization fraction在MIG/RBAC当前接口下未测量，使用实际吞吐和allocation时长。

## 阶段性结论

资源合同现在区分实际计算、预约配额与排队；最终费用与closure仍需任务全部终态读回。

## 下一步计划

完成最终模型统计、paired条件、decay分析后获取所有backend终态，汇总资源并清理native中间产物和过时AFS脚本。
