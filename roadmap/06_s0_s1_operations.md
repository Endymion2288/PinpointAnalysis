# 历史文档：S0/S1 实现与运行入口

此页对应已完成的 v1（S0–S6）路线。[阅读归档原文](archive/2026-11-14_v1/06_s0_s1_operations.md)；当前优化计划从[新版路线图](README.md)进入。

<a id="s0s1-实现与运行入口"></a>
<a id="环境与目录"></a>
<a id="全库身份盘点"></a>
<a id="四个-pilot-audit-完成并验证后"></a>
<a id="全部-scan-audit-完成且无文件错误后"></a>
<a id="原生转换数据加载和-b0"></a>
<a id="小样本验证通过后通过-condor-转换每类一个完整-chunk"></a>
<a id="b0-是-4000-event-级的小规模验证可在交互-gpu-执行"></a>
<a id="全部-pilot-的多-worker-io-测量通过-condor"></a>
<a id="最多-64-event-原生-tensor-传输检查可在交互-gpu-运行"></a>
