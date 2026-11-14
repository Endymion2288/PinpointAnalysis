# S6：推理policy与概率转换规则

## 阶段目标

用实测吞吐选择正式推理batch，落实单参数校准、prior scenario及独立validation工作点的数值定义，为完整calibration推理做准备。

## 工作内容

`configs/s6_inference.json`固定BF16、compiled encoder、batch64；calibration/test使用同一数值模式。每lane常驻13个模型并共享三种representation，allocation内多个lane处理独立分块。`s6/calibration.py`在inverse temperature上解单调NLL导数，等价于有界凸优化；calibration按四类均匀prior加权，ensemble先平均raw概率再校准。Prior变换仅使用预定scenario；每类阈值从既有完整validation按scenario的背景组成选择，整组ties保留，`>=`后FPR不超过目标。多类冲突与reject按已定规则执行。

`configs/s6_conditions.json`列出digitizer proxy、真实G4 transport及GENIE Standard/NUX比较参数。条件变体、transport和generator结果均尚未测量；本文件不代表最终analysis freeze，test仍未开放。物理exposure和当前schema不支持的timing项保持未测量。

## 详细数据

同一1024个calibration事件、H100 MIG47GB、13模型、workers6：batch16用41.485s（含首次编译）；后续batch64为11.117s、batch128为12.358s。64与128的throughput分别92.113及82.859events/s。GPU peak依次305.58、385.07、675.15MiB，峰值为进程自启动以来的累计值。相对batch16最大概率差8.345e-7，远小于0.003固定容差。batch16首轮有编译开销，不能把41.485s直接解释为稳定吞吐。

产物`audit/h100_batch_profile1024.json`及`runtime/pinpoint-s6-profile47.terminal.json`。Pod于12:36:12 UTC主container exit0、终态Succeeded。选择实测较快的batch64；多lane的最终利用率/吞吐尚未测量。

四项unittest通过：已知75%正确率、90%置信度的对称例子恢复temperature1.5；均匀概率经label-shift变换精确成为给定prior；固定阈值不拆ties；多类冲突和reject选择正确。命令`python -m unittest discover -s tests -p test_s6_calibration.py`，耗时0.016s。

从首个calibration shard读出的bar hardware map覆盖全部96个geometry panels，每panel对应40条bar；用于在包含未观测通道的完整硬件空间生成noise。ROOT tracker x/y/z原类型均为`vector<float>`，保存ideal float32没有附加精度损失；分辨率变体的nominal重放仍需实测。数学测试不替代真实事件变体smoke。

原bulk cluster2207854因bigbird21循环Requirements无法匹配，全部idle时取消；沿用S5的兼容性排除记录，重新提交cluster1582697到bigbird20。新申请2CPU/6000MiB、max_materialize64，已进入运行。29个分块共92697个calibration事件已完成时启动下一轮GPU安排。此前2CPU/8000MiB申请会被CERN自动改为3CPU/9000MiB，reroute使用2CPU/6000MiB以匹配线程预算。

## 阶段性结论

正式数值模式及概率规则可执行，真实batch变更保持分数一致。完整calibration尚在进行，没有最终temperature、threshold或test结论。

## 下一步计划

运行多lane完整calibration推理并按实际利用率调整并发；只在数值等价的policy内安排资源。完成真实变体smoke与G4接口后，拟合校准并冻结所有评价规则，再执行独立test。
