# S6：生成器真实 transport pilot

## 阶段目标

验证旧 GFaser 桥接、共同 PINPOINT 顶点及 current detector transport 的完整路径，量测批量资源。

## 工作内容

HTCondor 使用匹配 LCG107 的 PyROOT 写 current GFaser，再执行隔离 G4，转换为 native observation、ideal tracker 和 metadata。逐事件核对输出顶点与 benchmark 输入一致，删除暂存 ROOT。成功日志压缩，保留复现配置与 compact response。

另外补充 predeclared exact-target overlap 评价，沿用 S5 的固定 joint E/x/y/z/exact isotope bins 和每类每 cell 最少5事件。该诊断比较已固定的候选，不用 test 改模型或工作点；完整域仍为主结果。

## 详细数据

`s6_generator_transport_pilot_r1`、HTCondor1582717，task0共16事件，nominal真实 transport+转换46.903秒，status complete。输入为 matched NUX 子集，output顶点与输入在0.0002mm内相同。输出 `generator/shards/nominal/0000/`。Bridge PyROOT 类型读回以及9项S6测试通过。

Kubeflow主校准Pod的watch保留成功终态：2026-11-14 12:58:47Z至13:28:12Z、exit0；四lanes共104个已完成分块。Condor两lanes共122个分块，worker用时1352.689/1347.183秒，GPU为H100 NVL。终态完整CPU/GPU allocation计费待最终汇总，不把worker秒数直接相加当device-hours。

Calibration最后一个转换块仍在运行，冻结尚未发生。生成器分类效果、真实条件差异和test结果未测量。

## 阶段性结论

生成器输入到当前几何响应的完整路径通过16事件pilot，可以扩到已固定的3512事件。原GENIE事件独立，只有benchmark支持匹配，不作paired hard-scatter比较。

## 下一步计划

批量生成全部generator response；完成calibration tail和拟合后冻结。并行执行完整test、32k读出条件、1024事件真实field/material transport及固定评价。
