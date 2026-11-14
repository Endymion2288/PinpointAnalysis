# S6：统计流程集成验证与依赖调度

## 阶段目标

确认最终评价的输入合并、固定阈值、子集和文件输出能完整执行，建立转换/质量计算/推理之间的依赖收尾。

## 工作内容

新增 test finalizer，仅在全部正式推理和观测质量分块完成后合并 catalog/truth/probability/quality，创建共享 physical-block bootstrap。观测质量使用 claim worker 消费已经转换的分块；保持1个 shard 内存，避免重复读大 ROOT。模型评价仅读所需 probability columns，减少并发 EOS I/O。

## 详细数据

从 calibration 每类取8个事件，共32个，使用人工 Dirichlet 分数及临时目录的 fixture freeze 执行统计集成检查。完整流程包括 key merge、prior weighting、fixed selection、raw counts/exact intervals、reliability、conditional/cutflow、ROC/PR 和10次小型bootstrap；运行通过。人工分数不用于任何物理结论，不是实际 test 结果，未读取 test response。9项S6自动化测试再次通过。

生成器219个bulk任务中48个已运行，沿用通过的16事件pilot参数。Calibration tail重提的进程检查显示运行7分31秒、CPU5分44秒、RSS约1.40GB，说明它实际在计算；schedd上的CPU计数更新滞后，不能仅据旧计数定位原作业根因。仍待最后分块写出，未生成freeze。

## 阶段性结论

最终评价能够端到端处理四类数据，统计边界与文件接口通过检查。尚无完整calibration拟合或test评价结论。

## 下一步计划

完成temperature/阈值冻结，运行独立test和固定conditions，执行完整400次bootstrap、generator对照及score/energy response，量测资源并逐项完成S6合同。
