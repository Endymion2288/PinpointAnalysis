# S6：transport 输入并行与 test 推理启动

## 阶段目标

减少1024个固定transport事件逐文件读取的串行等待，保持冻结事件/顶点/物理条件不变。

## 工作内容

原单CPU输入准备进程运行9分17秒时CPU1分34秒、RSS170MB，多数时间等待I/O；改为每个16事件transport任务直接读取对应accepted primaries，保存小型input part后立即执行真实G4。单线程uproot executor与CPU1匹配，避免在大量worker中嵌套默认线程池。失败恢复可以复用input part及已完成条件，无须重读同一批ROOT。

## 详细数据

独立test已有20/227块完成转换。GPU提交：HTCondor1582732（两lane×三loader worker，CPU8/24GB）；Kubeflow `pinpoint-s6-test47-r1`（四lane×四loader worker，47GB MIG）。共享claim，按冻结BF16 compiled/batch64执行。实际test概率和新transport结果尚待读回。

原transport输入job1582725将取消，替换为小批并行输入和G4 pilot；原样本、subset、seed、条件幅度不变。Generator3512事件的真实transport、三seed推理和全部冻结prior对照已经完成，见stage220。

## 阶段性结论

Test进入转换/推理并行阶段。小批transport输入实现已完成，需要CC e与CC τ各16事件pilot确认实际fields和八种条件后扩展。

## 下一步计划

运行task0与32，核对原顶点、G4 field/density设置、τ helicity及紧凑truth读回；随后执行其余62批。继续全部test/paired统计，不用结果改工作点。
