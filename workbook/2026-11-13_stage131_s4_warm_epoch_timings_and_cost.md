# S4 缓存 worker 的训练实测与成本修正

## 阶段目标

测量缓存接入后的真实训练路径，核对 GPU 与 CPU allocation 的成本口径，等待独立 MAE seeds 完成。

## 工作内容与详细数据

HTCondor 1577948（bigbird20，b9pgpun206）使用 H100L 24GB MIG、6 CPUs、16,000 MB 申请内存，以冻结源码449e056同时执行MAE5的1k和10k标签训练；固定分类输入缓存上限为每reader 1024 MiB。没有修改运行中模型、采样或源码。

1k任务的实测记录为：

| Epoch | Train [s] | Data wait [s] | Forward/backward [s] | Validation [s] |
| --- | ---: | ---: | ---: | ---: |
| 1 | 154.62910 | 34.60512 | 116.85946 | 137.26988 |
| 2 | 50.42485 | 5.47645 | 43.75696 | 36.03473 |

这是同一正式任务不同epoch的实测，包含编译预热、事件次序、异步加载和缓存的共同影响；没有据此估计缓存的单独因果加速比。此前stage130的原始/缓存reader同输入CPU profile保持逐位等价，可独立支持重复解码开销的减少。24GB MIG 的GPU利用率未测量；nvidia-smi返回N/A。完整epoch时序保存在`roadmap/results/s4_training_cache_measurement.json`。

成本程序此前遗漏了新增的CPU-only `cache_profile` stage，把1577642的132秒allocation误列为GPU时间。本次去除0.0366667 device-hours，保留0.1466667 CPU core-hours。22:36:27 UTC刷新后的组织累计为12.50895646/60 device-hours、169.52835072 CPU core-hours；该数字包含之后仍在运行的worker截至快照时的费用。`tests.test_flavour_s4_accounting`的3项检查通过，覆盖CPU profile不产生GPU费用、活跃队列查询及终态history缓存。第一次使用系统Python3.9执行因缺pandas未导入测试；切换正式LCG108 Python3.12环境后通过。

Kubeflow登录有效，原47GB MIG继续运行。额外GPU请求受namespace配额限制，沿用已有HTCondor allocations。MAE3 seed17的1k训练已完成；seed2026仍在最后两轮。所有新增指标继续以完整10k validation统一读回，calibration/test冻结。

## 阶段性结论

缓存worker的重复训练/validation路径已出现较短的实测耗时，尚未隔离全部速度来源。CPU与GPU成本分类已纠正。独立seed的低标签分类收益仍需配对统计。

## 下一步计划

两个MAE独立seed完成1k训练后提交v7统一读回，覆盖新完成的matched-compute、channel Rel100k和frozen probes；继续执行MAE5、Rel与EMA下游矩阵，并检查内部健康、内存及实际训练耗时。
