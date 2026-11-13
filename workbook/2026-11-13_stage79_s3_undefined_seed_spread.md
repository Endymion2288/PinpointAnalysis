# S3未定义三seed统计的JSON表示

## 阶段目标

保证条件域或加权支持中缺少某类时，最终数值表保留未定义统计的含义，并能写入标准JSON。

## 工作内容

检查`analysis.flavour.s3.metrics.interval_report`发现：delta、CI和逐seed值已将非有限数转换为`None`，但`training_seed_sample_SD`直接保存`np.std`结果。缺类时该结果为NaN，与其余字段的null约定不一致，也不能通过最终导出工具的`allow_nan=False`。

将非有限SD明确写为`None`，保持有限值的计算不变。扩展已有缺类统计测试：在实际四类toy数据上将第四类权重设为0，经过`MetricCache.compute`和三seed区间报告，核对该类AP有效replicates为0、逐seed值为三个null、SD为null，并实际执行`json.dumps(..., allow_nan=False)`。未用0或丢弃缺失seed来替代未定义SD。

## 详细数据

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s3_metrics tests.test_flavour_s3_decisions
```

实际exit0，13项测试通过，unittest计时0.301 s。覆盖weighted/tied statistics与sklearn对照、物理block resampling、共享seed区间、缺类JSON、四分类晋级及完整域合同。最终真实条件数据中出现未定义三seed SD的次数尚未测量；当前训练未完成，未伪造完整评价输入进行导出。

本次不修改训练模型、数据、proxy、配置、checkpoint或已发布概率。运行中的训练模块与读回进程继续执行；finalizer后续启动独立评价子进程时会加载修复后的统计函数，无须重复提交GPU任务。

## 阶段性结论

未定义统计现在可用标准JSON的null完整表达，有限统计的数值算法保持原样。该修复使最终导出与既定物理分母约定一致，不改变模型质量或输入信息的科学结论。

## 下一步计划

继续等待当前四项确认训练，随后检查真实完整评价的未定义项、有效replicate数、条件域分母和导出结果。完成全部回读、区间、终态成本和图示复核后更新最终报告与roadmap。
