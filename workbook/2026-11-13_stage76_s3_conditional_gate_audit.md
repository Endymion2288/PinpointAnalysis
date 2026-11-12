# 阶段目标

在三seed确认训练继续运行时，检查最终验收是否逐项覆盖条件域的完整四分类比较，并核实恢复登录后的GPU进程与调度状态。

# 工作内容

`analysis.flavour.s3.accept.conditional_inventory`新增独立清单核对：条件域中必须包含最终评价的全部S3及ParT/ParticleNet模型；七个预定条件域不得缺失或重复；每个域都必须包含冻结配置中适用的全部单seed配对和三seed均值；每个比较须有raw/common-support的34项统计及400次bootstrap记录。物理上无定义的类统计保留`null`和有效replicate数，不能用缺失字段代替。最终gate另检查条件域ROC/PR PDF和曲线Parquet存在。

补充一个回归测试，分别删除条件域、配对和NC AP字段，确认不能通过；同时验证明确记录的无定义统计仍被接受。现有八项decision tests全部通过。当前运行中的finalizer在训练结束后通过独立Python进程调用accept，将读取新版本代码；训练配置、samples和proxy数值未调整。

# 详细数据

2026-11-12 23:35:00 UTC查询结果：Job `pinpoint-s3-confirm-mig47-parallel-v1`的Pod为Running，主容器重启0次。`nvidia-smi`确认4个CUDA训练进程；此前23:32:59 UTC读数为MIG显存15,595/47,488 MiB，分别约15.2/46.4 GiB。MIG的GPU utilization字段为N/A，无法从这个接口测得计算利用率；显存占用不能替代GPU utilization。包含原始输出与Pod状态的快照保存于：

`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/kubeflow/confirmation_gpu_process_snapshot.json`

当前并发任务为confirmation 0/1/6/7，均处于第二轮训练或该轮validation附近。最近progress中late两项为40,016 applied optimizer steps，tracker两项为27,016/27,266，AMP skips均为0；这些是运行中计数，尚非四轮最终结果。其余2/3/4/5由同一个supervisor在slot空出后自动启动。自动finalizer和readback monitor仍存活。17项screen已完成，八项confirmation尚未全部完成，S3没有最终验收。

验证命令（LCG108 CUDA环境，`PYTHONPATH=src`）：

```bash
python -m unittest tests.test_flavour_s3_decisions
git diff --check
```

实际结果：8 tests通过，diff格式检查通过。本次仅增加验收覆盖检查，没有重新执行screen bootstrap或训练。

# 阶段性结论

恢复登录后Kubeflow查询正常，四个训练进程继续执行。最终gate现在可识别“条件域汇总标记成功但冻结配对或四类字段不完整”的情况。当前没有新增的物理性能结论；最终成本和三seed比较仍未测得。

# 下一步计划

继续原Job与自动finalizer，等待八项确认训练和CPU读回完成。随后检查真实终态成本、所有四分类/条件域结果和英文图示，形成最终科学报告与方法晋级决定。
