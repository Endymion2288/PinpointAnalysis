# S4 推理统计依赖与独立seed调度

## 阶段目标

修复CUDA路径诊断的启动依赖，继续推进独立MAE初始化的下游验证。

## 工作内容与详细数据

首个附加CUDA推理执行`s4-functional-cuda-v2`在导入阶段exit1：`s2.evaluate`提前导入matplotlib/PIL，AlmaLinux9精简容器缺少`libjpeg.so.62`。未执行模型forward，未产生路径结果，故障不属于模型健康失败。执行命令和完整错误保留在输出根`audit/attached_inference/`，时间由正式父Pod allocation计入。

将`s2.evaluate`的matplotlib、PDF和`plot_evaluation`依赖移到实际绘图入口。纯统计函数、推理和配对bootstrap不再要求这些库；数值计算逻辑未修改。LCG108/Python3.12验证：路径回归6项通过（0.095秒），S2评价及S3统计回归16项通过（1.023秒）；将`matplotlib`及`PIL`显式设为不可导入的新Python进程仍可导入完整`functional_probe`。Python编译检查通过。

共享输入worker`s4-staged-confirmation-a`已正式提交bigbird21 cluster2202364，2并发，冻结a284928，优先MAE下游seed17/2026，再执行剩余全量矩阵。schedd即时负载、WRITE和EOS兼容性证据写入submission记录。原idle候选1575505及2199507使用`JobStatus==1`约束取消，不中断已经获配的worker；动作审计在`audit/staged_idle_replacement_20261113.json`。

新增完成的MAE3 seed42/100k fine-tuning程序耗时5046.655秒、best validation NLL1.20755884；统一checkpoint读回、配对区间及物理子道统计尚待下一次正式validation作业，暂不将该单值作为比较结论。

## 阶段性结论

CUDA诊断启动失败原因已定位并通过隔离导入检查修复。训练继续按原冻结配置运行；独立MAE下游seed收益仍未测量。

## 下一步计划

冻结修复源码后重启附加GPU诊断，检查完整10k cached Head一致性。尝试正式Kubeflow整卡flavor候选，与HTCondor共享claims，按实际调度结果减少排队时间。
