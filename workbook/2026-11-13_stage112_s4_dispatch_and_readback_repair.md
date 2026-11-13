# S4：配置分发与checkpoint读回修复

## 阶段目标

修复正式mixed campaign对旧监督配置的执行及读回分发，完成等预算监督和真实checkpoint检查。

## 实测问题与修复

Condor cluster2198296获得24GB H100 MIG后，三项matched-supervised在首个训练步骤前因`KeyError: program`失败，总allocation63秒。planner继承的旧Scratch配置没有显式program，GPU worker却强制读取。未产生训练checkpoint，不能作方法结果。生成器现显式写`program=train`，worker与validation统一按`objective=classification`解析旧配置；失败claims经终态核实后移入audit并释放，新提交不复用故障writer。

同一差异导致第一份CPU进度评价执行了四分类统计，却跳过旧Scratch的checkpoint readback，health summary为空。原先独立Scratch checkpoint检查仍保留，但第一份新进度产物不能称统一readback已完成。新CPU v2将补执行所有已完成分类模型的14项合同和内部health，再给出完整统计。

## 详细数据

恢复后的batch64 EMA pilot完成1024步，固定probe的teacher effective rank **4.24633524**、leading variance fraction **0.36441322**；student projection rank **6.22351613**、leading fraction **0.20381544**。全部integrity/internal和6项measured-learning checks通过，允许完整EMA训练；分类收益未测量。

24GB H100 MIG资源已由HTCondor获取，channel Rel补充wave cluster1574116于19:49开始运行两项；这也证明compact flavor扩展缩短了本轮等待。Kubeflow4并发继续，host peak约35GiB/58.67GiB。

## 验证与下一步

新增回归直接运行缺省program的worker分发，以及缺省program的已完成模型readback/health汇总。修复后25项S4测试全部通过（49.088秒）。另将图中单seed的SD记为未定义、不画误差棒，使用英文物理指标名称。重新提交matched-supervised与CPU v2，继续完成下游、Rel语义事件置换、最终配对图表和成本。

完整1k MAE3 seed42已完成，best epoch1：macro AP0.36179163、τ AP0.28432114、NLL1.40226192；Scratch对应0.29720868/0.25953440/1.51618843。仅为单seed初步比较，完整paired interval、独立seeds与matched-supervised尚待。

重提wave同时带上已通过pilot的完整EMA及MAE/Rel继续训练，复用原冻结任务配置与互斥claims，先获配的后端领取；分类matrix与资源预算不变。这样在HTCondor获得紧凑MIG时，也能提前推进关键预训练依赖。
