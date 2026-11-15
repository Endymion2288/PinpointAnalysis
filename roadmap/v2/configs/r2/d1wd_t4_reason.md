# D1WD T4：完整四类参考形成后的对应正则化控制

D1 T4完整endpoint和共同Train gap现已完成。Tune BA4/NLL4=.4237/1.337161873；共同4k Train诊断为.5455/1.012396882。条件三类Tune BA3/NLL3=.5373/1.015106150，低于D0的共同BA且NLL更差。τ在1% FPR下的效率从D0 .0511变为D1 endpoint .0438，best-screen为.0534。这里不能把同数据更长训练的失败归于τ不可识别。

初次只启动D1WD T3，是因为当时T4终点尚未形成。上述新证据触发相同的预定过拟合分流；现在补上T4，检验同一正则化干预下的三类/四类取舍和τ工作点。唯一学习因素仍为AdamW weight decay .01→.1。数据、seed42、BF16 encoder/FP32 head、batch8、1.2M训练曝光、0.9M非τ曝光、完整日程和架构均保持D1。系数与已冻结T3相同，没有根据本任务额外选择或扫描。

D1WD T4−D1 T4是单因素比较；与D0和D2的差异作为综合配方/数据选择单列。D1WD T3−conditional T4补齐共同任务配对。每组四个预定域、endpoint/best-screen分开评价，全部保留400次paired physical-block bootstrap、Train gap、同一五折readout、内部健康和实际资源。初始六模型主矩阵不变；完整交付成为8模型、16视图、128项配对。

资源：D0/D1的Kubeflow训练已于11:51 UTC完成，namespace GPU额度已释放。优先在该服务可用MIG上执行T4控制，与HTCondor上的T3控制并行；不等待整卡或同一型号。既有B1实际拟合门槛与共享staging方案复用，新的预检查仍独立测量准确updates与曝光。参考文献和固定系数依据沿用`d1wd_reason.md`。D3、1M、加宽或第二架构族未在此冻结。
