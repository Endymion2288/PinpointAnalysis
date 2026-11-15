# 阶段255：压力测试独立核算与MIG调度修复

**25条已完成条件路径通过2,268项独立核算，最大点估计差1.44×10⁻¹⁵；两个B1必要seed及B1诊断已在此前被排除的H100 1g.12gb MIG启动。**模型选择、样本和曝光规则保持冻结；本阶段补强统计读回并解除实际排队限制。

## 阶段目标

检验压力测试的逐条件指标、固定cut人数和模型间条件效应是否与保存概率一致；检查“允许MIG”的提交规则是否实际匹配当前池中的空闲资源。

## 工作内容

在`analysis.flavour.v2.r1.review`增加独立压力测试读回：重新计算class-mean BA/NLL、逐类recall、描述性FPR工作点、全Tune冻结cut的原始信号/背景/τ通过人数、条件差值和模型间条件交互点估计。独立路径不调用生产`Metrics`或`scalar_summary`。原始身份/标签与完成状态仍由完整review核对；置信区间的定义与配对设计另行验收。

新增`--stress-only`用于已有中期产物；最终`--require-full`会自动执行55条条件路径的独立核算，`accept`同时要求该产物通过。新增故障注入测试，验证错误通过人数及错误条件差值确实被检出。

查询完整Condor GPU ClassAds后，修正Capability缺失的已知H100 MIG匹配；根据实测主存调整R1控制器默认请求。仅修改尚未启动的三个作业资源属性，并保留修改前、held核对、修改后和release证据。

## 详细数据

### 独立压力测试核算

本次输入为Tune固定1,600事件、每类400，5个模型×nominal/noise/combined/remove_pixel/remove_tracker共25条路径。模型为HGB T3/T4、B4 T3/T4、B4-MAE T4；生产结果已有400个共享物理块draw。独立核算通过2,268项、失败0项，最大绝对差1.44329×10⁻¹⁵。

本次核算覆盖点估计和通过人数，未重新生成bootstrap；不将它描述为对训练方差或区间覆盖率的验证。新的测试用小型随机概率生成可核算结果，再故意把noise通过信号数加1、combined BA差改成0.05，两项均被检测；修正后的实际产物通过。

统计任务Condor1582906，1 CPU/3GB，allocation12 s，终态4、ExitCode0。产物为`R1/interim/stress_seed42/evaluation/independent_stress_review.json`。

```bash
python -m analysis.flavour.v2.r1.review \
  --root "$R1_ROOT/interim/stress_seed42" --stress-only
python -m unittest tests.test_flavour_v2_r1 tests.test_flavour_v2_r0
```

LCG108/Python3.12、PYTHONPATH=`src`、BLAS/OpenMP线程1；25项测试通过，耗时1.340 s。Cal-A/Cut-B/Confirm未读取。

### GPU广告与原有排队限制

当前池快照172个partitionable slots，包括A100、H100、T4、V100、H200及H100 1g.12gb/2g.24gb MIG；slot数包含不同slot广告，不直接等于独立物理卡数。快照与查询命令保存在`runtime/gpu_pool_stage255.json`。

实际MIG广告有`DeviceName`和`GlobalMemoryMb`，没有`Capability`。旧`Capability >= 7.5`对缺失字段不成立，排除了这些SM90分区。1g节点另有6 CPU/17,000 MB主存；旧16GiB请求经CERN `RebalanceMemCpu`变成6 CPU/18,000 MB，因此即使只修Capability，仍无法匹配。

实测aggregate RSS：B1 T3 11,738.27 MiB、B1无tracker FP32 12,647.21 MiB、B3 T4 12,220.94 MiB。R1 GPU默认改为15GiB；按实时CERN变换分配5 CPU/15,360 MB，覆盖已测峰值并能进入17GB节点。已有running作业不变；后续新作业采用新请求。

修订后的设备条件保留至少10,000 MB显存；已公布Capability时继续要求≥7.5；仅对缺失Capability且型号明确为NVIDIA H100 MIG的广告补充SM90识别。若用户明确要求能力>9.0，不使用这个fallback。V100仍受当前LCG108没有SM70内核的约束；实际运行先执行CUDA probe。

实际`countMatches`查询：新条件9个当前可用slot广告、旧条件3个，新条件中的H100 MIG6个、V1000个。查询原文和结果保存在`runtime/mig_matching_stage255.json`。这证明扩展了当前可匹配资源，不把slot数解释为保证可立即使用的物理GPU数。

### 三项pending作业的实际更新与启动

01:37 UTC附近核对1582901/1582904/1582905均Idle、零次启动，先hold并再次核对，再改`RequireGPUs`、`RequestCpus=5`、`RequestMemory=15360`，最后release。同一作业不产生新的并发writer；提交源快照保留原状，资源调整另存`runtime/pending_resource_edits_stage255.json`。

约01:38 UTC实际开始匹配，`-better-analyze`显示6个slot愿意运行；随后两个B1 seed的manifest均记录H100L-1-12C MIG1g.12gb、Capability[9,0]。seed17为1582904.0、seed2026为1582905.0；B1诊断1582901分配到同类MIG。它们均不要求整卡或相同设备型号。

依赖控制器先结束旧PID2015581，核对退出后启动唯一新PID2052979，exec session96303，日志`logs/finish_stage255.log`。它使用新的资源条件，继续等待真实任务完成；不会重复提交已经存在且成功提交的同名任务。

相关代码：`v2/submit.py`、`r1/finish.py`、`r1/review.py`、`r1/accept.py`；完整运行证据留results，源代码、测试和报告随本阶段提交。最新进度快照为`runtime/progress_stage255.json`。

## 阶段性结论

**事实：**已有压力测试的核心点估计、通过人数与保存概率一致；当前MIG实际发布方式与通用Capability筛选存在不匹配，修正后必要作业获得了资源并开始执行。

**研究判断：**资源兼容性需要核对实际广告与站点CPU/内存变换；“不指定GPU型号”本身不足以保证包含所有可用分区。不同MIG的训练速度、运行精度和曝光继续分别记录，异构小时不解释为FLOP等效。

## 下一步计划

保留已运行任务，完成B3 T3和最终排名、另一强族必要seed、全部fresh同计算及条件路径。最终统计后执行55路径独立核算，再完成图表、终态记账、R1验收与最终报告。新增B1 seed的泛化结果和最终成本尚未测量。
