# 阶段59：完整四分类诊断与先获配先运行

## 阶段目标

补齐S3条件域、完整四类工作点与最终输出门槛，并落实用户最新调度要求：“Kubeflow和HTCondor哪边能排上队就用哪个，尽量减少排队等待时间”。S3科学训练仍在进行中。

## 工作内容

新增`conditional.py`，在同一full-validation NLL checkpoint上评价pixel-centered、large scintillator、detector-active、all-empty、pixel≥1000、tracker-active/inactive七个条件域。每域保留完整四分类raw/common-support配对和三seed均值的物理block区间，报告各类1%/0.1%背景接受率、50%/80%/90% purity、domain阈值和固定inclusive阈值效率。全域包含空响应原分母；相同purity是在各域观察到的benchmark prior下定义。曲线绘图抽样，工作点使用全部tied-score thresholds计算，低计数尾部如实标记。

`MetricCache`增加四类两个固定FPR效率的配对区间。`accept.py`分别输出研究完成门槛和方法晋级，不能用τ提升替代macro质量/逐类AP/NLL门槛。效率路径必须有实际allocation，且质量区间处于预定近等效范围。所有17个screen、去重确认、14项CPU读回、全配对、128 blocks/400 replicates、条件域、英文图和native错误显示、终态计费及120 GPUh组织上限同时核对。

复用S2 native display绘图接口，新增可选groups路径和四类score标题；原S2默认行为保持。新增S3入口`analysis.flavour.s3.display`。

调度不再固定后端优先级。`backend --dispatch`在Kubeflow申请可接任务的整卡H100 slot，其worker实际CUDA/BF16环境通过后才写ready。`dispatch watch`每15秒观察ready；只有slot实际可运行，才从已记录的Condor队列选择idle任务。schedd上的hold约束明确包含`JobStatus == 1`，因此并发启动的running任务不被交接。确认held并移除原副本后才发布唯一assignment，worker执行同一冻结配置；Running作业保持原地训练。slot准备好后至多等待300 s，无安全可交接任务则结束。所有startup/失败/交接占用计入同一S3预算。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

新增统计/工作点/决策检查9项通过，0.146 s，`unit_diagnostics_decisions.log`；交接检查3项通过，0.014 s，`unit_dispatch.log`。交接测试包含“无ready保持队列不动”“任务已running或在hold前并发启动时不交接”“先hold/remove再发布assignment”。此处是mock schedd逻辑验证，不冒充Kubeflow运行证据。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s3_metrics tests.test_flavour_s3_decisions tests.test_flavour_s3_dispatch
```

两个只读已有S2基线的完整条件域诊断真实退出0：7个域、64行四类工作点、40行四类epoch/domain learning结果，400次共享block bootstrap，产物`conditional_baseline/summary.json`、`sampled_curves.parquet`和英文`four_class_domain_curves.pdf`。运行参数和日志在`conditional_baseline.log`；该对照用于执行验证，不用作S3 screen选择。Native display在既有ParT基线上真实绘制2页退出0，产物`evaluation_baseline/event_displays/part_native_joint_s42_e4_v2/`，标题显示四类score。

正式16项Condor队列1573835已提交，主task5先提交Kubeflow `pinpoint-s3-screen-task5-v1`。原Kubeflow probe约26分钟未获配，科学Job另约6分钟仍Pending无node；随后确认无writer、缩短deadline并观测真实DeadlineExceeded终态，再单独提交Condor1573837。原两项Kubeflow从未Scheduled，GPU占用0；证据`kubeflow/probe_release.json`、`task5_migration_release.json`及计费快照保留。

写作时Condor已有4项Running，最早配置首epoch已到12,250/20,016 steps，另三项为10,250/5,500/3,000，均0 AMP skips。速度约95.4/76.8/48.2/71.3 events/s。尚无完整epoch validation或S3分类收益结论。`campaign_accounting.json`在14:21:50 UTC时组织上界0.2407164746 GPUh、missing history=0；这是该时间快照，不当作最终训练成本。

## 阶段性结论

事实：完整四类条件诊断和native绘图接口已在真实预测/响应上运行；训练仍正常推进。两个后端的实际观察支持取消固定优先级，改为实际ready后交接idle任务。交接的真实Kubeflow GPU执行尚未测量。

假设：S3结构收益、粒度interaction和四类资源优势仍待全部screen及三seed确认。当前训练进度和基线诊断均不能替代该结论。

## 下一步计划

提交Kubeflow dispatch slot并运行ready watcher，与原Condor队列并行等待；哪边实际可运行就使用哪边，保持单writer。完成首完整epoch与全部screen，逐项CPU读回、全四类评价后冻结三seed确认。最终完成条件域/错误显示/成本与strict output gate，中文报告并提交。calibration/test冻结，S4–S6未执行。
