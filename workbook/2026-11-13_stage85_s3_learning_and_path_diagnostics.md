# S3 首轮有效学习与分支诊断

## 阶段目标

检查修复后的模型是否在完整validation上学习，并为高query相似度、latent弱梯度增加实际路径使用诊断。科学矩阵、accepted样本、detector-proxy和更新预算沿用冻结配置。

## 工作内容

新增`diagnose_repair_paths`：在每类16个、覆盖全部非空occupancy范围的固定validation事件上，缓存FP32 Head输入，并在相同模态presence事件之间置换regional、global latent、local和response-summary分支。Head输入重建必须与原forward一致；保留donor事件索引、预测变化和探针NLL差。另测两个Head路径分别移除的影响，以及module/latent token的未中心化奇异谱entropy rank。它们描述固定checkpoint的路径依赖，非重训消融，也不是总体分类指标；低rank或小置换效应本身均不否定训练。完整诊断尚未测量，待fit完成后执行。

最终入口对17个revision 2模型执行此诊断，并在验收中要求诊断完整、presence保持和Head重建一致。3个fresh ParT保留原结构，使用共同的内部健康及有效学习检查。

调度新增同一allocation下跨supervisor的并发上限和自动补位：worker本地kernel文件锁保护名额读取与启动，EOS保存进程区间；进程退出自动释放锁。只有空闲名额出现才领取确认任务，HTCondor仍可先领取。repair child失败后停止启动该supervisor的剩余队列，等已有child结束再报错，供审查。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13`。

12:04:58 UTC，6个模型已完成第一轮完整10,000-event validation：

| Model | Train NLL | Validation NLL | Balanced accuracy | Distinct score vectors |
|---|---:|---:|---:|---:|
| fixed 1 mm | 1.339820 | 1.299809 | 0.3761 | 9,561 |
| multiscale 1 mm | 1.341199 | 1.279591 | 0.3849 | 9,561 |
| multiscale native | 1.344952 | 1.285525 | 0.3796 | 9,561 |
| multiscale no direct | 1.342884 | 1.293324 | 0.3783 | 9,561 |
| multiscale region8 | 1.344821 | 1.283614 | 0.3814 | 9,561 |
| multiscale tracker | 1.322083 | 1.260383 | 0.4136 | 9,719 |

均匀四分类NLL为1.386294。这些首轮数值支持模型开始学习，尚不作结构、粒度或tracker净增益结论。机器可读快照：`audit/first_epoch_validation_snapshot.json`。fixed native和fresh ParT的首轮validation在上述快照时尚未完成。

11:57 UTC，8个训练占用35.55 GB / 63.00 GB Pod内存，22个CPU核；曾尝试提前启动4个确认训练。之后确认每个fit的train与validation会分别保留一个worker，使峰值驻留内存高于训练阶段的估计；新增fit还发生CUDA allocator NVML assertion / CUBLAS allocation failure。原8个训练无Traceback，继续运行；CPU cgroup的OOM/kill计数为0，不能将CUDA错误直接表述为已观察到的CPU OOM。

新增supervisor在首批4个任务失败后已启动第二批4个，12:01 UTC停止新增supervisor与其全部child，12:02:32 UTC确认writer已退出。4个exit=1由atomic claim终态保留；另4个exit code未观察，用确认进程退出时刻作为区间上界。失败尝试的实验目录与claims全部移入`audit/concurrency12_runtime_attempt/`，未纳入科学比较；进程费用作为独立overhead并入同一MIG allocation，不重复计费。之后确认任务从头执行。

调度回到总并发8；新的补位supervisor将复用结束实验的名额，避免等待最慢screen才开始所有确认训练。两组Condor确认作业仍Idle。NVIDIA在MIG下未提供可读利用率，未报告推测的百分比。

12:00:01 UTC成本快照：修复1.166805174 device-hours，累计S3上界32.405073852，余额下界87.594926148，missing history=0；这是该时刻的运行估计。后续终态费用继续刷新。

针对性验证16项、0.069 s通过，包含同一Pod补位等待、双supervisor成本守恒、presence置换、奇异谱诊断、故障学习验收和任务去重。此前模型/路径/决定测试19项、7.726 s通过。`compileall`和`git diff --check`通过。

## 阶段性结论

事实：已有6个修复模型的首轮完整validation优于常数均匀先验，预测非常数；原8个screen继续训练。新增并发的失败属于资源/运行尝试，已审查归档，不能用其判断模型能力。高query相似度及latent路径贡献仍待完成固定checkpoint诊断和重训对照。

## 下一步计划

以总并发8在现有MIG内自动补位，继续竞争HTCondor最先获配资源。完成全部20个训练、内部健康、分支诊断及四分类/条件域评价，复核英文图，再更新科学结论。Calibration/test继续冻结，S4–S6未执行。
