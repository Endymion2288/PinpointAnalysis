# S3 双后端调度与有效学习诊断

## 阶段目标

让冻结确认任务使用最先获配的GPU，同时防止两个后端写同一实验；为完整结果补充可解释的内部健康与学习证据。

## 工作内容

1. `parallel`对repair任务使用原子目录领取，`claims/<experiment>/state.json`记录writer、状态和终态。另一个获配后端遇到已有claim时跳过同一fit，等待其结果；失败claim需要审查，不会机械重启。`campaign_repair`先screen再确认，每波CPU读回及健康检查后继续。
2. 科学配置继续固定模型、sample/split、输入、4epochs/40,064updates。硬件允许H100（含47GB MIG）和A10040GB的原生BF16，Torch2.7.1；逐模型保留实际卡型，跨设备分配成本不声称模型加速。硬件允许列表在确认任务获配前记录，未按validation改变模型。
3. `inspect_repair`增加NLL相对constant类别先验的400次accepted物理块bootstrap诊断，完整validation的改善区间支持学习才通过对应项；单纯有限参数、非零浮点波动和checkpoint读回均不足以通过。区间条件于validation选定的checkpoint，不能称冻结test结果。
4. 新增六个三seed结构的overview与九个三seed对照的raw/common-support区间图；单seed点与training seed sample SD、物理事件CI分别标注。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13`。

八个编译版本2配置的最大占用反向压力测试全部finite且输入梯度非零：

| Configuration | Peak allocated MiB | Input gradient squared norm |
|---|---:|---:|
| fixed native | 761.882 | 699.156 |
| fixed 1 mm | 739.127 | 683.642 |
| multiscale native | 762.285 | 925.367 |
| multiscale 1 mm | 739.049 | 3130.447 |
| multiscale tracker | 762.294 | 963.906 |
| multiscale no direct | 760.719 | 1119.605 |
| multiscale region8 | 761.592 | 855.213 |
| fresh ParT | 1023.725 | 1.03509 |

八个pilot也全部通过。正式20配置入口`configs/s3_repair_all.json`与screen+confirmation逐项一致。Kubeflow job `pinpoint-s3r-campaign-compiled-v2`在11:32 UTC实际启动screen，单个H10047GB MIG内8个独立训练；不存在第二个GPU Pod。

确认任务同时提交HTCondor，`bigbird20.cern.ch`在提交时通过EOS policy与WRITE授权且空闲队列为0；先查询全部schedd，记录比较后选择。任务1574083对应seed17的6个模型，1574084对应seed2026的6个模型，各GPU并发3个fit。提交文件位于AFS `~/condor/Pinpoint/flavour_s3/repair_confirm_s17_v2/`与`repair_confirm_s2026_v2/`，日志EOS；当前两者仍Idle。`better-analyze`显示0个愿意运行的匹配slot；表面的Unclaimed/Idle不等于可调度。一个空闲A100已过staged-drain ShutdownTime；空闲H200有独立accounting-group限制，未据此声称有可用资源。

已移除旧Idle任务1574068对应的AFS小脚本目录`repair_screen_controls_v1`，清理1,556bytes；完整提交文本保留在EOS证据，记录`audit/afs_cleanup.json`。当前使用的两个确认任务脚本保留。

补读并缓存正式ICML论文：Zhai et al. 2023 *Stabilizing Transformer Training by Preventing Attention Entropy Collapse*、Dong et al. 2021 *Attention is not all you need: pure attention loses rank doubly exponentially with depth*；以及2026-01 v2 preprint *mHC: Manifold-Constrained Hyper-Connections*。正式URL、迁移范围和未采用组件写入`roadmap/10_s3_repair.md`。理论的纯attention rank结论不直接适用于有FF/残差的当前网络，不凭cosine高宣布collapse。

当前`audit/live_health_checkpoint.json`记录8个模型的最近采样，均通过；更新数2,501–6,401。这里仅为训练中检查，完整validation指标尚未发布，不据此判断native/tracker或架构增益。H100实际吞吐随结构约20–51events/s；ParT全局计算较慢，保留同事件/更新预算对照。

11:46:47 UTC成本：本轮0.946276632 device-hours、累计S3上界32.184545309、余额下界87.815454691，missing history=0。旧S3上界31.238268678并入，S2单列。

最新针对性验证20项、4.510s通过，包含原子claim避免重复fit、Condor/Kubeflow共享成本守恒、BF16硬件允许列表、健康门槛和训练resume。前一阶段35项相关模型/训练/决策测试通过。`compileall`和`git diff --check`通过。

## 阶段性结论

事实：八个修复结构具备真实先导拟合与完整最大占用反向能力，正式screen正在运行，确认任务已进入另一个后端队列。当前尚无完整重训的科学结果；所有旧S3方法判断继续撤回。

## 下一步计划

持续观察内部健康及第一轮完整validation，完成20个配置的拟合与检查。任何梯度断开、常数输出或明显学习失败都触发实现/架构/训练审查；健康后才能做四分类比较。全部结果和图表复核完成后更新`roadmap/results/s3_repair_summary.json`为实际科学结果。
