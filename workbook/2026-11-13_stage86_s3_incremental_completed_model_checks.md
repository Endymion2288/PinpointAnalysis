# S3 首个完整重训模型与增量诊断

## 阶段目标

在其余GPU训练继续时，对每个已完成模型立即检查有效学习，减少等到整批结束才发现问题的延迟；保留冻结的20模型、4epochs、40,064updates及accepted输入。

## 工作内容

新增`live_repair`依次执行已完成模型的CPU读回、内部健康/学习检查和固定checkpoint分支诊断。最终流程仅复用配置、best epoch和checkpoint NLL一致且全部通过的健康证据；分支诊断还要求最终complete状态、64个事件、非interim、Head重建及presence一致。配置/epoch变化或失败检查均不复用。中途诊断写入`audit/path_interim/`，与最终验收分开。

新增`summarize_repair`在严格科学验收通过后生成版本化数值摘要和中文表格；目前未执行最终汇总，避免将进行中的训练写成完成。对每个模型保留实际训练曝光，不用80,000乘以epoch数代替采样器实测曝光。

为避免旧campaign supervisor与新backfill竞争，在Pod内暂停只负责调度的PID1；其已启动的screen child继续运行。worker watchdog从冻结20配置读取模型名，全部complete后恢复PID1，执行原读回与正常退出；确认任务统一由总并发上限8的backfill或先获配的Condor领取。状态保存在`audit/supervisor_coordination.json`。另启动终态捕获，每30秒读取Pod，只保存去除注入认证信息后的资源/状态；用于保留实际终态及设备时，避免Pod清理后退化为时间上界。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13`。

首个完成模型`s3r_multiscale_1mm_s17_e4`：

- 4epochs、40,064 applied updates、零AMP skips；实际曝光320,304。
- best epoch4；validation NLL=1.243261893932，macro AP=0.439397272351。
- 14/14 CPU checkpoint/optimizer/epoch-prediction读回通过。
- 401/401计划健康采样存在且通过；全部非空encoder采样梯度非零，独立best CPU FP32 backward连通。
- NLL−constant prior=−0.143032468152，400次128块配对bootstrap CI95=[−0.154687669943,−0.132123570247]。这是对NLL选定validation checkpoint的有效学习诊断，不是冻结test测量。
- 64-event分支诊断在13:07 UTC执行中，尚未发布最终路径结论。

13:06:35 UTC累计33/80个完整epoch，1/20模型完成，15/20模型已启动；新启动seed17 tracker。Kubeflow为H100 47GB MIG，HTCondor分别为整张H100和A10040GB，资源差异按实际元数据保留。13:06 UTC最新终态捕获仍为Running。

12:46:41 UTC已有14模型、2,750次健康采样全部通过。13:02 UTC几个seed42模型的第3轮完整validation：multiscale native NLL1.247830、macro AP0.437694；tracker1.229365/0.459632；no-direct1.264002/0.424945。它们是中途结果，不据此取代最终三seed比较。

另在首轮checkpoint执行32-event临时分支诊断。Native模型：置换全部regional/local/global/summary分支的probe NLL差分别+0.151943/+0.024794/+0.006307/+0.229227；tracker模型分别−0.005632/+0.111775/−0.000010/+0.233335。两个checkpoint的Head缓存重建一致；probe按occupancy选取，非总体样本。负差不说明分支在总体上有害，小global差不单独证明latent collapse。原数据在`audit/path_interim/`；最终检查使用64个事件重新计算。

Token census新增padding占用统计；同一个最高occupancy train事件：

| Representation | Active nodes | Local blocks | Active / padded node slots | Regional tokens | Global tokens | Direct local blocks |
|---|---:|---:|---:|---:|---:|---:|
| fixed native | 36,547 | 14,597 | 7.8242% | absent | 24 | absent |
| multiscale native | 36,547 | 14,597 | 7.8242% | 958 | 64 | 14,597 |
| multiscale 1 mm | 8,417 | 6,169 | 4.2638% | 430 | 64 | 6,169 |
| multiscale tracker | 36,549 | 14,599 | 7.8235% | 962 | 64 | 14,599 |
| multiscale region8 | 36,547 | 14,597 | 7.8242% | 3,690 | 64 | 14,597 |
| fresh ParT | 36,547 | 2,641 | 43.2447% | absent | 2,641 | absent |

细窗口保留完整原始节点，但固定32槽padding有实际计算开销；这些是同一事件的计数，非总体occupancy分位统计。`audit/screen_token_census/`共3个train occupancy ranks×8配置=24行。后续ragged/sparse优化应独立验证窗口定义、输出和梯度，不在本轮中途改变科学结构。

12:41:51 UTC资源快照：本轮上界2.650698906 device-hours，累计S3上界33.888967583，余额下界86.111032417，missing history=0。

针对性17项测试、0.232s通过，覆盖缓存不能复用interim/不同best checkpoint/失败健康报告、presence置换、谱诊断、并发名额、成本守恒和任务领取。`compileall`、`git diff --check`通过。实际首个完成模型的检查进一步验证了完整数据路径。

## 阶段性结论

已确认修复结构可以完成全量四轮训练并取得统计支持的有效学习；首个模型的全部计划内部健康及checkpoint读回通过。其余19模型尚未全部完成，因此尚无最终结构/粒度/tracker净增益结论；旧S3结论继续撤回。

## 下一步计划

继续完成20个模型，逐一执行增量诊断；出现故障则审查修复。全部结束后完成四分类配对、七条件域、工作点、英文图与事件显示，保存终态费用，形成stage87科学报告和最终roadmap摘要。Calibration/test保持冻结。
