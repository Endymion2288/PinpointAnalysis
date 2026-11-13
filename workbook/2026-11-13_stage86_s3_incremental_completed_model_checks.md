# S3 首个完整重训模型与增量诊断

## 阶段目标

在其余GPU训练继续时，对每个已完成模型立即检查有效学习，减少等到整批结束才发现问题的延迟；保留冻结的20模型、4epochs、40,064updates及accepted输入。

## 工作内容

新增`live_repair`依次执行已完成模型的CPU读回、内部健康/学习检查和固定checkpoint分支诊断。最终流程仅复用配置、best epoch和checkpoint NLL一致且全部通过的健康证据；分支诊断还要求最终complete状态、64个事件、非interim、Head重建及presence一致。配置/epoch变化或失败检查均不复用。中途诊断写入`audit/path_interim/`，与最终验收分开。

JSON证据发布改为各writer独立临时文件后原子替换，避免原共享`.json.tmp`在读回observer重叠时被另一个writer抢先rename；不改变内容或训练计算。并发8个writer×8次发布的回归检查验证中文及完整payload、无残留临时文件。

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

JSON并发修复后18项相关测试、0.125s通过；13:16:58 UTC首个fixed native确认模型也完成全部CPU读回、健康及分支诊断，累计2个完整模型的增量检查通过。

13:44 UTC增量：10/20个模型、52/80个epoch已完整结束，全部20项已启动。8个完成模型已通过CPU读回、健康及分支诊断，其余依次检查。新增`plot_repair_paths`把分支置换的NLL/预测变化、事件间分支变化和token有效秩同时绘图；最终数值摘要保留逐路径梯度的min/median/max与精确零计数、采样group加权的token谱摘要。图不设置未经验证的rank失败阈值，也不把置换当重训消融。

实际三模型图输出到`audit/fitted_path_diagnostics_render.pdf`，第二页已转PNG人工检查，英文标题/坐标/说明可读。Native seed17在64-event probe中的global分支事件间RMS=0.0427503，global置换ΔNLL=−0.0091165，latent-read训练采样梯度平方范数median=0.0031144；局部和regional置换ΔNLL分别+0.2514955/+0.2120467。Fixed seed17相应global变化RMS仅0.0001833，global置换ΔNLL=−0.0000032。事实是部分拟合模型对global分支依赖弱；非零梯度、低rank、旁路冗余需要联合解释，不能将全模型有效学习写成global表示已充分利用。最终报告须保留这一限制。

新增图的三模型实际渲染、`compileall`与`git diff --check`通过；路径/科学验收相关9项既有测试全部通过（0.056s+0.009s）。只重启等待中的CPU finalizer以接入新图，GPU训练计算与冻结配置未改。

13:50 UTC对两个Condor allocation各连续采样21次、间隔1s：H100利用率均值80.00%（8–100%），设备显存16,889 MiB；A100均值42.14%（5–71%），设备显存18,596 MiB。当前均同卡3个训练；这是短时间NVML样本，不外推整批busy time。先前SSH带AssignedGPUs标识的探测未枚举设备；本次不指定该标识的实际`nvidia-smi`成功，证据在`audit/gpu_utilization_samples_2026-11-13T1350Z.json`。MIG利用率仍未测量，不从其他卡代推。

最终流程增加显式终态等待：所有fit/诊断完成后每30s刷新allocation，直到history完整、各分配真实终态，才生成依赖成本的评价，避免observer与supervisor退出之间的竞态留下过时资源字段。新增回归覆盖Running、缺失历史、allocation不完整及空分配列表；科学验收6项测试0.012s通过。等待中的CPU finalizer已更新，训练writer保持运行。

资源审查还定位到旧`parallel`在自己的writer全部结束后等待远端已领取的配置，从而可能保留闲置GPU。现改为默认回收本worker，只有显式`--wait-for-peers`才等待远端；campaign wrapper原本另有完整矩阵等待。当前运行的旧A100进程无法加载该更新，已启动观察器：仅在其process record全部exit0/reaped、全部owned manifest complete后查询并移除仍Running的allocation，证据保存`audit/idle_peer_reservation_release.json`。不会终止自身训练。共享资源9项测试0.024s通过，含远端仍Running但本worker无writer时立即释放的回归。

14:17 UTC把最慢的fresh ParT seed2026由MIG接续到已有Condor整张H100。迁移前CPU读取已提交checkpoint：epoch1、10,016updates，模型/optimizer/scheduler/RNG和配置完整；先验证目标CUDA，再终止源writer及其3个子进程，确认全部退出且旧supervisor已reap为−15，归档claim后才发布handoff。目标manifest记录`resumed_from_epoch=1`与`prior_gpu_name=H100 NVL MIG 3g.47gb`，剩余30,048updates未缩短。源端最后progress已观察750个未提交更新，这部分丢弃但资源成本仍保留。负退出是主动迁移，不作模型失败结论。

目标H100只暂停原调度supervisor，原训练child继续；迁移wrapper在新writer结束后恢复supervisor，避免其原任务先完成导致allocation关闭。目标父allocation合并计费，迁移前MIG区间另计，不产生同时writer。证据为`audit/migration_part2026/{plan,source_released,destination_ready,supervisor_paused}.json`及source manifest/health/progress归档；迁移终态和最终读回尚待实测。两次目的端启动准备失败（Python模块调用及进程匹配包含init wrapper）均发生在暂停调度或停止源writer之前，修正后目标ready；它们不改科学模型且处于同一已计费allocation。

针对global弱依赖补读NeurIPS2021 Gradient Starvation正式全文及CVPR2022/TPAMI2024 OPM/OGM，正式链接和后续可检验假设已加入[修复路线图](../roadmap/10_s3_repair.md)。这些论文支持审查分支间优化竞争，不构成本模型已证实gradient starvation的证据。当前冻结矩阵不加入新的分支dropout/gradient modulation。

## 阶段性结论

已确认修复结构可以完成全量四轮训练并取得统计支持的有效学习；首个模型的全部计划内部健康及checkpoint读回通过。其余19模型尚未全部完成，因此尚无最终结构/粒度/tracker净增益结论；旧S3结论继续撤回。

## 下一步计划

继续完成20个模型，逐一执行增量诊断；出现故障则审查修复。全部结束后完成四分类配对、七条件域、工作点、英文图与事件显示，保存终态费用，形成stage87科学报告和最终roadmap摘要。Calibration/test保持冻结。
