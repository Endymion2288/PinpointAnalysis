# R2 设计补充：分开数据、曝光与学习率轨迹

## 阶段目标

根据stage263已完成的B1/B4三seed结果，明确R2的单因素比较和实际预算。仅修改后续设计，R1继续按原冻结配置执行，未启动R2训练。

## 工作内容

重新核对`r1/train.py`与`SpatialBatchSampler`。R1四pass主任务按`updates/(total_steps−1)`定义学习率进度；等时控制按已经消耗的操作wall、扣除预计最终推理时间后定义进度。两者都使用warm-up及cosine，但进度变量不同。因此stage263同时改变了曝光次数和优化轨迹，不能单独回答重复曝光的因果增益。

联网读取M. Li、E. Yumer、D. Ramanan的 **Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints**，ICLR2020，arXiv:1905.04753v4；核对摘要及§3.1–3.2。论文以迭代预算定义进度`t/T`，研究按预算重新参数化学习率，并将cosine列为budget-aware schedule。其linear结果来自图像/视频等任务，不据此宣称PINPOINT应立即替换cosine。

将可执行的设计约束补入`roadmap/v2/04_execution_plan.md`和`experiment_plan.json`，并在文献表增加M6。R2 runner及其真实效果尚未实现/测量。

## 详细数据与设计

stage263的B1 compute T3−main T3为BA3 −.19444 pp [−.42774,+.03423]、NLL3 +.005171397；B4对应+.28222 pp [.08821,.46817]、−.002685275。它们支持分开控制优化配方；不支持宣布更多曝光普遍有害或有益。

首批仍只冻结一个族，当前优先B1；沿用相同初始化方式、输入、模型、loss、batch size、sampler、selected LR、AdamW与正则化。D0/D1/D2都从fresh初始化，学习率按其预定总updates定义5% warm-up及cosine，floor保留原5%。D1不能从已经退火到floor的D0尾部直接接着训练。wall只用于资源记账与工程deadline，不用于这一组数据/曝光比较的学习率进度。

| Contrast | Unique supervised rows T3 / T4 | Total label exposures T3 / T4 | Interpretation |
| --- | ---: | ---: | --- |
| D0, pool100k ×4 passes | 75k / 100k | .3M / .4M | 可复用的同数值配方基准；否则补新D0 |
| D1, pool100k ×12 passes | 75k / 100k | .9M / 1.2M | 同数据、扩展预算及其完整schedule |
| D2, pool300k ×4 passes | 225k / 300k | .9M / 1.2M | 与D1同曝光，更多unique事件 |

`SpatialBatchSampler`在mixed-shard window内形成batch，每个window保留尾批。故D1/D2相同总事件曝光不保证完全相同更新步数；U必须从冻结清单与sampler实算，并同时报告batch-size分布、总updates、LR轨迹及累计`sum(lr)`。首批将D2−D1称为**同曝光比较**；需要精确fixed-update控制时另行预定停止步数与实际曝光，不能同时声称固定passes、固定曝光和精确固定updates都已成立。当前300k的U、I/O、吞吐与最终wall均未测量。

每个模型每75k个非τ Train曝光进行一次固定Tune screen评价，T4不把τ曝光计入这个横轴；最终在完整40k Tune评价。中间窗口使用同一4k screen，终点使用固定预算checkpoint；best-screen checkpoint可另报，不能与固定终点混成一项比较。完整Tune的checkpoint/超参数选择仍按开发结果处理。

每个窗口保留train/Tune NLL、共同BA3、逐类argmax recall、AP/AUROC、原始概率与低occupancy表现；τ工作点及τ混入在完整Tune终点评价。三个连续窗口NLL变化小于.001且主指标无实际提高只触发plateau检查，不能据此认定物理信息上限。最终R1、预算、输入和小样本真实拟合通过后才启动首批。

R1的B1 seed42为T4/FP32，其余B1 seed为H100 MIG/BF16。新配对固定实际精度和训练设置，取得什么兼容GPU就使用什么；旧D0数值配方不一致时补一个新D0。不同型号不妨碍执行，但不将设备小时当FLOP等效。

按R1实测外推，B1一族D1/D2四项约10.087异构device-hours，原先两个族约23.493 h且尚未含新D0及转换。继续分批，完成首族后按未回答的问题与实际剩余预算决定第二族，不扩大成自动全矩阵。linear/cosine之类schedule对照需另立小型合同和匹配HPO额度，首批不同时改变它们。

## 验证与产物

论文abstract及PDF均HTTP200，PDF为724,324 bytes；真实来源、作者、版本、访问时间与文件路径位于`R1/audit/literature_stage264/retrieval.json`。PDF及文本只保存在results目录，不进入Git。源码证据为`src/analysis/flavour/v2/r1/train.py`、`src/analysis/flavour/s2/dataset.py`。设计字段经JSON解析，表中两种任务的unique/exposure关系经独立算术核对，修改后的Markdown链接与`git diff --check`通过。

本阶段没有新增GPU任务，没有读取Cal-A/Cut-B/Confirm payload。R1仍在执行，完整41模型与最终accept尚未完成。

## 阶段性结论

**事实：**R1等时对照同时改变曝光和schedule；当前sampler下相同曝光不等于完全相同updates。已有三seed结果未证明B1从更多重复曝光受益。

**待检验：**在明确预算与相同数值配方下，更多unique事件是否优于更多重复曝光，以及完整schedule/容量是否仍限制性能。论文提供实验设计依据，PINPOINT收益尚未测量。

## 下一步计划

完成R1剩余B2/B3模型、完整评价与验收。随后按本设计冻结R2首族及实算U/吞吐/成本，再执行D0/D1/D2；不提前启动R2。
