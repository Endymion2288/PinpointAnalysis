# R1：μ工作点、决策取舍与剩余时间迁移

## 阶段目标

检查B1的μ argmax召回损失是否也表现为μ分数辨别能力下降；用既有三seed统计完善R2诊断设计。同时按实际剩余完成时间执行候选设备迁移，保留全部旧成本。没有启动R2训练。

## 工作内容

新增`roadmap/tools/export_r1_decision_review.py`，按原evaluation config精确选择B0/B1/B4、T3/T4、seeds17/42/2026的18个main模型，导出6组μ指标和4项mean-seed配对。仅重组已完成的R1 Tune统计，没有拟合新的模型或阈值。产物为[r1_muon_decision_review.json](../roadmap/v2/r1_muon_decision_review.json)。

更新模型设计，分开检查互斥多类argmax与单类score cuts；补充R2分批预算，先在完成R1后冻结一族的D1/D2，再按问题与预算安排第二族。

## 详细数据

### μ的多类决策与one-versus-rest表现

全部指标在相同30k非τ Tune事件上，使用均匀三类benchmark；T4结果先在非τ上条件化。区间为既有400次配对物理块pointwise开发区间，条件于三个已拟合seed及实际设备/精度，不代表训练种子总体。

| B1 minus B4 | Mu argmax recall | Mu efficiency at 1% FPR | Mu efficiency at 0.1% FPR | Mu efficiency at 90% benchmark purity |
| --- | --- | --- | --- | --- |
| T3 | −2.410 pp [−3.209,−1.690] | +4.850 pp [+4.057,+5.823] | +2.443 pp [+1.507,+3.581] | +8.503 pp [+6.486,+10.052] |
| T4 conditional | −1.853 pp [−2.611,−1.063] | +5.383 pp [+4.560,+6.297] | +2.463 pp [+1.330,+3.585] | +8.720 pp [+7.431,+10.288] |

T3 μ的1% FPR效率差逐seed为+5.36/+4.32/+4.87 pp，argmax recall差为+1.37/−6.91/−1.69 pp；后者sample SD为4.187 pp，包含实际执行差异。μ AP平均B1=.648318、B4=.599953；AUROC为.759086/.727759。B1相对HGB的μ1%效率增益为+4.110 pp [3.165,4.977]；0.1%增益仅+0.543 pp [−1.016,1.625]，尾部优势尚不明确。

R1预设的B1 T3对B4逐类recall非劣仍未通过；本阶段没有更改该门槛。各类fixed-cut选择不是互斥三类输出，不能用它们宣称已经修复argmax；后续需预定冲突/拒判、coverage及全域质量。当前证据支持把训练schedule、类间offset/决策和表示机制分开诊断，尚不能认定为不可逆投影信息损失。

### B2 seed17迁移

新候选1582928在H100 NVL对完整Train相同前512 batches测得64.545255 events/s，activation finite。04:08:27 UTC旧2g的T4已曝光101,896次、训练2,533.553 s；以三项操作约1.1M曝光估算，继续约24,816.963 s，fresh加600 s余量约17,642.306 s，预计减少7,174.656 s（1.993 h），超过本次至少1 h判定值。这是线性资源推算，最终wall尚未测量。

随后旧1582921执行`condor_rm`，history状态3，allocation **2,829 s**，event log有eviction与abort；确认队列已无该任务后，将未完成目录归档到`R1/failed_attempts/final_b2_t4_s17_mig2g_stage262`。04:09:17 UTC写入与新allocation绑定的release；新T4 manifest已实际Running、allocation1582928.0。没有同时写同一模型目录。

证据为`R1/audit/b2_s17_handoff_stage262.json`及`R1/runtime/b2_s17_remaining_time_stage261.release.json`。此前1g尝试、2g尝试和所有候选profile全部记账。seed2026仍在2g的1582922有效训练，其新候选1582929排队；只有在新probe与当时余量满足要求时才迁移。

04:12:58 UTC总资源为**36.650785 allocation device-hours**、261.921988 allocated CPU core-hours，尚有5项GPU训练和1项Idle候选；终态成本未关闭。异构小时不作FLOP等效。Kubeflow只读取旧终态缓存，没有再次触发认证。

### R2成本设计

R1四pass实测：B1 seed17的T3/T4 wall分别为2,646.968/3,405.090 s（1g MIG）；B2 seed42为3,554.084/4,489.690 s（H100 NVL）。按每族D1/D2各T3/T4、总曝光各为D0的三倍估算，B1需10.087 h、B2需13.406 h，异构分配时间合计23.493 h，已超R2原18 h规划，且未含新D0、300k转换与失败。

因此仅保留分批设计：完整R1后先冻结当前有收益的一族，优先B1；完成D1/D2和必要数值配方对齐D0后，再判断第二族哪个对照值得运行。该估计未测量300k的实际I/O或wall，不作为等待整卡的理由，也未启动任何R2作业。

## 验证命令与产物

```bash
python roadmap/tools/export_r1_decision_review.py --root "$R1_ROOT" \
  --output roadmap/v2/r1_muon_decision_review.json
git diff --check
```

独立按6组原始mean/per-seed值复算4项配对的μ差，共80项值核对通过；R2两个成本外推与实测manifest读回一致；相关27个本地链接和导出脚本语法检查通过。此前stage261新增1,185项独立人数/工作点读回继续有效。本阶段不访问Cal-A/Cut-B/Confirm。

已建立24模型56配对的`R1/interim/b1_three_seed_compute/evaluation_config.json`视图，包含B1/B4全部三seed主任务、compute及HGB。两个B1 compute尚未全部complete；该视图尚未运行评价。完整41模型仍由唯一PID2052979控制器管理。

## 阶段性结论

**事实：**B1的μ分数在当前开发工作点优于B4，但多类argmax有逐类取舍及较大seed/runtime波动。B2 seed17已在实测更快的新资源启动，旧writer终态与归档可核对。

**待检验：**交叉拟合的决策诊断或明确update/exposure schedule能否改善互斥三类质量；更快资源的最终全程节省量。不能仅凭当前μ fixed-cut增益推断argmax已修复或信息上限。

## 下一步计划

收齐B1三seed compute，按已冻结视图完成同allocation计算比较与独立读回。继续B2剩余时间判断和训练，B3 compute继续运行。完整41模型/133配对、全部图表、独立复核和终态资源关闭后，才执行最终accept与汇总。R2保持设计状态。
