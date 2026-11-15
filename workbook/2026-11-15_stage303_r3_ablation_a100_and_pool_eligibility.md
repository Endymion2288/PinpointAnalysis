# R3：消融A100恢复与实际GPU池准入

## 阶段目标

继续利用能减少完成时间的兼容GPU，保留正式训练配方与有效预算。资源选择依据真实吞吐、剩余训练、准备/验证/恢复成本和实时准入条件。

## 工作内容

查询Condor实时GPU广告后提交两个有限Train-only探针，原三个writer继续运行。均不限定DeviceName，要求当前BF16环境的SM80+、至少30,000MiB显存，并接受backfill；12CPU/36GB RAM、80GB scratch。额外显存要求用于寻找不同容量的候选资源，原有MIG训练继续使用。

消融探针获得A100，通过两段实测后按剩余完成时间决定迁移；先结束原writer、核对queue消失和history/event终态，才release新writer。双模型探针没有获配，通过实际Start策略识别不能使用的空闲广告后结束零启动请求。Kubeflow现有双模型保持。

## 详细数据

根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。

两次提交前均检查schedd负载、WRITE和EOS策略，选择bigbird20。消融探针1583024.0于18:43:32 UTC在`slot1_1@b9g57n0029.cern.ch`执行，实际GPU为NVIDIA A100-PCIE-40GB；两段256updates实测66.199456/66.554833events/s，完整operation64.413513/63.628895s，GPU allocated peak1548.42MiB。Profile Train10k准备251.428365s，32散点精确输入检查通过。

原1583010.0消融近期速率25.601948events/s。工作节点的EOS日志曾只显示209,766曝光；通过`condor_ssh_to_job`直接读取旧worker，确认已记录217,760曝光。保存checkpoint为200,000曝光。根据两次4k screen的实际时间，剩余两个4k screen加两个40k完整表的原设备验证估计1,159.146309s。

迁移前冻结的计算：

| Forecast component | Seconds |
| --- | ---: |
| Old remaining train and validation | 8,277.354698 |
| New full preparation, 14×measured Train10k | 3,519.997104 |
| New recovery check allowance | 180 |
| Additional margin | 600 |
| New remaining including preparation/check/margin | 7,769.457687 |
| New with 10% train/forward throughput penalty | 8,154.953307 |
| Predicted advance | 507.897011 |
| Predicted advance under 10% penalty | 122.401391 |

准备估计对worker启动也线性放大，并使用未完整本地staging的profile；正式组会先复制压缩shards到scratch。较短的实际准备仍待测量。以上是当时剩余完成时间预测，未运行旧设备剩余反事实，不能写成已测时间节省；直接观测之后的旧未记录进度会进一步减少余量。

18:55:16 UTC，旧1583010.0已evict/abort，live queue消失、history JobStatus3；完整allocation10,943s全部计费。最终完整日志至少记录到27,500updates/219,758曝光，故保存200k后至少19,758曝光需重放；最后完整日志后精确曝光未测量。有效训练轨迹仍严格400k，不追加有效学习预算。重复计算的资源成本包括在allocation账本。

18:56:54 UTC写入新allocation1583024.0的release。新A100上真实连续/恢复fixture均完成9updates，最大参数差0、概率/曝光表逐位一致；这验证同新设备上的恢复路径，不宣称跨GPU任意轨迹逐位相同。19:01 UTC新组正在正式Train100k/Tune40k共享mmap准备，完整准备时间与正式恢复后吞吐待测；旧manifest在新正式runner开始前仍保留旧allocation ID，不把它作为旧writer仍活跃的证据。

同次资源准入审查：

| Observed resource | Actual policy evidence | Action |
| --- | --- | --- |
| Unclaimed H100 `b9pgpun015` and H200 `801/802` | Start requires BE.ABP.gpu or ATS.u_gpu group; submitted group is CMS.u_zh.users | Do not infer current user eligibility from Unclaimed |
| Unclaimed A100 `b9g57n0004` | InStagedDrain=true, ShutdownTime=1794543905, already expired | Does not offer the requested workday execution |
| Pair probe1583025.0 | Still Idle, NumJobStarts0 | Removed; history status3, allocation0 |
| Existing Kubeflow3g.47gb pair | Actual Pod Running; both saved200k exposures | Continue existing optimization |

这些是当次实时条件，不是GPU型号的永久限制。未修改accounting group，不将其他项目保留资源计作可用卡。

19:03:11 UTC账本快照为14.889452异构device-hours，活动allocation仅A100消融组和Kubeflow双模型组；其中旧1g所有尝试合计9.388056h、A100合计.485698h、3g.47gb合计4.988476h，已移除整H100尝试仍保留.027222h lifecycle上界。该快照不是最终成本，MIG/整卡hours不表示等效计算量。

产物：

- `audit/ablation_candidate_resources_stage303.json`：原始候选广告。
- `audit/profile_handoff_g2_ablation_c10/results.json`：两段实际测量及输入准备。
- `audit/native_migration_decision_c10.json`：当时的剩余时间计算。
- `audit/g2_ablation_c10_writer_handoff.json`：原writer终态及重放下界。
- `runtime/g2_ablation_c10.release.json`：与实际allocation绑定的恢复授权。
- `smoke/g2_ablation_c10_c0/integrity.json`：新A100真实恢复验证。
- `audit/pair_c11_eligibility_decision.json`：实际pool策略与零启动撤销依据。
- `audit/resource_handoff_stage303.json`：本阶段资源状态。

验证：新增profile配置source/live相等；正式模型配置、物理输入、seed、batch、LR轨迹、有效曝光和评价合同保持；现有handoff terminal-proof校验实际通过。上一阶段47项不同tests已通过，训练源码未变化，本阶段未重复完整tests。Cal-A/Cut-B/Confirm仍封存。

## 阶段性结论

事实：在先获得的A100上，消融实测训练速度为原1g的约2.59倍；原writer确认终态后才恢复到新设备，新设备确定性恢复通过。撤销了尚未获得实际可用资源的双模型探针，保留已运行Kubeflow双模型。

待测：新卡完整准备、恢复后正式吞吐和最终成本；三个正式G2模型质量、完整probe/health和最终科学决定。没有新的分类性能结论，R3尚未完成。

## 下一步计划

核对新缓存的事件覆盖、图摘要和散点tensor等价，确认消融在200k保存状态上真实恢复优化。完整三终点到达后，检查自动CPU评价、九指标独立复核及共享GPU上的最终probes，再完成条件扩展决定、英文图、终态账本和严格accept。
