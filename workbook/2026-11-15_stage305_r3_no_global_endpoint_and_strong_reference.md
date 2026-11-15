# R3：no-global完整终点与强B1参照

## 阶段目标

判断重训no-global模型在同100k事件、四遍训练预算下是否优于强B1 D0；记录完整终点、独立统计核对和实际A100终态成本。本阶段完成跨族比较，同族global路径效应仍需完整geometry评价。

## 工作内容

`b2_g2_global_ablation_t4_s42`已完成400,000次曝光和50,051次更新，四类各100,000次，`sum(lr)=7.865059679070895`，符合冻结预算。自己的T4 NLL最优screen为最终终点，两个预测表共用该终点。评价仅使用既有Train/Tune；Cal-A/Cut-B/Confirm payload继续封存。

CPU批任务`1583026.0`完成4视图、8配对、11域及400次physical-block bootstrap。独立points核对1,368项通过；九指标独立区间核对148项通过，含两项τ工作点，最大point/interval端点误差分别为`3.997e-15`/`2.200e-15`。

## 详细数据

运行根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。

### 全Tune固定终点

40,000事件，原始四类各10,000。BA3按非τ样本的条件三类输出计算；τ工作点使用完整非τ背景，均为开发Tune的raw score结果。

| Metric | No global | B1 D0 | Difference, no global minus D0 [95% interval] |
| --- | ---: | ---: | --- |
| BA3 | 55.1367% | 56.5333% | −1.3967 pp [−1.8301, −0.9905] |
| NLL3 | 0.901035 | 0.864586 | +0.036449 [+0.032634, +0.040582] |
| Electron recall | 60.82% | 64.43% | −3.610 pp [−4.3089, −2.9173] |
| Muon recall | 46.11% | 46.48% | −0.370 pp [−1.1656, +0.3368] |
| NC recall | 58.48% | 58.69% | −0.210 pp [−0.9018, +0.4195] |
| BA4 | 43.0475% | 44.7850% | −1.7375 pp [−2.1439, −1.3069] |
| NLL4 | 1.221309 | 1.185568 | +0.035741 [+0.032282, +0.039122] |
| Tau efficiency at 1% FPR | 2.77% | 5.11% | −2.340 pp [−2.9251, −1.7991] |
| Tau efficiency at 0.1% FPR | 0.47% | 1.16% | −0.690 pp [−1.0410, −0.2975] |

macro AP3差为−0.029973，macro AUROC3差为−0.018218。三类与τ晋级条件均失败；当前no-global配方不晋级。区间条件于已训练seed42和既有Tune，不包含训练seed不确定度。

### 稀疏事件的逐类取舍

`Accepted channels <100`域有12,153事件，原标签counts为`[2971,2801,3371,3010]`。BA3分别为39.0285%和39.1756%，差−0.1470 pp [−0.8395,+0.5867]。Electron recall差−9.0878 pp [−10.4514,−7.7110]；muon差+7.7829 pp [+6.5091,+9.0946]；NC差+0.8638 pp [−0.0988,+1.8097]。稀疏macro AP/AUROC没有净改善证据，τ在1% FPR下的差为+0.0890 pp [−0.3857,+0.7661]。μ recall提高伴随e损失，且μ在1% FPR下效率下降2.0707 pp，不能解释为稀疏μ分离整体改善。

### A100正式训练与终态成本

`1583024.0`已从queue消失，history为JobStatus4、ExitCode0，完整allocation为5,966s，申请12 CPU。它包含profile、恢复检查、完整缓存、启动、优化及验证。

从200k checkpoint恢复后，新200,000次曝光、25,023次更新消耗训练2,862.225683s，实测69.875692 events/train-second；同期验证186.872901s。该整段速率只描述A100上的恢复段。混合设备累计有效训练10,752.976482s、checkpointed operation wall 11,076.852852s；未保存后重放的历史成本另含在allocation账本，不能用混合设备全程平均数代表A100。

CPU评价allocation为109s、8 CPU，主评价64.664334s，独立区间复核14.758003s。提交前已检查schedd排队及EOS/WRITE兼容性。20:27:49 UTC全R3账本为17.629872异构device-hours、212.423294 allocated CPU core-hours、36条allocation；唯一活动allocation为Kubeflow `pinpoint-r3-g2-k4`。该值尚非最终成本。

### 产物与验证

- `audit/g2_ablation_endpoint_a100.json`：完整预算、最终输出、A100恢复段和Condor终态。
- `evaluation_global_ablation_projection/`：完整主评价、1,368项独立points、148项独立区间及CSV。
- `audit/scientific_review_m0_c1_ablation_c1.json`：三套已完成评价的局部review，最终四套review仍待齐备。
- `figures/r3_{endpoints,contrasts}_evaluation_global_ablation_projection.{png,pdf}`和对应class-guards图：三张英文图均检查PNG及经`pdftoppm`渲染的PDF，点值、区间、稀疏域和−1 pp guard一致，无标签裁切。记录为`audit/figures_stage305_review.json`，范围限这三图。

源码未改变，沿用stage302的47项不同tests；本阶段检查JSON解析、证据路径和`git diff --check`。

## 阶段性结论

事实：no-global完整终点及对同预算强B1的负结果成立。当前三类和τ均无晋级依据，稀疏事件的μ/e取舍也未构成整体改善。

仍待测：no-global−relations的同族差异，以及control/relations对强B1的完整结果。跨族比较同时改变表示与容量，不能据此断言global路径有用或几何信息无用；所有模型也尚未证明达到最佳效果。

## 下一步计划

完成Kubeflow两模型的最终400k终点、12视图/48配对geometry评价、三个native最终probe和六模型health。随后按完整证据决定是否触发seed/粒度/机制扩展，完成最终图审查、科学决定、终态资源账本和严格accept。
