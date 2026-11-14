# S6：全部校准分数与几何标签读回

## 阶段目标

合并全部calibration分数，量测S6几何标签修正影响；保存generator连续能量匹配残差，准备最终冻结。

## 工作内容

完成最后task129的13模型BF16 compiled推理并检查内部health。全227块合并后逐候选拟合temperature，沿用既有validation选择保守完整tie阈值。新增transport τ/charm分支读回字段供后续真实条件对照，成功日志压缩。补充独立后端allocation记账程序，MIG实例和整卡分别列出。

## 详细数据

Calibration全722,272个事件及227个推理分块合并通过，输出概率301,775,875 bytes；各任务forward时间累计8,327.555秒，任务wall累计9,108.745秒（并发任务时间不能相加成device allocation）。最后task129推理45.460秒，13项health全部通过。

转换累计144,926.356任务秒，最大单任务RSS2776.430MiB，native caches合计13,808,716,151 bytes。τ有64,901个有效decay vertex；S6将silicon thickness统一为0.01mm后修正2364个material location标签。观测、四类标签和checkpoint未改变。

首两个温度拟合：MAE seed17 T=1.064991，balanced NLL1.203920→1.203443；seed42 T=1.040432，NLL1.204652→1.204464；均未到边界。其余候选仍在拟合，本报告不预先声明全量校准完成。

Generator220/220小批完成3512事件response。匹配coarse energy cells后仍有连续能谱残差，四类KS统计量0.04297/0.04492/0.05455/0.04492，详见`generator/matching_diagnostics.json`。Charm primary counts Standard/NUX：e41/42、μ61/50、τ23/19、NC0/0。对照将保留这些残差作为解释范围，不当成逐事件相同hard scatter。

## 阶段性结论

完整calibration概率已可使用，初步temperature为接近1的小幅变换。Generator真实response已齐备。尚未打开独立test，也尚无系统变化分类结论。

## 下一步计划

检查全部15个拟合数值后保存freeze，提交全部test。继续paired读出、真实field/material/helicity和generator推理，执行完整统计和成本审查。
