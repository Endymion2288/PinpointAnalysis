# Stage282：R2更多unique数据的实际收益与稀疏域限制

## 阶段目标

用完整D2 T3终点回答“相同曝光下，更多unique数据是否优于重复旧数据，以及相对较短D0是否已有实用增益”。D2 T3、配对区间、共同Train gap和五折readout已完成；D2 T4及D1WD T3/T4仍在训练。当前不是完整R2交付，也没有冻结1M或加宽扩展。

## 工作内容

完成新增模型的完整Tune读回，将现已完成的D0 T3/T4、D1 T3/T4、D2 T3统一评价为10个endpoint/best-screen视图、11域、48项配对。使用与最终矩阵相同的400次paired accepted-family-connected-chunk bootstrap、seed20261116；每个区间只针对已训练seed42的开发样本，不估计训练seed或训练pool抽样变化。

独立BA/NLL、原始选择计数及sklearn weighted AP/AUROC通过2,298项；独立核心区间通过682项。相同五折readout也更新到五个模型。原始预测仍作为主评价，温度/bias只作开发诊断。

## 详细数据

### 全域更多数据与更多重复

同一40k Tune，非τ三类各10k。D0/D1分别使用75k非τunique、4/12 passes；D2使用225k非τunique、4 passes。D1/D2同为900k非τ曝光，实际updates为112,645/112,630，尾批导致小量差异，不称精确fixed-update。

| T3 endpoint | Unique events | Exposures | BA3 (%) | NLL3 | Macro AP | Macro AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 75,000 | 300,000 | 56.6500 | .871321793 | .656279067 | .769607053 |
| D1 | 75,000 | 900,000 | 53.1700 | 1.143738363 | .613771288 | .738472882 |
| D2 | 225,000 | 900,000 | 57.0967 | .855866803 | .670214658 | .778902048 |

| Comparison | ΔBA3 (pp), 95% interval | ΔNLL3, 95% interval |
| --- | --- | --- |
| D2−D1 | +3.926667 [3.380711,4.438761] | −.287871560 [−.300109592,−.275855644] |
| D2−D0 | +.446667 [.077522,.815348] | −.015454990 [−.018357127,−.012586502] |

D2−D0的e/μ/NC recall变化为−2.25/+5.28/−1.69 pp，BA增益低于既定2 pp实用门槛。统计正效应和足够大的实际收益是两个判断。D2相对D0同时改变unique数量与总曝光；真正同曝光比较是D2−D1，不把前者称为纯数据效应。

D2 best-screen位于75,089 updates、约600k非τ曝光，完整Tune BA3/NLL3=.568133333/.857964660；固定endpoint为112,630 updates，其BA/NLL均稍好。最后四个screen NLL为.868277018/.871104501/.864635238/.865114164，未满足三个连续变化均<.001的工程plateau。当前不能据此宣布全局收敛，也不将波动曲线外推为继续三倍曝光一定有益。

共同Train子集为100k pool内每个非τ类1k、eval mode。D2 endpoint Train BA3/NLL3=.590666667/.820945653，对应Tune−Train NLL=.034921150；best-screen Train BA3/NLL3=.577333333/.833755831。D1 endpoint的对应NLL gap=.516755766。更多unique明显减弱了当前重复曝光的过拟合，Train拟合和Tune泛化仍需分别解释。

### 稀疏域：recall改善不等于分离改善

原accepted channels<100，包含空响应，原四类支持为2971/2801/3371/3010。域内三类统一为均匀先验。

| T3 endpoint | BA3 (%) | NLL3 | μ recall (%) | μ AP | μ AUROC | μ eff @ 1% FPR (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 39.400819 | 1.078160482 | 10.317744 | .397356514 | .545741946 | 3.998572 |
| D2 | 38.687525 | 1.076379420 | 27.561585 | .401881584 | .548284293 | 4.605498 |

D2−D0：稀疏BA3为−.713293 pp，区间[−1.605710,+.083226]；NLL3为−.001781062，[−.004689169,+.001517205]；macro AUROC差−.000389931，[−.006920630,+.006623698]。μ recall上升17.24 pp，同时e/NC下降11.41/7.97 pp。

μ的1% FPR效率差为+.606926 pp，区间[−.109638,+1.078563]；.1% FPR效率差为−.035702 pp，区间[−1.107007,+.748188]。当前区间不支持把稀疏μ的工作点变化称为已建立的净改善。固定工作点仍为开发域内选阈值，R7另用Cut-B/Confirm，未将此处阈值带入封存集。

### 决策偏移与置信度诊断

与stage279同一五折，6,845个物理chunk，每折原四类各2k；仅非τfit。温度/bias的参数在fit折拟合，held折评价；基础模型选择本来已使用Tune，因此仍属于开发诊断。

| Model | Raw BA3 (%) | OOF bias BA3 (%) | Raw NLL3 | OOF temperature NLL3 | OOF bias NLL3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1 T3 | 53.1700 | 53.1933 | 1.143738363 | .919987376 | 1.107505702 |
| D2 T3 | 57.0967 | 57.0633 | .855866803 | .847037033 | .854078001 |

所有温度变换保持argmax。D1置信度失配解释部分NLL退化，但温度调整后仍落后于D0的.864601421和D2的.847037033；D1原始AP/AUROC同时退化，不能把过拟合全归于置信度。D2两个全局bias没有改善整体BA，其稀疏BA也从38.687525%变为38.403341%。未测量occupancy条件bias，未以这项有限诊断排除所有readout方案。

### 执行、产物和资源

结果根`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2`。新增Condor作业：`1582981`局部400次配对评价、`1582982` D2 T3 Train gap、`1582983`五折readout；均在记录队列与兼容性后选择`bigbird20.cern.ch`。这些工作均已完成。

- `evaluation_partial/`：当前五模型的10视图/48配对，后续进展可继续更新该临时入口。
- `readout_partial/`：五模型的相同合同诊断；独立score检查2,298项通过。
- [仓库只读快照](../roadmap/v2/r2_data_scaling_review.json)：固定本阶段模型、逐类/稀疏指标、配对和诊断。
- `gap/b1_d2_t3_s42/`：两个checkpoint的共同Train预测、报告。
- 英文图`r2_train_tune_gap_partial`、`r2_low_occupancy_partial`和其他`_partial` PDF/PNG。

资源快照累计6.097911异构device-hours，仍含活动allocation，不能作为终态成本。`r2.accounting`现在单列`rejected_before_allocation`，保留EosSubmit拒绝及零allocation性质；运行失败与真正分配时间仍由终态ads/Pod记账。当前D2、两个D1WD分别使用两块Condor MIG及一块Kubeflow MIG；有效运行未为换型号而中断。

验证命令沿用LCG108及单线程环境：

```bash
python -m analysis.flavour.v2.r2.plots \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2 --partial
python -m analysis.flavour.v2.r2.accounting \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2
```

新统计已通过独立数值与区间读回；图表为已完成模型的局部输出，完整正式图仍需等待全部模型。最新入口同步更新`roadmap/README.md`和`r2_execution.json`，避免把早期六模型快照当作当前执行状态。

## 阶段性结论

事实：同曝光下更多unique事件明显优于重复训练当前100k pool；相对D0的总体净BA收益较小，并伴随逐类取舍。稀疏排序和固定FPR工作点尚未出现足够证据的净改善。低维全局bias没有恢复这部分损失。

假设：当前稀疏特征路径可能比单纯增加同类数据更值得优先改进。该判断尚未排除更大数据、容量或条件readout的收益，不能称为物理上限或全局最优。

## 下一步计划

完成D2 T4及两个D1WD终点后，统一16视图/128配对、全部Train gap、readout与正式图表，作一次完整分流。当前不因D2−D1的大差值自动扩成1M；需同时考虑相对D0的绝对效应、逐类约束、τ工作点和实际成本。最终形成科学决策、资源终态与R2验收。
