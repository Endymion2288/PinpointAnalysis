# R3：完整结果、科学决定与最终验收

## 阶段目标

完成v2 R3的全部冻结范围，回答采样、首层降采样、native关系层、global路径和紧凑图像的实际效果；根据完整质量证据决定条件扩展，并交付最终probe、健康、英文图、终态成本和下一阶段路线。

## 工作内容

六个正式模型全部完成规定曝光和更新。M0/G1分别使用Train300k×4、150,117 updates；control/relations/no-global/C1各使用Train100k×4、50,051 updates。独立优化器、曝光和LR轨迹按五组冻结控制逐项核对；移除global路径后的预算保持，参数量变化单独记录。

Control/relations在Kubeflow H100 NVL MIG 3g.47gb双并行完成后，继续复用共享mmap对三个native最终权重作完整probe。CPU统计、独立复核及health由Condor执行，与GPU后续重叠。原1g、A100和Kubeflow的恢复、完整输入、真实拟合和单位修正证据继续保留。

Cal-A/Cut-B/Confirm payload全程封存。严格accept已通过249项；随后人工核对完整范围及科学解释，见`audit/final_delivery_review.json`。本阶段不新增训练源码或重复stage302已通过的47项不同tests。

## 详细数据

运行根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。最终资源快照为2026-11-15 20:57:45 UTC。

### 固定终点

Tune40k原始四类各10k。BA3/NLL3使用非τ样本的条件三类概率；τ工作点使用完整非τ背景。主决定使用固定曝光终点，自己的T4 NLL最优4k screen单独评价，不在完整Tune上选checkpoint。

| Model | Parameters | Unique Train | Exposures | BA3 | NLL3 | BA4 | NLL4 | Tau efficiency at 1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 mixing | 793,674 | 300,000 | 1,200,000 | 57.5100% | 0.845725 | 46.0275% | 1.163531 | 6.03% |
| G1 stride 2 | 793,674 | 300,000 | 1,200,000 | 57.0933% | 0.850461 | 45.6825% | 1.167762 | 5.97% |
| G2 control | 199,882 | 100,000 | 400,000 | 55.6100% | 0.891512 | 43.4350% | 1.213798 | 3.14% |
| G2 relations | 207,883 | 100,000 | 400,000 | 55.5867% | 0.892097 | 43.4700% | 1.214397 | 2.69% |
| G2 no global | 157,771 | 100,000 | 400,000 | 55.1367% | 0.901035 | 43.0475% | 1.221309 | 2.77% |
| C1 sensor rows | 797,594 | 100,000 | 400,000 | 56.6700% | 0.861747 | 44.9025% | 1.185646 | 4.84% |

自己的最优screen updates依次为125,098、137,748、50,051、50,051、50,051、50,051。模型数据量不同，表中绝对值不构成同预算跨族排名。

### 完整评价与独立复核

| Evaluation | Model views | Paired contrasts | Domains / model | Draws | Independent point checks | Independent interval checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 4 | 8 | 11 | 400 | 1,368 | 148 |
| Geometry | 12 | 48 | 11 | 400 | 4,104 | 876 |
| C1 | 4 | 8 | 11 | 400 | 1,368 | 148 |
| No global versus B1 D0 | 4 | 8 | 11 | 400 | 1,368 | 148 |
| Total across matrices | 24 | 72 | — | — | 8,208 | 1,320 |

24视图按矩阵计，含重复参照，独立权重不是24个。Bootstrap使用6,845个accepted physical blocks；独立区间验证覆盖BA3/NLL3、e/μ/NC recall、BA4/NLL4和τ的1%/0.1%工作点。Geometry复核最大point误差`9.104e-15`、interval端点误差`3.324e-15`。

Geometry CPU任务`1583027.0`终态Exit0，8CPU、305s allocation；主评价214.254960s。六模型health `1583028.0`为2CPU、33s；最终科学review `1583029.0`为2CPU、8s。提交前均自动检查schedd排队、WRITE及EOS策略，选择bigbird20。科学review覆盖四套评价的18项all-domain endpoint/best对比，九项主终点均未通过三类或τ晋级规则。

### Native同族与强B1比较

| Endpoint contrast | BA3 difference, pp [95% interval] | NLL3 difference [95% interval] | Tau efficiency difference at 1% FPR, pp [95% interval] |
| --- | --- | --- | --- |
| Relations − control | −0.0233 [−0.3084,+0.2191] | +0.000585 [−0.001075,+0.002403] | −0.450 [−0.7954,−0.1008] |
| No global − relations | −0.4500 [−0.7423,−0.1402] | +0.008938 [+0.006241,+0.011819] | +0.080 [−0.2391,+0.3221] |
| No global − control | −0.4733 [−0.7820,−0.1780] | +0.009523 [+0.007065,+0.012437] | −0.370 [−0.7279,−0.0290] |
| Control − B1 D0 | −0.9233 [−1.2798,−0.5710] | +0.026926 [+0.023312,+0.030426] | −1.970 [−2.5132,−1.4616] |
| Relations − B1 D0 | −0.9467 [−1.3375,−0.6022] | +0.027511 [+0.024087,+0.030924] | −2.420 [−2.9748,−1.8431] |
| No global − B1 D0 | −1.3967 [−1.8301,−0.9905] | +0.036449 [+0.032634,+0.040582] | −2.340 [−2.9251,−1.7991] |

Relations−control的e/μ/NC recall差为−0.210/−0.320/+0.460 pp；NC增加伴随其余类别取舍，未形成主任务净增益。No-global−relations的μ recall下降1.370 pp [−1.9833,−0.7613]。三组同族模型不是完整2×2，移除stack也降低容量；只对当前重训方案作结论。

稀疏域`Accepted channels <100`有12,153事件，counts `[2971,2801,3371,3010]`。Relations−control BA3为+0.2288 pp [−0.2515,+0.7029]，AP/AUROC差均跨零；no-global−relations BA3为−0.0153 pp [−0.5644,+0.6668]，e/μ recall分别−2.6927/+2.7133 pp。相对强B1的局部μ增益仍伴随e损失，未证明稀疏分离恢复。

### 有信息、实际使用与实际增益

六模型最终health通过；M0/G1各152个窗口，其余各52。所有分支观察到梯度及代表tensor参数更新、激活finite。G1 tracker有一个零梯度窗口，其余151个正梯度窗口，不能将个别空输入窗口解释为永久失效。Relations/no-global的gate、message、norm均52个正梯度和52个正更新窗口。

四组独立probe覆盖七个权重（六formal+D2 reference），Train4k拟合、Tune40k评价。Native local/regional/global/tracker维度均128；no-global不存在regional路径，未伪造该probe。Control/relations完整权重重算概率差0、argmax一致100%；no-global跨A100/H100最大差0.0037148、平均差0.0001532、argmax一致99.89%，通过预设容差。其它四个权重此前均差0。

| Native model | Local BA3 | Regional BA3 | Global BA3 | Tracker BA3 | Sparse local / global / tracker BA3 |
| --- | ---: | ---: | ---: | ---: | --- |
| Control | 47.9767% | 47.9900% | 48.9300% | 41.6700% | 35.6099% / 34.6032% / 38.3147% |
| Relations | 47.0200% | 47.8133% | 48.9700% | 41.8267% | 34.4327% / 35.0943% / 38.2234% |
| No global | 47.7133% | Absent | 48.5367% | 41.6067% | 34.8543% / 34.9026% / 38.0846% |

Probe可读性与训练梯度不等于增量收益；global为tracker融合前detector embedding，local/regional使用自己的池化定义。主质量由重训配对和完整预测决定。

### 资源与图

全部39条allocation终态，`active_or_unresolved=[]`，18.094166667异构device-hours、223.338333333 allocated CPU core-hours。失败、未保存重放、profile、迁移、缓存和最终probe均计入，一个allocation不因多worker重复计时。

| Device or partition | Allocated device-hours |
| --- | ---: |
| H100 1g.12gb, all attempts | 9.388056 |
| A100-PCIE-40GB, all attempts | 1.815556 |
| H100 NVL 3g.47gb, all attempts | 6.863333 |
| Previous h100-94gb lifecycle upper bound | 0.027222 |

最后Kubeflow `pinpoint-r3-g2-k4`成功后Pod已清理；已保存PodScheduled与live Job Complete支持11,692s保守allocation上界，包含同卡双训练及三个最终probe。该作业20:55:39 UTC结束。GPU utilization未测量，异构小时不作FLOP等效。A100恢复段69.8757 events/train-second见stage305；不把混合设备全程平均数称为A100吞吐。

合同14图及9张逐类guard图共23张英文图，PNG及`pdftoppm`渲染PDF均实际目视审查。检查指标、区间方向与单位、稀疏定义、−1 pp guard、真实健康范围及不存在的regional路径；记录为`audit/figures_final_review.json`，图在`figures/`。

## 阶段性结论

事实：完整R3交付完成；M0、G1、relations、no-global和C1均无满足预定质量规则的晋级证据。M0的BA增加伴随μ损失；G1没有净改善；relations同族BA近零且τ工作点下降；重训移除global降低总体质量。C1有明确成本优势，但e质量损失尚未恢复。

决定：保留R2 B1 D2真实T3/T4固定终点及原采样；不触发seeds17/2026、粒度扫描、同容量/打乱边、anchors、late fusion或branch modulation追加训练。C1继续作为未来成本候选，本轮不重复已失败的bias恢复。七项条件决定写入`conditional_decisions.json`，完整科学解释在`decision.json`。

限制：这些是seed42开发Tune结果，没有独立确认；没有证明充分收敛或容量上限。当前候选图的高覆盖不等于高正确关联，首贡献标签也不是完整轨迹truth。R3具体配方失败不推出硬件信息无用。

## 下一步计划

详细设计见[新的R3结果与R4–R7工单](../roadmap/v2/10_r3_results_and_next_steps.md)：R4先冻结Train/Tune各10k的标签支持、候选关联与robust fit评价，包含方向/uncertainty和真实spacing，交付整轨efficiency/fake/coverage后再选择一个raw+object比较；R5保留flat4并研究条件τ分支，新增子类以有效标签及inclusive收益为门槛；R6只针对明确缺口选择一项；R7继续保留Cal-A/Cut-B/Confirm独立确认。

R0已记录上界+R1/R2/R3 batch累计72.363744h，原80h规划名义余量7.636256h，完整交互开销未测量。后续先CPU小批次，再按实际预计完成时间选择异构/MIG/共享GPU，保留R7资源。

关键命令与最终证据：

```bash
python -m analysis.flavour.v2.r2.accounting --root "$R3_ROOT"
python -m analysis.flavour.v2.r3.accept --root "$R3_ROOT"
```

`acceptance.json`：249 passed、0 failed；`audit/final_delivery_review.json`：source/live/manifest配置相等、原revision2范围全部保留、四矩阵独立复核和真实终态成本已核对。机器结果回填`roadmap/v2/r3_results.json`、`r3_execution.json`和`experiment_plan.json`；报告、图和源码引用分别保留。

最终文档与metadata核对80项全部通过：3份JSON可解析，455个本地Markdown链接及其中11个anchor有效，六模型配置/预算/指标、四矩阵、七权重probe、科学决定和终态账本逐项对应。模型manifest与完整评价的NLL仅有浮点累加次序差，最大绝对差`3.109e-15`，采用`1e-12`核对容差；其余指标一致。原revision2合同与Git已提交版本及live版本相等。记录为`audit/documents_stage306_review.json`；`git diff --check`通过。
