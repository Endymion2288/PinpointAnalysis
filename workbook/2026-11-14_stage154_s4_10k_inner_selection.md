# S4 10k训练内选择完成与固定末轮重拟合

## 阶段目标

完成10k预算24项三折训练，按与1k相同的既定规则选择共同配方及epoch；冻结三seed重拟合、EMA/MAE5/Rel同设置下游和完整配对评价。

## 工作内容

10k全部完成12次完整遍历。每个配方/合格epoch先按fold留出数加权OOF NLL，再对Scratch/MAE3等权平均；共同选择`early`、第6次完整遍历，而非沿用1k选中的LP-FT。原独立validation没有参与这次选择。

生成[10k选择表](../roadmap/results/s4_low_label_selection_10k.json)、[六项三seed重拟合](../roadmap/configs/s4_low_label_refit_10k.json)、[12模型评价](../roadmap/configs/s4_optimized_10k_evaluation.json)、[12项配对与4项seed均值](../roadmap/configs/s4_optimized_10k_pairs.json)及[CPU评价任务](../roadmap/configs/s4_optimized_10k_validation.json)。完整预算每pass635次更新；六pass共3,810次更新、60,000次事件输入。两组都固定提交第6轮，不按validation另选epoch。

另外生成[10k三个SSL目标的六项下游](../roadmap/configs/s4_repaired_objectives_b.json)，fine-tuning采用相同early配方/固定第6轮；full/global probes保留固定C=1及收敛要求。将六项新refit、目标比较、计算对照及frozen probes合为[33任务dispatch](../roadmap/configs/s4_repaired_dispatch_b.json)，逐项保持原campaign配置和任务名，依靠共享claims与Condor协作。

## 详细数据

各配方在自身最优共同epoch的OOF NLL如下；它们是相同调参集上的描述性选择值，不能当作新的独立泛化结果。

| Recipe | Common epoch | Scratch OOF NLL | MAE3 OOF NLL | Equal-initialization mean |
| --- | ---: | ---: | ---: | ---: |
| **early** | **6** | **1.302972** | **1.277887** | **1.290429** |
| slow | 12 | 1.338998 | 1.266668 | 1.302833 |
| regularized | 12 | 1.336170 | 1.265023 | 1.300596 |
| lpft | 12 | 1.315446 | 1.275046 | 1.295246 |

10k所选encoder LR scale=1、weight decay=0.01、Head dropout=0.1、LP passes=0；readout LR3e-4及共同scheduler按冻结候选执行。不能只为MAE选取其单独更优的slow/regularized结果，再与另一套Scratch配方作初始化归因。

1k/10k共48项inner全部complete，合计137,088次逐步梯度检查、2,112,000次事件输入；每pass曝光与实际fold-fit事件数完全一致。1k/10k激活与梯度采样分别334/1,398项，全部healthy，其中明确LP采样52/96项；inner最大preclip gradient norm为264.337341，未触及10,000数值审查线。

可复现选择命令：

```bash
python -m analysis.flavour.s4.plan_low_label resolve \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 \
  --source roadmap/configs/s4_low_label_inner.json --budgets 10000 \
  --selection-report /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/evaluation/low_label_optimization/selection_n10000.json \
  --output roadmap/configs/s4_low_label_refit_10k.json
```

真实命令exit0。源码中的规则、完整遍历与预算步数重算已有阶段148的9项回归检查，本阶段只生成数据和配置，未新增模型代码或重复测试。新10k完整预算分类结果、τ及低FPR收益尚未测量；calibration/test保持冻结。方案在观察此前validation后设计，后续仍属于同validation的条件性重复评价。

Kubeflow `pinpoint-s4-inner-main`的Job completion为03:03:00 UTC，Pod live phase为Succeeded；已保存完整Job/Pod终态快照到EOS `audit/backends`，随后删除终态Job以接续使用唯一GPU单元。原运行campaign未修改。Condor inner终态与累计成本继续由账本刷新，不能以manifest完成代替allocation终态。

## 阶段性结论

事实：10k共同OOF规则选择完整fine-tuning六pass；1k选择LP-FT八pass，最合适的候选随标签预算不同。inner训练数值健康，完整预算的新10k收益尚未测量。

待检验假设：减少10k重复曝光可改善概率质量并保留排序收益；必须通过固定末轮三seed及完整同事件评价检验。不同预算选中不同配方不是LP-FT或预训练整体优劣的结论。

## 下一步计划

正式提交六项10k refits、冻结CPU评价及六项SSL目标下游，用已释放的Kubeflow GPU六进程接续33任务dispatch，并与Condor共享claims。完成1k配对结果后，再审查frozen probe的拟合/泛化误差及正则需求，最终汇总S4全合同与成本。
