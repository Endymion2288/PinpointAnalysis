# S4 1k三折优化结果与固定末轮重拟合

## 阶段目标

完成1k预算的24项train-inner对照，按[阶段146](2026-11-14_stage146_s4_train_inner_optimization_contract.md)固定规则选择共同训练配方，并生成六项三seed重拟合。10k预算继续独立运行，不等待它完成才执行1k决策。

## 工作内容

24项1k训练均完成12次完整遍历，包含四配方、Scratch/MAE3及三个物理family隔离的train-inner folds。选择只读取原train中的留出预测，原独立10k validation、calibration/test均未参与本次选择。

对每个配方及合格epoch，按各折留出事件数加权计算两种初始化的OOF NLL，再等权平均；两组共同选择LP-FT、第8次完整遍历。LP前两轮不进入该候选的选择，避免将尚未解冻的linear probe当作full fine-tuning。

生成重拟合时重新计算完整预算的采样步数：每折49/52步改为完整1k每遍74步。该转换已加入使用真实sampler的回归检查。另禁止监督continuation悄然重启LP，避免加载已有非零Head后破坏线性阶段的定义。

## 详细数据

完整选择表为[s4_low_label_selection_1k.json](../roadmap/results/s4_low_label_selection_1k.json)，运行原表位于输出根`evaluation/low_label_optimization/selection_n1000.json`。每个配方在其自身最佳共同epoch的结果如下，NLL为无量纲：

| Recipe | Common epoch | Scratch OOF NLL | MAE3 OOF NLL | Equal-initialization mean |
| --- | ---: | ---: | ---: | ---: |
| early | 3 | 1.493005 | 1.429028 | 1.461016 |
| slow | 4 | 1.489376 | 1.399914 | 1.444645 |
| regularized | 5 | 1.480959 | 1.391085 | 1.436022 |
| lpft | 8 | 1.433131 | 1.390133 | 1.411632 |

LP-FT相对early的共同OOF NLL减少0.049384。这是相同train-inner调参集上的描述性最优值，包含配方及checkpoint选择，未测量独立泛化增益。所选配方下，两组OOF NLL仍均高于均匀先验ln4=1.386294，低标签概率质量问题尚未解决。

1k矩阵合计14,688次更新、192,000次事件输入；每个完整epoch的曝光数均等于该折fit事件数。334项内部健康采样全部通过，其中52项为明确的LP阶段，解冻首步检查齐全。ντ argmax F1、低FPR效率和common-support收益尚未在新的完整预算模型上测量。

生成配置为[s4_low_label_refit_1k.json](../roadmap/configs/s4_low_label_refit_1k.json)：Scratch/MAE3×seeds 17/42/2026，共6项。每项固定8次完整遍历、592次更新、8,000次事件输入；前148次更新只训练线性readout，第149次开始完整fine-tuning。encoder LR scale=0.1，matrix weight decay=0.1，Head dropout=0.3；readout LR与scheduler沿用既定候选。**最终固定提交第8轮，不按独立validation重新挑checkpoint。** 新模型尚未完成，未测量其validation结果。

可复现命令：

```bash
python -m unittest tests.test_flavour_s4_low_label
python -m analysis.flavour.s4.plan_low_label resolve \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 \
  --source roadmap/configs/s4_low_label_inner.json --budgets 1000 \
  --selection-report /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/evaluation/low_label_optimization/selection_n1000.json \
  --output roadmap/configs/s4_low_label_refit_1k.json
```

`resolve`真实运行成功，9项针对性测试全部通过；覆盖原train隔离、完整遍历、线性LP、冻结/解冻及resume、预算独立选择、排除未解冻epoch，以及完整预算步数重算。CERN/Condor的精确启动状态和累计预算继续以EOS submission记录及账本为准。

## 阶段性结论

事实：1k的固定选择规则选中LP-FT八次完整遍历，inner NLL低于其他三种配方，但仍未优于均匀先验。此次选择未使用独立validation，训练数值健康。

假设：增加到完整1k拟合数据后能否降低NLL并保留MAE的排序收益，仍需三seed重拟合检验；不能把最优inner结果或冻结encoder期间的零梯度解释成最终质量或架构失效。

## 下一步计划

执行六项固定末轮重拟合，对相同完整10k validation做三seed配对评价，并与阶段145基线分别比较排序、概率和τ硬决策。10k矩阵完整后独立选择其配方及轮数。继续执行已提交的修复后frozen probes；只有分类训练及表示诊断有证据后才推进其他预训练目标和matched-compute控制，本轮不扩1M。
