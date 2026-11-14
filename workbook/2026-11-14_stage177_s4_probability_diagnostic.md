# S4 低标签frozen读出的过度自信诊断

## 阶段目标

判断固定C=1少标签读出的高validation NLL是否由少数离群事件主导，以及它与旧Head失活是否相同。

## 工作内容与详细数据

仅重新读取已完成预测，不训练、校准或选择阈值。MAE3的四项1k/10k full/global各使用相同10k validation，实测为：

| Labels / branch | Mean NLL | Median event NLL | Top 1% contribution to total NLL | Wrong predictions with confidence >0.9 | True-class probability <0.001 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k full | 3.246709 | 2.224457 | 5.0781% | 2,606 | 1,402 |
| 1k global | 1.579325 | 1.391716 | 3.5265% | 81 | 8 |
| 10k full | 1.381796 | 1.280947 | 3.2360% | 45 | 0 |
| 10k global | 1.316041 | 1.306600 | 2.2167% | 4 | 0 |

1k full的逐类概率标准差为0.322–0.346，并非常数预测。最高1%事件只贡献5.08%总损失，中位NLL仍2.224，不能将失败归为少数离群事件。其train NLL0.174783/accuracy0.977与validation间存在明显泛化差距；真实梯度驻点已通过，不能用收敛或finite系数宣称概率质量合格。

新增`analysis.flavour.s4.plots --frozen-readout-diagnostic`，输出英文逐事件NLL CDF和confidence reliability，不拟合calibration模型。命令为：

```bash
python -m analysis.flavour.s4.plots --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 --frozen-readout-diagnostic --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/evaluation/frozen_readout_diagnostic
```

实际生成并查看PNG，CDF与reliability均正确显示，PDF/PNG和逐confidence bin计数CSV保存在上述目录。数值见[精简诊断](../roadmap/results/s4_frozen_probability_diagnostic.json)。没有为此可逆绘图增加镜像实现的测试。

## 阶段性结论

事实：固定C=1的1k高维读出有广泛过度自信和泛化损失，与旧整层失活不同。10k与global较轻。假设：标签/维数与正则化设置是优先检查对象；目前观察不能独立证明其中单一成分的因果作用。

固定C=1仅保留为已冻结的诊断对照，不能据此判断预训练表示无信息，也不能推荐其1k full概率部署。后续应在各原标签预算内用分组train-inner选择regularization，再重新拟合并报告完整验证；同一次设计已观察过validation，后续统计须保留该设计条件。

## 下一步计划

完成正在运行的100k驻点和45预测/66配对。最终结论分别说明数值求解、表示排序与概率泛化，保存该失败诊断及后续修复路线。
