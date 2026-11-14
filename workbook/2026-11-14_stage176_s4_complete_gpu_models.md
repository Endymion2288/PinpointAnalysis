# S4 五种目标的100k模型完成与GPU释放

## 阶段目标

完成修复后的全部目标训练和一次性feature提取，保留内部健康、真实GPU终态及尚未解决的读出问题。

## 工作内容与详细数据

15项seed42目标分类配置（五种初始化×1k/10k/100k）均已complete，共147,130次更新、1,710次计划内部采样，采样健康全部通过。最大记录preclip norm307.565003，全部Head SiLU精确零比例为0。完整独立CPU checkpoint读回仍由正在等待readout的最终validation执行，不能把此处complete替代该检查。

100k五项均25,024 updates、282次内部采样、best第四epoch，完整10k nominal validation的实测如下，尚未在本报告形成配对显著性结论：

| Initialization | NLL | Macro AP | Tau AP | Maximum preclip norm |
| --- | ---: | ---: | ---: | ---: |
| Scratch | 1.214940 | 0.468453 | 0.350946 | 134.681381 |
| MAE3 | 1.207316 | 0.473777 | 0.359616 | 34.082367 |
| EMA3 | 1.224415 | 0.463002 | 0.351237 | 77.628159 |
| MAE5 | 1.204849 | 0.474457 | 0.361759 | 32.709156 |
| Channel Rel5 | 1.202688 | 0.475250 | 0.359629 | 34.750748 |

MAE5与Channel Rel5共用MAE3 parent并继续同样训练曝光，Rel另用模拟bar语义。单seed绝对数值不能代替配对区间，也不能将Rel称为纯自监督。

五份100k train/10k validation feature bank已完整写出，full/global数组合计3,590,402,560 bytes；后续CPU求解期间不占GPU。Kubeflow Job真实成功时间05:28:47 UTC，读取时Pod已由平台自动清理，保存成功Job与先前running Pod快照，以Job completion作为保守结束上界；未捏造Pod Succeeded记录。随后删除Job。两组Condor训练allocation2204998/1578819也已由history确认终态。05:37:08 UTC全部GPU成本固定为35.317084908/60 device-hours，CPU仍运行，CPU费用当时405.525387 core-hours。完整来源分别为`audit/completed_valid_objective_finetunes.json`、`feature_bank_sizes.json`、`newton_v2_feature_release.json`及`audit/backends/`。

## 已发现的读出问题

所有1k/10k frozen读出均收敛，但固定C=1的低标签full表示存在很大的train/validation差距。1k MAE3 full fit NLL0.174783、fit accuracy0.977，validation NLL3.246709；global为1.049773→1.579325。10k full为1.062646→1.381796，global为1.269359→1.316041。驻点通过证明数值求解完成，不能证明泛化或概率质量良好。这些固定配方的差结果保留为诊断，不能用来宣布encoder无信息。

事实：差距随标签预算与读出维数明显改变，且真实优化梯度达到原阈值；与旧Head整层失活、解冻断梯度是不同现象。假设：高维少标签下正则化和feature尾部尺度影响该读出，尚未做单变量train-inner选择，不能认定C不足是唯一原因。未来readout优化应仅用原标签预算内的分组train-inner选择C，并重新测试；本轮固定C=1比较完整保留，不在观察validation后静默修改配方。

## 阶段性结论

事实：三种额外目标的100k训练及五份feature bank均完成，旧失活现象未重现，GPU已释放。科学收益、30个磁盘readout概率回放及100k驻点仍等待统一CPU评价。内部健康、数值收敛和概率泛化分别判断。

## 下一步计划

完成五组100k驻点、30项磁盘概率回放、45预测/66配对及完整条件统计。明确固定C=1的低标签限制，随后清理暂存数组、AFS终态提交目录和最终成本记录。
