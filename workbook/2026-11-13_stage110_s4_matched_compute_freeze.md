# S4：等计算监督控制配置冻结

## 阶段目标

以实际ragged sampler/masks重放得到的矩阵运算代理，固定监督继续训练预算，避免用额外训练量解释预训练收益。

## 工作内容与详细数据

正式CPU matched-plan cluster1574117完成，用时888.1089秒、4CPU。完整每事件shape tables仅24,128 / 240,128 / 2,400,128 bytes，分别对应1k/10k/100k；监督sampler和cost无需重复读原始响应。预算结果写入`roadmap/results/s4_matched_budgets.json`，训练配置`configs/s4_supervised_matched.json`。

| Labels | Original supervised proxy | MAE3 + downstream proxy | Extra supervised epochs×steps | Extra proxy matching error | Extra train exposures |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1.0631279172×10^15 | 2.8674512265×10^15 | 8×873 | +0.0306506% | 97,065 |
| 10,000 | 3.0623671179×10^15 | 4.8666904272×10^15 | 8×742 | +0.0695624% | 93,613 |
| 100,000 | 7.8301416946×10^15 | 9.6344650038×10^15 | 4×1424 | +0.0083425% | 90,984 |

代理为eager Linear/attention forward/backward与checkpoint recomputation，未含elementwise、normalization与compiler fusion；同时报告实际allocation时间。预训练共享成本将区分单次部署全额和多个标签预算摊销，不把共享预训练记成免费。

已知1k Scratch/MAE后期过拟合，所以正式控制从validation NLL选定的Scratch **best**恢复完整模型与Head，fresh optimizer/schedule；原Scratch全部训练epochs的成本仍计入。此前planner提案使用last，尚未训练；在正式提交前改为best使继续监督具备已得到的最佳初始化。基准与continuation最佳点分别报告，并展示两者取较低validation NLL的监督策略结果，不将因过拟合而变差的last模型作为唯一对照。Sampler使用原监督训练epoch offset，未重复首轮sampler顺序。

分类效果目前未测量。只允许在完整训练、健康检查和相同10k validation配对统计后解释差异。

## 验证与下一步

23项S4检查此前通过，shape proxy与实际模型hooks一致；本次仅更改冻结初始化checkpoint选择、提交CPU额度与数值报告。正式提交三项控制和CPU400-replicate评价，继续执行S4。

正式提交：bigbird21、cluster2198296，2并发、6CPU/16,000MB、compact native BF16 GPU pool；EOS submission record保留配置与源码commit。新增评价虚拟策略 `__selection` 按原Scratch/continuation最低validation NLL选择，并计入两段完整计算量，raw continuation单独保留。
