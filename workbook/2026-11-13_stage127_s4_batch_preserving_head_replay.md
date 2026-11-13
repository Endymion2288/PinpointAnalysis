# S4 固定Head重放保留原始batch

## 阶段目标

在完整validation上保持严格FP32重放检查，并消除诊断自身改变GEMM形状产生的数值差异。

## 工作内容与详细数据

CUDA诊断已完成首个Scratch1k模型的10k事件encoder forward；随后cached Head检查失败：40,000个logits中2个超出原容差，所报最大absolute difference为2.890825e-6。该诊断把原16事件及尾batch的Head统一改成10,000事件GEMM，可能选择不同CUDA kernel。小样本相同batch的重放此前误差为0。

修复保留encoder时原始每个batch的事件位置、顺序和shape：canonical事件排序后建立逆映射，baseline及每次分支置换均按原batch执行Head，再回填同一canonical位置。保持原`rtol=1e-5, atol=2e-6`，不放宽检查。报告保存实际batch sizes。此更改只影响固定权重诊断，训练及保存的BF16验证预测不变。

新增回归用3/2/1的ragged batches、打乱的canonical位置检验GEMM调用shape、batch内顺序和最终事件回填。LCG108/Python3.12执行`python -m unittest tests.test_flavour_s4_paths tests.test_flavour_s3_paths`，7项通过，0.071秒；编译检查通过。完整CUDA重放修复结果：未测量。

Kubeflow新增整卡候选被server dry-run拒绝：namespace GPU quota最多1unit，现有47GB MIG父Pod已占用。未新建Job、未消耗额外allocation；错误记录为`logs/submission_pinpoint-s4-confirmation-whole-a.json`。现有Kubeflow与HTCondor训练继续。

21:55:41 UTC正式成本为9.451826device-hours/60，CPU allocation141.175793core-hours；MIG与整卡分列，附加推理只计父Pod一次。Scratch100k seed17已完成，best validation NLL1.21317920；新模型统一读回和配对统计待执行。

## 阶段性结论

目前发现的是诊断中更换Head批量尺寸导致的微小FP32一致性问题，尚无证据指向新的Head断连或输出饱和。必须先用原batch重放通过完整样本，再解释路径贡献。

## 下一步计划

重启冻结源码诊断，完成四个模型的严格重放及配对统计。提交共享输入small MIG候选，继续三seed与各目标对照；新完成模型集中进入下一次正式validation。
