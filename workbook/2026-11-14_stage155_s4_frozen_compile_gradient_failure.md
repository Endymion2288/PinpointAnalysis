# S4 LP解冻后编译编码层断梯度

## 阶段目标

定位正式1k评价发现的训练梯度故障，撤回受影响的LP配方选择及重拟合解释，修复后通过真实CUDA回归并重训。

## 工作内容

CPU评价1578768对首个新1k Scratch完成14项checkpoint读回及独立FP32 backward，随后核查实际训练健康记录时，在`informative input gradient absent`处失败。进一步逐组检查发现：LP解冻后的`local_pool`及后续层有梯度，而`input`与两个`local`块没有参数梯度。此前宽泛的健康规则将`input+local+local_pool`求和，错误地让局部断开通过。这是本轮实现和验收遗漏，之前“全部healthy”仅代表旧规则，不能等同有效full fine-tuning。

实际训练用`torch.compile`，首次局部forward发生在全部encoder参数冻结的LP阶段；解冻后编译路径继续缺少input/local参数backward边。独立checkpoint诊断关闭compile，因而能反传，无法证明真实训练更新过这些参数。现保留原始权重、记录与输出，另写[机器可读失效记录](../roadmap/results/s4_transfer_compilation_invalidation.json)。

修复：LP冻结期间局部编码使用相同eager算子，不调用编译局部函数；解冻后首次以可训练参数执行compile路径。LP验证同样不预热冻结编译图。`TransferHealth`新增独立的input/local组检查，不能再由local_pool梯度替代。OOF解析也要求每个informative full-FT采样存在正input/local梯度，避免失效候选进入选择。

新增真实CUDA回归程序`transfer_probe.py`：固定八个原train事件、相同初始化与batch，在FP32/BF16分别比较旧冻结编译路径、eager及修复编译路径；包含两次LP更新、LP eval和三次解冻更新，检查参数实际改变、真实梯度和eager/compiled logits。该正式GPU回归尚未执行，修复效果尚未在GPU测量，不能先称通过。

## 详细数据

48项inner记录回查中，错误集中于LP候选：1k有36项、10k有258项informative full-FT健康采样缺input/local梯度；其他36项non-LP候选的对应采样均有正input/local梯度。此前1,732项healthy和137,088次finite/preclip检查不足以认证full learning，这一限制应覆盖阶段154及之前的相关判断。

受影响范围：12项LP inner候选、六项1k优化refits、三个已提交1k SSL目标fine-tunes，共21项；尚未执行的部分也必须改用新合同。1k/10k此前OOF选择同时失效，因为LP竞争者未正确训练。1k NLL约1.404–1.405或MAE约1.352–1.364不作为新方案结论，需修复重训；更低NLL不能抵消断梯度。

已停止两个包含故障LP任务的正式分配：bigbird21 cluster2204965、bigbird20 cluster1578796。原claim和配置保持历史；新任务使用新名称。正在运行的10k early配方实测input/local梯度为正，可以保留训练作为候选结果，但共同配方选择需等修复LP竞争者重训后重新解析。其余100k non-LP、监督计算对照和frozen probes不依赖该冻结/解冻路径。

冻结[新inner合同](../roadmap/configs/s4_low_label_inner_v3.json)：36项non-LP配置逐字复用，12项LP使用新名、`local_execution_version=3`并要求[真实编译gate](../roadmap/configs/s4_transfer_compile_gate.json)。两个预算重新以原共同规则选择，后续refit前缀为`s4_transfer3_optimized`，避免覆盖故障训练。

实际运行`python -m unittest tests.test_flavour_s4_low_label`：10项通过，包含local_pool有梯度但input被detach时立即失败，以及OOF拒绝缺input/local梯度。真实CUDA回归和修复后LP训练尚未测量。原calibration/test及样本接受状态不变，不重做认证。

## 阶段性结论

事实：真实LP训练在解冻后局部参数未更新，独立FP32 checkpoint反传掩盖不了这一训练故障；旧LP选择及其refits失效。36项non-LP控制具有相应训练梯度，可以在更严格规则下复用。

修复假设：避免在冻结参数时建立编译局部图可以保留解冻后的参数梯度；需真实CUDA FP32/BF16回归与实际重训证据验证，不能仅靠CPU测试或权重读回。

## 下一步计划

正式运行真实CUDA gate；通过后重训12项LP inner并重新选择两个预算，再按新选定配方重拟合和完整评价。已有10k non-LP训练仅在配置和选择都相符时才可复用。补齐真实fit provenance、严格input/local梯度、内部激活和物理指标后再形成结论。
