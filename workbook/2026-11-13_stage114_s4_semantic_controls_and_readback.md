# S4：Rel事件信息诊断与冻结配置读回

## 阶段目标

检验bar语义解码器是否利用事件信息、量化其与常量及几何先验的关系，恢复完整分类checkpoint读回。仅用原train-inner重建probe，独立分类validation和calibration/test的边界保持原协议。

## 工作内容

CPU正式作业2198298在相同512个事件、37,567个有效bar通道和3,676种硬件channel位置上，保持queries与targets固定，循环置换完整decoder source及其mask。每模型使用原checkpoint、FP32重新计算；常量和逐channel几何均值均在同一probe拟合，作为所选先验族的描述性下界。400次accepted-family connected-chunk配对bootstrap使用44个chunk blocks，区间以固定权重、固定先验和probe样本为条件。

CPU进度评价v2（1574287）实际以exit1结束：14项checkpoint合同中只有`frozen_configuration`失败。原Scratch配置省略`program`，实际冻结manifest显式写了`program=train`；其余配置相同。评价端现只解析这一个已知缺省字段，先与冻结campaign逐项比较，再将记录的原配置交给严格readback继续核对manifest、best与last。seed、学习率和任何新增参数差异仍失败。没有修改训练checkpoint或原样本。

## 详细数据

| Rel route | Trained CE | Event-permuted CE | Permuted − trained，95% CI | Constant − trained，95% CI |
| --- | ---: | ---: | --- | --- |
| Global-only | 0.73061657 | 0.74213737 | 0.01152080 [0.00855484, 0.01439955] | −0.00673324 [−0.00866893, −0.00468705] |
| Kept local blocks | 0.72199309 | 0.73898691 | 0.01699382 [0.01201017, 0.02242097] | 0.00189024 [0.00083107, 0.00304727] |
| Kept individual channels | 0.71790266 | 0.74855685 | 0.03065419 [0.02465611, 0.03764834] | 0.00598067 [0.00424707, 0.00803701] |

常量CE为0.72388333，同probe逐channel几何均值CE为0.66291918，soft-target entropy为0.19901407。三个模型的geometry-oracle减trained的区间均在零以下，见[机器可读表](../roadmap/results/s4_rel_semantics.json)。各模型均利用事件信息；逐channel route的置换退化更大、相对常量的误差改善也更大。这些量描述各自拟合结果，尚未对route之间的差额做配对区间，不能据此声称新route已在统计上优于旧route。

实际产物根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`，完整诊断为`experiments/s4_rel_geometry_event_diagnostic_s42/semantics.json`，各模型的`bar_semantic_diagnostic.json`保留逐项结果。

等预算恢复wave已用commit`43fb563`正式提交至bigbird21.cern.ch，cluster2198680；实时schedd选择及WRITE/EOS兼容性证据保存在`logs/submission_s4-compute-recovery-b.json`。优先full EMA，随后三标签预算matched-supervised、MAE及local-Rel continuation，共6项，2并发，允许24GB H100 MIG/A100/H100/H200。其与Kubeflow共享互斥claims，最先获得资源的一方运行。

验证命令（LCG108 CUDA/Python3.12，`PYTHONPATH=$PWD/src`）：

```bash
python -m unittest tests.test_flavour_s4
```

27项通过，11.078秒。新增检查覆盖缺省program解析、实际传入readback的配置，以及拒绝seed/学习率/程序和新增参数不一致。原atomic-sync4项另已通过，见stage113。

## 阶段性结论

事实：两个新Rel route在固定train-inner probe上超过常量先验，且置换事件表示后误差上升；三者均未超过在同probe拟合的几何先验。该几何先验并非独立分类baseline，也没有用于训练模型或查询构造。下游四分类收益未测量，仍由正在运行的完整预训练及fine-tuning/frozen controls回答。

评价v2没有完成，不能引用其为统一checkpoint验收通过；v3将在正式CPU作业中重新执行。当前未支持1M扩量。

## 下一步计划

提交v3，完成全部已训练模型的14项合同、内部激活/梯度和400次bootstrap；持续检查正式训练终态、学习证据和内存。完整MAE、EMA、Rel下游及matched-supervised完成后再给最终标签效率和τ子道结论。
