# 阶段65：S3主模型预测塌缩的只读诊断

## 阶段目标

解释预定主模型第二轮macro AP0.25的程序与优化机制。正式17项配置、样本和训练继续冻结；诊断不替代完整screen或三seed确认。

## 工作内容

读取已有各轮validation预测，统计不同score向量数和argmax分布。新增`analysis.flavour.s3.diagnose`，只读已发布checkpoint，在每类前两个validation事件上以CPU FP32/eval mode执行一次前向和分类损失反向；记录各层范围、GELU零值比例、按模块的梯度和事件键。不执行optimizer step，不写训练checkpoint，不使用calibration/test，不重新认证输入。

该诊断使用相同冻结模型与输入代码。CPU FP32用于检查机制，不能取代正式H100 BF16性能评价。对已完成4 mm late作同事件检查，区分只读诊断本身的梯度连通性与主模型的训练状态。

最终评价入口另为每个已训练模型统一输出raw预测的不同向量数、逐类概率均值/标准差/范围和argmax分布，自动呈现其它配置与seed的预测状态。此项仅描述原始score，不修改排序、ties或任何冻结统计。已在真实late第四轮与主native第二轮的10,000事件表上分别读回9,561和1组不同向量；不以向量数单独宣称塌缩的因果来源。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

### 完整validation预测

| Model | Epoch | Events | Distinct score vectors | Argmax counts [e, mu, tau, NC] |
| --- | ---: | ---: | ---: | --- |
| 4 mm late | 1 | 10,000 | 8,683 | [1536, 6267, 1953, 244] |
| 4 mm late | 4 | 10,000 | 9,561 | [2007, 4498, 1737, 1758] |
| Fine/context/latent native | 1 | 10,000 | 13 | [440, 0, 9517, 43] |
| Fine/context/latent native | 2 | 10,000 | 1 | [10000, 0, 0, 0] |
| Fine/context/latent 1 mm | 1 | 10,000 | 336 | [759, 9149, 91, 1] |

主native第二轮NLL1.386318927、macro AP0.25。每类2,500事件；所有事件同一score时AP等于类prior0.25。Score统计原始证据为`prediction_spread_progress.json`，未对ties添加抖动。

### CPU FP32 checkpoint诊断

主native读取epoch2、40,032 optimizer updates的正式`last.pt`；4 mm late读取epoch4、80,064 updates。两次诊断均实际exit0，每类2事件，共8事件，事件键保存于JSON。

| Measurement | Primary native, epoch2 | 4 mm late, epoch4 |
| --- | ---: | ---: |
| Distinct score vectors on 8 events | 1 | 8 |
| Cross entropy | 1.386319399 | 1.142569661 |
| Head first linear minimum | −104.278465 | −18.921734 |
| Head first linear maximum | −18.343712 | 3.930080 |
| Head first linear mean | −58.610931 | −8.249426 |
| Following GELU zero fraction | 1.000000000 | 0.832682312 |
| Input-module gradient squared norm | 0 | 0.182401888 |
| Module-read gradient squared norm | 0 | 0.419109906 |
| Head gradient squared norm | 0.000012489 | 2.947325982 |

主native所有参数和前向层输出finite。Local-pool各批RMS约133–171，module read RMS3.024，最终latent global block RMS4.873；分类头GELU后8×192个元素全部精确为零。所有上游模块（local、module、context、latent）的分类损失梯度为零，head仅4个梯度元素非零，对应最后一层的bias。Late在同一8事件上各编码模块有非零梯度，观测节点数为[1204, 547, 37, 1934, 1644, 521, 135, 44]。

原始证据为`diagnostic_primary_epoch2.json/.log`与`diagnostic_late_epoch4.json/.log`。初次primary JSON使用每block活动节点统计；诊断入口随后改为每event汇总，避免冗长列表；前向和梯度算法未改。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m analysis.flavour.s3.diagnose --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12 --experiment s3_f16_context_latent_s42_e4 --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/diagnostic_primary_epoch2.json
python -m analysis.flavour.s3.diagnose --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12 --experiment s3_a4_late_s42_e4 --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/diagnostic_late_epoch4.json
```

`last.pt`会随正式训练推进，未来重跑须以输出的checkpoint epoch判断状态，不能将更后轮结果当epoch2的复现。以上均为实际读回值；首次进入塌缩的具体训练step、LR/归一化各自的因果贡献未测量。

## 阶段性结论

事实：主模型第二轮的完整validation预测已经塌缩；在8个真实响应事件上，FP32分类头GELU全零直接切断分类损失到编码器的梯度。机制在FP32也出现，不能只归为BF16概率量化。所有finite和零AMP skips本身不能证明模型保持可学习状态。

已定位的是这个训练状态中的分类头优化失败，不是native探测器信息无区分力的证据，也不能用它否定Perceiver或module-context结构。初期性能低仍不作为删除实验分支的理由。Late同事件的非零梯度支持诊断路径正常。

待检验假设：未归一化的modality/context拼接与当前优化设置可能推动分类头进入大负值区。仅凭本次checkpoint不能分离触发原因；若提出head normalization、残差缩放或平滑激活修复，必须作为新版本显式对照，不能静默修改现有screen及确认输入。

## 下一步计划

继续全部17项及预定确认，检查最终各seed的score spread、学习曲线与分类头状态。最终报告将优化塌缩与分辨率、几何、模态的物理结论分开；塌缩模型间的零差值不能作为native/1 mm等效的证据。当前不新增修复训练、不启动S4–S6。
