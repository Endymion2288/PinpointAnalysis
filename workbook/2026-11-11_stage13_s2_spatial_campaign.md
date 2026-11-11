# 阶段 13：V100 资源曲线与空间对照冻结

24 个占用分层事件 × 6 个表示 × 2 个模型族的完整 batch-1 资源测量已结束，共 576 行 inference/training 记录，loss 均有限。基于该测量和已通过的 GPU smoke，冻结 80k train/10k validation 的 12-run 空间筛选矩阵；先启动两族 native joint，观察实际 batch throughput，再提交其余已冻结配置。

## 阶段目标

在完整输入和明确预算下比较 native/1mm 与 joint/projection，并补 100/200 μm 中间点，分开判断粗化、关联损失与模型族。

## 工作内容

1573738 在 Tesla V100-PCIE-32GB 正常完成，软件为 LCG_107、Python 3.11.9、Torch 2.5.1、FP16。按总激活通道数的等间隔秩取 24 个 train events，包括空事件和最大总占用；这是压力曲线，未用来无偏估计总体平均资源。固定观测经六种表示后输入全部节点/块，没有 event cap。

`resource_profile/measurements.parquet` 保存逐事件转换/传输、forward/backward、显存、表示及事件 keys；`resource_curves.png/pdf` 已目视核对。首个样本有 CUDA warmup，最大耗时含该开销。

`roadmap/configs/s2_spatial_screen.json` 冻结两族各六个表示：native joint/projection、1mm joint/projection、100μm joint、200μm joint。所有实验 seed42、pixel+bar、count、固定 grid origin、全部目标域。ParT dim48=150,076 参数，ParticleNet dim64=159,620 参数，差 6.36%，处于预先设定的 10% 容量匹配范围；报告实际参数而不声称完全相等。

每配置 4 epochs、batch4、2 data workers、2-shard scratch cache、bucket256、AdamW lr=0.0005/wd=0.01/cosine、validation NLL 选 best。每 epoch 固定 20,016 attempted steps，共 80,064；该上限保证完整 80k 事件至少遍历一次，再确定性重复少量 batches 以统一步数。validation 每 epoch 全部 10k，calibration/test 保持关闭。记录实际 exposures、applied optimizer steps 和 AMP skipped steps，后续报告同时披露 attempted/applied 预算。

初轮空间筛选仍按约 100 GPU-hours 总预算组织。batch-1 压力测量未包含 optimizer/GradScaler，不能直接换算完整训练吞吐；两个 native 配置先给出每 250 steps 的真实 batch 进度，依据这些实测检查预算，再继续十个已冻结配置。后续关键三种子确认与架构/模态/训练域对照独立记录资源。

## 详细数据

下表为 24-event batch-1 median；training time 是 forward+backward，未含 optimizer 和完整 loader。显存为该模式 peak allocated memory。

| Family / representation | Inference median [s] | Training median [s] | Training median / max memory [MiB] |
| --- | ---: | ---: | ---: |
| ParT native joint | 0.01888 | 0.06936 | 73.82 / 204.22 |
| ParT native projection | 0.01897 | 0.07082 | 75.93 / 160.88 |
| ParT 100μm joint | 0.01868 | 0.06598 | 73.82 / 186.12 |
| ParT 200μm joint | 0.01857 | 0.06666 | 73.82 / 172.44 |
| ParT 1mm joint | 0.01911 | 0.06828 | 73.82 / 161.00 |
| ParT 1mm projection | 0.01845 | 0.06547 | 75.93 / 160.84 |
| ParticleNet native joint | 0.01832 | 0.04486 | 230.84 / 549.65 |
| ParticleNet native projection | 0.01744 | 0.04354 | 234.90 / 545.56 |
| ParticleNet 100μm joint | 0.01693 | 0.04144 | 230.84 / 549.61 |
| ParticleNet 200μm joint | 0.01718 | 0.04181 | 230.84 / 548.47 |
| ParticleNet 1mm joint | 0.01736 | 0.04205 | 230.84 / 547.53 |
| ParticleNet 1mm projection | 0.01710 | 0.04221 | 234.90 / 545.07 |

同一压力样本最大 blocks：native joint 2,641、100μm 2,425、200μm 2,225、1mm 1,830；native projection 998、1mm projection 679。这不是完整目录最大值；阶段 09 更大抽样已见 3,043 blocks。ParT native 最大训练时间 7.441 s 包含初次 CUDA warmup，不能直接当作 occupancy tail 的稳定延迟。

B0 错误组原生显示 1573740 完成，20 页 `evaluation/event_displays/b0_all_s42/native_error_groups.pdf`、`examples.csv` 和首例 PNG 已生成，首例已目视检查；之后补充空面板文字和 tracker station legend。显示只读 observations，truth E/vertex 在标题独立标注。每 confusion pair 的高置信例子和固定随机参照均保留，不以个例替代总体统计。

```bash
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS" \
  --campaign "$PWD/roadmap/configs/s2_spatial_screen.json" --tasks 0 6 \
  --lcg-view LCG_107_cuda --gpu-device 'Tesla V100-PCIE-32GB'
# 上述真实吞吐确认预算后，继续尚未提交的十项：
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS" \
  --campaign "$PWD/roadmap/configs/s2_spatial_screen.json" \
  --tasks 1 2 3 4 5 7 8 9 10 11 --lcg-view LCG_107_cuda \
  --gpu-device 'Tesla V100-PCIE-32GB'
```

## 阶段性结论

事实：当前两族实现能完整处理压力样本的所有节点和块，显存被 query tiling/checkpoint 控制在此测量范围内。粗化/projection 减少部分高占用事件的全局块数，median 资源差别小；它们的分类代价尚未测量。

## 下一步计划

完成冻结空间矩阵的真实训练，按配对 validation 概率给出主效应/interaction、matched GPU-time、共同支持及 τ 分支。根据结果选择进一步 controls 和三种子确认，不将 GPU smoke 或资源曲线当作分类验收。
