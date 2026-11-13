# S4 遮蔽、预训练目标与优化验证入口

## 阶段目标

实现可验证的 MAE、EMA self-distillation 和 MAE+barRel，并在正式预训练前测量真实事件优化能力。验收为无泄漏回归、真实事件 CPU smoke、S3 兼容性和 GPU pilot 入口；下游收益尚未测量。

## 工作内容

`s4/dataset.py` 在 observation 层按几何 hash field 遮蔽，再重新建立全部 tokens 和统计量。包含 0.5/0.75 masks、部分整模态与 plane/panel views。Queries 从 native pixel 邻近/跨层、4-mm pixel cell、bar 邻近和全几何混合抽取；先确定位置，再查 hidden targets。每种 query 的 75% 邻近、25% uniform 几何是 proposal 起点，mask/边界拒绝会改变实际分布；实际 near/uniform 比例和各自的 positive rate 单列。Loss 是所定义查询分布的加权 risk，未进行全探测器均匀 risk 的 importance 修正。

`s4/models.py` 的 decoder 只读 global latents，强制重建目标向该路径传播；S3 的 uncapped local/region 分类路径仍保留。EMA 教师看完整硬件 observation，student 对齐 event-global representation；MAE anchor 和 VICReg variance/covariance 共同抑制常量解。本实现借鉴 Sonata 的问题分析，未复现其自然场景 augmentation 或完整网络。

S3 编码器增加 `encode` 导出接口，保留默认 forward 和所有 state-dict 参数；teacher/decoder/projector 不参与下游。Full fine-tuning 对所有策略随机重置相同的 head、branch norms 和 summary。新增 frozen linear probe 和 train-only linear/MLP branch probe 入口。

## 详细数据

执行环境：LCG 108 CUDA、Python 3.12、Torch 2.7.1。CPU smoke 用两个真实 train 事件：

| Objective | Loss | Input projection gradient squared norm |
| --- | --- | --- |
| MAE | 0.8665058016777039 | 0.0871911495923996 |
| EMA + reconstruction/variance | 3.5662624835968018 | 181.5537872314453 |
| MAE + barRel | 1.2046208381652832 | 0.10171578824520111 |

上述初始数值由各自随机初始化产生，只证明真实数据路径可执行并有非零梯度。MAE batch 产生 41 个 positive queries。无法由它们判断收敛或泛化。

1k/10k label 集合实测嵌套成立，1k 四类各 250；τ 类为 43 electron、44 muon、163 hadronic，保留原 train 比例。Pretraining-fit 为 98k，inner reconstruction-probe 为另 2k，均属于原 train partition。

7 项 S4 回归检查通过：hidden-support 改变不影响 visible inputs/queries；负候选与 uniform coverage；global decoder 梯度回到 local/latent encoder；feature API 分类数值一致；EMA 不反向传播且 variance 反对常量解；bar contribution 精确累加；checkpoint 恢复 EMA/optimizer/scheduler/RNG 后下一步完全一致。

```bash
python -m unittest discover -s tests -p 'test_flavour_s4.py'
python -m unittest discover -s tests -p 'test_flavour_s3*.py'
```

S3 既有 **49 项**检查全部通过。大 occupancy 的 GPU FP32/BF16 stress、32-event 分类/重建拟合和 pilot 泛化结果尚未测量，不能用 CPU smoke 代替。

## 阶段性结论

新目标的损失可向 global 和 local 编码器传播，且没有以 hidden support 选择 query 的接口。实际训练仍需验证重建、特征方差和下游收益。S4 数据转换 16 tasks 已退出，40 份 Rel sidecar 齐全；独立 HTCondor 汇总 task 正在处理新增标签 join/能量检查，结果由 `data_summary.json` 给出。

## 下一步计划

使用 `roadmap/configs/s4_pilot.json` 的 5 项固定配置：共同 backbone 的真实分类拟合/压力验证，以及 MAE 50%、MAE 75%、EMA、MAE+barRel 的 256-step 真实事件优化和 512-step corpus pilot。提交两后端竞争领取互斥 claims；通过后冻结正式目标与训练量，并推进各 label budget 的 probe/full fine-tuning 和 matched-compute 对照。
