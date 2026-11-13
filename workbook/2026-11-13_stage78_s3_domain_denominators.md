# S3条件域分母、类别比例与尾部统计

## 阶段目标

在完整confirmation评价前，实测七个预定条件域的四类分母，明确1%/0.1%背景接受率对应的有限样本计数。该统计只读取accepted validation，不改变samples、split、proxy、checkpoint或冻结训练。

## 工作内容

读取`experiments/s3_a4_late_s42_e4/validation_predictions.parquet`，按`run/chunk/row`与accepted catalog一对一合并，复用`analysis.flavour.s3.evaluate.domains`的域定义。每类background为本域其他三类事件，计数上限为`floor(FPR × negative_events)`。该上限与模型无关，实际选择数由模型分数及ties决定。

## 详细数据

类别顺序固定为`CC nu e / CC nu mu / CC nu tau / NC`。

| Domain | Events | Class counts [e, mu, tau, NC] | Max background at 1% FPR | Max background at 0.1% FPR |
|---|---:|---|---|---|
| all | 10000 | 2500, 2500, 2500, 2500 | 75, 75, 75, 75 | 7, 7, 7, 7 |
| pixel_centered | 885 | 245, 237, 176, 227 | 6, 6, 7, 6 | 0, 0, 0, 0 |
| large_scintillator | 4494 | 1131, 1183, 1053, 1127 | 33, 33, 34, 33 | 3, 3, 3, 3 |
| detector_active | 9718 | 2444, 2438, 2410, 2426 | 72, 72, 73, 72 | 7, 7, 7, 7 |
| all_empty | 282 | 56, 62, 90, 74 | 2, 2, 1, 2 | 0, 0, 0, 0 |
| pixel_ge_1000 | 1201 | 526, 219, 247, 209 | 6, 9, 9, 9 | 0, 0, 0, 0 |
| tracker_active | 5630 | 1359, 1646, 1497, 1128 | 42, 39, 41, 45 | 4, 3, 4, 4 |
| tracker_empty | 4370 | 1141, 854, 1003, 1372 | 32, 35, 33, 29 | 3, 3, 3, 2 |

高占用域的e类prior为43.7968%，完整域为25%；tracker-active的mu类prior为29.2362%，tracker-empty的NC类prior为31.3959%；全空域tau类prior为31.9149%。这些比例均直接由上表计数计算。

输出为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/validation_domain_denominators.json`，包含原始计数、priors和两种FPR上限。实际执行环境为LCG108/Python3.12，CPU单线程，命令exit0。复现计算核心如下：

```python
frame = pd.read_parquet(root / 'experiments/s3_a4_late_s42_e4/validation_predictions.parquet')
catalog = pd.read_parquet(root / 'catalog.parquet',
    columns=KEY + ['all_empty', 'pixel_channels', 'bar_channels', 'tracker_channels'])
frame = frame.merge(catalog, on=KEY, validate='one_to_one')
masks = domains(frame, {'modalities': ['pixel', 'bar']}, True)
counts = frame.loc[masks[domain], 'label'].value_counts().reindex(range(4), fill_value=0)
negative = int(masks[domain].sum()) - counts
ceiling = (negative * acceptance).astype(int)
```

## 阶段性结论

事实：所有预定域均有四类事件；0.1% FPR处即使完整validation也至多选择7个背景，所有域均低于当前25个背景的尾部充分性标记。pixel-centered、高占用及全空域该阈值只允许0个背景，经验效率存在明显离散性。1% FPR处的pixel-centered、高占用与全空域同样达不到25个背景。

不同域的AP同时受到类别prior、物理接受度和难度影响。需要使用同域、同事件的冻结模型配对评价；域间AP高低不能替代tracker或局部编码的因果消融。零背景选择不代表真实背景率为零，当前计数也不支持部署级尾部精度或物理flux产额。

## 下一步计划

完成当前四个live confirmation训练，读取最终条件域的实际signal/background选择数、purity和固定inclusive threshold迁移；将本表与模型配对区间一起纳入最终科学报告。S3严格验收、终态成本和资源释放仍等待完整训练终态。
