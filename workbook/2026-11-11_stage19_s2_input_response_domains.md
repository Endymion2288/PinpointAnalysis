# 阶段 19：区分探测器响应域与模型输入响应域

validation 中有 158 个 tracker-only events。pixel+bar 主模型的输入空事件为 440，而三探测器全空事件为 282；两者的条件性能必须分列。此次补充评价分母，不改变训练或用户接受的响应参数。

## 阶段目标

报告 all-event、完整探测器有响应、当前模型输入有响应的条件效率，并明确 tracker-only 子样本。防止把未提供给模型的 tracker 响应计作模型已获得的信息。

## 工作内容

原 `detector_active`/`all_empty` 保留为 pixel、bar、accepted tracker 中任一响应/全部无响应，与模型配置无关。新增 `model_input_active`/`model_input_empty`：仅按该模型配置中的实际模态判断是否有通道响应。B0 的现有特征使用 pixel/bar，因此对应 pixel+bar；truth E/vertex 诊断不定义硬件输入空状态。

新增共同 `tracker_only_response` 条件域。每个模型记录 supplied modalities、input-active/input-empty 事件数，summary 保存这五种域的准确文字定义。上述域均从既有 channel counts 构建，不重验样本或 detector-proxy 参数。

## 详细数据

使用既有 `catalog.parquet` 的 train/validation 目录；class 顺序固定 e/μ/τ/NC。完整计数写入 results 根目录 `response_domains.json`。

| Split / response domain | Events | νe | νμ | ντ | NC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train / all three detectors empty | 2,253 | 543 | 471 | 686 | 553 |
| Train / pixel+bar empty | 3,545 | 883 | 755 | 1,088 | 819 |
| Train / tracker only | 1,292 | 340 | 284 | 402 | 266 |
| Validation / all three detectors empty | 282 | 56 | 62 | 90 | 74 |
| Validation / pixel+bar empty | 440 | 91 | 90 | 148 | 111 |
| Validation / tracker only | 158 | 35 | 28 | 58 | 37 |

pixel+bar 输入空域占 validation 的 4.40%，三探测器全空域占 2.82%。tracker-only 含 58/2,500=2.32% 的 validation τ，提供 tracker 后这些事件的输入响应状态会改变；实际分类收益尚未测量。

验证命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p test_flavour_s2_evaluation.py -v
```

10 tests passed，0.910 s，日志 `evaluation_unit_tests.log`。新增检查明确 tracker-only 事件对 pixel+bar 模型为空，对 +tracker 模型有响应；truth-only 参照无硬件输入响应域。

调度诊断：`condor_q -name bigbird20.cern.ch 1573747.8 -better-analyze` 当次观察 5,309 slots，0 个立即愿意执行，12 个 drained 后可匹配。现有五项科学作业仍在推进，保留队列；该排队状态不是训练失败。完整 epoch 的实际成本仍待测量。

## 阶段性结论

事实：完整 detector-active 与 model-input-active 分母不同，已测得差异及其类别组成。此前 282-event 空域结果仍正确地描述三探测器全空子集，但不能代表 pixel+bar 的全部输入空事件。新域评价待批量写出，深度模型的 native/coarse/模态增益仍未测量。

## 下一步计划

运行补充 B0 条件评价并记录 440-event 输入空域的分数行为；科学训练完成后，对所有模态控制同时报告两种 active 定义和 tracker-only 性能，形成 S2 最终解释。
