# S6：冻结评价、paired 变体和生成器桥接

## 阶段目标

实现独立 test 的固定评价，避免把 test 曲线上的最佳工作点当成预定阈值；建立 Standard/NUX 输入到当前 G4 的桥接，以及共享输入的 paired 读出变体推理。

## 工作内容

新增 `s6_evaluation.json` 固定 reliability、energy、occupancy、vertex 和 migration bins。评价使用已冻结校准和 validation 阈值，分别保存四类原始整数计数、prior 加权指标、reject/conflict、多条件子集、cutflow、完整 tie-aware ROC/PR、exact binomial 区间以及 400 次 physical-block bootstrap。Bootstrap 保存小型 block multiplicity 矩阵，并在每个 replicate 内按 true class 归一到指定 prior；所有模型复用同一设计，差值按 replicate 配对。训练 seed spread 单列。

新增观测质量计算，复用 S5 的 cluster/segment/vertex 和 tracker association。响应尾层/横向边缘活动只称 leakage indicators；不称为已测真能量包含。新增 14 种读出条件的共享 native 数据及三主 seed 推理，nominal replay 再与正式分数比较。激活 health 不通过将保存异常信息并终止该任务，不能写为成功。

Standard/NUX 对照按 signed flavour、flux source、energy bin 和 W isotope 的共同 cell 精确匹配事件数；同 cell 赋予相同 PINPOINT calibration benchmark 顶点。原 generator 事件独立，不按 event index 配对。旧 GFaser 的 incoming/target/primary lepton ancestry 用于补 current schema；缺少的 scattering mode 和 cross section 保持 unknown，不用于权重或模型输入。真实 transport 逐小批写暂存 ROOT，转换为 compact observation 后删除 ROOT。

## 详细数据

`python -m unittest discover -s tests -p 'test_s6_*.py'`：9 项通过，新增统计检查覆盖 resample prior、冻结 selection/reject、零背景 exact 区间和旧 GFaser 的 ancestry/单位。Calibration 的 2 个真实事件执行 14 variants，共 28 个 tokenization/collate 全部 finite；`audit/variant_dataset_smoke.json` 保存结果，未读取 test。

隔离 G4 构建通过，HTCondor `1582709`，61.675 秒；`g4/build.json`、`g4/build.log` 保存环境、源码和 patch。Generator 批量扫描、transport 批量运行和 test 评价尚未测量。旧 GFaser 的 ragged arrays 不支持直接 `np.isin`，实现已改成 Awkward 逐元素条件。

## 阶段性结论

冻结统计及输入桥接已经实现并通过小样本/数学检查；目前没有 test 分类结果或 generator/transport 差异结论。Calibration 仍在转换和推理，尚未生成最终 freeze。

## 下一步计划

提交 G4 单事件验证和有界 generator 扫描。完成 calibration、temperature 和 validation thresholds 后保存 freeze，开放 test。先量测完整评价与质量计算的单任务资源，再批量提交，完成后逐项核对 S6 合同。
