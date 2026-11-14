# S5 原始范围审查与遗漏诊断补齐

## 阶段目标

对照`roadmap/04_execution.md`第6节和`03_models_and_experiments.md`，检查完整S5交付。26模型训练与29结果统一评价已完成；本次补齐独立于训练矩阵的物理诊断证据。

## 工作内容

逐项核对发现，现有common-support只包含生成能量与真实顶点，没有包含原工单要求的target。读取已接受catalog确认存在多个target PDG，包括W同位素、C、Al及少量其他核。补充固定E/x/y/z bins与exact `tgtPDG`联合匹配；每格每类至少5事件，权重为每类count的最小值除本类count，权重上限1，支持外权重0。对现有29份预测执行400次物理block bootstrap及同一33配对，不改变模型、样本或split。

原生几何部分增加τ charged daughters在首个下游pixel面上的直线truth投影，报告相对parent外推的位移及daughter pair最小间距，并用实际pixel pitch归一化。保持材料/间隙、可见response和分道计数；所有projection仅作几何诊断，不作分类输入，不视为包含散射、磁场及shower的重建分辨率。用已有response flags连接观测可见性，沿用八passvertex+kinematics seed42的既有1%背景阈值。

## 详细数据

输入仍为110k缓存的10k validation，其中τ为2,500。新入口为`analysis.flavour.s5.scope_diagnostics`，输出写EOS `s5_2026-11-14/evaluation/scope/`。该模块只额外读取32份τ ROOT中的primaries/τ投影字段，不重新输出hit大数组。

本地轻量catalog读取已测得联合支持82格，保留3,179事件，Neff=2,886.875109，各类weight sum均562；较原能量/顶点诊断8,559事件明显缩小，后续结果限于该共同域。指标及总体几何比例等待HTCondor CPU批处理。

新增两项数值测试通过（0.014 s），核对已知target格权重、支持外排除、forward/backward/出接受度mask、已知直线间距和pitch单位。命令为LCG108环境下`PYTHONPATH=src python -m unittest tests.test_flavour_s5_scope`。真实4个validation τ事件的ROOT投影smoke正常返回，均有有效decay，首下游plane距decay为65.0484–179.2041 mm；这4例都出pixel横向接受度，保留0计数，没有作为总体结论。

## 阶段性结论

现有能量/顶点匹配不足以宣称完成target匹配。本次新增明确的联合target诊断；分类与方向学习结果保留原完整验证依据。来源认证和硬件标定沿用用户已接受值。

## 下一步计划

运行新增数值测试后冻结源码并提交CPU批作业；读取真实结果，形成原始S5要求到证据的完整映射，再更新执行入口、资源终态、最终验收及完成状态。S6 calibration/test不在本次补充范围内。
