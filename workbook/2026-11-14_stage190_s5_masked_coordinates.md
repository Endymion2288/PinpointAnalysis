# S5 坐标测量掩码与同容量对照

## 阶段目标

修复vertex-relative分支对bar未测坐标的处理，并分离额外模型容量与坐标居中的贡献。

## 工作内容

revision 2按原生node的measurement mask逐轴计算block centroid，保留三个axis-present特征；未测坐标在居中后归零。Tracker的ordinal z继续排除。absolute与relative采用同样10维输入和网络，后者只额外减去观测预测的vertex。新增回归测试固定vertex为零，任意改变bar未测x均不能改变该分支输入。

`s5_coordinates.json`冻结relative、relative+objects、kinematics-relative及两个同容量absolute控制。旧relative三项在训练前用claim标为`superseded_before_training`；旧主worker继续执行其余十项，后续新增worker使用新版源码。`s5_analysis_matrix.json`汇合十五项有效screen配置。

新增validation观测诊断CPU入口，计算pixel边缘占用、bar已测轴边缘signal、最后100 mm signal及centroid slope change。该量是观测leakage/轴变化proxy；不等同于真实containment或拟合粒子kink。辅助任务评价增加train-only constant-mean对照和输出方差。

## 详细数据

输出根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14`。revision 2全路径真实拟合使用16个非空train事件、每类4个、256 steps、Tesla T4：NLL1.633340597→0.000301073014，accuracy0.25→1.0，50.713453 s。`pilot/gate.json`及`health.json`保存现版本，v1保存在相邻`*_v1.json`。这是同事件学习检查，泛化未由此测量。

首批direct4完整10k validation：NLL1.240712302，balanced accuracy0.4139；四类概率标准差0.208470/0.135946/0.080426/0.121761。128-event selected-checkpoint replay最大差1.19209e-7；input/local-q/head权重差范数0.380958/1.142436/3.456020，内部健康检查全部通过。正式4-pass筛查与后续确认仍需完整比较，不将单seed差异当晋级结论。

验证命令：匹配LCG108 CUDA环境下执行`python -m compileall -q src/analysis/flavour/s5`及`python -m unittest tests.test_flavour_s5_model_objects tests.test_flavour_s5_truth`，7项通过；`python -m pytest -q tests/test_flavour_s5_model_objects.py tests/test_flavour_s5_truth.py`同样7项通过，16.39 s。

## 阶段性结论

坐标路径已保留不完整测量的物理边界，新增路径可在真实事件上学习。首批正式direct4产生非恒定概率且encoder实际更新；当前结果未显示S3原先的head失活问题。新增五项正式训练、观测诊断与三seed净贡献尚未测量。

## 下一步计划

冻结本次源码后启动五项coordinate screen和CPU测量诊断，读取全部screen与HGB结果，以共同预算选择关键候选进行三seed确认，再完成paired block bootstrap及困难通道分析。
