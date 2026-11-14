# S5 修复分量结果与方向读出诊断

## 阶段目标

核实新head的验证收益，并继续定位未形成可靠收益的py方向预测。

## 详细数据

CPU评价1582139在bigbird20完成20模型、24项分类配对及400次128-block bootstrap，121.654990 s。完整10k validation中运动学有效分母9,973，constant reference仅拟合既定10k train labels。MSE定义在asinh/log并除7的encoded空间。

| Head | px MSE | py MSE | pz MSE | E MSE |
| --- | ---: | ---: | ---: | ---: |
| Train mean reference | 0.028805437 | 0.028573225 | 0.046341829 | 0.046310322 |
| Old Linear + Huber | 0.033530915 | 0.031631366 | 0.033853693 | 0.034229011 |
| Standardized Linear | 0.028500253 | 0.028710575 | 0.030544473 | 0.030527875 |
| Standardized normalized MLP | 0.028464897 | 0.028672704 | 0.029460884 | 0.029446014 |

MLP减constant的px MSE差为−0.000340540，CI95[−0.000479664,−0.000201766]；py为+0.000099479，CI95[−0.000017839,+0.000242857]。Pz/E约改善36.4%，px约1.18%，py没有可靠改善。MLP分量相关系数为0.110869/0.059313/0.603739/0.603648，prediction std为0.024104/0.019334/0.128688/0.128625；目标std为0.169665/0.169016/0.215249/0.215175。新Linear的py差+0.000137350、CI[+0.000022312,+0.000269314]。所有新head负encoded pz/E计数为0。

MLP分类NLL1.239434145、macro AP0.449262083，不能将回归改善直接称为分类改善。Wrapped Δφ absolute median1.385622 rad、68% quantile1.936310 rad，有效pT>0.1 GeV的分母9,595；方位角仍弱。

产物：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14/evaluation/component_repair/`，包括完整summary、CSV及英文PDF/PNG。首个错误参数提交1582138在运行前移除，实际wall0；v2使用单一冻结合并矩阵，记录保留在runtime，不覆盖历史。

## 工作内容

继续审查py，新增`analysis.flavour.s5.regression_diagnostic`：对修复MLP最佳checkpoint读取完整所选train/validation的features与分量预测，核对现有validation predictions；固定encoder，用FP64 eigensolve求解带截距、train feature标准化的ridge读出。正则化候选0.001/0.01/0.1/1/10，选择仅使用train physical-family folds；encoder已见过全部所选train标签，该inner选择只是条件于该encoder的读出诊断，不能称独立encoder交叉验证。最终同时报告train/validation MSE、bias、std、correlation和paired constant差值。不保留大型feature bank。

联网查阅[EGNN, ICML2021](https://proceedings.mlr.press/v139/satorras21a.html)，用于将来检验显式vector readout的可行性，本轮未实现EGNN。LP-FT论文的OpenReview入口返回browser challenge，不能将HTTP200写成已读论文；另获取[arXiv正式条目](https://arxiv.org/abs/2202.10054)，来源状态在`literature/direction_diagnostic_sources.json`。固定encoder读出用于区分优化与表示问题，不把文献当成当前故障证据。

LCG108下`python -m unittest tests.test_flavour_s5_evaluation tests.test_flavour_s5_model_objects tests.test_flavour_s5_truth`：10项通过，3.821 s。新增收敛读出测试使用带常数特征的已知正负方向映射与独立外推点，检查截距和符号恢复。

## 阶段性结论

尺度、目标与head修复已减轻原有回归失败，px/pz/E有数据支持的constant-relative收益；尚未获得可靠py或方位角重建。是训练不足、编码信息限制还是目标可观测性限制仍需上述诊断，未给硬件信息不足的结论。

## 下一步计划

利用已分配GPU的空位执行一次固定encoder诊断；继续三seed确认。统一最终评价增加新head相对旧配方及彼此的同预算分类配对，保留探索性范围。
