# R3 材料单位修正与独立分支可读性

## 阶段目标

在G2正式分类训练前完成材料单位复核，保留有用的资源测量并重验受影响的学习gate。交付B1独立分支可读性，避免把非零梯度或可读性直接当作新增分支效益。

## 工作内容

源码逐字段复核发现新增材料特征误将silicon header的10当作mm。G4 `AnalysisManager.cc:708`为`GetSiliconThickness()/um`，另两种厚度为`/mm`；现有`utils/geometry.py:259`及`s6/truth_geometry.py:14`已正确区分。修复为10 μm=0.010 mm；W/scint均5 mm。阶段286厚度文字已追加明确更正。原生relations修订号从2升为3，正式G2尚未冻结/训练，原revision2的学习结果仅作诊断，不作为修订后模型的学习验收。

用有限float32最大值替换材料积分查询末尾的infinity sentinel，所有state tensors保持finite；名义物理z范围内积分不变。补充单位换算及finite-state检查。树索引优化只缓存各plane/signature的source行，16个真实Train事件与旧版本的完整edge数组和全部counts精确一致。

## 详细数据

产物根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。

修订后的control、relations及global-stack ablation真实学习gate由Condor `1583003.0`执行，配置为`probe_b2_{control,relation}_t4_s42_c5.json`及`probe_b2_global_ablation_t4_s42_c5.json`。受影响relations/ablation的结果尚未测量；control输入与结构没有使用材料特征，仍为复核矩阵的一部分。前一profile的tensor shape、计算路径、candidate图不因单位缩放改变，因此其吞吐/内存仍只作为该执行图的资源证据；正式模型另核对正确单位的学习和finite结果。

候选source-index优化在固定16个真实Train事件上总CPU时间从2.5897988 s降至2.1566215 s，下降16.73%；逐edge及counts完全相同，记录`audit/candidate_tree_optimization16.json`。这是小样本CPU实测，不外推整个Train的实际加速。

`probes/b1_reference_c1/results.json`由Condor `1583002.0`完成。冻结R2 B1 D2 T4 encoder，四个独立StandardScaler+multinomial logistic（C=1、max_iter1000），只在共同Train4k拟合，完整Tune40k评价；未用Tune选正则化。共同T3为各自T4概率的非τ条件化：

| Frozen features | Dimensions | Conditional BA3 | BA4 |
| --- | ---: | ---: | ---: |
| Local, first residual stage | 64 | 0.449833 | 0.351525 |
| Regional, second residual stage | 128 | 0.482267 | 0.370200 |
| Global, final image embedding | 520 | 0.502800 | 0.393800 |
| Independent tracker set | 128 | 0.415267 | 0.316000 |

这些是不同层冻结表示的独立线性可读性；深层同时含上游信息，不能将global−local差解释成独立global分支净贡献。原B1共同BA3为0.571267，包含非线性head和tracker融合，不能把probe上限当完整模型上限。改进后模型及原生族probe、重训ablation尚待完成。

M0/候选英文图已目视：`r3_endpoints_evaluation_m0.*`、`r3_contrasts_evaluation_m0.*`、`r3_candidate_coverage_graph_plane_policy_c2.*`；坐标、图例和标注均为英文。37项R1/R2/R3相关tests通过，包括匹配学习gate拒绝不同stride/材料设置、评价合同拒绝缺失/重复候选；修正单位后R3的12项tests再次通过。

## 阶段性结论

新增材料特征的硅单位确有程序错误，已在正式分类训练前修复并提交重新验证。候选图审计不依赖该厚度，M0/G1不使用材料特征，其已测结论保持原定义。

B1每个已测层和tracker均有可线性读出的任务信息；这没有证明新增原生边有效，也没有证明各层对最终预测的独立效益。实际训练梯度与重训结构对照继续分别测量。

## 下一步计划

读回修订单位后的五项真实学习gate和完整Tune资源profile，再冻结正式G2三模型。继续G1训练、原生训练后probe/ablation及400次配对评价。正式比较完成后按既定门槛决定seed、分辨率、anchor/fusion扩展，最终交付资源全终态和完整R3验收。
