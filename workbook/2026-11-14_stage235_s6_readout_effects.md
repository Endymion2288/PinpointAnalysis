# S6：全部readout效应与完整验收证据

## 阶段目标

完成14个readout条件统计、数值重放影响和最终图表证据，关闭计算后端。

## 工作内容

每条件32000固定test事件、四类各8000，主MAE三seed ensemble、四个冻结prior与名义cuts。保留400次paired block区间、三seed条件差和class migration/reliability。最终audit增加numerical replay效果与十组英文PDF/PNG的直接检查。

## 详细数据

全部14条件统计完成。Balanced nominal对照下，combined ΔBA−0.0160625、Δτ效率@1%−0.008；sparse noise ΔBA−0.01775、Δτ效率−0.00075；gain low ΔBA−0.001125、Δτ效率−0.0055；gain high ΔBA−0.00059375、Δτ效率+0.003375。各值是detector-proxy响应，paired区间和种子差均保存在`evaluation/conditions/variants/*.json`，不把设定幅度称作真实硬件误差。

同一32000事件的nominal重放：主ensemble四个prior下argmax改变数均0，两个冻结τ cuts的membership改变数均0；BA、τ效率与τ FPR的差均0。NLL重放−原分数约1.56–2.09×10⁻⁷。三单seed的概率最大差仍在原0.003容差内，单seed计数效应另保留。

全部S6 GPU作业已经终止；已保存Kubeflow终态并删除七个已结束Job。首test Pod缺少精确终态，按Job时刻给资源保守上界；其失败与较小并发恢复均计入成本。最新11项统计/校准/geometry单元验证通过。

## 阶段性结论

完整readout、真实transport和generator评价均已产出。主ensemble readout变化的选择计数没有被这次nominal重放数值残差改变；条件压力变化仍须与统计区间一起解释。

## 下一步计划

渲染并读回剩余五组英文图，汇总所有成本与backend closure，压缩native缓存后更新最终合同、路线图结果和中文最终报告。
