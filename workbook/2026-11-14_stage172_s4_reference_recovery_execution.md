# S4 严格参考恢复的正式执行

## 阶段目标

将参考梯度修复落实为可追踪的批处理，并保持完整目标比较的配置一致。

## 工作内容与详细数据

v2 CPU gate已正式提交至实时选择的bigbird21，cluster2205010，代码568c300；04:52:03 UTC在真实原train事件上完成Scratch组188/1,000个事件，仍在提取，未声明通过。独立名称和原失败结果并存。

已正式移除仅等待旧gate的五个CPU allocation：2205006/2205007/2205008及1578834/1578835；原held GPU1578832也已移除。旧完整评价2205009因冻结配置引用旧gate而单独移除。真实删除前queue ads和操作响应保存在输出根`audit/reference_gate_v2_removals.json`及`reference_gate_v2_validation_replacement.json`；随后由accounting从Condor history读取终态，未修改正在执行的训练。

新完整评价配置[s4_stationary_v2_objectives_evaluation.json](../roadmap/configs/s4_stationary_v2_objectives_evaluation.json)包含相同15训练配置、15 frozen读出配置，产生45组full/global预测，沿用原66配对。变化仅为数值回归gate依赖；新validation输出独立存放`evaluation/stationary_v2_objectives`。配置相等检查仍严格执行，避免旧冻结依赖与新manifest不匹配。新评价提交尚未执行。

执行`python -m unittest tests.test_flavour_s4 tests.test_flavour_s4_solver tests.test_flavour_s4_accounting tests.test_flavour_s4_parameter_audit`，实际48项14.315秒全部通过。没有扩大无关测试。

04:51:58 UTC原账本累计33.802762/60 device-hours、383.215053 CPU core-hours。两组GPU allocation中的三个100k模型仍运行；04:52:18 UTC最近进度：EMA3/MAE5均第三epoch第3,000/6,256步，Rel5第三epoch第500步。累计最大preclip norm分别77.628159、32.485485、34.750748；没有AMP skipped steps。这是运行中健康证据，尚无本轮100k终态结论。

联网获取[SciPy L-BFGS-B说明](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)和[sklearn1.5.2实现](https://github.com/scikit-learn/scikit-learn/blob/1.5.2/sklearn/linear_model/_logistic.py)，分别29,579和88,887 bytes，来源UTC见`literature/reference_gradient_stopping_sources.json`。最终独立梯度验收保留，即使optimizer报告成功也不替代此条件。

## 阶段性结论

事实：数值修复、正式gate和完整比较的v2依赖已落实，未放宽原目标与阈值。全部真实v2驻点及45预测完整比较仍待测。

## 下一步计划

真实gate通过后，让Kubeflow与Condor按共享claims竞争五个feature任务；GPU只提取，CPU求解各预算。完成完整评价、终态归档和临时数组清理，合并最终成本。
