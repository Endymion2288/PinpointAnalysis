# S4 最终内部健康检查与资源记录

## 阶段目标

使最终完成检查与已冻结的LP-FT训练语义一致，并核对执行中的资源和存储。

## 工作内容与详细数据

共享`accept.training_health`原先要求每次有响应采样的input梯度为正，未区分已明确配置的LP阶段。修正为只有显式`linear_probe_passes>0`时接受`intentional_encoder_freeze`，且冻结采样只能有正的linear_readout梯度，其余参数组梯度必须严格为0；后续informative FT采样仍全部要求input梯度为正，纯LP不能替代实际fine-tuning。冻结seed42目标矩阵不要求未计划的seed均值，实际/预期空集合可正确匹配。

执行`python -m unittest tests.test_flavour_s4_acceptance tests.test_flavour_s4_solver`，14项0.574秒全部通过。新增回归证明确认配置的LP+有梯度FT可接受，未配置LP、FT断梯度、只有LP均拒绝；此前的真实参数更新检查继续保留。没有改变已运行训练或选择规则。

05:14:19 UTC账本累计34.609842/60 device-hours、390.156421 CPU allocation core-hours。10个终态AFS提交目录已清理，共18,404 bytes；仅删除目录中恰为run.sh/job.sub且逐字匹配EOS冻结证据者。Active、idle、held及仅准备的提交保留，清理记录为输出根`audit/afs_terminal_cleanup_stationary_v2.json`。

05:16:26 UTC五路feature提取各完成35,061–36,275/100,000 train事件，含初始化的平均116.3–120.4 events/s/encoder。EMA3/MAE5第四epoch第4,750/5,000步，Rel5第四epoch第1,750步；各轮6,256步。完整分类与驻点读出尚未结束。

## 阶段性结论

事实：最终检查现能区分有意LP冻结与实际FT失败，不把符合配方的零梯度误判，也不允许冻结标记覆盖真正的FT断梯度。所有检查仍需真实模型与数值结果。

## 下一步计划

完成运行中的训练、特征提取和CPU求解；使用完整10k评价、磁盘readout回放、真实驻点与参数变化共同检查。随后关闭所有资源、清理临时特征并提交最终报告。
