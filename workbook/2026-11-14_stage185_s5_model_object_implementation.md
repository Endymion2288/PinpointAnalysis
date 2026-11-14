# S5 对象与多任务实现、全量truth提取

## 阶段目标

让S5所需对象、独立mask辅助任务、六分类合并和预测vertex-relative路径成为可运行对照。

## 工作内容

`objects.py`只用观测构造同层0.06 mm连通簇、跨plane相互最近细簇轨段、双轨back-projected vertex和三站tracker关联；所有事件保留84维测量moments、31维对象统计和失败mask。Truth matching在构造之后，统计cluster purity、track matched/fake/eligible/recovered、primary/τ vertex可重建性与quality/residual。Tracker的弯曲y容差20 mm、非弯曲x容差3.5 mm；没有声称q/p测量。阈值是当前固定算法设置，并非真实硬件标定。

`PhysicsModel`继承S4 revision3。Direct4/six-way head共享相同修复encoder；六类τ概率精确求和。新增vertex、bar语义、charm presence、visible primary-system momentum四组辅助头。固定λ=0.1和homoscedastic组合分别配置。对象分支及relative分支均先独立归一化，LayerScale融合后再LayerNorm；absolute/raw路径保留。Relative坐标用模型预测vertex，只处理有物理z的pixel/bar；tracker ordinal z不混作物理位置。

新增真实16事件拟合gate和正式训练入口；正式训练必须先通过gate。记录shared representation梯度cosine、encoder/head实际参数更新、内部健康及128事件checkpoint replay。CPU HGB提供moments/object-only/combined基线。跨后端claims避免同实验重复训练，GPU子进程共享一份本地压缩输入。

## 详细数据

6项测试通过，用时3.418 s：衰变标签3项，缺失辅助mask/六类概率和上游梯度1项，直轨与空事件fallback1项，已知交点back-projection1项。三事件synthetic backward实测：input梯度范数9.225703、relative梯度1.193943、object梯度1.318906；四项全mask auxiliary loss均精确0。该测试不代表真实拟合或泛化。

Truth v2在bigbird21的站点变换出现`PreCvmfsRequirements→PreLocalRequirements→PreAfsRequirements`循环，无机器可匹配。已在新v3提交成功后移除2206661，保留等待证据。V3在bigbird20 cluster1581684运行，160个分块Parquet和160个summary均已生成（共320文件），未看到提取异常。完整合并/对象结果尚未测量。

输出根为`results/flavour/s5_2026-11-14`。Screen配置为[11项同预算对照](../roadmap/configs/s5_screen.json)，每项10k labels、4完整passes、seed42；方法筛查用validation，test未打开。预训练语料与下游标签成本单列。

## 阶段性结论

新增mask与融合路径在数值合同测试中有梯度，原生输入仍有独立分类路径。真实数据拟合、对象纯度/效率、辅助任务负迁移和物理通道增益尚未测量；正式训练由gate约束。

## 下一步计划

正式合并truth并生成40个对象shards；先评价对象，再运行真实16事件拟合。Gate通过后，在HTCondor与Kubeflow中使用先可分配的GPU并发执行11项screen，保留关键候选三seed。
