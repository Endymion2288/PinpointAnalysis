# S4：并发提交的原子写入冲突修复

## 阶段目标

使两个后端作业共用同一committed source archive时可靠发布完整文件，恢复等预算训练提交。

## 已知故障与修复

两项正式提交同时为同一个commit创建source archive，共用了固定`source.tmp`。一个提交完成rename后，另一个提交删除/rename相同临时路径失败，抛FileNotFoundError；该compute recovery未创建Condor作业、未占GPU，不产生训练结论。已定位为提交端临时inode竞争，与样本或网络架构无关。

共享`atomic_output`现使用含唯一UUID的隐藏临时路径，各调用独占inode，完整write/flush/fsync后atomic replace；原EOS transient失败的full-write retry仍保留。现有checkpoint owner claims继续保证同一模型只有一个训练writer。并发回归让两个writer同步打开文件并交错写入，要求最终文件完整等于任一全量输入且无临时残留；已有四项EOS截断/重试/永久错误/保留旧checkpoint检查继续执行。

## 详细数据

独立MAE3 seeds17、2026均已完成完整3×6144steps并通过integrity/internal和measured-learning读回；其下游比较仍待完成。先前完整MAE seed42也通过，这三份预训练各自独立初始化。

本阶段未增加样本或训练量。失败提交无allocation，实际失败说明将与成功重提record共同保留。CPU v2已成功提交cluster1574287，将补齐旧Scratch缺省program的统一checkpoint读回。

## 验证与下一步

LCG108/CUDA环境执行`python -m unittest tests.test_flavour_atomic_sync tests.test_flavour_s4`：26项S4检查全部通过；同时加载的4项atomic检查中有1项仍使用修改前的临时文件名断言。更新该断言后单独重跑`python -m unittest tests.test_flavour_atomic_sync`，4项全部通过（0.168秒）。并发source archive回归通过，重试仍能完整重放写入且保留上次已提交checkpoint。

按顺序重提等预算训练，避免同一source archive初始化并行；继续统计Rel事件条件性、独立seed下游、全部allocation终态。CPU v2因旧Scratch配置未显式写入`program`而在冻结配置一致性检查停止；实际manifest比旧配置仅多`program=train`，训练参数未改变，下一阶段修正评价端默认值解析后重新执行，当前v2不称为已完成。
