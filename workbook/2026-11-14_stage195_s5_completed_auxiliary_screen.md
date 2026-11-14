# S5 六项辅助筛查完成与预定控制调度

## 阶段目标

在坐标对照运行时读取已完成辅助筛查，并利用两种后端安排不依赖候选选择的控制。

## 详细数据

新增六项4-pass模型均exit0、selected-checkpoint replay及实际encoder update通过，全部best epoch4。10k train/10k validation、seed42和同标签子集。

| Variant | NLL | macro AP | Tau AP |
| --- | ---: | ---: | ---: |
| charm | 1.242678076 | 0.446428439 | 0.333164950 |
| kinematics | 1.240855174 | 0.448339396 | 0.338145662 |
| combined | 1.238886391 | 0.447773886 | 0.334439062 |
| uncertainty | 1.239353355 | 0.449313370 | 0.336473171 |
| raw+objects | 1.239773406 | 0.445543595 | 0.333556199 |
| vertex+kinematics | 1.236937037 | 0.451092826 | 0.335514819 |

包含上述十个神经网络与三个HGB的auxiliary诊断评价已提交cluster1581762，用于检查kinematic/charm输出及constant-reference对照。正式十五项screen尚有五个坐标模型，当前不进行最低NLL候选选择。

## 工作内容

预定direct4/six-way不依赖剩余screen排序，先冻结各自17/42/2026的8-pass配置`configs/s5_confirmation_controls.json`，投递HTCondor候补。最终selection产生相同名称和配置；与Kubeflow共用claims，先拿到资源者认领任务，避免重复fit。HTCondor采用两fit并发、8 CPUs/24GB主存；大输出继续写EOS。

## 阶段性结论

辅助筛查已有完整结果，vertex+kinematics当前点估计最佳；相对direct4的差异仍需配对区间及三seed。当前候选统计raw+objects未表现出明确排序收益，其算法纯度限制继续保留。

## 下一步计划

读回auxiliary误差与完整坐标对照，完成候选选择；随后让已分配的后端执行尚未认领的确认任务，减少GPU等待。
