# S5 修复head完整训练与分量评价

## 阶段目标

评价kinematics revision 2的真实泛化，给出相对于train-only常数的逐分量配对区间，并准备完整矩阵的结果收尾。

## 工作内容

两种修复head均完成四个完整passes。评价增加有效监督mask内的MSE减constant MSE，以及400次相同physical-block bootstrap的CI95；每次使用有效事件的权重和归一化。增加四分量对照图与CSV，将新head纳入coverage、真实能量及位置分层图。Six-way同步给真实分母/先验、hard prediction数及概率方差。

新增完整矩阵收尾入口`analysis.flavour.s5.finalize`与去重入口`analysis.flavour.s5.cleanup`。前者对应26个神经网络、3个HGB及完整10k validation，后者在最终科学结果生成后合并重复产物并保留selected checkpoints、预测、学习诊断及后端证据。当前未执行删除。修正accounting中交互fit次数的过时文字，按实际四份gate记录累加。

## 详细数据

| Head | 四pass训练时间 [s] | 选择epoch | 128事件readback最大概率差 | 输入参数变化L2 | local Q变化L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standardized Linear | 889.424955 | 4 | 1.78814e-7 | 0.368072 | 1.135019 |
| Standardized normalized MLP | 889.478351 | 4 | 1.19209e-7 | 0.367250 | 1.255082 |

两项全部内部采样通过，最终train classification loss分别1.211734/1.214728，standardized kinematics MSE分别0.818639/0.806840。数值在encoded target空间定义，尚未测得完整validation分量区间，不能据此判断方向重建成功。

验证命令：在LCG108_cuda环境、`PYTHONPATH=src`、OMP/BLAS各1线程下执行`python -m unittest tests.test_flavour_s5_evaluation tests.test_flavour_s5_model_objects tests.test_flavour_s5_truth`，9项通过、3.585 s。新统计测试包含被mask的大误差事件与不等block权重，独立手算验证分母和配对CI；不是只检查输出finite。

产物根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14`。模型在`experiments/s5_kinematics_standard_{linear,mlp}_n10000_s42/`。10:30 UTC，三seed确认继续在Kubeflow47GB MIG与HTCondor24GB MIG并行执行，共同claims分工；两项修复完成后空位已自动执行six-way。

## 阶段性结论

两种修复均完成有效参数学习与完整训练；泛化评价待完成。共享validation用于screen选择与确认，独立test仍按S6冻结。MIG上的nvidia-smi utilization返回N/A，不能据此给出GPU利用率百分数。

## 下一步计划

先完成修复head的完整分量评价；等待九项8-pass确认结束后统一评价29结果及三seed差值，检查真实终态、释放资源、合并重复输出并写最终报告。
