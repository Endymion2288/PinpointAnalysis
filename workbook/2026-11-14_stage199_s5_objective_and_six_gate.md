# S5 匹配目标函数审查与六子类真实拟合

## 阶段目标

检查横向回归问题是否只源于评价目标不同，并核实six-way稀有输出通道的学习能力。

## 详细数据

CPU component-objective audit cluster1581776完成。旧vertex+kinematics的px/py Huber为0.099610599/0.095793407，train-mean reference为0.089045421/0.088753445；同一训练loss也更差。Px/py预测–目标相关仅0.0309708/0.0221049，pZ/E为0.539854/0.526770。六个旧kinematics配置的负encoded pZ/E计数均为0。故观察到的问题是横向任务未有效泛化，不是只换MSE指标产生的假象，也没有测到负能量输出。

完整4-pass six-way评价的9,999个已定义子类事件中，τ→e/μ的六类argmax计数为0，但各自AP为0.0722314/0.0658264，高于当前样本先验约0.0444/0.0395。六类softmax经τ三类求和与四类score最大差1.19209e-7。

为检查通道是否失活，新增六类均衡真实fit gate：CC e、CC mu、τ e、τ mu、τ had、NC各4个非空train事件，均少于2,000 channels。沿用全路径及kinematics revision2，在T4执行384 steps、73.044774 s：初始six accuracy0.166667，最终1.0；合并四类accuracy0.333333→1.0、NLL1.423353553→0.000152291。四分量MSE从[0.025939774,0.018062290,0.021581845,0.021553364]降至[0.001341081,0.000179309,0.000337615,0.000337087]，内部健康均通过。

最新gate位于`pilot/gate.json`；此前16-event kinematics gate保存在`gate_kinematics_v2.json`，coordinate gate在`gate_coordinate_v2.json`。每份对应一次真实fit，accounting将各自时长单列累加。新config为`configs/s5_pilot_six_balanced.json`。

## 工作内容

Pilot增加六类均衡选择及六类accuracy验收；统一评价增加六个输出的概率std与argmax计数，避免只看合并四类指标。修复版MSE的mean/std和预测均在既有asinh/log encoded target空间定义，映射回GeV后的bias与θ/φ residual另行报告，不将encoded均值等同于物理动量条件均值。

## 阶段性结论

Six-way所有输出通道能够拟合真实样本；验证集中稀有子类缺少hard prediction，需结合低先验和与prompt轻子的混淆解释。横向回归在匹配loss下仍未优于常数，支持继续测量新head的泛化，不能宣称已获得有效方向重建。

## 下一步计划

读取两项head修复的完整validation，继续三seed分类确认；最终报告同时保留rare-subclass概率质量、argmax计数与τ分道工作点。
