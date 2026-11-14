# S5 全路径真实拟合与对象质量结果

## 阶段目标

在正式模型训练前验证新增路径实际学习，并报告当前对象算法的效率、假关联与覆盖。

## 工作内容

40个对象分块已合并，110k事件均保留。16个真实非空train事件（每类4个，均少于2,000通道）在交互T4上执行256步全路径拟合：six-way、vertex/semantics/charm/kinematics、object及relative同时启用。训练和健康文件在S5根`pilot/`；不是validation泛化证据。

## 详细数据

Gate通过，初始四类NLL1.730599403，最终0.000381776481；accuracy由0.3125升至1.0，55.309953 s。第32步NLL0.111548且accuracy1.0，之后保持。全部16次内部采样健康；模型继续保留FP32归一化/SiLU与独立linear路径。正式compiled BF16执行仍将逐次检查上游梯度与实际参数更新。

完整validation每类2,500事件，当前**近邻候选算法**结果如下。Pixel matching要求候选至少75%通道由同一存储trackID贡献；该truth继承pixel第一贡献者限制，不能视为精确软混合真值。

| Class | Pixel candidates | Matched | Pixel purity | Tracker candidates | Matched | Tracker purity | Tracker recovered / eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CC e | 19,503 | 398 | 0.020407 | 3,145 | 966 | 0.307154 | 966 / 3,050 |
| CC mu | 11,122 | 596 | 0.053587 | 1,958 | 1,479 | 0.755363 | 1,479 / 2,177 |
| CC tau | 8,771 | 303 | 0.034546 | 1,836 | 1,161 | 0.632353 | 1,161 / 2,039 |
| NC | 8,583 | 279 | 0.032506 | 1,601 | 894 | 0.558401 | 894 / 1,814 |

Pixel recovered/eligible依次387/130,681、566/65,557、298/52,009、270/58,717。Eligible为存储trackID在至少3个pixel planes出现，包含级联次级轨迹，不能叫primary-track效率。

Vertex candidate事件按四类为226/203/191/173；sigma-z<20 mm且DCA<0.3 mm的quality事件95/77/59/63。Primary双轨可重建事件171/385/97/153；τ双直接prong可重建31/2,500。后者不包括单prong kink。具体primary/τ vertex残差与selection coverage保存在`objects_summary.json`，完整块区间将由统一评价生成。

## 阶段性结论

新增神经网络路径通过真实小样本学习检查。当前pixel近邻关联有严重组合fake，尚不能称为成功的粒子重建；tracker关联在CC mu中较可读，在CC e中混淆更大。对象分类实验明确使用**观测候选统计**，同时保留raw路径。该实验只能评价这份候选构造的贡献，不能据此否定可重建对象路线或硬件信息。后续改进需要方向相容关联/实例学习，以及独立fake–efficiency扫描。

## 下一步计划

正式并发执行13项同预算screen与CPU候选统计基线。任何常数输出/失活/泛化失败重新审查；候选对象的低纯度作为已知算法问题分层分析，不冒充已验收粒子对象。关键神经网络候选随后三seed确认。
