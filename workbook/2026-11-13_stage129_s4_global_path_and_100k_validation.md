# S4 global路径使用与100k监督对照

## 阶段目标

检验预训练是否让global latents在分类中发挥实际作用，并完成100k监督基线、MAE3及新增目标的进度评价。

## 工作内容与验证

正式CPU v6作业2202522（bigbird21）完成，程序217.259秒。14个分类模型均通过14项checkpoint合同；1733个训练监测samples全部健康，有效输入梯度均大于0。SiLU Head精确zero fraction最大值0，RMS范围0.116447–0.803596。33组预测进入完整10k validation，仍有128项模型/比较依赖待完成，`requested_complete=false`。代码冻结28dcf3e，配置`s4_validation_job.json`，产物为输出根`evaluation/progress_v6/`。

附加CUDA FP32路径诊断在已正式获配的47GB H100 MIG父Pod完成，程序334.689秒，四模型各约82–85秒、peak allocated均602.370MiB。全部10k事件按原batch重放Head，四个最大logit误差均为0。AMP、TF32、compilation关闭，参数固定，三个presence-preserving mappings，400次connected-chunk配对bootstrap；统计同时报告raw/common support。执行生命周期和父Pod UID见`audit/attached_inference/s4-functional-cuda-v2c.json`，device duration只计父allocation一次。

旧CPU诊断2200411实际在3615秒后被`SYSTEM_PERIODIC_REMOVE`因wall time移除，没有完整路径结果。其schedd终态虽保存`workday/MaxRuntime28800`，前次qedit没有延长运行端的有效时限。没有手动取消该已终态作业；成本保留，终态和CUDA替代关系见`audit/functional_cpu_superseded.json`。故障不归为模型学习失败。

## 详细数据

### 完整validation的拟合路径依赖

以下数值为置换后的NLL减原FP32 NLL；区间为固定checkpoint、固定三组donors条件下的recipient事件配对区间。

| Model / labels | Global ΔNLL [95% CI] | Regional ΔNLL | Local ΔNLL | Summary ΔNLL | All spatial ΔNLL |
| --- | --- | ---: | ---: | ---: | ---: |
| Scratch / 1k | 0.000427 [-0.001102, 0.001795] | 0.015868 | 0.021738 | 0.018792 | 0.048610 |
| MAE3 / 1k | 0.051899 [0.045364, 0.058185] | 0.031240 | 0.013063 | 0.004847 | 0.144440 |
| Scratch / 10k | 0.000028 [0.000011, 0.000045] | 0.041430 | 0.054670 | 0.222955 | 0.084804 |
| MAE3 / 10k | 0.035174 [0.030456, 0.040256] | 0.069094 | 0.040837 | 0.000913 | 0.352739 |

MAE global置换的macro AP变化为1k−0.044323、10k−0.036310，区间均小于0。MAE两预算的regional/local置换ΔNLL区间也均为正，说明global预训练没有使直接local路径失去作用。分支之间有相关性和交互，不能相加为贡献比例，也不当作重训ablation或物理因果重要性。

FP32与训练保存BF16预测的drift单独记录；例如Scratch1k的概率RMS变化0.0005685、argmax变化0.26%、NLL差+0.0001483。路径比较始终使用自身FP32 baseline，不替代正式BF16分类成绩。

### 标签效率与100k结果

| Labels | Scratch seed42 macro AP | MAE3 seed42 macro AP | Paired increment [95% CI] |
| ---: | ---: | ---: | --- |
| 1k | 0.297209 | 0.361792 | +0.064583 [0.057917, 0.071315] |
| 10k | 0.427100 | 0.445049 | +0.017949 [0.012548, 0.024082] |
| 100k | 0.469769 | 0.473139 | +0.003371 [-0.000117, 0.006865] |

Scratch100k三seeds的macro AP为0.46976869/0.47143515/0.46945353，均值0.47021912、sample SD0.00106483；tau AP均值0.35247252、SD0.00109737。MAE另外两个独立预训练seed的下游尚未完成，不能以一个MAE值和Scratch三seed均值替代配对。

100k seed42的MAE−Scratch NLL为−0.00624349，区间[−0.00924886,−0.00258471]；raw tau AP增量+0.00759334，[0.00096261,0.01415321]。common-support macro/tau AP增量分别+0.00025903/+0.00209488，区间均跨零。tau hadronic的raw AP增量+0.01061885，[0.00302916,0.01753086]，common-support区间跨零；e/μ子道AP增量均未确认。证据仍不足以支持kinematics-independent tau改善。

100k tau在1% FPR的通过背景均75/7500，Scratch/MAE通过信号75/89（各2500），效率3.00%/3.56%，配对效率增量区间[−0.006232,0.019122]。0.1% FPR各仅7个通过背景、4/10个通过信号，继续报告有限MC计数，不据此声称稀有工作点已改善。

10k matched-compute继续监督的raw macro AP0.40914390、NLL1.30524567，较原Scratch恶化。预先冻结的原始-or-continuation最低validation NLL选择保留原Scratch，完整继续训练成本仍计入。该选择与MAE3相比macro AP差−0.01794883，区间[−0.02408212,−0.01254786]；支持当前seed下MAE收益并非仅增加这部分监督计算即可取得。

完整channel Rel10k的macro AP0.44986819、tau AP0.33997416、NLL1.24504483。MAE5等parent/steps/views控制尚未完成，暂不将该点归因于Rel目标；Rel包含额外模拟channel语义监督。

## 阶段性结论

MAE3在低标签模型中建立了可测的global路径依赖，regional/local路径继续有效；旧S3的dead Head问题未在本轮健康记录中重现。seed42低标签验证收益成立，100k macro AP提升尚未由区间确认。tau排序与低FPR效率、原样本与common-support分别解释，后者尚不支持明确tau物理改进。

已生成并人工查看英文NLL/AP路径图及v6标签效率图，位于`evaluation/functional_seed42_v2/`与`evaluation/progress_v6/`。精简数值为`roadmap/results/s4_functional_paths_seed42.json`及`roadmap/results/s4_progress_validation_v6.json`。

## 下一步计划

完成MAE独立下游seeds、MAE5、两种Rel route、EMA及其frozen全/global probes。先通过固定分类输入缓存的正式等价与性能profile，再用于新worker，减少CPU等待；最后执行strict矩阵、物理子道、分设备成本和资源终态验收。本阶段不扩1M。
