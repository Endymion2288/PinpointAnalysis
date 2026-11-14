# S4 同一路由的预训练目标复核

## 阶段目标

在修复后的分类路由上比较EMA、MAE continuation和FASERCal式partial channel Rel，区分表示可读性、实际fine-tuning表现与额外语义监督。旧故障分类指标不参与比较。

## 工作内容

冻结[首批12项配置](../roadmap/configs/s4_repaired_objectives_a.json)：三种初始化×1k/100k预算，各有full fine-tuning和full/global frozen linear probe。先使用seed42；10k待固定train-inner矩阵完整后，直接采用与Scratch/MAE3相同的共同配方和末轮，不为每个SSL目标在validation另调设置。

1k使用已冻结的LP-FT八次完整遍历、固定末轮；100k使用阶段145共同的修复后baseline训练配方及validation-NLL checkpoint。两种预算的选择方式分别记录，不将其差别归因于标签数量本身。分类Head按相同seed重新初始化，SSL decoder/teacher均不进入分类输入。所有下游均使用`empty_response_bypass=true`和匹配的真实事件健康gate。

本批复用已完成学习readback的SSL encoder，在新路由上重新训练分类器及拟合frozen probes，没有重训这些SSL任务本身。若后续出现架构或数值异常，按内部证据定位；不以epoch完成或参数finite替代学习检查。

## 详细数据

既有encoder的实测来源如下，steps与曝光为该来源任务自身，不把parent重复累计：

| 初始化 | SSL任务 | Updates | 事件曝光 | 来源关系 |
| --- | --- | ---: | ---: | --- |
| EMA3 | s4_ema_m75_e3_s42 | 4,608 | 293,168 | 独立完整EMA训练，含teacher/reconstruction |
| MAE5 | s4_mae_cont_e2_s42 | 12,288 | 196,369 | 从MAE3继续2轮 |
| Channel Rel5 | s4_mae_barrel_channels_e2_s42 | 12,288 | 196,369 | 同MAE3初始化和2轮输入，另加bar语义loss |

MAE3 parent本身为18,432 updates、294,497次曝光；MAE5和Rel5完整来源均需另外计入这个parent，累计490,866次曝光。相同额外输入不等于相同计算量；Rel额外decoder、EMA teacher及不同batch设置在计算和allocation报告中单列。

三个来源的`ssl_readback.json`均实测`integrity_and_internal_health_pass=true`及`measured_learning_pass=true`。EMA的student projection和teacher均通过非近rank-one检查；MAE continuation通过重建学习检查；channel Rel同时通过重建、语义CE改善与优于常量prior检查。完整channel Rel在原512个train-inner语义probe上，trained/permuted/geometry-prior CE为0.642756/0.797911/0.662919；geometry−trained区间[0.011855,0.029564]，事件信息诊断见[阶段119](2026-11-13_stage119_s4_full_rel_event_information.md)。这些是固定train-inner重建/语义证据，不能代替新四分类结果。

Rel目标只在预训练loss使用模拟bar通道贡献，不进入input、mask或query位置；其额外模拟监督明确标记，不能称为仅用1k或10k总标注的纯自监督对照。首批不重复较弱的旧global-only/pooled Rel分类分支，使用已有train-inner事件信息证据支持的channel route。

新12项下游配置已生成并逐项关联真实来源与readback；尚未执行，未测量新分类和frozen结果。冻结probe使用C=1、tol=1e-5、max_iter=5000，必须记录收敛。配置直接继承已验证的训练/提取接口，本阶段未新增模型实现，未新增自动化测试。Calibration/test未读取。

## 阶段性结论

事实：三个健康且具有学习证据的SSL来源已确定，新路由下的共同下游合同已冻结。MAE5与channel Rel5具有共同MAE3起点和相同额外曝光，可用于区分目标改变与单纯继续输入的收益，但额外计算仍需单列。

待检验假设：EMA或partial channel Rel可能改善MAE3未解决的τ及低标签问题；目前没有修复后这三种初始化的分类证据。

## 下一步计划

正式提交首批12项，继续以HTCondor/Kubeflow先获配者运行；当前Kubeflow命名空间配额限制为一个GPU单元，现有inner Job终态后接续。10k共同配方确定后冻结余下六项。先完成三目标seed42的完整分类、full/global frozen、内部健康与成本，再按证据决定独立SSL seeds的扩展。
