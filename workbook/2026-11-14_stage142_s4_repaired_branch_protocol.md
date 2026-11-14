# 修复模型的global与局部路径重新诊断

## 阶段目标

检查修复后的global/regional/local/summary是否在最终分类中被利用，避免把总体encoder连通当作每条路径均充分学习。

## 工作内容与详细数据

冻结`s4_empty_functional.json`：对新Scratch/MAE3的1k/10k seed42四个最终模型，在完整10k validation做固定权重路径置换，三个presence保留的映射seed为2026/17/42。使用CUDA FP32、关闭compile和TF32，每个模型先做完整14项checkpoint读回和独立8事件best FP32 backward。各分支置换共用同一事件donor；Head重放保留原batch形状及事件位置，与完整FP32基线误差须≤2×10⁻⁶。沿用400次物理块bootstrap及raw/common-support指标，不更新参数或选择阈值。

新诊断为`functional_probe`正式campaign任务，加入独立dispatch入口`s4_empty_dispatch.json`的第18号任务，依赖上述四个训练完成；前18项训练配置逐字保持原冻结矩阵。它只在正式GPU batch入口启动，不通过SSH维持长期进程。`accept_empty`新增完整路径证据及真实四类工作点计数要求。

旧S4的数值依然保留为历史观测；roadmap第5/6节现明确说明旧排序收益和分支推论须由修复矩阵复核。当前100k三个seed首轮NLL的Scratch/MAE3分别为1.253265/1.239436、1.253160/1.240683、1.262912/1.258699；τ AP方向不一致。该中途观察不决定新配置或后续checkpoint。

正式新路径诊断结果未测量。此处是固定权重的依赖性检查，不等同于分支重训消融或物理因果重要性。

## 阶段性结论

最终验收将同时核对总体数值健康与完整validation的分支依赖。旧故障初始化对应的方法结论已暂停，不能由旧路径图或有限梯度绕过这一复核。

## 下一步计划

使用仍排队的GPU候选运行更新dispatch，四个依赖模型完成后自动开始路径诊断；其余18模型继续原配置训练。完成后与严格成对评价、GPU终态和成本一起报告。
