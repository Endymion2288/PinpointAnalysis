# S5 横向动量head审查与重训设计

## 阶段目标

处理auxiliary评价暴露的横向分量未学习问题，重新验证后再解释方向重建结果。

## 详细数据

旧vertex+kinematics的encoded px/py MSE为0.033530915/0.031631366，train-mean reference为0.028805437/0.028573225；pZ/E分别0.033853693/0.034229011，优于reference0.046341829/0.046310322。wrapped Δφ absolute median1.437732 rad，68% quantile2.030037 rad。横向输出尚未形成可靠的回归。该模型的分类预测/权重更新仍通过，不能把分类净贡献解释为已重建横向动量。

原loss为encoded四分量SmoothL1、beta0.1，再乘0.1。其对应的稳健location目标与MSE条件均值不同；旧head只有Linear。Shared raw representation处kinematics/main gradient norm比约0.52，乘权重后约0.052；median cosine约-0.0026，不支持强烈梯度冲突的说法。缺少分量级预条件、任务head表达力与训练不足均为待检验原因，尚不能归为物理信息缺失。

## 工作内容

联网重新阅读[GradNorm, ICML2018](https://proceedings.mlr.press/v80/chen18a.html)、[ReZero, UAI2021](https://proceedings.mlr.press/v161/bachlechner21a.html)、[FAMO, NeurIPS2023](https://arxiv.org/abs/2306.03792)，HTTP200页面和适配记录位于`literature/kinematic_repair_sources.json`。当前采用train-only target preconditioning及从constant reference开始的残差读出，没有声称实现完整GradNorm/FAMO。

Kinematics revision2用所选10k train有效标签的均值和std逐分量标准化MSE。比较同目标Linear与FP32 LayerNorm/SiLU MLP+独立Linear读出。末层零初始化，初始预测精确等于train mean；pZ/E经具有均值锚点的softplus，保持非负并在锚点匹配target std的导数。分类backbone与输入不变，旧结果只作旧配方诊断。

新增`auxiliary_audit.py`比较原始Huber/MSE、train mean/median、bias、correlation及负pZ/E计数，区分目标函数与架构问题。新增测试验证两种head的均值初始化、非负输出和可训练梯度。八项unittest全部通过，5.247 s。

真实16-event全路径gate在T4完成256步、53.211825 s：分类NLL1.599730372→0.000110387、accuracy0.25→1.0；四分量MSE从[0.018475946,0.010522908,0.029231661,0.029191413]降至[0.0000249471,0.0001375278,0.0006646081,0.0006601746]。本次gate同时要求各分量误差至少下降50%，已通过。此前coordinate revision2 gate保存在`pilot/gate_coordinate_v2.json`，最新gate保存本次kinematics revision2。

## 阶段性结论

新head在真实事件上可学习，且数值定义与component MSE/物理符号相符。validation泛化尚未测量；不把小样本拟合当方向重建成功。两项4-pass修复列为额外探索对照，预定十五项screen的candidate选择规则保持独立。

## 下一步计划

完成十五项后冻结三seed候选；将确认任务与两项head修复一起安排GPU。若整卡flavor能立即分配，依据平台实际CPU/RAM配置提高安全并发，否则继续47GB MIG。完整评价同时保留旧配方及新配方的分量误差和constant-reference差值。
