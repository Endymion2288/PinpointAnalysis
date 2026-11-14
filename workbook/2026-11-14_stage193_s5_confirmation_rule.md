# S5 完整筛查与三seed确认规则

## 阶段目标

在剩余screen完成前固定候选选择、训练预算及比较范围，保证确认结果有相同预算的control。

## 工作内容

新增`analysis.flavour.s5.select`：要求十五项screen均status complete、internal health、checkpoint replay及实际encoder update通过，再按validation NLL排序。选择direct4、最低NLL候选、预定six-way假设，去重后最多三种。每种使用17/42/2026，从相应MAE3 pretrained encoder全新初始化、10k共同标签、8完整passes。seed42也重新训练，不复用4-pass的cosine终点。固定label subset seed20261113。

`evaluate`按analysis cohort或passes/labels隔离配对，8-pass确认与4-pass筛查分别记录。每seed差异、三seed均值、训练spread和400次共同物理block区间均保留。验证集已用于screen选择，确认只支持当前validation及训练稳定性结论；calibration/test继续留给S6。

## 详细数据

恢复六项epoch2 NLL：charm1.269913、kinematics1.271445、combined1.271019、uncertainty1.274132、raw+objects1.264457、vertex+kinematics1.275891。首轮至第二轮均下降。09:47:27 UTC主存45,141,520,384 bytes、peak45,341,872,128，无OOM。实际candidate ranking尚未产生。

CPU初始诊断评价cluster1581757已运行；它仅用于检查完整评价入口与图表，不替代十五项完整矩阵。新选择程序静态编译通过；正式运行时将输出`selection.json`及独立确认配置。

## 阶段性结论

确认任务的选择依据和预算已明确，尚无新结构获准晋级。训练已恢复推进，完整候选优劣等待统一的同事件比较。

## 下一步计划

完成screen后执行冻结规则；若辅助输出接近常数或误差不优于train-only constant reference，先审查该任务的目标/尺度/优化，不能将未学会的head当物理分辨率。
