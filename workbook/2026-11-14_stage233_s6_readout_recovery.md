# S6：readout完整恢复与控制组解释

## 阶段目标

完成失败Pod遗留六个任务，保留完整readout覆盖，并把native/tracker/auxiliary控制的独立配对区间写入最终表。

## 工作内容

Kubeflow MIG47GB两lane×一loader完成任务2/3/4/9，HTCondor A100两lane×一loader完成任务5/8。两backend各lane均exit0，29个readout推理任务全部齐备。已提交finalize，需逐variant核对32000事件和与完整test的nominal replay差。

最终比较新增四个原预定seed42控制：native−1mm、tracker−native、vertex+kinematics−direct4、six−direct4；从现成400个配对bootstrap读出，不重新选模型/阈值。区分这些单seed固定阶段对照和100k-label三seed MAE−Scratch比较。

## 详细数据

Kubeflow任务2：32158条条件响应、295.392秒；任务3：17696条、174.835秒；任务4：6510条、46.506秒；任务9：7000条、49.783秒。HTCondor任务5：6888条、127.782秒；任务8：6902条、130.351秒，首batch含编译。新Pod实际memory current9.35GB、peak15.54GB、limit63GB；所有lane正常结束，已保存Succeeded终态。

Common-support固定925个joint energy/x/y/z/exact-target cells，保留686693事件、Neff411249.453；主ensemble BA0.418782808、Scratch0.416704496，原始完整域差异和这个支持域分别报告。Native/tracker等对照使用不同训练预算，不概括成总体架构优劣。

## 阶段性结论

失败任务全部完成恢复，完整test结果未改变。主ensemble仍有同预算小幅收益，但τ稀有prior选择纯度不足；后续结论须包含几何/occupancy与控制组证据。

## 下一步计划

完成readout coverage/replay、14条件统计，渲染剩余英文图；取得完整资源和backend closure，执行compact cleanup并最终逐合同读回。
