# S6：完整推理内部健康读回

## 阶段目标

把每个checkpoint、每个推理分块的内部证据汇总到最终报告，检查原S3的Head激活故障是否重现。

## 工作内容

新增 `health --root ...`，逐块核对13个checkpoint与事件覆盖，要求所有保存的TrainingHealth检查通过，汇总各层RMS范围、零比例、低于−5比例和finite状态。记录范围是每块首batch；推理阶段没有gradient测量，训练梯度与真实拟合证据仍引用已有修复训练阶段，不冒称本轮重新训练。

## 详细数据

实测test task227的MAE seed17：head_input_norm RMS0.898692、head.0 RMS3.554386、head.1 RMS1.006580、head.2 RMS0.350286；Head各层零比例0，全部finite。该块各层健康检查通过，不能单凭这个batch推断所有事件的内部行为。完整汇总待227个test块齐后执行。

约14:49 UTC：test推理223/227；transport compact truth354/368、generator213/220批。完整独立统计pilot尚未开始读分数，其依赖为全部test ready。

## 阶段性结论

已读首块没有原S3 Head全零激活故障，模型仍按冻结实现运行。完整逐块汇总、全部物理结果与资源终态尚待完成。

## 下一步计划

完成推理健康汇总、全test模型/prior统计与paired条件；保留内部范围及其采样范围，旧故障模型不参与结论。
