# S6：真实事件读出变体验证

## 阶段目标

验证预定14种读出条件可在真实observation上执行，确认nominal identity及tracker resolution重digitize保持原baseline，准备不依赖test分数的固定paired子集。

## 工作内容

`s6/conditions.py`用family seed保持事件随机扰动对应；dead-channel图对全部事件固定。Pixel noise遍历完整native地址空间采样，bar noise覆盖96 panels×40 bars，包括原来没有hit的地址。分辨率变体从ideal crossings使用原RNG重新digitize；额外inefficiency在原接受测量上thinning，保留幸存测量的噪声。坐标bias同步作用于测量与提供给模型的geometry，单独标为坐标proxy，不解释成field/material变化。

## 详细数据

从既有validation每类取最少及最多tracker测量的两个事件，共8事件，执行14个variant，112组变换全部finite。每事件nominal与原pixel/bar/tracker数组逐元素相同；sigma scale=1从ROOT原float32 crossing得到的tracker station/xy/sigma也逐元素相同。产物`audit/conditions_smoke.json`，不含test响应或模型推理。

命令`python -m analysis.flavour.s6.conditions smoke --root <S6> --config roadmap/configs/s6_conditions.json --source <S5>`。子集规划仅读取已有test metadata，以family hash选择每类8000个事件及transport每类256个事件，并保留完整family；最终准确计数写入对应manifest，原test不改变。

首次子集写出尝试发现LCG的PyArrow不含zstd codec；改为仓库既有COMPRESSION（gzip），同时修正尚未执行的推理合并writer。失败没有打开test响应。

H10094GB申请尚Pending，调度消息包含GPU/CPU/memory不足；准备改用先前已验证可立即运行的47GB MIG flavor。55个calibration分块已转换时GPU仍未开始正式推理；不将排队当成计算进度。

子集准确计数为32000（每类8000）和1024（每类256），均未读取test response。命名空间quota仅1 GPU unit，包含pending申请；取消未启动的94GB请求后47GB MIG立即调度。首个多lane作业在启动时发现`python -m __main__`模块路径错误，4个子进程均exit1，未读取模型输入；将launcher改为绝对模块`analysis.flavour.s6.infer`后重新提交。失败日志及pool exit codes已保留，不计作完成的推理。

## 阶段性结论

14种读出变体在真实输入上可运行，nominal与tracker基值保持精确。尚未测得它们的分类影响，尚无transport/generator结果。

## 下一步计划

使用可获得的GPU执行完整calibration；构建保持输入顶点的G4响应，完成field/material及τ helicity proxy对照。校准、工作点与全部规则冻结后执行独立test和paired变体。
