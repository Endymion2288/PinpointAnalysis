# S4：独立预训练种子确认和GPU排队证据

## 阶段目标

区分SSL初始化随机性与下游Head随机性；解释HTCondor显示Idle却不接收任务的原因，按真实可运行容量调度。

## 工作内容与详细数据

新的正式Kubeflow wave `pinpoint-s4-main-b` 已分配H100 47GB MIG（22CPU/63GB），10并发。原core Job成功完成于18:48:14 UTC；Pod被平台清理，Job completionTime与缓存Pod start用于allocation保守上界，证据保存于`audit/backends/pinpoint-s4-core-a.job.json`。

100k Scratch seed42的14项checkpoint检查全部通过，最佳epoch4，实际399,696 train exposures。三档seed42 Scratch均完成读回。

18:53:47 UTC accounting：总1.77764547 device-hours，CPU45.13532069 core-hours。本次修正了旧CPU统计只计Condor的问题，新增Kubeflow训练container实际request的CPU-hours；平台sidecars不计入该CPU列。MIG/整卡继续分别列出，不换算等价FLOPs。此前CPU值不可作完整campaign CPU总成本。

HTCondor一台看似Idle的A100 (`b9g57n0004`) 实际`InStagedDrain=true`且`ShutdownTime`已过去，所以Start拒绝新job，不能把它计为可分配GPU。18:59左右即时查询过滤drain后发现H100 NVL `b9pgpun015`可用；将据此提交额外独立预训练种子工作，而不打断Kubeflow。

`s4_mae_confirmation.json`冻结MAE3的17/2026两个独立SSL seeds，以及同seed的三档fine-tuning/global+full frozen probes。这样关键比较涵盖预训练与下游随机性，不把共享单一SSL checkpoint的不同Head seeds冒充完整三seed方法确认。额外SSL jobs使用2个stateless dataloader workers以减少I/O等待，batch/sampler/masks/optimizer与主协议一致，记录CPU成本。

## 验证与阶段性结论

新wave任务已claim，未出现重复writer。此前S3 49项、S4 19项测试通过；本阶段只补成本元数据和冻结确认配置。完整预训练收益仍需下游比较。

## 下一步计划

提交并跟踪独立MAE seeds；根据主wave语义/谱读回决定Rel/EMA扩量；完成paired event/seed统计、matched budget和总allocation终态。
