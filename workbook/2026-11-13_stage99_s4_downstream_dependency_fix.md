# S4：下游依赖和 global-only probe

## 阶段目标

避免 Rel continuation 在其语义 pilot 完成前启动；使预训练 global 表示能在独立 validation 上直接读出。

## 工作内容与验证

`roadmap/configs/s4_mae_downstream.json` 的 Rel continuation 显式依赖完整 MAE 与 `s4p5_barrel_local_s42` 的 complete 状态及通过的学习 readback。未获资源的旧 queued campaign 将按 JobStatus==1 约束替换为合并的 `s4_main_wave_b.json`；运行中任务不改变代码或配置。

frozen probes 额外拟合 global-only linear readout，与 full readout 使用相同 train 标签预算、StandardScaler 和固定 C=1 的 LogisticRegression。它没有额外读取数据或增加 validation 选择，单独保存 `global_linear_predictions.parquet`。评价管线检查其与同一 10k validation 的 event、family 和 common-support metadata 一致，然后报告四分类指标。这直接检验 S3 的弱 global 分支能否通过预训练变得可读；full readout 的提升不能单独说明 global 改善。

1k和10k Scratch seed42 均完成，14项checkpoint读回通过；10k模型最佳NLL出现在epoch3。CPU profile正在正式HTCondor运行。当前不重复运行已通过的测试；本阶段改动是明确依赖字段和同一已有probe/metric流程的另一分支。

## 阶段性结论与下一步

完整MAE重建验收完成；分类对照仍在执行。继续完成Rel/EMA修复pilots、下游三档标签曲线、三seed和资源比较，最终明确1M扩量决定。
