# 阶段 15：修正汇聚总量表达并保留实际计费

v1 的 mean/max 汇聚没有显式保留局部 multiplicity 和总信号。为避免这个实现遗漏混入 native/coarse-count 的比较，已停止 1573741/1573743（尚无完整科学 epoch），v2 在局部编码和全局分类头加入当前表示导出的 8 项 additive summaries。全部输入仍来自已接受的 observations，OR 对照不恢复原生计数。

## 阶段目标

保证 count 模式的不同表示在汇聚后仍可显式使用相同总 multiplicity/信号，同时保留 OR 的预定信息损失；将已停止/失败/重启的实际 GPU allocation 计入资源报告。

## 工作内容

v2 summaries 为：pixel multiplicity、bar channels、tracker channels、bar energy sum、bar PE sum、当前 token count、pixel x/y marginal multiplicity。先在线性尺度相加再 log/fixed-scale；bar 信号由现有 log 特征反变换计算，精度为 float32。局部 summaries 在 learned pooling 前连接，全局 summaries 在 event head 前连接。

joint pixel 的 multiplicity 是各 token count 之和；projection 将每个原生 pixel 计入两个 marginals，所以使用 x/y measured multiplicity 和的一半。count 模式因此在 native/100μm/200μm/1mm、joint/projection 均保持同一总量。OR 模式只对 occupied represented cells 计数，禁止读取 `raw_counts` 填回原生 multiplicity；新的合成测试明确要求 OR 的 4→1 损失仍存在。全空事件所有 summaries 为零。

v2 参数实测：ParT dim48=151,228，ParticleNet dim64=161,156，差 6.56%。12-run 训练数据、epochs、steps、种子和软件不变，独立命名为 `_v2`；配置为 `roadmap/configs/s2_spatial_screen_v2.json`。v1 文件及部分进度保留为已终止记录，其资源也进入总成本；没有使用它的未完成训练作性能比较。

评价程序增加 τ AP/ROC AUC、1%/0.1% FPR efficiency 的 paired CI，以及固定 common-support weights 条件下的 AP 差值区间。bootstrap 直接使用既有 accepted family annotations，将被同一 family 连通的 chunks 作为整体抽样；这不改变样本，也不重新认证来源。

新增 accounting 工具只查询本 S2 submission JSON 中记录的 ClusterIds，汇总 completed/failed/removed/running allocations，区分已完成 epoch history 与额外 allocation。matched-time 比较要求相同 GPU/软件/精度，并将 history 外成本保守计入首个可选 checkpoint 之前。运行中 allocation 是估计值，最终资源以 terminal history 为准；新提交同时保存精确 task/config mapping，避免后续配置修订误归属资源。

## 详细数据

| Verification | Measured result |
| --- | --- |
| Spatial/model tests | 7 通过 / 3.364 s，含 count 总量一致、OR 无原生计数泄漏、空事件 summaries 为零 |
| Sampler/resume tests | 3 通过 / 4.017 s，v2 最终权重/概率与连续训练一致 |
| Evaluation tests | 5 通过 / 0.385 s，含 existing family 跨 chunk 整体抽样、time overhead 禁用超预算 checkpoint |
| Accounting mapping test | 1 通过 / 0.003 s，部分 task index0/6 与四项 CPU queue 正确归属 |
| v2 scientific validation AP / efficiency | 未测量 |
| v2 GPU profile | 待执行，写 `resource_profile_v2/`，不覆盖 v1 数据 |

停止前的 allocation snapshot 为 0.66025 GPU-hours，含本 S2 已记录的 GPU smoke/profile 和当时运行时间；它不是最终停止时成本。`campaign_accounting.json` 后续按实际 history 更新，保留所有早期成本。

## 阶段性结论

事实：mean/max 单独不显式保存总量，v2 已将当前表示可测的总量直接接入模型。合成输入证明 count 表示间总量一致，OR 不恢复被丢失的 multiplicity。原生细布局是否提供净分类收益仍待科学训练结果；该修改不能提前证明收益。

## 下一步计划

先完成 v2 GPU smoke/profile，随后重启统一 12-run 矩阵；使用实际 allocation 与 expanded paired metrics 做主表，再完成 OR/shift/modal/architecture/domain/seed controls。用户接受的 samples、split 与 detector-proxy 参数继续直接使用。
