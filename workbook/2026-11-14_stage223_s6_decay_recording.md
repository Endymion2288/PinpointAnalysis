# S6：重模拟 decay truth 记录修复

## 阶段目标

补齐固定 transport 与 generator 对照的 τ/charm decay 证据，保留冻结条件、输入和已经完成的探测器响应。

## 工作内容

真实 G4 pilot 的16个 CC τ 事件在八条件下均读出 `resim_tau_branch=unknown`。原 `AnalysisManager::FillPythiaTree` 依赖 event trajectory container，而本轮宏关闭完整轨迹输出且没有启用 trajectory storage，导致 τ/charm 树为空。这是记录缺失，不能解释成粒子未衰变。

隔离构建增加 compact trajectory：只保留 primary τ/charm 及其直接子粒子的初始位置、动量，不累积 shower steps。沿用原 physics、随机种子、输入顶点、磁场、密度和 helicity。原 G4 源码和原隔离可执行均保留；新构建写入 `g4_truth/`。重放后逐值比较 observations、targets 和 ideal tracker，全部相同才保存独立的紧凑 truth sidecar。推理分数无需因此重算。

Geant4 v11.3.0 的 `G4TrackingManager::ProcessOneTrack` 在用户 PreTracking 后读取 `SetStoreTrajectory`，允许设置自定义 trajectory，见 https://github.com/Geant4/geant4/blob/v11.3.0/source/tracking/src/G4TrackingManager.cc 。来源副本在 EOS `literature/G4TrackingManager_v11.3.0.cc`。

另修复成本汇总：`condor_q` 没有匹配记录时可能成功退出并输出空文本，现按空数组解析。

## 详细数据

2026-11-14 约14:23 UTC：test 转换133/227块、正式推理88/227块、质量重建80/227块；transport 完成183个条件批次。HTCondor 首轮 GPU worker 两 lane 均 exit0，后续1582744已提交，Kubeflow test47-r1继续运行。实际 compact build、衰变有效数和响应相等结果尚未测量，下一次读回后另报告。

原 transport 输入1582725已在取消前完成，1024条记录已保存；`condor_rm` 返回作业不存在。没有重新读取全部输入，也没有实施stage222所述的替换取消。

## 阶段性结论

τ 全 unknown 的程序原因已定位。紧凑记录与完整响应的相容性仍需实际重放验证，当前不据 unknown 得出物理分支结论。

## 下一步计划

构建 compact executable，先重放CC e和CC τ各16事件；核对逐值响应一致、τ decay vertex与分支、primary charm和子粒子记录，再扩展其余固定事件。继续全部test与paired统计。
