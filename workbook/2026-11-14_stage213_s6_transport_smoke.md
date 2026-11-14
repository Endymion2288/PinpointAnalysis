# S6：G4 接口通过与校准收尾

## 阶段目标

确认隔离 G4 的观测几何兼容当前 accepted response，并准备不截断的全 calibration 合并和温度拟合。

## 工作内容

HTCondor 单事件运行使用 public GENIE 输入，仅验证 transport 接口；不重做样本认证。新增依赖等待的 CPU 校准收尾程序：全部 227 个推理分块写完、内部 health 全部通过后，合并全部 calibration truth/observation metadata/probabilities，再逐候选拟合 temperature 与 validation 工作点。该程序不生成最终 freeze，不开放 test。

真实 transport 和 generator response 推理只用预定三主 seed，沿用 nominal 校准/阈值；每个输入、条件和 chunk 保留对应 key。原 primaries float32 动量用于重模拟，原顶点由 double metadata 写入；新 transport nominal 与条件变体形成独立的 paired 基准。

## 详细数据

G4 smoke：1 事件，输出 geometry 的全部 22 项 scalar/vector 与 accepted observation geometry 在预定数值精度内相同，`g4/smoke.json` 为 passed。HTCondor `1582710`，输入不含 test。原 build 用时 61.675 秒。当前已完成 223/227 calibration 转换及 197 个全 13 模型推理分块；全部已保存 health 通过。准确终态由后续 completion 文件统计。

命令：`python -m analysis.flavour.s6.transport smoke --root <S6>`；`python -m unittest discover -s tests -p 'test_s6_*.py'`。完整 temperature/threshold、generator 支持范围和 test 结果尚未测量。

## 阶段性结论

隔离真实 transport 接口可以继续运行；几何兼容验证通过。校准数据仍在收尾，不把提交/等待解释成完整 S6。

## 下一步计划

在 CPU 合并全 calibration 并拟合 15 个 calibration mappings。核对冻结定义和数值状态后保存 analysis freeze，提交 test、transport 与 generator 小批验证，再扩展正式处理。
