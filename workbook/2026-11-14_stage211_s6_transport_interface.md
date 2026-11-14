# S6：transport 接口与几何单位修正

## 阶段目标

建立能保留输入顶点、改变真实磁场和 W 密度的隔离 G4 构建，用于预定 transport 对照；修正 S6 评价用的 τ 衰变位置单位。

## 工作内容

现有 G4 二进制会重新抽取 GFaser 顶点 z，不能用于同一事件的固定顶点比较，因此复制当前源码到 worker scratch 后构建。原源码及其已有构建不修改。新增可选环境变量保留输入 z、缩放 W 密度和设置电荷共轭的 τ helicity；磁场使用已有 `/Pinpoint/field/value` 命令。Pythia XML 路径指向匹配的 LCG107/Pythia8.313。Nominal 保留原物理设置。τ helicity 是压力范围，不是已标定的 polarization uncertainty。

`AnalysisManager.cc` 将 silicon thickness 除以 μm 写入 header，10 表示 10 μm。S5 truth helper 曾直接用该数与 mm 比较；S6 finalize 将统一使用 0.01 mm，并用实际各层的最前/最后边界定义纵向包含。修改只作用于 S6 评价标签。已验收 checkpoint 不变。

## 详细数据

`python -m unittest discover -s tests -p 'test_s6_*.py'`：5 项通过，包括温度/阈值计算和 10 μm silicon 邻近 0.1 mm 空气间隙的回归检查。首次检查发现 upstream 定义沿用了首个 pixel 中心，已改为材料最前边界后通过。`python -m analysis.flavour.s6.transport --help` 正常。

G4 源码为 `/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint`，commit `dbbdb5155a66f5329fd27ea5285d037bf8fb2f03`。构建、smoke、真实条件影响尚未测量；提交后记录在 S6 输出的 `g4/` 和 `runtime/`。全部 calibration 完成合并时测量更改标签的准确数量。

## 阶段性结论

S6 几何评价统一到 mm，单位检查通过。真实 transport 接口已实现，需 G4 编译和单事件接口验证才能用于批处理。

## 下一步计划

通过 HTCondor 构建及单事件 smoke，继续全部 calibration；冻结概率规则后开放 test。用同 primaries、同顶点运行 nominal/field/material 对照，并报告 generator Standard/NUX 的共同支持范围及非 paired 统计。
