# S6：可读取生成器对照与固定评价补充

## 阶段目标

量测可用生成器共同支持，固定四类/τ/charm 对照的输入和统计方式，并补齐条件差值与 score/energy migration 输出。

## 工作内容

对 public GENIE3.04 Standard/NUX、三种 flux source 各前 20 个编号文件做有界扫描。共同支持 cell 为 class、flux source、signed incoming PDG、W isotope 和固定 energy bin；两套各 cell 的抽样数严格相同，并使用相同 benchmark 顶点 slots。加入独立 source-file block bootstrap 的 generator 对照，及同事件条件变体的 paired bootstrap。冻结的场景 prior、temperature 和 validation cuts 在所有对照中保持。

补充 selected Neff、target-envelope/active/selected cutflow 和 W z-slab 描述。Target envelope 使用 accepted geometry 的大 scintillator 横向范围与 W 纵向范围；它是 truth 条件评价，不能作为推理输入，也不称精确 W corner survey。Score/energy response 保存原始计数；S5 辅助能量输出另查非物理值，并明确它回归的是 visible-primary proxy。Exposure、response inversion 和 unfolding 尚未测量。

## 详细数据

扫描 120 个 public 输入文件，运行 115.35 秒；共选择 3,512 个事件。每套：CC e512、CC μ512、CC τ220、NC512。τ 共同支持少于预定上限512，因此保留全部220，未补造事件。准确 cell counts 在 `generator/support_cells.parquet`，输入路径/类别数在 `generator/inventory.json`。NuWro compatible input 未找到；私人目录权限拒绝，不继续递归搜索。

真实旧 GFaser 1 个事件经 PyROOT bridge 写入 current schema，uproot 读回 vector/int/double/bool 类型和 incoming PDG 正确。9 项 S6 自动化测试通过。G4 smoke 22 个几何字段全部通过。生成器 response 尚未测量。

Calibration 已有226/227转换和推理；剩余 task129 的原 job `1582697.128` 运行约1810秒、记录 CPU1秒、log0 bytes，没有 shard 输出。取消并以相同事件、配置重提为 `1582715`；根因未定位，不归因于物理事件。校准 finalizer 等待该块，没有生成 freeze 或打开 test。

## 阶段性结论

可执行同 W target、四类覆盖的 Standard/NUX hadronization 对照。它没有覆盖全部 generator 理论差异，也不是 NuWro 比较。当前只有输入与接口证据，尚无分类影响结论。

## 下一步计划

先运行 generator16事件真实 transport pilot，再按测得资源扩展；等待 calibration tail，完成全量拟合与 freeze，然后执行独立 test、读出与真实 field/material 对照。
