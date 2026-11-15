# Stage292：R3正式原生任务启动与紧凑表示验证

## 阶段目标

读回stage291新增执行路径的真实验证，启动冻结G2与C1，保存模型访问终点前的配对评价合同。正式泛化结果仍以完整endpoint及相同Tune评价为准。

## 工作内容

三个G2任务都在提交后约一分钟内获得不同H100L 1g.12gb MIG。各allocation先执行实际共享mmap Train/Tune路径的continuous与中途恢复，随后stream构建100k Train与40k Tune本地缓存。C1独立完成响应映射、真实拟合、完整Tune forward资源及恢复，再提交同100k×4的正式任务。

新增正式评价合同：`contracts/geometry_evaluation.json`为6模型/12视图/48配对，`contracts/c1_evaluation.json`为2模型/4视图/8配对；每视图11域，配对含全域、pixel-centered、large-scintillator及总channels<100。endpoint为主、best-screen为辅，400次既定family-connected physical-block bootstrap，保留独立点值及核心区间复核。跨族项标为多因素模型比较，不能解释为单独几何边效应。

补充读取CoordConv官方摘要，记录题名/作者/原文及访问状态。C1只采用其显式坐标通道机制，不采用其视觉任务效果作为PINPOINT证据。更新v2执行路线及实际状态索引，旧阶段报告保留当时结果。

## 详细数据

所有路径相对`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。

| Measurement | Actual result |
| --- | --- |
| G2control1583008.0共享缓存恢复 | 参数max差0，概率/曝光表精确一致 |
| G2relations1583009.0共享缓存恢复 | 参数max差0，概率/曝光表精确一致 |
| G2global-ablation1583010.0共享缓存恢复 | 参数max差0，概率/曝光表精确一致 |
| 共享恢复fixture | 24个optimizer Train事件、8个synthetic validation事件，来自原冻结Train；不读取真实Tune |
| C1真实映射，CPU1583007.0 | 256个Train事件；1,490,680条响应index/值逐项可逆，包含最低/最高occupancy各32个事件 |
| C1 sensor网格 | 111×420；原1322×420；真实z/gaps/type显式输入 |
| C1真实T4拟合 | 32events、64steps、accuracy1.0、NLL0.000133872701554，六项健康检查全部True |
| C1资源，47gb MIG与G1共享 | 两次256update operation吞吐174.470371/170.551386 events/s |
| C1完整Tune forward | 40,000events，finite且覆盖完整，61.635732 s，GPU峰值121.929 MiB |
| C1确定性恢复 | 参数max差0、概率和曝光表精确一致 |
| C1正式提交 | 1583011.0；G2和C1均未限定GPU型号，允许12GB MIG及其他兼容SM80+卡 |
| 正式G1/G2/C1泛化收益 | 本报告形成时未测量完整终点 |
| 自动检查 | 40项tests通过，diff whitespace检查通过 |

恢复证据为`smoke/g2_{control,relations,ablation}_c9_c0/integrity.json`。C1证据为`audit/sensor_rows_c1.json`、`fit/sensor_rows_c1/B1_T4.json`、`audit/profile_sensor_rows_c1/results.json`及`smoke/sensor_rows_c1/integrity.json`。同allocation内额外gates/profile通过`audit/shared_allocation_*.json`关联，资源账本按allocation计一次。

正式使用本地file-backed数组：Train100k和Tune40k共同输入按冻结身份、配置及32个散点的实际collated tensors核对；构建峰值和耗时以正式manifest读回。尚未完成的缓存或训练不能写成模型效果。

主要命令：

```bash
python -m analysis.flavour.v2.r2.group --root "$R3_ROOT" \
  --name g2_control_c9 --configs "$R3_ROOT/configs/b2_g2_control_t4_s42.json" \
  --parallel 1 --smoke --share-r3-inputs --cache-workers 4 \
  --train-module analysis.flavour.v2.r3.train
python -m analysis.flavour.v2.r2.accounting --root "$R3_ROOT"
```

实际提交源快照/queue选择/命令均在`runtime/*.submission.json`，作业输出写results；提交文件仅在AFS home保留小型wrapper/sub。资源快照约4.02异构device-hours，仍有5项任务未终态，不作为最终成本。

## 阶段性结论

事实：共享缓存没有改变所测恢复路径；C1保留真实测量、能够学习，并已通过完整Tune的有限值、覆盖和资源检查。C1相对旧图像的CPU/GPU资源形状更小，但当前profile是在与G1共享时测得，不能以该吞吐直接作独占设备的公平速度收益。

待验证假设：缩短空行可改善泛化或资源Pareto；原生关系和global stack的作用须由正式配对、重训ablation及分支readability共同回答。真实小样本拟合本身只验证学习能力。

## 下一步计划

完成G1/G2/C1正式训练；在固定Train4k拟合local/regional/global/tracker probes并评价Tune40k；对完整结果执行400次配对和独立复核。按预定2pp共同BA或1pp τ效率及对应质量约束决定seed/分辨率/anchor/fusion。完整科学决策、成本终态、英文图与R3验收完成后再结束本阶段goal。
