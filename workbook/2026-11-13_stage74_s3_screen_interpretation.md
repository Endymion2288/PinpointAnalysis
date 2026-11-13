# 阶段目标

在确认训练运行期间，完成17项screen的四分类解释和图示检查，区分可观测分类性能、优化失败以及尚未得到支持的探测器信息结论。

# 工作内容

从已完成评价导出版本化数值表`roadmap/results/s3_screen_summary.json`，包含17项S3、两项S2深度基线和B0，保留四类指标、raw/common-support、配对区间和原始预测分布。评价使用10,000个validation事件，每类2,500个；所有checkpoint均由完整validation NLL选择。400次bootstrap共用128个accepted-family/chunk blocks。没有重新训练screen、修改samples或使用calibration/test。

生成33页英文四分类图示和3页资源比较图。资源图按实际GPU subtype分开显示，用编号与表格对应实验，避免长名称重叠；横轴明确包含shared allocation和保守成本边界。不能从这些混合调度成本直接推出模型独立加速倍数。资源图第一页已渲染检查，文字和编号完整可读。

# 详细数据

以下为seed42的最佳NLL checkpoint，尚非三seed结果：

| Model | macro AP | macro F1 | balanced accuracy | NLL | AP [e, mu, tau, NC] |
|---|---:|---:|---:|---:|---|
| S3 fine/context/latent, pixel+bar | 0.250000 | 0.100000 | 0.2500 | 1.386298 | [0.250000, 0.250000, 0.250000, 0.250000] |
| S3 fine/context/latent, +tracker | 0.368196 | 0.357890 | 0.3596 | 1.324486 | [0.452587, 0.376319, 0.290370, 0.353509] |
| S3 4 mm late | 0.358741 | 0.334573 | 0.3460 | 1.330101 | [0.467282, 0.360682, 0.287407, 0.319593] |
| S2 ParT native joint | 0.462286 | 0.418443 | 0.4184 | 1.221079 | [0.604781, 0.460651, 0.358584, 0.425130] |
| S2 ParticleNet native joint | 0.434524 | 0.392041 | 0.3974 | 1.239306 | [0.598822, 0.414593, 0.355105, 0.369575] |

固定common-support使用8,559个有正权重事件，Neff=7,959.413646535；tracker候选macro AP=0.361130957，4 mm late=0.346885998，S2 ParT/ParticleNet为0.446711645/0.416867624。该粗energy/vertex条件化没有匹配target Z，不能称生成器或完整相空间独立结果。

部分预定配对的macro AP差值为后者减前者，区间是给定训练seed的物理事件pointwise 95%区间：

| Treatment | Delta macro AP | CI95 |
|---|---:|---|
| 4 mm late → 16-pitch late | −0.020058880 | [−0.025400125, −0.015137736] |
| Fine joint → context joint | −0.014432757 | [−0.017992908, −0.011131464] |
| Fine context joint → context latent | −0.082316853 | [−0.089566596, −0.076320669] |
| Pixel+bar context latent → +tracker | +0.118196335 | [+0.109862158, +0.127902741] |
| Four → eight module queries | +0.030445213 | [+0.026691102, +0.035132058] |

Tracker增量的对照模型本身为常数预测。两项manifest均为749,868参数，增加tracker没有增加参数量；它改变输入响应，并改变本次训练得到的优化状态。此差值证明同参数量、同训练预算下两份模型性能不同，但尚不能将tracker信息与输入引起的优化轨迹变化分解。Fine context latent的geometry/pair和模态消融也受同类优化问题影响；零差不构成无几何贡献或无bar贡献的证据。

Tracker候选的若干条件域如下，均保留该域全部事件和原始class priors：

| Domain | Events | macro AP | balanced accuracy | NLL |
|---|---:|---:|---:|---:|
| All | 10,000 | 0.368196 | 0.359600 | 1.324486 |
| Pixel-centered | 885 | 0.503675 | 0.447556 | 1.200789 |
| Large scintillator | 4,494 | 0.427976 | 0.407377 | 1.270480 |
| Tracker active | 5,630 | 0.335919 | 0.321192 | 1.338869 |
| Tracker empty | 4,370 | 0.370482 | 0.339478 | 1.305955 |
| Pixel channels >=1000 | 1,201 | 0.369647 | 0.300974 | 1.230188 |
| All detector responses empty | 282 | 0.250000 | 0.250000 | 1.373621 |

条件域之间的AP受到各自prior和难度影响，不能把tracker-empty AP高于tracker-active当作tracker有害的证据。后续条件域bootstrap将比较同一事件域内两个模型。All-empty事件始终保留；native/1 mm主模型对全10,000事件都只有一个score向量。阶段65/69的小样本FP32激活与梯度诊断支持head GELU进入全零状态的机制，但不把8个事件的诊断扩展为完整的因果干预。

产物：

- `roadmap/results/s3_screen_summary.json`：版本化screen统计。
- `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/screen_figures/four_class_diagnostics.pdf`：33页，包含四类混淆、ROC、PR、NLL与macro AP学习曲线。
- 同目录`four_class_resource_tradeoff.pdf`：3页，整卡和MIG分开呈现。
- 同目录`screen_metrics.csv`：完整四分类描述统计表。

绘图由`analysis.flavour.s3.evaluate.plot_all`读取已计算summary和预测表生成，没有重复bootstrap。`pdfinfo`验证33/3页，首张资源图经`pdftoppm`渲染并检查。修改不影响冻结训练；后续finalize复用更新后的英文绘图函数。

# 阶段性结论

目前seed42的S3候选整体四分类质量低于既有S2深度基线。细窗口、module context和latent在本轮实现与训练预算下没有给出一致收益；head优化失败使多个物理增量对照失去解释力。这些结果不证明native信息无用，也没有确立shared 4 mm compression抹平native优势的因果链。

已冻结的三seed确认仍必要：它检验当前训练配方在不同seed下的性能与塌缩稳定性，而不是重新选择有利seed。当前8项确认仍在一个MIG Job内执行，finalize进程和Pod均已核实存活。

# 下一步计划

等待8项确认完成，联合报告三seed均值、training sample SD、四类配对区间和条件域结果。逐项审计最终strict gate和图示后写最终科学报告。归一化/激活干预属于后续独立版本，本阶段不静默替换训练配方。
