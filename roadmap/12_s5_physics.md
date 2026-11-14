# S5：τ、charm、辅助任务与可重建对象

本阶段沿用已接受的110k事件、split和detector-proxy；100k train、10k validation。Calibration/test留给S6。输出根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14`。S4 revision 3是原生响应backbone，MAE3为起点。既有故障结果不进入科学对照。

## 标签与输入边界

`truth.parquet`单独按`run/chunk/row`连接。模型只接收原生pixel/bar/tracker observation或其可测量派生对象；event ID、分道、生成能量、真实顶点/动量、trackID不进入输入。Truth用于loss、matching和分层评价。未知标签保留mask，six-way缺失τ分道不填成hadronic。

G4的`FillPythiaTree`只存primary τ/charm的直接子粒子，其中也包含非衰变次级粒子。`ProdXYZ/DecayXYZ`按子轨迹最小/最大z定义；不能把`DecayXYZ`当母粒子的衰变顶点。τ使用forward neutrino共同`ProdXYZ`定位，要求0.01 mm一致性，只保留同点直接子粒子；backward或缺失锚点保留无效mask。记录与S4分道的全部差异。Primary τ track ID从PDG和primaries连接，不假定为1。

`tau_direct_prongs`是该衰变点的带电子粒子数，`tau_direct_prongs_hit`要求对应trackID在存储响应出现；pixel第一贡献者标签有已知偏差，bar为逐track沉积，tracker使用存储crossing。任何fromTau响应与直接prong命中分别报告，不能等同于可分辨kink。

Charm presence从primary PDG的charm content定义；decay tree只覆盖源码列出的9种parent，覆盖mask单列。多charm分道按μ→e→had优先级保存，同时保留e/μ multi-label flags。

Visible target为非核G4初始primary system：排除neutrinos，以可定义的直接τ衰变替换τ，不重复相加τ与daughter；charm暂保留初始parent，故此target明确命名为`primary_system_proxy`，不是所有稳定末态或接受度内能量。分别报告ΣE、Σp、|Σp|。回归仅作simulation-supervised辅助及诊断。

## 对象与训练对照

先从原生测量构造同plane邻接簇、跨plane细轨段与back-projected vertex candidates，tracker独立按测量相容度关联。所有事件保留raw fallback；对象失败产生quality mask。评价在分类前输出cluster purity、track efficiency/purity/fake、vertex efficiency/resolution与all→reconstructable→quality cutflow。Pixel truth matching继承第一贡献者限制。磁区关联不以直线拟合冒充动量测量。

首轮固定seed42与同标签预算：direct4、six-way sum、+vertex、+bar semantics、+charm、combined、object-only、raw+object，以及absolute/raw+predicted-vertex-relative。每组λ=0.1起步、有效mask归一；combined另比较FASERCal式homoscedastic uncertainty weighting。辅助任务引入的标签成本明确单列。保留候选与no-aux对照用17/42/2026；若负迁移则记录而不挑选有利单指标。

完整十五项4-pass screen完成后，以validation NLL最低者为数据选择候选，six-way另作为预定FASERCal假设保留；与direct4去重后最多三种结构。三seed确认使用新的8完整passes配置，从各seed的MAE3 encoder重新初始化，不从4-pass cosine终点接续。评价按cohort隔离，同seed/同passes控制；主指标NLL、macro AP和τ工作点同时报告。全部selection仍使用既有validation，独立test仅在S6打开。

训练记录输入/局部/融合/head激活RMS和梯度、概率方差、有效aux样本数、实际参数更新及selected-checkpoint readback。新增路径做真实小样本拟合；出现整层失活或未学习则审查和修复。分支归一化、FP32 LayerNorm/SiLU head、LayerScale、QK normalization和空响应bypass继续使用。

坐标revision 2显式保留bar的逐轴measurement mask；未测量轴在居中之后归零，tracker的ordinal z不进入这一路径。`s5_coordinates.json`包含五项重训，并以同一10维输入、相同模块和参数量比较absolute/relative；旧三项relative在启动前标为superseded。相对于raw基线增加分支的收益单独报告，不能全部归因于坐标居中。revision 2全路径16个真实train事件256步拟合达到NLL0.000301073、accuracy1.0；正式validation结论等待全矩阵。

## 方法依据与适配

- [FASERCal论文及本地汇报](02_literature.md)：六类、共享encoder、vertex/visible momentum/charm、多任务uncertainty weighting；当前pixel语义不具备精确soft mixture，先用逐track bar沉积构造event semantic fractions。
- [Kendall et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)：不同回归/分类尺度的homoscedastic权重；显式mask和物理尺度先处理，learned weight不能弥补错误标签。
- [CAGrad, NeurIPS 2021](https://arxiv.org/abs/2110.14048)、[FAMO, NeurIPS 2023](https://arxiv.org/abs/2306.03792)：冲突梯度/任务收敛速率。先测shared梯度cosine和主任务负迁移，当前少任务起点用固定小权重及uncertainty；有冲突证据再增加复杂优化器。
- [Point Transformer V3, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html)、[PointINS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Towards_Foundation_Models_for_3D_Scene_Understanding_Instance-Aware_Self-Supervised_Learning_CVPR_2026_paper.html)：保留分层局部结构和实例线索。先量化观测对象的错聚/效率，再评价融合净贡献；不把shower硬分成完美粒子。

本轮网络获取页面和状态记录于EOS的`literature/sources.json`；未实现的方法不声称已测量。

## 交付与统计

四分类与六类合并使用同事件paired block bootstrap，400次；训练seed spread另报。τ分道、衰变材料/位置、prong、可见性、charm和containment给分母、通过数、CI95；1%/0.1% background acceptance工作点与efficiency–purity同时报告。能量/真实vertex common-support是条件诊断，不能输入或用于真实推理选择。

预测/重建px、py、pT、θ、wrapped Δφ分别给偏差和68/95% quantiles，近零pT mask明确；quality-cut扫描必须同时给coverage。absolute/raw与预测vertex-relative并存，真实vertex-centred只作oracle。可见/不可见secondary、正确/错误/随机组画原生响应，score与观测量关系以总体计数支撑。结论至少包括一个有区间的困难通道解释，并列待查原因。

Containment采用观测端边缘/末端signal fraction作为leakage proxy，保留有效mask，不称作真实shower containment。辅助回归同时报告预测/目标方差及相对于仅10k所选train标签均值的误差，避免把近常数输出当成功重建。正式评价入口为`analysis.flavour.s5.evaluate`，原生响应诊断由CPU批作业`analysis.flavour.s5.diagnostics`生成。
