# S4 修复模型、预训练对照与完整评价

本轮S4完成：五种初始化×三个标签预算的15项分类训练、30个驻点frozen读出、45套完整validation预测及66项预定配对全部完成。20项收尾检查通过，114条allocation全部终止。MAE3保留为后续初始化基线；EMA3未获得相同收益，继续MAE和channel Rel没有建立跨预算一致的额外排序收益。本轮不扩至1M。

## 阶段目标

回答修复后模型是否实际学习并保持梯度连通、FASERCal式预训练是否提供可读事件信息、收益能否转化为四分类与τ工作点质量。旧S3失活模型、旧空响应异常与LP编译断梯度模型继续只作故障诊断，不进入本报告结果。

## 架构、训练与执行修复

S3 revision 2已确认旧Head的整层数值失活，修复重训与20模型评价见[阶段87](2026-11-13_stage87_s3_repair_final_results.md)。本轮共同backbone保留Peri-LN、QK normalization、LayerScale、各分支及concat归一化、FP32 Linear→LayerNorm→SiLU Head和独立linear logits路径。论文与适配依据见[修复路线图](../roadmap/10_s3_repair.md)，包括ICML 2025 Peri-LN、ICCV 2021 LayerScale、ICML 2023 QK normalization及CVPR 2024 PTv3；预训练主要参考FASERCal、CVPR 2025 NOMAE/Sonata和VICReg，见[S4方法合同](../roadmap/11_s4_pretraining.md)。替换激活函数只是组合修复的一部分。

局部窗口内保留原生通道位置残差，所有local blocks均参与多尺度读出，并有独立regional与global路径。最大已测train事件为36,549个原生节点、14,599个local blocks、962个regional tokens和64个global latents。分类不再仅经过固定少量module/latent queries；池化仍会压缩信息，不能称为无损编码。

S4 revision 3对真实空响应使用显式bypass，修复零占位节点的异常反向尺度；共同Scratch对照也采用相同路由。LP-FT在解冻时清除冻结状态下生成的编译图，并检查input/local参数实际更新。修复后的train-inner选择、三seed低标签结果见[阶段167](2026-11-14_stage167_s4_valid_low_label_results.md)，完整10k固定权重分支检查见[阶段170](2026-11-14_stage170_s4_valid_path_results.md)。这里没有复用已作废的LP checkpoint。

固定C=1的100k高维读出曾触及L-BFGS迭代上限。恢复先通过同目标CPU Newton-CG/严格L-BFGS参考gate；为缩短实际耗时，再实现FP64设备gradient/Hessian-vector products，沿用sklearn 1.5.2 Newton-CG、Wolfe line search、200 outer/200 inner上限。8组真实同feature CPU/GPU对照全部通过，最大目标差2.5315×10⁻⁹、最大概率差1.5209×10⁻⁴；双方独立CPU梯度满足原1e-6要求，没有放宽阈值。5项100k full解分别用79/99/86/97/100次outer迭代，fit加CPU证书耗时134.35–163.63s。见[真实gate](../roadmap/results/s4_fp64_acceleration_gate.json)及[阶段180](2026-11-14_stage180_s4_actual_fp64_gate_pass.md)。

加速仅改变数值执行方式，目标仍为mean multinomial NLL加`||W||²/(2 C n)`、C=1，intercept不正则，scaler只fit当前标签预算内事件。1k/10k原CPU结果复用；100k使用独立实验名和冻结配置，未改写旧任务。替代解全部通过后，正式停止原五组耗时CPU任务及旧等待评价，保留其全部成本与终止证据。

## 样本、配置与统计口径

输出根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`，下文记为`ROOT`。沿用用户接受的split与detector-proxy，不重新认证样本或要求硬件标定。分类输入为原生pixel、合并后的bar通道和proxy tracker硬件观测。监督标签预算为嵌套1k/10k/100k；独立于train的固定validation共10k，每类2,500个事件。

本轮五初始化比较仅使用seed42。1k/10k使用修正后、在原标签预算内按物理family分组的train-inner共同配方；100k沿用validation NLL选择，五项best均为epoch4。MAE3是3个预训练epochs；MAE5从MAE3继续2 epochs；Channel Rel5与MAE5具有相同continuation曝光，但另外使用模拟bar channel语义监督，不能称纯自监督或相同总标注成本。四份SSL来源均通过健康及实际学习readback。

冻结入口为[30配置](../roadmap/configs/s4_fp64_objectives_evaluation.json)、[66配对](../roadmap/configs/s4_fp64_objectives_pairs.json)和[正式CPU评价](../roadmap/configs/s4_fp64_objectives_validation.json)。45套预测包括15项fine-tune、15项frozen full和15项frozen global；本矩阵没有预定seed-mean项目。此前有效的三seed结论保存在阶段167，与本单seed对照分开。

每项配对使用相同的400次、128个accepted-family/chunk物理块重采样，raw与common-support共用block multiplicities。Common-support在能量与顶点联合分箱后保留8,559事件，有效样本量7,959.414，四类权重和均为1,586；它是条件诊断，不是flux加权物理产额。区间为pointwise CI95，条件于训练权重、已复用的validation和设计；不包含训练seed不确定度或多重比较修正。设计已观察前期validation，不能解释成新的冻结test泛化证据。

## 分类结果

下表全部为本轮seed42、完整10k nominal validation，AP使用非插值one-vs-rest定义。

| Labels | Initialization | NLL | Macro AP | Tau AP |
| ---: | --- | ---: | ---: | ---: |
| 1,000 | Scratch | 1.407127 | 0.306829 | 0.274940 |
| 1,000 | MAE3 | 1.351380 | 0.339867 | 0.297408 |
| 1,000 | EMA3 | 1.376491 | 0.311661 | 0.270570 |
| 1,000 | MAE5 | 1.351196 | 0.339311 | 0.294451 |
| 1,000 | Channel Rel5 | 1.346140 | 0.344380 | 0.290537 |
| 10,000 | Scratch | 1.272804 | 0.423359 | 0.318489 |
| 10,000 | MAE3 | 1.236319 | 0.449259 | 0.338947 |
| 10,000 | EMA3 | 1.271015 | 0.421124 | 0.317642 |
| 10,000 | MAE5 | 1.234224 | 0.451484 | 0.342148 |
| 10,000 | Channel Rel5 | 1.234169 | 0.452457 | 0.342559 |
| 100,000 | Scratch | 1.214940 | 0.468453 | 0.350946 |
| 100,000 | MAE3 | 1.207316 | 0.473777 | 0.359616 |
| 100,000 | EMA3 | 1.224415 | 0.463002 | 0.351237 |
| 100,000 | MAE5 | 1.204849 | 0.474457 | 0.361759 |
| 100,000 | Channel Rel5 | 1.202688 | 0.475250 | 0.359629 |

MAE3相对共同Scratch的主要配对结果：

| Labels | Raw Δmacro AP [CI95] | Raw ΔNLL [CI95] | Common-support Δmacro AP [CI95] |
| ---: | --- | --- | --- |
| 1,000 | +0.033039 [0.027412, 0.039693] | −0.055747 [−0.063627, −0.048743] | +0.024739 [0.017230, 0.032757] |
| 10,000 | +0.025899 [0.020028, 0.032388] | −0.036485 [−0.042831, −0.030080] | +0.023804 [0.018141, 0.030549] |
| 100,000 | +0.005324 [0.002028, 0.008660] | −0.007624 [−0.010732, −0.004210] | +0.003505 [−0.000267, 0.006954] |

1k/10k的macro F1和balanced accuracy也提高，区间均为正；100k两者差值为−0.000223/−0.000800，区间跨零。MAE3的收益不能概括为所有硬决策指标同时改善。100k common-support macro AP未确认正增量，NLL差值−0.004972的区间[−0.008624,−0.000994]仍为负。

EMA3相对MAE3的raw Δmacro AP为−0.028207/−0.028135/−0.010775，ΔNLL为+0.025111/+0.034696/+0.017100；raw和common-support的对应区间均支持该退化。SSL内部学习通过没有转化为同等下游收益。该结果限于所测EMA目标、训练量与共同下游配方，不证明自蒸馏总体无效。

MAE5相对MAE3的raw macro AP和NLL差值在三个预算下均未排除零；100k common-support NLL有小改善。Channel Rel5相对MAE5在1k提高macro AP +0.005069 [0.003507,0.007135]、降低NLL −0.005056 [−0.006671,−0.003645]，同时τ AP降低−0.003914 [−0.007095,−0.000550]。10k未确认总体增益，100k raw NLL降低−0.002161 [−0.004120,−0.000287]，macro AP及τ AP区间跨零。因此不以最低单点NLL将Rel宣布为全面更好的分类器。

## Frozen readability与概率质量

100k固定C=1读出的完整实测：

| Initialization | Full NLL | Full macro AP | Global NLL | Global macro AP |
| --- | ---: | ---: | ---: | ---: |
| Scratch | 1.253389 | 0.441831 | 1.347254 | 0.340248 |
| MAE3 | 1.235023 | 0.457088 | 1.294122 | 0.404062 |
| EMA3 | 1.245888 | 0.446462 | 1.347556 | 0.338008 |
| MAE5 | 1.236345 | 0.456566 | 1.290500 | 0.406204 |
| Channel Rel5 | 1.236730 | 0.455035 | 1.284657 | 0.414909 |

MAE3相对Scratch的full Δmacro AP为+0.015257 [0.009965,0.020554]，global为+0.063814 [0.056307,0.072226]。10k global也增加+0.064867 [0.056388,0.073652]。这说明当前MAE global表示具有可读事件类别信息；结合此前完整10k固定checkpoint置换，可支持具体模型使用该路径，不能替代重训ablation或证明物理因果重要性。

Channel Rel5相对MAE5的global macro AP在10k/100k增加+0.010284/+0.008705，区间均为正；其full/fine-tune没有相同一致增益。不同路径的信息与完整协议的分类收益需要分别评价。Full fine-tune与frozen线性读出比较同时改变可训练参数、优化和Head，不能当作单变量架构消融。

固定C=1的1k高维full仍是失败的概率配方。例如MAE3的fit NLL0.174783、validation NLL3.246709，2,606个错误预测confidence>0.9；最高1% loss事件仅贡献总NLL的5.08%。这是广泛的泛化差距，与旧整层失活不同，不能因驻点与readback通过而验收为良好概率模型。详见[阶段177](2026-11-14_stage177_s4_probability_diagnostic.md)。目前优先假设是高维/少标签与正则化设置的影响，未独立证明唯一病因；后续仅在各原标签预算内分组train-inner选择C，再完整重拟合评价。

## τ子道与低背景工作点

MAE3相对Scratch的hadronic τ common-support AP增量在1k/10k/100k为+0.014350/+0.017222/+0.008202，对应CI95为[0.002950,0.027777]/[0.004230,0.032654]/[0.002629,0.013771]。Electron和muon τ子道的AP区间均跨零。该子道定义只保留目标τ分支与所有非τ背景，不与四类均衡prior直接比较。

MAE3的τ实际通过数如下，signal分母2,500、background分母7,500。每类的完整工作点及Clopper–Pearson计数区间均保存在精简JSON中。

| Labels | Tau selected at 1% FPR | Background selected | Tau selected at 0.1% FPR | Background selected |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 44 (1.76%) | 75 | 1 (0.04%) | 7 |
| 10,000 | 76 (3.04%) | 75 | 6 (0.24%) | 7 |
| 100,000 | 89 (3.56%) | 75 | 14 (0.56%) | 7 |

MAE3−Scratch的1%和0.1% FPR效率增量在三个预算的区间均跨零；高纯度τ收益尚未建立。阈值来自validation经验ROC，计数区间为描述性结果，不是独立选阈值后的coverage声明。τ AP或hadronic子道增益不能代替这些低通过率的限制。

## 内部健康、资源与清理

15项分类共147,130次optimizer updates，全部更新检查梯度；最大preclip norm307.565002。1,710次计划BF16激活/梯度采样全部健康，Head输入归一化RMS为0.867895–0.999995，Head隐藏LN RMS为0.947006–1.024976，SiLU RMS为0.111420–0.629792，采样精确零比例最大值为0。Informative FT最小input梯度平方范数0.0109263。LP阶段仅linear读出更新，解冻后的额外第149步采样按冻结配方检查。

15×14=210项checkpoint/optimizer/prediction读回、15次独立selected-checkpoint CPU FP32 backward、实际encoder参数更新均通过。30个full/global joblib在完整10k同一特征上按run/chunk/row/label回放，最大概率差0（要求≤1e-10）。这些是采样及选中checkpoint的实测证据，不能声称连续监控了每个事件的每层激活。

最终资源账本在06:26:24 UTC为：

| Allocation type | Device-hours |
| --- | ---: |
| HTCondor H100 MIG 2g.24gb | 18.664444444 |
| HTCondor H100 MIG 1g.12gb | 5.922222222 |
| Kubeflow H100 MIG 3g.47gb | 10.740418241 |
| HTCondor V100 early failed attempt | 0.071944444 |
| Total | 35.399029353 / 60 |

CPU allocation为423.043923533 core-hours；114条allocation均有终态证据，含等待、初始化、失败、旧执行与停止的重复任务。不同MIG时长不换算成等效FLOPs。两次新FP64任务均实际观测到Pod Succeeded并留存；此前已被平台GC的Pod继续使用明确标注的成功Job结束上界，不补造终态。MIG利用率查询返回N/A/权限不足，没有推测利用率百分比。

全部回放和20项检查通过后，删除5份临时feature bank及v2 gate小缓存，共36文件、3,643,405,382 bytes；小型readouts、预测、feature配置及来源manifest保留。重新执行feature回放须按冻结配置再生成cache。AFS本次收尾删除9个真实终态目录、16,627 bytes，均仅包含与EOS证据逐字一致的run.sh/job.sub；当前flavour_s4提交目录已空。清理清单与资源终态保存在`ROOT/audit/`。

英文图包括45页四分类diagnostics、三种label-efficiency图、训练曲线和内部健康范围。实际查看fine-tune label曲线、internal health和100k Scratch混淆矩阵/ROC/PR；显示正常、四类分数非常数。图与长日志保留EOS，仓库保存代码、配置、报告和精简数值。

## 可复现入口与产物

完整数值：[本轮45预测/66配对](../roadmap/results/s4_fp64_objectives_summary.json)、[20项收尾检查](../roadmap/results/s4_fp64_objectives_acceptance.json)、[终态与清理摘要](../roadmap/results/s4_final_closure.json)。原始评价位于`ROOT/evaluation/fp64_objectives/summary.json`，solver证书和磁盘readout位于相应`experiments/`目录。内部健康及回放汇总位于`ROOT/experiments/s4_fp64_objectives_validation/`。

实际最终命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PINPOINT_S4_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13
python -m analysis.flavour.s4.summarize \
  --evaluation "$PINPOINT_S4_RESULTS/evaluation/fp64_objectives/summary.json" \
  --health "$PINPOINT_S4_RESULTS/experiments/s4_fp64_objectives_validation/training_health_summary.json" \
  --output roadmap/results/s4_fp64_objectives_summary.json
python -m analysis.flavour.s4.accounting --root "$PINPOINT_S4_RESULTS"
python -m analysis.flavour.s4.accept_objectives --root "$PINPOINT_S4_RESULTS" \
  --campaign roadmap/configs/s4_fp64_objectives_evaluation.json \
  --comparisons roadmap/configs/s4_fp64_objectives_pairs.json \
  --evaluation "$PINPOINT_S4_RESULTS/evaluation/fp64_objectives/summary.json" \
  --validation-name s4_fp64_objectives_validation \
  --output "$PINPOINT_S4_RESULTS/fp64_objectives_acceptance.json" --require-complete
```

最终strict入口实际exit0、20项均true。新增汇总/解冻采样检查12项测试通过，FP64公式与既有solver/feature回放9项测试通过；实际CUDA的8项gate与30项磁盘回放另列为数据验证，不用单元测试替代。

## 阶段性结论与后续路线

事实：旧失活及后续已定位的空响应/LP断梯度均通过修复后的训练、梯度和checkpoint证据重新评价；MAE3能改善所测有限标签协议并使global表示包含可读类别信息。收益随标签量增加而减小，τ轻子子道和高纯度效率仍未改善到可确认水平。技术收尾通过与概率泛化、物理收益分别判断。

后续优先保留修正后的MAE3与同配方Scratch，在原标签预算内解决frozen readout正则化；再进入S5的τ混淆、charm、局部轨迹/顶点可测代理与重建覆盖率研究。额外MAE/Rel只在指定指标或路径的问题上作为控制，不因最低单点值扩量。方法、阈值及校准冻结后按S6打开test；目前calibration、冻结test、generator/detector系统包络及物理产额均未测量。
