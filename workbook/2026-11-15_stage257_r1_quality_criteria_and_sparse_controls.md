# 阶段257：R1研究质量条件与稀疏投影对照设计

**同一个B1三类模型相对HGB满足本seed的开发质量条件，相对B4则因μ recall退化未满足。**B1四类模型相对B4/HGB满足本seed的τ研究条件，但这不消除其μ代价。判定必须同时写明任务、参考模型、统计范围和失败项。

## 阶段目标

将路线图中的研究晋级条件落实为可复核的逐项结果，为最终R1交付准备紧凑汇总；根据stage256的稀疏μ定位补充具体结构控制及文献依据。完整排名、三seed统计和R1最终验收尚未完成。

## 工作内容

新增`analysis.flavour.v2.r1.summarize`，读取既有评估与验收，导出训练/设备/曝光、共同任务、条件域、配对区间、seed波动、压力测试及终态资源。最终导出要求`acceptance.status=complete`；本阶段没有生成最终交付JSON。

`quality_criteria`按`experiment_plan.json`读取阈值，逐项检查实用效果、区间、NLL、recall或实际FPR。它不改变冻结的Scratch架构排名，也不将Tune的开发区间解释为新独立确认。新增4项针对性测试，覆盖总体改善掩盖μ损失、零下界、τ目标的共同任务/FPR约束及未测量区间。

联网核对Zhang的anti-aliasing论文与Graham/van der Maaten的submanifold sparse convolution论文官方摘要和元数据，两项HTTP200。追加[来源记录](../roadmap/v2/source_update_stage257.json)、[文献P5/P6](../roadmap/v2/05_literature.md#p5)和[结构控制](../roadmap/v2/03_models_and_physics.md#21-r1发现的稀疏μ问题与后续对照)。本阶段未实现这些模型或测得其收益。

## 详细数据

以下直接使用stage254/252已经完成的400次accepted-family连接物理块区间；Tune40k，每类10k，seed42。完整逐项值见[r1_quality_criteria_interim.json](../roadmap/v2/r1_quality_criteria_interim.json)。

| Candidate / reference | Research criterion | Actual result | Interpretation |
| --- | --- | --- | --- |
| B1 T3 / B4 T3 | BA3点差≥2 pp，下界>0；NLL3增量上界≤0.01；逐类recall下界≥−1 pp | BA3 +2.270 pp；μ差下界−7.739 pp | 未满足，失败项为μ recall |
| B1 T3 / HGB T3 | 同上 | BA3 +2.030 pp；e/μ/NC下界+4.150/+0.299/−0.683 pp | 本seed全部满足 |
| B1 T4 / B4 T4 | τ1%效率点差≥1 pp且下界>0；BA3下界≥−0.5 pp；NLL4增量上界≤0.01；实际FPR交付 | τ效率差+2.400 pp，下界+1.813 pp；BA3下界+1.342 pp；NLL4上界−0.03234 | 本seed全部满足 |
| B1 T4 / HGB T4 | 同上 | τ效率差+2.260 pp，下界+1.630 pp；BA3下界+1.483 pp；NLL4上界−0.02349 | 本seed全部满足 |

这里的τ1%是各模型在完整开发Tune上选取的描述性工作点，实际FPR已保存。该τ研究条件约束共同BA，未包含逐类μ非劣要求；B1 T4相对B4 T4的μ recall仍低5.30 pp，继续如实报告。上述单seed结论不替代三seed结果、校准/选cut或R7独立确认。

稀疏控制以现有B1的真实结构为起点：1mm投影、首层stride4、累计stride32，已有totals/presence和tracker set。后续分别比较首层stride、栅格相位/滤波及高分辨率局部残差；保留原accepted occupancy分组、空事件与全域逐类验收。滤波可能削弱孤立响应，局部稀疏卷积也不会自动跨越plane间隔，二者均为待检验假设。

验证环境为LCG108/Python3.12，`PYTHONPATH=src`、BLAS/OpenMP线程1：

```bash
python -m unittest tests.test_flavour_v2_r1_summary
python -m py_compile src/analysis/flavour/v2/r1/summarize.py \
  tests/test_flavour_v2_r1_summary.py
git diff --check
```

4项测试全部通过；语法与diff检查通过。真实中期数据上的4项候选/参考判定已输出至`R1/audit/quality_criteria_stage257.json`并读回。完整summarize导出尚未执行，需等待最终验收。

02:17 UTC，Condor 1582873/1582880/1582883/1582904/1582905均Running、各1次启动。B1 seed42及B4 seed17执行fresh计算控制，B3 seed42执行T3；新增B1两seed在H100 1g MIG执行T4。唯一控制器仍为PID2052979，45/55条条件路径完成。Cal-A/Cut-B/Confirm未读取；R2/R3未启动。

## 阶段性结论

**事实：**开发质量条件的判定依赖参考模型；总体BA提高可以与某类显著退化同时存在。已有单seed数据支持继续比较B1的四类输出，但不能称为各类全面改进。最终汇总与判定代码已具备，完整输出未测量。

**待检验假设：**早期降采样、稀疏局部表示或融合决策可能影响有响应的低occupancy μ。论文为控制变量提供方法依据，当前数据尚未证明其中任一因果解释。

## 下一步计划

完成B3完整T3推理，按冻结规则选择另一强族并扩seed；补齐全部计算控制和55条诊断。完成全量400次统计、独立核算、英文图表与终态记账后运行验收，再生成最终汇总和中文报告。R2的训练规模、R3的稀疏结构控制由完整结果决定，本阶段只完成设计。
