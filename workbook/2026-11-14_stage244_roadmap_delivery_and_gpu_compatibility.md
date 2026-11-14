# 阶段244：路线图交付、归档链接与GPU兼容性

**独立审查与下一轮完整路线图已交付。**现有方法的改进成立，尚未建立最优性；下一步优先完成三类/四类同输入强基线、训练规模与显式跨层关系。R1全部16项短screen已完成，四pass及最终配对统计仍在执行。资源策略按用户要求采用能够尽快启动的兼容GPU、共享资源和MIG，HTCondor/Kubeflow并用。

## 阶段目标

将审查结论、归档与当前执行状态形成完整可读的交付，修正新版入口中“R1待启动”等滞后状态，落实减少排队的设备安排。核对两项V100启动失败、重提状态和成本，避免把软件构建失败解释成模型或硬件物理信息结论。

## 工作内容

完整路线由[总入口](../roadmap/README.md)、[证据审查](../roadmap/v2/01_evidence_review.md)、[任务/评价合同](../roadmap/v2/02_tasks_and_evaluation.md)、[模型设计](../roadmap/v2/03_models_and_physics.md)、[R0–R7执行](../roadmap/v2/04_execution_plan.md)、[文献](../roadmap/v2/05_literature.md)、[workbook索引](../roadmap/v2/06_workbook_index.md)、[资源操作](../roadmap/v2/07_resources_and_operations.md)及[JSON规格](../roadmap/v2/experiment_plan.json)组成。

保留已完成的stage237独立审查和stages238–243实际工作。本阶段再次联网核对FASERCal、NuGraph2 context-aware、Particle Transformer及pAUC论文入口，补充Exa.TrkX候选构造/筛选/整轨评价的参考；[追加访问记录](../roadmap/v2/source_update_stage244.json)说明访问范围。Exa.TrkX本阶段读取摘要与期刊元数据，具体实现前需补读构图和评估定义。

旧路线14份完整正文已位于[2026-11-14_v1](../roadmap/archive/2026-11-14_v1/README.md)。本次将20份历史workbook中的32个Markdown目标直接迁到归档，保留其历史正文、指标、命令路径和锚点；新增索引覆盖stages237–244。原docs资料继续作为研究输入。

资源代码新增实际CUDA启动算子检查、运行内核架构记录，将默认最低Capability从8.0改为7.5，当前构建下不允许降到SM70；保持运行时SM75的FP32/eager回退。Kubeflow stdout/stderr同步EOS，成本程序可读已保存的terminal Pod及failed-attempt manifest，保留失败allocation的设备归属。

## 详细数据

### 支持后续研究的已测事实

| Question | Measured evidence | Decision |
| --- | --- | --- |
| Existing improvement | S6 MAE3 ensemble BA4=43.5226%；相对Scratch +0.3354个百分点，CI95 [+0.2508,+0.4166] | 保留预训练成果；下一步建立强监督参照 |
| Tau difficulty | 冻结validation 1%背景cut下test τ效率3.5556%，实际FPR1.2080% | τ保留研究分支；三类近期交付并行推进 |
| Common three classes | 固定四类MAE3模型条件BA3=54.6933%；pixel-centered为74.4517% | τ不是唯一困难；测真实三类重训及条件域分母 |
| Saturation | 六个历史主模型均best=4/4，末轮NLL仍降0.00203–0.00779 | R2比较100k×12与300k×4，再按收益扩1M |
| Geometry and objects | 当前无显式跨层geometry context层；旧segment purity约2.0%–5.4% | R3显式测量关系；R4先改善候选与重建质量 |

前三项不是本阶段新模型结果，原始分母、统计口径和代码核对见stage237与证据审查。完整新三类/四类收益、几何结构收益、300k/1M收敛及物理flux产额均**未测量**。

R1全部16项screen的选中共同BA3为：B1 T3/T4=0.542333/0.546000；B2=0.491000/0.465667；B3=0.455667/0.455000；B4=0.540667/0.531000。每trial 300训练秒，设备与完成曝光不同，只用于冻结规则的短预算选择；完整Tune排名尚未形成。原表与实测JSON见[stage243](2026-11-14_stage243_r1_all_screens_and_full_training.md)。

22:10 UTC读回首个完整B4 seed42 T4结果：100k unique Train、每行四次曝光、共400k exposures，40k Tune全部推理完成。GPU为H100 NVL MIG3g.47gb，模型897,992参数，wall2024.190 s，完整Tune推理162.247 s，概率有38,918个不同向量。以下与同一Tune上的B0并列，仅为描述性单模型结果：

| Model / task | Common BA3 | Common NLL3 | BA4 | NLL4 |
| --- | ---: | ---: | ---: | ---: |
| B0 HGB T3 | 0.545900 | 0.897449 | — | — |
| B0 HGB T4 | 0.545300 | 0.897246 | 0.431025 | 1.214964 |
| B4 Scratch T4, seed42 | 0.546767 | 0.902870 | 0.424575 | 1.223468 |

B4的τ效率在同Tune描述性1% FPR下为0.0269，0.1%下为0.0031；这些cut不是最终Cut-B工作点。当前复杂模型相对summary HGB尚无全面优势证据，进一步支持先完成强基线、长训练与关系比较。配对区间、B4实际T3重训及其他神经网络仍待完成，不把此表作为最终晋级排名。原始记录为`R1/experiments/final_b4_t4_s42/manifest.json`及同目录predictions。

### 软件兼容性、失败与重提

2026-11-14 22:02:55 UTC交互T4实测：Torch2.7.1、CUDA build12.5、Capability7.5；`torch.cuda.get_arch_list()`为`sm_75/sm_80/sm_86/sm_89/sm_90/compute_90`。一个CUDA零张量归约返回0.0，启动通过。证据：`R1/audit/cuda_compatibility_stage244.json`。

原任务1582878和1582879均终态exit1，错误为`no kernel image is available for execution on the device`；GPU为V100/SM70。失败目录分别为`R1/failed_attempts/final_b1_t4_s42_v100_a1`和`final_b1_t3_s42_without_tracker_v100_a1`。重提1582881/1582882的提交文件实际要求`Capability >= 7.5`且没有DeviceName限制；不存在旧失败任务继续写入的情况。两次失败allocated GPU时间分别为0.027222/0.036111 h，合计0.063333 device-hours，计入成本。

当前构建缺少SM70内核，FP32本身不能使V100运行；这不限制未来使用已验证的其他软件环境。T4、A100、H100及兼容MIG继续调度，其他设备先做实际算子/学习验证。当前运行任务、按型号成本和UTC快照见`R1/resources.json`；本阶段的固定读回见[交付检查](../roadmap/v2/delivery_validation.json)。成本为快照时点的累计值，任务仍在运行。

22:08:19 UTC的R1资源快照合计5.161286 device-hours、73.719049 allocated CPU core-hours，其中V100失败0.063333 device-hours已正确归属。该合计用于组织成本，分型号/MIG明细另存，不是FLOP等效小时。

### 验证与命令

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p 'test_flavour_v2*.py'
python -m analysis.flavour.v2.r1.accounting \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1
```

21项v2相关测试通过，包含标签/层级loss、事件身份、缓存、投影物理、空响应反传、配对统计与ties。未新增镜像实现的测试。44份相关Markdown的646个本地链接、5个归档锚点、14份归档正文保持、JSON解析、修改Python语法及`git diff --check`通过，实测检查保存于`delivery_validation.json`。

## 阶段性结论

**事实：**审查、旧计划归档和R0–R7设计已经完成；R0与R1短screen有实际证据，完整R1尚在执行。当前软件最低SM75是实测构建要求；GPU型号、整卡或MIG不作为统一科学准入条件。

**研究判断：**先证明共同三类强基线、数据/曝光/容量瓶颈和跨层关系收益。四类尚未被现有证据排除，τhad specialist及条件衰变头值得在更强表示上比较；更多类别以合并后的τ/四类收益和共同三类非劣决定。具体晋级值、分母、先验、尾部统计与新独立Confirm规则已写入评价合同。

## 下一步计划

1. 读取已运行任务的四pass完整Tune、同allocation计算控制和实际seed重复；通过raw counts、学习健康、400次paired物理块统计及stress交付R1排名。
2. 将最强基线及B4参考进入R2的100k×4/12、300k×4矩阵，按实测吞吐控制候选数与预算，再触发1M和容量实验。
3. R3优先改跨窗口/跨层稀疏关系及几何anchors；R4–R6分别由对象质量、τ工作点和强监督对照触发。候选冻结后才开放Cal-A/Cut-B，最终Confirm用于独立确认。
