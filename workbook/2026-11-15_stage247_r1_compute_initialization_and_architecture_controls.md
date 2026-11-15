# 阶段247：R1同计算、初始化与架构中期对照

**ParticleNet T4比当前B4有约1个百分点的共同三类BA增益；MAE改善四类BA和NLL，但两者的τ尾部提升尚未建立。**B4相同总wall预算下，T3多训练约98k次曝光后共同BA接近T4。本阶段是seed42的开发比较，完整架构排名和三seed结果仍在执行。

工作点数值更正：相关400次配对统计已在[stage250](2026-11-15_stage250_r1_condition_interactions_and_operating_point_repair.md)修复浮点边界后重算；BA/NLL等核心结果逐项不变，FPR/purity工作点以小型JSON修订值为准，原始统计保留于results。

## 阶段目标

区分训练资源、预训练初始化和架构差异对分类的影响，并检查实际wall预算是否对齐。使用新增完整分数开展400次物理块配对统计；补齐失败作业的设备成本归属，继续利用最早释放的资源。

## 工作内容

对已完成的B4 T3/T4、B4同计算T3、B4-MAE T4和B2 ParticleNet T4，在全域及两种几何域做12项配对。统计由Condor1582889执行，8CPU/8 workers，统计wall52.452 s。每个模型都使用相同40k Tune，所有比较共享400次物理块权重；与stage246相同的6,845个分层物理块和seed20261116。

详细结果保存在`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1/interim/controls_seed42/evaluation/results.json`，主要指标小型读回见[r1_interim_controls.json](../roadmap/v2/r1_interim_controls.json)。架构排名仍须等待所有Scratch族的T3/T4完整结果；未按本中期比较提前扩展seeds。

## 详细数据

### 同计算控制

| Seed | Reference T4 wall (s) | Fresh T3 wall (s) | Relative error | T3 exposures | T3 BA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 2,024.190 | 2,018.502 | −0.281% | 398,022 | 0.547133 |
| 2026 | 2,088.722 | 2,085.571 | −0.151% | 397,408 | 0.546900 |

Seed42的参考和计算控制在`pinpoint-r1-extend-b4-s42-k12`、H100 NVL MIG3g.47gb同一allocation完成，seed2026在Condor1582874、H100 NVL完成。两个控制均从fresh初始化执行；计时包含加载、训练和验证，实际误差均通过±5%验收。

Seed42相同wall下，T3−T4 conditional的ΔBA3=+0.000367，CI95[−0.004014,+0.003902]，ΔNLL3=−0.005208 [−0.007358,−0.002887]。与T3固定四pass相比，额外训练曝光的ΔBA3=+0.003633 [+0.000172,+0.006900]，ΔNLL3=−0.003579 [−0.005244,−0.002019]。这些是包含wall-based schedule变化的完整资源配方效应，不将其全部归因于类别数。

### 初始化与架构

| Model, T4 seed42 | Common BA3 | Common NLL3 | BA4 | NLL4 | Tau eff at descriptive 1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4 Scratch | 0.546767 | 0.902870 | 0.424575 | 1.223468 | 0.0269 |
| B4-MAE | 0.547200 | 0.890817 | 0.430325 | 1.212520 | 0.0241 |
| B2 ParticleNet | 0.556700 | 0.891792 | 0.434475 | 1.215185 | 0.0282 |

| Difference | ΔBA3 [95%] | ΔBA4 [95%] | ΔNLL4 [95%] | Δtau efficiency at 1% FPR [95%] |
| --- | --- | --- | --- | --- |
| B4-MAE − B4 Scratch | +0.000433 [−0.003763,+0.003881] | +0.005750 [+0.001999,+0.008892] | −0.010948 [−0.012786,−0.008805] | −0.002800 [−0.005690,+0.001520] |
| B2 − B4 Scratch | +0.009933 [+0.005223,+0.014184] | +0.009900 [+0.005771,+0.013604] | −0.008283 [−0.011301,−0.005551] | +0.001300 [−0.003080,+0.005277] |

MAE对共同NLL3的差值为−0.012053 [−0.014205,−0.009471]；B2为−0.011078 [−0.014439,−0.007485]。三类和四类概率指标均有可复核的变化，τ fixed-FPR工作点仍需单独优化。这些中期效果小于路线中用于昂贵扩展的2 pp共同BA目标；完整R1仍依冻结排名完成必要seed重复。

三者Train unique和曝光相同，但不同架构的wall和设备不同：B2在H100 NVL为4,489.690 s，MAE在A100为5,498.810 s，B4 Scratch在H100 MIG为2,024.190 s。跨设备wall不解释为架构等计算效率。MAE只加载encoder，分类head、branch norms及summary readout重新初始化；历史SSL operation为0.635321 MIG device-hours，包含98k训练与2k train-inner reconstruction probe，未重复计作本R1新消耗。

### 资源归属与迁移

`r1.accounting`新增Condor event log的实际`DeviceName`回读。早期Python在manifest之前失败的profile也有分配证据；23:18:46 UTC快照中，所有Condor GPU allocation均已归到实际型号，未改变总小时。另从真实fit/profile结果补充Kubeflow型号；无runtime记录的早期Kubeflow尝试保留明确flavor。

该时点已用13.264916 device-hours（不同型号合计仅用于组织记账），分为A100 5.327226、V100失败0.063333、H100整卡4.438706、T4 1.075186、runtime已确认H100 MIG2.264631及仅有flavor证据的h100-47gb 0.095833。完整矩阵仍在运行，最终成本**未测量**。原18 device-hours是规划额度；当前必做训练、seed和诊断继续按实测记录，后续预算需根据完整矩阵成本重新分配。

B4 seed42 Kubeflow任务于23:09:58 UTC成功终止并释放配额。Condor1582884经核对仍Idle、NumJobStarts0、wall0，移除后迁移为`pinpoint-r1-b1-without-tracker-k14`，已在H100 MIG运行。迁移证据为`R1/audit/migrate_b1_without_tracker_k14.json`。普通CNN1582885随后在已释放的Condor H100启动；运行中的主训练继续完成。未为统一型号重新排队。

### 验证与复现

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 实际统计payload通过Condor运行。
python -m analysis.flavour.v2.r1.evaluate --root "$R1_ROOT/interim/controls_seed42" \
  --config "$R1_ROOT/interim/controls_seed42/config.json" --replicates 400 --workers 8
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT"
```

统计程序逐模型核对Tune身份及标签，400次paired权重复用；成本代码通过语法和真实事件日志回读，未新增镜像实现的测试。Cal-A/Cut-B/Confirm继续封存，当前cut为开发Tune描述性cut。

## 阶段性结论

**事实：**当前B2 T4比B4 T4更好地利用输入完成共同三类及四类分类；MAE初始化改善BA4/NLL；同计算T3的总体BA与条件T4接近。尚无明确τ尾部提升证据。

**研究判断：**继续完成强基线竞争是有效投入，数据/训练与关系设计仍比直接扩大预训练矩阵更优先。R1完整排名将决定R2参考，当前单seed比较不作为最终独立确认。

## 下一步计划

完成余下T3/T4、同计算及普通CNN/无tracker控制，按完整排名扩展seeds。压力推理与统计随已完成权重在批处理推进。完整R1交付包括合作组同构域对照、τ污染、种子波动、配对区间、全部图表、资源和后端终态。
