# 阶段250：条件敏感性、配对交互与工作点数值修复

**同一批1,600个Tune事件上，MAE相对Scratch的noise退化幅度多3.00个百分点，配对CI95为[−5.20,−0.92] pp。**这是seed42及已接受proxy下的描述性结果。已修复FPR边界的浮点比较问题，并重算相关中期统计；BA/NLL等核心结果逐项不变。完整R1排名、多seed及全部条件交付仍在执行。

## 阶段目标

检验名义收益与条件敏感性的关系，区分固定cut变化与同FPR区分能力，核查名义预测读回，修复实际复现的工作点数值不一致。

## 工作内容

完成B0 T3/T4、B4 T3/T4、B4-MAE T4的nominal、noise、combined、remove_pixel、remove_tracker，共25/55条真实诊断。`stress_evaluate`缓存各模型/条件在共同物理块draw上的统计向量，增加模型差值和“条件变化之差”的配对区间，保留协方差并减少重复nominal计算。跨模型交互是在首组条件读回后增加的描述性分析，未用于修改冻结的HPO、Scratch排名或训练配方。

## 详细数据

### 样本、条件和统计

冻结Tune family hash选取每类400个，共1,600事件；非τ分母1,200。1356个run/chunk物理块，已使用accepted-family连接，0个跨chunk块。400次class-signature分层重采样，模型/条件共享draw；区间条件于已训练seed42，训练seed总体波动未测量。

noise为每事件Poisson均值5个pixel噪声channel、每hardware bar channel均值0.002 PE噪声。combined另加bar gain=0.9、threshold=2 PE、pixel/bar survival=0.98、tracker efficiency=0.97、alignment=[+0.1,−0.1] mm、tracker sigma scale=1.2。沿用已接受S6 proxy，不是新的detector系统误差约束。名义tracker由相同ideal crossing和RNG重数字化逐值复现；未使用held-out pattern。

样本与条件为`R1/data/stress_rows.parquet`、`R1/stress/preparation.json`。完整统计为`R1/interim/stress_seed42/evaluation/stress_results.json`；小型结果：[r1_interim_stress.json](../roadmap/v2/r1_interim_stress.json)。

### 三类条件变化与输入依赖

下表为condition−paired nominal，单位pp，方括号为pointwise CI95。

| Model | Noise ΔBA3 | Combined ΔBA3 |
| --- | ---: | ---: |
| Summary HGB T3 | −2.083 [−4.438,+0.301] | −1.750 [−4.113,+0.703] |
| Summary HGB T4 | −1.417 [−3.750,+1.241] | −2.083 [−4.606,+0.247] |
| B4 Scratch T3 | −0.250 [−1.844,+1.516] | −0.167 [−1.722,+1.335] |
| B4 Scratch T4 | +0.333 [−1.041,+2.053] | +0.500 [−1.103,+2.147] |
| B4-MAE T4 | −2.667 [−4.454,−0.863] | −2.417 [−4.506,−0.584] |

直接配对的MAE−Scratch在noise下BA3为−2.500 pp，[−4.517,−0.627]；combined为−2.417 pp，[−4.540,−0.311]。比较相对各自nominal的变化，noise交互为−3.000 pp，[−5.196,−0.916]，combined为−2.917 pp，[−5.152,−0.709]。本proxy与样本支持MAE的共同三类决策对噪声更敏感；实际detector变化和跨训练seed的适用范围尚未测量。

冻结权重移除pixel/tracker时，B4 T3的ΔBA3为−2.250/−6.250 pp，B4 T4为−2.333/−7.417 pp，MAE T4为−2.583/−7.583 pp，六项区间均排除零。这些测量输入依赖；重训的tracker净增量继续使用stage246独立控制，全域B4 T3增益+3.897 pp。

英文图：`R1/interim/stress_seed42/figures/r1_interim_stress.{pdf,png}`；[脚本](../roadmap/tools/plot_r1_interim_stress.py)已执行，目视检查标签、图例与区间通过。

### 固定τ cut和重新选ROC工作点

cut来自完整40k nominal Tune的描述性1%非τ FPR，在各条件保持不变；子样本的实际FPR单独报告。

| Model / condition | Selected τ / 400 | Selected non-τ / 1200 |
| --- | ---: | ---: |
| B4 T4 nominal | 13 | 9 |
| B4 T4 noise | 9 | 5 |
| B4 T4 combined | 4 | 5 |
| MAE T4 nominal | 8 | 14 |
| MAE T4 noise | 6 | 11 |
| MAE T4 combined | 4 | 10 |

B4 combined在固定cut下τεff变化−2.250 pp，CI95[−3.801,−0.999]；背景通过率也从0.75%降至0.417%。每个条件重新选描述性1% FPR时，B4 nominal/noise/combined的τεff为3.5%/3.25%/4.0%，combined−nominal为+0.500 pp，[−1.527,+2.056]。因此观察到固定cut效率损失，尚未建立同FPR区分能力损失；下一步需同时检查score移动和形状变化。

MAE重新选1% FPR时为2.0%/2.25%/1.75%，noise/combined效率差区间均跨零。本批只有400个τ和1200个背景，不据此晋级τ候选。

### 名义读回

B0最大概率差不超过2.22×10⁻¹⁶；B4 T3/T4为4.39×10⁻⁵/1.12×10⁻⁵；MAE为2.05×10⁻³。主MAE在A100推理、诊断在H100 MIG，各批次构成也不同。五个模型的原任务和条件三类argmax变化均为0；三项T4在完整nominal 1%/0.1%τ cut下的选择变化均为0。条件效应以诊断自身的paired nominal计算，保留实际数值读回范围。

### 工作点修复和重算

审查发现MAE nominal表内τ效率为2.0%，旧配对向量归一化权重后得到不同工作点，使noise变化写成+1.0 pp，而2.25%−2.0%应为+0.25 pp。原因是`cumsum`、`bincount`和`sum`累加顺序不同：FP累计值与`FPR × background weight`在严格边界相差末位，恰好可接受的整个tie group被排除。

NumPy官方[源码文档](https://github.com/numpy/numpy/blob/v2.3.0/numpy/_core/fromnumeric.py)记录`cumsum(a)[-1]`与`sum(a)`可能因pairwise sum产生舍入差异；在线读回摘录为`R1/audit/numpy_cumsum_source.json`。

回归样本每类400事件，100个τ与12/1200背景处于可接受score组，真实效率25%、FPR恰好1%。旧实现把权重乘1/400后返回0%，已先复现测试失败，再将FPR/purity边界改为仅容许`4 n ε_float64`相对累加舍入。tie group仍整体选择，不能从超过真实预算的tie中取部分事件。修复后四种权重尺度的效率和阈值一致，原tie测试通过。

新版本为`r1-tie-aware-float-boundary-2`；最终验收要求nominal与stress均有此版本。原始概率、训练目标与参数不变。

已重算stage246/247全部24个配对、各400次bootstrap。216项BA3/NLL3/BA4/NLL4、macro AP/AUC、recall检查的point/CI最大变化为0；18项工作点结果发生可测变化。最大为pixel-centered e在0.1% FPR处的区间端点0.1625 pp；controls最大为NC在90% benchmark purity处0.0300 pp。修订值写回[r1_interim_b4.json](../roadmap/v2/r1_interim_b4.json)和[r1_interim_controls.json](../roadmap/v2/r1_interim_controls.json)，保留原小型schema。原始完整结果另存`results_before_boundary_fix.json`；核对表为`R1/audit/operating_point_boundary_repair.json`。旧报告的主要BA/NLL、tracker和MAE名义结论保持成立。

修复后的MAE noise/combined ROC效率变化为+0.25/−0.25 pp，与表内点值一致。跨模型BA交互结果保持完全一致。

### 实际执行、验证与资源

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest tests.test_flavour_v2_r0 tests.test_flavour_v2_r1
python -m analysis.flavour.v2.r1.stress_evaluate \
  --root "$R1_ROOT/interim/stress_seed42" --replicates 400
python roadmap/tools/plot_r1_interim_stress.py --root "$R1_ROOT"
```

R0/R1共23项测试通过，包含权重边界和共享波动相消的交互测试。完整统计通过batch执行：1582892 stress修复重算，1 CPU/3GB、allocation49 s；1582893 B4重算，8 CPU/67 s；1582894 controls重算，8 CPU/82 s，均终态4、ExitCode=0。后续完整stress按1 CPU/3GB申请，主nominal评估继续匹配8 workers。AST、JSON、文档链接及diff检查通过。

Kubeflow OIDC凭据在00:08 UTC前后失效。K17在Pod server dry-run认证阶段失败，未创建Job、零allocation；B2诊断1582895保留于Condor，未移除。其余15条已提交Kubeflow allocation均有缓存终态或保守Job生命周期证据。`accounting`在实时API不可用时使用这些证据，缺失或非终态记录仍标unresolved。K17记录为`R1/audit/k17_authentication_failure.json`。终态捕获器在已有Kubeflow作业全部终止后停止，恢复使用时再启用。

00:10:57 UTC成本快照为17.698579组织性device-hours，仍有运行作业；异构卡/MIG不作FLOP等效，最终成本未测量。新完成B1 T4的BA3=0.564933、NLL3=0.866249、BA4=0.446225；B2 T3的BA3=0.551200、NLL3=0.900973。完整四族任务及多seed尚未齐备，最终排名未确定。

## 阶段性结论

**事实：**MAE在当前noise/combined proxy的三类退化比Scratch更大。固定τ cut与同FPR能力需分别评价。名义读回没有改变本批类别或指定τ选择。工作点权重尺度问题已实测修复，主要名义结论保持不变。

**研究判断：**较好的名义NLL不足以单独确定后续初始化；R6需验证定向预训练和条件鲁棒训练能否同时保留名义质量并降低已观察到的敏感性。当前结果不终止MAE或τ分支，R1排名继续按冻结Scratch规则。

## 下一步计划

完成主T3/T4与同allocation计算控制，按实际排名补齐前两族及B4三seed。补齐B1/B2/B3剩余30条条件路径，用修复版本交付完整配对、合作组三域对照、英文图表、成本及终态审查。Cal-A/Cut-B/Confirm继续封存。
