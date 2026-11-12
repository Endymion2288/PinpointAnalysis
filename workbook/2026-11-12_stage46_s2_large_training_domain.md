# 阶段46：Large训练域、全域泛化与条件选点

按冻结的full-validation NLL选点规则，ParticleNet full-domain训练相对large-only训练的全域τ AP增量为+0.03136247，pointwise 95%事件区间=[+0.02176243,+0.04088759]；固定粗E/vertex支持后为+0.01862325，区间=[+0.00830622,+0.02916431]。Large区域内的AP配对也支持full训练，但large-only的区域内NLL随训练改善、全域NLL却恶化，导致全域选中较早的epoch1。主比较因而包含训练覆盖、独立事件数、重复采样和选点的共同影响，不能归因于单一空间因素。

## 阶段目标

完成large-only训练控制，在完整目标域、中心和large区域交叉评价；检查固定步数下独立样本量、条件域学习过程及checkpoint选择如何影响结论。

## 工作内容

`pn_native_largetrain_s42_e4_v2`对应`bigbird20.cern.ch`的`1573782.22`，实际JobStatus=4、ExitCode=0、NumJobStarts=1，节点`b9pgpun003.cern.ch`。训练完整四轮、14项CPU输出读回全部pass；选epoch1不表示只训练了一轮。输入仍为原生pixel+bar，ParticleNet dim64/161,156 parameters，除训练域及对应类别权重外沿用冻结配置。

新增`analysis.flavour.s2.domain_curves`，读取已完成且读回pass的逐轮validation概率，保存full/center/large的NLL、τ AP、balanced accuracy和计数，生成英文学习曲线。逐轮full指标必须与训练history重算一致；主选点不变，各条件域自身的NLL最优轮次仅作结果后诊断。

沿用既有geometry flags：中心域使用pixel边界内缩20 mm、目标前500 mm；large域使用scintillator横向边界内缩15 mm、下游内缩200 mm。定义来自`analysis.flavour.s2.prepare.regions`，不声称与历史报告fiducial完全相同。训练域过滤仅用于train，validation始终完整保留。

## 详细数据

seed42、4 epochs、batch4、20,016 attempted steps/epoch，H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。共同validation为10k、每类2,500；calibration/test未使用。

| Training domain | Unique train events | e / mu / tau / NC | Four-epoch exposures | Exposures / unique events |
| --- | ---: | --- | ---: | ---: |
| Full | 80,000 | 20,000 / 20,000 / 20,000 / 20,000 | 320,164 | 4.00205 |
| Large only | 35,936 | 8,864 / 9,479 / 8,341 / 9,252 | 320,039 | 8.90580 |

Large各轮exposures=80,010/80,009/80,010/80,010；两模型均80,064 applied updates、0 AMP skips。Loss使用各自训练集计数的inverse-class-frequency权重。匹配优化步数并未匹配unique事件数，也不代表同样的有效数据覆盖。

### 冻结checkpoint的完整目标域比较

| Metric | Large-only, epoch1 | Full, epoch4 |
| --- | ---: | ---: |
| Raw tau AP | 0.32374226 | 0.35510474 |
| Fixed E/vertex support tau AP | 0.30340506 | 0.32202830 |
| Tau ROC AUC | 0.57169589 | 0.63150771 |
| NLL | 1.69656875 | 1.23930563 |
| Balanced accuracy | 0.38250000 | 0.39740000 |
| Macro F1 | 0.37200067 | 0.39204149 |
| Brier | 0.80462806 | 0.67434646 |

| Class | Large-only AP | Full AP |
| --- | ---: | ---: |
| CC nu_e | 0.54032649 | 0.59882218 |
| CC nu_mu | 0.35072142 | 0.41459288 |
| CC nu_tau | 0.32374226 | 0.35510474 |
| NC | 0.29354084 | 0.36957463 |

所有四类AP点估计更高，不等于每类argmax recall均提高：NC recall为large-only 0.5872、full 0.2844，而τ recall分别0.1984/0.5664。非τ、综合指标和recall差值CI未测量。

400次、128个accepted-family/chunk blocks（每类32）的pointwise配对bootstrap；固定支持8,559 events、Neff=7,959.41365，仅控制既定粗E/vertex分箱，未控制靶核Z/同位素，区间以现有模型和权重为条件。

| Full minus large-only | Difference [95% CI] |
| --- | --- |
| Raw tau AP | +0.03136247 [+0.02176243,+0.04088759] |
| Fixed-support tau AP | +0.01862325 [+0.00830622,+0.02916431] |
| Tau ROC AUC | +0.05981181 [+0.04561638,+0.07221943] |
| Tau efficiency at 1% FPR | −0.00160000 [−0.01326047,+0.00479614] |
| Tau efficiency at 0.1% FPR | −0.00400000 [−0.00735471,+0.00196273] |

Raw AP区间支持超过+0.01的增量，固定支持区间支持正增量但下界小于+0.01。1%工作点large-only阈值0.39440164，选87τ/75背景；full选83/75，差值区间跨0。0.1%分别13/7和3/7，尾部背景不足25，不能据AP结论推定低背景效率提高。Large-only的electron/muon/hadronic τ在inclusive 1%阈值下通过10/9/68个，共同分母445/395/1,660；分支差值CI未测量。

### 相同条件域内的配对

条件域使用原有标记和full-validation选中的checkpoint，未重加权E/vertex。中心885 events/176 τ、large 4,494/1,053 τ；不同域有不同prior，跨域AP绝对值不直接构成配对。

| Domain | Large-only AP | Full AP | Full−large ΔAP [95% CI] |
| --- | ---: | ---: | --- |
| Centered | 0.30878123 | 0.31582532 | +0.00704409 [−0.02752698,+0.03935868] |
| Large | 0.36691177 | 0.38642610 | +0.01951434 [+0.00598887,+0.03493095] |

Large域ΔAUC=+0.02044807，区间=[+0.01232813,+0.02893532]；1%效率差=+0.00474834，区间=[−0.01509976,+0.01616629]，large-only/full分别选45/50个τ、各34背景。中心1%效率差区间也跨0，只有7背景通过。Large域0.1%效率差区间为负，但只有3个背景通过，保留原始结果而不作稳定尾部结论。

### 条件学习过程与结果后选点诊断

| Large-only epoch | Full NLL | Large NLL | Centered NLL | Full tau AP | Large tau AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.69656875 | 1.14955462 | 1.07221612 | 0.32374226 | 0.36691177 |
| 2 | 1.93142289 | 1.12120071 | 1.05001327 | 0.32723542 | 0.37947850 |
| 3 | 2.09053020 | 1.09837480 | 0.99440928 | 0.32829268 | 0.38500765 |
| 4 | 1.95051378 | 1.07124799 | 0.96726893 | 0.33248719 | 0.38937430 |

按NLL的事件可加性分解，`NLL_outside=(10000*NLL_full−4494*NLL_large)/5506`，large区域外5,506个事件的NLL依次为2.14304196/2.59272664/2.90032794/2.66817097；full训练对应1.37121733/1.35637105/1.33912812/1.33019439。这是由已测域计数/NLL计算的条件平均，不对AP作相同线性分解。

Large-only在自身域的NLL持续改善，却在全域选点中保留epoch1。若结果后仅按large NLL选择，会取epoch4，其large AP=0.38937430，full为0.38642610，差+0.00294819；中心对应0.35498384和0.31582532。两项均仅为诊断点估计，未计算差值CI，未替代主选点或主配对。当前不能声称full训练在所有区域选择规则下都优于large-only，也不能由posthoc点估计反向声称large-only稳定优胜。

### 资源、产物与验证

Large-only程序wall=6,057.38261 s，完整四轮allocation=6,065 s=1.68472222 GPUh，额外7.61739 s，peak allocated memory=1,009.92773 MiB；full为6,310 s、1,593.10254 MiB。全部训练成本计入，未只计算best epoch1。

共同allocation预算6,064.53299 s内，large-only按NLL仍选epoch1、full可选epoch3，τ AP差=+0.02344324，无CI。距所选早期checkpoint的预算差不表示实际节省了后续训练，四轮均已执行。09:59:25 UTC组织累计80.65506215 GPUh，H100视图44.41672881 GPUh，missing history records=0。

正式根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`large_training_comparison.json`保存模型、主/条件配对、学习曲线、域外NLL分解、manifest/history/readback、真实终态与成本；`conditional_large_training.json/.png/.pdf`、`domain_learning_curves.json/.png/.pdf`、`large_training_diagnostics/paired_comparisons.pdf`及首页PNG、large-only四分类诊断页均实际查看。日志为`readback_pn_native_largetrain.log`、`accounting_large_training.log`、`evaluation_large_training.log`、`conditional_large_training.log`和`domain_learning_curves.log`。

本批评价commit=`39f6ba9`，正常退出：28 results（4 references+24科学模型）、754行分域、84个历史三类条件表、30 paired/15 pending、2 factorial、0 seed means。新学习曲线入口在两模型上实际运行，8个逐轮full AP/NLL均与history一致；两个模型×两个条件域的主checkpoint AP/NLL与独立条件分析逐项一致，语法和`git diff --check`通过。新入口由实际数据验证，本阶段未重跑53项既有回归套件。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments pn_native_largetrain_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.conditional_pairs --root "$S2_RESULTS/h100" \
  --a pn_native_largetrain_s42_e4_v2 --b pn_native_joint_s42_e4_v2 \
  --output "$S2_RESULTS/h100/conditional_large_training.json"
python -m analysis.flavour.s2.domain_curves --root "$S2_RESULTS/h100" \
  --experiments pn_native_joint_s42_e4_v2 pn_native_largetrain_s42_e4_v2 \
  --output "$S2_RESULTS/h100/domain_learning_curves.json"
```

本批首次核算在用户恢复Kubeflow登录之前因OIDC凭据过期退出，主评价未启动；登录恢复后核算及评价重跑成功。失败日志去除授权URL后保留为`accounting_large_training_pre_login_refresh.log`。09:56:39 UTC已实际确认原v7 Job UID不变、Pod仍Pending且无node，记录`kubeflow/client_login_restored.json`；未重提或迁移科学任务。

10:12:40 UTC实时manifest为26 complete、3 running、6 not started：本批24项已评价科学输出之外，ParT native的seed17/2026随后也结束并通过14项CPU读回，尚未纳入此批28-result主评价与09:59核算。两项真实terminal及读回日志分别保留，下一批统一刷新核算、评价和三seed native spread；`stage46_execution_snapshot.json`记录这一时序，不能将26个训练终态直接当作26项正式评价完成。

## 阶段性结论

事实：按既定全域目标和选点规则，full训练在全域及large区域的τ AP有正事件区间；低背景效率没有同样的正区间。Large-only后续各轮的区域内NLL/AP仍改善，而域外NLL较差，全域选点因此与自身域选点不同。

待检验解释：训练覆盖、样本量、重复次数、能谱/顶点分布和未校准置信度均可能影响域迁移。相同步数不隔离这些因素。当前控制支持保留full-domain作为完整目标域参照，尚不足以单独解释两份历史报告的全部差异。

## 下一步计划

完成中心训练域控制，并复用条件曲线与交叉评价；继续既有额外种子和plane-only任务。新完成的ParT native种子先实际读回，再汇总三种子spread和既定配对，不能把种子当作新增物理事件。35项输出、41单seed配对和4三seed均值齐备后，执行严格评价与最终S2科学验收；S3–S6不开始。
