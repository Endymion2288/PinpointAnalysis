# 阶段 18：补齐历史三类与 τ 子道评价

新增评价实现通过 9 项统计检查。此次补齐路线图要求的三类历史诊断入口和 τ 子道排序指标；不改变训练、样本、硬件参数或科学比较配置。

## 阶段目标

回答当前四分类模型在 νe/νμ/NC 条件样本上如何表现，以及 τ→e/μ/had 相对同一非 τ 背景各自的可分性。三类历史参照见路线图 `05_local_studies.md` 的 S2 条目，主结果仍为四分类。

## 工作内容

`historical_three_class` 从已有四分类预测中移除 true τ，分别在 full、pixel-centered、large-scintillator 域计算诊断表。不重新训练模型。每类同时报告原四类 score 在三类样本上的 AUC/AP，以及三类概率重新归一后的 AUC/AP；三类 argmax confusion、precision/recall、balanced accuracy 与 conditional NLL 使用后者。若非 τ 概率和为零，以均匀三类分布定义该条件概率，并单列事件数。

这不复现旧模型/旧样本/旧工作点，不能与历史 PDF 的数值直接相减解释成方法收益。表中明确保存 `historical_model_reproduced=false`。

`tau_branch_metrics` 对每个 τ 分支保留该分支 signal 和全部 7,500 个非 τ background，排除其他 τ 分支，报告 ROC AUC、AP、branch prevalence、1%/0.1% FPR 工作点、signal/background counts 与尾部统计标记。同时保留以 inclusive τ 工作点计算的分支效率。输出中的 `events` 现为该 branch-vs-background 评价总分母；纯分支数量明确放在 `branch_signal_events`/`tau_events`，不混淆两个分母。

这些属于分层诊断，未追加新的主假设检验或训练；calibration/test 继续冻结至 S6。

## 详细数据

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p test_flavour_s2_evaluation.py -v
```

9 tests passed，0.998 s。日志：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/evaluation_unit_tests.log`。新增检查验证移除 τ 后的三类分母与正确 confusion、零条件概率的有限 NLL，以及分支评价使用相同非 τ 背景、排除其他 τ 的行为。已知分支示例的 AP/AUC 均为 1，inclusive threshold=0.8 下分支效率为 0.5。

另已人工查看 `resource_profile_v2/resource_curves.png`：英文轴与图例可读，完整事件 latency 与 training memory 均按真实 retained block 数呈现；首例 warmup 高耗时仍保留，不解释成低占用事件的稳定性能。该图原始压力样本为阶段 16 的 24 个占用秩事件，不能当作总体平均。

实际 neutrino 三类/分支参照指标见下表；完整深度训练仍进行中。

commit `37b89b5` 后提交补充评价 `1573753.0`（bigbird20），写入 `evaluation_final/`，使用完整比较配置并明确 `--allow-incomplete`。该作业现已正常完成，exit code=0，实际 allocation=45 s。进一步加强的已知样本检查确认 e-score 的 raw 三类 AUC=0.5、conditional AUC=1，避免把重新归一前后的排序混为一谈；再次运行全部 9 tests passed，0.940 s。

18:50:35 UTC accounting 实测累计 2.34086 GPU-hours，缺失 history records=0，当前三个科学 clusters 的全部 33 tasks 均已纳入核算。仍为运行中快照。

### 批量评价实际结果

`1573753.0` 已完成并写出 `evaluation_final/summary.json`：4 个已完成 reference models、12 张三类条件记录、43 项 pending comparisons；`requested_comparisons_complete=false`。实际评价记录代码 commit 为 `8ac9b00`，训练未改变。

完整 B0 的三类条件诊断如下。AUC 使用原四类输出的 e/μ/NC score，限定 true label 为这三类；balanced accuracy 与 NLL 使用三类重新归一概率。没有训练新的三分类器。

| Evaluation domain | Three-class events | Balanced accuracy | Conditional NLL | νe AUC | νμ AUC | NC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 7,500 | 0.49920 | 0.94030 | 0.78304 | 0.68289 | 0.65363 |
| Pixel-centered | 709 | 0.65701 | 0.72813 | 0.91141 | 0.82034 | 0.80490 |
| Large-scintillator | 3,441 | 0.60237 | 0.81372 | 0.88842 | 0.75938 | 0.73151 |

相同模型在条件域中排序更好，是当前样本的实测描述；条件域同时改变能谱/顶点/类别混合，不能解释为 pixel 的独立增益，也不能与不同历史训练直接相减。正式 native/coarse 对照仍需固定评价域的配对结果。

完整 B0 的各 τ 分支相对相同 7,500 个非 τ 背景：

| Tau signal branch | Signal events | Branch prevalence | AP | AUC | Signal selected at 1% FPR | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Electron | 445 | 0.05601 | 0.07169 | 0.57595 | 7 | 0.01573 |
| Muon | 395 | 0.05003 | 0.06174 | 0.55336 | 8 | 0.02025 |
| Hadronic | 1,660 | 0.18122 | 0.28391 | 0.65620 | 50 | 0.03012 |

三个 AP 对应不同 signal prevalence，不把它们直接横向比较成分支分类能力。1% 工作点均选中 75 背景；0.1% 工作点实际为 7 背景，尾部统计不足标记保留。这些分支结果使用 inclusive τ score，不是新训练的子道分类头。

## 阶段性结论

事实：评价所需分母与 score 定义已经显式区分，程序检查通过，四个 references 的补充评价正常完成。同一 B0 在不同条件域表现明显不同，说明历史比较必须统一评价域。假设：native topology 是否特别改善 leptonic τ 或历史三类中的某一类，仍待完整训练及配对结果。

## 下一步计划

随 33 项科学训练完成，更新同一套三类/τ 子道表、39 项配对和 4 项三种子均值，完成 S2 科学报告。样本与硬件响应继续使用用户接受的现有值。
