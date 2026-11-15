# Stage279：R2 D0完整评价、独立排序读回与交叉拟合诊断

## 阶段目标

区分当前模型的多类决策偏移与稀疏事件排序不足，并在首批规模比较结束前固定一项低复杂度诊断。D0的完整40k Tune结果和诊断均已完成；D1/D2的四个模型继续训练，本阶段不决定它们的优劣。

## 工作内容

- 新增`analysis.flavour.v2.r2.readout`：按accepted family连接的chunk分五折，仅在fit折拟合正温度或两个全局bias，held折评价；NC bias锚定为0。
- 正温度inverse范围`[.05,20]`；bias范围`[-3,3]`、L2=.001；目标为fit折的class-balanced mean NLL，没有超参数扫描或occupancy条件参数。输入为保存的条件三类概率之log，clip为1e-15。
- `evaluate.independent_points`增加独立sklearn weighted AP/AUROC，直接使用保存的概率及域内均匀类别先验，与原tie-aware实现逐类核对。
- `accept`要求每个实际family/task/precision/width对应的真实拟合门槛；解析与模型相同的默认`cnn_variant=resnet`，不能把width16的检查用于width32。
- 交付依赖增加完整readout诊断；图表增加`r2_readout_diagnostic`。本阶段局部图名均带`_partial`，完整批次保留独立输出位置。

该诊断承接[M4温度缩放](../roadmap/v2/05_literature.md#m4)和[M7 logit adjustment](../roadmap/v2/05_literature.md#m7)。全局平衡pool的frequency adjustment只是共同常数；这里拟合的两个自由bias用来检验已有表示的决策偏移。多类softmax加bias可以改变单类概率的排序，不能解释为增加了探测器信息。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2`。

### 完整D0原始模型

同一40k Tune，原e/μ/τ/NC各10k，条件三类均排除真τ。训练为fresh seed42、100k pool、4 passes、BF16 encoder/FP32 head。endpoint与各自best-screen为同一步。下列指标是开发比较，保留原始输出；Cal-A/Cut-B/Confirm均未开放。

| Model | BA3 (%) | NLL3 | e recall (%) | μ recall (%) | NC recall (%) | τ eff @ 1% FPR (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 T3 | 56.6500 | .871321793 | 62.70 | 50.16 | 57.09 | — |
| D0 T4 conditional | 56.5333 | .864586107 | 64.43 | 46.48 | 58.69 | 5.11 |

400次paired physical-block bootstrap的T3−conditional T4：ΔBA3=+.116667 pp，95%区间[−.194961,+.414475] pp；ΔNLL3=+.006735686，[+.004343466,+.009193097]。μ recall增加3.68 pp，同时e/NC分别下降1.73/1.60 pp。D0不支持“三类化已经改善整体任务”的结论。

低occupancy仍按原accepted `pixel_channels+bar_channels+tracker_channels<100`，包含空响应。完整Tune的原e/μ/τ/NC支持为2971/2801/3371/3010；以下采用域内均匀三类先验。它与stage278固定4k screen中的自然mixture AP有不同口径。

| Model | Sparse BA3 (%) | Sparse μ recall (%) | Sparse μ AP | Sparse μ AUROC |
| --- | ---: | ---: | ---: | ---: |
| D0 T3 | 39.400819 | 10.317744 | .397356514 | .545741946 |
| D0 T4 conditional | 39.175592 | 9.425205 | .395209637 | .543901877 |

空响应的原e/μ/τ/NC为256/241/326/260；两个模型条件三类均判NC，BA3=1/3、μ AUROC=.5。这与完全无观测输入一致，但空响应只占稀疏域的一部分，不能据此解释全部稀疏μ问题。

### 冻结的五折诊断

合同形成于2026-11-15 11:17:40 UTC，已在D0开发工作之后；基础模型/配方选择也用过Tune，因此五折结果仍是开发诊断。6,845个chunk、6,845个family连接component、跨chunk component=0；每折原四类各2,000事件，非τfit/held分别24k/6k。fold family和chunk完整性均通过。

| Model / domain | Raw BA3 (%) | OOF temperature BA3 (%) | OOF global bias BA3 (%) |
| --- | ---: | ---: | ---: |
| D0 T3 / all | 56.650000 | 56.650000 | 56.163333 |
| D0 T4 / all | 56.533333 | 56.533333 | 56.436667 |
| D0 T3 / sparse | 39.400819 | 39.400819 | 38.215828 |
| D0 T4 / sparse | 39.175592 | 39.175592 | 38.984268 |

温度缩放使全域NLL3分别从.871321793→.864601421和.864586107→.860533953，所有事件argmax保持不变。全局bias后的稀疏μ recall为8.139950%/10.853267%，均没有伴随稀疏BA改善。没有估计拟合readout的置信区间，也没有将它用于原始模型排名。条件bias或更复杂calibrator尚未测量。

### 运行、验证与产物

HTCondor三项CPU工作分别为完整D0局部评价`1582972`、分组合同`1582973`、D0 readout `1582974`，schedd均为`bigbird20.cern.ch`。提交工具每次记录全部队列及EOS/WRITE兼容性后选择兼容且负载最低者。

- `evaluation_partial/`：4个endpoint/best视图、11域、8配对；原独立point/core-interval检查分别554/116项通过。
- `readout_partial/independent_scores.json`：增加sklearn排序读回后994项通过，无失败。
- `readout/contract.json`、`readout/folds.parquet`：正式批次复用的诊断合同与分组。
- [紧凑读回](../roadmap/v2/r2_readout_review.json)：合同、两个模型的all/sparse结果、各折拟合及Train/Tune gap。
- 英文图：`figures/r2_readout_diagnostic_partial.{pdf,png}`；原学习曲线/规模/稀疏图同步保留`_partial`。

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest tests.test_flavour_v2_r2_readout tests.test_flavour_v2_r2
python -m analysis.flavour.v2.r2.plots \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2 --partial
```

12项测试通过，覆盖connected-chunk分组、held label不影响该fold拟合输出、已知偏置的held恢复、温度argmax不变、tie-aware/缺类排序、容量门槛及完整交付依赖。图表在交互CPU生成，模型训练与大样本评价通过batch执行。

## 阶段性结论

事实：D0三类化增加μ recall，但整体BA差值区间跨零，NLL更差。当前两个全局bias不改善BA；温度可以改善NLL但保留argmax。稀疏μ的域内AUROC仍接近.545。

假设：全局决策偏移不能充分解释当前稀疏域损失，局部表示/几何路径仍值得独立检验。这不是物理可分性上限，也没有排除所有条件readout。

## 下一步计划

完成仍在运行的D1/D2固定终点，依次形成共同Train gap、完整400次区间、全部endpoint的同一五折诊断及正式英文图。按实测尾端、Train/Tune变化及成本决定D3、更多unique或容量控制。只有这些科学判断、必要扩展和终态验收完成后，才交付完整R2。
