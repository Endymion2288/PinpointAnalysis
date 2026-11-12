# 阶段42：ParticleNet 固定图与 bar 增量

当前ParticleNet/seed42/四轮预算下，pixel+bar相对pixel-only的τ AP提高0.06890803，pointwise 95%事件区间=[0.05663332,0.08443211]；固定E/vertex common support后提高0.04851078，区间=[0.03602058,0.06154046]。1%背景接受率下τ效率由1.24%提高至3.32%，差值区间也为正。动态图相对固定几何图的AP区间跨0，固定图实际allocation减少24.06%。本批两项训练、14项输出读回与配对评价均完成；S2其余14项科学配置继续执行。

## 阶段目标

回答bar读出在pixel之上提供多少四分类和τ识别收益；独立比较动态图与固定几何图在同四轮及同allocation预算下的质量和成本。

## 工作内容

`1573782.17`（fixed-graph）和`.18`（pixel-only）在`bigbird20.cern.ch`正常结束，JobStatus=4、ExitCode=0、NumJobStarts=1。两项均完成14项CPU读回，包括冻结配置、四轮checkpoint/optimizer、NLL选best、10k最佳概率和40k逐轮概率；没有重新认证样本或proxy。

Fixed-graph保留全部pixel+bar输入及161,156个参数，`dynamic=False`。局部三层均用固定物理距离；全局三层均用几何图，包括既定的同平面/跨平面邻居配额。主模型局部首层及全局首层使用几何，其余层改用feature KNN，因此此对照同时涉及后续层是否保留几何邻域与跨层配额。它没有移除坐标或bar信息。

Pixel-only将bar读出及其加性汇总一起移除，其他模型参数、局部compression和全部全局块覆盖规则相同。Pixel+bar和pixel-only分别重训；测量的是这两个已训练系统的增量，不能直接解释为同一模型中某个事件的bar归因。

## 详细数据

80k train/10k validation，各类20k/2,500；seed42、batch4、4 epochs、20,016 attempted steps/epoch，H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。两项均80,064 applied、0 AMP skips；exposures=80,040/80,044/80,040/80,040，总320,164。NLL均选epoch4。calibration/test未使用。

| Model | Tau AP | Fixed common-support AP | Tau AUC | NLL | Balanced accuracy | Macro F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dynamic pixel+bar main | 0.35510474 | 0.32202830 | 0.63150771 | 1.23930563 | 0.39740 | 0.39204149 | 0.67434646 |
| Fixed geometry pixel+bar | 0.35435361 | 0.32331491 | 0.62822149 | 1.24349048 | 0.40110 | 0.39858246 | 0.67573443 |
| Dynamic pixel-only | 0.28619671 | 0.27351752 | 0.56983725 | 1.34070449 | 0.32930 | 0.31120173 | 0.72478579 |

| Model | CC nu_e AP | CC nu_mu AP | CC nu_tau AP | NC AP | Argmax recall e / mu / tau / NC |
| --- | ---: | ---: | ---: | ---: | --- |
| Main | 0.59882218 | 0.41459288 | 0.35510474 | 0.36957463 | 0.4920 / 0.2468 / 0.5664 / 0.2844 |
| Fixed | 0.59331711 | 0.41781131 | 0.35435361 | 0.37151094 | 0.5008 / 0.3144 / 0.5224 / 0.2668 |
| Pixel-only | 0.44906901 | 0.30389503 | 0.28619671 | 0.28385405 | 0.3352 / 0.4488 / 0.4484 / 0.0848 |

AP以每类25%的benchmark prior为条件。加入bar后四类AP、NLL、balanced accuracy的点估计均改善，但μ的argmax recall反而降低；排序质量与未校准四类决策须分开解释。非τ差值CI未测量。

### 预定配对区间与工作点

同400次、128 accepted-family/chunk blocks（每类32）的pointwise bootstrap。Common support仍为8,559 events、Neff=7,959.41365，固定E/vertex权重。区间以本次训练模型为条件，不包含跨seed波动，也不是同时覆盖区间。

| Main−control | Raw tau ΔAP [95% CI] | Fixed common-support tau ΔAP [95% CI] |
| --- | --- | --- |
| Dynamic−fixed graph | +0.00075113 [−0.00567937,+0.00685530] | −0.00128660 [−0.00712364,+0.00463922] |
| Pixel+bar−pixel-only | +0.06890803 [+0.05663332,+0.08443211] | +0.04851078 [+0.03602058,+0.06154046] |

动态图的raw/common-support AP增益区间上界均低于预定+0.01。Bar增量两个区间的下界均高于+0.01；共同粗E/vertex支持下仍保留增益，但这不能单独定位为拓扑、能量测量或接受度的某一来源。

Main−pixel-only τ AUC差=+0.06167045、区间=[+0.04999507,+0.07683968]。1% FPR效率差=+0.02080000、区间=[+0.01034517,+0.03158505]；主模型通过83/2,500个τ，pixel-only通过31个，二者背景各75/7,500。Fixed通过84个τ/75背景，main−fixed效率差=−0.0004、区间=[−0.01058548,+0.00472263]。阈值main/fixed/pixel分别为0.45402154/0.44904050/0.29916102。

0.1% FPR下main/fixed/pixel通过τ=3/12/0，背景各7；尾部统计不足。此时main−pixel效率差区间下界为0，不能从1%结果推出更严格尾部的收益。最终工作点保留至S6。

### 接受度、分域与分支

Pixel-only输入为空的事件为2,312（655 τ），有pixel响应的事件为7,688；pixel+bar空输入只有440（148 τ）。Pixel-empty组中pixel-only输出恒定，AUC=0.5、AP=655/2,312=0.28330450。以下组内比较使用相同事件，保留不同域prior：

| Region | Events / tau | Tau prior | Pixel-only tau AP | Pixel+bar tau AP |
| --- | ---: | ---: | ---: | ---: |
| Pixel empty | 2,312 / 655 | 0.28330450 | 0.28330450 | 0.32051583 |
| Pixel 1–99 channels | 4,219 / 1,073 | 0.25432567 | 0.30136601 | 0.37943686 |
| Pixel 100–999 channels | 2,268 / 525 | 0.23148148 | 0.27117750 | 0.33413379 |
| Pixel at least 1,000 channels | 1,201 / 247 | 0.20566195 | 0.27312373 | 0.36023522 |
| Pixel-centered | 885 / 176 | 0.19887006 | 0.21925256 | 0.31582532 |
| Large scintillator | 4,494 / 1,053 | 0.23431242 | 0.28051924 | 0.38642610 |

所有pixel-active占用组以及中心域的AP点估计也提高，因此all-event差异并非仅出现于pixel-empty事件。AP不是子域AP的线性平均，不能据表分解总增益比例；组内差值CI未测量。

在各模型inclusive 1%阈值下，electron/muon/hadronic τ通过数：main=7/4/72，fixed=9/6/69，pixel-only=1/0/30；共同分母445/395/1,660。分支AP及其他分域保存在独立快照；分支差值CI未测量，μ分支零通过不是其普遍不可分辨的结论。

### 学习曲线与资源

| Epoch | Fixed NLL / tau AP | Pixel-only NLL / tau AP |
| --- | --- | --- |
| 1 | 1.34472416 / 0.28574831 | 1.35446095 / 0.27467961 |
| 2 | 1.26798207 / 0.33987249 | 1.35423310 / 0.27307876 |
| 3 | 1.26020592 / 0.34789749 | 1.35947066 / 0.27217336 |
| 4 | 1.24349048 / 0.35435361 | 1.34070449 / 0.28619671 |

Pixel-only第三轮NLL变差，第四轮恢复并选best；目前不以单轮波动终止该输入分支。

| Model | Program wall [s] | Allocation [s / GPUh] | Extra allocation [s] | Peak allocated memory [MiB] |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 4,784.52235 | 4,792 / 1.33111111 | 7.47765 | 1,642.31445 |
| Pixel-only | 4,112.94476 | 4,120 / 1.14444444 | 7.05524 | 1,590.25049 |
| Main | 6,293.25696 | 6,310 / 1.75277778 | 16.74304 | 1,593.10254 |

Fixed/main allocation比=0.75942948，减少24.06%；peak allocated memory略高，不能将它描述为显存也更省。等allocation预算4,791.41274 s下fixed选epoch4、main选epoch3，main−fixed τ ΔAP=−0.00716810；预算4,119.12867 s下pixel选epoch4、main选epoch2，main−pixel ΔAP=+0.04151995。两项都是整轮分辨率点估计，CI未测量。

### 产物与复现

正式根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`graph_pixel_comparison.json`保存两项完整训练/读回、主模型、配对、分域与成本快照。`graph_pixel_paired_review.png`和fixed/pixel两张classification诊断页已实际查看。

合并评价commit=`799a30a`，正常退出：25 results（4 references+21科学模型）、673行分域、75个历史三类诊断、27 paired/18 pending、2 factorial、0 seed means。08:29:35 UTC组织累计72.13702406 GPUh、H100视图35.89869073 GPUh，missing history records=0；尚未整体验收。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" \
  --experiments pn_native_fixed_s42_e4_v2 pn_native_pixel_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.output_inventory --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json"
```

日志分别为`readback_pn_native_fixed.log`、`readback_pn_native_pixel.log`、`accounting_graph_pixel_controls.log`、`evaluation_graph_pixel_controls.log`和`output_inventory.log`。现有清单已核对21项完整输出，严格gate仍为false；新增文档通过`git diff --check`。

## 阶段性结论

事实：在本次ParticleNet条件下，bar detector-proxy带来高于+0.01的τ AP事件区间收益，并提高1% FPR效率。Shared-domain点估计显示pixel-active事件也获益。动态图没有获得τ AP正区间支持，固定图以较少实际训练allocation达到接近的τ排序质量。

待检验解释：bar可能通过能量/纵向发展、覆盖范围和形态提供互补信息；当前不能分离其来源。固定图的资源优势可能来自避免后续feature距离计算，实测wall还包含节点/I/O状态，需结合后续完整结果选择工程基线。以上均为seed42和当前局部compression条件，未测量其他模型族上的模态增量。

## 下一步计划

完成bar-only以测量pixel在bar之上的互补收益，再完成+tracker、center/large训练域、8项额外种子和2项plane-only，共余14项科学配置。继续沿用存活Condor任务及Kubeflow probe；无缺项严格评价、全部输出清单和最终科学报告通过后再验收S2，不进入S3–S6。
