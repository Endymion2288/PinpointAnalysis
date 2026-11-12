# 阶段 38：S2 十二项空间筛选完成

两族十二项正式空间配置全部完整，均有14项通过的读回证据。PN200μm τ AP=0.36010319；200μm−1mm原样本ΔAP=+0.00706716，pointwise 95% CI=[−0.00223045,+0.01497597]，固定common-support ΔAP=+0.00990883，CI=[+0.00165290,+0.01894474]。两族曲线均不单调，当前结果尚不能支持“越细越好”的一般结论。信息、模态、训练域、跨层控制与三种子仍待完成，S2整体尚未验收。

## 阶段目标

完成PN200μm及最后两项相邻粒度比较，汇总空间筛选的完整输出、性能、成本和可解释范围，继续冻结的23项后续配置。

## 工作内容

PN200μm（`1573782.9`）真实终态JobStatus=4、ExitCode=0、allocation=4,604 s；05:34:30 UTC通过全部14项CPU读回，包括冻结配置、四轮与恢复状态、checkpoint/概率、40k逐轮预测及NLL/AP重算。十二项空间配置逐项汇总为`complete_spatial_output_inventory.json`：全部manifest complete、配置匹配、H100 NVL、14项pass、80,064 applied updates、0 AMP skips、完整terminal成本；没有重新认证已接受样本或proxy。

资源核算及部分评价正常退出，commit=`ba5004f`：16 results（4 references+12空间模型）、430行分域指标、48个历史三类诊断、18项预定配对、27项pending、2 factorial。完整resolution图与paired-effects图已实际查看。

路线图README及执行入口同步标明十二项空间配置完成、控制与额外种子继续执行，避免将历史首轮/部分曲线状态作为当前状态。文档链接与`git diff --check`通过；此次同步没有改动运行配置。

## 详细数据

PN200μm为ParticleNet dim64、161,156 parameters、pixel+bar、count reducer、seed42、batch4、4 epochs、20,016 attempted steps/epoch；H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。沿用独立train80k/validation10k、每类20k/2,500及accepted split；各轮exposures=80,040/80,044/80,040/80,040，合计320,164。NLL选中epoch4。局部压缩仍为同平面4mm分块、canonical chunks≤32 nodes，随后处理全部全局块；粗表示不含原生cell残差。

| Epoch | PN200μm NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.27925357 | 0.31602544 | 1,164.53887 |
| 2 | 1.26232029 | 0.33759797 | 2,305.43329 |
| 3 | 1.25887887 | 0.35443209 | 3,454.08381 |
| 4 | 1.23876869 | 0.36010319 | 4,594.45564 |

程序wall=4,596.90210 s，allocation=4,604 s=1.27888889 GPUh，额外7.09790 s，peak allocated memory=1,584.43896 MiB。Balanced accuracy=0.40530、macro F1=0.40465178、Brier=0.67331925；e/μ/τ/NC AP=0.59693069/0.41931586/0.36010319/0.38251209，argmax recall=0.4904/0.2864/0.5188/0.3256。

### 完整空间主表

AP以25% τ benchmark prior为条件；固定common support为8,559 events、Neff=7,959.41365。表中allocation包括所有尝试，ParT native projection包含原输出失败与完整替换。

| Model | Tau AP | Fixed common-support AP | NLL | Actual GPUh |
| --- | ---: | ---: | ---: | ---: |
| ParT native joint | 0.35858391 | 0.32383619 | 1.22107879 | 1.94250 |
| ParT 100μm joint | 0.35025881 | 0.31715126 | 1.22682355 | 1.92167 |
| ParT 200μm joint | 0.35513011 | 0.31829366 | 1.22811129 | 1.71472 |
| ParT 1mm joint | 0.35502857 | 0.32086106 | 1.24293540 | 2.02167 |
| ParT native projection | 0.35733812 | 0.32248272 | 1.22337225 | 2.78250 |
| ParT 1mm projection | 0.35421688 | 0.31995100 | 1.22571671 | 1.80083 |
| PN native joint | 0.35510474 | 0.32202830 | 1.23930563 | 1.75278 |
| PN 100μm joint | 0.35548005 | 0.32148225 | 1.23904515 | 1.54806 |
| PN 200μm joint | 0.36010319 | 0.33309115 | 1.23876869 | 1.27889 |
| PN 1mm joint | 0.35303603 | 0.32318232 | 1.24970777 | 1.43639 |
| PN native projection | 0.34797906 | 0.31945295 | 1.24334514 | 1.02306 |
| PN 1mm projection | 0.35100848 | 0.32166112 | 1.23785700 | 1.09917 |

十二项全部尝试合计20.32222222 GPUh。这里的排序是单seed点估计，不能据此确定PN200μm总体优于其他架构；冻结的架构对比与三种子合同继续执行。

### 最后两项预定配对

沿用400次、128个accepted-family/chunk blocks（每类32）的配对bootstrap。所有区间均为pointwise事件区间；固定支持区间以既定权重为条件。

| PN B−A | Raw tau ΔAP [95% CI] | Fixed common-support ΔAP [95% CI] |
| --- | --- | --- |
| 100μm − 200μm | −0.00462314 [−0.01203929,+0.00303426] | −0.01160890 [−0.01888470,−0.00499589] |
| 200μm − 1mm | +0.00706716 [−0.00223045,+0.01497597] | +0.00990883 [+0.00165290,+0.01894474] |

原样本的两项区间均跨0；100−200μm可排除至少+0.01的100μm收益，200μm−1mm尚不能排除或保证该量级收益。固定支持上，200μm相对100μm及1mm都有正事件区间支持。粗E/vertex匹配并未消除所有可能的先验差异，这些结果仍以已训练模型和权重为条件。

同allocation预算均为4,601.55354 s。100−200μm比较选100μm epoch3与200μm epoch4，ΔAP=−0.00852892、100μm剩414.78429 s；200μm−1mm比较选200μm epoch4与1mm epoch3，ΔAP=+0.01135323、1mm剩705.46996 s。所有额外成本已计入，均为整轮分辨率点估计，CI未测量。

### PN200μm工作点、分支与域

Validation empirical 1% FPR阈值=0.43959561，通过95/2,500个τ与75/7,500个背景，τ效率=0.0380；200μm−1mm效率差=+0.0024，CI=[−0.00969755,+0.01180249]，没有正区间支持。0.1% FPR阈值=0.57993865，通过3个τ、7个背景，尾部统计不足。最终阈值留待S6。

τ electron/muon/hadronic的信号数445/395/1,660，在inclusive 1%下通过13/7/75个，分支AP=0.08162731/0.06109069/0.29841395，使用各分支信号与同一7,500个non-τ背景。相对1mm通过数变化为+3/−1/+4，合计+6；分支差值CI未测量，尚不能确定由何种拓扑带来收益。

| PN200μm domain | Events / tau | Tau AP |
| --- | --- | ---: |
| Pixel centered | 885 / 176 | 0.31826514 |
| Large scintillator | 4,494 / 1,053 | 0.39509155 |
| Pixel+bar input empty | 440 / 148 | 0.33636364 |
| Tracker-only response | 158 / 58 | 0.36708861 |

空输入与tracker-only域AUC=0.5，AP等于域prior。域内配对CI未测量，训练域控制继续等待。

### 产物与复现

正式根目录 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`：`complete_spatial_comparison.json`保存本次PN200μm训练/读回、全部空间模型与配对、交互、分域和核算快照；`complete_spatial_output_inventory.json`汇总十二项已完成输出。日志为`readback_pn_200um.log`、`accounting_complete_spatial.log`、`evaluation_complete_spatial.log`。完整表和图在`evaluation_final/`。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments pn_200um_joint_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
```

05:35:16 UTC组织累计57.12619820 GPUh、missing history records=0；05:39:22 UTC正式12 complete、2 running、21 idle、0 held。运行者为ParT1mm OR和半格平移控制；Kubeflow v7仍Pending、无node。没有迁移或重复writer。

## 阶段性结论

空间筛选已完成，完整输出和实际成本可复核。已观察到的正逐项事件区间包括ParT native−100μm，以及固定支持上的PN200μm相对100μm/1mm；原生输入的收益随模型和对比条件变化，两族曲线均不单调。两族native−1mm及四格交互仍无正区间支持，native架构优劣未确定。量化可能改变有限模型的归纳偏好和优化过程，这是待控制的解释，不能把AP排序直接当作探测器信息量排序。跨种子稳定性、multiplicity、局部关联、跨层连通、模态及接受度机制继续由已冻结控制回答。

## 下一步计划

完成余下23项控制/额外种子，包括当前OR和平移任务；逐项读回并更新41项单seed配对与4项三种子均值。全部实际结果、资源、图表和物理解释齐备后执行无`--allow-incomplete`的严格评价，才能验收S2；calibration/test保留至S6，不进入S3–S6。
