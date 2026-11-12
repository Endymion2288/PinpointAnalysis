# 阶段69：S3完整2×2对照与分类头优化状态

四个单元均完成四轮训练和输出读回。Seed42下，4 mm窗口的native−1 mm四分类macro AP为−0.060502435，95%配对事件区间[−0.067034736, −0.054427614]；fine窗口两个模型均为常数预测。交互项虽为正，但本次结果没有确立“4 mm压缩抹平native优势”：fine单元的分类头已失去上游梯度，4 mm/native也有明显激活饱和。该统计差异描述当前编码与训练条件，不能单独归因于探测器信息。

## 阶段目标

补齐native/1 mm×4 mm/16-pitch窗口的冻结2×2对照，回答四类指标如何变化，并用学习曲线、完整validation预测和只读梯度诊断区分优化状态。继续使用实际先获配的计算后端。

## 工作内容

复用接受的80,000 train、10,000 validation事件，run10081/10082/10083/10084分别对应CC nu e/CC nu mu/CC nu tau/NC；validation每类2,500事件。沿用accepted split、geometry和detector-proxy，不重新认证样本或校准proxy。Calibration/test未使用。

四个单元均为749,868参数、seed42、H100 NVL/LCG108/Torch2.7.1/BF16、四轮80,064 applied updates、零AMP skips；全部以full-validation NLL选择第四轮。每项14个CPU输出读回检查通过。1 mm为`reducer=count`，保留bin内hit数；16-pitch窗口约0.3328×0.3648 mm，小于1 mm coarse中心间距。

使用冻结`roadmap/configs/s3_pairs.json`执行400次配对bootstrap，128个accepted-family/chunk blocks、每类32个；所有模型共享物理block multiplicities。保持原始score ties。区间均为条件于已训练seed42模型的pointwise事件区间，未作多重比较修正，不包含训练seed波动。Common support沿用固定E/vertex权重：8,559个正权重事件、Neff7,959.413646535195，target Z未匹配。

另对四个已选`best.pt`各做每类2事件、共8事件的CPU FP32/eval前向及一次反向。诊断不执行optimizer step；四项使用相同事件选择规则。训练更新数从正式history和readback读取；轻量best checkpoint不含optimizer-step字段，诊断JSON的该字段为null。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

### 四单元最终结果

下表NLL来自训练history；配对NLL则由发布的概率以FP64重新计算，存在浮点舍入差异。

| Window / pixel representation | Macro AP | NLL | Macro F1 | Balanced accuracy | Actual training GPUh |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-pitch / native | 0.250000000 | 1.386298356 | 0.100000000 | 0.2500 | 1.562500000 |
| 16-pitch / 1 mm | 0.250000000 | 1.386298203 | 0.100000000 | 0.2500 | 1.106666667 |
| 4 mm / native | 0.261014083 | 1.384043492 | 0.167641404 | 0.2675 | 1.405000000 |
| 4 mm / 1 mm | 0.321516517 | 1.356914366 | 0.314125560 | 0.3289 | 0.875000000 |

| Window / pixel representation | AP e | AP mu | AP tau | AP NC | Distinct probability vectors / 10,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-pitch / native | 0.250000000 | 0.250000000 | 0.250000000 | 0.250000000 | 1 |
| 16-pitch / 1 mm | 0.250000000 | 0.250000000 | 0.250000000 | 0.250000000 | 1 |
| 4 mm / native | 0.256964918 | 0.259722285 | 0.264397081 | 0.262972047 | 970 |
| 4 mm / 1 mm | 0.395244044 | 0.307243947 | 0.288937201 | 0.294640877 | 9,536 |

4 mm/native真实终态为Condor `1573835.6`，JobStatus4/ExitCode0，allocation5,058 s；4 mm/1 mm为`1573835.7`、3,150 s。两个fine单元的终态见阶段67。四项成本均已终态，不能把整个S3进行中成本当最终成本。

### 冻结配对与交互

区间列格式为delta [CI95]。正macro AP表示前项较好，正NLL表示前项较差。

| Contrast | Raw macro AP | Common-support macro AP |
| --- | --- | --- |
| Fine native − 4 mm native | −0.011014083 [−0.014274134, −0.008500752] | −0.010573706 [−0.014668787, −0.007620121] |
| Fine 1 mm − 4 mm 1 mm | −0.071516517 [−0.079198341, −0.064878833] | −0.064172735 [−0.072241079, −0.057373601] |
| Fine native − fine 1 mm | 0.000000000 [0, 0] | 0.000000000 [0, 0] |
| 4 mm native − 4 mm 1 mm | −0.060502435 [−0.067034736, −0.054427614] | −0.053599029 [−0.060528169, −0.047309640] |

| Contrast | Raw NLL | Common-support NLL |
| --- | --- | --- |
| Fine native − 4 mm native | +0.002254871 [+0.001340165, +0.003073414] | +0.001697197 [+0.000663641, +0.002702738] |
| Fine 1 mm − 4 mm 1 mm | +0.029383883 [+0.025887736, +0.032884795] | +0.026289620 [+0.023032601, +0.029833224] |
| Fine native − fine 1 mm | +0.000000093 [−0.000001677, +0.000001851] | +0.000000093 [−0.000002027, +0.000002115] |
| 4 mm native − 4 mm 1 mm | +0.027129106 [+0.023962508, +0.030459314] | +0.024592516 [+0.021460796, +0.027951863] |

交互定义严格为`(fine_native − fine_1mm) − (4mm_native − 4mm_1mm)`。

| Interaction metric | Raw delta [CI95] | Common-support delta [CI95] |
| --- | --- | --- |
| Macro AP | +0.060502435 [+0.054427614, +0.067034736] | +0.053599029 [+0.047309640, +0.060528169] |
| NLL | −0.027129013 [−0.030459442, −0.023962492] | −0.024592423 [−0.027952507, −0.021461866] |
| Macro F1 | +0.146484156 [+0.138007679, +0.155816988] | +0.144991121 [+0.135808811, +0.154429313] |
| Balanced accuracy | +0.061400000 [+0.053720289, +0.070249461] | +0.060255778 [+0.051522429, +0.069949013] |
| AP e | +0.138279126 [+0.123185481, +0.155596392] | +0.132954209 [+0.117435582, +0.150885892] |
| AP mu | +0.047521662 [+0.035882339, +0.058911968] | +0.042059515 [+0.029606656, +0.054365750] |
| AP tau | +0.024540120 [+0.016211613, +0.032265691] | +0.014836443 [+0.005872294, +0.023055678] |
| AP NC | +0.031668830 [+0.021660056, +0.040877948] | +0.024545950 [+0.013314372, +0.036813979] |

4 mm窗口的native−1 mm在表中八项指标均劣于零方向；AP e/mu/tau/NC的raw差分别为−0.138279126/−0.047521662/−0.024540120/−0.031668830，其区间为上述交互AP区间取负并交换端点。Fine两项AP差为零，源于各自对所有事件给出常数score，不能解释为原生与coarse输入等效。

### 学习曲线与优化状态

| Window / pixel representation | Epoch 1 macro AP | Epoch 2 | Epoch 3 | Epoch 4 |
| --- | ---: | ---: | ---: | ---: |
| 16-pitch / native | 0.248447945 | 0.250000000 | 0.250000000 | 0.250000000 |
| 16-pitch / 1 mm | 0.248754312 | 0.248297109 | 0.250000000 | 0.250000000 |
| 4 mm / native | 0.250619803 | 0.252018204 | 0.259987409 | 0.261014083 |
| 4 mm / 1 mm | 0.271294480 | 0.274224186 | 0.308823180 | 0.321516517 |

完整validation上，4 mm/native的argmax计数[e, mu, tau, NC]为[7644, 1, 2353, 2]，1 mm为[2362, 4529, 1577, 1532]。两个fine模型均[0, 0, 0, 10000]。970个不同score不能证明4 mm/native已经充分学习；其mu/NC argmax几乎缺失，但排序AP仍分别0.2597/0.2630。

以下为同8事件、第四轮best checkpoint的CPU FP32诊断。Gradient列是分类损失的梯度平方范数，没有更新参数。

| Window / pixel representation | Head first linear mean | GELU exact-zero fraction | Input gradient squared norm | Module-read gradient squared norm | Distinct score vectors / 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-pitch / native | −44.972729 | 1.000000000 | 0 | 0 | 1 |
| 16-pitch / 1 mm | −34.972504 | 1.000000000 | 0 | 0 | 1 |
| 4 mm / native | −24.818609 | 0.990885437 | 3.624865e−12 | 3.282716e−5 | 5 |
| 4 mm / 1 mm | −7.108082 | 0.648437500 | 8.068929e−5 | 5.783104e−3 | 8 |

Fine/native和fine/1 mm的head第一层最大值分别−13.976823/−13.882498；随后8×192个GELU输出均精确为零。两个模型的input、local、module和context等上游分类损失梯度均为零。该状态在FP32中仍出现。4 mm/native不是完全常数模型，在此小样本上有非零上游梯度，但分类头99.09%的GELU输出为零；该观测与受限学习相符，不能单凭它量化饱和对完整validation差异的因果贡献。

### Screen、后端与成本

17:40:40 UTC的monitor：10/17项完成、10项CPU读回通过，共140个检查；其余7项仍在运行或排队。含tracker配置也已完成，macro AP0.368196335、NLL1.324485925、macro F1 0.357889767、balanced accuracy0.3596，四类AP为[0.452587225, 0.376318814, 0.290370228, 0.353509074]。其改善相对的是已塌缩的primary配置，尚不能作为独立、稳定的tracker净物理增量。

17:42:27 UTC的非strict评价含10项S3和16项只读基线，共26项结果；15/22个冻结单seed配对与1项交互已可计算，另有7项实验、7项配对pending。全部可计算配对与交互均保留raw/common support各34项指标、400个有效replicates。没有据此提前筛选确认候选。

17:41:39 UTC成本快照：实际累计13.982794144 GPUh，含首次交互profile保守上界为14.163936758 GPUh，missing history0；组织预算余量下界105.836063242 GPUh。失败及恢复attempt继续计入。此为进行中快照。

同轮后端查询中，Condor `1573835.8/.9`正在运行task9/10，其余5项idle；Kubeflow `pinpoint-s3-anyqueue-slot-v1`仍Pending、没有node。Scheduler报告H100匹配节点的GPU/CPU/RAM资源不足，以及其它节点的affinity/taint限制。继续保留同一Kubeflow Job的排队位置及唯一dispatch watcher；真实CUDA/BF16 ready后才从Condor交接idle任务。已Running任务原地继续，未重复提交训练。当前CERN flavor仍为`h100-94gb`对应的`NVIDIA-H100-NVL`、整卡`nvidia.com/gpu:1`，admission分配46 CPU和127G内存。Job时限43,200 s包含排队，训练四轮和120 GPUh组织预算保持原合同。

### 复现与产物

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S3_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12
python -m analysis.flavour.s3.accounting --root "$S3_ROOT"
python -m analysis.flavour.s3.evaluate --root "$S3_ROOT" --campaign roadmap/configs/s3_screen.json --comparisons roadmap/configs/s3_pairs.json --output "$S3_ROOT/evaluation_factorial_progress" --bootstrap 400
# 对四单元分别运行，只读best checkpoint；四个EXPERIMENT名见下文。
python -m analysis.flavour.s3.diagnose --root "$S3_ROOT" --experiment "$EXPERIMENT" --checkpoint best --output "$S3_ROOT/diagnostic_factorial_${CELL}_best.json"
```

四项`EXPERIMENT`为`s3_{f16,a4}_context_latent{,_1mm}_s42_e4`的四种展开；`CELL`依次为`f16_native`、`f16_1mm`、`a4_native`、`a4_1mm`。Bash执行时为每项显式设置这两个变量。

产物相对上述结果根：

- `evaluation_factorial_progress/summary.json`：所有四类、配对、交互、条件域、预测分布和pending清单。
- `evaluation_factorial_progress/bootstrap_statistics.npz`：各模型共享重采样下的34指标统计，不保存重复事件表。
- `evaluation_factorial_progress/four_class_diagnostics.pdf`：英文混淆矩阵、ROC/PR和学习曲线；`four_class_resource_tradeoff.pdf`为英文资源图。
- `diagnostic_factorial_{f16_native,f16_1mm,a4_native,a4_1mm}_best.json`：真实事件键、逐层范围、梯度与8事件概率。
- `factorial_backend_snapshot.json`：本轮精简后端队列快照；`campaign_accounting.json`和`monitor_state.json`继续随进度更新。

以上评价、四项诊断及完成模型的输出读回均实际执行成功。未测量：本2×2的训练seed总体波动、首次饱和step、归一化或优化修复的因果效应、真实硬件标定；不以这些未测量项目重新否定用户接受的输入与proxy。

## 阶段性结论

事实：当前seed42和冻结四轮下，4 mm/1 mm比4 mm/native的四类AP、macro F1、balanced accuracy和NLL均更好，raw和固定E/vertex支持下方向一致。两个fine模型的完整validation预测已塌缩，且所检8事件分类损失到编码器的梯度为零。

正交互主要表达“fine下两者均失败，4 mm下coarse较好”这一训练结果。它不支持原生与1 mm输入的物理等效，也没有确立4 mm压缩造成原生信息优势消失。4 mm/native的渐进学习、少量score向量与分类头大面积零输出，应与其较低指标一起报告。

待检验假设：输入表示改变了局部节点数、汇总量和分类头的数值条件，当前归一化/优化设置可能使native更易饱和。需要显式版本的归一化或激活干预及匹配训练对照，才能分离这些因素。当前17项screen保持冻结；本阶段不插入修复训练。

## 下一步计划

完成剩余7项screen及真实终态成本后，按冻结规则选两个候选，再与primary native/1 mm去重，执行seed17/2026确认。最终同时报告训练spread和共享事件区间，不把事件CI当训练随机性。

继续“哪个后端先实际获配就用哪个”的调度。现有watcher有独立时限，后续若到期先确认旧进程退出再续接；保留已有Pending Job，避免重排队。最终完整四类、三seed、模态、条件域、错误显示、资源和strict gate齐备后再结束S3；S4–S6仍未执行。后续若开展优化修复，另冻结版本、对照和预算。
