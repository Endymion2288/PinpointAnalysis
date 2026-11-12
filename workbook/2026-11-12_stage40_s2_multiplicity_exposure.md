# 阶段 40：S2 粗格实际移除的信息与模型差异

1mm OR在validation的5,537个事件中移除了multiplicity，占7,688个pixel-active事件的72.02%；按全部active pixel通道汇总，4,085,982个native pixels对应2,635,984个occupied cells，移除的额外计数为1,449,998（35.49%）。因此，[阶段39](2026-11-12_stage39_s2_coarse_response_controls.md)两族count−OR τ AP区间跨0，不能简单归因为粗格归约几乎没有改变输入。该测量量化信息处理本身，不证明被移除的计数具有额外分类能力。

## 阶段目标

回答OR/count干预覆盖多少事件、信息移除程度如何随粒度/类别/域变化，并区分表示处理与重新训练模型的输出变化。此项是在主要OR结果后增加的探索性解释诊断，不改变35项训练和41配对+4种子均值合同。

## 工作内容

新增`analysis.flavour.s2.response_diagnostics`，直接复用当前`pixel_view`，读取accepted validation的原生pixel通道；不重新读取生产ROOT、不认证样本或proxy、不拟合新模型。计算100μm、200μm、1mm reference-origin及1mm半格平移的active native数、occupied-cell数、merged-cell数、最大格内multiplicity和excess fraction。标签/域仅用于输出分层，不进入表示处理。

每个shard只读取pixel和必要的provenance列，使用2个process并行，Arrow/BLAS线程各为1，与CPU申请匹配。输出合并为一个40k-row Parquet及一个JSON，没有逐事件小文件。两个已训练模型族的count/OR预测按run/chunk/row/label对应，分层列出τ prior与AP点估计，不额外生成探索性显著性结论。

提交前实查schedd负载：`bigbird21`的idle更少但拒绝当前直接EOS路径，`bigbird20`的EOS policy=True且WRITE ALLOW，按兼容者最小idle选择后提交`1573792.0`。JobStatus=4、ExitCode=0、NumJobStarts=1，实际节点`b9p14p4348.cern.ch`。运行基于`67dbfea`和本次新增模块；没有改动正在训练的模型或冻结配置。

## 详细数据

每种表示均为10,000个validation events，每类2,500；pixel-active=7,688、pixel-empty=2,312。这里pixel-empty允许bar响应，不能等同于pixel+bar输入空的440个事件。

定义`excess_fraction=(N_native−N_occupied)/N_native`，pixel-empty约定为0。这个分数描述OR丢失的额外active-pixel计数；空间内部布局在count和OR中都已由粗中心替代。它不是模拟能量损失，也不是该比例的分类信息损失。

| Representation | Events with merged cells | Fraction of pixel-active events | Occupied cells, pooled | Excess native counts, pooled | Pooled excess fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100μm | 5,032 | 65.45% | 3,848,578 | 237,404 | 5.8102% |
| 200μm | 5,223 | 67.94% | 3,637,589 | 448,393 | 10.9739% |
| 1mm reference origin | 5,537 | 72.02% | 2,635,984 | 1,449,998 | 35.4871% |
| 1mm half-cell shift | 5,527 | 71.89% | 2,637,410 | 1,448,572 | 35.4522% |

每行pooled native均为4,085,982。1mm reference-origin的pixel-active事件excess fraction分位数：median=0.10526316、p75=0.22580645、p90=0.38947965、p99=0.65731239；最大格内multiplicity的median=3、p90=30、p99=160.13、max=469。分位数插值可为非整数。1mm平移的最大格内multiplicity=549；pooled统计接近不意味着两种原点具有相同的逐事件cell分配。

### 1mm reference origin的类别与域

| Stratum | Events | Pixel-active | Events with merges | Pooled excess fraction | Median excess fraction among active events |
| --- | ---: | ---: | ---: | ---: | ---: |
| CC nu_e | 2,500 | 2,020 | 1,591 | 42.9612% | 0.17320513 |
| CC nu_mu | 2,500 | 1,957 | 1,335 | 30.0195% | 0.08333333 |
| CC nu_tau | 2,500 | 1,845 | 1,334 | 28.7170% | 0.09600000 |
| NC | 2,500 | 1,866 | 1,277 | 29.0823% | 0.08223513 |
| Pixel-centered | 885 | 882 | 855 | 43.3243% | 0.22511967 |
| Large scintillator | 4,494 | 4,379 | 3,897 | 40.4867% | 0.16374269 |

电子类及中心域的通道汇总移除比例更高，说明干预覆盖随类别和接受度变化。它们的能谱、占用和shower构成也不同；当前测量没有分离这些原因，不将差异直接命名为某种已确认的物理拓扑。

### 按干预程度分层的预测差异

以下为同一1mm reference-origin分组内重新训练的count−OR τ AP点估计；区间未测量。各域prior不同，不能把不同域的AP大小直接当作模型能力排序。

| Input group | Events / tau | Tau prior | ParT ΔAP | ParticleNet ΔAP |
| --- | --- | ---: | ---: | ---: |
| Pixel empty | 2,312 / 655 | 0.28330 | −0.00379789 | +0.01100198 |
| Pixel active, no merge | 2,151 / 511 | 0.23756 | −0.00901536 | +0.01907428 |
| 0 < excess fraction ≤ 0.25 | 3,905 / 1,015 | 0.25992 | −0.00339518 | −0.01268061 |
| 0.25 < excess fraction ≤ 0.75 | 1,623 / 318 | 0.19593 | +0.00441594 | −0.00419133 |
| Excess fraction > 0.75 | 9 / 1 | 0.11111 | 0 | 0 |

无合并的pixel-active域在count与OR下拥有相同输入，但两族平均绝对τ score差分别为0.02113269/0.01845904；pixel-empty域也分别为0.01176598/0.01012862。原因在比较定义上明确：这是在不同训练表示分布上重新拟合的两个模型，权重可以不同；并非在同一已训练模型上逐事件开关信息。因而这些域的预测差不是该事件自身丢失信息的直接归因。最强干预组只有9个事件/1个τ，不能从其AP稳定作结论。完整组内score差写入JSON。

### 验证、产物与运行成本

40k行表的event/representation keys无重复，四种表示各10k行且各类2,500；`N_native−N_occupied=N_excess`、excess非负、merged-cell内pixel数≥excess均逐行成立。用真实ROOT lattice上相邻的3个native pixel作独立手算例：落入同一1mm格，native=3、occupied=1、excess=2，count特征为3、OR为1，与实现一致。这些检查只验证新增诊断表及变换含义，没有重设数据验收。CDF和类别分布图已实际查看，英文标注完整。

输出目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/response_diagnostics/`：`coarse_multiplicity.parquet`、`summary.json`、`output_readback.json`、`coarse_multiplicity.png/pdf`。提交证据为上一级`submission_response_diagnostics_073436.json`，日志在`h100/logs/response_diagnostics.1573792.*`。

程序wall=119.79587 s，actual allocation=138 s×2 CPU=0.07666667 CPU-core-hours，GPU allocation=0；Condor记录RemoteUserCpu=186 s。新任务会随下一次统一核算进入组织台账。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.submit response_diagnostics --root "$S2_RESULTS/h100"
```

已有完成输出无需重提。训练仍为正式15 complete、6 running、14 idle，下一项临近完成者为ParticleNet平移；Kubeflow v7仍Pending且无node。`campaign_output_inventory.json`还汇总当前35项配置的既有读回/评价/成本证据，15个完整模型均匹配冻结配置、14项pass且成本完整；不是新的样本认证。

## 阶段性结论

事实：粗格OR确实移除了广泛且可测的multiplicity，尤其在高占用与电子类输入中；两族模型的inclusive τ结果仍没有count正收益支持。信息处理的强度与有限训练模型能否利用信息是两个实测问题，不能互相替代。

待检验解释：bar和occupied-cell形态可能已编码相关信息，或者当前容量、局部压缩与优化预算限制了额外计数的利用。当前数据未区分这些机制；事后分层差异没有配对CI，不能替代冻结的主配对结论。

## 下一步计划

继续完成ParticleNet平移、两族shuffle/pair/graph/plane控制、模态和训练域以及额外种子。结合主要事件区间、各类/分支/接受度和资源完成最终解释，执行无`--allow-incomplete`的严格评价后验收S2；不开始S3–S6。
