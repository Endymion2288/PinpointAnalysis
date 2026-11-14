# S2 运行状态与复现入口

S2已完成并验收。35项H100/LCG108/BF16科学配置均四轮完成、各14项CPU读回pass；严格评价与`--require-complete`真实exit0。最终39结果、41单seed配对、4三seed均值、2项交互、0 pending，组织终态成本92.73055556 GPUh，149条后端记录全部终止、无缺失历史。[阶段55总报告](../../../workbook/2026-11-12_stage55_s2_final_acceptance.md)与[版本化数值表](../../results/s2_summary.json)给出完整结论。沿用用户接受的samples/split/proxy。本文记录S2结束时的calibration/test冻结状态；后续S3–S6已完成，见[S6最终结果](13_s6_final_evaluation.md)。下文运行记录为历史快照。

## 当前产物

结果根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11`。正式科学根目录为上述父目录的 `h100/`。代码：`src/analysis/flavour/s2/`。配置合同：[s2.yaml](../../configs/s2.yaml)。

| Artifact | State / location |
| --- | --- |
| Frozen subset | `manifest.parquet`，80k train + 10k validation，各类相等；不改既有 split |
| Complete observations | `shards/000`–`031`，约 1.765 GB combined Parquet，原生 pixel、bar、accepted tracker |
| Occupancy / diagnostics | `catalog.parquet`、`catalog_summary.json`，保留 2,535 全空事件 |
| Information toys | `toys/`，420 点曲线、projection pairing collision、双轨/kink/shower support |
| Coarse response exposure | `h100/response_diagnostics/`，10k validation ×4表示；1mm OR在72.02%的pixel-active事件移除multiplicity，pooled excess fraction=35.49%；事后score分层和解释范围见[阶段40](../../../workbook/2026-11-12_stage40_s2_multiplicity_exposure.md) |
| B0 / prior reference | `experiments/b0_*_s42` 和 `truth_energy_vertex_s42`，均已完成 |
| Paired reference evaluation | `evaluation/summary.json`、`metrics.csv`、`tau_PR.png`、`error_event_groups.parquet` |
| Comparison report figures | `h100/evaluation_final/`粒度、配对/等allocation、三seed均值/spread图完整；跨层控制图见阶段54 |
| Native response displays | `evaluation/event_displays/b0_all_s42/`，20 页 PDF、事件 keys 和首例 PNG |
| V100 resource study | `resource_profile/` 为 v1；`resource_profile_v2/` 的 576 行已完成，两者分别保留 |
| H100 resource comparison | `1573759.0` exit0，H100/LCG108/BF16 共 576 行；与 V100 匹配后去 warmup 保留552行，batch1 training compute 比值1.69–2.33；原始/配对图已查看，见[阶段 24](../../../workbook/2026-11-11_stage24_s2_alternative_gpu_profile.md)；后续两族整轮门槛已通过，见下一行 |
| H100 selected execution | 两族完整首轮于2026-11-12 00:01:53 UTC总体pass。ParT=2,223.38 s（V100的45.76%），ParticleNet=1,604.22 s（54.60%），均20,016/20,016 applied、0 skips、23项检查通过；无AP/accuracy/NLL硬件选择，见[阶段25](../../../workbook/2026-11-11_stage25_s2_h100_epoch_gate.md) |
| H100 complete matrix | [冻结的35项同配置合同](../../configs/s2_h100_v2.json)；原tasks0/6保留为 `1573760.0/1`，其余33项已提交为 `1573782.0–32`，schedd=`bigbird20.cern.ch`。`h100/activation.json` 保存门槛、完整task/proc映射、优先级和移除旧idle证据；正式评价读取 `h100/` |
| First completed H100 model | `1573760.1` exit0，ParticleNet native joint四轮完成，80,064 applied、0 AMP skips，12项最终checkpoint/概率读回通过；τ AP=0.35510、NLL=1.23931、allocation=1.75278 GPUh。正式部分评价为5 results、1 pair/44 pending；相对B0 raw ΔAP=+0.01039，CI=[−0.00250,+0.02294]，固定common-support ΔAP=+0.01358，CI=[+0.00324,+0.02458]。20页原生图与代表性错误页已查看；见[阶段29](../../../workbook/2026-11-12_stage29_s2_first_h100_model.md) |
| V100 supplementary execution | 九项辅助训练均已完成，26个被替代idle已移除；两族四格及ParT 100 μm的实际结果、全部尝试成本见阶段28。不替代H100正式配置 |
| Kubeflow execution | `ml-prod/xcheng`登录正常；CPU路径与GPU flavor Pod server dry-run已通过。v9始终无node，全部任务在Condor运行后主动结束释放配额，0 GPUh；GPU runtime未测量，见[flavor说明](08_kubeflow.md) |
| Spatial screen | [current 12-run configuration](../../configs/s2_spatial_screen_v2.json)；v1 在首个 epoch 完成前停止，计费保留 |
| Spatial paired evaluation | 12配置/18空间配对完整，统一纳入`h100/evaluation_final/`的41项最终配对 |
| Prespecified controls | [13 runs](../../configs/s2_controls_v2.json)：两族 OR/shift/shuffle、各自 pair/graph，以及 ParticleNet modality/domain 控制 |
| Three-seed confirmation | [8 additional runs](../../configs/s2_confirmation_v2.json)：两族 native/1mm joint × seed17/2026；复用已有 seed42 |
| Original comparison contract | [39 individual contrasts and 4 three-seed means](../../configs/s2_final_pairs_v2.json)；原有 33 项科学训练，保留此文件以对应既有部分评价 |
| Complete S2 contract | [41 individual contrasts and 4 three-seed means](../../configs/s2_complete_pairs_v2.json)；另含两族 [plane-only controls](../../configs/s2_cross_plane_controls_v2.json)，共 35 项科学配置，见[阶段 21](../../../workbook/2026-11-11_stage21_s2_cross_plane_control.md) |
| Plane-only GPU execution | 正式H100 `1573782.31/32`均四轮/exit0、14项读回pass；早期smoke只作执行验证，见阶段54 |
| Historical / branch diagnostics | 最终39结果、117历史三类条件表、195行τ分支、1051行全条件诊断；历史三类并非重训旧模型 |
| Historical drawing integration | `1573758` exit0（commit `9d67e8a`），当时4个 references、45项 pending；后续科学评价见下一行 |
| First scientific model / current evaluation | `1573747.6` exit0，ParticleNet native-joint 四轮完成，τ AP=0.35784；相对B0 raw ΔAP=+0.01313，CI95=[−0.00097,+0.02446]；fixed-common ΔAP=+0.01862，CI=[+0.00686,+0.02971]。22:07 UTC 本地部分评价正常退出，5个完成结果、1项配对/44项 pending；20页原生图和3个错误页已查看，见[阶段27](../../../workbook/2026-11-11_stage27_s2_first_scientific_model.md) |
| Second scientific model / updated evaluation | `1573747.1` exit0，ParT native-projections 四轮完成，τ AP=0.34989、NLL=1.22890、allocation=3.84833 GPUh。最新部分评价正常退出，6个完成结果、160行分域指标、1项配对/44项pending；新resolution图已查看。22:23 UTC为2 complete/5 running/28 idle，见[阶段28](../../../workbook/2026-11-11_stage28_s2_spatial_results.md) |
| Third scientific model / latest evaluation | `1573747.2` exit0，ParT 1mm joint四轮完成，τ AP=0.35845、NLL=1.22922、allocation=4.71833 GPUh。部分评价为7个完成结果、187行分域指标、21个历史三类诊断，仍为1项配对/44项pending；resolution图已查看。23:15队列3 complete/4 running/28 idle，见阶段28 |
| Fourth scientific model / first spatial contrasts | `1573747.3` exit0，ParT 1mm projection四轮完成，τ AP=0.35210，allocation=3.84250 GPUh。8个完成结果、214行分域指标、3项配对/42项pending；1mm joint−projection AP=+0.00634，pointwise CI=[+0.00101,+0.01250]；native−1mm projection=−0.00221，CI=[−0.00935,+0.00544]。原`.7`已接续运行，累计23.56117 GPUh且missing=0，见阶段28 |
| Fifth V100 model / complete ParT factorial | `1573747.0` exit0，native joint四轮完成，τ AP=0.35070、NLL=1.21797、allocation=5.70722 GPUh（含中断）。9个完成结果、241行分域指标、7项配对/38项pending；native pairing ΔAP=+0.00081，CI=[−0.00570,+0.00729]；native−1mm joint=−0.00774，CI=[−0.01397,−0.00010]，固定common-support区间跨0；交互效应−0.00553，CI=[−0.01488,+0.00375]。更新图已查看，01:10组织累计34.92273 GPUh、missing=0，见阶段28 |
| Input response domains | `1573754` 已完成；新增 model-input-active/empty 与 tracker-only，pixel+bar validation 输入空为 440 events，三探测器全空为 282；见[阶段 19](../../../workbook/2026-11-11_stage19_s2_input_response_domains.md) |
| First full scientific epoch | ParT 1mm projection epoch1：τ AP=0.30649、NLL=1.32943、3,063.325 s；继续既定 4 epochs，尚未作为完成模型纳入配对，见[阶段 20](../../../workbook/2026-11-11_stage20_s2_first_complete_epoch.md) |

完整 B0 的 τ AP=0.34471、1% FPR 下 τ efficiency=65/2,500=2.60%；truth E/vertex 参照 AP=0.44106。25% τ benchmark prior 下，粗 E/x/y/z common support 为 8,559 events、Neff=7,959.41，二者 weighted AP 为 0.30845/0.36474。truth 参照单列为先验诊断，不算硬件模型。详细 counts、CI 和条件指标见[阶段 12](../../../workbook/2026-11-11_stage12_s2_reference_results.md)。

## 科学训练合同

两家族共享当前表示的局部 learned compression（每块最多 32 nodes）和全部全局块；这是明确的 baseline adaptation，不是未经压缩的全事件 ParT。模型输入为 18 项 observation 特征及其导出的 8 项计数/信号和，后者在局部汇聚与全局头显式保留。只从当前表示计算，不读取 raw counts 恢复 OR 丢失的 multiplicity。truth/provenance 仅做 split/诊断。粗表示不携带原生 cell 残差，投影移除 x–y pairing；空事件保持先验型输出。

v2 ParT dim48=151,228 parameters；ParticleNet dim64=161,156，相差 6.56%。空间筛选每 run 4 epochs、batch4、20,016 attempted steps/epoch、相同 seed42 与数据顺序；记录实际 applied steps 和 AMP skips。bucket256 用同一原生通道数排序，完整覆盖后重复少量 batches 匹配步数。模型按 validation NLL 选 best；每 epoch 四类概率支持同型号 GPU、共同时间预算内的比较。

正式完整矩阵固定NVIDIA H100 NVL，复用 `LCG_108_cuda/x86_64-el9-gcc13-opt`：Python3.12.11、Torch2.7.1、BF16；依据为预先冻结的双家族完整首轮门槛，已总体通过。早期V100辅助结果使用LCG107/Python3.11.9/Torch2.5.1/FP16。下面保留V100先导首次提交与部分评价的复现入口，存活或已移除任务均不按此重新提交；正式最终命令见后文。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
S2_CAMPAIGN="$PWD/roadmap/configs/s2_spatial_screen_v2.json"
# 仅用于首次提交；已存活的任务不重复提交。
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS" --campaign "$S2_CAMPAIGN" \
  --tasks 0 6 --lcg-view LCG_107_cuda --gpu-device 'Tesla V100-PCIE-32GB'
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS" --campaign "$S2_CAMPAIGN" \
  --tasks 1 2 3 4 5 7 8 9 10 11 --lcg-view LCG_107_cuda --gpu-device 'Tesla V100-PCIE-32GB'
# 部分 runs 完成时只报告已有结果，并显式保留 pending；最终验收去掉 allow-incomplete。
python -m analysis.flavour.s2.submit evaluate --root "$S2_RESULTS" --evaluation-tag evaluation_spatial \
  --comparisons "$PWD/roadmap/configs/s2_spatial_pairs_v2.json" --allow-incomplete
# 某任务终止后，确认队列无存活副本，再仅重提该 index；自动恢复 atomic last epoch。
# 已完成同名/同配置任务会直接返回，不重新训练。
```

提交器每次保存当前 schedd/EOS/WRITE/GPU pool、精确 task/config 映射与最终 ClusterId；训练 `manifest.json` 保存代码、软件、参数、样本数。`progress.json` 每 250 steps 更新；`history.json` 为完成的 epochs；`last.pt` 为恢复依据，`best.pt`/`validation_predictions.parquet` 为 selection 结果。中途停止不可视作完成 epoch。`s2.accounting` 只查询本 S2 目录记录的 ClusterIds，保留失败/重启成本，汇总到 `campaign_accounting.json`；时间匹配将 history 外的 allocation 成本也计入。

对照在读取空间矩阵完整 validation 结果前冻结。modality/domain 最小控制预先选用实测计算较省的 ParticleNet，结论以该 backbone 为条件。空间初筛上限 100 GPU-hours，确认与控制追加上限 100，总组织上限 200，旧失败/替代尝试计入。每次监测以实际 allocation 为准，并在首个完整 epoch 后更新完成成本估计；这些数值是组织上限，不是已使用算力。

三种子均值比较对 seed17/42/2026 的 **AP 差值**取平均，每次 bootstrap 对三者使用相同物理事件块；不是把三份 prediction 拼成 30k events，也不是 ensemble。分别报告事件区间、逐 seed 差值及训练 spread。固定 common-support weights 的区间为条件区间。最终评价使用 `s2_complete_pairs_v2.json` 并去掉 `--allow-incomplete`，缺少任一指定结果时不能声称完成。旧 39-pair 合同保留对应当时的部分评价。

18:44 UTC 提交状态：空间 `1573747.0–11`、控制 `1573750.0–12`、额外种子 `1573751.0–7`，均在 `bigbird20.cern.ch`。当时 4 running/29 idle。`.6/.8` 的空间 JobPrio=10，其余尚未启动空间项=5，追加任务=0，以便先获得完整筛选证据。`1573752` 为 `evaluation_final/` 的部分结果集成检查，显式启用 `--allow-incomplete`。这些 ClusterIds 仅用于查状态；存活任务不再次提交。

20:17:53 UTC，跨层控制已通过专用 H100/BF16 smoke 后提交为 `1573757.0/1`，正式任务仍是 V100/FP16。对应记录 `submission_train_201753.json` 保留新的 schedd 负载、EOS/WRITE 检查和配置。35 项科学配置均已提交；20:18 UTC 为 5 running、30 idle，尚无四轮科学 run 完成。两项 smoke 的 terminal allocation 共 833 s，按实际成本计入；小样本执行结果不进入分类主表。

20:43:59/20:45:19 UTC，`1573747.3/.0` 因 `Startd was shutdown`（1013）自动返回 idle，原 JobId 保留。分别可从 epoch2/epoch1 恢复，CPU checkpoint readback 已通过；只将这两个 idle JobId 的 JobPrio 提高到20，等待原任务重新匹配 V100。20:47 UTC 为 3 running、32 idle。中断 allocation 纳入成本，未进入已提交 checkpoint 的部分共 1.50795 GPU-hours，含准备与部分 epoch；见[阶段 23](../../../workbook/2026-11-11_stage23_s2_startd_shutdown.md)。当前执行状态以 `condor_q` 为准，重排后训练 manifest 可暂留 `running`。

正式H100全部训练完成后的入口：先在父目录更新所有历史成本（自动写出H100成本视图），再对 `h100/` 严格评价与两族原生错误显示。

21:44:20/21:50:04 UTC，原 `1573747.0/.3` 已分别从 epoch1/epoch2 在 V100 恢复，optimizer/AMP counters 与保存状态衔接一致；见阶段23补充与 `startd_shutdown_resume_execution.json`。Kubeflow 自21:58起已有成功 CPU 环境证据，GPU资源仍待调度；用户已授权在Condor排队时使用。`s2.accounting` 现按记录的 Job UID 合并两个后端，未分配节点的等待为0，已分配 Pod 的启动/初始化/失败成本保留。

22:40:04 UTC，`.3`恢复后已提交完整epoch3，并进入epoch4。CPU读回的新checkpoint参数、optimizer、概率、history/config和恢复状态8项检查通过，见 `startd_shutdown_post_resume_checkpoint.json` 与阶段23。22:35:58 UTC累计18.97207 GPU-hours，missing history records=0；5项V100科学训练继续。

22:43:49 UTC，仍idle且未启动的 `1573747.7/.9` 原任务JobPrio从5提高到10，优先补齐ParticleNet native/1mm双投影这两个主factorial成员；其余6个主factorial成员已完成或运行中。依据实验角色调整顺序，不改模型/数据/步数，也不重提任务。证据为 `primary_factorial_priority.json`，见阶段28。

22:54:35 UTC，`.0`恢复后的完整epoch2也已CPU读回通过8项检查，累计40,012 applied/20 skips，随后进入epoch3；见 `startd_shutdown_native_post_resume_checkpoint.json`。两项Startd中断任务均已提交并读回恢复后的新完整checkpoint，四轮科学训练继续。

```bash
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.submit evaluate --root "$S2_RESULTS/h100" \
  --evaluation-tag evaluation_final --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json"
# 等上述评价成功后提交 displays。
python -m analysis.flavour.s2.submit display --root "$S2_RESULTS/h100" \
  --evaluation-tag evaluation_final --experiment part_native_joint_s42_e4_v2
python -m analysis.flavour.s2.submit display --root "$S2_RESULTS/h100" \
  --evaluation-tag evaluation_final --experiment pn_native_joint_s42_e4_v2
```

02:08 UTC，第二个正式 H100 模型 ParT native joint（`1573760.0`）exit0，四轮完成、12 项 checkpoint/概率读回通过。τ AP=0.35858；ParT−B0 raw ΔAP=+0.01387，CI95=[+0.00152,+0.02523]，fixed-common ΔAP=+0.01539，CI=[+0.00636,+0.02482]。ParT−ParticleNet raw ΔAP=+0.00348，CI=[−0.00369,+0.01036]，架构优劣尚未确定。正式部分评价为 6 个结果、3 项配对/42 项 pending。两族正式 native 的 20 页错误事件图均完成并实际查看固定混淆页；阶段证据见[阶段 30](../../../workbook/2026-11-12_stage30_s2_native_h100_comparison.md)。正式队列为 2 complete、4 running、29 idle，S2 尚未完成。

02:31 UTC，正式 ParT native projection（`1573782.0`）exit0，τ AP=0.35733812；best checkpoint/10k predictions 一致，13 项读回中 12 项通过，epoch 概率表的 EOS 两个副本写回不完整。原节点恢复 `1573790.0` 等待资源；输出改为 flush/fsync 后原子发布，四次 EOS 替换与直接 XRootD 读回通过。当前正式部分评价为 7 个结果、4 项配对/41 项 pending；native joint−projection ΔAP=+0.00124580，CI=[−0.00501470,+0.00830354]。完整输出状态仍待恢复，见[阶段 31](../../../workbook/2026-11-12_stage31_s2_output_integrity.md)。V100 辅助已完成 8 个模型，两族四格均已评价，剩 1 项辅助训练，新增结果见阶段 28。

02:37 UTC，原节点恢复 `1573790.0` 因 GPU 被其他任务占用而无法匹配，已在 0 starts/0 allocation 时移除。projection 原尝试已完整归档至 `h100/recovery_projection/original_training_attempt/`；正式同名目录等待冻结 task1 替换 `1573791.0`，使用输出修复 `d9a5d77`，JobPrio=30。替换依据是两副本文件不完整，原 5,618 s 与新尝试成本共同保留。正式已完整模型数为 2，先前 7-result 评价的 projection 数值属于归档原尝试；替换完成后更新有效主表，见阶段31补充。

02:55 UTC，正式 ParT 1 mm joint（`1573782.1`）exit0，14 项 CPU 读回通过，40k 行表与原节点同步副本逐值相同。τ AP=0.35502857；native−1 mm joint ΔAP=+0.00355534，CI95=[−0.00186106,+0.00868839]，fixed-common ΔAP=+0.00297513，CI=[−0.00192093,+0.00846325]。当前 seed42/四轮/表示条件下可排除至少 +0.01 的 AP 收益，跨种子与机制待控制。正式有效部分评价为 7 个结果、4 项配对/41 项 pending，原归档 projection 不进入当前主表；详见[阶段 32](../../../workbook/2026-11-12_stage32_s2_joint_resolution.md)。当前正式 3 complete、2 running、30 idle。

03:10 UTC，正式 ParticleNet native projection（`1573782.5`）与 ParT 1 mm projection（`1573782.2`）均 exit0，14 项读回与最终源表/原节点副本比较全部通过；两族 native joint 也已补齐当前 14 项检查。PN native joint−projection 的 raw τ ΔAP=+0.00712568，pointwise CI95=[+0.00003031,+0.01448372]，固定 common-support 区间跨 0；ParT 1 mm pairing ΔAP=+0.00081170，CI=[−0.00433001,+0.00629580]。当前正式有效评价为 9 results、6 paired/39 pending，5 complete、1 running、29 idle，见[阶段 33](../../../workbook/2026-11-12_stage33_s2_projection_comparisons.md)。V100 九项辅助全部结束，新增 100 μm 结果见阶段28；03:09:45 UTC 累计 46.45342 GPU-hours、missing=0。Kubeflow v6 仍待 GPU。

正式 `metrics.csv` / `summary.json` 现显式区分 `program_gpu_hours` 与 `allocated_gpu_hours`，并列出 allocation 秒数与额外开销。旧 `gpu_hours` 保留为程序时间别名；实际记录不完整时为空，详见[阶段34](../../../workbook/2026-11-12_stage34_s2_resource_columns.md)。原有配对的实际成本匹配继续使用同一 accounting。

04:18 UTC，ParT native projection冻结替换`1573791.0`正常结束、14项读回全部pass。正式ParT四格齐全，native pairing τ ΔAP=+0.00124580，pointwise CI=[−0.00501470,+0.00830354]；四格交互=+0.00043410，CI=[−0.00790547,+0.00875624]。四项主配对的raw/common-support区间均跨0、各自上界低于+0.01，跨种子与机制仍待控制。原尝试+替换合计10,017 s=2.78250 GPUh；projection粒度的同成本比较因无法容纳首轮而缺测。有效评价为10 results、8 paired/37 pending、1 factorial；6 complete、6 running、23 idle，组织累计52.32724 GPUh、missing=0。详见[阶段35](../../../workbook/2026-11-12_stage35_s2_part_factorial.md)。

04:26 UTC，ParticleNet 1mm projection（`1573782.7`）exit0、14项读回全部pass。双投影native−1mm的τ ΔAP=−0.00302942，pointwise CI=[−0.01277175,+0.00658906]，固定common-support区间也跨0且上界低于+0.01。当前11 results、9 paired/36 pending，7个正式模型完整；组织累计53.11281 GPUh、missing=0。完整结果、分支与同allocation比较见[阶段33补充](../../../workbook/2026-11-12_stage33_s2_projection_comparisons.md)。

04:44 UTC，ParticleNet 1mm joint（`1573782.6`）exit0、14项读回全部pass，两族正式八模型四格齐全。PN native−1mm joint τ ΔAP=+0.00206870，pointwise CI=[−0.00485307,+0.00996035]；1mm pairing为+0.00202755，CI=[−0.00647042,+0.00934805]；PN交互=+0.00509813，CI=[−0.00510744,+0.01683727]。1mm ParT−PN区间也跨0；同allocation整轮点估计与固定四轮排序不同，详见[阶段36](../../../workbook/2026-11-12_stage36_s2_both_factorials.md)。当前12 results、12 paired/33 pending、2 factorial；8 complete、4 running、23 idle，组织累计54.57364 GPUh、missing=0。Kubeflow仍待GPU，无科学任务迁移。

05:03 UTC，ParticleNet 100μm joint（`1573782.8`）exit0、14项读回全部pass，τ AP=0.35548005。native−100μm ΔAP=−0.00037532，pointwise CI=[−0.00552033,+0.00501993]；固定common-support区间也跨0且上界低于+0.01。同allocation预算内native选epoch3、100μm选epoch4，ΔAP=−0.00829455（无CI）。当前13 results、13 paired/32 pending；9个正式模型完整、3 running、23 idle，组织累计55.82911 GPUh、missing=0。详见[阶段37](../../../workbook/2026-11-12_stage37_s2_intermediate_resolution.md)。

05:15 UTC，ParT100/200μm均exit0、14项读回pass，ParT完整粒度曲线齐备。native−100μm raw τ ΔAP=+0.00832510，pointwise CI=[+0.00282045,+0.01378043]，固定common-support也支持正收益；100−200μm与200μm−1mm区间跨0，曲线不单调，解释需结合控制和训练波动。当前15 results、16 paired/29 pending；正式11 complete，PN200μm与ParT1mm OR控制运行，其余22 idle，累计56.37795 GPUh、missing=0。详见[阶段37补充](../../../workbook/2026-11-12_stage37_s2_intermediate_resolution.md)。

05:17:35 UTC，Kubeflow v6到达DeadlineExceeded，保留Job终态与scheduler证据后核算全程0 GPU allocation；`kubeflow/probe_v6_terminal.json`为终态快照。v7按相同2 CPU/4 GiB/1 NVIDIA与六小时deadline继续申请资源，实际probe仍限时300 s。科学任务未迁移，见阶段26补充。

05:35 UTC，PN200μm（`1573782.9`）exit0、14项读回pass，十二项正式空间配置全部完整，全部尝试合计20.32222 GPUh。PN200μm τ AP=0.36010319；200μm−1mm raw ΔAP=+0.00706716、pointwise CI=[−0.00223045,+0.01497597]，固定common-support ΔAP=+0.00990883、CI=[+0.00165290,+0.01894474]。固定支持上200μm也优于100μm，两族曲线均不单调；模型/种子/机制范围见[阶段38](../../../workbook/2026-11-12_stage38_s2_complete_spatial_screen.md)。当前16 results、18 paired/27 pending、2 factorial；12 complete、ParT OR/shift两项running、21 idle，组织累计57.12620 GPUh、missing=0。

06:56 UTC，ParT1mm OR（`1573782.10`）exit0、14项读回pass，正式13个科学模型完整。count−OR τ ΔAP=−0.00003083，pointwise CI=[−0.00529256,+0.00474363]；固定common-support ΔAP=+0.00253247，CI=[−0.00249439,+0.00724162]，当前模型下没有正收益支持。17 results、19 paired/26 pending；OR全部allocation=6,325 s=1.75694444 GPUh，组织累计62.68373474 GPUh、missing=0。07:02 UTC为13 complete、6 running、16 idle，ParT no-pair已接续启动；Kubeflow v7仍Pending、无node。详见[阶段39](../../../workbook/2026-11-12_stage39_s2_coarse_response_controls.md)。

07:07 UTC，ParT1mm半格平移（`1573782.11`）exit0、14项读回pass，正式14个科学模型完整。reference-origin−shift τ ΔAP=−0.00063541，pointwise CI=[−0.00587022,+0.00528142]；固定common-support为−0.00055234，CI=[−0.00760615,+0.00632436]。18 results、20 paired/25 pending；shift allocation=5,824 s=1.61777778 GPUh，累计63.75366560 GPUh、missing=0。正式14 complete、6 running、15 idle，ParticleNet fixed-graph已接续；详见[阶段39补充](../../../workbook/2026-11-12_stage39_s2_coarse_response_controls.md)。

07:21 UTC，ParticleNet1mm OR（`1573782.13`）exit0、14项读回pass，两族multiplicity控制完整。PN count−OR τ ΔAP=−0.00289658，pointwise CI=[−0.00956869,+0.00254101]；固定common-support为−0.00094003，CI=[−0.00773349,+0.00441296]。19 results、21 paired/24 pending；PN OR allocation=4,515 s=1.25416667 GPUh，累计65.22408182 GPUh、missing=0。正式15 complete、6 running、14 idle，pixel-only已接续；详见[阶段39补充](../../../workbook/2026-11-12_stage39_s2_coarse_response_controls.md)。

07:57 UTC，ParticleNet1mm平移（`1573782.14`）exit0、14项读回pass，两族OR和平移四项粗格控制完整。PN reference−shift τ ΔAP=+0.00148250，pointwise CI=[−0.00590075,+0.00831700]；固定common-support为+0.00145118，CI=[−0.00559692,+0.00809617]。20 results、22 paired/23 pending；PN shift allocation=6,187 s=1.71861111 GPUh，累计68.77621452 GPUh、missing=0。ParT shuffle随后也结束，tracker/center-training已启动，08:07为17 complete、7 running、11 idle；详见[阶段39补充](../../../workbook/2026-11-12_stage39_s2_coarse_response_controls.md)。

## S2 科学验收清单

| Requirement | Current evidence / remaining work |
| --- | --- |
| Shared subset / complete input / empty events | 完成，阶段 08–09 |
| ParticleNet / ParT execution | CPU/GPU smoke 和完整压力测量完成，阶段 11–13 |
| Both families native/1mm × joint/projection | 两族正式八模型四格与交互已完成，14项读回全部pass；新增区间与同allocation比较见阶段36 |
| 100/200 μm curves | 两族完整四点曲线及相邻粒度配对完成（阶段37–38），十二项空间输出全部14项readback pass |
| OR/count, grid shift | 两族四项控制均完整，raw/common-support的τ AP差值区间均跨0且上界低于+0.01；工作点CI和限制见阶段39 |
| Pixel / bar / pixel+bar / +tracker | PN四项均完整，双向增量见阶段42–44；+tracker改善四分类点估计，τ AP的raw/common差值区间跨0且上界低于+0.01，见阶段45 |
| No-pair / fixed-graph / local-shuffle | 全部四项均完整、14项读回pass；两族shuffle和ParT no-pair见阶段41，PN dynamic−fixed的raw/common AP区间跨0，资源及评价见阶段42 |
| Explicit cross-plane messages | 两族完整四轮、14项读回pass、主配对/条件域/分类图审阅完成，见阶段54 |
| Full / centered / large evaluation and training domains | 三域已完成、14项读回均pass；full−center的全域/固定支持AP及全域1%效率区间为正，中心域自身AP区间跨0；三域条件学习曲线、重复次数、类别权重与占用见阶段46/49 |
| Three seeds for retained architectures and coarse inputs | 12模型/4组均值全部完成；区间与训练spread分列，见阶段53/55 |
| Toys / priors / τ branches | toy420行、39结果195行τ分支、273组target诊断完整；先验与适用范围见阶段55 |
| Paired CI / interaction / matched time | 41单seed+4三seed均值、2项交互完成，pending=0；完整同allocation整轮比较随主表保留 |
| Resource / error groups | 576行压力测量、35项真实allocation与终态成本、两族20页原生错误显示及新增分类图审阅完成 |
| Scientific stage report and gate decision | 通过，见[阶段55](../../../workbook/2026-11-12_stage55_s2_final_acceptance.md)；strict输出门槛与方法收益分别判定 |

calibration/test 始终保留至 S6。统计不足时报告区间和可排除的收益范围，不用“不显著”直接判为等效。用户接受的 sample/proxy 决定已固定，后续实验直接使用。

## 最新关联控制与输出清单

阶段46完成large-only控制，14项读回pass，24项科学输出进入主评价：28 results、754行分域、84个历史三类条件表、30 paired/15 pending。Full−large训练的全域τ ΔAP=+0.03136247、pointwise CI=[+0.02176243,+0.04088759]；固定E/vertex为+0.01862325、CI=[+0.00830622,+0.02916431]。冻结选点下large区域配对也为正，但large-only自身域NLL持续改善、全域NLL恶化，全域选epoch1与自身域的posthoc epoch4不同。新`analysis.flavour.s2.domain_curves`量化这一差异，不替代主选点；独立样本量/重复次数、尾部与解释范围见[阶段46](../../../workbook/2026-11-12_stage46_s2_large_training_domain.md)。`large_training_comparison.json`保存快照。09:59:25 UTC组织累计80.65506215 GPUh、missing=0。额外ParT native种子随后完成，先实际读回，再纳入后续评价；S2严格gate仍为false。

阶段45补齐PN +tracker，14项读回pass、23/35科学输出完整。27 results、727行分域、81张历史三类条件表、29 paired/16 pending、2 factorial、0 seed means。+Tracker的四类balanced accuracy点估计0.3974→0.4367，τ AP差=−0.00424866、pointwise CI=[−0.01307753,+0.00369210]；固定E/vertex差=−0.00282447、CI=[−0.01101520,+0.00480946]。两者可排除至少+0.01的当前模型收益，不能判等效。输入空事件440→282，四类/分支/条件域与实际成本见[阶段45](../../../workbook/2026-11-12_stage45_s2_tracker_increment.md)。`tracker_comparison.json`保存独立快照；09:34:40 UTC组织累计78.59287089 GPUh、missing=0。训练域、额外种子和plane-only继续，严格gate仍为false。

阶段44补充已有模态模型的条件配对：large域pixel+bar−bar的τ ΔAP=+0.01429830，pointwise CI=[+0.00374257,+0.02424773]，1% FPR效率差也为正区间；中心域ΔAP=−0.01544100、CI=[−0.05592653,+0.02131973]，方向未确定。Bar在pixel之上的AP增量在两个区域均为正区间。新入口`analysis.flavour.s2.conditional_pairs`复用full-validation NLL选点与既有fiducial标记，未重加权E/vertex；这些结果后的条件诊断不增加主合同。计数、尾部限制、实际验证及英文图见[阶段44](../../../workbook/2026-11-12_stage44_s2_conditional_modality_intervals.md)。原有53项回归测试全部通过；仍为22/35科学模型完整，+tracker/训练域/额外种子继续，严格gate尚未通过。

阶段43补齐PN bar-only：14项读回pass，26 results（4 references+22科学模型）、700行分域、28 paired/17 pending。Pixel+bar−bar raw τ ΔAP=+0.01073932，pointwise CI=[+0.00463872,+0.01687213]；固定E/vertex支持为+0.00180202，CI=[−0.00491768,+0.00812752]。1% FPR效率差区间跨0；等allocation预算下排序与固定四轮不同。`modality_comparison.json`保存三模态完整快照，资源/分域/历史三类及解释见[阶段43](../../../workbook/2026-11-12_stage43_s2_modality_and_target_diagnostics.md)。08:49:23 UTC组织累计74.11387479 GPUh、missing=0；严格gate仍为false。

新增`analysis.flavour.s2.target_diagnostics`只读现有provenance和概率，输出`target_diagnostics/target_counts.csv`与`summary.json`。90k子集有14个target PDG/8个Z，validation W占92.76%；当前26个完成结果覆盖182个Z条件组。Target用于truth诊断，未加入主E/vertex匹配或模型输入，未改变接受样本。完整结果增加后再次运行此入口以更新条件表。

08:35 UTC输出清单已确认21/35科学模型完整，全部14项读回pass且进入评价；6 running、8尚未启动。阶段42新增25 results（4 references+21科学模型）、673行分域、27 paired/18 pending。PN pixel+bar−pixel-only τ ΔAP=+0.06890803，pointwise CI=[+0.05663332,+0.08443211]；固定common support为+0.04851078，CI=[+0.03602058,+0.06154046]。1% FPR效率从31/2,500升至83/2,500，差值CI=[+0.01034517,+0.03158505]。Dynamic−fixed的raw/common AP区间均跨0，fixed allocation减少24.06%。分域、四类混淆、资源和限制见[阶段42](../../../workbook/2026-11-12_stage42_s2_fixed_graph_bar_increment.md)，独立快照`graph_pixel_comparison.json`。08:29:35 UTC组织累计72.13702406 GPUh、missing=0；严格输出gate仍为false。

2026-11-12阶段41合并评价：23 results（4 references+19科学模型）、619行分域、25 paired/20 pending、2 factorial、0 seed means。ParT main−shuffle τ ΔAP=+0.00847852，pointwise CI=[+0.00281484,+0.01296334]；固定common support为+0.00794723，CI=[+0.00257296,+0.01269657]。PN同项AP及ParT main−no-pair的raw/common区间跨0，1% FPR效率差的三项区间也均跨0。No-pair参数减少2,520且显存/wall下降，不能只按τ AP判断取舍；详细指标和解释范围见[阶段41](../../../workbook/2026-11-12_stage41_s2_association_structure_controls.md)。

`association_structure_comparison.json`为本批独立快照；新增`analysis.flavour.s2.output_inventory`读取既有35项配置、输出读回、评价与backend成本，生成`campaign_output_inventory.json`。其backend状态来自注明时刻的accounting，不替代实时`condor_q`。实际未完成状态下`--require-complete`已按预期退出1。最终流程追加：

```bash
# 先执行上文不含 --allow-incomplete 的严格评价，再核对全部35项输出。
python -m analysis.flavour.s2.output_inventory --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --require-complete
```

## CERN Kubeflow 文档核对

09:56:39 UTC用户通知重新登录后，已实际查询确认访问恢复；既有v7 Job UID未变，Pod仍Pending且无node，继续原请求。精简快照为`h100/kubeflow/client_login_restored.json`。此前09:46–09:47 UTC的OIDC凭据过期曾使非交互查询失败，已解决；历史失败保留在`client_auth_expired.json`，未据客户端失败终止或替换Job。

根据用户提供的[命令行入门](https://ml.docs.cern.ch/getting-started/#command-line_1)，并核对[GPU flavors](https://ml.docs.cern.ch/#available-flavors)与[单节点训练](https://ml.docs.cern.ch/ml/training/)，单GPU训练可直接创建`batch/v1 Job`。整卡H100 94GB为`nvidia.com/gpu: 1`、`nodeSelector: {nvidia.com/gpu.product: NVIDIA-H100-NVL}`；入门页的`nvidia.com/mig-3g.47gb`示例是47GB分区资源，与本S2整卡预算分开。后续科学Job使用`--gpu-product NVIDIA-H100-NVL`；执行前仍由已有runtime检查确认GPU/Torch/BF16。

文档支持`mount-eos`/`mount-cvmfs`/`mount-afs`标签；现有CPU容器已实测EOS/CVMFS可读，GPU probe v7仍存活，不重提。当前Pod的调度反馈为6节点GPU不足、33节点selector不匹配等，无node；NVIDIA quota=1且被该排队请求占用。`h100/kubeflow/cern_documentation_review.json`保存文档URL、整卡/MIG区别和精简实时证据。没有调整平台注入的`large-images`约束，也没有迁移科学训练。GPU可用后仅承接仍idle的Condor任务，先结束其原副本再启动Kubeflow writer。


## ParT native 三seed训练波动

本批30 results=4 references+26科学配置，808 strata、90历史三类表、30 pairs/15 pending、2 factorial、0 seed means，target诊断210 model×Z组；26项output_complete，完整gate仍false。ParT native的17/42/2026均通过14项读回，raw τ AP=0.35583088/0.35858391/0.34930306，均值0.35457262、sample SD=0.00476665；固定common-support均值0.31906188、SD=0.00413959。逐seed图已实际查看，SD不代表事件CI或ensemble；τ argmax recall=0.4448/0.5488/0.6672，不能以AP spread替代决策稳定性。四组预定三seed配对仍未齐，见[阶段47](../../../workbook/2026-11-12_stage47_s2_native_training_seed_spread.md)与`h100/part_native_seed_repeatability.json`。


## GPU flavor 配置修正

用户明确要求相应flavor后，已修正提交入口为`--gpu-flavor h100-94gb`，同时设置product和GPU resource，GPU请求省略CPU/RAM、由CERN自动注入46 CPU/127G。v7通用请求与v8被拒绝的旧CPU/RAM请求已主动结束并核算各0 GPUh；不是自然排队超时。v9实际Pod已创建，10:33:58 UTC为Pending、无node，selector=`NVIDIA-H100-NVL`、GPU=`nvidia.com/gpu:1`。14项Kubeflow与2项成本测试通过，实际Pod准入dry-run通过；GPU runtime尚未测量。完整flavor表、文档来源、配额/存储和迁移入口见[08_kubeflow](08_kubeflow.md)及[阶段48](../../../workbook/2026-11-12_stage48_s2_kubeflow_flavors.md)。


## 三域训练控制完整结果

[阶段49](../../../workbook/2026-11-12_stage49_s2_center_training_domain.md)补齐center-only，14项读回pass；31 results=4 references+27科学配置、835 strata、93历史三类表、31 paired/14 pending、2 factorial、0 seed means，217个target条件组，27 output_complete。Full−center的全域τ ΔAP=+0.05449639，pointwise CI=[+0.04020247,+0.06874907]；固定E/vertex为+0.03210627，CI=[+0.01756088,+0.04732816]；全域1%效率差+0.018，CI=[+0.00576943,+0.02511136]。中心条件域AP区间跨0，large域AP/1%效率区间为正；三域学习曲线已生成并查看，不改变主选点。Center只有7,429 unique train、319,233 exposures、loss使用该域类别权重；其原生通道均值为full的2.84倍。`center_training_comparison.json`保存完整证据，`stage49_execution_snapshot.json`保存5 running/3 idle的实际状态。Kubeflow v9保持正确H100 94GB flavor并Pending、无node；科学任务未迁移。严格S2 gate仍false。

## ParT 1 mm seed17 配对

[阶段50](../../../workbook/2026-11-12_stage50_s2_part_resolution_seed17.md)：`1573782.25` exit0、14项读回pass；32 results=4 references+28科学配置、862 strata、96历史三类表、32 paired/13 pending、0 seed means，target 224组，28 output_complete。Seed17 native−1 mm raw τ ΔAP=+0.00440110，pointwise CI=[−0.00207130,+0.01145529]；固定E/vertex为−0.00138996，CI=[−0.00768450,+0.00504632]。尚不能确认正收益或排除raw +0.01收益；三seed均值继续等待2026。1 mm综合四分类点估计较好，不能由τ AP代表所有指标。`part_resolution_seed17_comparison.json`及`stage50_execution_snapshot.json`保存主表、终态和实际4 running/3 idle状态。11:15 UTC组织85.50462006 GPUh、missing=0；Kubeflow正确H10094GB flavor仍Pending，严格gate=false。

## ParT native−1 mm 三seed确认

[阶段51](../../../workbook/2026-11-12_stage51_s2_part_resolution_three_seeds.md)：`1573782.26` exit0、14项读回pass，首个预定三seed差值完整。Raw平均τ ΔAP=+0.00079004，pointwise事件CI=[−0.00282494,+0.00445758]；固定粗E/vertex平均−0.00201089，CI=[−0.00526091,+0.00111996]。区间条件于三次已拟合模型，排除此条件下+0.01平均收益；逐seed差值+0.00440110/+0.00355534/−0.00558631，sample SD=0.00553825，不作一般信息等效或训练seed总体结论。`part_resolution_three_seed_comparison.json`保存六模型与比较，四张新图已实际查看。本批33 results、889 strata、99历史三类表、33单seed+1均值、11 pending，target 231组，29 output_complete，严格gate=false。11:34 UTC组织86.79951360 GPUh、missing=0；11:45实际5 running/1 idle，ParT plane-only已在Condor启动，Kubeflow仍Pending。备用task33配置未提交，不启动运行中任务副本。

## Native架构三seed比较与排队终态

[阶段52](../../../workbook/2026-11-12_stage52_s2_native_architecture_three_seeds.md)：PN native17/2026均exit0、14项读回pass。ParT−PN三seed平均raw τ ΔAP=+0.00219679，pointwise事件CI=[−0.00401226,+0.00795043]；固定E/vertex平均−0.00244385，CI=[−0.00832807,+0.00308100]，差值训练sample SD=0.00518276。区间条件于已训练seed，排除该条件下+0.01平均收益，不作一般等效判断。ParT三seed四分类均值更好、实际成本更高，详情与非τ差值CI未测量项均列入报告。主表35 results、943 strata、105历史三类表、35单seed+2均值、8 pending；31 output_complete。PN 1 mm17随后训练/读回完成，故实际32 training complete/3 running；target扫描先纳入它，252组对应36结果，主配对待下批同步。12:11组织89.75616158 GPUh、missing=0。所有剩余任务已在Condor运行，Kubeflow v9在12:03主动结束释放配额，0 GPUh、runtime未测量；task33/34备用配置未提交，详见[flavor入口](08_kubeflow.md)。

## 全部三seed确认完成

[阶段53](../../../workbook/2026-11-12_stage53_s2_complete_seed_confirmations.md)：PN 1 mm17/2026均exit0、14项读回pass，12个三seed模型完整。新增PN native−1 mm平均raw τ ΔAP=−0.00013880，CI=[−0.00454523,+0.00421196]；1 mm下ParT−PN为+0.00126795，CI=[−0.00485443,+0.00824882]。全部四组raw/common区间均跨0、上界低于+0.01，条件于当前三次拟合与固定权重，训练spread另列，不作一般信息等效结论。`three_seed_four_class_summary.json`补齐四类AP/recall及综合指标的均值与sample SD，相关综合差值CI未测量。主表37 results、997 strata、111历史三类表、39单seed+4均值、2 pending；target 259组、33 output_complete，严格gate=false。13:05 UTC两项plane-only仍running；12:52组织91.86595936 GPUh、missing=0。`complete_seed_confirmation_comparison.json`与`stage53_execution_snapshot.json`保存证据。Kubeflow备用训练Pod在释放配额后server dry-run通过、自动46CPU/127G/1整卡GPU，实际未提交；详见flavor入口。

## S2最终验收

[阶段54](../../../workbook/2026-11-12_stage54_s2_cross_plane_controls.md)完成两项跨层控制；[阶段55](../../../workbook/2026-11-12_stage55_s2_final_acceptance.md)给出完整科学结论和gate decision。最终35/35科学输出完整、490读回checks pass、39结果、1051 strata、117历史三类表、41单seed配对、4三seed均值、2交互、0 pending；target 273组与主模型集合一致。`strict_final_pipeline.json`记录accounting/evaluate/target/inventory顺序exit0，严格评价不含allow-incomplete、inventory含require-complete。组织成本92.73055556 GPUh，149终态记录、missing=0，低于200上限。

S2通过不等于宣布所有新表示有收益。四组三seed τ AP均值raw/common区间均跨0、上界低于+0.01，条件于现有拟合与支持；bar增量、训练域覆盖及四分类/跨层结果在总报告分别说明。S3–S6未执行，calibration/test冻结。
