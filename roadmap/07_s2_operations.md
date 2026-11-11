# S2 运行状态与复现入口

S2 进行中。沿用用户接受的 sample/source-family split 与 detector-proxy，不重新设置来源认证或真实硬件标定门槛。科学验收需要完成以下实验及其实际评价。

## 当前产物

结果根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11`。代码：`src/analysis/flavour/s2/`。配置合同：[s2.yaml](configs/s2.yaml)。

| Artifact | State / location |
| --- | --- |
| Frozen subset | `manifest.parquet`，80k train + 10k validation，各类相等；不改既有 split |
| Complete observations | `shards/000`–`031`，约 1.765 GB combined Parquet，原生 pixel、bar、accepted tracker |
| Occupancy / diagnostics | `catalog.parquet`、`catalog_summary.json`，保留 2,535 全空事件 |
| Information toys | `toys/`，420 点曲线、projection pairing collision、双轨/kink/shower support |
| B0 / prior reference | `experiments/b0_*_s42` 和 `truth_energy_vertex_s42`，均已完成 |
| Paired reference evaluation | `evaluation/summary.json`、`metrics.csv`、`tau_PR.png`、`error_event_groups.parquet` |
| Comparison report figures | 新增 `s2.plots` 并接入 `evaluate`：粒度曲线、配对 CI/等 GPU-time、三种子均值图；真实 B0 的 3 项配对图已查看，完整空间/种子图待结果，见[阶段 22](../workbook/2026-11-11_stage22_s2_comparison_figures.md) |
| Native response displays | `evaluation/event_displays/b0_all_s42/`，20 页 PDF、事件 keys 和首例 PNG |
| V100 resource study | `resource_profile/` 为 v1；`resource_profile_v2/` 的 576 行已完成，两者分别保留 |
| H100 resource comparison | `1573759.0` exit0，H100/LCG108/BF16 共 576 行；与 V100 匹配后去 warmup 保留552行，batch1 training compute 比值1.69–2.33；原始/配对图已查看，见[阶段 24](../workbook/2026-11-11_stage24_s2_alternative_gpu_profile.md)；整轮收益待测 |
| H100 full-epoch candidate | 独立 `h100/` 共享既有 accepted 输入和 B0 references；[35 项同配置合同](configs/s2_h100_v2.json)已冻结，先行 tasks0/6 已提交为 `1573760.0/1`；两者首轮耗时均须 ≤V100 的80%且执行完整，见[阶段 25](../workbook/2026-11-11_stage25_s2_h100_epoch_gate.md)。通过后统一环境完成全矩阵，当前 V100 任务继续 |
| H100 candidate execution / gate | `1573760.0` 于22:56:49 UTC实际开始，H100 NVL/Torch2.7.1/BF16和冻结配置一致；`.1`仍idle。新 `s2.hardware_gate` 的5项测试通过，23:34真实首轮验收：ParT 2,223.38 s（V100的45.76%）、20,016/20,016 applied、23项检查通过；另一族pending，总体pending。已从首轮checkpoint暂时hold原ParT任务，让另一族优先取得资源；实际启动或10分钟无收益后自动release原任务。`h100/execution_gate.json` 保存状态 |
| Kubeflow execution | `ml-prod/xcheng` 已登录，NVIDIA quota=1；CPU 容器的 LCG108/Python3.12/Torch2.7.1、EOS/CVMFS 路径已实际成功，GPU probe v5全程未分配node，23:14:58 UTC到达deadline（GPU reservation=0）；科学任务未迁移。v6以相同资源保留6小时排队机会，容器探测仍限时300秒。新提交/核算入口与失败环境探测见[阶段 26](../workbook/2026-11-11_stage26_s2_kubeflow.md) |
| Spatial screen | [current 12-run configuration](configs/s2_spatial_screen_v2.json)；v1 在首个 epoch 完成前停止，计费保留 |
| Spatial paired evaluation | [18 paired contrasts](configs/s2_spatial_pairs_v2.json)，将输出到 `evaluation_spatial/`，缺项显式列 pending |
| Prespecified controls | [13 runs](configs/s2_controls_v2.json)：两族 OR/shift/shuffle、各自 pair/graph，以及 ParticleNet modality/domain 控制 |
| Three-seed confirmation | [8 additional runs](configs/s2_confirmation_v2.json)：两族 native/1mm joint × seed17/2026；复用已有 seed42 |
| Original comparison contract | [39 individual contrasts and 4 three-seed means](configs/s2_final_pairs_v2.json)；原有 33 项科学训练，保留此文件以对应既有部分评价 |
| Complete S2 contract | [41 individual contrasts and 4 three-seed means](configs/s2_complete_pairs_v2.json)；另含两族 [plane-only controls](configs/s2_cross_plane_controls_v2.json)，共 35 项科学配置，见[阶段 21](../workbook/2026-11-11_stage21_s2_cross_plane_control.md) |
| Plane-only GPU execution | `1573755.0/1` 均 exit0，实际 H100/BF16、各 15 applied steps、0 skips，checkpoint/概率有限；`smoke/cross_plane_smoke_checks.json`。两项 V100/FP16 科学配置已提交为 `1573757.0/1`，见阶段 21 |
| Historical / branch diagnostics | 初次 `1573753` 已完成；`evaluation_final/summary.json` 含 4 个 references 的 12 张三类表和 τ 分支 AUC/AP |
| Historical drawing integration | `1573758` exit0（commit `9d67e8a`），当时4个 references、45项 pending；后续科学评价见下一行 |
| First scientific model / current evaluation | `1573747.6` exit0，ParticleNet native-joint 四轮完成，τ AP=0.35784；相对B0 raw ΔAP=+0.01313，CI95=[−0.00097,+0.02446]；fixed-common ΔAP=+0.01862，CI=[+0.00686,+0.02971]。22:07 UTC 本地部分评价正常退出，5个完成结果、1项配对/44项 pending；20页原生图和3个错误页已查看，见[阶段27](../workbook/2026-11-11_stage27_s2_first_scientific_model.md) |
| Second scientific model / updated evaluation | `1573747.1` exit0，ParT native-projections 四轮完成，τ AP=0.34989、NLL=1.22890、allocation=3.84833 GPUh。最新部分评价正常退出，6个完成结果、160行分域指标、1项配对/44项pending；新resolution图已查看。22:23 UTC为2 complete/5 running/28 idle，见[阶段28](../workbook/2026-11-11_stage28_s2_spatial_results.md) |
| Third scientific model / latest evaluation | `1573747.2` exit0，ParT 1mm joint四轮完成，τ AP=0.35845、NLL=1.22922、allocation=4.71833 GPUh。部分评价为7个完成结果、187行分域指标、21个历史三类诊断，仍为1项配对/44项pending；resolution图已查看。23:15队列3 complete/4 running/28 idle，见阶段28 |
| Fourth scientific model / first spatial contrasts | `1573747.3` exit0，ParT 1mm projection四轮完成，τ AP=0.35210，allocation=3.84250 GPUh。8个完成结果、214行分域指标、3项配对/42项pending；1mm joint−projection AP=+0.00634，pointwise CI=[+0.00101,+0.01250]；native−1mm projection=−0.00221，CI=[−0.00935,+0.00544]。原`.7`已接续运行，累计23.56117 GPUh且missing=0，见阶段28 |
| Input response domains | `1573754` 已完成；新增 model-input-active/empty 与 tracker-only，pixel+bar validation 输入空为 440 events，三探测器全空为 282；见[阶段 19](../workbook/2026-11-11_stage19_s2_input_response_domains.md) |
| First full scientific epoch | ParT 1mm projection epoch1：τ AP=0.30649、NLL=1.32943、3,063.325 s；继续既定 4 epochs，尚未作为完成模型纳入配对，见[阶段 20](../workbook/2026-11-11_stage20_s2_first_complete_epoch.md) |

完整 B0 的 τ AP=0.34471、1% FPR 下 τ efficiency=65/2,500=2.60%；truth E/vertex 参照 AP=0.44106。25% τ benchmark prior 下，粗 E/x/y/z common support 为 8,559 events、Neff=7,959.41，二者 weighted AP 为 0.30845/0.36474。truth 参照单列为先验诊断，不算硬件模型。详细 counts、CI 和条件指标见[阶段 12](../workbook/2026-11-11_stage12_s2_reference_results.md)。

## 科学训练合同

两家族共享当前表示的局部 learned compression（每块最多 32 nodes）和全部全局块；这是明确的 baseline adaptation，不是未经压缩的全事件 ParT。模型输入为 18 项 observation 特征及其导出的 8 项计数/信号和，后者在局部汇聚与全局头显式保留。只从当前表示计算，不读取 raw counts 恢复 OR 丢失的 multiplicity。truth/provenance 仅做 split/诊断。粗表示不携带原生 cell 残差，投影移除 x–y pairing；空事件保持先验型输出。

v2 ParT dim48=151,228 parameters；ParticleNet dim64=161,156，相差 6.56%。空间筛选每 run 4 epochs、batch4、20,016 attempted steps/epoch、相同 seed42 与数据顺序；记录实际 applied steps 和 AMP skips。bucket256 用同一原生通道数排序，完整覆盖后重复少量 batches 匹配步数。模型按 validation NLL 选 best；每 epoch 四类概率支持同型号 GPU、共同时间预算内的比较。

科学 GPU 固定 Tesla V100-PCIE-32GB，复用 `LCG_107_cuda/x86_64-el9-gcc11-opt`：Python3.11.9、Torch2.5.1、FP16。这是复用现成 V100 构建的运行环境，项目 Python3.12 声明未修改。LCG_108/Torch2.7.1 不含 sm70，不能用于 V100；108 仍用于 CPU reference/evaluation。

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

20:43:59/20:45:19 UTC，`1573747.3/.0` 因 `Startd was shutdown`（1013）自动返回 idle，原 JobId 保留。分别可从 epoch2/epoch1 恢复，CPU checkpoint readback 已通过；只将这两个 idle JobId 的 JobPrio 提高到20，等待原任务重新匹配 V100。20:47 UTC 为 3 running、32 idle。中断 allocation 纳入成本，未进入已提交 checkpoint 的部分共 1.50795 GPU-hours，含准备与部分 epoch；见[阶段 23](../workbook/2026-11-11_stage23_s2_startd_shutdown.md)。当前执行状态以 `condor_q` 为准，重排后训练 manifest 可暂留 `running`。

全部训练完成后的入口：先更新实际 terminal allocation，再严格评价与两族原生错误显示。

21:44:20/21:50:04 UTC，原 `1573747.0/.3` 已分别从 epoch1/epoch2 在 V100 恢复，optimizer/AMP counters 与保存状态衔接一致；见阶段23补充与 `startd_shutdown_resume_execution.json`。Kubeflow 自21:58起已有成功 CPU 环境证据，GPU资源仍待调度；用户已授权在Condor排队时使用。`s2.accounting` 现按记录的 Job UID 合并两个后端，未分配节点的等待为0，已分配 Pod 的启动/初始化/失败成本保留。

22:40:04 UTC，`.3`恢复后已提交完整epoch3，并进入epoch4。CPU读回的新checkpoint参数、optimizer、概率、history/config和恢复状态8项检查通过，见 `startd_shutdown_post_resume_checkpoint.json` 与阶段23。22:35:58 UTC累计18.97207 GPU-hours，missing history records=0；5项V100科学训练继续。

22:43:49 UTC，仍idle且未启动的 `1573747.7/.9` 原任务JobPrio从5提高到10，优先补齐ParticleNet native/1mm双投影这两个主factorial成员；其余6个主factorial成员已完成或运行中。依据实验角色调整顺序，不改模型/数据/步数，也不重提任务。证据为 `primary_factorial_priority.json`，见阶段28。

22:54:35 UTC，`.0`恢复后的完整epoch2也已CPU读回通过8项检查，累计40,012 applied/20 skips，随后进入epoch3；见 `startd_shutdown_native_post_resume_checkpoint.json`。两项Startd中断任务均已提交并读回恢复后的新完整checkpoint，四轮科学训练继续。

```bash
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.submit evaluate --root "$S2_RESULTS" \
  --evaluation-tag evaluation_final --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json"
# 等上述评价成功后提交 displays。
python -m analysis.flavour.s2.submit display --root "$S2_RESULTS" \
  --evaluation-tag evaluation_final --experiment part_native_joint_s42_e4_v2
python -m analysis.flavour.s2.submit display --root "$S2_RESULTS" \
  --evaluation-tag evaluation_final --experiment pn_native_joint_s42_e4_v2
```

## S2 科学验收清单

| Requirement | Current evidence / remaining work |
| --- | --- |
| Shared subset / complete input / empty events | 完成，阶段 08–09 |
| ParticleNet / ParT execution | CPU/GPU smoke 和完整压力测量完成，阶段 11–13 |
| Both families native/1mm × joint/projection | 配置冻结，真实训练进行中 |
| 100/200 μm curves | 两族均已冻结，待训练及 paired evaluation |
| OR/count, grid shift | 待科学对照 |
| Pixel / bar / pixel+bar / +tracker | B0 reference 已有；深度模型消融待完成 |
| No-pair / fixed-graph / local-shuffle | 实现入口已有，待实际对照 |
| Explicit cross-plane messages | CPU 与专用 H100/BF16 smoke 通过；V100/FP16 科学任务 `1573757.0/1` 已提交，配对待结果 |
| Full / centered / large evaluation and training domains | 分域评价已在 B0 运行；深度训练域控制待完成 |
| Three seeds for retained architectures and coarse inputs | 两族 native/1mm joint 的 17/42/2026 已冻结并全部提交，待实际结果 |
| Toys / priors / τ branches | toy 与 B0 诊断完成；深度模型物理解释待结果 |
| Paired CI / interaction / matched time | 程序已有，reference CI 已完成；空间主表待运行 |
| Resource / error groups | GPU 曲线与 B0 原生显示完成；深度模型结果待补 |
| Scientific stage report and gate decision | 尚未完成，不把 smoke/reference 当作 S2 整体验收 |

calibration/test 始终保留至 S6。统计不足时报告区间和可排除的收益范围，不用“不显著”直接判为等效。用户接受的 sample/proxy 决定已固定，后续实验直接使用。
