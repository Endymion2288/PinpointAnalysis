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
| Native response displays | `evaluation/event_displays/b0_all_s42/`，20 页 PDF、事件 keys 和首例 PNG |
| V100 resource study | `resource_profile/` 为 v1；`resource_profile_v2/` 的 576 行已完成，两者分别保留 |
| Spatial screen | [current 12-run configuration](configs/s2_spatial_screen_v2.json)；v1 在首个 epoch 完成前停止，计费保留 |
| Spatial paired evaluation | [18 paired contrasts](configs/s2_spatial_pairs_v2.json)，将输出到 `evaluation_spatial/`，缺项显式列 pending |
| Prespecified controls | [13 runs](configs/s2_controls_v2.json)：两族 OR/shift/shuffle、各自 pair/graph，以及 ParticleNet modality/domain 控制 |
| Three-seed confirmation | [8 additional runs](configs/s2_confirmation_v2.json)：两族 native/1mm joint × seed17/2026；复用已有 seed42 |
| Final comparisons | [39 individual contrasts and 4 three-seed means](configs/s2_final_pairs_v2.json)；共 33 项科学训练，尚未全部完成 |

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

三种子均值比较对 seed17/42/2026 的 **AP 差值**取平均，每次 bootstrap 对三者使用相同物理事件块；不是把三份 prediction 拼成 30k events，也不是 ensemble。分别报告事件区间、逐 seed 差值及训练 spread。固定 common-support weights 的区间为条件区间。最终评价使用 `s2_final_pairs_v2.json` 并去掉 `--allow-incomplete`，缺少任一指定结果时不能声称完成。

18:44 UTC 提交状态：空间 `1573747.0–11`、控制 `1573750.0–12`、额外种子 `1573751.0–7`，均在 `bigbird20.cern.ch`。当时 4 running/29 idle。`.6/.8` 的空间 JobPrio=10，其余尚未启动空间项=5，追加任务=0，以便先获得完整筛选证据。`1573752` 为 `evaluation_final/` 的部分结果集成检查，显式启用 `--allow-incomplete`。这些 ClusterIds 仅用于查状态；存活任务不再次提交。

全部训练完成后的入口：先更新实际 terminal allocation，再严格评价与两族原生错误显示。

```bash
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.submit evaluate --root "$S2_RESULTS" \
  --evaluation-tag evaluation_final --comparisons "$PWD/roadmap/configs/s2_final_pairs_v2.json"
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
| Full / centered / large evaluation and training domains | 分域评价已在 B0 运行；深度训练域控制待完成 |
| Three seeds for retained architectures and coarse inputs | 待筛选及 17/42/2026 确认 |
| Toys / priors / τ branches | toy 与 B0 诊断完成；深度模型物理解释待结果 |
| Paired CI / interaction / matched time | 程序已有，reference CI 已完成；空间主表待运行 |
| Resource / error groups | GPU 曲线与 B0 原生显示完成；深度模型结果待补 |
| Scientific stage report and gate decision | 尚未完成，不把 smoke/reference 当作 S2 整体验收 |

calibration/test 始终保留至 S6。统计不足时报告区间和可排除的收益范围，不用“不显著”直接判为等效。用户接受的 sample/proxy 决定已固定，后续实验直接使用。
