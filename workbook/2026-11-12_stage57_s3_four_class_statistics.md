# 阶段57：S3四分类统计与H100执行验证

## 阶段目标

把已冻结的四分类合同落实为可执行的评价与确认选择程序。S3尚未验收；本阶段不报告新模型分类收益。

## 工作内容

新增`analysis.flavour.s3.metrics/evaluate/select/accounting`。指标覆盖四类macro AP、macro AUROC、macro F1、balanced accuracy、NLL、四类Brier和逐类AP/AUROC/precision/recall/F1、混淆矩阵。固定权重的common support、所有条件域、τ工作点与τ分支一并保留。所有模型及训练seed共享同一组物理block bootstrap权重；训练seed的sample SD另列。冻结22主配对、窗口×表示四分类interaction和完成后才执行的Pareto筛选；确认配置独立写入，不覆盖screen。

AP/AUROC预计算排序并严格保留ties。独立sklearn对照发现，重新归一化每行概率可能改变原本并列的score；现在仅NLL遵循log-loss归一化，排序使用文件中的原score。缺少某类时该类AP/AUROC和四类macro对应统计输出null，不把有事件的子集称为完整四类平均。

修正S3 Kubeflow计费的label/container名称，使S2默认路径保持原样。增加profile/smoke完整程序wall时间；组织预算保留此前T4 profile的独立保守上界0.1811426134 GPUh，不将其称作精确reservation。

2026-11-12再次读取[CERN available flavors](https://ml.docs.cern.ch/#available-flavors)。整卡H100为`NVIDIA-H100-NVL`+`nvidia.com/gpu:1`，平台注入46 CPU/127 GB；47 GB为独立`nvidia.com/mig-3g.47gb:1`。当前科学比较固定整卡H100。文档副本保存在results/literature/cern_flavors_latest.html。

## 详细数据

输入仍是已接受的80,000 train/10,000 validation和原detector-proxy；calibration/test未使用。结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

27项自动检查通过，8.695 s，日志`unit_stage57.log`，包括完整覆盖/空输入/批独立性、BF16 autocast真实backward+AdamW、加权ties与sklearn对照、物理block复制等价、三seed共享重采样、Pareto fallback和原S2计费逻辑。命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s3 tests.test_flavour_s3_metrics tests.test_flavour_s2_accounting tests.test_flavour_s2_kubeflow
python -m analysis.flavour.s3.evaluate --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12 --campaign roadmap/configs/s3_screen.json --comparisons roadmap/configs/s3_pairs.json --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/evaluation_baseline --no-plots
```

评价在16个已有S2模型上真实退出0，28.319 s。每模型400次raw/common-support bootstrap，128个accepted-family/chunk blocks、每类32。固定common support仍是8,559事件、Neff=7,959.413646535195，103个有效cell，每类权重和1,586。17个S3实验、22主配对和1个interaction共40项pending，如实保存；不能据此验收S3。

下表为已有S2模型seed42的四分类基线重表达，来自已有预测文件，不是S3拟合结果：

| Model | Macro AP | NLL | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| B0 all | 0.435711 | 1.245042 | 0.404577 | 0.402900 |
| ParT native | 0.462286 | 1.221079 | 0.418443 | 0.418400 |
| ParT 1 mm | 0.442603 | 1.242935 | 0.398800 | 0.403300 |
| ParticleNet native | 0.434524 | 1.239306 | 0.392041 | 0.397400 |
| ParticleNet 1 mm | 0.429536 | 1.249708 | 0.394003 | 0.397700 |

产物`evaluation_baseline/summary.json`、`bootstrap_statistics.npz`、`error_groups.parquet`；英文混淆/四类ROC/PR/learning curves与资源图使用同一结果单独绘制，避免再次bootstrap。所有S2模型symlink只读，未改写S2产物。

HTCondor实际提交前重新检查schedd与EOS/WRITE，选择兼容且idle最少的bigbird20；依据完整保存于`submission_*.json`。首轮H100 profile1573830和smoke1573831真实失败，allocated wall分别18 s/13 s，退出码1：BF16 local输出与FP32 query/embedding residual提升后的module readout在indexed packing处类型不匹配。已改为由module readout类型创建目标buffer，并增加CPU BF16 optimizer回归；随后重提profile1573833与smoke1573834。H100修复后的最终验证结果本报告写作时尚待终态，不把CPU通过替代正式GPU检查。

Kubeflow `pinpoint-s3-h100-probe-v1`已通过Pod admission，写作时约23分钟仍Pending且无node；实际scheduler事件显示GPU/CPU/RAM不足和selector/taint约束。尚未获得GPU runtime证据，不称其排队更快；未调度部分GPU计费为0。未创建重复probe或同一训练的双writer。

## 阶段性结论

事实：四分类统计在真实基线预测上完整运行；所有27项检查通过。实际H100验证暴露并已定位一个混合精度packing错误，修复后已重新排队验证。尚无S3训练完成或新模型收益证据。

假设：fine plane windows、module context或latent fusion能改善四类识别，仍须由完整screen、配对CI与三seed确认回答。原生信息可能受后续module/global瓶颈限制，不能仅凭节点保留推断优势。

## 下一步计划

完成修复后的H100最大占用profile及optimizer smoke；全部finite后提交冻结17项screen。实际终态、四轮80,064 updates、零AMP skips与CPU读回通过后评价四分类、2×2交互和资源Pareto，独立冻结确认配置，追加seed17/2026。按`roadmap/09_s3_operations.md`完成全套科学验收后再决定晋级。S4–S6未执行。
