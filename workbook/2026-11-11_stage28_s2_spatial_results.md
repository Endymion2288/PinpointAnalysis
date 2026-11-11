# 阶段 28：S2 空间矩阵的部分完成结果

12项空间实验中已有3项完成，另4项正在运行。新增完成的 ParT 1mm joint 模型 τ AP=0.35845、NLL=1.22922，实际占用4.71833 GPU-hours。当前完成模型尚未形成预定的粒度、投影或架构受控配对。

## 阶段目标

按冻结的12项空间矩阵，逐项积累完整四轮结果、terminal allocation 与预定配对。完整 S2 的35项科学配置和45项比较合同保持不变。

## 工作内容

核对 `1573747.1` 的真实终态：JobStatus=4、ExitCode=0、RemoteWallClockTime=13,854 s。该任务使用 `s2_spatial_screen_v2.json` task1，pixel为原生 pitch 的 xz/yz 双投影，移除x–y pairing；bar配置与其他空间任务相同。架构为当前 ParT adaptation，151,228 parameters，seed42、batch4、4 epochs、20,016 attempted steps/epoch，按 validation NLL 选择。

更新两个后端的 allocation 后，本地完整比较入口以 `--allow-incomplete` 正常退出。`evaluation_final/` 现在有6个完成结果（4个 references、2个科学模型），160行分域指标、18个历史三类诊断。配对仍为1项完成/44项pending：新模型的预定比较对象尚未完成，不临时增加事后配对。更新后的 resolution 图已实际查看，图中仅显示2个完成的原生点；100μm、200μm、1mm尚无完成点。

两个空出的 V100 已实际接续运行原队列：`1573747.4` 为 ParT 100μm joint，`1573747.8` 为 ParticleNet 1mm joint。未重复提交科学任务。Kubeflow 的2 CPU/4 GiB GPU环境探测仍因 GPU 资源不足排队；两族 H100候选的 Job 文件已预先生成并通过 `bash -n`，实际科学请求仍为4 CPU/12 GiB，未提交第二后端副本。

## 详细数据

冻结数据仍为80k train/10k validation，各类20k/2.5k，calibration/test未使用。模型均使用 pixel+bar；以下是分别测量的数值，行间差异不能单独归因于架构或投影。

| Completed model | Tau AP | Tau ROC AUC | NLL | Balanced accuracy | Common-support AP | Allocated GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ParticleNet native joint | 0.35783732 | 0.63023309 | 1.24514857 | 0.397600 | 0.32706642 | 3.16083 |
| ParT native projections | 0.34988948 | 0.62859563 | 1.22890107 | 0.404000 | 0.31567223 | 3.84833 |
| ParT 1mm joint | 0.35844742 | 0.63260771 | 1.22922198 | 0.412800 | 0.32293880 | 4.71833 |

ParT双投影模型在validation empirical 1% FPR阈值0.454139下保留76/2,500个τ、75/7,500个背景，τ效率为3.04%。0.1% FPR对应10个τ和7个背景，尾部统计不足标记保留。Common support使用既有8,559 events和固定权重，本报告没有增加新置信区间。

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] | Train exposures |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.28103238 | 0.34389407 | 3,555.22416 | 80,040 |
| 2 | 1.26347936 | 0.33943845 | 6,948.12612 | 80,044 |
| 3 | 1.24554213 | 0.35465650 | 10,322.50053 | 80,040 |
| 4 | 1.22890107 | 0.34988948 | 13,822.20189 | 80,040 |

按预定NLL规则选择第4轮，尽管τ AP最大值出现在第3轮。最终程序wall为13,823.16650 s，Condor allocation为13,854 s，额外allocation为30.83350 s；peak allocated GPU memory为418.88330 MiB。80,064 attempted steps中80,028 applied、36 AMP skips，applied比例99.9550%。GPU为Tesla V100-PCIE-32GB，LCG107/Torch2.5.1/FP16。

训练exposures总计320,164，包含匹配步数的少量重复，独立训练事件数仍为80k。核对完整history后同步修正[阶段27](2026-11-11_stage27_s2_first_scientific_model.md)原先将每轮概写为80,040的合计；已运行同seed配置在对应轮次具有一致exposures，训练输入与实现未改动。

22:23:36 UTC 原空间矩阵状态：

| Family | Representation | State |
| --- | --- | --- |
| ParT | Native joint | running，epoch2/step11,750 |
| ParT | Native projections | complete，selected epoch4 |
| ParT | 1mm joint | running，epoch4/step6,500 |
| ParT | 1mm projections | running，epoch3/step14,250 |
| ParT | 100μm joint | running，epoch1/step250 |
| ParT | 200μm joint | idle |
| ParticleNet | Native joint | complete，selected epoch4 |
| ParticleNet | Native projections | idle |
| ParticleNet | 1mm joint | running，epoch1/step750 |
| ParticleNet | 1mm projections | idle |
| ParticleNet | 100μm joint | idle |
| ParticleNet | 200μm joint | idle |

全35项科学配置为2 complete、5 running、28 idle；独立H100候选两项idle。22:21:34 UTC accounting累计17.77179 GPU-hours，missing history records=0。上述成本包含早期尝试、重启、资源测量，未分配节点的 Kubeflow 排队耗时不计GPU reservation。

22:43:49 UTC 调度调整：原 `1573747.7/.9`（ParticleNet native/1mm projections）均仍为idle、NumJobStarts=0，JobPrio从5提高到10；`condor_qedit`返回2 matching jobs，查询确认两者为10。主factorial已有的其他6项已完成或正在运行，因而优先让这两个原任务在下一批slot上启动，再补余下中间粒度；依据是预定实验角色，未使用validation分数决定排队。运行任务和科学配置不变，未重提。调度证据为 `primary_factorial_priority.json`；22:35:58 UTC累计18.97207 GPU-hours，missing history records=0。

```bash
condor_qedit -name bigbird20.cern.ch \
  -constraint 'ClusterId == 1573747 && (ProcId == 7 || ProcId == 9) && JobStatus == 1' JobPrio 10
```

产物位于 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/`：

- `second_scientific_model.json`：固定保存新增模型指标、完整history和terminal allocation，评价代码commit=`ff11eb1`。
- `evaluation_final/summary.json`、`metrics.csv`、`resolution_comparison.png/pdf`：本次成功部分评价；后续随完成结果更新。
- `training_observation.json`：上述调度和进度快照。
- `kubeflow/pinpoint-s2-h100-{part,pn}-native.job.json`：已生成、尚未提交的候选执行配置。

复现入口与阶段27相同：先 `s2.accounting`，再用 `s2_complete_pairs_v2.json` 运行 `s2.evaluate --allow-incomplete`。最终验收去掉该参数；当前显式保留44项pending。

## 第三个完成模型与刷新评价

23:11 UTC，`1573747.2` 四轮完成；Condor history 为JobStatus=4、ExitCode=0、RemoteWallClockTime=16,986 s。该模型为ParT 1mm joint，参数151,228，其他训练合同不变，按NLL选中epoch4。

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.33085119 | 0.28580370 | 4,293.01853 |
| 2 | 1.26453682 | 0.34462893 | 8,607.14306 |
| 3 | 1.25458553 | 0.35316891 | 12,801.78146 |
| 4 | 1.22922198 | 0.35844742 | 16,952.73098 |

最终程序wall=16,953.67862 s，额外allocation=32.32138 s，peak=1,587.11963 MiB。80,064 attempted steps中80,029 applied、35 AMP skips，applied比例99.9563%；exposures仍为320,164。validation empirical 1% FPR保留82/2,500个τ和75/7,500个背景，τ效率3.28%；0.1% FPR为21个τ、7个背景，保留尾部统计不足标记。固定common-support AP=0.32293880。

刷新后的 `evaluation_final/summary.json` 有7个完成结果（4 references+3科学模型）、187行分域指标、21个历史三类诊断，仍为1项预定配对完成、44项pending。已实际查看更新后的resolution图，新增1mm joint点；没有临时增加跨架构/跨表示比较。完整指标、history和terminal成本固定在 `third_scientific_model.json`。

23:13:23 UTC累计22.33457 GPU-hours，missing history records=0。23:15队列仍为3 complete/4 running/28 idle；原ParticleNet投影 `.7/.9` 仍在队列等待。H100 ParT候选运行中、另一族idle，尚无完整首轮，环境门槛未通过。Kubeflow GPU probe v5于23:14:58 UTC以DeadlineExceeded结束，全程未分配node；终态证据为 `kubeflow/probe_v5_terminal.json`，GPU reservation=0，科学任务没有迁移。

## 阶段性结论

已完成三项冻结配置的四轮执行及独立验证评价，新的ParT模型验证NLL随四轮下降，但τ AP不单调，预先固定选模规则得到保留。当前证据尚未回答“联合位置是否优于双投影”“原生是否优于1mm”或“ParT是否优于ParticleNet”。

资源侧当前有4项V100科学训练和1项H100候选实际运行。Kubeflow环境探测因GPU不足排队至deadline，未得到GPU执行证据；H100两族整轮门槛仍待完成。

## 下一步计划

等待运行中的完整epoch与终态，优先补齐可形成受控对比的空间结果，再更新41项单seed和4项种子均值合同。H100候选严格沿用执行/耗时门槛，GPU探测或资源profile不代替整轮候选。两族native-joint模型均完成后补齐对应原生错误显示；所有35项、45项比较与科学交付齐全后进行S2整体验收。
