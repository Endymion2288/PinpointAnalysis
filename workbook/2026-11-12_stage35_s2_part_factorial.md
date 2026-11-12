# 阶段 35：S2 正式 ParT 四格与交互效应

正式 H100 ParT 的 native/1 mm × joint/projection 四格已齐全。native pairing 的 τ ΔAP=+0.00124580，pointwise 95% CI=[−0.00501470,+0.00830354]；四格交互效应=+0.00043410，CI=[−0.00790547,+0.00875624]。四项主配对的 raw/common-support AP 区间均跨 0、上界均低于预定 +0.01；当前 seed42、四轮及既定局部压缩条件下可排除这些比较至少 +0.01 的收益，跨种子和信息机制仍待完成。

## 阶段目标

完成文件不完整任务的冻结替换，恢复正式 ParT 四格，评价 pairing、粒度及其交互，并将原失败尝试保留在实际资源成本中。沿用已接受的样本和 detector-proxy，calibration/test 保留至 S6。

## 工作内容

`1573791.0` 执行冻结 `s2_h100_v2.json` task1，真实终态 JobStatus=4、ExitCode=0、NumJobStarts=1。ParT dim48、151,228 parameters、pixel+bar、seed42、batch4、4 epochs、每轮20,016 attempted steps，NLL 选中 epoch4。环境 H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。表示经同平面4 mm、canonical chunks≤32 nodes的局部 learned compression 后处理全部全局块；投影移除 x–y pairing。

替换依据是原 `1573782.0` 的 epoch 表两个 EOS 副本不完整，决策与归档见阶段31。原尝试保存在 `recovery_projection/original_training_attempt/`，没有按质量重选。替换使用已修复的 flush/fsync 与原子发布。04:17:22 UTC，14项 CPU 读回全部通过，包括四轮完整性、冻结配置与环境、optimizer/恢复状态、NLL选模、10k validation keys、best checkpoint/概率一致性、40k四轮概率及逐轮重算指标、exposures。

实际核算与部分评价随后正常退出，评价 commit=`778c05b`。10个有效结果（4 references+6正式模型）、268行分域指标、30个历史三类诊断，8项配对完成、37项pending，1组完整 factorial。新 resolution 图与 paired-effects 首页均已实际查看。当前主表使用完整替换，原归档尝试只计成本。

## 详细数据

独立 train 80k、validation 10k，每类20k/2,500；每轮 exposures 80,040/80,044/80,040/80,040，合计320,164。80,064 applied updates，0 AMP skips。

| Epoch | Validation NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.28522510 | 0.33062624 | 1,120.18385 |
| 2 | 1.26784029 | 0.32789119 | 2,212.79580 |
| 3 | 1.24075242 | 0.34899397 | 3,290.12002 |
| 4 | 1.22337225 | 0.35733812 | 4,381.19039 |

替换程序 wall=4,391.96231 s，本次 allocation=4,399 s=1.22194444 GPUh，peak=443.35596 MiB。原尝试5,618 s加本次4,399 s，总 allocation=10,017 s=2.78250 GPUh；相对当前 committed 程序时间的额外 allocation=5,625.03769 s。最后一轮 history 时间与程序结束时间的差额也计入实际成本。

### 四格与四类指标

| ParT configuration | Tau AP | NLL | Balanced accuracy | Fixed common-support AP | All-attempt allocation GPUh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native joint | 0.35858391 | 1.22107879 | 0.41840 | 0.32383619 | 1.94250 |
| Native projection | 0.35733812 | 1.22337225 | 0.41500 | 0.32248272 | 2.78250 |
| 1mm joint | 0.35502857 | 1.24293540 | 0.40330 | 0.32086106 | 2.02167 |
| 1mm projection | 0.35421688 | 1.22571671 | 0.41370 | 0.31995100 | 1.80083 |

原生投影的 macro F1=0.41465904、Brier=0.66463187。以下AP以每类25%的validation mixture为条件，recall使用四类argmax。

| Class | AP | ROC AUC | Recall |
| --- | ---: | ---: | ---: |
| CC nu_e | 0.60368818 | 0.78179776 | 0.5080 |
| CC nu_mu | 0.45228808 | 0.68750867 | 0.3548 |
| CC nu_tau | 0.35733812 | 0.63248459 | 0.5056 |
| NC | 0.41279783 | 0.65542275 | 0.2916 |

### 预定配对与交互

区间沿用400次、128个accepted-family/chunk blocks（每类32个）的配对事件bootstrap。固定common support保留8,559 events、Neff=7,959.41365，权重固定。

| Prespecified B−A | Raw tau ΔAP [pointwise 95% CI] | Fixed common-support ΔAP [pointwise 95% CI] |
| --- | --- | --- |
| Native joint − projection | +0.00124580 [−0.00501470,+0.00830354] | +0.00135347 [−0.00498468,+0.00777324] |
| Native joint − 1mm joint | +0.00355534 [−0.00186106,+0.00868839] | +0.00297513 [−0.00192093,+0.00846325] |
| 1mm joint − projection | +0.00081170 [−0.00433001,+0.00629580] | +0.00091006 [−0.00469003,+0.00678403] |
| Native projection − 1mm projection | +0.00312124 [−0.00240699,+0.00842901] | +0.00253172 [−0.00267478,+0.00788205] |

四项raw与固定支持区间均跨0，上界均小于预定+0.01；只在当前设置下排除至少+0.01的收益，不判定一般等效。四格交互 `AP_NJ−AP_NP−AP_CJ+AP_CP`=+0.00043410，CI=[−0.00790547,+0.00875624]，即native pairing收益减去1mm pairing收益。当前没有粒度依赖的区间支持；训练种子、表示压缩与模型提取能力仍待控制。

Native joint−projection的1% FPR τ效率差=+0.0044，CI=[−0.00365371,+0.01089213]；projection native−1mm为−0.0040，CI=[−0.00988984,+0.00489023]。工作点差值也未得到正区间支持。

### 实际成本匹配

Native joint−projection的共同预算=6,992.77439 s。把原尝试和准备开销计入后，projection仅epoch1可用（6,745.22154 s，剩247.55285 s），joint按NLL选epoch4，τ ΔAP=+0.02795768。这是含本次文件失败成本的整轮分辨率点估计，CI未测量；不能解释为一般的架构算力优势。

Native projection−1mm projection的共同预算=6,482.63405 s，低于native replacement含原失败开销后的首个完整epoch成本，结果为 `no_complete_epoch_within_shared_time`，同成本指标未测量。该项保留缺测，不移除原失败成本以获得可比较点。其余两项成本配对见阶段32–33。

### 工作点、τ分支与域

Native projection的validation empirical 1% FPR阈值=0.46394488，通过82/2,500个τ、75/7,500个背景，τ效率=0.0328；native joint为93个τ。0.1% FPR通过14个τ、7个背景，尾部统计不足，最终阈值仍待S6。

| Tau branch | Signal events | Native projection selected | Native joint selected | Projection branch AP |
| --- | ---: | ---: | ---: | ---: |
| Electron | 445 | 7 | 7 | 0.07989121 |
| Muon | 395 | 8 | 4 | 0.05956384 |
| Hadronic | 1,660 | 67 | 82 | 0.29691899 |

通过数使用各自inclusive 1% FPR阈值。joint相对projection的通过数变化为electron 0、muon −4、hadronic +15，合计+11；分支差值CI未测量，机制尚未确定。分支AP使用该分支prior与同一7,500个non-τ背景。

| Native projection validation domain | Events / tau | Tau AP |
| --- | --- | ---: |
| Pixel centered | 885 / 176 | 0.34800415 |
| Large scintillator | 4,494 / 1,053 | 0.39123665 |
| Pixel+bar input empty | 440 / 148 | 0.33636364 |
| Tracker-only response | 158 / 58 | 0.36708861 |

空输入与tracker-only域AUC=0.5，AP等于各域prior。分域AP包含样本组成变化，域内配对CI未测量。训练域、模态与local-shuffle/pair/plane-only控制继续等待冻结矩阵。

### 产物与复现

正式根目录 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`：

- `first_part_factorial.json`：替换manifest/history、终态、完整读回、全部尝试成本、四格模型、配对、交互、分支与域。
- `experiments/part_native_projection_s42_e4_v2/final_checkpoint_checks.json`、`readback_replacement_native_projection.log`：14项pass。
- `campaign_accounting.json`、`evaluation_final/summary.json`、`metrics.csv`、图表和`evaluation_first_part_factorial.log`：本次实际部分评价。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments part_native_projection_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
```

04:18:10 UTC组织累计52.32724 GPU-hours，missing history records=0。正式6 complete、6 running、23 idle、0 held；空出的GPU已接续运行ParticleNet 200 μm joint。Kubeflow v6仍Pending、无node。

## 阶段性结论

首个正式ParT四格与完整输出证据齐备。当前四轮、seed42及既定压缩下，四项主AP增量都没有正区间支持，可排除至少+0.01；交互效应未确定。文件失败已实际解决，原成本保留使部分同预算比较只能使用早期checkpoint或缺测，科学主表与资源表均保留这一事实。

## 下一步计划

继续其他29项未完整正式配置，补齐ParticleNet四格、中间粒度、信息/模态/训练域控制与三种子。最终41项单seed+4项种子均值严格评价、资源与物理解释齐备后验收S2；当前goal保持active。
