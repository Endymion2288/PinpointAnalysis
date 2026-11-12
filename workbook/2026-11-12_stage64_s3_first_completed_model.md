# 阶段64：S3首个完成模型的四分类结果

## 阶段目标

读回首个完成模型，报告完整四分类和真实训练成本。其余screen与确认仍待完成，本阶段不选择候选、不作S3验收。

## 工作内容

`s3_a4_late_s42_e4`完成四轮训练，Condor `1573835.0`在bigbird20返回JobStatus4、ExitCode0。监视器对发布模型执行14项CPU检查，checkpoint、optimizer、冻结配置、NLL选择、validation事件键、各轮预测、有限概率和exposures全部通过。评价仅使用已接受的10,000 validation事件；calibration/test未使用。

用S3评价入口生成该模型及16个只读S2结果的诊断。输出保留全部冻结请求，其中39项实验、配对或交互仍pending；非strict运行完成只证明当前产物可评价，不代表S3通过。英文混淆、ROC、PR、learning与资源图位于`evaluation_progress_first/`。

补充明确冻结1 mm表示：`reducer=count`保留bin内native hit数量，丢弃bin内细坐标残差；不是binary OR。16-pitch窗口约0.3328×0.3648 mm，小于1 mm coarse中心间距，coarse表示在该窗口中多数为单node。2×2交互应解释为这一具体表示与编码组合的作用。仅澄清文档，不改变输入、模型或配置。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

训练为seed42、H100 NVL、Torch2.7.1/BF16、80,000 train事件。373,060参数；80,064次optimizer更新、0 AMP skips；320,164 training exposures（固定bucket/batch合同含重复曝光）。NLL选择第四轮checkpoint。

| Model, seed42 | Macro AP | NLL | Macro F1 | Balanced accuracy | Allocated GPUh |
| --- | ---: | ---: | ---: | ---: | ---: |
| S3 4 mm late | 0.358741130 | 1.330100921 | 0.334572800 | 0.346000 | 0.987500000 |
| S2 ParT native | 0.462286490 | 1.221078794 | 0.418442672 | 0.418400 | 1.942500000 |
| S2 ParticleNet native | 0.434523607 | 1.239305631 | 0.392041487 | 0.397400 | 1.752777778 |
| S2 B0 all | 0.435710686 | 1.245042368 | 0.404577173 | 0.402900 | GPU未使用 |

以上为完整validation上的点值，未将这一临时表当作冻结配对或三seed晋级判断。

| Class | AP | Precision | Recall | F1 | Efficiency at 1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| CC nu e | 0.467282092 | 0.463876432 | 0.3724 | 0.413135123 | 0.1044 |
| CC nu mu | 0.360682338 | 0.319030680 | 0.5740 | 0.410117176 | 0.0492 |
| CC nu tau | 0.287407299 | 0.292458261 | 0.2032 | 0.239792306 | 0.0076 |
| NC | 0.319592792 | 0.333333333 | 0.2344 | 0.275246595 | 0.0296 |

每类2,500事件。Macro AUROC0.609195273，Brier0.719555655。固定E/vertex common support有8,559个正权重事件、Neff7,959.413646535184，macro AP0.346885998、NLL1.336476307。Target Z未匹配，条件AP的prior随选择改变。

当前模型pixel-centered域885事件，macro AP0.485819348；large-scintillator域4,494事件，0.414471203；all-empty域282事件，0.250000000。它们与总体分母和prior不同，不将域间AP差直接解释成几何的因果增益。

真实allocation3,555 s，程序3,531.034599 s，差23.965401 s；peak allocated GPU memory397.883301 MiB。2026-11-12 15:22:44 UTC账本：两后端累计5.567020872 GPUh，加入首次交互profile保守上界后5.748163485 GPUh，missing history0。此为进行中快照，包含已记录失败尝试，不是最终成本。

复现命令（使用仓库匹配的LCG108环境及`PYTHONPATH=src`）：

```bash
python -m analysis.flavour.s3.accounting --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12
python -m analysis.flavour.s3.evaluate --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12 --campaign roadmap/configs/s3_screen.json --comparisons roadmap/configs/s3_pairs.json --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/evaluation_progress_first
```

评价实际exit0，统计程序wall32.822447 s（图生成另计）。原始证据：模型`manifest.json`、`history.json`、`final_checkpoint_checks.json`、`evaluation_progress_first/summary.json`、`campaign_accounting.json`及Condor event log。

## 阶段性结论

事实：首个S3模型训练与读回完整。其四类AP均低于S2 ParT native；较低成本伴随显著点值质量差距，当前结果不支持该模型晋级。该配置为4 mm局部窗口和late fusion，不能代表尚未完成的fine-window、context或latent模型。

待检验假设：shared 4 mm压缩是否抹平native优势，仍须完整2×2与三seed比较。当前观测没有隔离该因果。其余配置继续按冻结合同训练，不因初期质量低而停止分支。

## 下一步计划

继续17项screen；确认task3恢复实际读取epoch1，并完成四轮和14项读回。HTCondor已有实际运行任务；Kubeflow当前Pending、无node，等待真实ready才交接still-idle任务，保持单writer。全部screen读回与终态成本齐全后才冻结确认配置，执行完整四分类、三seed、条件域、错误显示与strict验收。
