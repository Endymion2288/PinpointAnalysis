# S4 三个 Scratch seeds 的统一验收与 MAE continuation

## 阶段目标

确认低标签监督的seed波动，读回新增模型，明确partial Rel的监督资源口径。

## 工作内容与详细数据

正式CPU作业2200713（bigbird21）完成v5，程序169.048秒，allocation207秒，exit0。9个train模型的14项checkpoint检查全部通过，共819个训练健康samples；21组预测进入完整10k validation统计，仍有145个模型/比较依赖待完成。`requested_complete=false`，本轮为进度评价。

| Labels | Seed | Scratch macro AP | Scratch tau AP | Scratch NLL |
| ---: | ---: | ---: | ---: | ---: |
| 1k | 17 | 0.28666413 | 0.26370358 | 1.54826812 |
| 1k | 42 | 0.29720868 | 0.25953440 | 1.51618843 |
| 1k | 2026 | 0.28788000 | 0.26030673 | 1.55833801 |
| 10k | 17 | 0.42583541 | 0.32450086 | 1.27288043 |
| 10k | 42 | 0.42710017 | 0.32129605 | 1.27706020 |
| 10k | 2026 | 0.42693990 | 0.32533589 | 1.27282898 |

1k的macro AP三seed均值为0.29058427、sample SD为0.00576903；tau AP为0.26118157±0.00221800。10k对应为0.42662516±0.00068862和0.32371093±0.00213262。SD描述固定样本下三个训练初始化的波动，不是MC事件误差或均值置信区间。

MAE3 seed42的1k/10k macro AP为0.36179163/0.44504900。相同seed配对增量为+0.06458296（95%区间[0.05791726,0.07131453]）和+0.01794883（[0.01254786,0.02408212]）。不以一个MAE seed减去三个Scratch均值代替配对。独立MAE downstream seeds尚待完成。

819个samples中，FP32 SiLU `head.2`精确zero fraction均为0，RMS总范围0.138907–0.803596。完整validation预测均非恒定，informative输入梯度均大于0。低标签晚期梯度随拟合减小，train loss降低而validation NLL升高，按过拟合解释；不能仅凭最小梯度断言Head失效。

`s4_mae_cont_e2_s42`完成2×6144updates，并通过integrity/internal-health与measured-learning readback。它从MAE3同一parent开始，fresh optimizer/schedule，使用新views3/4；固定inner probe仍为view0。细pixel patch/4mm/bar AP为0.46793925/0.78890544/0.90150251，global effective rank8.23681。四分类收益尚未测量。

## 验证与产物

完整结果为`results/flavour/s4_2026-11-13/evaluation/progress_v5/`，精简数值为[s4_progress_validation_v5.json](../roadmap/results/s4_progress_validation_v5.json)。已人工查看`label_efficiency_train_full.png`：Scratch1k/10k显示三seed SD，MAE目前单seed，100k MAE尚未画入。

新增路径干预绘图接口，生成NLL/AP的英文PDF、PNG和CSV。Python编译检查通过；实际路径图未生成，CPU全量功能诊断仍在执行。

正式命令为`python -m analysis.flavour.s4.submit --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 --backend condor --stage validation_job --name s4-validation-progress-e --campaign roadmap/configs/s4_validation_job.json --tasks 0 --apply`。

21:09 UTC清理30个已核实终态的AFS小提交文件，共25793bytes；提交文本仍存EOS。记录为`audit/afs_cleanup_20261113T210931.json`。

## 阶段性结论与监督口径

修复后的低标签模型能够拟合；Scratch在1k存在可测seed波动，10k更稳定。单seed MAE收益继续成立，tau低FPR效率增益尚未由区间确认。

MAE/EMA仅从observation构造目标；partial bar Rel另使用98k pretrain-fit模拟事件的channel语义，以及2k原train内probe的语义评价。它衡量“事件类别标签效率 + 额外模拟语义监督”，不属于仅用1k/10k总标注的纯自监督比较。模拟贡献不进入推理输入。MAE5和Rel的parent、steps、mask views相同，附加Rel目标与decoder计算量另列，不称为相同FLOPs。

## 下一步计划

完成独立MAE downstream seeds、等计算监督控制、Rel/EMA下游和全量路径诊断，再执行strict矩阵验收与资源清理，明确本轮是否扩至1M。
