# S4 修复后的 1k train-inner 重选与固定重训合同

## 阶段目标

用完整修复 LP 候选与正常 non-LP 控制重新共同选择 1k recipe/epoch，再启动有效的固定末轮重拟合。

## 工作内容

六项 `s4_inner_lpft4_*_n1000_*` 已完成全部12个 complete passes。与18项未受冻结编译故障影响的 non-LP候选一起，通过 `plan_low_label resolve` 验证冻结配置、完整曝光、真实 input/local 训练梯度后，按原预设规则重做选择：三折按留出事件数加权，Scratch/MAE3 两种初始化等权，共同选择 recipe/epoch，不读独立 validation。

## 详细数据

新选择为 LP-FT、8个 complete passes，mean OOF NLL 为 **1.419278126**，其中 Scratch **1.448790994**、MAE3 **1.389765258**。这是两个初始化共同采用的设置，不是为每个初始化各选最优轮次。选择仍与旧失效选择同名同轮次，但由新的有效训练重新得到；旧故障 checkpoint 不复用。完整候选表见[新1k选择](../roadmap/results/s4_transfer4_selection_1k.json)。

固定重拟合每 pass 为74次更新，8 pass 共592次更新、8,000次事件曝光，前148次只训练 linear readout，后444次解冻训练。新命名 `s4_transfer4_optimized_{scratch|mae3}_n1000_s{17|42|2026}`，配置见[六项refit](../roadmap/configs/s4_low_label_refit_v4_1k.json)。选定末轮提交，不以validation再选epoch。

另使用相同共同设置重新 fine-tune 已通过学习 readback 的 EMA3、MAE5、channel Rel5 encoders，均为 seed42，见[目标对照](../roadmap/configs/s4_transfer4_objectives_1k.json)。这些是下游重训，SSL本身未重训；channel Rel包含额外模拟语义监督，其含义保留于配置。

[九项正式dispatch](../roadmap/configs/s4_transfer4_refit_1k_dispatch.json)使用修复源码和gate依赖。冻结[12模型评价](../roadmap/configs/s4_transfer4_1k_evaluation.json)、[配对及seed均值](../roadmap/configs/s4_transfer4_1k_pairs.json)与[CPU任务](../roadmap/configs/s4_transfer4_1k_validation.json)，比较阶段145六项基线与六项新重拟合，要求readback、独立FP32、实际训练健康及完整10k validation；400次物理块bootstrap、τ子道与低FPR计数沿用预设评价。新重拟合的分类结果尚未测量。

执行命令：

```bash
python -m analysis.flavour.s4.plan_low_label resolve \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 \
  --source roadmap/configs/s4_low_label_inner_v4.json --budgets 1000 \
  --selection-report /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/evaluation/low_label_optimization/selection_v4_n1000.json \
  --output roadmap/configs/s4_low_label_refit_v4_1k.json
```

## 阶段性结论

事实：修复后完整LP训练已通过真实前端梯度要求；1k新共同选择为LP-FT 8 pass。其 OOF NLL 仍接近或高于均匀四分类的 ln4，低标签概率质量仍是待解决的实测问题，不把完成训练视作泛化改善。

研究设计在此前观察S4 validation后形成，后续重复使用原validation的结论保持条件性。Calibration/test继续冻结，原samples与detector-proxy接受状态不变。

## 下一步计划

正式提交九项重训和CPU评价，优先使用先可用的后端并共享claims；10k六项LP完成后按同规则重选。核查最终参数更新与每阶段梯度，完成完整物理统计后再判断收益。
