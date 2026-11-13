# S4：保留逐 bar channel 的语义 decoder

## 阶段目标

检验FASERCal式kept-hit语义目标是否需要pooling前的逐通道表征，而非仅使用block/global压缩结果。

## 工作内容

`PlaneModuleClassifierV2.encode_channel_features`导出原有input和两层local attention的逐通道features；原有`encode_features`随后执行完全相同的mean/max pooling和summary。参数名、state dictionary及分类计算保持兼容。新增Rel选项`global_and_bar_channels`仅在预训练decoder中读取global latents与全部visible bar的pooling前features，使用相同encoder参数；不增加truth输入、truth-based routing或event截断。主MAE reconstruction仍只读取global latents。

保留前一个`global_and_bar_local`作为block-level对照；新的`s4p6_barrel_channels_s42`与它使用相同1024steps、mask/query/loss。`s4_rel_repair.json`、`s4_main_wave_b.json`及独立标注的FP32原型配置中均记录。两项Rel结果目前未测量。

主wave还冻结conditional EMA3与三个label budgets的fine-tuning/frozen probes。完整EMA从fresh initialization开始，只在batch64修复pilot通过重建与非近rank-one谱检查后启动；不重用pilot权重。若失败，依赖停止，保留具体优化和特征谱证据。

## 详细验证

与改动前`e9d264c`的实际classifier实现载入同一state，在相同合成ragged batch上比较FP32 forward与全部parameter gradients：输出max absolute difference=0，所有gradients逐值相同，strict state dictionary load通过。记录于输出根`audit/channel_API_regression.json`。

`python -m unittest discover -s tests -p 'test_flavour_s3*.py'`：49项通过。`python -m unittest tests.test_flavour_s4`：19项通过。新增语义route检查覆盖detach global后bar route仍能反向传播，并核对block/channel两种Rel的shape预算与真实forward/backward hooks一致。

## 阶段性结论

逐channel接口没有改变既有分类输出和梯度；新增语义route解除的是kept-bar目标的局部pooling瓶颈。它能否超过prior及block route，须由pilot语义readback决定，不依据接口测试作物理结论。

## 下一步计划

在下一正式GPUwave运行两项Rel和EMA修复，按学习readback启动continuation和下游；完成完整validation、paired seeds、成本和1M decision。
