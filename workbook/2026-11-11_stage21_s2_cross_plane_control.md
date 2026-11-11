# 阶段 21：补齐显式跨层消息传递对照

为落实 S2 中“量化 layer 内局部结构、跨层连通和能量/位置先验各自贡献”的要求，增加两项同层消息控制。默认模型通过旧版数值一致性检查，原有 33 项配置继续执行；追加两项后共 35 项科学训练、41 项单 seed 配对及原定 4 项三种子均值。

## 阶段目标

估计显式跨层消息传递相对于 plane-local 编码、全局总量汇聚和末端分类头的额外贡献。已有 no-pair-bias 与 fixed-graph 控制仍保留跨层消息，不能单独回答这一问题。

## 工作内容

ParT global attention 增加同层 mask，按既有 normalized z 的 `1e-6` 容差判断同层；ParticleNet 的物理和 dynamic 全局 graph 都限制在同层，保留最多 k=16 的同层邻居。local blocks、完整事件覆盖、response totals、全局各模态 mean/max pooling 和分类头保持同一结构。参数量仍为 ParT 151,228、ParticleNet 161,156。

该控制保留 plane z 和全局分布，末端头仍能组合多个 plane 的汇聚信息。它测量“显式跨层交流的增量”，不是移除一切纵向信息，也不将差值解释为某一种 τ 衰变特征的纯因果贡献。

新增 [两项科学配置](../roadmap/configs/s2_cross_plane_controls_v2.json)：`part_native_planeonly_s42_e4_v2`、`pn_native_planeonly_s42_e4_v2`，同一 80k/10k、4 epochs、20,016 attempted steps/epoch、V100/LCG_107/FP16，seed42。配置在读取 native-joint 的首轮 validation 前固定，依据是验收条目缺少直接对照。

旧 39-pair 文件保留，新增 [完整 41-pair 合同](../roadmap/configs/s2_complete_pairs_v2.json)，加入 plane-only→native-joint 两个差值；原 4 个三种子均值不变。总组织上限仍为 200 GPU-hours，控制与确认追加上限 100，旧尝试继续计入；没有新增全笛卡尔扫描。

## 详细数据

CPU 检查：9 项 spatial tests passed，5.307 s；3 项 training/resume tests passed，5.114 s。新增检查直接验证同层 attention、physical graph、dynamic graph 的输出对另一层输入的梯度为零，同时同层梯度非零，并覆盖 300-block query tiling、空事件、padding 和完整模型反传。

以 commit `022dca7` 的模型源码为参考，dim16 的两模型分别在同一 3-event、multi-plane/multi-modal、empty/padded、最大 300 blocks 的 CPU batch 上比较默认行为：输出最大绝对差=0，参数梯度最大绝对差=0。结果保存为 `cross_plane_default_equivalence.json`，环境 Torch2.7.1。默认新参数为开启跨层交流，控制选项显式关闭。

首次小规模本地 CUDA 检查使用 LCG_107/Torch2.5.1、T4；查询时空闲 877 MiB，执行期间降至 22.62 MiB，在线性层申请 20 MiB 时 OOM。该检查退出，未完成 optimizer step，不能当作新控制 GPU 验证通过。独占 GPU allocation 不适用于该共享交互节点，实际 GPU 运行秒数未测量，单列为交互验证失败，不伪造 Condor allocation。

改用 [专用 GPU smoke 配置](../roadmap/configs/s2_cross_plane_smoke_v2.json)，复用已有 29 train/3 validation 的小样本、原生 pixel+bar；模型 dim48/64 与科学配置一致。它只验证 FP16 执行与 checkpoint，不用于分类性能判断。提交器增加 `--training-flavour espresso` 支持短 GPU 检查，科学任务仍使用默认 `tomorrow`。

commit `ff84b5f` 后按当前 schedd/EOS/WRITE/负载检查提交为 `1573755.0/1`（bigbird20），JobFlavour=espresso、JobPrio=20，当前仍在排队。精确提交记录为 `smoke/submission_train_194416.json`，归入 S2 的 Condor allocation 核算。两项新科学配置暂待 smoke 结果，原有 33 项训练均已提交。

进一步在科学运行环境 LCG_107/Torch2.5.1 执行 9 项 spatial tests，全部通过，3.937 s；包含 padded query 与所有真实 key 不同层、没有任何允许 key 的情况，输出与梯度仍有限。日志为 `spatial_unit_tests_lcg107.log`。这项 CPU 检查补充软件版本兼容性，专用 GPU 执行结果仍待测量。

两项 smoke 持续 idle 后，仅对这两项 idle jobs 调整为任意 `Capability>=7.0`、显存至少 14,000 MiB 的兼容 GPU，并按零 workers 的小样本需求申请 2 CPUs、6 GiB RAM、2 GiB disk。精确覆盖记录在 `smoke/cross_plane_smoke_resource_override.json`，JobFlavour 仍为 espresso，实际 GPU 型号将由训练 manifest 记录。该执行检查使用 LCG_107/Torch2.5.1；全部科学对照的硬件合同仍为 V100、4 CPUs、12 GB RAM。没有重复提交 smoke。

```bash
python -m unittest discover -s tests -p test_flavour_spatial.py -v
python -m unittest discover -s tests -p test_flavour_s2_training.py -v
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS/smoke" \
  --campaign "$PWD/roadmap/configs/s2_cross_plane_smoke_v2.json" \
  --lcg-view LCG_107_cuda --training-flavour espresso
```

`$S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11`。CPU 日志为该目录 `spatial_unit_tests.log`、`training_unit_tests.log`。

## 阶段性结论

事实：新的限制在 CPU 检查中阻断跨层消息；默认分支在上述数值样例中与旧版完全一致。新增控制的科学增益尚未测量，专用 GPU 执行待验证。此前已测的首轮指标只用于运行监测，不据此修改默认训练架构或选择最终模型。

## 下一步计划

专用 GPU smoke 完成后在 V100 上提交两项科学对照，与其他指定 runs 一起完成。最终使用 41-pair 完整合同，配合 4 个三种子均值、粒度×关联 interaction、输入响应域、τ 分支、实际成本和错误事件组，交付 S2 科学结论。
