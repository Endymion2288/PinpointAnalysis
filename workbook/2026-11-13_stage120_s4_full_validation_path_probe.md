# S4 完整 validation 的固定权重分支诊断

## 阶段目标

检验 fine-tuned 分类器是否实际利用 global 表示，以及 regional/local/summary 路径的相对依赖。S3 的 64-event occupancy probe 和 S4 的 frozen linear readability 分别回答不同问题；本诊断覆盖完整固定 10k validation，不重新拟合模型。

## 工作内容

新增 `analysis.flavour.s4.functional_probe`，复用 S3 的 presence-stratified derangement、分类 context 切片和 Head 重放。首批为 seed42 的 Scratch/MAE3 × 1k/10k labels，均已完成 14 项 checkpoint readback。编码器与 Head 使用完整 best checkpoint、CPU FP32、eval 模式；关闭 compilation，不改变参数。

缓存 Head 输入归一化之前的 context，验证重放 logits 与原 forward 在 `rtol=1e-5, atol=2e-6` 内一致。逐项置换 global、全部 regional、全部 local、response summary、全部 spatial。每组使用三个固定 donor mappings（2026/17/42），donor 来自相同 pixel/bar/tracker presence 组；singleton 保持原值，空事件保留。

使用既有 common-support weights 和 400 次 accepted-family connected-chunk bootstrap。先逐 mapping 计算指标差，再对三个 mapping 取均值；相同 recipient block multiplicity 同时应用于原始及干预预测。记录各 mapping 差额，不能将其当作独立训练 seeds。区间条件化于固定模型和 donor mappings。

FP32 baseline 与正式 BF16 保存预测的 RMS/max 概率差、argmax 变化和 NLL/AP 差额单列实测；所有路径干预均减去同一 FP32 baseline。诊断不替换正式分类成绩。

配置为 `roadmap/configs/s4_functional_probe.json`。正式作业通过 HTCondor 的独立 CPU stage 执行，4 CPU，12 GB RAM；代码、详细统计与 event-aligned donor positions 写入 EOS。成本脚本已将该 stage 标记为 CPU-only。

## 详细数据与验证

2026-11-13 20:55 UTC，LCG108 CUDA/Python3.12 执行：

```bash
python -m unittest tests.test_flavour_s4_paths tests.test_flavour_s3_paths tests.test_flavour_s4
```

34 项通过，11.297 秒。新增回归验证只替换指定 context 区间、多个分支使用同一 donor，以及 paired bootstrap 的固定 mapping 均值和 common-support weights；既有 S3 诊断合同未修改。四模型真实 10k-event 功能结果：未测量，待正式批作业。

同期资源状态：Kubeflow `pinpoint-s4-recovery-c` 正常运行 4 个 fits；HTCondor 1574116 与 1574593 运行，1575505、2198680、2199507 排队。未中断正常训练。

## 阶段性结论

已具备直接测量拟合路径依赖的实现和回归证据。置换会破坏分支间相关性；结果描述该 checkpoint 对路径的依赖，不能当作重训 ablation 或硬件的物理因果重要性。global 置换变化小也可能来自冗余，需结合 frozen readability 与 downstream 配对结果解释。

## 下一步计划

提交正式 CPU 诊断，检查缓存重放、完整事件覆盖和 precision drift，再结合独立 seeds 与等计算监督控制解释预训练收益。Calibration/test 保持冻结。
