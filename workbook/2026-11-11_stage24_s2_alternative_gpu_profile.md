# 阶段 24：H100 的完整事件资源曲线

V100 上三项科学任务继续训练，两项因 Startd 关闭重排。为判断后续资源安排是否有可测量的改进空间，增加独立 H100 资源测量；正式科学训练仍沿用当前 V100 合同。

## 阶段目标

在同一 train occupancy-rank 子样本、模型宽度和表示上，测量 H100 的完整事件延迟与显存，比较既有 V100 曲线。先取得实际执行和资源数据，再决定后续安排。

## 工作内容

提交器新增 `--profile-events` 与 `--profile-tag`，默认仍为 24 events、`resource_profile_v2`。辅助测量写入 `resource_profile_h100_v2`，使用 LCG_108/Torch2.7.1 与 NVIDIA H100 NVL。该软件构建用于 H100 的执行与现有 V100/LCG_107 结果分别记录。

测量沿用现有 `s2.profile`：train partition 中按完整 pixel/bar/tracker channel occupancy 的等距 rank 抽取 24 个事件，覆盖全空与最大占用；六种表示 × 两个模型 × inference/training 两种模式，目标为 576 行。ParT dim48、ParticleNet dim64，单事件 batch，不截断。training 模式只反传以测资源，不执行 optimizer fitting。

## 详细数据

20:54 UTC 左右的 Condor 资源快照中，`b9pgpun014/015` 的 H100 parent slots 为 Unclaimed/Idle，AvailableGPUs 分别为 2/4；这只说明 collector 快照，实际可调度性由提交与匹配结果确认。V100 上的两项原 JobId 仍 idle，其他三项仍 running。

提交 `1573759.0` 于 20:56:54 UTC，39 s 后在 `b9pgpun501` 开始执行，21:00 UTC 已 exit0。实际 GPU 为 NVIDIA H100 NVL，Python3.12.11、Torch2.7.1、BF16、PyArrow20.0.0。Condor history 的 terminal allocation 为 167 s（0.04639 GPU-hours）；实际 RequestCpus=3，软件仍只使用一个 Torch thread。全部 576 行完成、loss 全部有限；H100 原始曲线和配对图均已实际查看。

新增 `s2.compare_resources`，按 event/model/representation/mode 与既有 V100 数据逐行配对。576 行完全对应，nodes、blocks、参数量及 dim 一致。每组去掉首事件后保留 552 行，即 24 组各 23 个事件，避免首事件 warmup 主导稳态比较。

| Native joint, training | V100 median compute [ms] | H100 median compute [ms] | Sum(time V100) / Sum(time H100) | V100 peak allocated [MiB] | H100 peak allocated [MiB] |
| --- | --- | --- | --- | --- | --- |
| ParT | 67.42449 | 37.06868 | 1.74987 | 204.22803 | 241.13574 |
| ParticleNet | 43.95782 | 23.44264 | 1.69249 | 549.66064 | 597.21387 |

training compute 为 forward+backward，六种表示的总耗时比范围分别为 ParT **1.72447–2.32543**、ParticleNet **1.69249–2.22478**。这是 batch1、occupancy-rank stress sample 的执行环境比较，GPU、Torch 版本、precision 和 CPU host 均不同；loading/collation 与整轮训练不在该 compute 量中。其结果支持完整训练吞吐试跑，整轮加速倍数仍未测量。

既有 V100 数据保留在 `resource_profile_v2/`；H100 原始输出为 `resource_profile_h100_v2/`；配对明细、CSV、PNG/PDF 与 JSON 为其中 `comparison_reference/`。首次比较导出遇到 NumPy boolean JSON serialization 错误，已将分组字段转换为 Python 标量；相同真实输入重新运行正常退出，summary 确认 matched=576、noncold=552、all losses finite=true。

复现入口：

```bash
python -m analysis.flavour.s2.submit profile --root "$S2_RESULTS" \
  --profile-events 24 --profile-tag resource_profile_h100_v2 \
  --lcg-view LCG_108_cuda --gpu-device 'NVIDIA H100 NVL'
python -m analysis.flavour.s2.compare_resources \
  --reference "$S2_RESULTS/resource_profile_v2" \
  --target "$S2_RESULTS/resource_profile_h100_v2"
```

`$S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11`。提交器保存当前 schedd/EOS/WRITE、GPU 资源快照、精确命令与 ClusterId，后续 allocation 由现有 accounting 汇总。

## 阶段性结论

H100 的实际调度与完整事件执行已验证，batch1 training compute 在上述同事件比较中较快、显存占用略高。完整训练的 I/O、batch4 与 validation 成本需要另外测量。该结果只支持资源安排的后续试跑，正式科学模型的架构/粒度结论仍待完成训练。

## 下一步计划

在独立目录运行两族 native-joint 的同配置完整 epoch，依据预先固定的耗时与执行标准决定后续硬件安排；最终科学比较须在同一执行环境中完成全部 35 项配置。继续观察既有 V100 任务，将两条执行路径的全部 allocation 计入 200 GPU-hours 总组织上限。
