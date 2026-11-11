# 阶段 24：补测 H100 的完整事件资源曲线

V100 上三项科学任务继续训练，两项因 Startd 关闭重排。为判断后续资源安排是否有可测量的改进空间，增加独立 H100 资源测量；正式科学训练仍沿用当前 V100 合同。

## 阶段目标

在同一 train occupancy-rank 子样本、模型宽度和表示上，测量 H100 的完整事件延迟与显存，比较既有 V100 曲线。先取得实际执行和资源数据，再决定后续安排。

## 工作内容

提交器新增 `--profile-events` 与 `--profile-tag`，默认仍为 24 events、`resource_profile_v2`。辅助测量写入 `resource_profile_h100_v2`，使用 LCG_108/Torch2.7.1 与 NVIDIA H100 NVL。该软件构建用于 H100 的执行与现有 V100/LCG_107 结果分别记录。

测量沿用现有 `s2.profile`：train partition 中按完整 pixel/bar/tracker channel occupancy 的等距 rank 抽取 24 个事件，覆盖全空与最大占用；六种表示 × 两个模型 × inference/training 两种模式，目标为 576 行。ParT dim48、ParticleNet dim64，单事件 batch，不截断。training 模式只反传以测资源，不执行 optimizer fitting。

## 详细数据

20:54 UTC 左右的 Condor 资源快照中，`b9pgpun014/015` 的 H100 parent slots 为 Unclaimed/Idle，AvailableGPUs 分别为 2/4；这只说明 collector 快照，实际可调度性由提交与匹配结果确认。V100 上的两项原 JobId 仍 idle，其他三项仍 running。

H100 的实际 latency、memory、loss finite 与 allocation 均未测量。既有 V100 数据保留在 `resource_profile_v2/`；辅助输出目录为 `resource_profile_h100_v2/`。

计划执行入口：

```bash
python -m analysis.flavour.s2.submit profile --root "$S2_RESULTS" \
  --profile-events 24 --profile-tag resource_profile_h100_v2 \
  --lcg-view LCG_108_cuda --gpu-device 'NVIDIA H100 NVL'
```

`$S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11`。提交器保存当前 schedd/EOS/WRITE、GPU 资源快照、精确命令与 ClusterId，后续 allocation 由现有 accounting 汇总。

## 阶段性结论

当前事实是两个 V100 任务因 Startd 关闭而重排，H100 collector 存在空闲 slot；H100 的实际资源收益尚未测量。资源测量不会提供分类性能结论，科学合同暂不调整。

## 下一步计划

提交并检查 H100 曲线，按相同物理事件、模型、表示和模式与 V100 比较，明确首事件 warmup 与 batch1 stress sampling 的范围。继续观察既有 35 项科学任务；只有测量支持实际收益时再评估后续资源安排。
