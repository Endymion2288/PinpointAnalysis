# S4 完整预训练语料与正式 GPU pilot

## 阶段目标

完成 100k train 的预训练语料及 partial Rel targets，启动真实事件拟合和短预训练 pilot。只在此阶段报告数据与执行事实，下游分类收益尚未测量。

## 工作内容

HTCondor 16 个转换任务全部 exit 0，产生 8 个新 mixed observation shards，并为原始 train 和新增 train 合计写出 40 个 Rel sidecars。原始 32 shards 通过只读链接复用，10k validation 未改动。

汇总的第一版逐行构造 Python 对象过慢，改为 Arrow list-buffer 和 NumPy 批量对应。调度重启时根据缓存的 queue 状态判断第二个汇总仍 idle，实际上它已执行约 31 秒，因移除而中断；这次额外 CPU 消耗保留在资源记录。最终汇总 `2198166` 在 bigbird21 执行。后续调度依据同时核对 event/worker progress，避免用单个缓存状态迁移正在前进的任务。

新增提交布局将小型 wrapper 和 scheduler metadata 留 AFS，scheduler stdout/stderr 为 `/dev/null`，worker 直接将全部实际日志重定向到 EOS。因此可按 CERN 的 NoEos policy 正常使用原先不接受 EOS submit paths 的 schedd，未关闭其提交检查。每次仍保存实时 load 和 WRITE 依据。

## 详细数据

输出根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。

| 项目 | 实测 |
| --- | --- |
| Train | 100,000；四类各 25,000 |
| Downstream validation | 原有 10,000；四类各 2,500 |
| Rel train events | 100,000 |
| Matched bar channels | 69,948,793 |
| Mixed-contribution bars | 34,294,631 |
| Other-category energy fraction | 1.7010620725474244e−5 |
| 最大硬件/Rel 总能量相对差 | 0.0，双方以相同 float32 存储比较 |
| 完整组装检查 | 4/4 通过 |

质量依据为 `data_summary.json`；per-track contributions 到 hardware channel 的对应只验证新增 target 实现，不重新审查已接受的样本认证或响应参数。

正式 pilot Job `pinpoint-s4-pilot-a` 已获 Kubeflow `NVIDIA-H100-NVL` + `nvidia.com/mig-3g.47gb:1`。平台注入 22 CPUs、63 GB RAM。5 项配置共享分区，互斥 claims 防止与 HTCondor 备用任务重复训练。仍 idle 的 Condor GPU copy `1574091` 在 Kubeflow 领取全部 5 项后移除。GPU 等待数据、编译、失败重启和所有后端成本由 `s4/accounting.py` 记录，累计不得超过 60 device-hours。

数据等待有十分钟上限。本次数据在超时前完成，真实 32-event 分类和重建拟合已开始；这些拟合、最大 occupancy FP32/BF16 stress 和 corpus pilot 的最终数值尚未测量。

## 阶段性结论

本轮 100k train 的 bar 软贡献标签可用，其中约半数 channel 有多类别能量贡献；将其硬化为首次贡献者会丢失实际混合信息。是否改善四分类和 ντ 仍需下游对照。

## 下一步计划

检查 5 项 pilot 的拟合曲线、内部梯度、跨事件方差、正负查询 AP 与先验，以及 maximum-occupancy stress。通过后冻结主 mask 和正式预训练量；按 1k/10k/100k 标签预算执行同 backbone 的 Scratch/MAE/EMA/MAE+barRel、frozen linear probes、full fine-tuning 和 matched-compute 控制。保留 S3 train-only branch probes 和 ντ 子道评价。
