# R3：按真实终点依赖提交CPU评价

## 阶段目标

减少GPU训练结束后的排队与串行等待。未完成终点不进入正式评价；CPU评价与同allocation的GPU最终probe尽量重叠，全部冻结科学要求保持。

## 工作内容

新增`analysis.flavour.v2.r3.finish`，从当前revision2完整合同读取四套评价和六模型。只轮询metadata和预测表是否存在，不在工作节点读取事件payload或做大样本统计。

每套评价要求相关模型manifest为complete且endpoint/best-screen两个预测表非空，再通过统一submit提交HTCondor任务。Global-ablation-projection可在消融完成后先运行，不等待其余原生模型。完整六模型的health另提交CPU任务。最终scientific review等待四套results、independent points与independent core intervals全部完成且无失败；应用原有质量和实际效应门槛。

已有submitted记录不重复提交，已失败的提交不静默复用；单observer锁避免重复调度。状态只有在依赖改变时写入，减少小文件I/O。该调度器不决定科学晋级、扩展、图审查或最终验收。

## 详细数据

运行根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。

2026-11-15 18:19:59 UTC从Kubeflow容器内部核对两个真实训练进程均为running，CPU累计各约971.6 s；其当前本地日志记录为：

| Model | Updates | Exposures | Cumulative train seconds |
| --- | ---: | ---: | ---: |
| Native control | 16,500 | 131,844 | 5,450.118238 |
| Native relations | 16,250 | 129,844 | 5,508.983184 |

同一时间前的工作节点EOS日志读回仍为15,500/15,250 updates；打开文件的可见进度有延迟，不能据此认定进程停滞。两模型在`pinpoint-r3-g2-k4-f47p9`的3g.47gb上双并行，100k保存状态之后仍在真实优化。正式有效目标各50,051updates/400,000exposures不变。

Condor消融作业1583010.0在bigbird20继续JobStatus2，RemoteHost为`slot1_1@b9pgpun106.cern.ch`。18:19 UTC本地最近完整JSON行是19,750updates/157,822exposures/6,214.509811 train s；下一行读取时尚不完整，不从半行推算结果。GPU/CPU利用率未测量：namespace metrics API返回Forbidden，CUDA process query返回Insufficient Permissions；已有吞吐、实际进程和终点记录继续作为运行证据。

18:19:15 UTC完整allocation账本快照为13.229062异构device-hours，活动GPU allocation仅`r3_g2_ablation_c9`和`pinpoint-r3-g2-k4`。其中整H100未调度尝试继续保留98s lifecycle上界，不记为确定零成本。该快照尚非最终成本，使用`r2.accounting`保留共享allocation只计一次的扩展字段。

后续资源请求：每项评价8CPU/12GB，health和scientific review各2CPU/6GB；实际提交前由统一入口查询schedd负载、WRITE与EOS策略。尚未到达依赖的任务未预先占CPU或GPU。

启动命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
R3_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3
python -m analysis.flavour.v2.r3.finish --root "$R3_ROOT" --watch
```

18:18:49 UTC首次实际依赖检查：M0/C1评价完成，geometry缺三个G2终点，补充强B1比较缺消融终点；health/scientific review均未完成。状态记录为`runtime/r3_finish_dependencies.json`。观察进程已启动，未因running终点提交任何正式评价。

验证命令：

```bash
python -m unittest tests.test_flavour_v2_r3 tests.test_flavour_v2_r2 tests.test_flavour_v2_r1 -q
```

45项通过，用时6.449s。新增验证覆盖running加部分表不触发、complete但缺best表不触发、终点齐备才提交、独立复核未齐不触发科学review、复核失败拒绝、已提交attempt不重发。真实数据统计未在本阶段运行。

## 阶段性结论

事实：两个恢复模型继续有效运行，依赖调度器已按完整revision2合同实际启动，部分终点未提前进入正式评价。CPU工作可在各自终点完成后与GPU最终probes重叠。

待测：重叠调度的最终时间节省、三个G2完整质量、全部正式模型health和G2最终probe尚未测量。当前没有新的分类性能结论，R3未完成。

## 下一步计划

核对自动提交与真实CPU终态、完整12视图/48配对及补充4视图/8配对的独立复核；检查Kubeflow自动三个原生模型的完整Tune重算与局部/跨块/global/tracker probes。由实际最终收益决定是否需要seed、粒度及机制重训；随后交付全图、条件决定、终态成本和严格accept。Cal-A/Cut-B/Confirm继续封存。
