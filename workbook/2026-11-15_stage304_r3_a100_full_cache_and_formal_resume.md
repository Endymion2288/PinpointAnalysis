# R3：A100完整缓存核对与正式恢复训练

## 阶段目标

核实新A100上的完整输入与原训练输入一致，并用正式优化日志确认消融模型从保存状态继续学习。区分短profile、完整缓存准备、局部正式训练吞吐与最终分类效果。

## 工作内容

继续运行Kubeflow的control/relations双模型及Condor的no-global消融。新A100完成Train100k、Tune40k共享mmap构建后，与原1g和Kubeflow缓存比较事件数、数组字节数、候选图完整摘要及实际散点输入检查；随后核对新manifest、恢复位置和真实滚动优化日志。

直接从所属Condor worker和Kubeflow容器读取最新日志。当前交互节点上的EOS追加日志读取曾滞后，不能仅据其较旧行推断训练停滞；完成状态仍使用原子发布的manifest和完整预测表。

## 详细数据

运行根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。审查快照为2026-11-15 19:35:04 UTC。

### 完整输入准备

| Split | Events | Array bytes | Preparation seconds | Scattered exact input checks |
| --- | ---: | ---: | ---: | ---: |
| Train | 100,000 | 13,925,383,580 | 1,253.584144 | 32 |
| Tune | 40,000 | 5,622,779,908 | 576.291912 | 32 |
| Total | 140,000 | 19,548,163,488 | 1,829.876056 | 64 |

两份缓存均保留精确事件覆盖；上表数组规模、完整候选图totals、最大out-degree及edge-count quantiles与原1g、Kubeflow缓存逐字段相同。每份缓存各32个散点事件的实际collated tensors精确核对通过。该证据不等同于对140k事件逐一重算并比较全部tensor。

Train候选图仍有109,547,012个节点、20,418,294个真实blocks；Tune为44,188,833个节点、8,153,150个真实blocks，两者最大out-degree均为14。候选图摘要一致说明本次缓存没有改变已审计图的统计量，不构成分类收益或完整truth ancestry效率的证据。

完整缓存准备实测1,829.876056s；stage303的14倍Train10k线性估计为3,519.997104s。二者之差只比较准备估计与准备实测，不是完整迁移节省的反事实测量。缓存计时之外的profile、恢复fixture、staging、进程启动和后续验证均继续包含在allocation成本中。

### 正式恢复与优化进展

新manifest已指向`1583024.0`、NVIDIA A100-PCIE-40GB；新增attempt明确记录`resumed_at_update=25028`。恢复的保存状态为200,000次曝光、四类各50,000次、`sum(lr)=6.073386147373478`。正式配置与冻结JSON相等，目标仍为400,000次曝光、50,051 updates、`sum(lr)=7.865059679070928`。

| Model | Actual resource | Latest recorded exposures | Window updates | Window exposures | Window train seconds | Events / train second |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G2 no global | Condor A100-PCIE-40GB | 211,766 | 1,250 | 9,990 | 143.864231 | 69.440471 |
| G2 control | Kubeflow H100 NVL MIG 3g.47gb, shared pair | 283,706 | 1,750 | 13,986 | 420.233142 | 33.281525 |
| G2 relations | Same Kubeflow allocation, shared pair | 273,716 | 1,750 | 13,991 | 435.189253 | 32.149231 |

以上均为直接读取所属worker后，用首末完整滚动记录的曝光差除以训练计时差。它们排除缓存、staging、验证和进程启动，且只覆盖当前窗口，不能称为完整训练吞吐。A100短profile此前为66.20/66.55events/s；本阶段的69.44来自正式100k训练池上的实际窗口。

原消融writer `1583010.0`已在stage303确认终态；其10,943s和至少19,758条未保存曝光的重放成本继续计入。新模型的有效学习预算没有追加。新GPU上的连续/恢复fixture最大参数差0已在stage303记录；该fixture与本阶段的正式优化进展共同支持恢复路径有效，不证明任意跨GPU轨迹逐位相同。

### 资源与验证

19:35:42 UTC的`r2.accounting`快照为15.973047异构device-hours、183.611524 allocated CPU core-hours，35条allocation记录；活动项仅`r3_g2_ablation_c10`和`pinpoint-r3-g2-k4`。

| Resource label | Device-hours in this snapshot |
| --- | ---: |
| H100 1g.12gb, all attempts | 9.388056 |
| A100-PCIE-40GB, all attempts | 1.027496 |
| H100 NVL 3g.47gb, all attempts | 5.530274 |
| Previous h100-94gb lifecycle upper bound | 0.027222 |

最后一项保留未调度整卡尝试的98s生命周期上界。所有并行worker按allocation计一次，异构整卡/MIG小时不作FLOP等效；两项活动allocation尚未终止，此表不是最终成本。

证据：

- `audit/g2_ablation_c10_formal_execution.json`：完整缓存对照、正式恢复计数、所属worker滚动记录及局部吞吐。
- `groups/g2_ablation_c10.json`与`groups/g2_k4.json`：实际资源和完整缓存准备。
- `experiments/b2_g2_global_ablation_t4_s42/manifest.json`：实际新attempt及保存状态；`training_cache`仍保留初始保存记录的来源，本次新缓存准备以group证据为准。
- `resources.json`：当前完整allocation账本。
- `runtime/r3_finish_dependencies.json`：自动CPU后续依赖；观察进程已确认存活。

主要核查命令：

```bash
condor_q -name bigbird20.cern.ch 1583024.0 -json \
  -attributes ClusterId,ProcId,JobStatus,RemoteHost,NumJobStarts
kubectl --context ml-prod -n xcheng get pod pinpoint-r3-g2-k4-f47p9
python -m analysis.flavour.v2.r2.accounting --root "$R3_ROOT"
```

文档及JSON核对：三个正式G2配置、campaign和completion contract的source/live解析一致；完整缓存对照十项通过；`git diff --check`通过。训练源码未改变，本阶段未重复stage302已通过的47项不同tests。Cal-A/Cut-B/Confirm的payload继续封存。

## 阶段性结论

事实：A100完整Train/Tune准备及实际输入核对完成，消融已从200k保存状态恢复并继续真实优化。Kubeflow同卡双模型仍在训练。完整输入覆盖这一项已完成，不再列为待办。

待测：三个G2完整终点的分类效果、完整特征probe和最终训练健康；当前吞吐及中间训练损失不能决定结构晋级。R3仍未完成，不形成新的分类性能结论。

## 下一步计划

完成三个400k曝光终点；检查自动提交的完整geometry及no-global对强B1评价、九指标独立区间复核和六模型最终health；复用Kubeflow缓存完成三个native最终probe。随后按实际质量决定必要扩展，完成英文图审查、科学决定、终态账本和严格accept。
