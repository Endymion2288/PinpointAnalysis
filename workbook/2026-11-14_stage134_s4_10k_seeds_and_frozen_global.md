# S4 10k标签独立seeds与frozen global表示

## 阶段目标

检验低标签收益能否在10k标签复现，区分预训练表示可读性、实际fine-tuning收益与Rel的额外作用。

## 工作内容与详细数据

v8 CPU作业2204029（bigbird21，源码e87a2a0）完成69组预测、44个单配对和8个三seed配对，65项待齐；24个模型各14项checkpoint读回及2,578次健康采样通过。精简数据为`roadmap/results/s4_progress_validation_v8.json`，完整结果及英文图表位于输出根`evaluation/progress_v8/`。已人工查看global frozen标签效率图，横轴仅含实际测量的1k/10k/100k预算。

10k下Scratch/MAE3的macro AP三seed均值±sample SD为0.426625±0.000689 / 0.441230±0.003676；NLL为1.274257±0.002428 / 1.251373±0.005508。

| MAE3 − Scratch, 10k labels | Raw delta [CI95] | Common-support delta [CI95] |
| --- | ---: | ---: |
| Macro AP | +0.014604 [0.010073,0.019366] | +0.012688 [0.008083,0.017899] |
| NLL | −0.022883 [−0.028124,−0.017563] | −0.019537 [−0.024658,−0.013913] |
| τ AP | +0.004274 [−0.002470,0.011125] | +0.005302 [−0.001186,0.011780] |
| τ argmax F1 | +0.000793 [−0.008578,0.009589] | −0.000756 [−0.011507,0.009767] |

三个macro AP增量为0.015088、0.017949、0.010776。总体分类排序及NLL改善得到本配方下的独立seed支持；τ改善未被区间确认。此处与stage133的1k结论按各自标签预算分别解释。

Frozen MAE3与random encoder均拟合同样StandardScaler+固定C=1线性readout，各budget三独立seeds，full/global分别评价：

| Labeled events | Full Δmacro AP [CI95] | Global-only Δmacro AP [CI95] | Global-only Δτ AP [CI95] |
| --- | ---: | ---: | ---: |
| 1,000 | +0.013282 [0.008424,0.018560] | +0.032246 [0.027299,0.037677] | +0.013606 [0.005853,0.020747] |
| 10,000 | +0.033763 [0.027785,0.039310] | +0.056820 [0.050409,0.063224] | +0.027033 [0.017054,0.036387] |
| 100,000 | +0.016530 [0.012704,0.021452] | +0.057396 [0.051217,0.064131] | +0.030666 [0.020712,0.040447] |

这是冻结表示的可读性比较，不是global单分支重训ablation。结合stage129完整路径置换，支持MAE确实向global表示写入有用事件信息，并在已训练分类器中利用；仍不能把分支收益相加或归为物理因果重要性。低标签线性readout的NLL仍有过拟合，未按AP单独宣布概率可用。

MAE5−MAE3在10k seed42的Δmacro AP=+0.001690，CI95=[−0.001264,0.004520]；ΔNLL=−0.001339，[−0.004543,0.001466]，未确认单纯延长MAE的收益。Channel Rel−MAE5在10k的macro AP为+0.003129，[0.000508,0.005643]，NLL为−0.008224，[−0.011078,−0.005285]；τ AP区间跨零。1k相同对照macro AP区间跨零，τ AP反而下降0.009167，[−0.013986,−0.004748]。这些为seed42比较，Rel还使用额外模拟channel语义，不称为纯1k总监督的自监督优势。

### 执行与资源

23:10:18 UTC累计15.03812943/60 device-hours，CPU allocation192.80465423 core-hours。Device时长分列：Kubeflow47GB MIG 5.87872304 h、Condor24GB MIG 6.20377092 h、Condor12GB MIG 2.88369102 h、先前不兼容V100失败0.07194444 h；没有把不同MIG换算为等效FLOPs。

24GB候选2204028仍排队；新12GB候选1578171（bigbird20，e87a2a0）以1并发竞争相同未领取任务。已有冻结worker继续运行。Kubeflow重新登录后Job API正常，metrics API因RBAC返回Forbidden；这不影响训练。GPU利用率未测量。

只读检查1577948的实际训练cgroup：current14,218,698,752bytes、peak18,758,344,704bytes，父allocation有限上限37,748,736,000bytes。先前Condor `host_memory`只查挂载根而未发现有限上限，本次改为从`/proc/self/cgroup`寻找最近有限ancestor；Pod namespace仍回退到挂载根。5项验收/内存单元测试通过。改动只供后续冻结worker使用，未替换运行源码。

23:07清理AFS已终态10文件、9,374bytes，逐项核对EOS保存原文相同，审计位于`audit/afs_cleanup_20261113T231000.json`。

## 阶段性结论

MAE3在1k/10k总体排序的收益已跨独立预训练seeds复现，global表示具有可读信息。τ的收益依赖标签预算与指标；Rel在1k出现退化证据，需要保留MAE对照，不能整体宣称优于MAE。

## 下一步计划

完成100k独立MAE、EMA、local Rel及MAE5全部矩阵。统一strict readback、真实τ工作点计数和终态成本后结束S4；扩量决策须考虑高标签收益与低标签概率问题。
