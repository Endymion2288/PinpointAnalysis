# R2 恢复元数据与依赖任务自动释放

## 阶段目标

在长训练继续执行时核对恢复与成本口径，并按真实完成状态释放Train gap/评价。科学模型的冻结配方与现有任务不变，R2仍未完成。

## 工作内容

审查发现原恢复代码保留旧manifest的顶层GPU/allocation/staging，且最终wall只记录恢复后的当前attempt。`attempts`虽已保留来源，顶层和累计operation wall仍需修正。现在恢复后刷新实际运行设备与allocation，保存各attempt的checkpointed wall和累计wall，GPU peak取可用历史与当前的最大值。旧checkpoint缺wall字段时仅从已记录train+validation给出下界，并显式标识；完整消耗仍以实际allocation统计为准。

fresh模型在第一步前保存初始恢复点；best checkpoint先更新其选择元数据再保存。它们不改变模型、输入、sampler、预算或优化公式。正在运行的六项仍使用各自提交时源码snapshot，没有热替换训练过程。

新增`r2.finish`本地轻量watcher，每30s只读状态：一对模型实际完成后才提交共同Train gap；全部完成后才提交400次配对评价；评价和独立读回完成后才生成图表。CPU/GPU任务仍通过标准提交器查询schedd并保存不可变源码。它不自动选择扩展，也不代替最终科学判断与验收。watcher日志为`logs/finish_dependencies_a1.log`。

## 详细数据

结果根目录仍为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/`。

`smoke/interactive_resume_meta_a3/`在同一实际T4/FP32上运行9updates的T3连续/中途恢复检查，人工切换allocation标签以检验记账，未测跨GPU参数逐位一致性。连续与恢复的最终parameter最大差0，endpoint/best预测、18行×3passes曝光均一致。恢复后的顶层标签为`smoke-simulated-recovery-label`，首attempt保留`smoke-simulated-initial-label`。

连续operation wall为58.755s；恢复两段累计98.959s，其中当前attempt60.457s、先前checkpointed段38.502s。恢复后的train/validation累计为34.412/49.024s，累计operation wall包含两段。原代码仅报告当前段会小于已经记录的train+validation；本测试明确检验了这一关系。实际Tune与Cal-A/Cut-B/Confirm均未使用。

当best-screen与endpoint为同一update时，训练器复用一份完整Tune预测表；gap推理先核对两份checkpoint参数逐位相同，再复用同一Train预测，避免对同一模型重复执行。`smoke/interactive_reuse_a4/`的T4/FP32、T3、workers0完整流程通过，两种运行的best预测均为endpoint的相对symlink，parameter差0。它属于小型运行检查，未代替正式模型的workers2设置。

8项R2单元测试通过（.044s）。新增依赖测试验证：另一对模型未完成时只释放已完成对的gap；完整模型、评价及独立读回齐备后才释放图表。正式gap与完整评价尚未运行。

六项第二个75k间隔的同一4k Tune screen如下；这些仍是中间窗口：

| Model | Non-tau exposures | NLL3 | BA3 (%) | Low-occupancy mu recall (%) |
| --- | ---: | ---: | ---: | ---: |
| D0 T3 | 150000 | .899748229 | 55.933333 | 48.727273 |
| D0 T4 | 150000 | .890848680 | 55.866667 | 38.181818 |
| D1 T3 | 150000 | .911009995 | 55.800000 | 35.636364 |
| D1 T4 | 150000 | .911876535 | 54.600000 | 32.000000 |
| D2 T3 | 150007 | .910304568 | 55.066667 | 38.909091 |
| D2 T4 | 150001 | .896521341 | 55.766667 | 17.090909 |

各模型已采样的branch activations均finite；已通过的小样本拟合继续作为优化有效性的必要证据。低occupancy真μ分母保持同一275行。与stage276首窗口相比，D0 T3 recall由7.64%升至48.73%，D2 T4则由46.55%降至17.09%，整体NLL同时改善。因此不能把中间窗口某一类recall直接解释为探测器信息或架构的上限。

复现：

```bash
python -m unittest tests.test_flavour_v2_r2
python -m analysis.flavour.v2.r2.smoke --root "$R2_ROOT" --r1 "$R1_ROOT" --r0 "$BASE" --attempt UNIQUE --tasks T3 --simulate-migration
python -m analysis.flavour.v2.r2.finish --root "$R2_ROOT" --watch
```

## 阶段性结论

**事实：**恢复参数与数据顺序门槛通过，新的元数据检查也通过；累计operation和真实allocation成本已分开。六项模型继续学习，低occupancy逐类取舍对训练状态敏感。没有完整预算结果可以据此选定最终模型。

**待查：**低occupancy变化中局部类别比例、readout偏置和输入压缩各自的贡献。需要终点AP/AUROC、Train gap和必要的匹配对照区分。

## 下一步计划

继续监测六项模型，等待完整预算endpoint和best-screen预测。依赖watcher完成gap和独立评价后，依据真实曲线、共同任务和成本作条件扩展决定；保留原始合同与新配置。最终终态记账、英文图表、科学决策和R2验收仍待完成。
