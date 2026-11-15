# Stage280：R2重复曝光的过拟合证据与最小正则化对照

## 阶段目标

核对D1延长同一100k pool曝光后的实际效果，根据已经完成的参考启动必要条件扩展。D1 T3/T4完整终点已完成；两者均未超过D0共同任务。T3的共同Train评价确认拟合增强而泛化退化，已冻结并启动一项只改变weight decay的T3对照。D2和扩展仍在执行，本阶段不作完整R2验收。

## 工作内容

`campaign extend`支持选择`--tasks T3/T4`以及`--weight-decay`；只要求实际选择的参考任务完成。初始T3/T4配对与后续扩展独立保留，书面理由必须记录因果比较、综合选择比较和启动次序。本次将预定过拟合分流提前到对应T3终点形成后启动，以并行使用当前资源；完整R2选择等待初始六模型和这个控制全部完成。

`finish`按缺失模型释放Train-gap，并识别已提交的单任务gap，避免两个任务同时写入相同目录。最终评价矩阵将随七个实际模型扩到14个endpoint/best视图、96项配对。原六模型的72项配对全部保留，新增3组×4域×2选点；Cal-A/Cut-B/Confirm保持封存。

## 详细数据

### 已完成的D1参考

相同40k Tune、seed42、BF16 encoder/FP32 head、100k四类pool；T3使用全部75k非τ行。D1是fresh 12 passes，有独立完整warmup/cosine日程，不是从D0低学习率尾部继续。

| Task / checkpoint | Common BA3 (%) | Common NLL3 | Common macro AP | Common macro AUROC |
| --- | ---: | ---: | ---: | ---: |
| D1 T3 endpoint | 53.1700 | 1.143738363 | .613771288 | .738472882 |
| D1 T3 best-screen | 55.3367 | .868712994 | .654632545 | .767338374 |
| D1 T4 endpoint | 53.7300 | 1.015106150 | .628620943 | .748502116 |
| D1 T4 best-screen | 55.6333 | .873765811 | .655708904 | .769022341 |

D0 T3/conditional T4 endpoint的BA3为56.6500%/56.5333%，NLL3=.871321793/.864586107。D1的endpoint退化包含排序指标退化，不能全归于温度失配。D1 own-task best-screen位置分别为T3 46,937 updates、T4 87,589 updates；完整Tune未用于checkpoint选择。D1 T4 endpoint的BA4/NLL4=.4237/1.337161873，best-screen为.440325/1.195163062。完整配对区间仍等待最终批次评价。

D1 T3的同一冻结Train100k诊断子集每类1k、合计3k，在eval mode下测得endpoint BA3=.696333333、NLL3=.626982597、macro AP=.837344608、macro AUROC=.895158667。对应Tune NLL−Train NLL=.516755766；best-screen的Train BA3/NLL3为.567666667/.833462931。训练集诊断支持当前长曝光出现过拟合；这是共同4k Train screen的抽样诊断，不是对全部Train的精确泛化界估计。

### D1WD合同与精确预算

[书面理由](../roadmap/v2/configs/r2/d1wd_reason.md)和[配置](../roadmap/v2/configs/r2/b1_d1wd_t3_s42.json)在首次训练前冻结，名称`b1_d1wd_t3_s42`。唯一模型/优化因素为AdamW weight decay从.01→.1；保留架构width16、同一数据、12 passes、seed42、LR=.0003、batch8及完整update日程。系数为固定一点控制，不是已知最优值。

preflight核对：112,645 updates、900,000实际训练/非τ曝光、每行12次、75,000 unique events、sum(LR)=17.701315577074674，均与D1 T3相同。D1WD−D1为weight-decay单因素；D1WD−D0、D1WD−D2用于后续综合选择，分别改变曝光/配方或unique/配方，不解释为纯正则化效应。

独立理论参考：Loshchilov–Hutter，Decoupled Weight Decay Regularization，[arXiv:1711.05101](https://arxiv.org/abs/1711.05101)。2026-11-15 11:33:25 UTC在线读取摘要HTTP200、43,412bytes；HTML和访问记录在`R2/audit/literature_stage280/`。当前本来就使用AdamW，这项实验研究衰减强度，不把AdamW本身作为新增算法。

### 资源、故障与恢复

资源根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2`。11:43:01 UTC查询142条含GPU的slot ads，存在可用A100、H100/H200及1g MIG，也有当前LCG构建不支持的SM70 V100。ad数量不是独立物理卡数。D1WD申请`Capability>=8`或已知SM90 MIG，显存下限4,000 MiB，没有固定型号或整卡限制；此BF16配方复用已验证数值路径，未静默改成FP32。

| Operation | Handle | Actual result |
| --- | --- | --- |
| D1WD preflight a1 | 无cluster | `bigbird24` EosSubmit策略拒绝，未创建作业 |
| D1WD preflight a2 | 1582976 / bigbird20 | 完成精确sampler与LR预算 |
| D1 T3 common Train gap | 1582977 / bigbird20 | endpoint/best-screen完成 |
| D1WD fresh T3 training | 1582978 / bigbird20 | 已开始，H100L MIG2g.24gb，单进程 |
| D1 T4 common Train gap | 1582979 / bigbird20 | watcher仅提交缺失T4，执行中 |

a1失败原因：原`choose_schedd`只确认`NoEos=True`和WRITE可用，漏掉同时存在的`AllEos/AbsPathInEos`。该schedd要求exec等路径全在EOS，与保存在AFS的提交脚本冲突。修复为额外读取`SUBMIT_REQUIREMENT_NAMES`并排除这类不兼容布局；保留原拒绝记录，不覆盖或隐藏失败。

依赖watcher更新为PID2456785，旧PID2437362终止；`runtime/finish_watcher_a3.json`记录原因。现有训练writers不受影响，初始Kubeflow作业会在D1完整交付后正常结束。D2仍在原Condor allocation中双并行。D1WD配置和代码snapshot独立保存，没有修改其他模型的活动配方。

```bash
python -m analysis.flavour.v2.r2.campaign extend \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2 \
  --design D1WD --reference-design D1 --tasks T3 --weight-decay .1 \
  --reason-file roadmap/v2/configs/r2/d1wd_reason.md
python -m unittest tests.test_flavour_v2_r0 tests.test_flavour_v2_r2 tests.test_flavour_v2_r2_readout
```

与stage279相同的LCG108及PYTHONPATH环境，21项相关测试通过；随后新增的单任务gap去重测试也通过。详细D1参考、preflight及合同入口见[r2_conditional_extension.json](../roadmap/v2/r2_conditional_extension.json)。

## 阶段性结论

事实：在当前固定LR日程与数据上，12 passes不会自动优于4 passes。D1两种任务的完整Tune endpoint均下降；T3共同Train明显改善，支持过拟合解释。最佳screen不能恢复为超过D0的整体BA。

假设：更强weight decay可能延后或减弱100k重复曝光的过拟合，仍需D1WD实际终点判断。D2更多unique能否改善绝对效果尚未测量完成，不据此宣布大数据无效或容量已饱和。

## 下一步计划

完成D2和D1WD，形成全部14视图/96配对、共同Train gap、独立排序与区间读回、五折readout、正式图表和资源终态。先依据D2相对D0/D1及D1WD的证据判断是否有理由扩展unique规模或容量，不自动启动D3/1M/第二族。完整R2决策仍须给出所选参考、收敛和局限以及后续R3输入。
