# S4：完整逐channel Rel预训练读回

## 阶段目标

完成MAE3之后的两epoch逐bar-channel Rel训练，核对学习证据，并准备相同probe上的几何/事件条件诊断。

## 工作内容与详细数据

正式Condor 1574116在24GB H100 MIG上完成`s4_mae_barrel_channels_e2_s42`，源码63aa241。训练2×6144=12,288 optimizer steps，原MAE3 seed42 parent，fresh optimizer/schedule，mask views3/4，固定train-inner probe仍使用view0。任务实际2178.035秒，峰值allocated tensor显存818.994MiB；这是共享设备上的task时长，不作为独立allocation成本重复相加。

CPU读回10项integrity/internal和6项measured-learning全部通过。固定2,000-event probe的三个尺度AP如下；目标仍是定义的near/uniform混合query risk。

| 目标 | Initial AP | Selected checkpoint AP | Positives/queries |
| --- | ---: | ---: | --- |
| 16×16 native pixel patch | 0.45474529 | 0.46561161 | 2257/128000 |
| 4mm pixel occupancy | 0.78090039 | 0.78790980 | 10144/128000 |
| Native bar occupancy | 0.89860147 | 0.90145242 | 42643/128000 |

Rel有效bar channels=144,196，其中mixed=62,073；trained CE=0.64457977，same-probe oracle constant CE=0.71935207，差额−0.07477230，target entropy=0.20368420，fraction RMSE=0.27434459。Global features effective rank=8.46550751。Constant在同probe拟合，仅为描述性先验基线；不据此声称超越几何先验或独立分类增益。

产物根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/experiments/s4_mae_barrel_channels_e2_s42/`，含manifest、best/last、history、reconstruction和`ssl_readback.json`。新增`s4_rel_full_diagnostic.json`将完全复用旧诊断的512个原train-inner events、FP32 CPU和400次connected-chunk bootstrap，先查询固定几何/visible位置后读取targets，保持模拟贡献只进loss。

优化源码25835ec的完整66项候选wave已提交bigbird21 cluster2199507，2并发、native BF16 compact pool。与原Kubeflow、24GB MIG和12GB MIG共享冻结任务claims，优先推进MAE continuation、local Rel、独立MAE seeds的低标签下游。已运行任务继续使用各自归档源码，不重启；真实输入位级等价证据见stage117。CPU v4 cluster2199506正在执行新增完成模型的严格读回和四分类配对统计。

## 验证与阶段性结论

完整channel Rel在原固定probe上有重建及语义学习证据，健康检查通过，允许下游继续。语义CE不是四分类指标；与同probe几何先验的比较、独立分类收益、Rel相对MAE continuation的收益尚未测量。当前仍不支持1M扩量。

## 下一步计划

提交完整channel Rel的CPU诊断并检查真实条件性；完成MAE/Rel/EMA及监督等预算下游、独立seeds、full/global frozen probes。全部必要实验完成后给出严格完整的物理和成本结论。
