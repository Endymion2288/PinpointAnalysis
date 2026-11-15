# R1：训练采样的类别混合审查

## 阶段目标

审查当前two-shard窗口采样是否造成大段按类别训练，避免在解释μ recall和训练波动时把未经核对的采样问题当成原因。仅重放metadata顺序，不增加训练或读取探测器响应。

## 工作内容

`SpatialBatchSampler`在两个shard内混洗；`r1.prepare.assign_shards`已显式按类别负载分配完整ROOT chunk。读取冻结100k Train的provenance及Parquet footer，用真实sampler重放T3/T4、seeds17/42/2026各四pass，共24种顺序。记录窗口类频率、batch组成、相邻batch类别比例相关性和末窗口，并用相同batch-size序列的全池随机排列作metadata控制。

完整窗口计数在`R1/audit/sampler_class_mix_stage271.json`；[紧凑结果](../roadmap/v2/r1_sampling_review.json)保留24种顺序及控制统计。没有读取observations payload、拟合模型或访问Cal-A/Cut-B/Confirm。

## 详细数据

40个Train shards中，T4有36个包含四类、4个包含三类；排除τ后，T3有36个含三类、4个含两类。逐case曝光总数均与其75k/100k池大小相符。

| Metadata statistic over 12 cases per task | T3 | T4 |
| --- | ---: | ---: |
| Single-class batch fraction | .0959%–.2983% | .0320%–.0799% |
| Events in windows missing at least one target class | 0–4,680 / 75,000 | 0–7,923 / 100,000 |
| Maximum fraction of events in such windows | 6.240% | 7.923% |
| Lag-1 correlation of per-batch class fraction, range across classes/cases | .01598–.22413 | .01919–.16540 |

相同batch-size序列的全池随机排列控制中，24种顺序的最大绝对lag-1相关为.03096。此控制只比较标签序列，不代表已实现的I/O方案、吞吐或分类结果。

最后一轮的末窗口中，各类频率如下；三个seed均含全部目标类别，不能用“最后窗口没有μ”解释现有argmax损失。

| Seed | Last T3 window e / mu / NC (%) | Last T4 window e / mu / tau / NC (%) |
| --- | --- | --- |
| 17 | 28.10 / 34.36 / 37.54 | 22.24 / 27.19 / 20.86 / 29.71 |
| 42 | 31.78 / 34.62 / 33.60 | 24.86 / 27.08 / 21.79 / 26.28 |
| 2026 | 29.65 / 35.32 / 35.03 | 23.30 / 27.75 / 21.43 / 27.53 |

全24种顺序的单类别batch都少于.3%，不支持“R1基本按类别分段训练”的假设。少数窗口缺类、局部类比例有正自相关是实测事实；它们对最终概率偏移、μ recall和泛化的影响**未测量**，不是已确认系统性故障。

## 验证与产物

metadata重放使用LCG108、单线程BLAS、真实`R1Dataset`初始化与`SpatialBatchSampler`；没有调用dataset的`__getitem__`。全池排列固定控制种子`seed+100000+pass`，保留实际每batch大小。完整审计包含窗口原始人数，可重新从人数计算频率、缺类事件质量和总updates；本阶段没有评价或训练代码改动。

`R1_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。此时两个B2 T3仍实际训练中，完成数量37/41；最终41模型评价及accept未执行。`git diff --check`通过。

## 阶段性结论

**事实：**现有shard准备已实现有效混类，大段单类别训练假设不成立；two-shard窗口仍引入一定类别比例相关性。**待检验：**这种相关性是否实质影响优化或逐类决策，不能据此归因μ问题或重写R1排名。

## 下一步计划

R2提交前除U、batch-size和LR轨迹外，再保存窗口/滚动类频率及末段组成；在固定Train-inner或Tune screen检查head/logit偏移是否随窗口先验系统摆动。只有出现对应模型证据时，才增加一个相同事件、曝光、batch-size序列、updates和LR轨迹的重混顺序控制，并先profile其缓存/I/O成本。此诊断不自动扩大R2矩阵，不改变正在执行的R1训练。
