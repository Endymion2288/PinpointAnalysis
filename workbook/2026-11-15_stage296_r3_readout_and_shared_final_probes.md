# R3：固定五折输出诊断与训练后共享输入probe

## 阶段目标

检验C1/M0的逐类取舍能否由简单全局输出偏移恢复，并减少后续原生probe重复构图的成本。C1经固定class-bias修正后仍没有取得2pp实用BA收益，e recall取舍仍存在；不改变原raw模型负决定。

## 工作内容

新增`r3.readout`，直接复用R2已冻结的family-connected五折、temperature/bias目标、边界及正则化，不扫描超参数。四个已完成endpoint分别在四折拟合、第五折评价，结果仅作重复开发Tune诊断。每个raw条件三类指标先与正式manifest独立核对，全部一致。

扩展现有group入口：原生训练组完成后，可在同一allocation内使用同一份只读Train/Tune mmap做完整特征probe与checkpoint概率重算。probe使用共同Train4k子集的native index映射读取full缓存，保留完整Tune；共享缓存已经存在时不再次tokenize/build candidates。该post-fit路径只在显式启用时执行，训练本身的配置、曝光和checkpoint不变。共享数组的确定性与子集映射已由先前真实runner恢复验证。

同步总入口、模型设计和当前决策文档中的C1状态，保留原历史阶段报告；验收另明确检查M0/geometry/C1核心矩阵分别为4/12/4视图、8/48/8配对，防止以缺项但内部自洽的合同替代完整交付。

## 详细数据

输出根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/`。CPU任务1583020.0完成，合同及结果为`readout/early_c1/{contract,results}.json`。

| 权重 | Raw BA3 | Positive-temperature BA3 | Bias BA3 | Raw NLL3 | Temperature NLL3 | Bias NLL3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 56.6700% | 56.6700% | 56.7367% | .861747 | .857632 | .861356 |
| R2 D0 | 56.5333% | 56.5333% | 56.4367% | .864586 | .860534 | .864135 |
| M0 | 57.5100% | 57.5100% | 57.2633% | .845725 | .842876 | .845586 |
| R2 D2 | 57.1267% | 57.1267% | 57.0267% | .849030 | .845444 | .847592 |

五折各自只拟合两个global class biases，NC固定0；temperature为单个正scalar，argmax逐行完全相同。bias会改变多类归一化及排序，不表示新增了detector信息。

| 权重/域 | Raw e/μ/NC recall | Bias e/μ/NC recall |
| --- | --- | --- |
| C1 all | .6101/.5010/.5890 | .6540/.4981/.5500 |
| D0 all | .6443/.4648/.5869 | .6855/.4849/.5227 |
| C1 pixel occupancy0–100 | .307304/.154945/.687708 | .419051/.126740/.611628 |
| D0 pixel occupancy0–100 | .406597/.094252/.674419 | .528105/.108533/.532890 |

同样bias处理后的C1−D0全域BA3点差+.3000pp，e recall仍−3.1500pp；低occupancy e recall仍−10.9054pp。拟合readout的bootstrap区间未测量，没有用其替代原始paired合同或晋级标准。

资源方面，尝试在G1仍占MIG配额时对whole-H100做server Pod dry-run，Kyverno的`namespace-gpu-quota/check-total-gpu-units`明确拒绝超过1个GPU unit。未创建Pod、未取得allocation。证据为`audit/handoff_whole_quota_c2.json`；常规ResourceQuota显示0个整卡请求，并不包含这一跨MIG/整卡策略，因此不能据此再启动第二个GPU Job。继续等待有效G1自然完成，然后提交后备资源候选。

验证：43项R1/R2/R3 tests通过，`r3.handoff --help`正常；新CLI复用已有受测试的fold/fit/crossfit数学实现，实际四模型全部raw核对及五折拟合完成。post-fit共享probe将在取得的实际native allocation内执行并记录，当前尚未测量该完整交付操作。没有读取Cal-A/Cut-B/Confirm。

## 阶段性结论

事实：正temperature只改善本次NLL，保持argmax；两个global biases不足以恢复C1相对D0的e recall或取得实用BA收益。已观察到的取舍不能简单归为一个可轻易修复的全局偏置。

假设：更丰富readout、输入几何或表示可能影响质量，但本诊断没有证明哪个机制主导。C1当前仍仅保留成本研究候选，不自动加深readout、扩大曝光或放宽质量门槛。原生训练后的缓存复用预计减少重复准备，实际耗时待测量。

## 下一步计划

完成G1 endpoint及匹配评价；配额释放后测后备claim吞吐，计入native重建缓存、validation、失去的未保存工作及余量，决定是否迁移。完整G2评价/健康/probes完成后，再形成最终条件扩展决定及R3终态验收。
