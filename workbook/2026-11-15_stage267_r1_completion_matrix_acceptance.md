# R1：最终验收覆盖冻结完整矩阵

## 阶段目标

在剩余训练完成前，确保最终验收检查完整R1合同，而不是仅检查评价文件恰好列出的子集。原41模型、133配对、全部域及三seed要求保持不变。

## 工作内容

`r1.deliver.evaluation_config`增加只返回计划而不写文件的模式；原命令默认行为不变。`r1.accept`在最终完整模型检查后重新从冻结排名、B0选择及精度匹配控制生成预期矩阵，增加五项验收：

1. 实际evaluation config与冻结模型/配对枚举一致。
2. 实际evaluation包含全部且仅有预期模型。
3. 配对身份、方向、域与reason完整，各出现一次；不能用重复pair顶替缺项。
4. 每个模型完整交付11个域。
5. 三seed平均配对完整且唯一，各自的seeds精确为17/42/2026。

保留模型学习、身份、曝光、±5%同allocation预算、独立原始读回、55条件路径、图表及终态资源等已有检查。该完整性检查不改变模型排名、研究晋级门槛，也不把“完成实验交付”解释成“模型全面优于基线”。

## 详细数据

从真实R1冻结状态重新枚举得到**41模型、133配对、21项三seed平均配对**；与已有`R1/evaluation_config.json`的模型及配对列表精确一致。枚举前后文件内容完全一致，没有改写冻结计划。

11域为all、pixel_centered、large_scintillator、active、empty、tracker_active、tracker_empty、occupancy_0_100、occupancy_100_1000、occupancy_1000_5000、occupancy_5000_inf。

以已经真实完成并通过1,431项独立读回的`R1/interim/b1_three_seed_compute/evaluation/results.json`作负对照：该文件为24模型、56配对，新的矩阵完整性检查正确拒绝其中四项结果覆盖要求，冻结config一致性一项通过。这里是预期的子集拒绝回归，不是最终R1验收失败；最终accept尚未运行。

新增三个回归测试，分别覆盖“内部一致但缺少一个seed的子集”“pair总数相同但一个重复且另一个缺失”“域缺项和三seed名单重复”。连同既有R1及summary测试，**26项全部通过**，本次实际测试时间.783 s。实际数据验证记录为`R1/audit/completion_contract_stage267.json`。

## 验证命令与产物

```bash
python -m unittest discover -s tests -p 'test_flavour_v2_r1*.py'
git diff --check
```

实际枚举使用`evaluation_config(root, check_complete=False, write=False)`，只读取已冻结排名和已完成控制的manifest，不启动训练。完整验收将使用默认complete检查及`write=False`，因此尚未完成的模型不能通过。

代码为`src/analysis/flavour/v2/r1/deliver.py`、`accept.py`；测试为`tests/test_flavour_v2_r1_acceptance.py`。`R1`为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。本阶段没有访问Cal-A/Cut-B/Confirm payload，也没有新增GPU作业。

## 阶段性结论

**事实：**当前冻结交付要求可以从实际状态精确重建，内部一致的中期子集不会被误判为完整R1。已有完整26测试通过。

**未完成：**剩余B2/B3模型、完整评价和最终accept；当前检查合同通过不代表这些产物已经完成。科学质量条件仍由最终实际结果逐项判定。

## 下一步计划

继续跟踪三个已运行GPU作业；收齐41模型后，由唯一finish控制器运行400次全矩阵统计、完整独立复核和英文图表。资源全部终态后执行增强后的accept、最终导出与报告。
