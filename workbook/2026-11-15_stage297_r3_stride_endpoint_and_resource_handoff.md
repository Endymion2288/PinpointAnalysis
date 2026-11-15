# R3：首层 stride 正式终点及后备资源实测

## 阶段目标

回答将 B1 首层 stride4 改为 stride2，能否在相同事件、曝光和优化配方下改善分类。完整 R3 仍须交付三个 G2 模型、全 geometry 评价、全部特征审查及终态验收。

## 工作内容

G1 在 Kubeflow `pinpoint-r3-g1-k1`、H100 NVL MIG 3g.47gb 完成训练及完整 Tune 推理。使用 endpoint 前冻结的 G1−D2 子合同，经 CPU Condor 1583021.0 计算 400 次 physical-block 配对区间及独立 point/core-interval 复核。该子合同不替代最终 geometry 的 12 视图、48 配对。

另外完成实际训练健康审查并绘制英文 endpoint、配对和逐类质量图。核对分组实现时发现 `r3.plots` 将 `occupancy_0_100` 图注误写成 Pixel occupancy；实际定义一直为 accepted pixel+bar+tracker 总通道数 <100，包含全空事件。现已修正图注并重绘 M0/C1/G1；事件选择和全部统计数值不变。

## 详细数据

产物根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/`。

G1 与 R2 D2 均为 seed42、T4、Train300k×4、batch8、BF16 encoder/FP32 head，原 spatial sampler。G1 唯一架构干预为首层 stride4→2，累计 stride32→16；参数同为 793,674。实测 150,117 updates、1,200,000 次曝光、每类 300,000 次，LR sum 23.589806175917317。训练时间 8,456.192748 s，全部 validation 429.201169 s，runner wall 9,008.477520 s；GPU allocation 成本另由全账本统计，不能直接用 runner wall 替代。

| 全域指标 | G1 endpoint | G1−D2 | 95% 配对区间 |
| --- | ---: | ---: | --- |
| BA3 | 57.0933% | −0.0333 pp | [−0.3657,+0.3533] pp |
| NLL3 | 0.850460698 | +0.001430215 | [−0.000761242,+0.003621739] |
| e recall | 59.0700% | −1.4300 pp | [−2.0481,−0.7673] pp |
| μ recall | 56.0900% | +1.7600 pp | [+1.0177,+2.4931] pp |
| NC recall | 56.1200% | −0.4300 pp | [−1.0160,+0.1795] pp |
| τ efficiency at 1% FPR | 5.9700% | −0.0800 pp | [−0.5153,+0.4240] pp |
| NLL4 | 1.167762168 | +0.001528990 | [−0.000467516,+0.003462559] |

低 occupancy 的 BA3 差 +0.1595 pp [−0.6821,+1.0404]；e/μ/NC recall 差分别 −1.8849/+1.4995/+0.8638 pp。pixel-centered 域的 τ 工作点差 −1.5852 pp [−3.6112,−0.1457]。这些分层结果为同 Tune 开发诊断，不能由局部 μ 增益宣称共同质量已恢复。

两项独立统计复核均 `complete`、`failed=[]`。沿用未放宽的 2 pp BA 或 1 pp τ 收益及质量门槛，`audit/scientific_review_g1.json` 没有 eligible primary；当前 stride2 配方不晋级，不为其自动扩 seed 或分辨率。

实际 optimizer 健康有 152 个采样窗口、8 个分支。全部分支观察到正梯度及代表 tensor 更新，激活均 finite；tracker 有 151/152 正梯度窗口，另一个零窗口完整保留，不能把它写成全程正梯度。四类概率 std 的逐窗口最小值为 0.0156943/0.0137810/0.0154879/0.0110356，没有常数输出证据。该检查说明存在有效优化路径，不代表已证明收敛。

关键产物：

- `evaluation_g1/{results,independent_points,independent_core_intervals}.json`。
- `health/g1_c1/{results.json,branches.csv}`。
- `audit/scientific_review_g1.json`。
- `figures/r3_{endpoints,contrasts}_evaluation_g1.{pdf,png}` 和 G1−D2 class-guards 图。

资源方面，whole-H100 后备 `pinpoint-r3-g2-k2` 无可调度节点，主动结束后改试 MIG；保存了 Pending Pod 与 scheduler events。终态 Pod 被平台移除，账本保留 Job lifecycle 的 98 s 分配上界（0.027222 h），没有将它当作已测 GPU 性能或确定零成本。`pinpoint-r3-g2-k3` 的 3g.47gb 随即获得资源；三个真实架构先依次测单进程，再测三并行。三个旧 1g G2 writers 继续优化，没有因申请后备而停止。最终迁移决定及实际速度另存后续记录，当前不预设较大设备必然更快。

复核与绘图命令：

```bash
R3_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3
python -m analysis.flavour.v2.r3.evaluate --root "$R3_ROOT" \
  --contract "$R3_ROOT/contracts/g1_evaluation.json" \
  --output evaluation_g1 --replicates 400 --workers 8
python -m analysis.flavour.v2.r3.health_review --root "$R3_ROOT" \
  --experiments "$R3_ROOT/experiments/b1_g1_stride2_t4_s42" --attempt g1_c1
python -m analysis.flavour.v2.r3.scientific_review \
  --evaluations "$R3_ROOT/evaluation_g1/results.json" \
  --output "$R3_ROOT/audit/scientific_review_g1.json"
```

## 阶段性结论

事实：在本次匹配配方下，减小首层 stride 没有带来整体 BA 或 τ 工作点收益，μ 改善伴随电子 recall 损失；当前配方不采用。这个负结果只针对已测干预，不能推出原生细粒度、局部残差或跨层关系均无用。更细栅格通过不同结构、对象关系或监督目标产生收益的可能性尚待独立检验。

## 下一步计划

完成 G1 全 Tune 权重重算与冻结特征 probe；在后备 claim 上取得三并行实测，用旧 checkpoint 进度、缓存重建、验证、未保存工作和余量估计完成时间后决定是否迁移。三个 G2 正式终点齐备后执行完整 geometry 合同，审查重训 global ablation、真实特征路径和梯度；据此决定 seed、粒度、anchors/fusion/branch competition 等条件实验，最后完成全部终态成本和 R3 验收。
