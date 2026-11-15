# R1：三 seed 稀疏事件取舍的原始计数读回

## 阶段目标

验证seed42发现的低occupancy μ损失是否在三个seed重现，区分总体均值、分层差值及各层对总体的贡献。只读取已完成R1模型的Tune预测，不拟合新模型，不改变排名或晋级规则。

## 工作内容

新增`analysis.flavour.v2.r1.occupancy_seed_review`，对B1/B4的T3 main、T4 main与T3 matched-compute共18个模型逐一读取预测。T4只在e/μ/NC概率中argmax，与条件三类输出一致。按原accepted的pixel+bar+tracker channel数分成0–99、100–999、1,000–4,999、5,000+；empty另作重叠诊断。

独立重算6个域的原始类人数和3×3 confusion，与stage263已通过1,431项读回的评价结果逐项核对，并检验四个occupancy域的support、correct数与配对correct差之和均恢复全域。导出[紧凑结果](../roadmap/v2/r1_occupancy_seeds.json)，生成英文两面板图。

## 详细数据

低occupancy域的e/μ/NC人数为2,971/2,801/3,010；三个seed顺序为17/42/2026。

| Contrast | Mu recall differences below100 channels, by seed (pp) | Mean difference (pp) | Contribution to all-muon recall (pp) |
| --- | --- | ---: | ---: |
| B1 T3 main minus B4 T3 main | −11.4602 / −24.2413 / −12.0314 | −15.91098 | −4.45667 |
| B1 T4 conditional minus B4 T4 conditional | −8.4613 / −16.8154 / −6.3549 | −10.54385 | −2.95333 |
| B1 compute T3 minus B4 compute T3 | −4.4984 / +13.4595 / −8.2114 | +.24991 | +.07000 |
| B1 compute T3 minus B1 main T3 | +15.8515 / +31.3816 / +2.2135 | +16.48221 | +4.61667 |

第一行低occupancy的μ正确数差为−321/−679/−337；T4对应−237/−471/−178。T3的全域μ差为+1.37/−6.91/−1.69 pp；即使seed17全域为正，低occupancy仍为负，不能用全域均值掩盖条件失败。

| Original channel count | Mu events | B1 minus B4 T3 main mean mu recall (pp) | Contribution to all-muon recall (pp) |
| --- | ---: | ---: | ---: |
| 0–99 | 2,801 | −15.91098 | −4.45667 |
| 100–999 | 3,653 | −.19162 | −.07000 |
| 1,000–4,999 | 3,336 | +5.86531 | +1.95667 |
| 5,000+ | 210 | +7.61905 | +.16000 |
| All | 10,000 | −2.41000 | −2.41000 |

四项贡献准确求和到全域−2.410 pp。低occupancy组占全部μ的28.01%，贡献的损失大于全域净损失，由较高occupancy部分抵消。241个全空μ事件的正确差在全部seed及这四种对照均为0。

B1同结构compute−main T3在低occupancy的e/μ/NC平均recall差为**−10.18737/+16.48221/−8.32780 pp**；正确数差分别为e −153/−703/−52，μ +444/+879/+62，NC −367/−367/−18。该组在三个seed均表现为μ恢复及e/NC损失，支持继续审查训练/决策取舍。原始投影与模型结构相同，但预算和wall-based schedule一起改变，不单独归因于曝光。

B1/B4两个compute模型分别使用本族T4的实际wall预算，其跨族差值**不是共同计算预算对照**；这里用来描述实际结果及seed变化。B1与B4 compute的低occupancy均值接近零，但seed差分别为−4.50/+13.46/−8.21 pp，不能宣布低occupancy问题稳定修复。

以上均为post-hoc开发点估计与实际seed sample SD，没有新增事件区间，没有训练seed总体推断。所有模型使用同一accepted分组；没有按输出或扰动后计数重新定义域。

## 验证命令与产物

```bash
python -m analysis.flavour.v2.r1.occupancy_seed_review --root "$R1_ROOT"
condor_history -name bigbird20.cern.ch 1582932 -limit 1 \
  -af ClusterId JobStatus ExitCode RemoteWallClockTime RequestCpus
git diff --check
```

本阶段新增入口为`occupancy_seed_review`；旧`occupancy_review`的单seed配对区间功能完整保留。原始读回通过HTCondor1582932的同一三seed实现运行，Exit0、allocation18 s、1 CPU。18模型、4配对组共**1,980项原始计数/求和检查通过**；紧凑结果另有156项独立算术核对通过。完成后仅补充跨族compute预算文字并重新绘图，未重复预测读回或训练。

数值产物为`R1/audit/occupancy_seed_review.json`，英文图为`R1/figures/r1_occupancy_seeds.{pdf,png}`，已目视核对；代码语法、JSON解析及修改链接通过。`R1`为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。Cal-A/Cut-B/Confirm payload继续封存。

## 阶段性结论

**事实：**共同曝光下，B1相对B4的低occupancy μ损失在三个seed重现；同结构增加计算及改变schedule后，三个seed均恢复部分μ，但同时损失e/NC。分组的原始人数和correct差准确还原全域结果。

**待检验：**训练长度、学习率轨迹、类间决策和投影压缩分别造成多少差异。当前证据不支持只归因于几何信息丢失，也不支持用compute均值宣称逐类缺陷已修复。

## 下一步计划

继续完成R1剩余B2两seed与B3 compute、完整41模型/133配对及最终验收。后续R2保留逐类/occupancy诊断及明确schedule对照；R3再做stride/采样相位/稀疏局部残差单因素研究。
