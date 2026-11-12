# 阶段54：显式跨层消息传递的完整控制

## 阶段目标

在同一原生pixel+bar、共享局部压缩和四轮更新预算下，量化global blocks中的跨平面消息增量，补齐S2最后两项预定配对。

## 工作内容

ParT将global attention限制在同z平面；ParticleNet将所有global graph邻居限制在同z平面。局部learned compression、全局计数/信号和、各探测器late pooling/head仍保留。这个控制检验显式跨平面消息，不等于去除全部跨层信息；所有事件和全局块均处理，没有按占用截断。

80k train/10k validation，每类20k/2,500；seed42、batch4、四轮，每轮20,016 attempted steps。H100 NVL/LCG108/Torch2.7.1/BF16；按full-validation NLL选checkpoint。accepted samples/split/proxy保持冻结。两项新增训练均四轮、best epoch4、80,064 applied updates、0 AMP skips、14项CPU读回全pass。

## 详细数据

差值均为cross-plane enabled减plane-only。400次accepted-family/chunk bootstrap、128 blocks，每类32。固定粗E/vertex支持8,559 events、Neff=7,959.41365；未匹配target Z/同位素。以下均为条件于已拟合模型的pointwise事件区间。

| Family | Raw tau AP difference [95% CI] | Fixed E/vertex difference [95% CI] | Efficiency difference at 1% FPR [95% CI] |
| --- | --- | --- | --- |
| ParT | +0.00803127 [-0.00049395,+0.01671763] | +0.00574564 [-0.00196088,+0.01402978] | +0.00640000 [-0.00282937,+0.01270446] |
| ParticleNet | +0.00574514 [-0.00169270,+0.01314704] | +0.00521402 [-0.00106270,+0.01159757] | +0.00120000 [-0.00730016,+0.00636519] |

| Model | Tau AP | BA | Macro F1 | NLL | Tau at 1% FPR | Allocation [s] | Program wall [s] | Peak [MiB] |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| part_native_joint_s42_e4_v2 | 0.35858391 | 0.4184 | 0.41844267 | 1.22107879 | 93/2500; 75/7500 background | 6993 | 6903.39536 | 1589.18994 |
| part_native_planeonly_s42_e4_v2 | 0.35055264 | 0.3928 | 0.39216781 | 1.25746343 | 77/2500; 75/7500 background | 6103 | 6095.66839 | 1599.60938 |
| pn_native_joint_s42_e4_v2 | 0.35510474 | 0.3974 | 0.39204149 | 1.23930563 | 83/2500; 75/7500 background | 6310 | 6293.25696 | 1593.10254 |
| pn_native_planeonly_s42_e4_v2 | 0.34935959 | 0.3897 | 0.38574807 | 1.25382265 | 80/2500; 75/7500 background | 4750 | 4743.34227 | 1592.86426 |

Plane-only两项的逐轮exposures均为80,040/80,044/80,040/80,040，总320,164。ParT对应Condor `1573782.31`，ParticleNet对应`1573782.32`，均JobStatus4/exit0/starts1。Allocation为6,103/4,750 s；实际program wall为6,095.66839/4,743.34227 s。没有用program时间替代真实分配。

| Model | e AP | mu AP | tau AP | NC AP |
| --- | ---: | ---: | ---: | ---: |
| part_native_joint_s42_e4_v2 | 0.60478117 | 0.46065084 | 0.35858391 | 0.42513003 |
| part_native_planeonly_s42_e4_v2 | 0.58867953 | 0.37024650 | 0.35055264 | 0.35493613 |
| pn_native_joint_s42_e4_v2 | 0.59882218 | 0.41459288 | 0.35510474 | 0.36957463 |
| pn_native_planeonly_s42_e4_v2 | 0.58534927 | 0.37957230 | 0.34935959 | 0.35279583 |

四分类指标和非τ AP为点估计，相关差值CI未测量。1%与0.1%工作点分别在validation经验曲线上定义；0.1%最多7个背景，低于25个尾部事件标准，不作稳定物理选择结论。

两项raw/common AP区间均跨0且上界超过+0.01，因此既未确认正收益，也不能排除预定+0.01收益。次要τ ROC AUC差的区间均为正：ParT +0.00763389 [0.00104624,0.01421328]，PN +0.00550640 [0.00021440,0.01092955]；这是pointwise次要指标，不能替代主AP结论。

同allocation整轮比较中，ParT在plane-only/main epoch4/3时ΔAP=+0.00529714；PN在epoch4/2时为−0.02164294，均为点估计、CI未测量。PN主模型在这个预算内只到epoch2，说明四轮预算排序与有限allocation排序可不同。完整budget/unused seconds和额外分配在主配对JSON中保留。

### 条件域与τ分支

以下是在既有中心/large flags上的posthoc配对，仍使用full-NLL所选checkpoint；每域各自定义经验背景阈值，未重配E/vertex或重新选checkpoint。中心885事件（四类245/237/176/227），large 4,494事件（1,131/1,183/1,053/1,127），两域可重叠。

| Family | Domain | Tau AP difference [95% CI] | Efficiency difference at domain 1% FPR [95% CI] |
| --- | --- | --- | --- |
| ParT | pixel_centered | +0.02560435 [-0.00780216,+0.06047087] | +0.02272727 [-0.02330733,+0.06487368] |
| ParT | large_scintillator | +0.01256755 [-0.00152050,+0.02643332] | -0.00759734 [-0.02022778,+0.01666600] |
| ParticleNet | pixel_centered | -0.01391863 [-0.05123121,+0.02468710] | -0.05113636 [-0.08678476,+0.00607531] |
| ParticleNet | large_scintillator | +0.00662967 [-0.00466764,+0.02181013] | +0.00854701 [-0.00293328,+0.02312996] |

这四个域的AP区间均跨0；不能把inclusive结果外推为每个区域均有收益。分支AP使用本分支τ与同域全部非τ背景，其prior和分母不同于inclusive。

| Model | Tau branch | Signal events | Branch AP | Efficiency at inclusive 1% FPR |
| --- | --- | ---: | ---: | ---: |
| part_native_joint_s42_e4_v2 | electron | 445 | 0.08132916 | 0.01573034 |
| part_native_joint_s42_e4_v2 | muon | 395 | 0.06071660 | 0.01012658 |
| part_native_joint_s42_e4_v2 | hadronic | 1660 | 0.29800798 | 0.04939759 |
| part_native_planeonly_s42_e4_v2 | electron | 445 | 0.07899842 | 0.01573034 |
| part_native_planeonly_s42_e4_v2 | muon | 395 | 0.05819474 | 0.00759494 |
| part_native_planeonly_s42_e4_v2 | hadronic | 1660 | 0.28931885 | 0.04036145 |
| pn_native_joint_s42_e4_v2 | electron | 445 | 0.07676873 | 0.01573034 |
| pn_native_joint_s42_e4_v2 | muon | 395 | 0.05989884 | 0.01012658 |
| pn_native_joint_s42_e4_v2 | hadronic | 1660 | 0.29638621 | 0.04337349 |
| pn_native_planeonly_s42_e4_v2 | electron | 445 | 0.07681458 | 0.01797753 |
| pn_native_planeonly_s42_e4_v2 | muon | 395 | 0.06290275 | 0.01518987 |
| pn_native_planeonly_s42_e4_v2 | hadronic | 1660 | 0.28645434 | 0.03975904 |

### 验证与产物

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100`。`cross_plane_comparison.json`合并四模型metrics/manifest/history/readback、两项主配对、条件域和分支；终态见`part_planeonly_terminal.json`、`pn_planeonly_terminal.json`，读回见`readback_*_planeonly.log`。

`cross_plane_controls/cross_plane_effects.png`和两个`cross_plane_*_conditional.png`均已实际查看。新增plane-only分类PDF页号如下，已导出PNG并实际查看混淆矩阵、ROC、score及学习曲线：

- `part_native_planeonly_s42_e4_v2`：第16页；`part_native_planeonly_s42_e4_v2_diagnostic_review.png`。
- `pn_native_planeonly_s42_e4_v2`：第35页；`pn_native_planeonly_s42_e4_v2_diagnostic_review.png`。

严格accounting/evaluate/target/inventory最终流程均真实exit0；不含`--allow-incomplete`，inventory使用`--require-complete`。最终39结果、41单seed配对、4三seed均值、0 pending；35科学配置全部14项读回pass。完整S2质量和验收决定在下一阶段总报告统一给出。评价代码commit为`d7fed400d327158e8871332613f99b53bc6a31fe`；本阶段未修改训练/评价代码。

```bash
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign roadmap/configs/s2_h100_v2.json \
  --experiments pn_native_planeonly_s42_e4_v2
python -m analysis.flavour.s2.conditional_pairs --root "$S2_RESULTS/h100" \
  --a part_native_planeonly_s42_e4_v2 --b part_native_joint_s42_e4_v2 \
  --output "$S2_RESULTS/h100/cross_plane_part_conditional.json"
# ParticleNet conditional comparison uses the corresponding pn names.
```

分类图显示electron ROC较强，mu/tau/NC score重叠；ParT的τ AP在epoch3略高于epoch4，但NLL选epoch4，未按AP重选。PN学习曲线非单调，最小NLL仍在epoch4。`cross_plane_controls/review.json`保存五张图的实际审阅记录。

## 阶段性结论

ParT的显式跨平面消息：inclusive τ AP区间跨0，未确认正收益；固定E/vertex支持区间跨0，未确认正收益。四分类和域内结果须与主指标一起评价。

ParticleNet的显式跨平面消息：inclusive τ AP区间跨0，未确认正收益；固定E/vertex支持区间跨0，未确认正收益。四分类和域内结果须与主指标一起评价。

物理解释仍为假设：跨层关联可能帮助连接shower/track拓扑；当前控制没有重建或直接拟合track、kink、secondary vertex，也没有隔离其中哪个物理机制起作用。

## 下一步计划

综合全部35科学配置、四组seed确认、信息/模态/训练域控制和终态成本，写S2总报告及gate decision；只完成S2，S3–S6和calibration/test继续冻结。
