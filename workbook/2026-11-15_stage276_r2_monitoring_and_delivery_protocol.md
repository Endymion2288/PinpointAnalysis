# R2 首个开发窗口与交付协议

## 阶段目标

检查首批模型是否在真实Train/Tune上学习，准备完整评价、条件扩展和资源验收。六个模型仍在训练，本阶段不作终点排名或R2完成判断。

## 工作内容

在stage275提交`022f92e`基础上新增`r2.plots/accept/accounting`。图表全英文，按非τ曝光显示NLL、BA3、逐类和低occupancy μ recall，endpoint与best-screen结果分开。交付验收要求六份原合同、逐行曝光/准确U/LR、完整40k Tune身份、真实学习与恢复门槛、各分支健康、endpoint/best checkpoint位置、Train gap、全部配对和独立区间、终态资源及有证据的扩展决策。

`campaign extend`只在参考T3/T4终点完成且有书面科学理由时冻结后续小矩阵。它保留原合同，新增fresh训练配置及reference对照，支持同数据增加passes、B1 width32或B2/B3原生族控制；不会自动扩矩阵。加宽模型的real-fit门槛使用实际width/dim，不能用width16的拟合结果代替width32。新扩展机制尚未用于科学配置。

资源统计复用R1的实际Condor ads/Pod终态，再加入group并发数。一个allocation只计一次device-hours。交互操作以实测train/validation或profile phase wall给出下界，启动及完整allocation时长未测量，单列，不加成精确总成本。

## 详细数据

结果根目录沿用`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/`。六项都使用固定同一4k Tune screen，真e/μ/τ/NC各1k。首个窗口如下；T3仅监督非τ，T4额外包含τ监督。

| Model | Non-tau exposures | NLL3 | BA3 (%) | NLL4 | Low-occupancy mu recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 T3 | 75000 | .933269119 | 51.666667 | — | 7.636364 |
| D0 T4 | 75000 | .930999850 | 53.966667 | 1.248701768 | 14.909091 |
| D1 T3 | 75000 | .923738729 | 54.500000 | — | 5.454545 |
| D1 T4 | 75000 | .930151168 | 53.933333 | 1.247207541 | 11.636364 |
| D2 T3 | 75004 | .926986957 | 53.633333 | — | 27.636364 |
| D2 T4 | 75005 | .917836317 | 53.800000 | 1.244964741 | 46.545455 |

low occupancy定义为accepted pixel+bar+tracker channels<100；同一screen内e/μ/τ/NC计数为285/275/342/309。六个模型都在这275个真μ上评价。此时τ在1% FPR的screen效率分别D0/D1/D2 T4为2.4%/2.3%/2.2%；background总计3000，尾部粗粒度明显，完整40k Tune终点尚未评价。T3表内“—”表示没有四类head，未将τ重标为NC。

T3第二窗口已到达：D0在150000非τ曝光时BA3=55.9333%、NLL3=.8997；D1同曝光BA3=55.8000%、NLL3=.9110；D2在150007曝光时BA3=55.0667%、NLL3=.9103。精确值与各分支内部健康位于`experiments/b1_d*/manifest.json`的history/health。D1/D2此时尚未完成其各自预算；D0与D1的学习率轨迹按总预算不同，不能把中间窗口当作曝光单因素效果。

Kubeflow四并行在首次screen后cgroup memory.current=39,007,813,632bytes，peak=39,267,102,720bytes，limit约63GB；memory.events的max/oom/oom_kill均0。每个进程的两名screen workers常驻后仍在申请范围内，完整Tune loader峰值继续监测。

资源快照UTC `2026-11-15T10:06:53.830499+00:00`：已分配GPU总时长.719392771h，其中H100L MIG2g24 .260390830h、A100-PCIE-40GB .061666667h、H100 NVL MIG3g47 .397335274h；两份科学allocation仍在运行，因此不是终态成本。交互operation下界.139298571h另列，见`resources.json`。

### 评价与图表验证

完整评价smoke使用Train-only的8行人工validation，重复构造六个候选的endpoint/best视图以覆盖pipeline，不是六份真实泛化结果。12视图×11域、72项配对、400 bootstrap均完成，独立point checks1152项和core interval checks1020项全部通过。三套PDF/PNG在`audit/evaluation_integration_a1/figures/`生成，终点对比图已人工查看。真实模型的首批learning curves位于`figures/r2_learning_curves_partial.{pdf,png}`。

7项R2单元测试通过，覆盖精确pass/尾批、完整预算LR、恢复顺序、checkpoint政策隔离、occupancy边界、缺类bootstrap和条件扩展保持原合同。R1/R2相关合并25项测试通过，耗时.859s。正式`r2.accept`须等真实产物齐备后运行，尚未验收。

### 联网论文复核与诊断边界

读取Menon等 **Long-tail Learning via Logit Adjustment**，ICLR2021，arXiv:2007.07314v2；abstract和PDF均HTTP200，PDF1,402,421bytes。抽读§3–4.1，式(7)–(9)将balanced error的Bayes决策与`logit−log(prior)`联系，并说明概率校准前提。重新读取Guo等ICML2017正式页面，HTTP200。原文、摘取文本及UTC记录保存在`audit/literature_stage276/`；文献表增加M7。

当前各训练pool全局类别平衡，因此基于全局frequency的logit adjustment是共同常数，不能改变argmax。正scalar temperature同样不能改变argmax。R1已见的μ排序/argmax分歧和本阶段早期低occupancy差异，支持在终点持续时增加按physical family交叉拟合的低维bias诊断；不支持直接采用long-tail loss或把差异归为物理不可分。交叉拟合仍属于Tune开发诊断，最终校准继续使用封存Cal-A，阈值继续使用Cut-B。

复现入口：

```bash
python -m unittest tests.test_flavour_v2_r2
python -m analysis.flavour.v2.r2.evaluate --root "$R2_ROOT" --replicates 400 --workers 4
python -m analysis.flavour.v2.r2.plots --root "$R2_ROOT" --partial
python -m analysis.flavour.v2.r2.accounting --root "$R2_ROOT"
python -m analysis.flavour.v2.r2.accept --root "$R2_ROOT"
```

后两项正式验收/终态统计需要对应科学任务结束；CPU大批量评价与诊断通过HTCondor，交互只做小型检查和运行监测。

## 阶段性结论

**事实：**六个模型均给出优于无信息NLL的首窗口Tune结果，T3第二窗口继续改善。低occupancy μ recall在中间窗口差异很大。尚无完整预算终点结果，不能由这些点确定最终优劣或具体原因。两份异构MIG allocation继续运行，无OOM记录。

**假设：**学习率/曝光轨迹、局部类别比例、readout偏置或输入压缩可能参与低occupancy逐类取舍。需要终点曲线、共同Train gap和适当配对对照逐项区分。

## 下一步计划

完成初始六模型后计算全部40k Tune配对与Train gap，检查实际学习健康及plateau。根据D1−D0、D2−D1和低occupancy结果，决定是否执行D3、width32或第二原生族；有300k的实际收益与独立成本计划后再考虑1M。任何扩展保留明确理由和准确预算，不自动全矩阵。完成终态记账、独立读回、科学决策和验收后结束R2，R3继续等待。
