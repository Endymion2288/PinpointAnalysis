# R3：紧凑 sensor-row 正式结果、配对质量与同设备成本

## 阶段目标

回答111×420紧凑图像是否在保留分类质量的同时节省计算。C1已完成100k×4正式训练、完整Tune配对和checkpoint重算；当前配方不晋级，保留成本研究价值。G1/G2仍在训练，本阶段不代表R3完成。

## 工作内容

使用预先冻结的`contracts/c1_evaluation.json`，比较C1与R2 D0的固定终点和各自best-screen。二者共同Train100k、4passes、50051updates及LR轨迹，主结果为固定终点。评价覆盖11域，8项预定配对，每项400次物理块重采样；点值和核心区间的独立实现均通过。Tune继续作为重复开发集，Cal-A/Cut-B/Confirm payload未打开。

在同一H100L 1g.12gb allocation内顺序执行旧物理网格和C1的1024updates，固定Train10k、batch8、BF16、mmap、workers及CPU线程。另用saved exposure权重和新准备的输入重新计算全部40k Tune概率，随local/regional/global/tracker特征提取一起核对。

新增`r3.scientific_review`，直接应用v2原有的2pp三类或1pp τ实用门槛及质量约束；对recall差值使用95%下界检验最大退化，避免正BA抵消单类损失。补充一个能够检出该错误的科学判定测试。图中增加逐类约束和同设备成本。

## 详细数据

所有本阶段运行产物位于：

`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/`

正式任务1583011.0已结束；配对评价1583013.0、probe1583014.0及同设备成本1583015.0均完成。主配对数据如下，recall/BA/效率差值单位为百分点，NLL为无量纲；区间为开发Tune的配对物理块95%区间。

| 指标 | C1 | R2 D0 | C1−D0及95%区间 |
| --- | ---: | ---: | --- |
| 共同三类BA | 56.6700% | 56.5333% | +0.1367 [−0.2134,+0.5036] pp |
| 共同NLL3 | 0.861746946 | 0.864586107 | −0.002839 [−0.006386,+0.000427] |
| e recall | 61.01% | 64.43% | −3.4200 [−4.0318,−2.8025] pp |
| μ recall | 50.10% | 46.48% | +3.6200 [+2.9590,+4.2998] pp |
| NC recall | 58.90% | 58.69% | +0.2100 [−0.4757,+0.8013] pp |
| τ效率，1% FPR | 4.84% | 5.11% | −0.2700 [−0.7416,+0.1933] pp |
| NLL4 | 1.185645760 | 1.185567841 | +0.000078 [−0.002667,+0.002673] |

在pixel occupancy 0–100域，BA3差值−0.8437 [−1.4652,−0.1539] pp，e recall为−9.9293 [−11.2686,−8.6500] pp。全域macro AUROC3差值+0.2155 [+0.0305,+0.4055] pp，但排序小幅变化不能替代argmax逐类质量。C1和D0的best-screen恰为endpoint，因此两种视图结果相同，并非两次独立证据。

正式C1每类曝光100000，总曝光400000，LR总和7.865059679070928；实际训练1200.9974 s、validation62.9726 s、data wait38.8988 s，训练吞吐333.0565 events/s，operation wall1272.8914 s。该operation不包括group输入staging/smoke。正式GPU allocated峰382.27 MiB。不能拿它与历史D0共享较大分区时的单worker速度直接计算表示收益。

同设备短任务成本实测：

| 指标 | Physical1322×420/stride4 | Sensor111×420/stride2 |
| --- | ---: | ---: |
| Training events/s | 119.391986 | 325.103994 |
| 含准备/进程启动的operation events/s | 72.372119 | 126.427075 |
| Training seconds | 68.614320 | 25.198091 |
| 完整phase wall seconds | 113.192761 | 64.796247 |
| GPU allocated峰MiB | 1084.955566 | 373.676758 |
| Train10k mmap准备seconds | 27.343308 | 28.498696 |

两者均曝光8192，逐类[2046,2030,2068,2048]，1024updates；训练吞吐比2.722997，operation吞吐比1.746903，显存比0.344417。独立重启的速度波动及相同formal训练的总成本比未测量。

全部40k Tune重算概率与已保存分数最大/平均差均0、argmax agreement1.0。正式训练52个健康窗口中，8个实际分支每个窗口均有正梯度及代表参数更新，激活finite。Train4k拟合scaler+logistic C1的四个独立readout在Tune上的BA3为local47.6567%、regional48.9700%、global49.7267%、tracker41.7200%；它们衡量冻结特征可读性，不能解释为删除分支后的效用。

证据为`evaluation_c1/{results,independent_points,independent_core_intervals}.json`、`probes/sensor_rows_c1/results.json`、`health/sensor_rows_c1/results.json`、`audit/profile_c1_cost_c1/results.json`、`audit/compact_matched_cost.json`及`audit/c1_decision.json`。英文endpoint/contrast/class-guard/cost图已生成PDF/PNG并目视核对。

验证命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_v2_r3 tests.test_flavour_v2_r2 tests.test_flavour_v2_r1 -q
python -m analysis.flavour.v2.r3.scientific_review \
  --evaluations /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/evaluation_m0/results.json \
                /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/evaluation_c1/results.json \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/audit/scientific_review_m0_c1.json
```

41项tests通过。此时G2三个allocation均已完成Train100k/Tune40k mmap准备并进入正式runner，第一次rolling日志已记录实际optimizer更新；完整正式模型健康与泛化尚未测量。

## 阶段性结论

事实：C1确实节省本次同设备计算，训练有效，完整权重/输入/概率核对一致；BA点增量小且区间跨零，e recall损失超过预定1pp限制，τ工作点没有改善。因此不采用当前配方，不触发其三seed或粒度扩展。

假设：紧凑图像的成本优势可用于后续质量恢复实验，但尚无证据说明增大数据、加宽或增加曝光能够恢复e recall。本轮不据此自动扩大矩阵。G1的stride单因素与C1的组合表示仍分别解释；三类/四类选择不由此单一负结果改变。

## 下一步计划

完成G1和G2的冻结曝光；运行已冻结的geometry全域配对，审查逐类、稀疏排序、τ和成本。对G1/native正式权重完成独立可读probe、checkpoint重算和实际健康汇总，利用重训global ablation判断效用。最后依据全部证据决定matched seeds、粒度、容量/打乱边机制控制、anchors和fusion的投入，再交付终态资源与R3验收。
