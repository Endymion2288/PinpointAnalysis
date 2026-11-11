# 阶段 22：S2 粒度与配对效应图

已将最终比较图接入评价程序。现有 B0 的三项真实配对已绘制并查看，10 项评价检查通过；空间模型尚未完成四轮，因此原生粒度和三种子图的实际数据验证仍待训练结果。

## 阶段目标

为 S2 最终报告直接展示粒度、联合位置/投影、配对区间和共同 GPU 时间下的差值，使每张图可以追溯到既有 `summary.json`。训练配置、checkpoint selection 和统计合同沿用已冻结版本。

## 工作内容

新增 `analysis.flavour.s2.plots`，由 `evaluate` 完成统计与常规诊断后调用，也可单独读取已有评价目录。

- `resolution_comparison.png/pdf`：仅绘制完成的 seed42、response-summary v2 空间筛选配置，按 Native/100 μm/200 μm/1 mm 分类轴展示 τ AP、common-support AP、1% FPR efficiency 和 balanced accuracy。原生 pitch 保留为 Native 类别。projection 只画实际点；缺失模型留空。
- `paired_comparisons.pdf`：每页最多十项，明确 A/B 与 B−A 方向，分别显示原样本和固定 common-support 的 95% paired event bootstrap CI。存在实测时间匹配比较时增加第三列，展示 epoch 分辨率的点估计。
- `seed_mean_comparisons.pdf`：读取既有的逐 seed AP 差值均值及共享物理事件 bootstrap 区间，训练 spread 仍单列在统计结果中。
- `comparison_figures.json`：保存来源评价 commit、绘制配置/配对、pending 数量及页码。全部图中文字为英文。

图形程序直接读取已计算的数值。区间绘制为独立线段，允许 bootstrap 区间与原始点估计的位置不同；图中标明 pointwise intervals、common-support 条件区间，以及预定 +0.01 AP 收益线。GPU 时间匹配没有计算过的区间保持未测量状态。

## 详细数据

输入：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/evaluation/summary.json`。该 reference 合同有 4 个完成模型、3 项配对；新图为 1 页 PDF 和首屏 PNG，已实际查看，A/B 标签、区间、零线和 +0.01 线正常。CPU reference 的 GPU 时间匹配未测量，因此此页仅显示前两列。统计数值来自既有阶段 12/16，未重新拟合或 bootstrap。

在当前 `evaluation_final/summary.json` 上执行同一程序正常退出：4 个完成 references，45 项 pending，绘制空间点=0、配对页=0、种子均值页=0，`requested_comparisons_complete=false`。对应索引已写出；完整空间和种子图的实际数据渲染尚未测量。

执行环境为 LCG_108、单线程 CPU；命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.plots --evaluation "$S2_RESULTS/evaluation"
python -m analysis.flavour.s2.plots --evaluation "$S2_RESULTS/evaluation_final"
python -m unittest discover -s tests -p test_flavour_s2_evaluation.py -v
```

10 tests passed，0.931 s；日志：`evaluation_plot_integration_tests.log`。这组检查覆盖原有指标、配对/种子统计、τ 分支和时间匹配逻辑；图形版式由实际 PNG 检查。当前代码增加绘图入口和调用，科学训练模块未改动。

commit `9d67e8a` 后，正式 `evaluate` 入口的 Condor 集成检查提交为 `1573758.0`，20:27 UTC 已 exit0，Condor log 的 TimeExecute=49 s。使用当前完整合同并显式启用 `--allow-incomplete`；`evaluation_final/summary.json` 与新图形索引均记录 source commit `9d67e8a`，4 个完成 references、45 项 pending，完成标记仍为 false。该检查已包含调用端，完整空间与三种子图仍待真实完成模型。

## 阶段性结论

现有 reference 的统计结果可直接转为可审阅配对图。最终合同的未完成项在图形索引中仍明确为 pending。尚无完成四轮的深度模型，本阶段没有新增架构、粒度或物理增益结论。

## 下一步计划

继续完成 35 项已提交科学任务，以完成模型更新评价，查看真实粒度图、GPU 时间匹配列和三种子图。最终交付仍须覆盖 41 项单 seed 配对、4 项种子均值、两家族 factorial interactions、条件域与 τ 分支，以及两家族原生错误事件显示。
