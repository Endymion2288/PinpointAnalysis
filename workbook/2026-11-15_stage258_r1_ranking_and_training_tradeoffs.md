# 阶段258：R1完整架构排名与同结构训练取舍

**冻结的四pass排名为B1、B2、B3、B4，按合同扩展B1/B2及共同参考B4。B1的同结构fresh计算控制显著恢复μ recall，同时降低e和NC recall；先前的低occupancy μ问题还需排查训练与决策。**完整R1交付仍在执行。

## 阶段目标

完成seed42八项主任务的实际排名，核对B1 fresh计算控制，提前枚举最终评估合同，并把新证据落实到R2/R3的控制变量设计。

## 工作内容

排名使用启动前冻结的规则：Scratch，T3和T4 conditional三类BA的算术平均，平均NLL仅用于平局。新增B2 seeds17/2026由唯一控制器提交，无并行writer。Cal-A/Cut-B/Confirm继续封存，R2未启动。

`deliver evaluation_config --plan-only`提前生成41模型、133配对的明确合同；该选项不声明模型完成。`evaluate`和`stress_evaluate`均读取manifest并要求complete。实际提前执行nominal评估正确拒绝尚未完成的`final_b1_t3_s17`，证据为`R1/audit/planned_evaluation_contract_check_stage258.json`。

最终汇总增加实际学习曲线、capability、memory、概率健康及真实bootstrap design。occupancy英文图支持实际候选/参考标签，贡献总和从数据读取。R2优先profile B1/B2，按各自实际容量设计加宽，并在精度改变时补相应D0；R3保留稀疏结构控制，同时加入训练/决策机制待查。

## 详细数据

输入为冻结Train100k、Tune40k，每类25k/10k。T3实际训练75k非τ事件，四pass共300k曝光；T4为100k四类事件，四pass共400k曝光。

| Family | Mean BA3 | Mean NLL3 | T3 BA3 | T4 conditional BA3 |
| --- | ---: | ---: | ---: | ---: |
| B1 | 0.565567 | 0.869108 | 0.566200 | 0.564933 |
| B2 | 0.553950 | 0.896383 | 0.551200 | 0.556700 |
| B3 | 0.551750 | 0.895169 | 0.550600 | 0.552900 |
| B4 | 0.545133 | 0.902055 | 0.543500 | 0.546767 |

完整实测见[r1_architecture_ranking.json](../roadmap/v2/r1_architecture_ranking.json)。B2比B3高0.220 pp，这是点排序；该排序本身未提供架构差的统计区间。八项主模型固定Tune-screen的末轮NLL仍下降，B3 T3/T4末轮分别下降约0.01640/0.01606，收敛与架构上限均未建立。

### B1同结构fresh计算控制

seed42，T4 GPU、FP32、同一结构、fresh初始化，主任务和控制均在allocation1582883.0。控制同时改变总曝光与wall-based LR schedule，效果不能仅归因于训练时长。

| T3 recipe | Exposures | BA3 | NLL3 | e recall | mu recall | NC recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four passes | 300000 | 0.566200 | 0.871967 | 0.6301 | 0.4998 | 0.5687 |
| Fresh timed control | 391633 | 0.561267 | 0.872934 | 0.5655 | 0.6126 | 0.5057 |

控制wall为5199.13075 s，相对T4的5193.24387 s预算误差+0.11336%。51个内部健康样本中，image/tracker/head均有正梯度与参数更新，activation全部finite；概率不同向量38,918个。

同一400次accepted-family连接物理块draw，seed20261116、Tune40k、6,845块，timed减four-pass：

| Metric | Point difference | 95% interval |
| --- | ---: | --- |
| BA3 | −0.4933 pp | [−0.8764, −0.1051] pp |
| NLL3 | +0.000966 | [−0.001256, +0.002910] |
| Macro AP3 | +0.002390 | [+0.000646, +0.004065] |
| e recall | −6.460 pp | [−7.110, −5.828] pp |
| mu recall | +11.280 pp | [+10.523, +11.961] pp |
| NC recall | −6.300 pp | [−6.964, −5.597] pp |

低occupancy按原accepted的pixel+bar+tracker channel数<100定义。2,801个μ事件中，timed正确1,205、four-pass正确326；新增正确893、反向14，净增加879。域内μ recall差+31.3816 pp [29.5251,33.0366]，对全域μ贡献+8.790 pp [8.2174,9.2951]；同组e/NC分别少识别703/367个。全空响应仍全部判NC，原类人数为[256,241,326,260]。每draw分组贡献加总恒等式最大误差2.78e−17。

timed T3相对T4 conditional的BA3差−0.3667 pp [−0.7590,+0.0314]，NLL3差+0.006684 [+0.004018,+0.009211]。相对B4 T3的BA3差+1.7767 pp，但NC recall仍低1.13 pp；设备和训练预算不同，该项为描述性比较。

紧凑结果见[r1_b1_compute_controls.json](../roadmap/v2/r1_b1_compute_controls.json)。独立读回116项通过，最大工作点误差1.063e−14。英文图`R1/interim/b1_compute/figures/r1_occupancy_localization.{pdf,png}`已生成并目视核对。

### 全部压力路径与运行状态

11模型×5条件共55条路径已complete；`R1/evaluation/stress_results.json`完成400次配对统计、21项模型比较，metrics revision为`r1-tie-aware-float-boundary-2`。独立stress读回作业1582917正常退出，**5,666项通过，最大绝对误差1.4433e−15**，核对类别均值、选择计数、冻结cut和配对点差。区间的统计口径为固定拟合seed的开发物理块bootstrap。

B1 T4三seed均已完成：seeds42/17/2026的BA3为0.564933/0.562633/0.561567，NLL3为0.866249/0.870914/0.864153；μ recall为0.4694/0.5128/0.5503。seed42在T4/FP32，17和2026在H100 1g MIG/BF16 encoder+FP32 head；逐类波动包含seed和实际执行精度/设备差异，不作纯RNG不确定度解释。

03:00:33 UTC资源快照为30.321282 allocation device-hours，尚有训练未关闭。按型号为A100 12.530923 h、H100整卡7.018889 h、H100 1g MIG3.627859 h、H100 3g MIG2.685833 h、T4 4.298611 h、失败V100 0.063333 h、启动前历史h100-47gb 0.095833 h。异构小时不作FLOP等效；原18 h规划已超过。历史MAE SSL 0.635321 h单列。最终成本待全部终态。

### 验证命令与产物

使用LCG108/Python3.12、`PYTHONPATH=src`，OpenMP/BLAS线程1：

```bash
python -m unittest tests.test_flavour_v2_r1_summary
python -m py_compile src/analysis/flavour/v2/r1/summarize.py \
  src/analysis/flavour/v2/r1/deliver.py src/analysis/flavour/v2/r1/evaluate.py \
  src/analysis/flavour/v2/r1/stress_evaluate.py \
  roadmap/tools/plot_r1_occupancy_localization.py
python -m analysis.flavour.v2.r1.review --root "$R1_ROOT" --stress-only
git diff --check
```

4项针对性测试和语法检查通过。一次误用未source的系统Python因缺少pandas未能导入，改用上述匹配环境后通过。批量独立读回经HTCondor提交。根目录为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`；完整统计、manifest、预测与图表保存在该目录。

## 阶段性结论

**事实：**B1在固定四pass开发排名领先，B2按冻结规则进入扩seed。相同B1结构可以在训练配方改变后大幅改变μ/e/NC取舍，低occupancy μ损失不能直接作为不可逆投影信息损失的结论。完整stress点值与计数已通过独立核算。

**待检验假设：**训练规模、LR schedule、类别决策与表示压缩可能共同影响性能。学习曲线尚未饱和；现有证据不足以选择其中唯一原因。R2的逐窗口概率、AP/ROC和recall、必要的交叉拟合readout诊断，以及R3单因素结构对照分别检验这些机制。

## 下一步计划

完成B1/B2所需seed和B3 fresh compute控制，执行41模型nominal统计、full独立读回、英文图与终态记账。完成accept后导出最终紧凑结果与中文报告，并依据完整三seed证据冻结R2设计。继续接受不同GPU、共享设备和MIG，以实际吞吐及预计完成时间安排资源。
