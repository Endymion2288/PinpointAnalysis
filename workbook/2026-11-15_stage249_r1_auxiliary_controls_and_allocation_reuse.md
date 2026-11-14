# 阶段249：完整CNN参照、同精度控制与资源接续

**普通CNN的T3/T4四pass均完成，BA3分别为54.75%/54.41%；B1无tracker的FP32重训完成，BA3为52.69%。**这些是冻结Tune上的开发结果。主ResNet及全部架构的T3/T4尚未齐备，tracker配对差值、CNN–ResNet差值和最终架构排名仍待完整统计。

## 阶段目标

补齐R1辅助实验的实际结果，核查任务和计算控制的初始化，接续使用释放出的GPU；修复资源汇总中已观察到的长时间历史查询。

## 工作内容

读回Condor1582885的完整CNN四pass、Kubeflow K15的FP32无tracker权重和全Tune预测。两个CNN使用B1已选配方，作为成本参照；不加入冻结的四族Scratch排名。

另外在CPU重新构造四族seed42的T3/T4模型，逐tensor核对共有形状的初始参数；再次构造T3，验证fresh计算控制的初始化可完全重现。该核对不读取数据或训练后checkpoint。

资源汇总原实现对60个ClusterId设置复合history约束，却要求最多120个匹配，可能继续扫描整个schedd历史。本次查询实际出现持续等待。修改为复用已保存的终态ad，仅对新终止且不在实时队列中的ClusterId分别执行`condor_history ID -limit 1`，最多4个并发查询；继续查询全部相关live ads。提交名和ClusterId共同索引缓存，输出显式记录schedd。

## 详细数据

### 四pass辅助模型

Tune为同一40,000事件、每原始类别10,000个；共同非τ分母为30,000。BA3/NLL3在e、μ、NC上评价，τ仍保留于预测和污染分析。所有结果为raw、未校准概率。

| Model | Actual precision / GPU | Unique Train | Exposures | BA3 | NLL3 | Operation wall (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Plain CNN T3 | BF16 / H100 NVL | 75,000 | 300,000 | 0.547500 | 0.910778 | 481.590 |
| Plain CNN T4 | BF16 / H100 NVL | 100,000 | 400,000 | 0.544133 | 0.906572 | 640.748 |
| B1 T3 without tracker, original | BF16 / H100 NVL MIG 3g.47gb | 75,000 | 300,000 | 0.531867 | 0.898937 | 见小型JSON |
| B1 T3 without tracker, matched precision | FP32 / H100 NVL MIG 3g.47gb | 75,000 | 300,000 | 0.526933 | 0.898594 | 821.980 |

CNN T4的BA4=0.426925、NLL4=1.225412、描述性1%非τ FPR处τεff=0.0266。两个CNN的参数数为179,241/179,370，峰值allocated显存为1,228.258/1,228.123 MiB。共同allocation为1582885.0；Condor实际终态JobStatus=4、ExitCode=0、allocation=1,148 s，包含两项拟合门槛及正式训练。

两个CNN分别有39/52个内部健康采样；每个采样的image、tracker、head梯度和所监视参数更新均非零，最终概率的各类标准差分别为T3[0.2693,0.1967,0.1818]、T4[0.2379,0.1655,0.1005,0.1502]。这些健康证据与stage248的真实小样本拟合共同支持训练确实有效。

FP32控制使用`final_b1_t3_s42_without_tracker_fp32`，只按预先确定的精度对齐规则进入tracker比较。BF16原控制仍保留供审查，不按较高BA替换选中的FP32控制。异构设备允许的TF32/kernel路径仍可能不同；相同浮点dtype不代表逐位相同的硬件执行。B1主模型与控制的最终配对区间**未测量**。

实际主文件为`R1/experiments/<name>/manifest.json`及`predictions.parquet`；完整数值和健康摘要保存于[r1_auxiliary_results.json](../roadmap/v2/r1_auxiliary_results.json)。同一T4的成本测量继续见[stage248](2026-11-15_stage248_r1_projection_cost_and_precision_alignment.md)，不同GPU的完整wall不能单独解释为CNN的结构加速比。

### 初始化核对

| Family | Common-shape tensors | Exactly equal T3/T4 | Repeated fresh T3 mismatches |
| --- | ---: | ---: | ---: |
| B1 | 68 | 68 | 0 |
| B2 | 68 | 68 | 0 |
| B3 | 116 | 116 | 0 |
| B4 | 181 | 181 | 0 |

输出类别数改变的tensor仅为最终分类层，以及B4的linear readout。其余初始tensor完全相同。原始记录：`R1/audit/initialization_readback.json`；执行方式为`torch.manual_seed(42)`后分别调用`build_model(recipe(...))`，比较`state_dict`的名称、形状及`torch.equal`。本检查验证构造契约；训练中的曝光、同allocation和预算对齐仍由最终实际记录验收。

### 资源接续与记账核对

CNN终止后，B4诊断1582886在其释放的H100节点开始。K15 Pod `pinpoint-r1-b1-no-tracker-fp32-k15-qx67w`实测Succeeded，终态快照已保存；随后把仍Idle且NumJobStarts=0的MAE诊断1582888移除。其history为JobStatus=3、NumJobStarts=0、RemoteWallClockTime=0，零allocation；新任务`pinpoint-r1-diagnostics-mae-k16`随即在Kubeflow MIG Running。迁移证据为`R1/audit/migrate_mae_diagnostics_k16.json`，终态和实时Pod由资源汇总持续保存。已运行的训练保持推进。

修改后的资源汇总已真实完成一次读回：60条Condor allocation无重复，CNN的1,148 s终态与独立history查询相同，所有零启动移除任务仍计零allocation。23:41:30 UTC快照的组织性device-hours合计15.577295；此时仍有运行作业，**最终成本未测量**。不同型号/MIG不作FLOP等效。18 h是原规划值，完整R1剩余矩阵按实际成本继续记账，下一阶段资源安排据最终实测重估。

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT"
condor_history -name bigbird20.cern.ch 1582885 -limit 1 \
  -json -attributes ClusterId,JobStatus,ExitCode,RemoteWallClockTime
```

上述完整训练、模型初始化核对、实际资源汇总与迁移已执行。Python AST、JSON读取及`git diff --check`通过；未新增镜像实现的测试。

## 阶段性结论

**事实：**CNN及FP32模态控制已补齐完整Train/Tune结果，四族任务共享的初始参数一致。释放的GPU持续用于后续必要诊断，未产生重复writer。资源汇总不再为已终止作业重复扫描整库。

**研究判断：**当前各辅助模型BA3处于0.53–0.55量级，概率质量和逐类行为仍需完整配对比较。精度、硬件、表示和训练曝光分别记录后，才能解释模型贡献；当前结果不建立最优性或收敛。

## 下一步计划

完成全部seed42主T3/T4与计算控制，按冻结规则确定前两族并补齐三seed；完成55条压力条件路径、400次配对统计、合作组三域对照、英文图表和最终资源终态审查。Cal-A/Cut-B/Confirm继续封存。
