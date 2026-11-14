# 阶段248：投影模型同设备成本与模态对照精度对齐

**同一T4上的普通CNN训练步吞吐比ResNet高20.3%，但峰值显存更高。B1无tracker对照补跑FP32，使其与当前主模型的训练精度一致。**CNN完整四pass性能与B1 tracker净效应尚待最终分数和配对统计。

## 阶段目标

补充可解释的CNN成本参照，避免仅从不同GPU的wall判断结构效率。核对模态重训对照除输入以外的实际训练配置，消除已发现的精度差异。

## 工作内容

新增`roadmap/tools/profile_r1_projection_cost.py`，使用R0冻结的32个真实Train事件、同一物理投影/TrackerSet、batch8，在同一T4比较普通CNN与ResNet。两个模型各16步warm-up、64步计时，包含forward/backward、梯度裁剪及AdamW，lr=0。输入batch预加载到GPU，不读取Tune/Cal-A/Cut-B/Confirm，不产生泛化结论。

主B1训练在T4运行，实际FP32；已完成的B1无tracker训练在H100 MIG运行，实际BF16。保留该完整BF16结果，并用`r1.extend --precision FP32`补充同精度重训，独立命名`final_b1_t3_s42_without_tracker_fp32`。主训练与其余已运行实验继续完成。`deliver`按主模型的实际精度选择模态对照，`finish`等待对应权重，`accept`核对两者精度一致。此项是输入增量比较的执行补充，不改变主HPO、架构排名及seed规则。

## 详细数据

### 同设备实测成本

设备Tesla T4、SM75、FP32；两个模型均为T4输出头，相同32事件与batch顺序。总操作wall18.634 s，包括读取/组装和两个模型的warm-up；下表计时只含稳态训练步。

| Backbone | Parameters | Median step (s) | Mean step (s) | Measured events/s | Peak allocated GPU (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet | 793,674 | 0.084102 | 0.084037 | 95.196 | 1,077.432 |
| Plain CNN | 179,370 | 0.069916 | 0.069860 | 114.515 | 1,260.631 |

稳态step的10%–90%分位分别为ResNet[0.083288,0.084811] s、CNN[0.068609,0.071127] s。CNN参数约为ResNet的22.6%，实测吞吐为1.203倍。CNN前端stride2保留更大的中间feature maps，峰值allocated显存高约17.0%；参数少不直接等同于更低显存。本样本仅覆盖固定32个informative事件，完整数据occupancy、I/O及validation成本由正式训练另测。

原始记录及事件键：`R1/audit/projection_cost_profile.json`；小型版本：[r1_projection_cost.json](../roadmap/v2/r1_projection_cost.json)。该18.634 s交互小样本操作已纳入`r1.reference_costs`的操作时间下界，不作为共享交互GPU的allocation小时。

普通CNN在Condor1582885、H100完成两项真实学习门槛：

| Task | Train events | Updates | Final fit NLL | Fit accuracy | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| T3 | 24 | 64 | 0.002229 | 100% | 179,241 |
| T4 | 32 | 64 | 0.016501 | 100% | 179,370 |

两个case的有限激活、分支梯度、参数更新及非恒定预测检查通过。正式T3/T4四pass随后在同一H100运行；其BF16完整质量与主ResNet的FP32质量比较标明实际精度/设备，成本结构差异以上述同T4小样本计时补充。

### 模态对照精度

已完成BF16 B1无tracker的BA3为0.531867、300,000 exposures。该值不直接用作当前FP32主B1的单因素tracker增量。新增FP32任务`pinpoint-r1-b1-no-tracker-fp32-k15`在前一MIG任务成功结束后启动，复用冻结事件、学习率、四pass和AdamW配置；同精度BA3及差值当前**未测量**。新增控制不进入架构排名。

实现保留提交配置与实际runtime精度；命名后缀对应请求精度，最终以manifest实际值验收。B4主模型与无tracker控制均为BF16，原控制继续有效。单设备内的T3/T4和计算控制保持同一精度与allocation；跨族、跨seed的混合设备配方按实际precision报告，作为本轮可运行配方比较。

### 命令与验证

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python roadmap/tools/profile_r1_projection_cost.py --root "$R1_ROOT"
# 下列训练payload已通过Kubeflow提交。
python -m analysis.flavour.v2.r1.extend --root "$R1_ROOT" --family B1 --seed 42 \
  --drop-modality tracker --precision FP32
python -m analysis.flavour.v2.r1.reference_costs --root "$R1_ROOT"
```

真实计时中所有梯度裁剪的finite检查通过；新增FP32 CLI已生成独立配置并在MIG启动。依赖控制器在核对唯一进程后重启加载新依赖，已运行训练未重启。修改代码通过AST解析、JSON和本地报告链接检查及`git diff --check`；未为文档和计时脚本新增镜像测试。

## 阶段性结论

**事实：**普通CNN参数更少、同T4稳态训练步更快，当前结构峰值显存更高。已有BF16无tracker结果和FP32主B1之间存在训练精度差异；已补充对应重训及验收约束。

**研究判断：**本轮异构资源策略继续有助于启动任务。科学比较需分别记录实际配方，并对模态等单因素解释补齐可控制的精度条件。三类/四类及最强架构的最终选择仍等待完整R1矩阵。

## 下一步计划

读回CNN及FP32无tracker控制的完整Tune分数，完成剩余主训练、按冻结排名的seeds和压力诊断。最终交付同时保留统一事件、曝光、计算口径、精度与资源证据。
