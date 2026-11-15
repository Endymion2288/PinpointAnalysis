# 完整路线整理、文献复核与修正单位后的学习验收

## 阶段目标

将已有独立审查、R1/R2最终结果和R3最新证据整理为可直接阅读的完整路线，回答是否有改进、是否已到最优、如何选择类别及下一步如何投入资源。完成stage289硅厚度修正后受影响学习路径的实际验收，明确正式泛化结果的状态。

## 工作内容

重写`roadmap/README.md`总入口，统一九份v2详稿的阶段状态与研究命名，补齐stage286–290索引。保留v1的14份完整归档及原configs/results/tools路径。当前路线继续使用v2的R0–R7结构，已完成、正在执行和条件设计分别列明，不另建一套重复阶段编号。

重新核对本地合作组报告的图像表示、网络和训练条件，以及FASERCal本地`main.tex`。联网读取FASERCal、NuGraph2 context-aware全文、pAUC会议页、INFERNO官方摘要和CERN训练文档。新增紧凑sensor-row图像的具体screen设计：响应不变，显式保留真实z/间距，先核对映射和实测成本，再做100k×4比较。当前未实现、分类与吞吐均未测量。新增过程产额拟合的条件路线，先交付完整score模板/响应矩阵，具有物理曝光和nuisance模型后再考虑推断目标。

读回Condor `1583003.0`的五个revision3真实学习结果。更新R3状态中的M0终点计数，移除已完成项目的pending标记；旧revision2只保留诊断和相同执行图资源口径。通过现有accounting入口实时更新R3资源，保留正在运行的G1/G2 profile。

## 详细数据

已有研究的核心数值已整理到路线入口，原始独立test与新Tune开发结果分别表述：S6 MAE3相对Scratch的BA4为+0.3354 pp；R1三seed B1 T3 BA3=56.5633%；R2 D2真实T3/conditional T4 BA3=57.0967%/57.1267%，T4 BA4=45.7650%，开发集1% FPR的τ效率6.05%。M0的BA3+0.383 pp伴随共同μ recall−4.08 pp，保留原采样。详见stages236/274/284/288。

材料单位来自G4 `AnalysisManager.cc`：silicon header为μm，10 μm=0.010 mm；W/scintillator header为mm。以下均为修正后revision3、BF16 encoder/FP32 head、seed42的真实Train拟合：

| Variant | Task | Events | Steps | Accuracy | NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| Native control | T3 | 24 | 96 | 1.000 | 0.006426919 |
| Native control | T4 | 32 | 128 | 1.000 | 0.008801566 |
| Prelocal relations revision3 | T3 | 24 | 128 | 1.000 | 0.000607550 |
| Prelocal relations revision3 | T4 | 32 | 128 | 1.000 | 0.009574102 |
| Relations without global stack, revision3 | T4 | 32 | 160 | 1.000 | 0.024306824 |

每项均通过fit NLL/accuracy、非恒定概率、活动分支梯度、参数更新和finite激活六类检查，共30项布尔检查通过。数据是固定informative Train事件；不是正式Tune分类排名。正式G2分类训练和分类收益仍未测量。

产物：

- R3根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。
- 五项gate：`fit/g2_c5/gates.json`及`fit/g2_c5_c{0,1,2}/B2_T*.json`。
- 官方来源、正文及独立gate读回：`/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_review_2026-11-15/`；小型来源清单纳入`roadmap/v2/source_update_stage290.json`。
- 2026-11-15 15:22:38 UTC的R3账本：2.804796异构device-hours、38.243919 allocated CPU core-hours；活动项目为`r3_g2_diagnostic_c4`与`pinpoint-r3-g1-k1`，最终成本尚未测量。按GPU/MIG分别记账，不作FLOP等效换算。

五个官方网址均HTTP200。R3的Kubeflow实时查询成功，G1使用H100 NVL MIG 3g.47gb，Condor profile使用H100L MIG 1g.12gb；无需为统一型号等待。profile的完整Tune峰值仍在测量；其revision2配置只作同shape/operation资源证据，修正单位后的正式模型须记录实际运行健康和峰值。

验证命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_v2_r3 tests.test_flavour_v2_r2 tests.test_flavour_v2_r1 -q
python -m analysis.flavour.v2.r1.accounting \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3
git diff --check
```

37项相关单元检查全部通过；路线图Markdown的本地链接、索引及JSON解析另作静态读回，汇总见`roadmap/v2/delivery_stage290.json`。未新增测试或修改训练架构。Cal-A/Cut-B/Confirm payload未读取。

## 阶段性结论

事实：已有小幅独立泛化收益和较强开发基线；复杂模型未在当前预算下胜出。三类化未稳定改善共同任务，保留四类研究有现有证据支持。硅单位修正后的五条学习路径均有效，消除了此次错误单位作为正式学习验收依据的问题。R3正式结构增益和新独立确认仍待形成。

设计判断：下一批优先局部保留、显式跨层关系及紧凑图像控制；再做候选质量合格的对象融合、τ分道和一种定向预训练。紧凑表示、关系层或更多类别是否改善结果都属于待检验假设。没有证据可以宣布当前方法达到最优或判定硬件四类信息上限。

## 下一步计划

读回G1固定曝光终点及G2完整资源profile，按已修正单位和匹配本族控制冻结G2最小矩阵；比较逐类、稀疏排序、τ工作点和成本。只对最有价值改变补seed/粒度。紧凑图像先做映射、真实拟合、Train-only计时和单候选screen。R4/R5以候选recall、整轨fake–efficiency–coverage及inclusiveτ收益触发；R7在候选冻结后才开放校准、阈值与新确认集。全部资源按预计完成时间选择，Condor/Kubeflow、混合型号、共享设备与MIG继续并用。
