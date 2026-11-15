# 阶段251：四类τ选择的先验情景与共同基线配对

**已完成的seed42模型中，B1的τ效率为5.09%（1%背景接受率）和1.34%（0.1%背景接受率）。**在固定条件响应、πτ=0.001、其余三类等先验的情景下，对应purity为0.507%和1.324%。平衡样本的purity不能直接解释为稀有τ样本的purity。当前完整排名、多seed和剩余条件路径仍在执行。

## 阶段目标

补齐四类τ工作点的显式prior情景，检查不平衡条件域中的归一化，并使领先架构对HGB和投影基线的seed配对可以直接评价。

## 工作内容

`evaluate.domain_report`原先已给出三类选择中的τ污染情景；本阶段增加`T4_tau_prior_scenarios`，保留1%/0.1% benchmark FPR下的原分数、完整tie group和cut，输出四类原始通过人数、支持人数及πτ=0.25/0.01/0.001的重权结果。每个情景先计算各类的条件接受率，再乘明确的四类prior。缺少任何类别支持的域，其外推情景记为未测量。

新增的四类情景改变混合权重，不改变网络的分数、argmax或cut；也没有实施posterior prior correction。完整nominal和stress交付均会包含该字段，验收增加对应检查。

`deliver.evaluation_config`补充B1/B2/B3对同任务、同seed的HGB比较，并将最强native与B1的同构比较扩到实际共同具备的seeds17/42/2026和T3/T4。seed42保留all/pixel-centered/large三域。这些差值复用已有模型、物理块draw和缓存，不新增训练或改变Scratch排名；它们是在首组结果公开后的描述性比较。

## 详细数据

### 全Tune的已完成τ工作点

样本为冻结Tune40k，四类各10k，非τ背景30k。下表由已完成训练manifest的同分数工作点读回；两个工作点分别通过300/30个背景，保留完整ties。B3 T4仍在完成推理，未进入本表。当前数值为seed42的开发点估计，完整配对区间和训练seed波动待R1最终交付。

| Model | τ efficiency at 1% FPR | Purity at πτ=0.001 | τ efficiency at 0.1% FPR | Purity at πτ=0.001 |
| --- | ---: | ---: | ---: | ---: |
| B1 ResNet | 5.09%（509） | 0.506927% | 1.34%（134） | 1.323588% |
| B2 fixed-graph ParticleNet | 2.82%（282） | 0.281488% | 0.38%（38） | 0.378939% |
| B4 Scratch | 2.69%（269） | 0.268546% | 0.31%（31） | 0.309350% |
| B4-MAE | 2.41%（241） | 0.240661% | 0.32%（32） | 0.319298% |
| B0 HGB | 2.83%（283） | 0.282483% | 0.26%（26） | 0.259585% |

括号为通过τ的原始人数。B1在平衡prior下的两项purity分别为62.9172%和81.7073%；πτ=0.01时为4.89000%和11.9217%。所有情景固定其余三类相对背景组成，不包含实际flux或cross-section权重。

全域平衡背景的独立算式为：

```text
purity(pi) = pi * tau_efficiency /
             (pi * tau_efficiency + (1-pi) * actual_non_tau_FPR)
```

数据来源为`R1/experiments/{model}/manifest.json`的`metrics.T4.per_class[2]`。逐模型、逐工作点完整读回保存于`R1/audit/tau_prior_scenarios_stage251.json`。上述表没有用该算式替代不平衡条件域的逐类接受率重权。

### 验证

新增端到端测试使用四类人数100/200/40/300、通过人数1/2/20/3的非平衡域。三个非τ类接受率均为1%、τ效率50%；三个prior情景均与独立purity算式一致，并核对完整通过人数。删除一个背景类后，外推情景返回未测量。这一检查区分逐类条件响应与直接重权原始混合人数。

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest tests.test_flavour_v2_r1
```

16项R1测试通过。配方构造检查使用假定B1/B2前两名的fixture，得到41模型、133配对，全部引用有效；新增HGB配对26条，native/projection配对10条。这个fixture检验配置连通性，不是已测架构排名。本阶段读取小型训练manifest及运行CPU单元检查，没有新增GPU allocation。最终完整评价将直接从概率表核对新增情景。

## 阶段性结论

**事实：**B1在本seed的两个τ工作点均有较高点效率；低τ prior下的purity仍低。名义BA、balanced purity和稀有样本的可用性回答不同问题。

**研究判断：**当前结果支持保留四类研究，并继续完成多seed、共同三类非劣和条件鲁棒性比较。是否将τ作为近期物理交付，仍按冻结的R5/R7效果与确认合同判断。R1不会据此启动更多主类别或提前使用Cal-A/Cut-B/Confirm。

## 下一步计划

继续完成四族主任务和同allocation计算控制，按实际Scratch排名扩展前两族及B4共同参考。最终表直接给出领先模型对HGB、B4和投影基线的配对差值、seed sample SD、三域结果及新增τ先验情景；补齐55条条件路径、英文图表与终态成本后验收R1。
