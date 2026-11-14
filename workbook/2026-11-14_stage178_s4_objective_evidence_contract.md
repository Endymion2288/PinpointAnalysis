# S4 有效目标矩阵的收尾检查

## 阶段目标

在既有15项分类训练完成后，核对五种初始化、三个标签预算的统一评价证据。将数值求解、磁盘模型回放、训练健康与泛化结论分开记录。

## 工作内容

新增`analysis.flavour.s4.accept_objectives`，复用已有训练健康、checkpoint和冻结矩阵检查，适配本轮30项配置、45套预测及66项预定配对。读取四份已完成SSL readback、实际encoder参数更新、同目标solver回归及30个磁盘读出回放。配置与正式validation worker冻结的输入逐项比对。统计要求沿用400次、128个物理块、raw/common-support、三个τ子道和逐类有限MC工作点。此检查不把技术完成等同于方法晋级。

发现并修正汇总代码的计划采样计数：`TransferAdamW`在LP解冻后的首次更新主动要求额外健康采样，旧`training_health`只列定期采样。五项1k LP-FT配置均有592次更新、每pass74步、2个LP passes；首次FT更新为第149步，记录确实包含149。现按冻结配置增加该必需记录，并逐条检查LP/FT阶段标记；漏掉149或额外加入任意记录仍不能通过。训练代码和已完成模型未改变。

## 详细数据与验证

结果根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。15项分类训练的147,130次更新、1,710次健康采样已完成；独立新目标checkpoint回放由已提交的统一validation执行，不在此重复计算。五项100k full读出在06:01 UTC仍为`fitting_full_linear`；短只读进程检查确认CPU进程活跃，没有在SSH中启动训练。1k/10k二十个full/global读出已完成，数值驻点通过。

Kubeflow当前认证正常，PINPOINT GPU作业已结束。06:00:27 UTC成本刷新为35.317084908/60 device-hours、413.685985619 CPU allocation core-hours；111条allocation中仅五组CPU读出和统一validation尚未终止。最终CPU成本未测量。

实际执行：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s4_objective_acceptance tests.test_flavour_s4_acceptance
```

12项测试通过，包含驻点标志与真实梯度不一致、C改变、缺失分支、错误回放、回放事件不足、放宽误差阈值、漏掉解冻采样与阶段标记错误的失败情形。另对五份真实LP-FT健康记录调用修正后的汇总函数，计划记录、阶段标记和梯度策略均通过。

## 阶段性结论

事实：额外149步记录来自训练时已定义的解冻监测，旧汇总器漏计，不能据此声称训练有故障。分类重训已完成；100k frozen驻点、统一配对、全部磁盘回放与最终CPU终态尚未完成。

固定C=1的低标签full读出有明显泛化差距与过度自信，见[阶段177](2026-11-14_stage177_s4_probability_diagnostic.md)。优化和回放通过不使该概率模型合格；保留固定配方诊断，后续在原标签预算的分组train-inner选择正则化。

## 下一步计划

完成现有CPU工作、统一评价和新收尾检查；保存条件区间、真实资源终态和英文图，随后清理五份临时feature bank与终态AFS提交目录。最终报告给出科学结果和概率泛化限制，不扩至1M。
