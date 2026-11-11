# 阶段 10：S2 信息参照和训练域的步数控制

四种 CPU 参照模型入口已完成，训练域控制可在更小的几何区域内重复完整周期，以达到 full-domain 相同的 optimizer steps。新增 sampler 测试和已有恢复测试均通过；参照模型的真实分类指标尚未测量。

## 阶段目标

将全局信号/空间形状与 sample energy/vertex priors 分开，给神经网络表示对照提供同一 80k/10k split 的参照；避免中心区域样本少、训练 steps 也随之减少的混杂。

## 工作内容

`s2.reference` 只读 S2 shards 已有的 B0 features 和 provenance，使用 HistGradientBoostingClassifier：200 iterations、learning rate 0.08、15 leaves、leaf 最少 40 events、L2=1，每 10 iterations 计算 validation NLL，保留 best。四个模型均使用全部固定 train/validation，不做 response cut。

| Mode | Input | Interpretation |
| --- | --- | --- |
| b0_all | 原有 22 项 pixel/bar summary | 硬件 response 的低维参照 |
| b0_signal | log channel counts、bar edep/PE、存在标志 | 全局信号和 multiplicity，不带坐标分布 |
| b0_shape | pixel/bar mean/std、pixel plane count、bar H fraction | 可观测空间形状，不含总信号；仍有几何接受度信息 |
| truth_energy_vertex | true initE/initX/initY/initZ | 样本先验诊断；不计作硬件模型，不进入 hardware 增益结论 |

CPU reference 与空间网络的模型预算不同；评价程序明确不将 CPU 时间标为 matched GPU-time。空间网络之间的时间对照要求同型号 GPU，并只选择共同 allocated-time 预算内的完整 epoch；更晚 checkpoint 即使质量更好也不能使用。

训练域 sampler 接受固定 `steps_per_epoch`。某区域完整周期耗尽后，从新确定性随机顺序开始下一周期，直至固定 steps；主域默认仍每 epoch 遍历全部事件。较小域重复样本，不增加独立物理事件，manifest 同时记录 unique train count、steps 和重复策略。validation 始终为完整 10k，再按各区域分层。

## 详细数据

- 输入仍为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/shards/`，不重新读取 ROOT。
- sampler/recovery 测试 2 项通过，运行 6.320 s。新增测试要求前两个完整周期分别恰好覆盖全部小样本，且达到指定 5 steps；恢复测试仍得到连续运行的精确权重与概率。
- 参照模型 AUC、NLL、τ efficiency：未测量。
- GPU smoke 1573726 和 resource profile 1573732 已提交、当前仍等待兼容 GPU；没有重复提交。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p test_flavour_s2_training.py -v
python -m analysis.flavour.s2.submit reference --root "$S2_RESULTS"
```

## 阶段性结论

事实：参照和训练域控制使用同一冻结子集及 validation 选择规则。步数控制保留完整周期，独立事件数不因重复而增加。truth 先验模型在 manifest、汇总表与名称中单列。

假设：若 truth-only energy/vertex 或仅全局信号已经解释多数分类能力，native pixel 的净结构增益可能较小；需要真实 validation 结果和共同支持诊断才能判断。

## 下一步计划

通过 HTCondor 运行四种 CPU 参照，记录每类及 τ 指标、共同 energy/vertex 支持、错误事件组。GPU 获得执行资源后完成 smoke/profile，冻结空间网络科学 campaign；S2 整体验收继续保持未完成。
