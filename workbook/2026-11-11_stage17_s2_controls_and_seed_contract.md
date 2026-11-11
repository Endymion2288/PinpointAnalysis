# 阶段 17：冻结 S2 最小控制与三种子统计合同

新增统计实现通过 7 项检查；S2 的剩余训练与最终对照清单已冻结。当前空间训练尚未完成首个 epoch，本阶段没有新的分类性能结论。

## 阶段目标

在读取空间矩阵完整 validation 指标前，固定足以回答 S2 问题的最小追加实验，以及共享物理事件的三种子差值统计。验收是配置完整、每个名称可追溯、统计实现通过检查；S2 整体验收仍需真实训练结果。

## 工作内容

[13 项控制](../roadmap/configs/s2_controls_v2.json)：两家族分别进行 1mm OR、半格平移、native plane 内局部 shuffle；ParT 移除 pair bias，ParticleNet 固定 geometry graph；ParticleNet 进行 pixel-only、bar-only、pixel+bar+tracker，以及 pixel-centered/large-scintillator 训练域控制。pixel+bar/full-domain 复用空间主矩阵。modality/domain 最小控制预先选用实测计算较省的 ParticleNet，其结论以该 backbone 为条件。训练域较小时重复完整域循环匹配步数，不把重复曝光算作新的独立事件。

[8 项种子确认](../roadmap/configs/s2_confirmation_v2.json)：两家族 native joint/1mm joint，各追加 seed17/2026，复用空间矩阵 seed42。同一 80k train/10k validation、4 epochs、每 epoch 20,016 attempted steps；保持 GPU、软件、模型宽度、优化器与数据顺序规则一致。sample/source-family 与 detector-proxy 使用用户接受的版本。

[最终比较合同](../roadmap/configs/s2_final_pairs_v2.json)含 39 项单 seed 配对和 4 项三种子均值：两族 native 相对 1mm 收益、native/1mm 下 ParT 相对 ParticleNet 收益。差值方向统一 B−A。三种子统计先分别计算每个模型的 τ AP，再平均配对差值；同一次 bootstrap 对所有 seeds 抽取相同 physical blocks，不拼接重复事件，也不平均概率形成 ensemble。逐 seed 差值和标准差单列，事件区间以这些已训练模型为条件。common-support 区间还以固定的 E/vertex 权重为条件。

科学训练共 33 runs。空间初筛组织上限 100 GPU-hours，确认与控制追加上限 100，总上限 200，含旧失败/替代成本；这些是计算安排，实际完成成本未测量。实际 allocation 随 `campaign_accounting.json` 更新，首个完整 epoch 后更新完成成本估计。

## 详细数据

18:40 UTC 队列快照：`1573747.0–3` running、`.4–11` idle；`.6/.8` 为获取 ParticleNet 两个主要表示而设置 JobPrio=10。不重复提交这些 tasks。

18:41 UTC 读取的 ParT native 进度为 epoch1/step3000：11,997 event exposures、2,998 applied optimizer steps、2 AMP skips、累计 train loss=1.39044、16.61 events/s、722.29 s。这是部分 epoch 的执行进度，不是 validation NLL/AP，也不能据此判定模型收敛。

验证命令与实际结果：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p test_flavour_s2_evaluation.py -v
```

7 tests passed，0.827 s；日志为 results 根目录的 `evaluation_unit_tests.log`。其中三种子检查包含物理事件数不扩大、乱序对齐、perfect/tied 分类器的已知 AP 差值 `[+0.75, −0.75, 0]` 与 seed std=0.75、同一事件块相消区间，以及事件不一致时拒绝计算。配置检查确认 33 个唯一 experiment names、39 个合法配对与 4 个各含 17/42/2026 的均值组。

新增科学 runs 的 AP、τ branch efficiencies、factorial interaction、完整 allocation 和三种子 spread 均未测量。calibration/test 保持 S6 前冻结。

## 阶段性结论

事实：追加实验与差值统计已成为可执行、可审阅的合同，统计检查通过。此前 B0 结果仍是目前完成的科学参照；本阶段不增加对 native/coarse 或架构优劣的判断。

## 下一步计划

提交这些尚未存活的控制/确认任务，保留已有空间队列。依据完整 epoch 与实际 allocation 检查成本，逐项生成 39 个配对、4 个种子均值、条件域/τ branch 表及错误事件显示。所有指定结果到齐后运行无 `--allow-incomplete` 的最终评价，再写 S2 科学结论与阶段 gate。
