# S4：几何查询对照与等待期间的优化实验

## 阶段目标

量化重建中的几何查询捷径；在正式 BF16 GPU 排队时用 FP32 小规模实验继续检查 Rel 和 EMA。

## 工作内容与详细数据

输出根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。

`experiments/s4_geometry_query_baseline_s42`：CPU HTCondor bigbird21 cluster 2198169，2048 fit events、1024 steps、固定完整 2k train-inner reconstruction probe。所有事件共享常量 latents；不读取 encoder、分类标签或 downstream validation。

| Target | Query-only AP | Full MAE3 AP |
| --- | ---: | ---: |
| fine native patch | 0.02819256 | 0.45497851 |
| 4-mm pixel cell | 0.11153190 | 0.78089867 |
| scintillator bar | 0.44580716 | 0.89860048 |

Query-only elapsed 413.86821 s，4 CPU allocation，GPU=none。该控制与 full MAE 不等 fit exposure 或计算预算；它单独度量已训练 decoder 从 query geometry 获得的能力。结合 MAE 的同 checkpoint event-permutation ΔBCE，可确认主重建依赖事件编码，而非只读取查询位置。

1k Scratch seed42 已完成8epochs，checkpoint 14项读回通过，最佳 validation NLL 在 epoch1，为1.51618843，macro AP0.29720868，ντ AP0.25953440；后续 train loss下降至约0.045伴随泛化变差，表示小样本过拟合。内部非恒定输出与数值检查通过；该现象不归因于死 Head。低标签改进需看同协议预训练和 frozen probes，不能用训练拟合作结论。

A100/H100 队列即时 better-analyze：可兼容的空闲机器当前不愿接收该任务。继续保留正式 BF16 下游 campaign；不修改运行中的 Kubeflow core。空闲 V100 可执行 FP32 eager 小规模实验，因此新增 `s4_fp32_prototypes.json`，最多1epoch/1024steps，单独命名、精度、成本和运行配置。Rel-local、EMA batch64、相同精度 MAE/EMA batch16 控制分别执行。正式下游主协议仍为同一 BF16 backbone，FP32 prototypes 不冒充等算力主要对照。

后续 source archive 首次读取新 Git tree 时出现短暂 EOS 可见性延迟（随后同对象恢复为560bytes且可读）；已将 `core.createObject=rename`，保留 `core.fsync=committed`。未重写历史。CPU compute profile 已提交 bigbird21 cluster2198171。

## 验证与阶段性结论

阶段97 matrix-operation test通过；阶段96的16项回归检查通过。FP32只用于显式prototype配置，程序限制最多1024steps。事实：MAE重建明显超过geometry-only control；1k Scratch显示过拟合；Rel语义和EMA优化仍需新的实测。

## 下一步计划

完成修复pilots，冻结通过学习readback的策略；完成下游label curves和三seeds；按矩阵运算proxy和实际allocation分别比较训练成本，决定是否扩量。
