# 阶段 09：S2 完整目录、信息 toy 与统一对照工具

90,000 个冻结事件已全部转换为 32 个混类 shards，表示/model/evaluation/resume 共 11 项测试通过。τ 诊断分道已修正，所有事件及四类 targets 保持原样。正式模型比较尚未测量，S2 整体验收仍在进行。

## 阶段目标

完成固定子集的完整响应目录和运行工具，回答表示压缩是否移除细结构，并为相同输入、参数/步数及时间预算的模型对照准备评价程序。沿用用户确认的 samples、source-family split 和 detector-proxy 参数，不重新要求来源认证或真实硬件标定。

## 工作内容

- 完成 16 个 HTCondor conversion tasks，合计 32 shards。新目录与冻结子集逐行对应，保留空事件和全部原生 pixel/bar/accepted tracker channels。ROOT 按 32-row windows 读取；窗口内可能同时解码未选择行，但只输出选定 train/validation 事件。
- 汇总 `catalog.parquet` 和 1,024-event 表示成本抽样。发现超过 3,000 个局部块的事件后，全局 KNN 与 attention 改为每次 256 queries、完整 keys/candidates；attention tile 使用 checkpoint。节点/块仍全部保留，计算量仍随块数增长。
- τ 分道要求 primary τ 的直接 daughters 中同时有 charged lepton 和对应 flavour neutrino：e 需要 abs(PDG)=11 与 12，μ 需要 13 与 14；hadronic 需要 16、无 12/14 且有 hadron。同时满足两个 leptonic 条件记 ambiguous，其余 unknown。以 primary τ PDG/track ID 匹配 ancestry，不假定 ID=1。电离电子不再改变 μ/hadronic 分类；这些字段只用于诊断。
- 添加 two-track separation、matched-straight/kink support difference、Gaussian shower-support overlap toys。明确是无噪声几何信息研究，未拟合轨迹、不计算真实 kink detection efficiency。joint-pairing 的 projection 碰撞反例通过。
- 新增评价程序：四类与 τ 指标、1%/0.1% background acceptance 的实际通过计数、共同 energy/vertex 支持权重、分域/占用/τ branch、paired chunk bootstrap、三种子 spread、2×2 interaction、同型号 GPU 的 epoch-resolution 时间匹配，以及可追溯的错误事件组。
- 训练保存每 epoch validation 概率、optimizer steps 和分项时间；原子 epoch checkpoint 包含 optimizer/scheduler/scaler/RNG 和 best model/predictions。模拟第二 epoch 中断后的恢复与连续运行得到逐 tensor 完全一致的最终权重和相同概率。已完成同名配置不会重复训练。
- 新增 GPU resource profile 工具，在固定占用秩样本（包括尾部）上测完整事件 batch-1 inference/training、显存、转换/传输时间；这是压力曲线，不能当作无偏平均吞吐。

## 详细数据

结果根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11`。

| Item | Measured value |
| --- | --- |
| Train / validation | 80,000 / 10,000；各类分别 20,000 / 2,500 |
| Shards / combined Parquet bytes | 32 / 1,765,254,890 |
| Frozen-subset join | 90,000 一一对应，split/label/accepted family signatures 未改变 |
| All-empty train / validation | 2,253 / 282；总计 2,535 |
| Pixel-centered validation | 885；各类 245 / 237 / 176 / 227 |
| Large-scintillator validation | 4,494；各类 1,131 / 1,183 / 1,053 / 1,127 |
| τ train e / μ / hadronic | 3,433 / 3,507 / 13,060 |
| τ validation e / μ / hadronic | 445 / 395 / 1,660 |
| Selected τ unknown / ambiguous | 0 / 0 |
| Catalog audit duration / peak RSS | 175.807 s / 1,531.74 MiB |
| Scientific AUC, τ efficiency, seed spread, paired interval | 未测量 |

完整目录占用如下；channels 是本事件已激活/保留的读出数。

| Detector | Median | p90 | p99 | p99.9 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pixel | 23 | 1,146 | 4,908 | 11,568.027 | 34,177 |
| Bar | 404 | 1,903 | 2,974 | 3,435.001 | 3,710 |
| Tracker | 1 | 13 | 76 | 315.002 | 1,100 |

1,024 个固定抽样事件的表示成本（pixel+bar）：

| Pixel representation | Median nodes | Maximum nodes | Median blocks | Maximum blocks |
| --- | ---: | ---: | ---: | ---: |
| Native joint | 567 | 15,780 | 90 | 3,043 |
| Native projection | 615.5 | 10,167 | 99 | 815 |
| 100 μm joint | 566 | 13,288 | 90 | 3,043 |
| 200 μm joint | 566 | 10,956 | 90 | 3,043 |
| 1 mm joint | 564 | 5,905 | 90 | 3,043 |
| 1 mm projection | 580 | 4,415 | 99 | 785 |

bar 主导部分事件的总 nodes，因此 median nodes 几乎不变不能说明 pixel 粗化没有损失信息。以上 maximum 来自该抽样，不是完整目录的表示 maximum。

交互式繁忙 T4 上，两真实事件/755 blocks 的一次 forward+backward：ParT dim=48，150,076 parameters，5.4999 s / 156.566 MiB；ParticleNet dim=64，159,620 parameters，2.6104 s / 545.081 MiB。两者 loss/gradients 有限。测量先于最终 query tiling，且受其他 GPU 工作影响，不用于预算或吞吐结论。

toy 输出为 `toys/toy_curves.csv`（420 行）、`toy_topology.png/pdf` 与 `summary.json`。native pitch 为约 20.8×22.8 μm；shower toy 每个 support 为 100 deposits、σ=0.12 mm，未含材料散射、噪声或无效率。图已目视核对，英文标注完整。

作业记录：bigbird20 上 planning 1573722、conversion 1573724、toy 1573727、catalog 1573729、τ refresh 1573731 均正常完成。初次 GPU smoke 1573725 被调度至 V100，LCG Torch 无匹配 CUDA kernel；修正 `Capability >= 7.5` 后的 1573726 两任务仍在排队。未重复提交仍存活的 smoke。每次新提交继续读取当前 schedd 兼容性/负载。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p test_flavour_spatial.py -v
python -m unittest discover -s tests -p test_flavour_s2_evaluation.py -v
python -m unittest discover -s tests -p test_flavour_s2_training.py -v
python -m analysis.flavour.s2.submit profile --root "$S2_RESULTS"
# 科学 campaign 完成后：
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS" \
  --output "$S2_RESULTS/evaluation" --comparisons "$S2_COMPARISONS"
```

测试实测：spatial 6 项 / 6.246 s；evaluation 4 项 / 0.260 s；training recovery 1 项 / 6.009 s。spatial 包含 dense/tiled attention、KNN 数值一致及 checkpoint 有限梯度；evaluation 包含相同模型 paired CI 为零、共同支持权重、factorial identity 和时间预算禁止使用超预算 checkpoint。只验证 S2 新实现。

## 阶段性结论

事实：完整事件的原生响应已可重复读取，projection 和粗 cell 的信息损失在构造反例与 toys 上可复现；完整占用尾部要求限制计算中间量，不能只按 median 设计。τ 分道的电离电子污染已在诊断定义中修正。

假设：native joint 的细结构可能改善 τ 分类。当前没有 neutrino 分类比较结果支持该假设；toy support 的差别也不等于网络会利用该信息。

## 下一步计划

1. 完成已排队的 GPU smoke 与 resource profile，按实测吞吐冻结 epochs、batch 与预算，再启动 seed-42 两族的完整 2×2/100/200 μm 对照。
2. 完成 OR、原点平移、pair/graph、局部 shuffle、模态与训练域对照，B0 在同一子集重训；主要架构及 coarse 对照重复 17/42/2026。
3. 用真实 validation 概率运行评价，补错误组原生显示、资源曲线、配对区间和 S2 工单逐项验收。最终 calibration/test 继续保留至 S6。
