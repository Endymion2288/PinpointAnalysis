# 阶段 11：使用现成 V100 软件环境推进 GPU 对照

LCG_107_cuda 的 PyTorch 2.5.1 包含 `sm_70` kernel，空间模型与恢复共 8 项测试通过；两个真实事件也完成两个模型族的 CPU forward/backward。GPU smoke 与 profile 改用该现成环境，以使用当前可调度的 V100。科学分类指标仍未测量。

## 阶段目标

解决 LCG_108 的 Torch 不支持 V100 导致的执行资源限制，复用 CVMFS 构建而不安装/编译新 framework。

## 工作内容

- 读取现成环境的 CUDA architecture list：LCG_108/PyTorch 2.7.1 从 `sm_75` 起；LCG_107/PyTorch 2.5.1 支持 `sm_50/60/61/70/75/80/86`。后者满足 V100 的 7.0 capability。
- 新增 `s2.submit --lcg-view` 和可选 `--gpu-device`，将软件环境与 GPU requirement 绑定。不同软件环境使用不同 wrapper 文件，避免修改已排队作业的启动脚本。
- 项目声明 Python 3.12；本次 GPU 作业为复用 V100 兼容构建，采用 LCG_107 已有 Python 3.11.9/PyArrow 15.0.0。S2 模块直接经 PYTHONPATH 运行，不在此环境重新打包项目。训练/profile manifest 保存实际 Python、Torch 和 Arrow 版本；CPU reference 继续使用 LCG_108。
- 取消旧环境中仍 idle 的 1573726 两任务、1573732 profile，再提交相同小样本/压力测量的兼容版本；属于执行环境替换，不将多个存活副本重复运行。

## 详细数据

| Verification | Measured value |
| --- | --- |
| LCG_107 Python / Torch / Arrow | 3.11.9 / 2.5.1 / 15.0.0 |
| Spatial tests | 6 通过 / 7.670 s |
| Sampler and resume tests | 2 通过 / 5.463 s |
| Real-event compatibility input | S2 smoke 的相同两个 train events，完整 native pixel+bar |
| ParT dim48 CPU CE / gradients | 1.3507480621 / 全部有限 |
| ParticleNet dim64 CPU CE / gradients | 1.4301966429 / 全部有限 |
| Compatibility artifacts | `compat_LCG107_tests.log`, `compat_LCG107_native.json` |
| Batch GPU throughput / science comparison | 未测量 |

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m analysis.flavour.s2.submit train --root "$S2_RESULTS/smoke" \
  --campaign "$S2_RESULTS/smoke/campaign.json" --lcg-view LCG_107_cuda
python -m analysis.flavour.s2.submit profile --root "$S2_RESULTS" \
  --lcg-view LCG_107_cuda --gpu-device 'Tesla V100-PCIE-32GB'
```

Git 在一次提交后短暂读到 zero-length loose tree object；数十秒后同一对象可正常读取，未删除或重建对象。后续检查确认其内容类型为 tree，并运行本地对象完整性检查；后续提交显式使用 loose-object fsync。没有修改历史或输入数据。

## 阶段性结论

事实：同一代码可在 V100 兼容软件栈读取现有 Parquet，并通过模型梯度/置换、query tiling、训练恢复测试。实际 V100 smoke 和 profile 的结果仍须由 batch 作业给出。

## 下一步计划

完成兼容 GPU 作业并据吞吐冻结科学训练预算；记录 CPU reference 的实际结果；继续主 2×2、分辨率曲线与三种子确认，不以软件兼容检查替代 S2 物理验收。
