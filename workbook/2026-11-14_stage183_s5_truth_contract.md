# S5 标签、衰变顶点定义与正式批处理入口

## 阶段目标

建立独立truth sidecar和mask合同，避免τ衰变树坐标解释错误污染S5。沿用用户接受的样本、split和detector-proxy。本阶段不重做样本认证。

## 工作内容

新增`s5/truth.py`、`prepare.py`和`submit.py`。源码确认`FillPythiaTree`的`DecayXYZ`为子轨迹最大z位置；τ衰变点采用forward neutrino共同产生点，同点子粒子定义prong与分道。独立记录原S4分道、未知mask、charm覆盖、hit visibility、visible primary-system proxy和bar semantic fractions。元数据不进入observation。

正式提交保存Git archive，AFS只放小wrapper/job.sub，所有日志和结果写EOS。每次提交检查schedd负载、EOS布局和WRITE权限；Kubeflow使用product+GPU resource flavor及Pod server admission。研究合同见[路线图](../roadmap/archive/2026-11-14_v1/12_s5_physics.md)。

## 详细数据

输出根`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14`。目录初始化得到110,000事件、160个run/chunk任务，40个原生shards。原partition为100,000 train和10,000 validation，各四类均衡。Calibration/test未使用。

命令在LCG108 CUDA/Python3.12环境执行：

```bash
python -m unittest tests.test_flavour_s5_truth
python -m analysis.flavour.s5.prepare initialize --root <S5_ROOT> --source <S4_ROOT>
python -m analysis.flavour.s5.prepare create --root <S5_ROOT> --task 80 --limit 16
```

3项物理合同测试通过，覆盖电离电子/其他parent排除、backward-neutrino无效mask和charm PDG。真实`10083_010.root`前16个选中事件：16个有效τ decay，分道与catalog差异0，1个charm事件。读取/生成耗时0.824457 s。首次Parquet写入发现LCG Arrow未编译zstd，改用可用gzip后成功。正式分块全量、其他三类和对象重建尚未测量。

网络查询已获取FAMO、CAGrad、CVPR2018 uncertainty、Mask3D及PTv3页面；方法适配选择与FASERCal来源记录在路线图。GPU未分配。Kubeflow凭据已恢复，读取成功。

## 阶段性结论

已用源码和16个真实τ事件支持修正后的衰变位置定义。该smoke仅支持标签实现的局部正确性；完整标签分布、visible/invisible比例及对象效率尚未测量。Visible target明确是primary-system proxy，不冒充稳定末态接受能量。

## 下一步计划

HTCondor生成160个分块sidecar；合并核对event key和mask计数，完成对象关联/vertex coverage后启动辅助任务对照。保留原生raw路径，按S5合同评价负迁移和τ轻子分道。
