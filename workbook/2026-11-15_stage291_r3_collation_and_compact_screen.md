# Stage291：R3精确组批加速、共享缓存与紧凑图像screen

## 阶段目标

降低原生关系模型的执行开销，完成正式G2配方冻结，并落实v2的紧凑sensor-row候选。验收分别要求输入逐位等价、真实学习、确定性恢复和完整Tune资源检查；训练收益在正式终点形成后评价。

## 工作内容

旧R3组批对每个block扫描整个事件，padding后又为边索引重复扫描。新实现使用稳定的组内顺序和直接scatter，保持32×18 padding、mask、事件/块编号、tracker及所有relation索引完全相同。R1/R2历史组批实现保留，供独立核对。

新增allocation-local mmap缓存：worker返回的批次直接写入scratch连续数组，训练进程读取只读视图；进程间序列化只传小型metadata及路径。共享Train/Tune缓存按完整身份行和实际输入配置核对。Tune screen的局部index由原native index映射，避免截取screen后错误读取缓存行。R3恢复检查增加这一实际共享路径；结果须由GPU作业读回。

正式G2冻结三个100k×4、T4、seed42模型：本族control、压缩前relations、重训去global stack。保留M0决定的spatial sampler，三项各50,051 updates、400,000 exposures、LR sum 7.865059679070928；每行实际曝光要求4次。共同候选数组可共享，control模型不读取边；共同构图成本与各模型forward成本分别记录。

C1按accepted geometry生成111×420的sensor-row表示，物理基线为1322×420。保持每个横向1mm bin、view及响应值的顺序，显式提供真实z、前后间距、pixel/bar类型五个静态通道；ResNet首层stride2。它是表示组合screen，不能把差异解释为单一stride收益。参照R2 D0的同100k×4 T4；配置已在本候选首次Tune访问前冻结。

## 详细数据

输出根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`，下表均使用Train或原冻结Tune，Cal-A/Cut-B/Confirm保持封存。

| Evidence | Actual measurement |
| --- | --- |
| Train10k组批，Condor1583006.0 | 1,250批、全部collated tensors逐位相同 |
| 相同批次CPU组批耗时 | 原实现82.592662 s；新实现4.286592 s；19.267674倍 |
| 上述实际batch尾部 | max33,075 nodes，398,455 edges |
| 上述aggregate RSS峰值 | 8,584.707 MiB，含构建worker |
| 完整Tune资源probe，1583000.0 | control与relations各40,000事件，finite且覆盖完整；862.019328/879.895736 s |
| 完整Tune显存峰值 | 1,086.747/1,136.014 MiB；旧revision2，只作相同执行图的资源证据 |
| 3g.47gb共享allocation上的c7，单项operation吞吐 | control24.153392、relations24.481977、去global28.479608 events/s |
| c7三模型并行operation吞吐 | 43.046863 events/s；aggregate RSS13,139.254 MiB |
| c7运行条件 | G1正式CNN同时使用同一MIG；不是独占设备benchmark |
| revision3真实学习 | 前阶段五项/30检查全部通过，见stage290 |
| 自动检查 | 新增完整组批等价、乱序/空事件、mmap只读/序列化、sensor映射；40项测试通过 |
| C1完整真实映射/拟合/资源/恢复 | 已实现并安排验证，本报告形成时未全部完成 |
| 全100k mmap共享运行峰值/正式G2效果 | 未测量 |

精确组批证据：`audit/collate_c1.json`；完整Tune资源：`audit/profile_g2_c4/results.json`；同卡并行：`audit/profile_g2_c7_on_g1/results.json`。c7在原`pinpoint-r3-g1-k1` allocation内执行，worker wall不另加device-hours；完整终态成本由阶段账本计算。独立大资源c6探针在NumJobStarts=0时结束排队，allocated device-hours=0，见`audit/parallel_c6_superseded.json`。

正式任务已提交到实时查询后选出的`bigbird20.cern.ch`：control1583008.0、relations1583009.0、global-ablation1583010.0。各自允许所有兼容SM80+、10GB以上GPU，申请15GB RAM、4CPU，站点可按内存提升CPU。每项先完成实际共享Train/Tune mmap恢复，再在同allocation进行正式训练；不是仅提交即通过验收。CPU sensor映射审计为1583007.0。新源代码包含40项通过的测试。

主要命令：

```bash
python -m unittest tests.test_flavour_v2_r3 tests.test_flavour_v2_r2 tests.test_flavour_v2_r1 -q
python -m analysis.flavour.v2.r3.collate_audit --root "$R3_ROOT" \
  --config "$R3_ROOT/configs/probe_b2_relation_t4_s42_c5.json" --attempt c1 --cache-workers 4
python -m analysis.flavour.v2.r3.campaign G2 --root "$R3_ROOT" \
  --mixing spatial --cache-workers 4 --reason '<stage291 evidence recorded in frozen JSON>'
python -m analysis.flavour.v2.r3.campaign C1 --root "$R3_ROOT" \
  --mixing spatial --cache-workers 4 --reason '<stage291 representation contract>'
```

实际完整reason、UTC、配方和preflight见`contracts/g2_freeze.json`、`contracts/c1_freeze.json`和`preflight/`，仓库配置副本见`roadmap/v2/configs/r3/`。

资源检索补充：[CERN GPU文档](https://batchdocs.web.cern.ch/gpu/index.html)，2026-11-15访问。当前部分看似空闲的H100节点受`SCond_vo_filter`限制，不能把公开空闲数当作当前用户可用数。继续接受兼容的混合型号、MIG、共享卡，按实际可访问资源和测得吞吐布局。

## 阶段性结论

事实：CPU组批在同10k真实输入上加速19.27倍，所有模型输入不变。较大MIG在已有G1共用时，独立三模型并行获得更高合计吞吐；1g上的旧双并行没有同样收益。二者分别回答CPU路径和实际调度问题，不能将CPU倍数当作训练倍数。

假设：共享mmap可降低正式100k多模型重复缓存的RAM和构建成本；当前有格式及逐tensor检查，完整运行成本仍待测量。C1压缩空行的训练效率与分类效果同样待测量。

## 下一步计划

1. 读回三种G2实际mmap恢复，提交最早可用兼容GPU上的正式训练；更大allocation仅在实测完成时间有优势时迁移。
2. 完成C1真实响应映射、T4拟合、完整Tune资源及恢复，随后运行冻结100k×4 screen。
3. G1/G2/C1完整终点评价、local/regional/global/tracker probes及实际梯度，与重训global ablation共同形成R3决策；再按预定门槛决定seed/分辨率/anchor/fusion扩展。
