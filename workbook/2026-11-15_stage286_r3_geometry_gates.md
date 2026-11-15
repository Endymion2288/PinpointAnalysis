# R3 几何干预的学习、相位与资源诊断

## 阶段目标

确认stride对照可以有效学习、测量完整validation峰值，并检查原生关系在压缩前的索引、测量轴和梯度。正式分类收益仍待固定曝光训练；本阶段不完成R3。

## 工作内容

实现保测量padding相位诊断，B1首层stride4/2真实拟合及同allocation单/双进程profile；实现B2原生候选图、带符号及测量mask的边特征、压缩前消息和匹配控制。32-event cache/recovery合同沿用stage285。另实现显式R3评价合同，复用已核对的400次physical-block配对评价及独立统计复核，保留endpoint主比较和best-screen次比较；三seed平均仅在匹配的17/42/2026全部存在时生成。

候选图不读取truth：同plane跨window、前后三个可见plane，degree≤12；bar使用真实区间和共同测量轴。新增raw坐标差的xy也乘共同测量mask。候选覆盖率与recorded trackID recall分开；后者尚未测量。原图可能因大量bar plane夹在pixel plane之间而缺少远端pixel连接，已提交1024个平衡Train事件的独立审计，不据少数事件终止分支。

## 详细数据

产物根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。G1诊断由HTCondor `1582993.0`、`bigbird20.cern.ch`执行，实际设备H100L-1-12C MIG 1g.12gb、BF16 encoder/FP32 head。提交器记录了实时队列与WRITE/path兼容性。

四项真实拟合均在64 steps达到100%：stride4 T3/T4 NLL为0.0000369491/0.000158652，stride2为0.0000811524/0.000566220；T3为24个非τ、T4为32个真实Train事件。梯度、更新、非恒定概率及finite激活检查均通过，结果在`fit/g1_c1_c{0,1}/`。这回答学习可行性，不是generalization测量。

| Measurement | Stride4 | Stride2 |
| --- | ---: | ---: |
| 256-update training events/s | 123.1194 | 39.8553 |
| Complete operation events/s, including setup | 45.6580 | 26.2910 |
| Training allocated peak MiB | 1084.598 | 3465.779 |
| Full Tune forward events | 40000 | 40000 |
| Full Tune forward seconds | 154.3270 | 421.4318 |
| Full Tune allocated peak MiB | 409.671 | 851.963 |

两模型并行的aggregate operation吞吐41.2143 events/s、总wall99.3830 s，分别运行需122.7526 s；在这个已获MIG上节省19.04% operation wall。并行aggregate RSS峰6871.797 MiB；单进程full validation的aggregate RSS峰5044.266/5151.434 MiB。40k forward身份和finite检查通过，随机初始化分数不用于选择结构。

G0使用R2 D2 T4固定endpoint及256个真实Train事件、7个synthetic点/短线；原forward精确重建，logits最大差0。固定扩展canvas内16种相位均保留全部测量与totals。真实Train中最敏感相位的平均逐事件最大概率差0.0714187、最低argmax agreement85.15625%；synthetic相应0.0254298/71.42857%。这是固定权重对padding采样相位的依赖，不能证明stride2分类收益。详细激活在`diagnostics/phase_g1_c1/`。

M0仍在Kubeflow H100 NVL MIG 3g.47gb训练，读回checkpoint已达499780/1200000曝光、训练秒1201.2848；最终指标未测量。新G2拟合/profile为Condor `1582994.0`；候选图审计为`1582995.0`。两者状态以终态资源表为准。

权威G4源码核对：`PixelSD.cc`对同channel后续hit只累加能量，保留首贡献者trackID；因此审计明确称recorded-first-contributor recall，不称完整真实关联效率。accepted geometry有15个pixel、96个scint plane、108个tungsten中心；silicon/tungsten/scint厚度10/5/5 mm。`DetectorConstruction.cc:ComputeZPositions`给出不均匀z布置。当前G2 prototype尚未加入材料积分特征，其作用未测量。

新增文献[M10](../roadmap/v2/05_literature.md#m10)官方摘要已联网核对。33项R1/R2/R3相关tests通过，包括原生control精确输出、匹配backbone初始化、新关系真实梯度、跨event索引隔离、skip-plane和bar mask。命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m unittest tests.test_flavour_v2_r3 tests.test_flavour_v2_r2 tests.test_flavour_v2_r1 -q
```

## 阶段性结论

事实：stride2能学习，但相同参数下计算及显存明显增加；单卡并行在已测小批次上有收益。固定权重存在相位敏感。原生关系实现通过局部正确性与梯度检查。

待检验：全程重混、减少压缩和原生关系能否改善共同三类、稀疏排序或τ工作点。尚无正式终点证据，不进行结构晋级。

## 下一步计划

M0完成后执行完整Tune配对并确定后续采样；以相同采样冻结G1正式对照。读回G2真实学习/profile及候选审计，必要时修正pixel被bar可见层隔断的问题，再冻结正式本族对照。继续完成local/regional/global可读probe、实际梯度和重训ablation；依预设收益触发种子、分辨率、anchor及fusion扩展，最终核对全部资源终态与R3验收。Cal-A/Cut-B/Confirm保持封存。
