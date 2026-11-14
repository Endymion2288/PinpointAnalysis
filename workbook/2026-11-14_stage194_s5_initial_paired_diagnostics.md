# S5 初始完整评价与辅助学习证据

## 阶段目标

核对统一评价的分母、内部学习与统计含义，得到完整screen前的程序诊断。

## 工作内容

CPU job1581757在bigbird20完成7模型评价：4个首批神经网络及3个HGB，400次accepted physical-block bootstrap，128 blocks（每类32）、10k相同validation事件。评价89.970203 s，产物`evaluation/initial_diagnostic/`；原生事件图也已完成。人工查看τ效率–purity和visibility图，计数与表一致。

评价增强保留单模型τ分道block区间、charm decay-branch假τ表，并明确validation选择阈值上的Clopper–Pearson仅为描述性计数区间。语义/charm的预测方差改为概率空间，另存logit方差。绘图将4-pass/8-pass分开，增加四分类计数及按真实能量/位置的component residual图。

## 详细数据

| Variant | NLL | macro AP | Tau AP | Tau efficiency at 1% background |
| --- | ---: | ---: | ---: | ---: |
| direct4 | 1.240712304 | 0.447628556 | 0.336006743 | 0.0240 |
| six | 1.237478997 | 0.450023980 | 0.334041026 | 0.0288 |
| vertex | 1.240485224 | 0.447285940 | 0.335054200 | 0.0288 |
| semantics | 1.237785574 | 0.451110705 | 0.336566020 | 0.0304 |
| HGB moments | 1.253048883 | 0.436265490 | 0.323559199 | 0.0264 |
| HGB moments+objects | 1.247475827 | 0.440891270 | 0.325976501 | 0.0216 |
| HGB objects | 1.338692566 | 0.347610249 | 0.275754727 | 0.0124 |

Six减direct4 NLL=-0.003233307，block CI95[-0.005621228,-0.000844770]；macro AP差0.002395425、CI[-0.000543340,0.005380562]。Semantics NLL差-0.002926730、CI[-0.004832435,-0.001175452]，macro AP差0.003482150、CI[0.001674292,0.005446405]。这是选中seed及固定权重条件区间，尚未完成训练seed确认。

Vertex辅助在9,718个有监督validation事件上，normalized xyz MSE分别0.027118087/0.062945845/0.001983572；train-only constant mean对照0.295032461/0.295125485/0.012598701。预测std0.515671/0.485965/0.105647，目标std0.543012/0.543161/0.112232。Head已学会非恒定的顶点预测；其分类NLL差-0.000227080、CI[-0.002026486,0.001632570]，两件事分开判断。Semantics soft CE0.600373884，对照0.712913192。

Direct4在1%背景接受度选择75/7,500个背景及60/2,500个τ。无daughter response的τ为0/873，固定阈值描述性CI95[0,0.004216606]；有response为60/1,627（3.68777%），CI[2.82573%,4.72162%]。该分层仅描述此模型与accepted proxy，不能解释成硬件不可分辨性的上界。τ→e/μ/had分别5/444、6/395、49/1,660；unknown1个单列。

当前对象vertex candidate覆盖793/10,000；primary reconstructable806个中29个满足xy<1 mm且|dz|<10 mm，效率3.5980%、block CI[2.5087%,4.8707%]。τ双prong eligible31个中2个匹配，6.4516%、CI[0,16%]；该对象算法尚未形成可靠的secondary vertex重建。

资源历史缺项已定位为Condor late materialization：v2计划160行、`max_materialize=24`，历史24个ProcId=0–23均removed且wall0；proc0保存`JobMaterializeNextProcId=24`、`NumJobStarts=0`、用户condor_rm原因。剩余136行没有实例化，不能要求它们有独立作业历史，也不计成运行失败。证据`runtime/s5-truth-v2.proc0.full.json`及24条历史；accounting按实际materialized数核对。

## 阶段性结论

评价入口与真实辅助学习正常。初始seed42中six/semantics的概率误差有小幅改善，τ排序净收益未建立。主要困难是未观察到τ daughter response的域与低纯度pixel候选；它们分别需要物理接受度分析与算法改进，不能合并成“τ无法识别”。完整screen和三seed仍待完成。

## 下一步计划

完成剩余坐标对照、执行冻结selection、三seed确认，再将全部同事件结果用于最终困难通道解释与S5验收。
