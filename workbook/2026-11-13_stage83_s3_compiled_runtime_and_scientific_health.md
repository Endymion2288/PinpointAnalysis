# S3 编译执行、健康观测一致性和科学验收

## 阶段目标

缩短完整事件重训耗时，并验证内部观测不改变训练计算。四分类评价必须独立要求有效学习，不能以checkpoint读回代替。

## 工作内容

全量训练入口增加匹配架构、表示、编译版本的32-event pilot和完整占用压力测试。八种screen配置均独立验证，不以主结构gate替代coarse/tracker/no-direct的拟合。冻结8个screen与12个confirmation，`s3_repair_all.json`仅合并配置便于单GPU顺序运行；科学对照仍是14项单seed与9项三seed。

新增`inspect_repair`检查所有计划内部采样、实际encoder梯度、best checkpoint的独立CPU FP32 backward，以及全validation NLL相对constant prior的学习。`accept_repair`将内部健康与有效学习加入必需条件；`finalize_repair`执行完整读回、四分类bootstrap、条件域/工作点、压缩统计、错误显示与英文健康图。共享/混合backend成本不用于声称模型本体加速。旧配置中残留的“已验收”当前状态改为历史故障记录；`s3.yaml`指向revision2。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13`。

最初4个eager训练在Kubeflow H100 47GB MIG并发启动，每模型约26–32 events/s。四者均未完成第一个epoch；停止时输出与配置已移到`audit/eager_execution_before_compile`，不作为科学结果。停作业容器终态已保存；child结束时间使用容器termination上界，明确标记未观察到child exit code。所有占用时间计入成本。

Kubeflow第二个GPU作业因namespace quota=1 GPU unit被平台拒绝，无第二个训练Pod获配。HTCondor控制组1574068提交到`bigbird20.cern.ch`：空闲队列为0、直接EOS policy=True、WRITE授权ALLOW；其他候选`bigbird21`的NoEos policy不兼容。一个H100上合并4个训练。`better-analyze`显示无愿意运行的匹配slot；停止eager配方时该作业仍Idle，已移除，未消耗remote GPU时间。AFS只存小提交脚本，产物均EOS。

在相同权重、相同8个train事件（原始通道数[10,59,557,557,1589,2831,6685,36549]）上，编译局部编码与计算chunk测试：

| Execution | Warm repetitions, forward+backward s | Peak allocated MiB |
|---|---|---:|
| eager chunk256 | 1.64258, 1.63737, 1.93076 | 244.374 |
| eager chunk2048 | 1.13632, 1.22160, 1.30823 | 1056.329 |
| compiled chunk256 | 0.53193, 0.58825, 0.50560 | 231.042 |
| compiled chunk2048 | 0.32147, 0.35406, 0.33735 | 772.402 |

这是同设备、并发背景仍在运行的固定压力批次测量，不是全样本吞吐或最终模型效率结论。首个编译warmup约37.92 s，不计入warm repetitions但计入allocation。编译区内部结果受浮点归约顺序影响，不要求bitwise相同。配置显式记录`compile_local=true, local_chunk_blocks=2048, local_execution_version=2`，worker数改为1；所有科学模型统一，从头拟合。该chunk是执行分批，不改变每个局部块最多32个节点及任何物理窗口/region。

最小Alma9镜像缺少Triton、C headers和linker支持。复用LCG108的Torch2.7.1，匹配Triton3.3.1只安装worker scratch；公用EL9头文件约3.48MB和linker文件约99kB放EOS，worker解压到scratch。`prepare_repair_runtime.py`可重建这些资产，`runtime_repair.sh`提供一致执行环境；不向Git写二进制依赖。

编译pilot版本1中，fixed虽多次100%拟合，末次step768降到accuracy0.84375/NLL0.612442；该gate判失败。程序检查发现health采样切换回eager路径，观测改变了训练实现。将local pool hook移到编译特征编码外后，版本2健康与普通更新使用同一路径，三个主pilot重跑通过：fixed和multiscale末次accuracy1/NLL显示−0.0；fresh ParT accuracy1/NLL0.0000100847。失败证据在`audit/compiled_pilot_execution_switch`，费用照计。

补充5个表示/结构的版本2先导均为accuracy1、32组不同分数，并通过所有健康要求：

| Configuration | Final train-probe NLL | Final probe wall s |
|---|---:|---:|
| fixed 1 mm | −0.0 | 106.928 |
| multiscale 1 mm | 0.0000127558 | 106.030 |
| multiscale tracker | 0.000000506640 | 106.349 |
| multiscale no direct | −0.0 | 105.607 |
| multiscale region8 | 0.000282117 | 105.903 |

八份`pilot/<name>/gate.json`全部通过；完整编译压力测试正在运行，正式全量训练要求其终态通过。小样本拟合不能替代泛化评价。

健康图`audit/current_health.pdf`已人工查看multiscale页：归一化分支RMS约1、Head第一Linear RMS约5，Head LN后约1，上游梯度非零；同region的queries cosine接近1、variation随训练减小，需在完整学习后与压缩控制一起解释。此前query cosine矩阵乘法受外层BF16 autocast影响，现显式FP32，避免把舍入到1当作完全相同。Attention entropy只描述第一group至多8 queries/128 keys的条件分布，不能视为全事件entropy。

验证：35项focused tests通过，8.212 s；覆盖模型、训练resume、四分类决策、共享allocation、空输入与内部断开。`compileall`、`bash -n runtime_repair.sh`和`git diff --check`通过。11:29:48 UTC刷新时本轮allocation0.676747730 device-hours、累计S3上界31.915016408、余额下界88.084983592，missing history=0；压力测试仍运行，成本需终态再刷新。

## 阶段性结论

事实：八个冻结表示/结构的修复模型均能拟合真实train先导；编译执行减少固定压力批次耗时；健康采样切换执行路径的问题已修复。未测量：完整重训的validation指标、三seed效应、native/tracker/压缩增量及其CI。

## 下一步计划

等完整编译压力测试通过后，在单GPU中并发执行8个screen，再执行12个事先选定的确认任务。每波完成先做读回和内部健康检查；异常中止后审查修复，不能归纳为物理信息无效。完成全部四分类、条件域及健康验收后写新的方法结论；旧结果继续只作故障诊断。
