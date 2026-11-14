# R1：B4筛选完成与共同参考扩展

**阶段目标**

按冻结合同完成首个神经族的学习率选择，衔接完整曝光与共同参考控制，并明确复用SSL和交互验证的成本口径。完整架构比较仍待其余模型、四pass、三seed与配对评价完成。

产物根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。本阶段未访问Cal-A、Cut-B或Confirm payload。

**工作内容与详细数据**

Kubeflow `pinpoint-r1-screen-b4-k11`已成功终止。四项B4训练共用H100 MIG3g.47gb，每项300训练秒；下表均为固定Tune1k/class，其中共同非τ为3000事件。它们不是全Tune最终表。

| Task | LR | Actual Train exposures | Common BA3 | Tau efficiency at 1% FPR |
| --- | ---: | ---: | ---: | ---: |
| T3 | 0.0003 | 48,469 | 0.540667 | Not defined |
| T3 | 0.001 | 65,614 | 0.538667 | Not defined |
| T4 | 0.0003 | 66,739 | 0.531000 | 0.017000 |
| T4 | 0.001 | 68,934 | 0.513667 | 0.017000 |

按共同BA3最大、再按NLL3最小的冻结规则，两任务均选 **LR=0.0003**。首次编译、数据加载和warm cache影响实际曝光；不将这些短预算结果解释为纯LR效应或架构终点。完整训练会从同一初始化重新开始，给每个非τ事件四次曝光。

B4正式seed42在Kubeflow `pinpoint-r1-extend-b4-s42-k12`继续执行T4、T3及fresh T3总wall-budget控制。Condor已提交共同参考seed17/2026（1582873/1582874）、B4-MAE T4（1582875）与无tracker T3（1582876）。B1/B2/B3 screen继续分别使用A10040GB、H100 NVL、H100 NVL；不限定exact型号或整卡。正式扩展记录精确allocation ID及GPU/精度，按各族完成时间衔接，不等待全部族一起启动。

固定1600-event stress准备已完成：400/class，耗时453.038秒。nominal tracker从ideal crossings按相同family RNG重数字化，与accepted nominal逐项精确相同；combined使用gain0.9、bar threshold2PE、pixel survival0.98、bar survival0.98、tracker efficiency0.97、alignment(+0.1,−0.1)mm、tracker sigma scale1.2，并叠加accepted sparse noise。保留全部事件和分母；未使用新的heldout pattern。B0的10项nominal/noise/combined/去pixel/去tracker推理也已完成，实际配对统计与科学解释在完整评价给出。

独立的八组Train-only checkpoint replay完成：B1/B2/B3最大概率差均为0；B4 T3/T4分别为2.98023×10⁻⁸ / 1.19209×10⁻⁷。证据为`smoke/a1/checkpoint_readback.json`。这一步读取独立保存权重，不仅是在原训练进程中继续预测。

新增`reference_costs.json`单列复用MAE3的既有成本：源checkpoint为S4 `s4_mae_m75_e3_s42/best.pt`；H100 MIG3g.47gb，98,000个SSL Train events、2,000个inner reconstruction probe events，18,432 optimizer updates，operation **2,287.156994秒=0.635321 device-hours**，峰值818.441 MiB。其共享historical allocation还运行其他实验，不能把整个allocation都归给本权重；历史SSL不记为新R1支出，但模型总方法成本单列。

成功交互拟合、Train-only runner与独立replay的已记录operation合计 **1,153.285803秒**，是共享交互GPU上的测得耗时下界；launcher、setup和未记录失败wall time尚未测量，不将它冒充完整allocation-hours。全部batch setup、编译、失败与成功成本继续由`resources.json`统计；缺少早期GPU名称的失败保留“unknown”成本，不置零。资源收集现可从唯一profile命令与实际manifest补齐设备来源。

**验证与阶段性结论**

完整screen与扩展命令、精确源码归档、选定配置和row exposures保存在`runtime/`、`code/`、`selected_recipes.json`及各experiment目录。实际学习健康与checkpoint replay通过；真实四pass泛化、三seed波动及完整终态成本尚未测量。

已测事实：B4当前短预算内τ工作点较弱，已有三类分数具备区分力。四类任务是否值得保留、哪个架构最强，须由完整曝光、同计算和seed重复回答。短screen只用于既定学习率选择，不能据此排除更慢架构或宣称达到最佳效果。

**下一步计划**

完成其余12项screen，逐族衔接T3/T4四pass。B4共同参考并行推进；全部Scratch主结果齐备后按冻结排名扩展另两族seed17/2026。完成全部诊断、400次paired物理块区间、学习曲线/ROC/PR/raw-score/资源图及R1验收。
