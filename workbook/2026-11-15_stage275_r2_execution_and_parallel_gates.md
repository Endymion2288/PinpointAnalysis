# R2 首批合同、真实运行门槛与异构 GPU 并行

## 阶段目标

落实stage264的单族D0/D1/D2合同，分开unique事件与重复曝光的影响。先验证真实学习、完整训练器和恢复，再启动六项科学训练；本报告记录实现与启动，R2科学验收仍待完成。

## 工作内容

新增`analysis.flavour.v2.r2`。`prepare`复用R1的100k Train和40k Tune原始observations，只转换新增冻结200k；`campaign/schedule`冻结六份配置并metadata重放实际sampler；`train`以完整预算U定义学习率，记录实际逐事件曝光、滚动类比例/logit/NLL、分支梯度/更新/激活和固定Tune screen。`profile/group`在一个实际allocation中运行独立训练进程，共享一次压缩输入staging，各自保留bounded decoded LRU。

全部fresh seed42、B1 ResNet width16、batch8、LR .0003、AdamW weight decay .01、clip1，BF16 encoder/FP32 head；每任务5% warm-up、cosine、5% floor按其完整U参数化。无wall驱动退火。D1不从已经退火的D0继续。GPU型号不设同卡限制，MIG可用；BF16要求兼容指令集，SM75的FP32配方另立匹配D0，不静默改变本组数值合同。

每75k非τ曝光评价R1固定4k Tune screen；主结果使用完整预算终点，best-screen以本任务NLL最小选择并单列。完整40k Tune只用于两种checkpoint终点评价。新增恢复状态包括optimizer、CPU/CUDA/numpy RNG、sampler位置、逐行计数和未完成滚动窗口；独立DataLoader generator避免恢复时worker创建消耗dropout RNG。

新增`smoke`，在32个冻结Train事件上构造人工validation，实际运行screen、best、endpoint、预测表和中途恢复，未用于泛化结论。`evaluate`实现12个checkpoint结果、11域、72项配对和400次physical-block bootstrap；`gap`冻结共同Train100k内每类1k的eval-mode诊断。二者的正式科学结果尚未产出。独立核心区间读回新增occupancy域支持，保留既有R1算术。

## 详细数据

结果根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/`。R0根目录为其父目录，R1位于`../R1/`。

### 数据与sampler

8个HTCondor转换任务1582949–1582956各处理10个新shard，4 processes×2 threads；原56个shard原样symlink复用，新增80个shard。finalize作业1582959核对通过：300k Train+40k Tune，共340000事件，冻结event/family身份和标签一致，PE守恒。完整证据`preparation.json`、`data/plan.json`及各新shard summary。六份sampler预检由1582958、1582960完成，保存JSON和逐update NPZ。

| Configuration | Unique Train | Updates U | Exposures | Non-tau exposures | sum(lr) |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 T3 | 75000 | 37550 | 300000 | 300000 | 5.900605657864 |
| D0 T4 | 100000 | 50051 | 400000 | 300000 | 7.865059679071 |
| D1 T3 | 75000 | 112645 | 900000 | 900000 | 17.701315577075 |
| D1 T4 | 100000 | 150153 | 1200000 | 900000 | 23.595463349912 |
| D2 T3 | 225000 | 112630 | 900000 | 900000 | 17.698958418677 |
| D2 T4 | 300000 | 150117 | 1200000 | 900000 | 23.589806175917 |

每行在metadata重放中精确出现指定passes。D1与D2同曝光，尾批分别造成15/36步差异，不能称为精确fixed-update比较。batch-size分布、窗口类别、末100batch和lag1详见`preflight/`；小型合同和测量表已复制到`roadmap/v2/configs/r2/`及`r2_execution.json`。

### 单卡并行实测

K2使用H100 NVL MIG 3g.47gb、BF16、Train-only真实forward/backward/AdamW，每child512updates、4096曝光，按1/2/4/1顺序执行，全部exit0。

| Phase | Concurrent | Phase wall (s) | Aggregate operation events/s | Sum child train events/s |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 21.553595 | 190.037903 | 251.722216 |
| 1 | 2 | 31.755413 | 257.971766 | 313.549100 |
| 2 | 4 | 52.614382 | 311.397745 | 354.829833 |
| 3 | 1 | 21.458206 | 190.882687 | 250.905800 |

前后单任务一致，四并行在这项短测中总operation吞吐约为单任务的1.63倍；各child GPU peak1084.956MiB、aggregate host RSS约5.04GiB。K1的256步首phase有明显启动差异，因此不采用其表观3倍加速。正式四并行启动后cgroup读数24,237,424,640 bytes、limit62,999,998,464 bytes，须继续检查后续validation loaders的峰值。Kubeflow metrics API无读取权限，使用实际cgroup与进程监测。

Condor的A100-PCIE-40GB两并行profile亦成功。交互T4/FP32的128步短测，总operation吞吐1/2并行为21.906/36.555events/s，说明旧卡也有并行空间。它们未代替同配方BF16科学对照。所有profile保存于`audit/parallel_profile_{k1,k2,c1,interactive_t4_a1}/`。

### 完整流程与真实学习

`smoke/d01_k3/integrity.json`：T3/T4各9updates，在第4步中途保存并重新启动。连续与恢复最终parameter最大差值均0，endpoint/best预测表和曝光表一致；T3为18行×3passes，T4为24行×3passes，人工validation均8行，screen次数分别6/5。实际Tune未使用。

`fit/d01_k3/`：B1 BF16真实Train-only拟合T3/T4均在64updates达到accuracy1.0，NLL分别0.000038002112、0.000162464901；分支梯度、参数更新、激活finite和非恒定输出检查通过。

交互T4第一次FP32流程检查未满足精确恢复容差：parameter最大差1.71177e-6、概率差4.17233e-7、logit差1.81794e-6。曝光与执行已完成，尚不能将微小数值差异归为恢复故障。投影CUDA `index_add_`浮点累加是待核对原因；另加仅用于smoke的deterministic algorithms与CUBLAS workspace后，T3/T4重测最大parameter差均0，预测/曝光表一致；这支持原微小差来自非确定性数值执行。正式BF16合同与已通过结果未改变。结果位于`smoke/interactive_fp32_a1/`、重测`interactive_fp32_a2/`。

6项有针对性的单元测试通过（0.026s）：完整pass/尾批、按完整U参数化LR、恢复剩余batch顺序、endpoint/best配对隔离、occupancy边界和缺类bootstrap。`compileall`通过。正式评价/Train gap/独立区间仍待真实产物验证。另用Train-only人工validation构造12个checkpoint视图、72配对、11域的完整评价smoke，覆盖稀疏domain中bootstrap抽样后缺类的情况；发现原scalar_summary直接读取不存在工作点会报KeyError，已改为对无分母的recall/工作点返回NaN，最终JSON为null，不把无支持估计当0。既有有支持样本的公式保持一致。该评价smoke的独立point checks通过1152项，core interval checks通过1020项，72项配对均核对通过。

### 已启动的科学任务

| Backend | Attempt | Actual GPU | Concurrent models |
| --- | --- | --- | --- |
| Kubeflow | pinpoint-r2-d01-k3 | H100 NVL MIG 3g.47gb | D0/D1 × T3/T4，4进程 |
| HTCondor | 1582961 / r2_d2_c2 | H100L-2-24C MIG 2g.24gb | D2 × T3/T4，2进程 |

Condor每次提交前重新读取schedd排队、WRITE和EOS策略，选择兼容且负载最低的bigbird20，依据写入`runtime/*.submission.json`。科学任务各自唯一writer，源码使用提交时不可变archive；日志、checkpoint、预测保存在results。训练初段滚动NLL已经变化；完整40k Tune指标、收敛判断、终态wall与device-hours尚未测量。

可复现入口（先按AGENTS.md启用LCG108并设置`PYTHONPATH=src`、线程数1）：

```bash
python -m unittest tests.test_flavour_v2_r2
python -m analysis.flavour.v2.r2.campaign preflight --root "$R2_ROOT" --configs CONFIG.json
python -m analysis.flavour.v2.r2.profile --root "$R2_ROOT" --r1 "$R1_ROOT" --attempt UNIQUE --counts 1 2 4 1 --updates 512
python -m analysis.flavour.v2.r2.group --root "$R2_ROOT" --name UNIQUE --parallel 4 --smoke --real-fit --configs CONFIG1.json CONFIG2.json CONFIG3.json CONFIG4.json
python -m analysis.flavour.v2.r2.evaluate --root "$R2_ROOT" --replicates 400 --workers 4
```

已冻结的preflight和attempt拒绝覆盖；历史准确提交命令、资源申请和code archive位置在runtime记录。GPU成本按allocation计算，单卡四进程只计一次device-hours；不同MIG/整卡小时不作FLOP等效。

## 阶段性结论

**事实：**数据扩展与六份准确预算已完成；BF16真实学习和恢复门槛通过；单卡独立多进程在实际profile中提高总吞吐。六模型已分配到两种兼容MIG并开始执行。Cal-A/Cut-B/Confirm payload继续封存。

**待检验：**D1更多曝光是否提高Tune、D2更多unique是否优于D1、低occupancy μ损失是否随训练规模改变，以及τ工作点和共同三类质量是否能同时改善。当前运行状态不足以决定收敛或全局最优。

## 下一步计划

完成六模型endpoint和best-screen完整40k Tune评价，72项配对采用400次共同physical-block抽样，并独立读回概率/计数、核心指标和区间。用共同Train样本eval-mode结果分析Train/Tune gap、最后三个screen的NLL变化、逐类低occupancy和分支健康。

按实际结果决定D3=300k×12、D4=1M×4、width32或第二原生族的必要对照，保留明确问题、fresh初始化、匹配数值配方和实测成本。完成终态资源统计、英文图表、科学结论及独立验收后才能完成R2；R3尚未启动。
