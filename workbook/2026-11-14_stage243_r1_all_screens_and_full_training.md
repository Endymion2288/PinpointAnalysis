# R1：全部screen完成，进入完整曝光比较

**阶段目标**

完成预定16项screen，按冻结规则选择每族、每任务的LR并实际启动完整训练。完整四pass排名、神经网络三seed和最终统计尚未完成。

所有输出根目录为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`；screen摘要另保存到`roadmap/v2/r1_screen_results.json`。

**工作内容与详细数据**

每族四项试验共用同一allocation，每项300训练秒，seed42、batch8、相同fixed Tune1k/class；按共同非τ BA3最大、NLL3最小选择。下表仅为选中trial在screen的实测结果，不是完整Tune或最终架构排名。

| Family | Task | Selected LR | Common BA3 | Common NLL3 | Train exposures | GPU |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B1 | T3 | 0.0003 | 0.542333 | 0.915568 | 74,022 | A10040GB |
| B1 | T4 | 0.0003 | 0.546000 | 0.912598 | 78,751 | A10040GB |
| B2 | T3 | 0.0001 | 0.491000 | 1.022665 | 27,815 | H100 NVL |
| B2 | T4 | 0.0003 | 0.465667 | 1.040096 | 29,489 | H100 NVL |
| B3 | T3 | 0.0003 | 0.455667 | 1.045927 | 17,599 | H100 NVL |
| B3 | T4 | 0.0003 | 0.455000 | 1.050650 | 19,024 | H100 NVL |
| B4 | T3 | 0.0003 | 0.540667 | 0.947691 | 48,469 | H100 MIG3g.47gb |
| B4 | T4 | 0.0003 | 0.531000 | 0.956985 | 66,739 | H100 MIG3g.47gb |

全部16项正常完成，未出现nonfinite loss/gradient停止。选中T4 trial的τ效率在1%非τ接受率下分别为B1=0.020、B2=0、B3=0、B4=0.017；screen只有1000个τ及3000个非τ，完整工作点仍待全Tune。B2/B3短预算不足一个完整pass，零尾部效率不表示架构不能识别τ。

完整seed42族序列：B1 Condor1582878、B2 1582877、B3 1582880、B4 Kubeflow `pinpoint-r1-extend-b4-s42-k12`。每序列重新初始化执行T4四pass、T3四pass与fresh T3同allocation总wall-budget控制；B3使用tomorrow/86400秒限制，避免较慢设备在默认8小时截止前未完成。B4 seed17/2026、MAE和无tracker控制亦已获得A100或H100运行。B1无tracker控制1582879已提交。更长B3/B2序列提高了相对JobPrio；正在运行的任务持续执行。等待任务的CPU/RAM按实际两loader加main线程与profile RSS调整，现场修改证据保留在`audit/extension_priorities_a1.json`、`extension_queue_resources_a1.json`。

统一evaluation新增三seed配对均值：同一组400次物理块resamples应用到每个已训练seed，再对三个seed的指标差取均值，给条件于这些权重的事件区间；每个seed差值及sample SD另外报告。main、same-compute和无tracker结果保持各自口径。

τ contamination同时保留实际域混合与明确prior情景：πτ=0.25/0.01/0.001，剩余先验均分e/μ/NC。按各类实际acceptance重权后计算τ污染与coverage；这是假定条件响应固定的benchmark情景，不是物理flux预测。缺少某类观测的域不外推该情景。fixed-FPR报告同时包含实际purity/coverage。

**验证结果**

30项相关测试通过，新增工作点purity与实际selected weights的一致性核对。`smoke/stats_seeds_a2`只复用Train-only模拟validation的相同预测作三个mock seeds，验证两项三seed配对、400次resamples与精确零seed波动；不将它作为真实seed重复。独立权重replay和branch健康的实际结果仍见stage240/242。

成本记账补充处理已被平台清理的Pod：只在Pod已不存活且Job有真实终态时，从cached PodScheduled到Job终态构成保守allocation上界；存活Pod继续按实际生命周期记录，不由Job状态提前置零或结束计费。GPU名称优先使用allocation ID关联的CUDA runtime记录，早期失败缺名时保留未知设备成本。

**阶段性结论**

已测事实：相同短设备时间中，CNN/B4完成更多曝光并取得较高共同三类分数；不同GPU小时不构成FLOP等效。B1的T4条件三类分数高于其T3短screen，尚不支持“去τ一定改善三类”的结论。最终物理判断需完整曝光、同计算和seed重复，当前R1继续进行。

**下一步计划**

完成所有四pass主结果后冻结Scratch排名，仅将前两族与B4扩到真实seed17/42/2026。全部frozen-weight条件与模态诊断、400次paired区间、三类合作组同构对照和原始scores/图表、终态成本与R1验收依次完成；届时给出R2训练规模与三类/四类投入建议。
