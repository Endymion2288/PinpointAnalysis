# R1 三类合作组对照

**R1已完成真实三类重训及相同事件上的四类条件评价。完整41模型、133配对、21项三seed平均配对、11域及452项验收均通过。本页交付W08的三类同类方法对照；原合作组研究的严格复现所缺输入另列。** 最终证据见[stage274](../../workbook/2026-11-15_stage274_r1_final_results.md)及[r1_results.json](r1_results.json)。

## 已完成的对照范围

所有R1模型使用相同accepted readout、Train100k四类池及Tune40k。神经网络主T3实际监督75k unique非τ事件，每个事件曝光4次；主T4监督100k事件，包含完全相同的75k非τ事件且曝光4次。HGB共用相同unique事件，boosting预算另列。Compute T3另按本族T4操作wall配对，实际曝光分别报告。Tune的三类评价使用相同30k非τ事件，四类模型的条件概率为`p(e,mu,NC)/(1-p(tau))`。另将10k真τ输入T3统计污染，未把τ重标成NC。

R1的benchmark三类prior为1/3、1/3、1/3。几何域内重新等类加权；人数、raw confusion与weighted confusion分别保留。工作点在该开发域的raw scores上描述性选取，非Cal-A/Cut-B冻结工作点。完整uncalibrated ROC/PR、固定1%/0.1% FPR及80%/90% purity效率与coverage由R1评估文件提供。

| Domain | Raw e / mu / tau / NC | Non-tau events | Selection |
| --- | --- | ---: | --- |
| All | 10,000 / 10,000 / 10,000 / 10,000 | 30,000 | 全部accepted Tune |
| Pixel-centered | 930 / 1,050 / 757 / 1,028 | 3,008 | pixel边界内缩20 mm，target前500 mm |
| Large scintillator | 4,424 / 4,764 / 4,136 / 4,678 | 13,866 | scintillator边界内缩15 mm，排除target末端200 mm |

域边界来自本批geometry，target z为[−2993.00,−1804.88] mm；具体实现为`analysis.flavour.s2.prepare.regions`，冻结说明为`R1/data/plan.json`。边界文字与Mahlstedt报告对齐，原研究的事件与坐标实现尚未逐项核对。

## 三类实际结果

以下为seed42、四pass的实测值，按e/μ/NC顺序。`T4 conditional`保留四类训练，评价时在非τ上条件化。B1为ResNet，CNN为同一投影/融合输入的低成本网络；without tracker通过实际重新训练得到，使用与B1 seed42相同的FP32配方。各项实际GPU不同，wall不能用于FLOP等效排名。

| Domain | Model | BA3 (%) | AUROC e / mu / NC | Efficiency at 90% benchmark purity (%) e / mu / NC |
| --- | --- | ---: | --- | --- |
| All | B1 T3 | 56.620 | .8151 / .7594 / .7339 | 24.52 / 17.61 / 10.46 |
| All | B1 T4 conditional | 56.493 | .8169 / .7595 / .7369 | 24.78 / 16.94 / 10.80 |
| All | B1 T3 without tracker | 52.693 | .8070 / .7311 / .7017 | 22.98 / 12.66 / 9.56 |
| All | Plain CNN T3 | 54.750 | .8039 / .7265 / .7139 | 21.20 / 11.34 / 0.54 |
| All | B4 T3 | 54.350 | .8065 / .7297 / .7185 | 21.55 / 10.02 / 5.96 |
| All | HGB T3 | 54.590 | .8025 / .7288 / .7171 | 22.49 / 12.72 / 2.88 |
| Pixel-centered | B1 T3 | 73.817 | .9203 / .9089 / .8644 | 59.35 / 52.76 / 29.09 |
| Pixel-centered | B1 T4 conditional | 74.185 | .9224 / .9099 / .8658 | 51.83 / 50.86 / 34.63 |
| Pixel-centered | B1 T3 without tracker | 68.243 | .9144 / .8620 / .8249 | 53.44 / 31.33 / 26.65 |
| Pixel-centered | Plain CNN T3 | 72.611 | .9118 / .8805 / .8423 | 55.38 / 47.90 / 3.79 |
| Pixel-centered | B4 T3 | 72.944 | .9165 / .8917 / .8512 | 52.58 / 48.57 / 25.49 |
| Pixel-centered | HGB T3 | 72.767 | .9136 / .8914 / .8458 | 49.35 / 51.05 / 19.75 |
| Large | B1 T3 | 68.891 | .9106 / .8593 / .8211 | 45.82 / 32.05 / 24.43 |
| Large | B1 T4 conditional | 69.038 | .9124 / .8594 / .8235 | 47.22 / 32.03 / 24.84 |
| Large | B1 T3 without tracker | 64.350 | .9065 / .8250 / .7909 | 43.72 / 24.08 / 20.91 |
| Large | Plain CNN T3 | 65.995 | .8996 / .8175 / .7931 | 43.49 / 22.50 / 0.81 |
| Large | B4 T3 | 66.158 | .9029 / .8246 / .7997 | 42.95 / 21.96 / 12.80 |
| Large | HGB T3 | 66.199 | .8978 / .8223 / .7969 | 42.74 / 24.03 / 8.89 |

来源：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1/evaluation/results.json`及[r1_results.json](r1_results.json)的域表；18行126个数值已再次与最终输出核对。原配对区间及解释见[stage254](../../workbook/2026-11-15_stage254_r1_projection_controls_and_domains.md)。固定purity的开发效率受尾部统计和阈值选择影响，表中零或较低效率不表示所有阈值的分类能力都相同。

上述18行的126个实测数值已逐项核对；另对B1/B4/HGB三seed的18模型、11域从预测独立读回1,185项，包含raw/weighted confusion与固定purity效率/coverage，全部通过，见[stage261](../../workbook/2026-11-15_stage261_r1_collaboration_and_count_audit.md)。

全域三seed的较稳健结论见[stage259](../../workbook/2026-11-15_stage259_r1_three_seed_task_evidence_and_scheduling.md)：B1真实T3平均BA3为56.563%，T4 conditional为56.304%；相对HGB的T3平均增益1.9733 pp，低于预设2 pp实用门槛。对B4的T3平均增益2.1711 pp，同时μ recall下降2.410 pp，未满足逐类非劣。不能只凭平均BA宣布全面优于基线。

[stage263](../../workbook/2026-11-15_stage263_r1_three_seed_compute_comparison.md)的三seed同allocation等时预算结果：B1 compute T3平均BA3为56.369%，相对T4 conditional只高.064 pp [−.179,+.302]，NLL更差.008444；e/μ/NC recall差为−.627/+6.583/−5.763 pp。B4 compute T3的BA3为54.674%，相对其T4高.188 pp [−.072,+.413]。24模型56配对、1,431项独立读回通过；增加三类重复曝光没有消除逐类取舍。这里比较的是本allocation fresh T4对应的操作wall，设备、实际精度与逐seed波动见[紧凑结果](r1_b1_three_seed_compute.json)。

最终结果补齐B2：T3/T4 conditional/compute T3的三seed BA3为55.0689/55.3911/55.0778%；compute−T4为−.313 pp [−.514,−.113]、NLL3 +.006649 [.005003,.008099]。B3 seed42 compute为55.3700%，相对T4高.080 pp但区间跨零。三类化未稳定改善共同任务，继续保留四类研究；R1的wall-based schedule与曝光同时变化，不单独归因于重复曝光。最终10项compute预算误差均在±5%内。

## 与合作组原报告的逐项关系

| Item | Collaboration documents | Actual R1 implementation / measured scope |
| --- | --- | --- |
| Task | e CC / mu CC / NC | 真实T3，加T4 conditional共同评价；另外给τ污染 |
| Validation identities | Mahlstedt两种输入使用相同validation | R1所有候选共用冻结Tune；与合作组事件列表尚未对齐 |
| Strip / pixel images | Wu：strip为68×40与56×40；pixel为15×250与15×200 | 两个物理XZ/YZ投影，每个4通道，1322×420，pitch1 mm；pixel count及bar count/energy/PE |
| Tracker | Wu：12个简化track reconstruction特征 | digitized tracker set branch；不等同于原12维重建量 |
| ResNet | Wu：5×5 stem，3 stages×2 blocks，width32→128，带pixel约2.9M参数 | 7×7 stride4 stem，4 stages×2 blocks，width16→128；T3共793,545参数 |
| Plain CNN | Wu：4 layers，24→96，约0.2M参数 | 4 layers，24/48/72/96；含共同tracker融合为179,241参数 |
| Training | Wu：40 epochs、cosine、相同class weights，80/10/10 split | 冻结新Train/Tune、4 passes；两LR screen；实际完整收敛未建立 |
| Fiducial / prior | Mahlstedt有center/large几何描述；报告的实际类prior与scores未提供 | 几何文字对齐，具体accepted geometry冻结；每域uniform三类benchmark另给raw人数 |
| Efficiency / purity | Wu表格各项purity不同；Mahlstedt给80%/90%曲线 | 统一benchmark purity及固定FPR；不能把Wu各模型的不同选择点直接排名 |
| Pixel increment | Mahlstedt关闭pixel并从scratch重训 | R1有固定权重remove_pixel压力诊断；未完成同配方无pixel重训，二者是不同问题 |

来源：[Wu PDF p2–4](../../docs/FORTVNE_10212026.pdf)、[Mahlstedt PDF p2–7](../../docs/FORTVNE_pixel_vs_scintonly_meeting_260916.pdf)。R1的B1是明确描述的CNN/ResNet同类方法对照；严格原研究复现尚未完成。当前结果不能量化相对合作组原模型的增益。

## 后续复现所需输入与执行

1. 获得原研究train/validation/test的事件身份及family映射、生产/geometry/readout版本和完整选择实现，先核对分母及重叠，不用Tune替代原validation。
2. 获得每类权重/先验、每个效率–purity点的score/cut、网络和训练配置、tracker特征定义。能取得原权重时先做固定权重读回，否则在同事件上从scratch重训并报告复现范围。
3. 按原分离图像尺寸与12维tracker构建adapter，与R1统一accepted-response版本另做配对；分别改变表示、容量或训练长度。原pixel removal按同配方实际重训，保留固定权重压力诊断供解释。
4. 在现有R1同输入比较后，R2先以B1的fresh D0/D1/D2检查数据、曝光和训练配方；B2按实际问题和费用追加。R3再检验投影stride和稀疏局部信息路径。未得到原研究输入不阻塞这些已定义的模型改进。

W08的R1对照已完成；严格原研究复现仍依赖上述原始输入。41模型完整结果和同计算表已交付，122条allocation全部终态，batch成本45.708889异构device-hours。Cal-A/Cut-B/Confirm继续封存；R2已完成D0/D1/D2及必要控制，198项验收通过，见[最终结果](r2_results.json)。其开发结果与R1共同域表分开；严格原研究复现仍需前述输入。
