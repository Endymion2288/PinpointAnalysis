# 研究结论整理、τ物理路线与R2稀疏μ进展读回

**已有模型改进成立，但最优性未建立；保留四类研究，同时交付三类对照。下一步完成已冻结的R2，再按训练饱和、unique数据收益和局部辨别能力决定扩展。**集中入口为[研究结论与下一批决策](../roadmap/v2/09_decisions_and_handoff.md)，旧14份路线继续完整保存在[v1归档](../roadmap/archive/2026-11-14_v1/README.md)。

## 阶段目标

在R1完整结果和R2首批执行基础上，整理原任务需要的完整研究结论、类别选择、下一批分流及资源策略。继续核对稀疏μ的训练/决策与表示两类解释，补充直接相关的τ识别文献。验收为结论有证据入口、设计与机器规格一致、旧路线与记录可定位、新读回算术和文档链接通过检查。

## 工作内容

沿用stage237对S0–S6的236份记录、14份旧路线、5份本地PDF共80页及FASERCal TeX/Methods的整理，复核当前v2、stage274的41模型/133配对完整结果，以及stages275–277的R2实现、运行和恢复记录。本阶段没有重跑历史实验，也不改变已经冻结的R1评价口径。

新增`roadmap/v2/09_decisions_and_handoff.md`，明确回答已有改进、最优性、3/4/更多类别的选择及R2–R7下一步。整理了D1−D0与D2−D1的分流表、endpoint/best-screen的选择机会差别、B1降采样与原生几何边的族内对照、raw/object质量门槛、τ两条物理路径和条件细分前提。`experiment_plan.json`同步这些设计，不修改六份已冻结训练配置。

新增小型只读工具`roadmap/tools/review_r2_progress.py`，从六份manifest提取固定4k Tune screen历史，并从低occupancy confusion重新核对计数、recall和BA。AP/AUROC来自原history，没有从事件预测独立重算。输出小型JSON及全英文PDF/PNG；没有读取checkpoint、response、Cal-A/Cut-B/Confirm。

资源说明继续遵循用户补充：优先预计完成时间短的兼容GPU，HTCondor/Kubeflow、异构卡、共享设备与MIG并用；单allocation内按实际aggregate吞吐和完整validation峰值选择并发数。修正了旧文档中“R2尚未启动”和Kubeflow历史凭据过期的过时描述，补齐workbook索引至stage278。

## 详细数据

### R2只读快照

源目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/`。快照UTC为`2026-11-15T10:51:48.885896+00:00`，六个模型共23个screen，**162项算术检查通过**。D0 T3在快照中complete，其他五项running；这是形成时的状态，不是R2最终验收。低occupancy固定分母e/μ/τ/NC=285/275/342/309；条件三类为869行，真μ275行，μ占比31.64556962%。

| Model | Latest non-tau exposures | All NLL3 | Low-occupancy mu recall (%) | Low-occupancy mu AP | Low-occupancy mu AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 T3 | 300000 | .879377554 | 8.363636 | .345856 | .550168 |
| D0 T4 | 300000 | .872906114 | 8.000000 | .345465 | .550787 |
| D1 T3 | 300000 | .907920444 | 87.272727 | .376934 | .568411 |
| D1 T4 | 300000 | .876633196 | 48.000000 | .349528 | .542639 |
| D2 T3 | 300004 | .882435377 | 18.181818 | .354272 | .546820 |
| D2 T4 | 225000 | .887253168 | 38.181818 | .357095 | .555384 |

精确值以[原始小型JSON](../roadmap/v2/r2_progress_review.json)为准。以上都是固定screen指标；D0 T3/T4已经达到预定曝光，但表格仍不替代完整40k Tune终点评价。D1/D2仍在更长的完整LR轨迹中，不用其300k中间点排名。

D0 T3同一275个低occupancy μ上的recall在75k/150k/225k/300k非τ曝光为7.6364%/48.7273%/10.9091%/8.3636%，AUROC为.528130/.565069/.558096/.550168。D1 T3最新点把84.2348%的低occupancy非τ事件判为μ，μ precision为32.7869%；87.2727%的recall不能单独作为强辨别能力证据。全域μ排序的既有优势也不能外推为该子域有充分信息。

模型和输入未改变时这些变化已经出现，说明训练配方与多类决策必须继续检查；低子域AUROC仍弱，也保留局部信息路径的研究必要。这里只给描述性点估计；本阶段没有新增窗口间配对区间、seed总体区间或因果归因。

### 联网论文与类别设计

通过INSPIRE查到并核对OPERA/NOMAD发表记录，分别下载arXiv摘要、PDF及INSPIRE记录，共6个URL均HTTP200。OPERA PDF444,435bytes、7页，抽读p2–3；NOMAD PDF751,604bytes、46页，抽读p4/8及hadronic selection片段。[来源记录](../roadmap/v2/source_update_stage278.json)保存UTC、URL、发表信息与采用决定；全文及提取文本位于`R2/audit/literature_stage278/`。

- OPERA，*Phys. Rev. Lett.*120,211801(2018)，arXiv:1804.04912：按1h/3h/e/μ可见形态使用拓扑与运动学BDT，检查charm、强子再相互作用、大角度μ散射和primary μ漏识别。本阶段只采用方法启发，未移用其cuts/性能/显著性，关联Erratum亦记录。
- NOMAD，*Nucl. Phys. B*611,3–39(2001)，arXiv:hep-ex/0106102：衰变顶点无法空间分离时，采用精确可见末态运动学及其相关性识别τ。PINPOINT须先验证观测方向/动量及接受度，不能将truth missing-pT作为输入。

进一步审查`s5/truth.py`发现`tau_direct_prongs`的实现是共同衰变点的直接子粒子中的带电粒子计数。它与经共振展开的最终charged prongs、可见prongs和实际重建tracks之间的一致性**未测量**。现有直接计数仍是有效诊断；未来had1/had3标签需先验证定义和覆盖，不能直接据此启动七类均衡softmax。四类coarse监督和unknown mask继续保留。

### 命令、图表与验证

使用已有LCG108环境，无依赖安装；只读取数十KB级manifest摘要：

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python roadmap/tools/review_r2_progress.py \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2 \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/audit/progress_stage278.json \
  --figures /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/figures
```

产物为`audit/progress_stage278.json`及`figures/r2_sparse_muon_progress.{pdf,png}`，PNG已经人工查看；仓库保留小型JSON、工具、来源记录和路线文档。162项计数/recall/BA检查通过；另核对本报告表格36个数值单元。27份Markdown的628个本地链接/锚点、41份v2 JSON解析、工具AST语法、278份workbook索引完整性及`git diff --check`均通过。14份v1归档与`37dd763`原文逐份比较，除Markdown链接目标调整外正文完全一致。80 h原规划之和核对通过，后续按实际成本重新分批，不将它写成新增额度。最终检查保存于`audit/delivery_stage278.json`。

本阶段新增模型0，没有修改或重启已有六个科学模型。先前stage277的watcher继续按真实完成状态释放gap/完整评价/图表；本阶段期间D0完成后已释放Train gap作业1582970，来源和schedd选择见`runtime/r2_gap_b1_d0_a1.submission.json`。完整R2终态成本、D1/D2终点效果、R3–R7收益、had1/had3新标签一致性均**未测量**。

收尾状态检查中，D0 T3/T4均已complete，两者endpoint Train gap报告也均complete；D1/D2四个模型继续运行。上表保留10:51:48 UTC快照，未用后来的窗口替换其中单个模型，从而保持读回证据与图表一致。

## 阶段性结论

**事实：**已有MAE和tracker改善有历史独立证据；R1当前预算下B1较强但逐类非劣未全部通过；三类化没有稳定改善共同任务。R2已存窗口中的低occupancy μ排序仍弱，argmax又有较大波动。新路线、文献、类别设计和资源策略已整理为连续入口。

**研究判断：**先完成R2，在训练/决策与局部表示两方面分别定位；可见τ拓扑与未分辨顶点的观测运动学都值得保留。类数调整应服务合并后实际任务，不以增加类别替代缺少的重建或可靠标签。现有证据不足以判定四类达到信息上限。

## 下一步计划

1. 完成初始六模型的完整40k Tune endpoint/best-screen评价、共同Train gap、72项配对/11域和独立读回，保留固定预算为主比较。
2. 根据D1−D0、D2−D1及低occupancy真实排序/取舍，选择D3、更多unique、width32、第二原生族或进入R3；每一新增批次均有具体科学问题和实测成本。
3. R3优先B1 stride/相位诊断，原生族以本族baseline检验压缩前几何边；R4先测对象质量，R5先条件衰变头与完整背景，满足条件才细分had1/had3。
4. Cal-A/Cut-B/Confirm继续封存，最终候选冻结后执行R7；三类合作组交付不等待τ分支成功。
