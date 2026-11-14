# 阶段237：现有结果独立审查与下一轮优化路线 v2

**现有方法有小幅、可复核的泛化改进，尚未建立最优性。下一轮先交付实际三类强基线及同输入四类对照，优先检验训练规模和跨层几何关系；τ 保留为独立优化分支，按实测工作点决定继续四类或条件细分。**完整路线见 [roadmap v2](../roadmap/README.md)。

## 阶段目标

独立整理原 S0–S6 资料，回答已有改进是否成立、哪些设计尚有系统性不足、三类/四类/更多类别如何选择，并设计可执行、可评价的下一轮完整路线。验收为旧计划完整归档、新计划与历史证据可定位、关键结果的小规模独立复核、文献依据与实验决策对应。

## 工作内容

审查基准 Git `37dd763`，branch `xcheng/neutrino_flavour`。全量编目 236 份 workbook 并提取结论，重点核对 S2 stage55、修复 S3 stage87、S4 stages163/170/181/182、S5 stages203/205、S6 stage236 及相关诊断、结果 JSON 和当前实现。重点审查代码包括 `s3/dataset.py`、`s3/models_repair.py`、`s2/representation.py`、`s5/objects.py`、`s5/model.py`、`s5/truth.py`。本轮没有重跑历史全部实验。

5 份本地 PDF 共 80 页完成文本提取，结合 FASERCal 本地 TeX、在线 Methods 和合作组三类报告核对输入、训练规模、接受度及指标口径。联网查阅期刊/会议正式页面、arXiv、INSPIRE 和作者软件；[文献表](../roadmap/v2/05_literature.md)逐项记录用途与采用次序，包含 FASERCal、ParticleNet、ParT、PTv3/LitePT、NuGraph2/context-aware、Object Condensation/GravNet、PDG τ、OGM、pAUC、GroupDRO、校准/label shift、NOMAE 和 Sonata。[来源记录](../roadmap/v2/source_review.json)中的 27 个外部链接全部 HTTP 200。论文方法证据与本项目待测收益分开记录。

原根目录 README 和 01–13 共 **14 份完整旧路线**移入 [archive/2026-11-14_v1](../roadmap/archive/2026-11-14_v1/README.md)，只调整移动所需的 Markdown 相对链接。旧 01–13 保留兼容入口与锚点；configs/results/tools 继续使用原路径。新路线包含总入口、6 份专题文档、机器可解析实验规格与本次小型复核结果。新增研究工具只读既有 validation 预测和训练记录。

所有既有 accepted samples、source-family/split 与 detector-proxy 继续有效。新独立确认集的使用记录核对针对历史训练/调参暴露，不重开样本来源或硬件认证。

## 详细数据

### 历史结果的核对

| Quantity | Measured result | Evidence / interpretation |
| --- | --- | --- |
| Full inventory | 7,237,572 events / 7,240 ROOT | stage04；全库数量，非本轮训练规模 |
| S6 independent test | 724,441 events；e/μ/τ/NC = 86,905/416,305/65,558/155,673 | stage236；全部 accepted 分母 |
| MAE3 ensemble BA | 0.435225878，CI95 [0.433704723, 0.436685364] | 已修复有效学习模型 |
| MAE3 − Scratch BA | +0.003353622，CI95 [+0.002507538, +0.004166165] | 同 100k 监督标签协议，MAE 另有 SSL 成本 |
| MAE3 − Scratch NLL | −0.007486826 | 小幅概率改善 |
| τ frozen validation 1% cut | 2,331/65,558 = 0.035556301 efficiency；test actual weighted FPR 0.012080097 | 不把 nominal 1% 当 test 实际 FPR |
| Rare prior τ purity | πτ=1% 时 2.8099%；πτ=0.1% 时 0.2867% | 既有情景分析，非 flux 测量 |
| Empty test response | 18,852/724,441；active-domain BA 0.438654 | 空响应不能解释全部性能差距 |
| Tracker − native BA | +0.026564，CI95 [+0.025494, +0.027827] | 相应 S3 预算下输入收益成立 |
| Native − 1mm BA | +0.000511，CI95 [−0.000539, +0.001444] | 当前结构下硬分类粒度收益未确认 |
| Sparse-noise / combined ΔBA | −0.017750 / −0.016063 | S6 固定 32k 子样本；已接受 proxy 压力，不称实测硬件误差 |

完整数值和条件见 [S6 stage236](2026-11-14_stage236_s6_final_results.md) 与 [v2 独立评估](../roadmap/v2/01_evidence_review.md)。原生信息、τ 细分、对象重建的负结果都只约束其已测配方，不作为硬件信息上限。

### 本轮新增只读计算

工具：[review_current_results.py](../roadmap/tools/review_current_results.py)。读取 `roadmap/configs/s6_candidates.json` 中预定的六个最终 revision3/empty-response-bypass 模型，每个已有 validation 文件 10,000 events；三 seed 按保存的原始概率均值形成 ensemble。本轮没有拟合温度、重新选阈值、读取 test scores 或执行新推理。

从 S6 存储 confusion counts 重算 test BA=0.4352258778442265，与汇总值差约 5.6×10⁻¹⁴；逐类行计数与原 test counts 一致。

| Model / seed | Train events | Parameters | Best / total epochs | Last-epoch ΔNLL | Recorded data-wait / total |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAE3 / 17 | 100,000 | 897,992 | 4 / 4 | −0.002033871 | 46.3553% |
| MAE3 / 42 | 100,000 | 897,992 | 4 / 4 | −0.002726188 | 46.3846% |
| MAE3 / 2026 | 100,000 | 897,992 | 4 / 4 | −0.006557449 | 31.1608% |
| Scratch / 17 | 100,000 | 897,992 | 4 / 4 | −0.005272446 | 46.6420% |
| Scratch / 42 | 100,000 | 897,992 | 4 / 4 | −0.004932730 | 46.3396% |
| Scratch / 2026 | 100,000 | 897,992 | 4 / 4 | −0.007793493 | 31.6403% |

这些时间比例来自现有 manifest 的 loader 计时，不当作 GPU idle 百分比。末轮最优说明有必要测饱和，不能单凭这一点断定最佳训练长度。

条件三类定义：删除真 τ 后，把固定四类模型的 `p[e,mu,NC]` 重归一化；不是实际三类重训，也不是在有 τ 的部署样本上获得该性能。域内 AP/AUC 保留实际类别混合，BA 平均逐类 recall。

| Validation domain | Non-tau e / μ / NC | MAE3 conditional BA3 | Scratch BA3 | MAE3 AUC e / μ / NC |
| --- | --- | ---: | ---: | --- |
| All | 2,500 / 2,500 / 2,500 | 0.546933 | 0.543733 | 0.809711 / 0.723572 / 0.718968 |
| Pixel-centered | 245 / 237 / 227 | 0.744517 | 0.741334 | 0.932970 / 0.890671 / 0.864807 |
| Large scintillator | 1,131 / 1,183 / 1,127 | 0.667814 | 0.667343 | 0.909284 / 0.823739 / 0.802449 |

MAE3 全域四类 validation BA=0.434300，pixel-centered 四类 BA=0.584691。三类删除操作改变了类别数与分母，不把其差值当作减少类别后的净训练收益。已有 validation 重复用于开发，这些新增数字属于描述性诊断。

六个最终 manifest 的 `local_blocks` 写着 `same_plane_4mm...`，实际 config 为 `window_pixels=16`，native window 约 0.3328×0.3648 mm；multiscale 代码中 `context` 为空，旧 `cross_plane_message_passing=true` 文本不等同于存在显式跨层几何层。未来 metadata 应从真实实现生成。S4 已有 global 置换 ΔNLL=+0.060310 的正向使用证据，不沿用早期“global 没学到”的判断。

### 命令、产物与验证

使用现成 LCG108_cuda Python 3.12.11，无新增依赖安装；轻量只读复核命令：

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python roadmap/tools/review_current_results.py \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_review_2026-11-14/current_results_review.json
```

结果 JSON 为 57,301 bytes：[仓库精简副本](../roadmap/v2/current_results_review.json)。研究资料目录为 `/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_review_2026-11-14/`，保留复核 JSON、PDF 编目、网络来源记录和必要摘录。已清理本轮 26 份临时下载/重复全文，共 7,229,946 bytes；清理时目录保留 12 份小型证据、合计 310,958 bytes，记录为该目录的 `cleanup.json`；用户原 docs 未修改。新的 [experiment_plan.json](../roadmap/v2/experiment_plan.json) 是设计规格，不能直接提交到旧 runner。

本轮新增训练 0、HTCondor/Kubeflow 作业 0、checkpoint 0；新模型效果、实际三类重训收益、300k/1M 饱和、新确认集可用数量、未来运行成本和物理 flux 产额均**未测量**。JSON/实验依赖一致性、脚本 AST 语法、236 份索引完整性、14 份归档正文保持、48 组 confusion/metric 读回和 `git diff --check` 全部通过；37 份相关 Markdown 的 540 个本地链接及 34 个历史入口链接/锚点全部有效。80 device-hours 和 440k 非训练预留样本的设计数值一致，实际可用性仍待 R0 测量。

归档和数值的最终核对见 [review_validation.json](../roadmap/v2/review_validation.json)。

## 阶段性结论

**事实：**MAE 在当前标签预算下有小幅泛化收益；tracker 有总体分类增量；τ 低背景效率弱；共同三类也未达到强分类水平；当前架构没有显式跨层几何关系层，训练饱和与同输入架构排名尚未建立。此前学习故障已有修复，评价基线应使用最终有效模型。

**研究判断：**先做强监督、公平三类/四类和规模曲线，比继续围绕少标签 probe 或堆叠 event-level 辅助头更直接。跨层连续性、几何锚点、可靠 tracker/track 对象和局部监督是优先可检验方向。四类尚无被排除的理由；条件细分应保留 coarse τ label，且以合并后的实际任务增益决定晋级。

路线使用 R0–R7：研究使用记录/效率→同输入三类四类强基线→数据与训练规模→显式几何关系→可靠对象→τ 专家与条件衰变→定向预训练和鲁棒性→新独立确认。三类交付可在 R1/R2 完成，不等待 τ 成功。建议 6–10 周、首轮 80 device-hours 是排期和资源规划，实际吞吐尚未测量。

## 下一步计划

1. stage238 起落实 R0：输入历史 accepted manifests、SSL/训练/probe 使用清单；输出新 Train/Tune/Cal-A/Cut-B/Confirm 的 family 记录和实际可用数量，确认评价分区也不与复用初始化的训练 family 相交。
2. 实现真实 representation metadata 与小规模 loader profile，核对窗口尺寸、跨层模块和各阶段计时；由实测吞吐确定 R1 任务的 CPU/GPU 与预算。
3. 实现 T3/T4 统一评价及 B1/B2/B3 的 tracker late fusion，验证真实小样本拟合和分支梯度；按同曝光/同总计算两口径完成合作组三类对照。
4. 只将胜出的基线扩展到数据规模与几何关系研究；以预先冻结的 BA3、τ 固定 FPR 效率、非劣约束和资源门槛决定后续对象、τ 和 SSL 投入。
