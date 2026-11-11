# 阶段 03：更新本地 PDF 后的证据核对与路线图修订

四份更新后的 PDF 均可读取，共 61 页；已阅读提取全文，并渲染查看 27 页关键图表。新增 [本地汇报核对](../roadmap/05_local_studies.md)，修订 [研究路线图](../roadmap/README.md)及数据、模型、执行和配置合同。主要变化是增加 pixel 粒度×联合/投影对照、fiducial 分层、FASER 同工作点比较，以及 reconstruction resolution–coverage 评价。

## 阶段目标

补齐上一阶段无法取得的四份本地汇报，核对它们的输入、样本选择、工作点和数值，明确哪些结论可支持当前原生硬件四分类研究，并将新增证据转化为可执行实验。

## 工作内容

- 用 `pdfinfo` 确认文件类型、页数、大小，用 `pdftotext -layout` 提取四份全文。渲染运动学 raster plots、pixel ablation curves、selection/confusion matrices 和 FASERCal 工作点，核对文字提取未覆盖的图中条件与计数。
- 对比 2026-09-16 与 2026-10-21 两份 flavour 研究：分别记录 fiducial、1 mm 投影、FASER 特征、网络、类混合和不同工作点，避免将未统一条件的差异单独归给 pixel。
- 将 2026-10-07 的运动学优化结果转化为 predicted-vertex、component direction、θ/φ 分开评价和 quality-cut coverage 工单。
- 按实际封面日期将 FASERCal workshop 归为 2025-11-14，和 2026 arXiv v2/正式论文分别引用，核对 τhad 与 inclusive τ 的效率分母、kinematic target 与 residual 符号。
- 更新 `roadmap/README.md`、`01_data_and_hardware.md`、`02_literature.md`、`03_models_and_experiments.md`、`04_execution.md` 与 `configs/study.yaml`，新增 `05_local_studies.md`。上一阶段报告与审计结果保留当时的记录。

## 详细数据

### 资料实测与图表核对范围

| PDF | Pages | Size (bytes) | Rendered PDF pages |
| --- | ---: | ---: | --- |
| `FORTVNE_pixel_vs_scintonly_meeting_260916.pdf` | 8 | 618,051 | 3–8 |
| `FORTVNE_Update_kinematic_variables_2026-10-07.pdf` | 9 | 3,546,193 | 2–7 |
| `FORTVNE_10212026.pdf` | 7 | 564,085 | 2–6 |
| `FASERCal/FASERCal_wshopRun4.pdf` | 37 | 8,122,725 | 19–25、30–32 |
| Total | 61 | 12,851,054 | 27 pages |

四个 `pdftotext` 均 exit 0，提取页数与 `pdfinfo` 一致。FASERCal 文件给出 PDF 数字格式提示，但 37 页提取完整，所选图表渲染成功。PDF 页码从 1 开始，与部分页脚编号不同。所有输入文件未修改，未加入本次 Git 提交。

### 汇报中的数值及本次算术核对

**2026-09-16，PDF 第 3–7 页：**同验证样本的三类任务，在 pixel-centered volume 加入 pixel 后，νe/νμ/NC ROC-AUC 从 `0.944/0.842/0.801` 变为 `0.970/0.906/0.876`，增量为 `+0.026/+0.064/+0.075`。在 large volume，νμ/NC 为 `0.747/0.698→0.774/0.719`，νe 为 `0.925→0.923`，变化接近零。中心体积 90% purity 工作点的 νe/νμ efficiency 约 `50%/51%→70%/80%`。记录了旁注与表格舍入差异，没有将其当作精确增量。

第 7 页两种输入的 confusion counts 逐行总数一致：中心体积 `3,368/18,821/7,234`，共 29,423；large 为 `20,130/107,826/40,551`，共 168,507。这是对幻灯片计数求和，未重新读取原研究事件。large 的 NC AUC 增加，但 argmax recall 约 `0.68→0.45`，支持将 ranking 与 selection 分开评价。

**2026-10-21，PDF 第 2–6 页：**pixel 输入为 `15×250` 与 `15×200` 的 1 mm 双投影。ResNet 的 M1（strip+FASER）νe efficiency/purity 为 `0.571/0.979`，M2（再加 pixel）为 `0.519/0.978`。νμ 的 M2 为 `0.431/0.971`，M3（strip+pixel、无 FASER）为 `0.890/0.813`，说明表中的高纯度与高效率是不同工作点，尚不能量化同工作点的净增益。Base M1 νe selection-efficiency 行和 `0.518+0.003+0.004=0.525`，有未进入 selection 的事件，不能当作完整 argmax confusion。

旧 FASER 假设包括 `σx=0.58 mm、σy=17 μm`、99% efficiency、无 noise、0.1 mm step 合并、3 mm straight-line gate 和 2.4 TeV momentum cap；匹配当前 field/geometry 的有效性尚未测量。旧 pixel 图像范围与本次 ROOT 的 `266×196 mm²` 不同，需要追溯旧几何/图像边界，不能由 shape 独自认定裁剪错误。

**2026-10-07，PDF 第 2–9 页：**报告指出 pT magnitude 改善而 azimuth 泛化不足、hit/hybrid direction 精确但覆盖有限，建议保留 pixel/component 信息。图中有 Score WP，精确阈值、全样本分母和 small/large 边界在该文件未给出；尚不能将 selected resolution 当作全样本结果。新计划分别测 θ/φ，并只用预测/重建 vertex 居中。

**2025-11-14 FASERCal，PDF 第 17–25、30–32 页：**早期样本为约 1.5M unbiased+68k τ-only，区别于 2026 v2。其 on-axis、780 fb⁻¹ 的 τhad preliminary selection 给出 `p(τhad)>0.95`，signal/background 约 `1.9/1`、purity 66%、τhad efficiency 11%。按舍入表格计算，全部 τ 的分母为 `17.7+4.4+4.6=26.7`，inclusive efficiency 约 `1.9/26.7=7.1%`。这是该研究配置下的数值，不是 PINPOINT 预期性能。第 25/30 页使用 class score cuts，第 32 页 Gaussian fit 使用所有事件且 residual 符号相反，均已记录。

### 命令、产物和验证

四份文件分别执行 `pdfinfo <PDF>` 与 `pdftotext -layout <PDF> <task-output>`。例如：

```bash
pdfinfo docs/FORTVNE_Update_kinematic_variables_2026-10-07.pdf
pdftotext -layout docs/FORTVNE_Update_kinematic_variables_2026-10-07.pdf \
  /eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_pdf_refresh_2026-11-11/scratch/FORTVNE_Update_kinematic_variables_2026-10-07.txt
pdftoppm -f 2 -l 7 -scale-to 1600 -png \
  docs/FORTVNE_Update_kinematic_variables_2026-10-07.pdf \
  /eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_pdf_refresh_2026-11-11/scratch/FORTVNE_Update_kinematic_variables_2026-10-07
git diff --check
```

运行证据目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_pdf_refresh_2026-11-11/`。最终保留 `pdf_review.json`（页索引、范围、来源数值、计数核对与未决元数据）和 `validation.json`；提取全文与渲染图为临时核对副本，核对后清理。原 PDF 继续放在用户的 `docs/`。

| Validation / measurement | Result |
| --- | --- |
| Updated roadmap | 8 files，6 个 Markdown，117,225 bytes，1,100 lines；不含本报告 |
| Roadmap local links / explicit anchors | 27 个检查通过 |
| YAML | 解析、四类映射、split 比例、train-only 预训练/normalization、2×2 输入对照和新评价要求通过 |
| Python audit tool | `ast.parse` 通过；本轮未改脚本 |
| PDF metadata / source arithmetic | 4 份文件大小、61 页索引、confusion row totals 和 τ 分母核对通过 |
| Whitespace / final newline / code fences | 检查通过；提交前再执行 `git diff --cached --check` |
| New classification / kinematic metrics | 未测量 |
| GPU / throughput / hardware calibration / systematics | 未测量 |

本轮没有重跑旧模型或 ROOT 审计，没有训练与 HTCondor 提交。数值的来源已区分为文件元数据实测、汇报引用和由汇报表格作出的算术计算。

## 阶段性结论

**事实：**中心 fiducial 的既有研究已经显示 pixel 有分类增量；1 mm 双投影 CNN 研究未显示相同收益，两者条件尚未统一。FASER 的高纯度、τhad 的分支效率和 selected-sample 的运动学精度，都需要各自的分母与工作点。FASERCal 支持完整 readout/reconstruction/pretraining 链路，但其谱仪与三视图体素输入不能直接替代 PINPOINT 响应。

**假设：**原生联合位置、局部 component direction 与受控多模态融合可能改善 τ 识别及泛化；新增 2×2 因子对照、接受度分层和 resolution–coverage 测量可判别这些增益的来源。新四分类性能与统计显著性仍待实验。

## 下一步计划

| Task | Required input | Acceptance criterion |
| --- | --- | --- |
| S0：历史/当前数据合同对照 | L1–L3 配置、run/production、fiducial/score/weights、tracker conditions | 边界、版本与未决状态可追溯；不依赖旧假设继续全库身份/标签审计 |
| S2：表示与接受度实验 | 共享 native catalog、split、同预算模型与几何 | native/1 mm × joint/projections、中心/large/全目标域的收益分别有计数和区间 |
| S3：FASER 增量 | digitized hits/fit 与 uncertainty、固定类混合 | 同 purity/background acceptance 的效率及 track coverage/fake rate 可复核 |
| S5：局部分量与 τ 分道 | 预测 vertex、component labels 与 quality masks | θ/φ、resolution–coverage、τhad/inclusive 分母及 charm 混淆分别报告 |
