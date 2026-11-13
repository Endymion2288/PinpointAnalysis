# S4：配对子道统计与监督预算匹配

## 阶段目标

使预训练收益能够区分 global 表示、下游适配、τ 子道和额外计算量，补齐可复核的统计对照。

## 工作内容

冻结 `s4_pairs.json`：54 项同事件配对、三个标签预算各一项独立三 seed 均值比较。MAE3 使用独立 SSL 与下游初始化；seed42 的 EMA/Rel/MAE continuation 作为方法筛查，不把单 seed 差异当稳定收益。Frozen global 与 full readout 分开统计和画图。新增 τ electron/muon/hadronic 的 raw/common-support 配对 bootstrap；保留原 connected-chunk/family multiplicities、score ties，其他 τ 子道从相应二分类分母中去除。区间以固定训练 seeds 和固定 common-support weights 为条件，训练 seed SD 单独报告。

实现 `matched_plan.py`，从完整 MAE 确定性 shape replay 读取新增预算，只读取一次各标签事件的精确 block/region shapes，再重放监督 sampler，冻结额外步骤数。目标为 MAE3 的 Linear/attention forward/backward/checkpoint-recompute proxy，匹配误差要求 ≤0.5%；该 proxy 不包括 normalization、elementwise、compiler fusion，不能称为完整硬件 FLOPs。监督控制从 Scratch **last** 的完整模型（含 Head）继续，fresh optimizer/schedule，sampler epoch offset 接续原训练。原 Scratch 最佳点和 continuation 最佳点分别报告；不会暗中用差异较大的短采样均值当精确预算。

## 详细数据

1024-step Rel pilots，同 512 个 train-inner probe events、37,567 valid bar channels、15,921 mixed：

| Route | Semantic CE | Constant-prior CE | CE minus constant | Fine/4mm/bar query AP |
| --- | ---: | ---: | ---: | --- |
| visible bar local blocks | 0.72199172 | 0.72388333 | −0.00189161 | 0.12178995 / 0.36802475 / 0.69662431 |
| visible bar channels before pooling | 0.71789485 | 0.72388333 | −0.00598848 | 0.12972270 / 0.37991017 / 0.69725559 |

两者 integrity/internal 与 measured-learning 检查均通过。MAE decoder 继续仅读 global；bar decoder 仅新增可见 observation 特征，truth 仍只进 loss。这是训练内语义学习证据，下游分类收益尚未测量。

独立 MAE17/2026 确认已提交 bigbird20、cluster1574110；排队时将 downstream concurrency 改为6、CPU18、RAM30GB，修改前后 ClassAd 保存于 `logs/mae_confirmation_concurrency.json`。即时发现 H100 `b9pgpun015` 虽无 staged drain，实际 `Start=false, Requirements=false`，因此不能把 Idle 视为可用容量。增加原生 BF16 的 H200/24GB H100 MIG 候选，保留单 GPU 两训练的紧凑资源选项。

旧 V100 FP32 prototypes cluster2198172 已在首个 optimizer step 前失败：LCG108 Torch2.7.1 CUDA 编译仅支持 sm75+，V100 是 sm70。四项均无训练结果，不参与方法评价；保留实际 allocation 成本。后续 FP32 prototype pool 改为当前 runtime 支持的 T4。Kubeflow H100 MIG 主 wave 不受影响。

产物根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。预算步骤尚待完整 MAE replay 输出，未测量；完整分类矩阵仍运行。

## 验证

执行 `python -m unittest discover -s tests -p 'test_flavour_s4.py'`。21项全部通过（46.134秒）。新增检查覆盖 compact shapes 与实际 collate/hook 预算一致、监督预算整数 rounding、τ score ties/weighted branch exclusion、global/full 配对和无支持子道的空区间。

## 阶段性结论与下一步

Rel 的 pooling 前局部信息修复获得训练内支持，改善幅度有限，不能代替下游分类结论。完成独立种子训练、matched-supervised 控制、400次完整配对统计与成本终态后再判断扩量。
