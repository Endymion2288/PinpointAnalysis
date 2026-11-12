# 阶段56：S3四分类合同与分层编码实现

## 阶段目标

检验S2共享4 mm局部压缩是否限制native信息利用，落实用户要求的Plane-aware局部编码、Module-aware context与latent fusion；在正式拟合前冻结完整四分类评价、筛选、三seed确认和晋级规则。

## 工作内容

新增`analysis.flavour.s3`：从当前观测表示重组同层16/64-pixel物理窗口，保留所有局部块；局部坐标/Fourier+set attention；按已有hardware layer/module归组，4个module readout queries；可消融module embedding、z/pair relations，分别比较late、joint、24-latent fusion。8-query控制检验后续module压缩。主模型dim96、749,868参数；这个小规模实现采用FASERCal/Perceiver的层次原则，不声称复现其31.9M模型。

17项首轮配置和22项配对在`roadmap/configs/s3_screen.json`、`s3_pairs.json`中记录。数据由symlink复用S2同一80k train/10k validation；S2四组native/1 mm三seed和参照模型只读引用。主四分类指标为macro AP/NLL，联合报告macro F1、BA、macro AUROC、Brier、每类AP/AUROC/precision/recall/F1、混淆、τ工作点和分支；checkpoint仍按四分类full-validation NLL选点。

`roadmap/09_s3_operations.md`冻结两候选Pareto选择规则、去重三seed确认、主模型native/1 mm确认、400次物理事件bootstrap与四分类晋级/近等效界限。方法是否晋级和S3研究是否完整分别判定。组织上限120 GPUh，含失败与两后端；不启动S4–S6，不使用calibration/test。

训练复用S2 atomic checkpoint/resume loop，增加显式dataset/model/collator hooks，默认S2行为保留。S3 collator按contiguous block counts一次scatter，替代每块扫描全部nodes；测试与已有collate完整结果逐项一致。新增Kubeflow flavor/Pod准入入口和HTCondor实时schedd选择入口，probe已在Kubeflow等待整卡H100。

## 详细数据

### CPU回归与完整事件压力测量

`tests.test_flavour_s3`共5项，验证完整节点与排列不变性、粗表示不泄漏native residual、真实module与signal消融、线性packing等价、三种fusion的空/缺失模态与梯度及batch独立性。复用3项S2训练/恢复测试，合计8项通过，真实exit0，6.510 s。初版7项也通过；新packing增加等价测试后为8项。

T4/FP32、batch1，train总占用秩0/中位/最大三个事件，在6配置各测inference和forward+backward，36行均finite；未执行optimizer fitting。最大总占用36,549包含2个tracker观测；pixel+bar完整输入36,547。以下固定为最大占用事件，首个空事件含CUDA warmup，不能用整张表最大wall作为稳态成本。

| Configuration | Parameters | Nodes | Local blocks | Module/type groups | Inference [s] | Forward+backward [s] | Backward peak [MiB] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 mm / context / latent | 749868 | 36547 | 2641 | 31 | 0.066953 | 0.234088 | 125.483398 |
| 16 pixel / joint | 523020 | 36547 | 14597 | 31 | 0.280731 | 0.826658 | 369.606934 |
| 16 pixel / context / latent | 749868 | 36547 | 14597 | 31 | 0.283962 | 0.838571 | 371.346191 |
| 16 pixel / context / latent / 1 mm | 749868 | 8417 | 6169 | 31 | 0.145322 | 0.410165 | 158.861816 |
| 64 pixel / context / latent | 749868 | 36547 | 5097 | 31 | 0.132108 | 0.396203 | 133.284668 |

16-pixel窗口约0.3328×0.3648 mm，64-pixel约1.3312×1.4592 mm。最大事件4-query module读取产生31×4=124个module tokens；相同主latent数24。完整通道进入局部编码，module pooling仍是有损bottleneck，故专门配置8-query控制。上述为T4/FP32、完整事件压力点，不是H100/BF16科学训练吞吐或总体均值。

压力测量在collator线性packing优化前进行；网络计算与输入tensor由新增等价测试核对，未将CPU packing节省写成实测收益。Interactive GPU完整wall未在初版profiler单列，记录compute/transform实测值，并在组织预算中保守计入从S3目标开始到该profile终态的上界，不以其冒充精确GPU reservation。

### 后端与文献证据

Kubeflow `pinpoint-s3-h100-probe-v1`已真实创建；server Pod dry-run通过，`NVIDIA-H100-NVL`、`nvidia.com/gpu:1`，平台注入46CPU/127G RAM。当前Pod仍Pending/无node，scheduler事件为GPU/CPU/RAM不足及selector/taint约束；GPU runtime未测量。登录正常，单GPU配额空闲后已由该probe请求占用。正式训练尚未提交，不会产生双writer。

本次重新读取FASERCal本地`main.tex` architecture Methods，并联网阅读Perceiver IO第3节/复杂度、LitePT Methods与PTv3正式页、CERN flavors/training。具体迁移判断、URL与保存路径见S3操作合同；未复制自然场景均匀体素化或未知硬件信息。

### 命令与产物

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s3 tests.test_flavour_s2_training
python -m analysis.flavour.s3.profile --root "$S3_RESULTS" \
  --campaign roadmap/configs/s3_screen.json --tasks 0 2 5 6 7 9 --events 3 \
  --output "$S3_RESULTS/profile_interactive_t4"
python -m analysis.flavour.s3.backend --root "$S3_RESULTS" \
  --name pinpoint-s3-h100-probe-v1 --flavor h100-94gb --apply
```

前两项真实exit0。`unit_packing.log`、`profile_interactive_t4/measurements.parquet`与`summary.json`、`profile_interactive_t4.log`保留实际数据；`kubeflow/execution_pinpoint-s3-h100-probe-v1.json`保存提交UID、selector、resources和Pod准入。科学性能、H100吞吐、三seed与最终成本尚未测量。

## 阶段性结论

事实：S3已实现按plane/window编码、物理module context与三种fusion，少量完整事件与CPU回归通过，保留最大事件全部pixel+bar输入。资源压力点证明执行可行，未证明任何分类增量。粒度、窗口、module/readout压缩与模态贡献已经拆成可评价的对照。

假设：细化局部窗口和module context可能提高四分类的可用结构。S2未证明4 mm压缩是native收益较小的原因；必须等待2×2与完整四类配对结果。

## 下一步计划

先在H100/BF16完成最大占用资源测量和真实optimizer smoke；再按固定合同提交正式17项。Kubeflow优先使用真实可用H100，HTCondor在每次提交前重新检查兼容schedd负载。随后补齐四分类配对/条件域/交互、选择两Pareto候选和三seed确认。最终全部终态、读回、评价、成本、图与总报告齐备后才验收S3。
