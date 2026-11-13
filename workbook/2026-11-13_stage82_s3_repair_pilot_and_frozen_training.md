# S3 修复先导、完整事件压力测试与重训矩阵

## 阶段目标

回答修复架构是否恢复梯度与真实样本拟合、是否完整处理最大占用事件。正式四分类性能仍须全量重训评价。旧 S3 方法结论继续撤回。

## 工作内容

在 stage81 的 Peri-LN、LayerScale、QK normalization、多尺度 region 与独立 local readout 基础上，完成三个真实 train 小样本拟合及六种配置压力测试。补充 query 相似度、query variation 与有明确采样范围的 attention entropy。健康检查区分空输入批次与非空输入梯度断开，resume 恢复连续零梯度计数。ParT reference 的 manifest 明确描述原 ParT body/head，避免误写为 V2 归一化 Head。

冻结 `roadmap/configs/s3_repair_screen.json` 的 8 个 seed42 对照，以及 `s3_repair_confirmation.json` 的 12 个确认训练。确认结构事先选定 fixed native、multiscale native/1 mm/tracker/region8、fresh ParT，各加 seed17/2026。14 个单 seed 对照和 9 个三 seed 对照分别在两个 pairs 文件定义。所有模型 80k accepted train、10k validation，4 epochs、batch8、每 epoch 10,016 updates，合计 40,064 updates；warmup500、AdamW 0.0003、逐 update cosine、clip1。同新配方 ParT 是主要参照；不能把旧故障配方与新配方之差当作纯架构收益。Calibration/test 未使用，S4–S6 未运行。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13`。

32 个固定非空 train 事件、每类8个、balanced batch8、768 updates：

| Model | Accuracy | NLL on 32 train events | Distinct score vectors | Pilot elapsed s |
|---|---:|---:|---:|---:|
| normalized fixed | 1.00000 | −0.0 in FP32 score computation | 32 | 59.6171 |
| multiscale + direct | 1.00000 | −0.0 in FP32 score computation | 32 | 58.2219 |
| fresh ParT | 1.00000 | 0.00000165776 | 32 | 72.3806 |

三份 `pilot/<name>/gate.json` 均通过5项先导要求，包括末次非零 encoder 梯度；这只证明可优化与可拟合。首次 fixed/multiscale 运行被健康检查中 `isfinite.float().mean()==1` 误报，GPU 全1求均值可能舍入到0.99999994；改用 boolean `isfinite().all()` 后重跑通过。原错误证据保存在 `audit/pilot_health_false_positive/`，失败分配也计费。

H100 NVL MIG 3g.47gb、PyTorch2.7.1、BF16 压力测试，每结构3个 train 占用秩事件，含空、居中和最大占用；每事件 inference/backward。下表为最大占用事件的反向实测，MiB 是单进程 `max_memory_allocated`，不含其他 CUDA 进程、context 或批量8的总显存。

| Configuration | Represented nodes | Local blocks | Module groups | Regions | Peak MiB | Input gradient squared norm |
|---|---:|---:|---:|---:|---:|---:|
| fixed native | 36,547 | 14,597 | 31 | 479 (unused by fixed encoder) | 475.0063 | 720.0465 |
| multiscale native | 36,547 | 14,597 | 31 | 479 | 231.4648 | 939.4333 |
| multiscale 1 mm | 8,417 | 6,169 | 31 | 215 | 209.1333 | 3123.3225 |
| multiscale + tracker | 36,549 | 14,599 | 33 | 481 | 231.4741 | 951.6133 |
| multiscale region8 | 36,547 | 14,597 | 31 | 1,845 | 260.5044 | 856.3480 |
| fresh ParT | 36,547 | 2,641 | 31 | 105 (unused by ParT) | 222.9165 | 1.02358 |

六结构全部18次 forward、18次 backward finite。原始占用列为 pixel+bar+tracker 共36,549；不用 tracker 的表示保留其中36,547个观测节点。非空压力样本 encoder 梯度非零，空输入梯度零属预期。Region32/8 各保留958/3,690个区域 tokens，另有全部 local blocks 旁路；不再用36,547→24作为唯一信息路径描述。原 profile 中 `encoded_regional_tokens` 对 fixed/ParT 也机械写入了区域数，修复其字段语义；上表明确未使用项。

实际命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s3_repair tests.test_flavour_s2_training tests.test_flavour_s3_dispatch
python -m analysis.flavour.s3.backend --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13 --name pinpoint-s3r-occupancy-stress-v1 --flavor h100-47gb --campaign roadmap/configs/s3_repair_pilot.json --tasks 0 1 2 --concurrency 3 --program profile --deadline 7200 --apply
python -m analysis.flavour.s3.backend --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13 --name pinpoint-s3r-occupancy-extra-v1 --flavor h100-47gb --campaign roadmap/configs/s3_repair_screen.json --tasks 3 4 7 --concurrency 3 --program profile --deadline 7200 --apply
```

最新 focused tests：14项，11.086 s，通过；含 BF16 backward、极端 Head/projection 尺度、完整 region 覆盖、空/缺失模态、batch isolation、旧 GELU 断开捕获、空批次健康检查、warmup resume。`git diff --check` 通过。

预算：10:41:32 UTC 的 terminal allocation 快照含前3个 jobs，本轮0.0583333 device-hours，加旧 S3 上界31.2382687，累计31.2966020；额外压力 job 的 terminal allocation 需登录恢复后刷新。120上界不变，S2成本单列。并发进程只分摊一次预留时长。

## 阶段性结论

事实：修复模型恢复真实 train 拟合与非零梯度，六结构能够处理最大占用测试事件。先导结果不能用作泛化、native增益、tracker增益或瓶颈收益结论。完整 validation 指标、三 seed spread 与物理事件CI目前未测量。

## 下一步计划

启动8个冻结 screen 对照，按实际 GPU 利用率与 CPU/I/O 调节并发；检查早期健康和第一轮完整 validation。完成12个事先冻结的确认训练。随后运行完整 checkpoint readback、34项四分类指标、配对 bootstrap、条件域/工作点、内部健康与压缩轨迹及英文图复核。只有健康重训结果可进入本轮方法结论。
