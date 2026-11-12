# 阶段61：S3精确验收清单与完成输出读回监视

## 阶段目标

在全部screen训练期间落实逐项输出验收，避免用配对数量或泛化状态代替实际完成证据。S3尚未完成。

## 工作内容

`analysis.flavour.s3.accept`新增精确核对：单seed配对集合必须与冻结spec相等且无重复；三seed组的名字和每个seed配对必须完整对应spec；每项对照的raw/common support均须包含34项完整四类统计、finite point/95%区间和400个有效replicates。确认矩阵由已选择的两个结构及预定主native/1 mm推导，必须逐项包含seed17和2026的正确冻结配置。最终命令新增必传`--comparisons`指向独立冻结后的确认配对文件，不修改任何收益门槛。

新增`analysis.flavour.s3.monitor`，仅在manifest真实发布complete后调用已验证的S2 CPU readback，对best/last/optimizer/predictions/逐epoch记录完成14项检查；检查失败立即保留报告并退出，不将其写成完成。compact `monitor_state.json`记录各配置已完成epoch及最新四类点估计，默认每60秒观察，完整训练未结束不触发读回。该文件不是调度器存活或allocation证据，后者仍显式查询Condor/Kubernetes与accounting。

## 详细数据

结果根`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。验收逻辑6项检查真实通过，0.014 s，`unit_acceptance_inventory.log`；测试覆盖完整与缺失/重复配对、某类仅399个replicates、主native/1 mm缺少确认seed，以及原四类晋级/NLL/资源门槛。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_s3_decisions
python -m analysis.flavour.s3.monitor --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12 --campaign roadmap/configs/s3_screen.json
```

monitor的单次真实执行exit0，14:42:49 UTC快照为0/17 complete models、0/17 final readbacks，符合当前训练状态。此时a4 late完成首轮，NLL1.3770788120、macro AP0.2940118071；f16 late完成首轮，NLL1.3784285483、macro AP0.2912937667。其余模型仍在训练或排队，不能从这两个首轮点估计进行screen选择。

实时调度已重新查询：6项Condor H100 JobStatus2，包括预定主模型1573837.0；其他11项idle。Kubeflow slot `pinpoint-s3-anyqueue-slot-v1-pkmfx`仍Pending、无node，约11分钟。队列交接watcher PID1170732用真实`ps`确认存活，按15秒频率等待ready。未根据超时重提已有作业。

## 阶段性结论

事实：正式训练继续运行，两个late控制已进入第二轮；暂无完成的4-epoch模型。精确验收逻辑可识别不完整的配对和确认矩阵。监视入口已真实读取当前结果，不把未完成模型当作通过CPU读回。

假设：两个late模型首轮接近的AP不能回答细窗口或module context的收益。该结论需要全部冻结配置和三seed区间。

## 下一步计划

持续监视新完成模型并执行14项CPU读回；保留Kubeflow ready后接走idle任务的机制。全部17项终态后更新真实allocation、严格四类screen与2×2交互，按原规则冻结候选确认文件并提交，再进行确认训练。最终全套条件域、错误显示、成本、精确gate与中文报告完成前保持S3进行中。
