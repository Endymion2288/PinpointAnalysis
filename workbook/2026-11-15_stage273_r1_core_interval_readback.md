# R1：完整配对核心区间的独立算术读回

## 阶段目标

在最终交付中核对nominal配对差值、percentile区间和三seed波动。原有独立审查已读回逐模型指标、人数、曝光和压力对照；本阶段补齐完整nominal核心区间的独立算术，并在已有实测结果上验证。

## 工作内容

新增`analysis.flavour.v2.r1.interval_review`。复用冻结physical-block设计与相同400次RNG draws，直接对逐事件正确判定、log-loss和逐类权重求和；每draw重新等类加权，不调用`Metrics/scalar_summary/interval`。对每个配置配对重算点差及95% percentile端点，三seed组再重算逐seed差、平均区间及sample SD。

检查的七项为BA3、NLL3、e/μ/NC recall及BA4、NLL4。不适用的四类比较要求保持null。其范围是核心统计量的独立算术复核，block设计及抽样本身沿用冻结实现；其他工作点及压力结果仍由原有指标/计数审查覆盖。

`review --require-full`纳入该检查，输出`evaluation/independent_core_intervals.json`；增强accept要求该结果完整、400 draws、全配对和21项mean-seed组；最终summary同时导出审计摘要。没有修改模型、预测、bootstrap合同或晋级门槛。

## 详细数据

先以已有`interim/b1_three_seed_compute`的24模型、56配对、12项三seed比较进行实际验证，涉及40个model-domain。

| Validation item | Measured result |
| --- | ---: |
| Independent checks | 1,156 passed; 0 failed |
| Largest absolute point difference | 6.995e−15 |
| Largest absolute interval-endpoint difference | 7.161e−15 |
| Largest absolute seed-SD difference | 3.484e−15 |
| Replicates | 400, same frozen physical-block draws |
| CPU operation time | 31.379070 s |

任务1582933在`bigbird20.cern.ch`执行，申请1 CPU、3GB，无GPU。提交前自动比较schedd负载与EOS/WRITE兼容性，选中最少idle/held/running的兼容schedd，证据在submission JSON。最终完整41模型复核尚未执行。

新增数值测试明确检验：每draw须重新等类加权；重采样中完全缺失的类不能进入BA/NLL分母；逐类recall保留不支持状态。接入完整review/accept/summary后，**27项R1与summary tests通过，.875 s**。

此时R1为**40/41**：seed17 compute实际wall4,983.629454 s，相对目标+0.00107093%，382,172次曝光，BA3=.5480、NLL3=.9074658511；allocation1582928已终态Exit0，完整14,182 s、5 CPU。seed2026最后compute已停止优化、379,402次曝光，正在full Tune推理；其最终wall、指标和验收尚未测量。Cal-A/Cut-B/Confirm保持封存。

## 验证命令与产物

```bash
python -m unittest discover -s tests -p 'test_flavour_v2_r1*.py'
python -m analysis.flavour.v2.r1.interval_review --root "$R1_ROOT" \
  --config "$R1_ROOT/interim/b1_three_seed_compute/evaluation_config.json" \
  --evaluation "$R1_ROOT/interim/b1_three_seed_compute/evaluation/results.json" \
  --output "$R1_ROOT/audit/interval_validation_stage273.json"
git diff --check
```

第二条为1582933的实际batch payload；提交通过`analysis.flavour.v2.submit --name r1_interval_validation_stage273 --cpus 1 --memory 3GB --job-flavour microcentury`执行。审计、日志和不可变source snapshot在`R1`下，`R1_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。

## 阶段性结论

**事实：**已完成子矩阵的BA/NLL/recall配对及平均区间与独立逐事件算术一致。新复核已接入最终流程，补充科学结果表的数值证据。**尚未完成：**最后compute、完整矩阵统计、全图目视、终态成本与accept；不以子矩阵验证宣称R1完成。

## 下一步计划

等待seed2026 compute完成后由唯一finish控制器启动完整评价。审查41模型、133配对和21项平均配对的全量读回与新增核心区间复核；检查所有英文图，关闭CPU/GPU allocation并执行accept。全部通过后才导出最终R1结果和研究结论，不启动R2训练。
