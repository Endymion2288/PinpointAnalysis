# Stage285：R3合同、M0执行与实际恢复验证

## 阶段目标

按roadmap v2完成R3的采样、局部保留和原生几何对照；先从M0分离训练顺序因素。当前M0已开始训练，R3完整科学结果尚未测量。

## 工作内容

复核R2最终验收、R3全部工单、模型与数据接口、Kubeflow实际状态和Condor实时schedd负载。新增`analysis.flavour.v2.r3`：全池重混sampler、allocation-local transformed cache、配置冻结、metadata preflight和实际runner恢复验证。共享R2固定预算训练器通过显式factory接入新dataset/loader/model，原入口默认行为保持同一数值路径。

M0从fresh D2 T4开始，唯一科学干预是每pass label-independent全局事件重混，保持原每update batch大小、总updates、曝光、LR和优化器配方。G0/G1/G2以及条件G3仍按已有路线实施，各干预分别配对。增加B1首层stride2适配和架构匹配的fit入口；真实拟合与GPU profile尚未完成。

## 详细数据

M0为300k四类pool×4 passes，T4曝光1.2M、非τ曝光900k、150,117 updates，width16、batch8、AdamW decay=.01、LR=.0003、seed42、BF16 encoder/FP32 head。全程global order seed=20261119+epoch。预定5% warm-up/cosine/5% floor，`sum(lr)=23.589806175917296`；全部逐update batch大小与原D2完全一致，逐行均四次曝光的metadata合同通过。

实际恢复测试使用既有32个Train事件构成的synthetic validation fixture，不使用实际Tune。连续运行与第4步中断后恢复的最终参数最大差为0；endpoint/best概率表和曝光表完全一致。该检查覆盖实际cache、global sampler、screen、checkpoint和resume，不作为有效学习或泛化证据。模型有效学习依据仍沿用R2 matching width16/BF16的真实拟合；G1新stride需另做匹配拟合。

M0在Kubeflow `pinpoint-r3-m0-k1`运行，Pod=`pinpoint-r3-m0-k1-ht9wc`，13:57:14 UTC获配H100 NVL MIG3g.47gb。访问正常，未等待整卡；同namespace的R2任务已完成。300k缓存数组10,425,150,000 bytes，构建145.583275 s；32个分散事件缓存前后全部模型输入一致。缓存只保留在allocation-local内存。完整validation峰值、终态资源及分类效果尚未测量。

运行环境为LCG108、Python3.12、Torch2.7.1/CUDA12.5，OMP/BLAS单线程。实际执行：

```bash
python -m unittest tests.test_flavour_v2_r1 tests.test_flavour_v2_r2 tests.test_flavour_v2_r3
python -m analysis.flavour.v2.r3.campaign freeze --root "$R3_ROOT" --r2 "$R2_ROOT"
python -m analysis.flavour.v2.submit --root "$R3_ROOT" --stage R3 \
  --name pinpoint-r3-m0-k1 --backend kubeflow --flavor h100-47gb --gpu \
  analysis.flavour.v2.r3.launch --root "$R3_ROOT" \
  --config "$R3_ROOT/configs/b1_m0_t4_s42.json" --attempt m0_k1
```

29项unit tests通过，包括global order每轮无重复/遗漏、逐步batch大小不变、label-independent、确定性及中途offset；B1默认adapter与旧模型逐参数/输出精确一致，stride2仅改变空间采样，初始化参数不变。真实恢复在上述实际MIG上通过。

`R3_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。完整合同、preflight、日志、模型和后续产物写入该目录；仓库保存[冻结合同](../roadmap/v2/configs/r3/campaign.json)、[M0配置](../roadmap/v2/configs/r3/b1_m0_t4_s42.json)和[执行状态](../roadmap/v2/r3_execution.json)。Cal-A/Cut-B/Confirm继续封存。

## 阶段性结论

事实：M0顺序干预的预算、缓存输入等价和实际恢复链路已验证，训练已启动。更细stride模型的参数可与原模型一一对应。尚无M0泛化或G1学习结论。

## 下一步计划

完成G0相位、G1真实小样本拟合和单卡并发profile；同步实现G2原生压缩前关系、候选覆盖及边数尾部诊断。M0达到预定终点后做同40k Tune配对，决定采样参考，再提交匹配G1。根据实际效果和成本冻结必要seed、粒度及后续结构对照，最终交付科学决策、英文图、终态资源和完整R3验收。
