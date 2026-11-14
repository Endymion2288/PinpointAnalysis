# S4 优化重拟合的配对评价合同

## 阶段目标

检验train-inner选出的1k LP-FT方案能否改善阶段145观察到的过拟合，同时测量MAE3相对Scratch的排序、概率、τ硬决策和低FPR表现。完整预算模型固定第8轮，评价不再选择checkpoint。

## 工作内容

固定[12模型评价配置](../roadmap/configs/s4_optimized_1k_evaluation.json)、[配对合同](../roadmap/configs/s4_optimized_1k_pairs.json)及[CPU评价任务](../roadmap/configs/s4_optimized_1k_validation.json)。六项阶段145模型与六项优化重拟合沿用原训练配置，全部采用修复后的revision 3；`initialization`保持真实含义。

增加独立的`comparison_group`用于绘图和精简统计。原默认按initialization分组会把同预算、同初始化的旧/新方案合并为六个seed，掩盖方案差别；现在四组分别保留三个seed。评价记录每个模型真实的checkpoint选择规则及epoch，精简报告不再把固定末轮写成validation-NLL选择。

## 详细数据

- 样本：相同完整10k validation，四类各2,500事件；原family划分和detector-proxy直接沿用。Calibration/test未读取。
- 合同：12项单seed配对、4项三seed均值比较，分别为旧MAE3−Scratch、新MAE3−Scratch、新−旧Scratch、新−旧MAE3。
- 区间：400次相同物理connected-chunk bootstrap；同时报告raw/common-support、τ electron/muon/hadronic分支、四类1%/0.1% FPR的实际通过计数及计数区间。训练seed SD与条件物理块区间分别报告，seed均值不是ensemble。
- 选择差别：旧模型使用validation最低NLL，优化模型的配方及轮数来自原1k标签预算内的train-inner OOF NLL。新旧比较包含训练长度、优化配方及checkpoint选择方式的共同改变，不拆分归因。
- 新方案在查看旧validation后设计，本次沿用validation属于条件性重复评价；该范围已写入机器可读评价合同。
- 评价前执行严格checkpoint/config读回、逐epoch曝光检查、训练梯度/激活汇总和所选权重的独立FP32 backward。LP阶段只有线性readout应有梯度，解冻后检查informative encoder梯度。

运行环境为LCG_108_cuda、Python 3.12、PyTorch 2.7.1。命令`python -m unittest tests.test_flavour_s4`实际运行30项测试，15.036秒全部通过；新增测试用相同initialization、相同seed的不同方案验证真实绘图CSV与精简报告均保留两组，并检查固定末轮元数据。`git diff --check`通过。

02:44 UTC live records：Kubeflow六进程继续10k inner；Condor重拟合2204899已Running，两个进程开始1k重训；frozen probes 1578756已Running，并开始10k预算的两项特征提取。1k新重拟合的完整validation收益尚未测量。所有产物继续写入`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`，预定评价目录为`evaluation/optimized_1k`。

## 阶段性结论

事实：配对、分组和checkpoint选择范围已经固定，评价代码30项测试通过。训练内最优值与资源获配均不能替代完整预算的分类证据。

待检验假设：LP-FT及减少曝光有助于缓解1k过拟合，且能保留MAE排序收益。新的NLL、τ决策和低FPR改善尚未测量。

## 下一步计划

正式提交CPU评价任务，在六项重拟合完成后执行完整读回及配对评价。10k的24项inner训练齐全后独立解析共同配方和末轮，再生成三seed重拟合及同结构评价合同。完成新路由的frozen readability后，继续EMA/partial Rel及matched-compute比较。
