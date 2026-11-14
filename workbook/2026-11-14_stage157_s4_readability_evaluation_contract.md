# S4 修复后frozen readability完整评价

## 阶段目标

完成不依赖LP冻结/解冻的表示可读性对照，比较随机初始化、MAE3预训练和阶段145正常fine-tuning，并保留数值优化与分类统计的边界。

## 工作内容

冻结[12项fit的评价配置](../roadmap/configs/s4_repaired_readability_evaluation.json)、[18项同事件配对](../roadmap/configs/s4_repaired_readability_pairs.json)及[正式CPU任务](../roadmap/configs/s4_repaired_readability_validation.json)。六项frozen encoder各输出full/global两种readout，另含六项seed42的阶段145修复监督模型，总18套完整10k预测。

每个预算报告full/global的MAE3−Scratch，两个初始化分别报告full−global，以及fine-tuned−full frozen。Frozen高维Head、固定StandardScaler+Logistic Regression及完整fine-tuning的训练方法不同，这些对照反映可读性和训练协议，不能当作重训的branch ablation或因果硬件重要性。

评价前新增readout验收：冻结manifest/config一致；每个branch求解器passed、无ConvergenceWarning、系数finite、fit样本数等于原预算；预测事件keys/label与原完整validation逐项一致。仅对既有接受目录核对输出，不重做sample认证。Phase155失效LP模型不进入本合同。

## 详细数据

固定条件：预算1k/10k/100k、seed42、同detector-proxy和family split；C=1、tol=1e-5、max_iter=5000，不用validation调C。完整10k四分类、raw/common-support、τ子道及低FPR实数计数、400次connected-chunk配对bootstrap均采用现有实现。

截至03:17 UTC，1k/10k四项converged probes均已complete，各full/global迭代数分别为Scratch1k 580/323、MAE3 1k 732/414、Scratch10k 2232/296、MAE3 10k 2849/1024；均无收敛告警。100k两项仍在运行，完整18预测的配对统计尚未测量。仅seed42，物理块区间条件于这些权重，不替代训练seed重复。

两项新增针对性验证实际运行0.102秒通过：求解告警即使被错误标成passed也被拒绝；改变预测catalog row被拒绝；正式等待遇到optimization-review状态立即失败。`git diff --check`通过。此前probe优化规则不变，新验收只检查其已保存的证据。

LP修复相关真实CUDA gate仍排队：cluster1578805/1578806已降低为2 CPUs、8,000 MiB，依据是预载八个事件、torch单线程且无多进程dataloader，GPU产品和科学合同不变。十二项新inner依赖gate的正式提交为1578808（两并发）和1578809（单并发），均在bigbird20；已检查的终态与所有实际allocation继续计入原60 device-hours账本。

## 阶段性结论

事实：frozen readability评价合同与收敛/事件输出验收已固定，相关两项检查通过。现有1k较高的probe NLL仍需结合拟合误差和正则解释；不能据此断言encoder或硬件信息无效。

待测：全部三个预算的配对结果、100k收敛状态、完整成本。LP修复效果和重新选择仍属于独立待完成任务。

## 下一步计划

正式提交readability CPU评价，等待100k两项结束后执行。继续优先取得LP编译gate，并将空出的Kubeflow单元接续到gate与12项重训，随后恢复两个预算的OOF选择及重拟合。
