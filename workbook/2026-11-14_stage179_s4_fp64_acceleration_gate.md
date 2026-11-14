# S4 相同凸目标的FP64设备加速验证

## 阶段目标

100k full Newton-CG在CPU上持续计算超过半小时，尚未产出中间迭代。验证能否仅加速同一目标的gradient/Hessian products，并保留原1e-6独立CPU驻点阈值。

## 工作内容

新增`accelerated_linear`：沿用scikit-learn 1.5.2的Newton-CG、200 outer/200 inner上限与Wolfe line search，解析multinomial梯度及Hessian-vector product用Torch FP64在设备计算。保留原fit-only StandardScaler的float32变换舍入后再转FP64；目标仍为mean NLL加`||W||²/(2 C n)`，C=1且intercept不正则。每次outer迭代记录梯度、HVP次数和耗时。最终系数保存为普通sklearn pipeline，驻点由既有CPU NumPy/SciPy公式独立重算。

正式gate配置为[8项实际对照](../roadmap/configs/s4_fp64_acceleration_gate.json)：Scratch/MAE3、1k/10k、full/global。直接复用已完成CPU readout的同一feature bank和fit事件键；通过条件为双方CPU梯度≤1e-6、目标差≤1e-5、最大fit概率差≤0.005。此处不读取validation标签作选择。GPU gate尚未测量，当前五组CPU100k任务继续执行，没有替换正在运行的冻结配置。

## 详细数据与验证

`tests.test_flavour_s4_accelerated_linear`的2项测试通过（0.425s）。独立Torch autograd检验loss、梯度与HVP至1e-13；含相关features的四分类拟合与sklearn相同目标比较通过，最大迭代1的未收敛实例不能通过。测试使用CPU，不能代替真实CUDA gate。

正式提交使用已提交source快照、Kubeflow GPU flavor和batch campaign；该任务只读小型feature bank，不重复复制native事件。所有新增allocation计入同一60 device-hour账本。后续若采用加速器，须新名称和冻结配置，并保存验证证据；旧CPU结果不改写。

## 阶段性结论与下一步计划

事实：解析公式和CPU执行与独立自动微分/既有求解器一致。CUDA实测、吞吐和100k收敛尚未测量。先运行实际8项gate，依据真实结果决定是否加速剩余100k读出；保持现有CPU任务直到替代产物通过。
