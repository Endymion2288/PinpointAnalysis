# S4空响应初始化梯度诊断与成对修复矩阵

## 阶段目标

解释梯度范围图中有限但异常大的初始化梯度，修复其程序原因，并为新的成对重新训练冻结配置。训练完成、有限参数或非零梯度不能替代这一检查。

## 工作内容

扫描已有全部训练健康记录：S3 revision2为8,020次计划采样，未发现单组参数梯度平方范数超过10⁸；S4在本次扫描时为4,805次，两项Scratch在step1出现异常，其他已记录采样未出现同量级异常。这个范围只覆盖实际采样，不能证明未采样的所有updates均正常。异常项为`s4_scratch_n10000_s17`与`s4_scratch_n1000_s2026`。

新增`analysis.flavour.s4.gradient_replay`，用原seed、原sampler第一个batch、CPU FP32及eval模式重放初始化，分别反传空事件损失、有响应事件损失和全batch损失；损失分解保留全batch分母。每个新旧路由从同一初始化出发，记录LayerNorm输入方差、参数梯度、全局裁剪系数和首个warmup AdamW参数变化。诊断只读取train，不拟合validation。

确认原因是`plane_tokens`为空事件构造的一行零占位节点，其collate local mask为true。零bias初始化使该节点进入多次零方差LayerNorm和Q/K normalization。LayerNorm在零方差方向的Jacobian可含`1/sqrt(eps)`增益（eps=10⁻⁵时约316.23），多层组合可放大梯度。它不携带真实硬件响应，随后global clipping却缩小整个模型的梯度。

新路由使用`empty_response_bypass=true`：local mask排除occupancy=0的占位节点、空local输出归零、regional/global读取屏蔽detector type3的假token。`StableAttention`在source全空时只跳过attention残差更新，保留query residual/FF；SDPA使用数值安全key并在输出处断开该更新，避免all-masked实现差异。空事件仍参与分类，使用summary/presence和可学习global prior。未增加参数、未删除事件；没有修改旧配置加载行为。非空batch新旧输出逐位相同的测试通过。

联网读取[ICML2020 Xiong等](https://proceedings.mlr.press/v119/xiong20b.html)和[ICML2025 Peri-LN](https://proceedings.mlr.press/v267/kim25u.html)正式来源页（HTTP200）。它们支持检查归一化位置、方差和梯度传播，但不直接证明本空占位病因；本结论来自损失路径分解。PyTorch文档请求返回HTTP403，未将其写成已读取来源。来源记录位于输出根`literature/empty_response_sources.json`。

## 详细数据

输出根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。完整事件keys与分解记录：`audit/initial_gradient_decomposition_cpu.json`。

| Original fit | Empty / batch events | Old full-batch preclip norm | Repaired norm | Old / repaired input gradient squared norm |
| --- | ---: | ---: | ---: | ---: |
| Scratch 10k seed17 | 10 / 16 | 6.4264979×10¹² | 159.475891 | 4.12998568×10²⁵ / 54.530282 |
| Scratch 1k seed2026 | 2 / 16 | 1.8054894×10¹² | 59.649490 | 3.25979063×10²⁴ / 163.941600 |

旧路由的空事件损失单独产生同量级全部input梯度；修复后的空事件input梯度平方范数精确为0。有响应事件损失的完整参数gradient norm新旧相同，分别为29.181843、40.092079。修复屏蔽了虚假路径，同时保留真实事件学习路径。

旧路由全局clip系数为1.55606×10⁻¹³和5.53866×10⁻¹³；同warmup学习率AdamW的Head首步参数变化平方范数分别为4.55611×10⁻²³、7.14665×10⁻²²，修复后为3.92457×10⁻⁷、3.97576×10⁻⁷。这说明裁剪保证finite并不保证所有分支获得有效更新。重放为FP32/eval，训练实际使用BF16/Dropout，二者数值不要求相同；已有实际BF16 step1 input梯度平方范数为4.20649×10²⁵、4.00931×10²⁴。

执行命令（LCG108 CUDA环境、`PYTHONPATH=src`）：

```bash
python -m analysis.flavour.s4.gradient_replay \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 \
  --names s4_scratch_n10000_s17 s4_scratch_n1000_s2026 \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/audit/initial_gradient_decomposition_cpu.json
python -m unittest tests.test_flavour_empty_response tests.test_flavour_s3_repair \
  tests.test_flavour_s4_acceptance tests.test_flavour_s4_paths
```

21项测试通过，包括非空输出与state dictionary兼容、空事件梯度隔离、混合batch独立性、FP32/BF16空事件backward、有限巨大梯度拒绝和既有架构/验收检查。正式新模型训练与指标：未测量。

## 阶段性结论

确认存在空响应占位引起的程序性初始化梯度缺陷。原“finite且非零”的训练标记漏掉了它；现在验收另检查裁剪前norm，原两项不通过该检查。此前stage133/134的三seed S4方法收益暂停作为结论，原数值保留为历史测量，不能用现有checkpoint的健康backward覆盖初始化异常。

配置`s4_empty_pilot.json`与`s4_empty_repair.json`冻结新gate和18次重新训练。所有模型在前32 updates记录健康，此后每100次采样；每个update都在裁剪前检查norm≤10,000，超出即留证、停止并审查。此阈值是数值审查边界，不是性能筛选阈值。新S4验收同时独立重建采样范数，不相信单一通过标记。

三套MAE3预训练已完成且当前采样未见本量级异常，新比较复用其权重并对所有下游组应用同一空响应路由，不声称预训练本身改用了该路由。全部事件标签、nested子集、sampler、更新预算和checkpoint选择沿用原合同。此次协议在看到旧S4结果后设计，后续validation仍是同一数据集的条件评价。

## 下一步计划

先在H100/A100原生BF16完成新真实32-event拟合gate及最大占用压力验证，再执行18项Scratch/MAE3成对重训和完整10k validation。完成14项checkpoint读回、独立best FP32 backward、梯度全范围图、逐类/τ指标与三seed物理块bootstrap后再更新结论。旧矩阵剩余任务继续完成诊断，实际GPUallocation全部计入原60 device-hour上限；不重新认证用户已接受的samples或detector proxy。
