# S6：条件选择区间与完整 test 依赖

## 阶段目标

为已经冻结的物理分层补充固定工作点的400次block区间，并保证正式统计只在完整test准备完成后执行。

## 工作内容

按block、true class聚合选择计数，在每次重采样内恢复固定prior；直接计算各类效率、FPR、纯度、resolved效率与reject率。这与显式复制block事件的统计完全相同，省去400×事件数的常驻权重。保持原分层、cuts和probability mapping。

新增 `pipeline evaluate_one`，等待 `evaluation/test_ready.json` 后运行指定模型/prior统计，便于批处理pilot与正式test转换重叠。GPU提交时检查candidate campaign含checkpoint列表，防止把数值推理policy误用为模型配置。

## 详细数据

验证命令：`python -m unittest tests.test_s6_evaluation tests.test_s6_calibration tests.test_s6_truth_geometry`，10项通过。新增独立参照测试显式复制block中的事件，包括缺类replicate与不均衡prior；所有区间与聚合实现一致。

Compact G4构建55.714秒。τ task32 nominal重放16/16 decay vertex有效，分支为hadronic11、muon4、electron1；observations、targets、ideal tracker三文件逐值相同，原16条unknown来自记录缺失。其余条件仍在运行。

补提GPU1582744在读模型配置时退出：提交命令误用了`s6_inference.json`，该文件只有precision/batch等policy，没有`models`。故障发生在模型加载前，没有新test分数；失败资源计入accounting，随后按原candidate配置恢复，不改分析冻结。

## 阶段性结论

固定工作点的条件区间有独立事件重采样参照。首个真实τ重放补齐了decay证据且探测器响应完全一致；全部条件覆盖仍待完成。

## 下一步计划

使用正确candidate配置补提GPU，执行完整test的统计pilot。完成compact truth pilot后扩展transport和generator重放，再合并truth sidecars与冻结分数。
