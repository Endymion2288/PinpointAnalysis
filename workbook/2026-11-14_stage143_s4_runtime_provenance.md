# 修复重训的首对完成与诊断执行位置

## 阶段目标

推进18项冻结重训，保证完整validation分支诊断记录实际GPU allocation，继续缩短尚未启动任务的排队时间。

## 工作内容

清除`configs/s4_empty_functional.json`及`configs/s4_empty_dispatch.json`中继承的旧`pinpoint-s4-recovery-c`执行位置。该字段未参与模型计算，但已不适用于新正式batch作业。诊断manifest与最终JSON改为从实际claim、hostname、Condor job ID和Kubernetes环境记录backend及allocation；资源时长仅向父submission计费一次。dispatch前18项配置与原修复矩阵逐项相等，训练数值实现保持冻结。

## 详细数据

2026-11-14 00:43 UTC左右，首对10k seed42模型完成：Scratch/MAE3最佳epoch均为3，validation NLL分别为1.2777388766和1.2534117277。各完成10,048次更新和逐步梯度检查，最大preclip norm分别为186.1028137和38.9563293。这是单seed中间观测，完整18项、9配对和三seed平均评价未测量。

Kubeflow主Pod继续运行6并发，两个H100 24GB MIG Condor allocation各运行2并发；完成任务后由claim队列补入剩余任务。00:41:22 UTC账本累计21.398000 device-hours和253.917833 CPU core-hours；60 device-hour上限包含本轮此前全部失败和初始化。Kubeflow resourcequota显示整卡资源requests.nvidia.com/gpu为0/1，主训练使用47GB MIG。新的整卡候选将与Condor竞争领取未开始任务，依赖完成后执行正式GPU分支诊断；GPU获配与终态仍待实测。

验证命令：LCG108 CUDA环境、仓库src加入PYTHONPATH后，`python -m unittest discover -s tests -p test_flavour_s4_paths.py`，3项通过；`py_compile`通过。JSON检查确认dispatch前18项与`s4_empty_repair.json`完全一致。

## 阶段性结论

首对修复模型完成了固定更新数，未重现空占位导致的巨大梯度。是否改善四分类、τ子道和各分支实际贡献，继续由完整冻结评价回答。

## 下一步计划

完成18项训练、checkpoint与FP32 backward读回、完整10k validation及4模型分支诊断；随后收取所有终态、清理本任务已无用提交文件，更新资源账本和最终研究结论。
