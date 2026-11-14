# S4 冻结读出任务的运行阶段记录

## 阶段目标

区分GPU特征提取和CPU读出求解，定位长任务耗时并安排后续并发。

## 工作内容

为后续冻结probe每30秒覆盖写入一次小型`progress.json`：分别记录fit/validation已提取事件数和吞吐，读出求解时记录branch、维数及CPU求解标志，完成后保存终态。每个branch结束输出收敛迭代数与fit/heldout NLL。不改变features、fit标签、标准化、求解器或超参数，不修改运行中的冻结任务。

## 详细数据

`/usr/bin/python3 -m py_compile src/analysis/flavour/s4/probes.py`和`git diff --check`通过。本阶段仅增加运行记录，不新增预测或性能测量。此前100k两个converged probes仍在正式Condor运行；只读查看约58分钟时两个worker均存在、CPU占用约75%，尚无最终收敛结果，不能根据任务存活判定学习或solver通过。

## 阶段性结论

后续任务会直接记录GPU提取与CPU求解阶段，减少将求解耗时误判为GPU前向耗时的可能。当前旧任务没有这些阶段记录，具体剩余时长未测量。

## 下一步计划

10k inner结束后优先正式补位1k refits及正常readout恢复，shared claims避免与Condor重复。依据新任务记录判断CPU/GPU瓶颈，完成收敛和分类评价后再解释表示学习效果。
