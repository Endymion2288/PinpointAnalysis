# S4 磁盘读出模型的概率回放

## 阶段目标

确认最终评价使用的概率能够由保存的readout模型重新产生，再清理临时feature数组。

## 工作内容

完整CPU validation对使用feature bank的full/global frozen模型增加独立joblib读取。在完全相同的validation特征上重新计算四类概率，按run/chunk/row/label一一对齐预测文件，要求最大绝对差≤1e-10；四类顺序必须为0/1/2/3。模型或预测不同即停止完整评价。

feature加载接口增加可选择split的参数，使该回放只读取10k validation，避免重复加载100k训练表示。此步骤不训练模型、不使用新标签、不调阈值，数值驻点验收仍由原fit梯度证据负责。临时数组清理仍在完整评价之后。

## 详细数据

执行`python -m unittest tests.test_flavour_s4_solver tests.test_flavour_s4`，42项测试13.926秒全部通过。新测试先将同一预测逆序写入磁盘，重载回放误差为0，随后只修改一个概率0.001且保留全部事件键，实际触发拒绝。

LCG108 CUDA、Python3.12，OMP/BLAS/MKL一线程。真实30项磁盘读出回放尚未测量。05:03:46 UTC的真实训练仍正常：EMA3/MAE5已进入第四轮，Rel5第三轮；solver gate第二组完成420/1,000事件。

## 阶段性结论

事实：保存模型与评价概率的一致性已有独立数值回放路径，可检测仅靠manifest和事件键检查无法发现的概率改动。真实feature提取及完整回放等待严格solver gate。

## 下一步计划

放行features和CPU readouts，完成45组预测的全量统计与30项回放，保存误差和驻点证据后移除临时数组。
