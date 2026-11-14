# S6：完整通道推理接口与 checkpoint 回放准备

## 阶段目标

为全部 calibration/test 准备共享输入变换的多 checkpoint 推理；先在既有 validation 回放检查数据接口、数值精度和模型内部输出。Test 尚未开放。

## 工作内容

`s6/infer.py`直接严格载入13个既定 checkpoint，不重新载入预训练初始化。只读取 observation 做模型输入，标签及事件键仅作输出连接。按 pixel/bar/tracker、native/1 mm 分成三个 representation 组，同一组各模型共享一次完整通道变换；保留所有通道。输出每事件四类概率及 S5 的 vertex/kinematics/charm 预测，首批记录内部 activation 与 query 诊断。

推理执行模式先比较 eager/compiled、FP32/BF16。正式 calibration/test 使用同一冻结模式；不能将精度差异未核清的分数混入最终结果。Replay 的既有容差沿用0.003，不因本次失败放宽。小样本选择补充各类 observation occupancy 尾部。

## 详细数据

T4、PyTorch2.7.1、eager FP32，16个既有 validation 事件、batch4、workers2，13个模型共10.458 s（含读取与首批诊断），GPU peak194.68 MiB，父进程 peak RSS1397.02 MiB。三个 representation 共10444/10164/10397个节点。各模型 head LayerNorm 输出 RMS 0.967–1.094，head 激活未出现非有限值；该小样本不代替完整压力验证。

本次回放最大概率差：MAE seed17 0.002104、seed42 0.004582、seed2026 0.003070；Scratch 三seed分别0.001405/0.002777/0.001698；S5三项0.002297/0.002679/0.003156；S3 1 mm/native/tracker为0.001182/0.008585/0.005663；Rel-channel5为0.004159。6/13超过0.003，因此结果为`review_required`，未用于 calibration 或 test。已有分数在支持 BF16 的 GPU 上产生；精度/执行模式差异是待核实原因，不能据此认定架构错误或通过回放。

产物：`results/flavour/s6_2026-11-14/audit/t4_eager_replay16.json`。命令为`python -m analysis.flavour.s6.infer replay --root <S6> --campaign roadmap/configs/s6_candidates.json --source <S5> --precision fp32 --workers 2 --batch-size 4 --events-per-class 4 --name t4_eager_replay16`。

Calibration 转换 pilot1582692已进入bigbird20的worker运行，完整分块吞吐和RSS尚未测得。papawan AFS可列名的GENIE科研目录内实际文件读取均返回Permission denied；public旧GENIE输入可读且具有neu/tgt/Ev等metadata，Pythia6 Standard/NUX旧schema需桥接，尚未获得兼容的独立generator响应。

## 阶段性结论

多 checkpoint 推理和真实事件内部输出可运行；T4 FP32回放未通过固定容差，需要在原 BF16 模式下定位差异。尚无最终 S6 test 或 generator 对照结论。

## 下一步计划

申请可用的BF16 GPU执行包含occupancy尾部的回放，并检查batch/order与eager/compiled一致性；通过后固定正式执行模式。转换pilot完成后按测得资源扩大calibration转换。继续调查generator schema及真实paired G4 transport接口。
