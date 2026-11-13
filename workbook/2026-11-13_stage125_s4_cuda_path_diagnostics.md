# S4 CUDA FP32路径诊断与真实输入等价检查

## 阶段目标

缩短四个固定checkpoint在完整10k validation上的分支依赖诊断时间，并验证共享压缩输入可以用于后续正式worker。

## 工作内容

`functional_probe`增加CPU/CUDA设备选择。完整encoder和cached Head重放使用同一设备的FP32，关闭AMP、TF32和compilation；context和概率移至CPU计算配对统计。原始forward与cached Head保持`rtol=1e-5, atol=2e-6`检查。报告记录设备、precision和本进程GPU峰值，绘图读取实际precision。

配置`s4_functional_gpu_probe.json`冻结Scratch/MAE3、1k/10k、seed42四个模型，全部10k validation，三个presence-preserving mappings及400次connected-chunk bootstrap。拟在正式获配的Kubeflow父Pod中并行执行固定权重推理；父Pod UID、源码、命令及终态另存audit，device-hours只按父allocation计一次。父Pod其余训练保持原配置。完整路径结果：未测量。

## 详细数据与验证

LCG108/Python3.12运行`python -m unittest tests.test_flavour_s4_paths tests.test_flavour_s3_paths`，6项通过，0.074秒；两个修改模块的Python编译检查通过。

当前交互节点Tesla T4执行三个合成事件（含一个空事件）的CUDA FP32 smoke test：完整forward与cached Head的最大logit误差为0，分支置换后的logits全部有限，AMP/TF32关闭。这验证CUDA执行接口，完整validation诊断仍待正式作业。

共享暂存真实profile的HTCondor cluster1577031（bigbird20）已exit0，allocation49秒，程序26.669秒。四个固定shards为000/013/026/039，每shard8事件、两次交替EOS/local读取；所有native tensors和metadata逐位一致。首次复制214,994,816bytes，0.689秒；native读取/解码/拼接为EOS12.915秒、local12.676秒。decode分别12.017/11.887秒；本次I/O主要耗时仍在decode，不能据此宣称端到端训练显著加速。

产物位于`results/flavour/s4_2026-11-13/experiments/s4_native_staging_profile_v1/`。正式提交命令为`python -m analysis.flavour.s4.submit --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 --backend condor --stage staging_profile --name s4-native-staging-profile-a --campaign roadmap/configs/s4_staging_profile.json --tasks 0 --apply`，冻结源码b9b2246。

## 阶段性结论

共享暂存保留真实native输入与事件顺序，可供后续worker使用。CUDA FP32路径诊断接口已通过小规模重放检查；分支实际贡献及CPU/CUDA完整样本差异尚未测量。

## 下一步计划

启动有父allocation审计的CUDA固定权重诊断，获得完整结果后回收旧CPU诊断，保留已花资源记录。新worker优先领取独立MAE下游seeds，再完成剩余frozen、Rel/EMA与matched-compute控制。
