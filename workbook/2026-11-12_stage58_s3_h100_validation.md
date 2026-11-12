# 阶段58：H100验证通过与S3正式screen启动合同

## 阶段目标

完成真实H100 NVL/BF16完整事件与optimizer执行验证，满足冻结screen的启动条件。分类收益尚未测量。

## 工作内容

阶段57的混合精度packing修复已在H100验证。formal screen仍使用`roadmap/configs/s3_screen.json`的17个配置、seed42、80k/10k事件、4 epochs、80,064 updates和full-validation NLL选模。未修改训练合同、proxy值、split或calibration/test。

启动分配为主模型task5优先Kubeflow，其余16项HTCondor。已有Kubeflow环境probe仍未调度，将保留UID、Pod/事件及计费证据，再主动缩短其deadline以释放唯一Nvidia GPU配额；确认终态后才创建训练Job。其余配置不重复提交。若Kubeflow训练仍长期Pending，将先确认无writer并终止旧副本，再迁移到新检查选择的HTCondor schedd。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

profile1573833和smoke1573834均真实终态exit0，作业日志在`logs/`。环境为NVIDIA H100 NVL、Torch2.7.1、BF16。profile程序33.278923 s，实际slot allocation43 s；smoke程序17.566059 s、slot26 s。原失败1573830/1573831 allocation18/13 s分别保留。

profile覆盖8配置×3 train占用秩×inference/backward，共48行全部finite；包含原始36,549通道的最大事件。主模型该事件pixel+bar输入36,547 nodes、14,597 local blocks。下表为每配置压力样本的最大backward时间及最大峰值；第一空事件含初始化，a4 late的2.786288 s不能当稳态速度。

| Configuration | Max backward [s] | Max peak [MiB] |
| --- | ---: | ---: |
| a4 late | 2.786288 | 139.482422 |
| f16 joint | 0.399120 | 358.200684 |
| f16 context latent | 0.376943 | 359.939941 |
| f16 context latent 1 mm | 0.170165 | 224.499512 |
| a4 context latent | 0.091958 | 142.375977 |
| f64 context latent | 0.150896 | 195.847168 |
| f16 context latent tracker | 0.372618 | 371.243164 |
| f16 context latent q8 | 0.377853 | 360.032715 |

smoke覆盖tasks0/2/5/6/14/16，每配置16 train/16 validation、4个真实AdamW updates，全部finite gradients与四类预测正常。此tiny样本仅验证执行，不参与选模。产物`profile_h100_v2/summary.json`、`measurements.parquet`和`smoke_h100_v2/smoke.json`。

执行入口为`python -m analysis.flavour.s3.condor profile|smoke`，完整提交命令、schedd实时选择依据、code commit、wrapper与job submit分别在`submission_profile_h100_v2_141723.json`和`submission_smoke_h100_v2_141745.json`。最初27项CPU统计/模型/后端检查也已通过，见阶段57。

## 阶段性结论

事实：完整高占用事件与optimizer在冻结正式硬件/精度下可运行，首轮执行缺陷已修复并在H100验证。48行压力结果未出现非有限数值。正式17项screen可以启动。

假设：局部细化、module context或latent fusion能改善完整四分类仍未验证。没有把压力测试吞吐、tiny metrics或只读S2结果用作S3收益证据。

## 下一步计划

提交全部17项、监控首完整epoch和组织120 GPUh预算；全部4 epochs/零AMP skips和14项CPU读回通过后执行四分类screen评价与2×2交互，按预定规则冻结确认配置。保留主native/1 mm三seed、模态/几何/信号解释和英文图。S3最终验收前不执行S4–S6。
