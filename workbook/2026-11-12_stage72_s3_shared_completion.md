# 阶段目标

完成同一H100 MIG分区内的五项S3 screen训练，提高已获配资源的使用程度，补齐四分类评价并正确计算共享成本。

# 工作内容

task 11/12/13/15/16在同一个`pinpoint-s3-anyqueue-mig47-v3` Pod内运行；没有创建第二个GPU Job，原HTCondor idle副本已移除。五项全部完成4 epochs、80,064 optimizer updates、零AMP skips。17项screen模型均完成CPU checkpoint、optimizer、预测及冻结配置的14项readback，共238项检查通过。

Job于2026-11-12 22:44:30 UTC Complete，Pod退出并释放GPU。旧手动并发执行曾出现DataLoader/resource-tracker暂停带来的共同停顿，以及resource-tracker阻塞主进程最终退出；恢复后训练与退出正常。相关占用全部纳入成本。后续新增`analysis.flavour.s3.parallel`作为父进程，等待所有子训练退出，保留独立日志与起止时间。

# 详细数据

输入沿用accepted 80k train/10k validation，calibration/test未使用。Torch2.7.1、BF16、冻结seed42及campaign保持不变。五项实际GPU名称为`NVIDIA H100 NVL MIG 3g.47gb`。

| task | treatment | macro AP | macro F1 | balanced accuracy | NLL |
|---:|---|---:|---:|---:|---:|
| 11 | no pair | 0.250000000 | 0.100000000 | 0.2500 | 1.386298204 |
| 12 | pixel only | 0.250000000 | 0.100000000 | 0.2500 | 1.386298194 |
| 13 | bar only | 0.251354399 | 0.125677402 | 0.2539 | 1.386008136 |
| 15 | no bar signal | 0.251845683 | 0.132108791 | 0.2535 | 1.385918639 |
| 16 | 8 module queries | 0.280445213 | 0.240555539 | 0.2865 | 1.375602265 |

screen中tracker候选macro AP=0.368196335、macro F1=0.357889767、balanced accuracy=0.3596、NLL=1.324485927；4 mm late候选分别为0.358741130、0.334572800、0.3460、1.330100921。各类别AP/AUROC/precision/recall/F1、混淆矩阵、raw/common-support及22项配对的400次accepted-family/chunk bootstrap均保存在`evaluation_final/summary.json`。screen pending为空；三seed确认尚未完成。

五项CUDA峰值分别为1418.95、716.96、146.54、1418.96、1419.17 MiB；这是每个进程分别观测到的峰值，不能当同一时刻的总显存读数。MIG中NVML utilization查询返回N/A，未取得并发后的硬件利用率百分比，不声称已经达到某个利用率。

一次共享Job的start-to-completion上界为2.816944444 MIG device-hours，组织成本只累计一次。模型成本按同一reservation各时间段活跃进程均分；未运行训练的空档等分给所有参与者。旧手动任务起止时间通过complete manifest的mtime及记录的elapsed重建，后续由supervisor显式记录。

总组织成本上界为27.277435344 device-hours，包含失败、停止、退出等待及初始interactive profile。旧整卡Pending任务的Pod终态已丢失，使用launch到首次观测Job不存在的完整8.269070509小时作保守上界；失败MIG和成功共享MIG使用归档Job时间上界。全部明确标注`conservative_upper_bound`，不宣称其为实际GPU忙碌时间或整卡/MIG吞吐等价。accounting `missing_history_records=0`表示所有尝试均有终态或明确保守边界。

运行输出根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

验证命令：

```bash
python -m unittest tests.test_flavour_s3_shared tests.test_flavour_s2_accounting tests.test_flavour_s2_kubeflow tests.test_flavour_s3_dispatch tests.test_flavour_s3_metrics
python -m analysis.flavour.s3.monitor --root <root> --campaign roadmap/configs/s3_screen.json
python -m analysis.flavour.s3.accounting --root <root>
python -m analysis.flavour.s3.evaluate --root <root> --campaign roadmap/configs/s3_screen.json --comparisons roadmap/configs/s3_pairs.json --output <root>/evaluation_final --bootstrap 400 --no-plots
```

相关单元测试30项通过，覆盖并发区间重叠、空档分摊及device/CPU/resource-hours总量守恒；编译及`git diff --check`通过。

# 阶段性结论

同一个MIG Pod内可以完成多个独立冻结训练，显存足够，避免了其余四项的GPU排队。尚无相同任务串行/并发的配对吞吐实验，因此不报告确定加速倍数。多个latent消融仍塌缩，不能据此推出native与1 mm物理等效。模型质量结论以完整四分类与后续三seed确认为依据。

# 下一步计划

按冻结Pareto规则选择两项候选，连同task5/6完成seed17/2026确认。新GPU Job统一使用supervisor，并在删除资源前保留终态证据。完成条件域、英文图表、三seed统计及strict gate后再结束S3。
