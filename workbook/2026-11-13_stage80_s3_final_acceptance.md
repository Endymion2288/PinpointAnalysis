# S3最终验收与收尾审计

S3完成，研究结论为保留S2 ParT native reference及1 mm控制；8项新模型晋级决定均false。25模型、350项CPU读回、完整四分类统计和19项strict输出门槛通过。最终科学解释见[阶段77](2026-11-13_stage77_s3_scientific_analysis.md)，机器可读数值见[最终摘要](../roadmap/results/s3_summary.json)。Calibration/test继续冻结，S4–S6未执行。

## 阶段目标

将完整confirmation产物、图示、终态资源证据与科学判断统一验收，修正收尾审计发现的PR绘制和计费元数据问题，提交最终报告、代码及状态。模型质量未晋级与研究输出完成分别判断。

## 工作内容

冻结screen 17项及confirmation 8项已全部结束；本轮逐项读取25份history、manifest与readback报告。每项四轮、80,064 applied optimizer updates、零AMP skips，14项checkpoint/optimizer/predictions CPU检查全部通过；配置符合冻结合同，calibration/test未使用。

最终评价包含41模型、22单seed配对、9三seed比较和1窗口×表示交互，pending=0。条件域包含37个S3/ParT/ParticleNet模型、七预定子域、1,184条逐类工作点记录，pending=0。Strict gate逐项核对冻结对照身份、raw/common-support的34指标及400有效bootstrap记录，null指标按定义处理。19项检查全部true，8项promotion全部false；没有用tau单类收益替代完整四分类决定。

修正`analysis.flavour.s2.kubeflow`的共享计费工具：Pod清理后的Job终态推定不再生成Pod Failed/Succeeded观测。新记录及已有fallback缓存统一标注`pod_phase=null`、`pod_phase_observed=false`、`termination_inferred_from_job=true`及`conservative_upper_bound`，另存controller Complete/Failed条件。S3 shared allocation保留reservation原始证据字段，分摊不覆盖上界来源。只刷新S3结果根，S2结果数据未重写。

修正S3主图及条件PR曲线绘制为step-post。原始AP、阈值、配对区间和bootstrap均未改变；常数分数的线性连接会制造倾斜PR形状，现在recall>0的precision保持本域prior。重绘从现有预测/curve Parquet读取，未重新训练、选择checkpoint或计算bootstrap。

## 详细数据

结果根（以下记为`S3_ROOT`）：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

| Audit item | Actual result | Evidence under S3_ROOT |
|---|---|---|
| Frozen scientific models | 25/25 complete; 4 epochs and 80,064 applied updates each | `review/final_training_audit.json` |
| AMP skips | 0 in every model | `experiments/*/history.json` |
| CPU readback | 350/350 checks pass | `experiments/*/final_checkpoint_checks.json` |
| Main evaluation | 41 models; 22/9/1 single-seed/three-seed/factorial; 0 pending | `evaluation_final/summary.json` |
| Conditional evaluation | 37 models; 7 subdomains; 1,184 operating records; 0 pending | `evaluation_final/conditional/summary.json` |
| Strict output checks | 19/19 true at 2026-11-13 03:11:28 UTC | `final_acceptance.json` |
| Model promotions | 0/8 | `final_acceptance.json` |
| Confirmation children | 8/8 exit0 | `kubeflow/parallel_pinpoint-s3-confirm-mig47-parallel-v1.json` |
| Confirmation controller | Complete=True, succeeded=1, ready=0; completionTime 02:49:33 UTC | `review/job_terminal_evidence.json` |
| Live S3 Pods | 0 at final inspection | `review/job_terminal_evidence.json` |
| Missing accounting history | 0 | `campaign_accounting.json` |
| S3 accounted device-hours | 31.057126064, including labelled bounds | `campaign_accounting.json` |
| Initial interactive T4 upper bound | 0.181142613 device-hours | `interactive_accounting_bound.json` |
| S3 organization upper bound | 31.238268678 / 120 device-hours | `campaign_accounting.json` |

Confirmation调度时间2026-11-12 22:51:55 UTC，Job Complete condition时间2026-11-13 02:49:34 UTC，按14,259秒=3.960833333 MIG device-hours计保守上界。Controller的completionTime比condition早1秒，上界使用后者。Pod已GC，未保留终态phase/exit code；成功证据来自Job及8个子进程，不能编造终态Pod观测。

Metadata修正前后成本完全一致。显式上界行从8增为18，原因是补充了遗漏的两条零小时fallback标签及确认reservation的8条分摊标签；最终18行对应7个Job，其中5个有正成本、2个为零。共享reservation仍只计一次，不等于新增10次运行或增加计费。Kubeflow费用15.153514953，含MIG6.884444444及删除整卡attempt的8.269070509生命周期上界；HTCondor15.903611111。整卡/MIG小时不互换为吞吐等效。精确GPU busy time、初次T4 context分配时长未测量。S2成本92.73055556另列。

### 图示人工复核

`pdfinfo`实测主诊断41页、资源4页、条件曲线40页，native主模型8页、tracker20页。主模型只预测NC，缺少其他预测组，按存在的confusion pair及四个固定random references只生成8页；20页不是每个模型必须达到的数量。

用`pdftoppm -scale-to 1600 -png -singlefile`重渲染并实际查看以下页：主诊断9/33/36页（ParT、主常数模型、tracker）、资源第4页（MIG）、条件曲线1/40页，以及两个native显示的首/末页（1/8和1/20）。共10张代表页，图题、坐标、图例与单位均为英文，可读。常数主模型PR在recall>0处平坦于precision=0.25，混淆矩阵全部流入NC；tracker显示完整四类取舍。Native图展示binary pixel centers、bar extents及proxy tracker，truth能量/顶点明确作为diagnostic annotation；缺少tracker hit时显示空响应说明。事件图不承担总体统计推断。

审计文件为`review/pdfinfo_final.json`、`review/plot_regeneration.json`、`review/manual_review_final.json`及同目录10个PNG。早期图重绘前的观察以这次最终复核为准。资源MIG末页仅一个模型，对应分GPU subtype及分页后的实际条目，不代表其他确认模型缺失。

### 执行命令与验证

以下为实际使用环境；完整训练和bootstrap已由finalizer完成，本轮没有重复启动finalizer。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S3_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12

python -m analysis.flavour.s3.accounting --root "$S3_ROOT"
python -m analysis.flavour.s3.accept --root "$S3_ROOT" \
  --campaign roadmap/configs/s3_screen.json \
  --confirmation roadmap/configs/s3_confirmation.json \
  --comparisons roadmap/configs/s3_confirmation_pairs.json \
  --evaluation "$S3_ROOT/evaluation_final/summary.json" \
  --conditional "$S3_ROOT/evaluation_final/conditional/summary.json" \
  --require-complete
python roadmap/tools/export_s3_summary.py --root "$S3_ROOT" \
  --output roadmap/results/s3_summary.json

python -m unittest tests.test_flavour_s2_kubeflow \
  tests.test_flavour_s2_accounting tests.test_flavour_s3_shared \
  tests.test_flavour_s3 tests.test_flavour_s3_metrics tests.test_flavour_s3_decisions
```

46项focused tests通过（15.200 s）；之后补充shared allocation证据字段断言，其4项tests再次通过（0.009 s）。新增回归覆盖Job成功/失败时的Pod GC推定、旧缓存元数据修复且成本守恒、共享reservation分摊及保留上界来源。首次编辑时新增断言误放到零Pod admission测试，定位后移到相应GC测试；修正后全部通过。Strict acceptance与export均exit0；`git diff --check`通过。大型模型/Parquet/PDF/PNG保留EOS，仓库只保存代码、配置、中文报告及1.61 MB最终数值摘要。用户本地`docs/`未加入提交。

Finalizer状态`strict_gate_passed`，进程已退出；原日志路径被unlink后从打开的FD恢复为`confirmation_finalize_recovered.log`，87,309 bytes，不需要重跑计算。未保留活跃S3 GPU Pod。

## 阶段性结论

事实：S3冻结实验、完整评价、资源闭环与人工图示复核均完成。Tracker候选AP三seed均值0.370623±0.002252，ParT0.456496±0.005208；差−0.085873，95%配对事件CI [−0.094278,−0.077817]。四类AP及1%背景接受率效率均退化，同时NC argmax F1有正收益。研究完成与新架构晋级是不同决定。

假设：head/context尺度及固定query压缩可能限制学习；主native/1 mm的优化失败使硬件分辨率原因仍未分离。本轮没有证实native等效、4 mm抹平优势或tracker信息无用。

## 下一步计划

本轮停止在S3。后续先冻结独立优化稳定性与压缩控制实验，成功学习后再评价预训练需求，具体建议见阶段77末节。Roadmap状态改为completed，保留ParT reference与1 mm控制；S4–S6未执行。
