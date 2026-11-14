# S4 修复后监督计算对照合同

## 阶段目标

回答MAE3收益是否能由相近额外计算量的监督continuation取得。采用阶段145修复后的Scratch/MAE3作为这项对照，保留原训练配方，使计算对照与train-inner优化实验分开解释。

## 工作内容

冻结[修复后matched-plan配置](../roadmap/configs/s4_empty_matched_plan.json)，三个预算均采用seed42、revision 3、`empty_response_bypass=true`及匹配的真实事件健康gate。监督continuation将读取阶段145完整Scratch checkpoint及Head，不重启LP，不额外引入标签。Parent全部训练计算计入成本，即使选中较早checkpoint。

继续使用原MAE3的确定性sampler/mask/group-shape预算：18,432 updates、294,497次事件曝光、1,804,323,309,279,552次声明矩阵操作。该proxy包括Linear、attention的forward/backward及eager checkpoint重算，不包括归一化、逐元素运算、编译融合与非训练时间；实际设备allocation时长另外报告，不将proxy称作实测硬件FLOPs。

原任务已保存每个标签事件的三列shape表。新增受约束的cache读取：仅允许源计划complete，label budget、seed、label-subset seed、train domain、各模态及token分组设置相同，且事件数/表尺寸一致时复用。空响应修复改变mask和梯度路由，不改变声明proxy中的dense矩阵尺寸；不重读100k原生响应以生成相同shape表。新continuation采用独立名称，避免覆盖旧故障结果。

## 详细数据

实际执行3项测试，6.519秒全部通过：cache允许空响应修复但拒绝改变label subset或region分组；compact shape与真实ragged batch预算及prefix取整一致；声明proxy与实际forward/backward hooks一致，分类case使用修复后的empty bypass。首次命令误写一个测试方法名，unittest报告AttributeError；随后使用真实方法名运行上述3项，全部通过。`git diff --check`通过。

新CPU计划尚未运行，新的matching误差、continuation步数和分类结果未测量，不引用旧故障continuation指标作为新结果。允许的预算匹配误差仍为0.5%，由实际固定prefix计算验收。

新增作业状态：更新源码的1k CPU评价已提交`bigbird20.cern.ch` cluster1578768；收敛修复probes为同schedd cluster1578769，均由提交程序先检查即时schedd负载及WRITE策略。Kubeflow对第二个GPU Job的server dry-run返回`Total GPU units ... cannot exceed ... 1`；未创建GPU allocation，非登录错误。现有Kubeflow继续完成inner任务，新probes由Condor竞争领取，待现有Job终态后再接续使用Kubeflow。精确响应与源码快照保存于EOS `logs/submission_*.json`。

## 阶段性结论

事实：修复后计算对照合同及shape复用约束已固定，相关三项测试通过。尚未生成新matched-prefix测量，尚无监督continuation新结果。

待检验假设：相近的额外声明矩阵操作可能获得与预训练相当的分类收益；该比较需要新路由下真正重训及同事件评价。

## 下一步计划

通过正式CPU作业运行计划并冻结三个新continuation配置，再由可先获配的GPU后端执行。继续完成1k新模型评价和10k train-inner选择；之后在相同下游配方下复核EMA、MAE continuation与partial bar Rel。
