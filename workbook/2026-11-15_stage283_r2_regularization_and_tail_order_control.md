# Stage283：R2正则化结果与末轮顺序的必要控制

## 阶段目标

完成D2 T4、D1WD两个终点及正式八模型比较，区分数据、过拟合和末段决策偏移。八模型已完成，正式16视图/128配对、共同Train gap、五折readout均完成；模型末轮出现与局部类别组成一致的偏移，已启动固定prefix的顺序控制。完整R2尚未验收。

## 工作内容

读取正式`evaluation/`的400次paired family-connected physical-chunk bootstrap，seed20261116，11个评价域；独立概率/计数及sklearn排序3,976项、核心区间1,808项通过。新正则化比较保持D1数据、seed42、batch8、完整LR轨迹及曝光，只将AdamW weight decay .01→.1。

新增`r2.sampling`，冻结末轮原顺序/全池随机排列的两次重放。复用D1WD T4第11轮末checkpoint；两个分支共享prefix模型、AdamW状态和RNG，保持相同事件、batch大小序列、updates及LR。先将真实输入顺序转换为allocation-local数组，避免随机访问导致shard反复解码；核对32个分散事件的模型输入。原顺序与已有endpoint配对，重混与原顺序配对。该诊断单独交付，不改写原主矩阵。

`r2.accept`增加已触发顺序控制的必需产物；完整导出工具`roadmap/tools/export_r2_final_review.py`需验收通过后才能导出最终结果。R3计划增加G0相位、G1首层stride和G2原生关系的具体对照、真实学习及晋级要求，尚未执行R3训练。

## 详细数据

### 完整40k Tune终点

| Model | BA3 (%) | NLL3 | BA4 (%) | NLL4 | Tau eff @ 1% FPR (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 T3 | 56.6500 | .871321793 | — | — | — |
| D1 T3 | 53.1700 | 1.143738363 | — | — | — |
| D2 T3 | 57.0967 | .855866803 | — | — | — |
| D1WD T3 | 54.6133 | .962710252 | — | — | — |
| D0 T4 | 56.5333 | .864586107 | 44.7850 | 1.185567841 | 5.11 |
| D1 T4 | 53.7300 | 1.015106150 | 42.3700 | 1.337161873 | 4.38 |
| D2 T4 | 57.1267 | .849030484 | 45.7650 | 1.166233178 | 6.05 |
| D1WD T4 | 54.9767 | .915119484 | 43.6075 | 1.226066265 | 5.29 |

T4表中的BA3/NLL3使用真非τ共同样本及重归一化概率，不把τ重标NC。所有工作点仍为Tune域内开发读回；Cal-A/Cut-B/Confirm未打开。

| Endpoint comparison | ΔBA3 (pp), 95% interval | ΔNLL3 |
| --- | --- | ---: |
| D1WD−D1 T3 | +1.443333 [.994768,1.942180] | −.181028110 |
| D1WD−D0 T3 | −2.036667 [−2.557283,−1.544521] | +.091388459 |
| D1WD−D1 T4 conditional | +1.246667 [.752100,1.720469] | −.099986666 |
| D1WD−D0 T4 conditional | −1.556667 [−2.097752,−1.003947] | +.050533377 |
| D2−D0 T4 conditional | +.593333 [.181383,.981433] | −.015555624 |
| D2 T3−conditional T4 | −.030000 [−.392504,.340599] | +.006836319 |

D2−D0 T4的ΔBA4=.980000 pp [.622120,1.375480]；τ效率差=.940000 pp [.419277,1.566227]。点估计小于预定1 pp门槛，不能四舍五入称通过。D1WD−D1 T4的τ效率差=.910000 pp [.438544,1.412150]；相对D0则为.180000 pp [−.268902,.727458]。正则化有实测帮助，当前endpoint仍不如短D0或更多unique的D2。

### Gap与readout

共同Train每类1k，eval mode。D1WD T3 endpoint Train BA3/NLL3=.626333333/.780739158，Tune−Train NLL gap=.181971094；相对D1的.516755766缩小。D1WD T4条件三类Train BA3/NLL3=.587333333/.844116768，gap=.071002716；四类gap=.070346988。D2 T4条件三类gap=.019469321、四类gap=.017612684。

D1WD T3 best-screen完整Tune BA3/NLL3=.555633333/.877850495；T4 best-screen条件BA3/NLL3=.5613/.864870277，τ效率5.67%。这些点由own-task固定screen NLL选择，不能替代endpoint主比较。

同一physical-chunk五折readout中，D1WD T3 raw→OOF global bias BA3由54.6133%升至55.3967%，NLL .962710252→.917961395；D1WD T4为54.9767%→56.2233%，NLL .915119484→.869241001。全局bias在这两个末端恢复部分决策损失，因此不能把stage279/282的“D0/D2无净BA收益”泛化为所有模型无收益。D2 T4 raw/bias BA3=57.1267%/57.0267%，温度仅降低NLL到.845443834，保持argmax。readout本身仍是Tune开发诊断。

### 末轮顺序触发条件与冻结合同

D1WD T4最后两个screen分别在825k/900k非τ曝光，条件三类e/μ/NC recall为.633/.575/.500与.854/.289/.492，NLL3=.869399604/.923521756。D1及D1WD T3也出现最后一轮e上升、μ下降，且非只有少数离群事件。

训练最后100 batch的e/μ/τ/NC为289/97/213/199，最后1,222行444/167/314/297；全pool依然严格平衡。它与模型末段偏移相符，构成执行既有采样分流的证据；相关性尚不是原因。

重放父checkpoint为`experiments/b1_d1wd_t4_s42/best.pt`：position=(10,12513)、updates137640、1.1M曝光、每行11次。两个重放均追加100k行、12,513次updates，到150,153总updates和每行12次；LR沿原完整schedule，AdamW=.1，不扫描系数。全池随机排列seed20261118后末100 batch变为207/209/189/193，lag-1 class-proportion相关均≤.01123；原顺序最大.10930。原/重混整轮均每类25k，不进行按类别再采样。

这个prefix原先按Tune screen选出，因此只回答条件末轮问题。尚未测量fresh12轮重混的泛化，不把它作为新架构或独立确认。理由在[冻结说明](../roadmap/v2/configs/r2/sampling_tail_reason.md)，完整合同/顺序位于`R2/sampling/contract.json`与`orders.npz`。

### 执行与验证

D1WD T3 Condor1582978及T4 Kubeflow `pinpoint-r2-d1wd-t4-k4`均完成；后者13:01:50 UTC Complete。正式评价Condor1582989、Train gap1582988、readout与图表均已交付。提交器查询实时queue和EOS/WRITE兼容性后选择bigbird20，未重复启动已有依赖watcher。

新重放Job `pinpoint-r2-sampling-k5`，Pod `pinpoint-r2-sampling-k5-277xd`在Kubeflow H100 NVL MIG3g.47gb有效运行。13:13 UTC cgroup memory.current=8,003,674,112bytes、peak=10,817,314,816bytes、limit≈63GB，OOM/kill均0；最终cache量、吞吐和终态成本尚未测量。没有为等待同型号或整卡中断训练。

随后cache审计已形成：100k事件数组共3,482,748,960bytes，构建46.037197 s，32个分散事件的缓存前后模型输入全部相同。原顺序已完成8,000次updates/63,939次曝光，纯训练154.200 s；尚未完成两个分支及正式评价，终态成本继续等待。

LCG108、`PYTHONPATH=src`、OMP/BLAS单线程：

```bash
python -m unittest tests.test_flavour_v2_r0 tests.test_flavour_v2_r2 \
  tests.test_flavour_v2_r2_readout tests.test_flavour_v2_r2_sampling
python -m analysis.flavour.v2.r2.sampling plan --root "$R2_ROOT"
python -m analysis.flavour.v2.submit --root "$R2_ROOT" --stage R2 \
  --name pinpoint-r2-sampling-k5 --backend kubeflow --flavor h100-47gb --gpu \
  analysis.flavour.v2.r2.sampling run --root "$R2_ROOT"
```

24项tests通过，包含重混无丢失/重复和每个batch位置大小不变。完整导出工具在现有12视图上验证90个task-domain摘要schema，正式导出仍等待最终accept。原始预测、大checkpoint与图留results；[紧凑证据](../roadmap/v2/r2_regularization_review.json)固定本阶段结果。

联网读取Mishchenko等 *Random Reshuffling: Simple Analysis with Vast Improvements*，arXiv:2006.05988，HTTP200、43,602bytes；摘要/访问记录`R2/audit/literature_stage283/`。文章支持将顺序作为优化因素检验，其gradient-descent理论不能直接证明本项目AdamW的偏移机制。

## 阶段性结论

事实：更多unique优于原重复配方，增强weight decay缓解重复训练退化，但这两个固定终点仍落后于D0。D2四类共同任务与三类差值区间跨零；τ增益为正但不足1 pp。部分末端损失可由全局bias恢复，且末轮偏移与局部采样组成相符。

待检验：末轮次序是否实际造成这部分偏移，以及修正后是否仍有过拟合/稀疏表示限制。两个因素可能同时存在，不以metadata相关性重写数据规模结论。

## 下一步计划

完成两次末轮重放、原顺序复现与8项开发配对，核对实际输入、prefix、曝光、健康和Train gap。根据结果决定是否需要fresh采样配方对照，再形成科学分流、参考模型、终态记账及完整R2验收；R3执行继续等待该决策。
