# R2末轮顺序控制：相同prefix的两次重放

决策时点：2026-11-15 13:08 UTC后，原六模型及D1WD T3/T4全部完成；原八模型16视图/128配对已产出，独立概率/排序3,976项与核心区间1,808项通过。尚未开始本控制。

D1与D1WD的两个任务均在12轮末端出现共同决策偏移。D1WD T4固定4k screen从825k到900k非τ曝光，e/μ/NC recall由.633/.575/.500变为.854/.289/.492，NLL3由.869400变为.923522；对应全40k终点BA3/NLL3=.549767/.915119。原D1 T4的末次变化同样为e上升、μ下降。训练最后1,222行e/μ/τ/NC为444/167/314/297；最后100 batch为289/97/213/199，μ明显少于全pool的25%先验。D2末段未出现同样的缺μ组成。此前R1纯metadata审查只说明这种影响尚未测量；这里出现实际模型证据，触发既有重混顺序分流。

本控制只回答“在当前已训练prefix下，末轮优化顺序是否实质影响终点”。固定`b1_d1wd_t4_s42/best.pt`，其position=(10,12513)、1.1M曝光、每行11次、updates137640；使用第12轮原顺序与一次label-independent全池随机排列（seed20261118），均沿用每个update原有batch大小、12,513次updates、100k事件、原150,153总updates参数化LR及相同AdamW/RNG状态。两分支均从同一prefix开始，不根据结果调整系数或顺序。后者更改事件所属batch和时间次序，属于sampling intervention；不称为纯分类阈值或增加物理信息。

先顺序生成一次allocation-local transformed cache，32个分散事件核对缓存前后模型输入完全相同，再由两个顺序共用。缓存数组量、构建时间、训练吞吐及完整validation峰值实际记录；不在EOS长期保存该cache。原顺序重放与已保存endpoint配对，可检验更改执行路径及异构设备下的复现误差；不能把这个误差直接归给重混效果。完整40k Tune、共同Train4k、逐类/稀疏排序与τ工作点均评价，400次原physical-block bootstrap及独立概率/核心区间读回。原八模型主矩阵保持原样；该诊断单独输出`R2/sampling/`，属于完整R2科学判断的必要补充。

这是Tune开发诊断，prefix本来按own-task screen NLL选出；不等同fresh12轮全程重混训练，也不能用于重新定义原D1−D0比较。若末轮顺序显著改变终点，则选择后续配方前须明确其用途和限制，必要时再冻结fresh训练对照；若没有改变，则继续保留过拟合/表示等解释。Cal-A/Cut-B/Confirm封存。

参考：K. Mishchenko、A. Khaled、P. Richtárik，*Random Reshuffling: Simple Analysis with Vast Improvements*，arXiv:2006.05988；2026-11-15读取arXiv摘要，HTTP200、43,602bytes，来源`R2/audit/literature_stage283/`。该文支持检查有限样本优化中的顺序效应，其gradient-descent理论不直接证明PINPOINT AdamW的偏移原因或预期增益。
