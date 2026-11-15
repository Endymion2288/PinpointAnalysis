# D1WD：D1三类终点后的单因素正则化对照

决策时点：2026-11-15，D1 T3已完成完整40k Tune endpoint/best-screen，D1 T4及D2 T3/T4仍在运行。D0 T3的BA3/NLL3为.5665/.871321793；D1 T3 endpoint为.5317/1.143738363，macro AP/AUROC为.613771288/.738472882。D1 best-screen对应完整Tune的BA3/NLL3为.553366667/.868712994，整体BA仍低于D0。D1后段screen AP/AUROC随重复曝光下降，同时rolling Train loss继续降低；对应共同Train的eval-mode gap此时尚未测量，后续统一补齐。

这些结果触发已有R2路线中的过拟合分流。以fresh D1 T3为唯一因果参考，D1WD仅把AdamW的decoupled weight decay从.01改为.1；Train100k四类pool内全部75k非τ事件、12 passes、seed42、BF16 encoder/FP32 head、batch8、LR、完整update schedule、模型宽度和输入表示均沿用D1。名称/design/preflight/extension_reference作为执行元数据更新。新系数为事先固定的一点控制，没有进行系数扫描，不称其为最优值。文献依据为Loshchilov–Hutter的Decoupled Weight Decay Regularization，arXiv:1711.05101，见研究来源M8及`R2/audit/literature_stage280/`。

执行顺序补充：把预定过拟合分流的最小控制提前到对应参考终点完成后启动，以并行利用可用GPU；初始六模型的固定终点比较单独保留，完整R2判断等待初始模型和本控制全部结束。D3/1M/第二架构族没有在这里冻结。仅补T3，避免在尚无终点证据的其他任务上复制相同对照。

主要配对为D1WD−D1，同unique数据、passes和LR日程，只改变weight decay。为选择后续参考，追加D1WD−D0及D1WD−D2，分别标注为配方/曝光的综合比较，不能解释为纯weight-decay效应。三个配对均给all、pixel-centered、large-scintillator及occupancy<100域，endpoint为主、best-screen为次；保留原始输出和同一400次paired physical-block bootstrap。后两组参考D2尚在训练，配对只在其完成后评价。

验收使用实际完整预算、原始预测、共同Train gap、相同五折readout诊断、分支学习健康和终态成本。改善与否同时评价BA/NLL、逐类取舍和局部排序。Cal-A/Cut-B/Confirm继续封存，事件bootstrap不估计训练seed总体波动。
