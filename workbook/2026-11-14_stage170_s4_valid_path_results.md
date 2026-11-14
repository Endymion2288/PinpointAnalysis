# S4 有效重拟合的多尺度路径依赖结果

## 阶段目标

判断修复后的分类是否实际使用global、regional、local及summary，而非仅具备连通梯度。

## 工作内容

Kubeflow正式两并发完成四个有效seed42 checkpoint的完整10k固定权重FP32检查，禁用TF32/compile，三套presence-preserving映射，400次128物理块bootstrap。Job于04:37:48 UTC成功结束，真实Job/Pod终态保存后删除并释放GPU；不是提前推断成功。

## 详细数据

下表为changed minus original NLL，均使用同一FP32 replay baseline。各项并非可加性贡献。

| Labels / initialization | Global latent | Regional | Local | Response summary | All spatial |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k Scratch | 0.036853 | 0.011889 | 0.003478 | 0.007145 | 0.065524 |
| 1k MAE3 | 0.024609 | 0.028646 | 0.015774 | 0.008286 | 0.064346 |
| 10k Scratch | 0.000085 | 0.033427 | 0.052724 | 0.226352 | 0.084117 |
| 10k MAE3 | 0.060310 | 0.069442 | 0.038145 | 0.002064 | 0.332015 |

10k MAE3的global增量95%物理块区间为[0.055079,0.065735]，common-support为0.056311 [0.050572,0.062514]；Scratch global虽区间为正，其量值仅约8.5e-5，实际依赖很小。10k Scratch主要依赖summary，MAE3则更依赖空间路径；此处置换幅度包含跨分支相关性被打破的影响。

1k MAE3的local增量0.015774 [0.012096,0.019255]，regional为0.028646 [0.025351,0.031990]；Scratch local为0.003478 [−0.001695,0.008752]，区间跨零。Global在1k两者均有正影响，不能将10k依赖模式推广到1k。

[精简数值](../roadmap/results/s4_transfer4_functional_paths.json)保存raw/common-support、AP/F1差、三套映射差、概率变化及重放检查。完整源为输出根`experiments/s4_transfer4_functional_n1000_s42/functional_paths.json`及`...n10000...`。英文独立图位于`evaluation/transfer4_paths_n1000/`、`evaluation/transfer4_paths_n10000/`，含`functional_paths_NLL.pdf/png`及`functional_paths_macro_AP.pdf/png`。四个模型始终固定权重，无重新训练、拟合或阈值选择。

## 阶段性结论

事实：有效模型存在真实多尺度路径依赖，10k MAE3明确使用global表示，与scratch的summary依赖不同。该结果说明当前Head使用了这些表示，不能证明重训删除该分支的收益或物理因果重要性。仅seed42，区间条件于该权重及三套映射，不代替训练seed不确定度。

## 下一步计划

继续同目标solver gate与CPU/GPU分阶段readout，完成100k三个预训练目标对照。将路径证据与完整分类结果分开解释，最终合并总成本与真实终态。
