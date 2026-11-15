# R3：完整交付合同、实际恢复与全Tune重算

## 阶段目标

固定R3完整终态的验收范围，补齐已完成模型的独立权重/输入读回，准备按预计完成时间选择更快资源。六个正式模型中M0/C1已完成评价，G1/G2仍在训练；本阶段没有将R3标为完成。

## 工作内容

冻结`roadmap/v2/configs/r3/completion_contract.json`及results同内容文件。合同包括六个正式模型、M0/geometry/C1三个完整评价合同、五组匹配optimizer控制及24项必要证据。条件seed/粒度/机制实验若触发，需在各自endpoint前追加合同和campaign，保留原有实用门槛。

另在G1 endpoint前派生原geometry合同中完全相同的G1−D2子合同，供G1完成后及早评价；最终完整geometry比较仍须交付。没有新增或按候选结果选择对照/域。

加强`r3.accept`：检查每个模型的真实拟合六项checks、全部冻结optimizer预算、实际逐行Train曝光、完整Tune输出、checkpoint位置、梯度更新和封存状态；五组干预须拥有相同共同配方和逐行有标签曝光。所有正式模型均须有完整Tune checkpoint重算和实际readability路径，完整health矩阵与campaign一致；从全部评价重新计算科学门槛并核对保存结果。终态还须有条件扩展决定、英文图及全部allocation终态。

原生global ablation没有regional层，probe工具现在明确只提取真实存在的local/global/tracker，仍进行全部Tune概率核对；不构造一个不存在的regional结果。新增该行为和迁移release拒绝存活writer/错误claim/不完整覆盖的测试。

## 详细数据

产物根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/`。

Condor1583016.0完成M0及R2 D2的完整Tune重算/独立readout；1583017.0完成G1实际runner中途恢复。

| 检查 | 样本/条件 | 实测结果 |
| --- | --- | --- |
| M0完整checkpoint概率重算 | 全部Tune40000，重新准备accepted输入，H100L1g.12gb | 最大/平均概率差0，argmax agreement1.0 |
| R2 D2完整checkpoint概率重算 | 同上，同一allocation的独立权重 | 最大/平均概率差0，argmax agreement1.0 |
| G1实际中途恢复 | 源Train32→optimizer Train24+synthetic validation8，stride2/BF16 | 参数最大差0，概率与曝光表逐位相同 |
| 新增ablation readout测试 | 去除global_blocks的真实模型对象，合成输入fixture | local/global/tracker存在，regional不存在，全部finite |
| 迁移release测试 | 错误claim、遗漏model、缺少旧writer终态、旧JobStatus2 | 均拒绝启动替代writer |

上述GPU操作均未使用Cal-A/Cut-B/Confirm。恢复fixture也没有使用真实Tune。旧integrity文字曾固定写`global sampler`，实际使用的采样以各config和preflight为准；工具已改为记录实际`event_mixing`并使用`configured sampler`说明。

两套冻结特征readout均只在共同Train4k拟合StandardScaler+logistic C1，完整Tune评价，没有Tune超参数选择：

| 权重 | Local BA3 | Regional BA3 | Global BA3 | Tracker BA3 |
| --- | ---: | ---: | ---: | ---: |
| M0 full-pass mixing | 44.2300% | 48.3667% | 50.5300% | 41.5867% |
| R2 D2 spatial | 44.9833% | 48.2267% | 50.2800% | 41.5267% |

这里global是tracker融合前的detector encoder。表格只衡量冻结特征的独立可读性，不能由微小差值宣称采样改变了物理信息或某分支效用。正式M0逐类质量仍以已完成配对负决定为准。

新图工具绘制分层readout和实际训练分支gradient/update RMS。梯度按全部分支参数数目归一化，update仍是runner记录的代表tensor；health报告保留全部窗口和零值，log图使用正值范围并标明存在零值的步骤。M0/C1预览使用实际记录，未替代待完成的全部模型矩阵。

资源查询`audit/live_slots_stage294.json`记录了当前逻辑slots、可用GPU属性及VO filter。空闲H100/H200 ads仍含特定AccountingGroup筛选，不能由空闲数量推断本账号可分配数量。已有A100测试没有建立迁移优势。三个native正式任务继续在独立1g MIG有效优化。

新增`r3.handoff`资源流程及三个架构/精度与正式模型一致的profile配置，计划在G1释放Kubeflow配额后使用。流程先在已取得的候选claim测单模型及三并行，再公开ready证据；只有核对预计完成时间、确认所有旧allocation终态并写入匹配release后，才以保存的状态继续正式训练。拒绝或未release则限时退出；同一claim计一次成本。候选整卡是否可立即获得、实际并行吞吐、迁移净节省时间均尚未测量，尚未提交此后备任务。

该claim可在profile完成后并行执行已完成G1的必要probe，利用等待决定及native缓存构建期间的GPU。此辅助操作单独关联allocation，既不改变正式曝光，也不额外计一次device-hours。

验证命令：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_v2_r3 tests.test_flavour_v2_r2 tests.test_flavour_v2_r1 -q
python -m analysis.flavour.v2.r3.health_review \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3 \
  --experiments /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/experiments/b1_m0_t4_s42 \
                /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/experiments/b1_c1_sensor_rows_t4_s42 \
  --attempt delivery_preview_c1
```

43项tests通过。完整合同尚有G1/G2 endpoints、相应完整评价/probes/health、最终条件决定和终态成本等未交付项，未运行最终accept。

## 阶段性结论

事实：已完成权重的独立重算与保存分数一致，G1实际恢复路径通过；验收已明确覆盖整个六模型矩阵，避免用已完成子集替代R3终态。M0/C1负科学结果继续有效。

假设：更大的已获GPU可能通过三模型并行缩短剩余完成时间；目前只有准备好的测量和单writer切换机制，尚无本次候选claim证据。迁移决定将包括重建缓存、验证、旧未保存工作和余量成本。

## 下一步计划

G1完成后立即运行匹配D2的完整评价及自身probe/health，释放Kubeflow配额后测必要资源候选。G2完成后运行完整geometry合同、真实原生分支probe与重训global ablation解释。依据全部结果作seed、粒度、机制控制、anchors、fusion和紧凑图像质量恢复的明确决定，完成英文图、全账本终态和最终accept。
