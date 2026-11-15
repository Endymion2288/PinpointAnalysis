# R3 原生候选关系修订与交付合同

## 阶段目标

修正可见层定义对pixel通信的限制，确认修订后的真实学习，并准备原生分支的独立probe、重训ablation及完整验收。R3仍在执行；正式分类收益、候选recall、原生完整Tune峰值尚未测量。

## 工作内容

accepted geometry中15个pixel plane之间可夹入多于三个bar plane；因此将原生候选图的前后三个可见层按同类传感器计算，再分别加入最近的前后跨传感器层。出度上限从12改为14，同plane跨window最多4条，前后同类第1/2/3层分别2/1/1条，跨传感器前后各1条。旧all-visible与新modality-visible在同一1024个平衡Train事件上作候选审计，不使用分类分数选择图。

边特征新增accepted geometry给出的名义轴向tungsten/silicon/scintillator厚度积分，单位mm；只表示两z位置之间的名义slab深度，不推断真实轨迹长度、动量或辐射长度。使用完整bar区间和共同测量mask。候选树缓存复用，保留所有原生节点和原global图。

实现独立local/regional/global/tracker LogisticRegression probes：固定common Train4k拟合StandardScaler与C=1的线性读出，完整Tune评价，未进行Tune超参数选择。B2没有独立region分支，regional明确指第一层inter-block消息之后的表示；global指最终event表示。实现移除inter-block stack后重新训练、保留prelocal关系和完整event pooling的ablation。两项实际结果均待正式模型完成。

新增R3完成验收入口：要求完整正式模型矩阵、exact Train/Tune、曝光/LR、匹配真实学习、分支梯度/更新、完整评价、独立复核、科学决策、图表和allocation全终态；不会因已有代码或profile结束把R3标为完成。

## 详细数据

所有产物位于`results/flavour/v2_2026-11-14/R3`，绝对results根为`/eos/home-x/xcheng/FASER/Pinpoint/results`。

修订版G2真实BF16学习gate由Condor `1582998.0`在H100 MIG执行，记录在`fit/g2_c3_c{0,1}/`：

| Variant | Task | Real events | Steps | Accuracy | NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| Native control | T3 | 24 | 96 | 1.000 | 0.00642692 |
| Native control | T4 | 32 | 128 | 1.000 | 0.00880157 |
| Prelocal relations revision2 | T3 | 24 | 96 | 1.000 | 0.00479975 |
| Prelocal relations revision2 | T4 | 32 | 128 | 1.000 | 0.01814557 |

四项全部通过finite激活、非恒定概率、各活动分支梯度与实际更新检查。上述NLL不是generalization排名。完整40k峰值及单/双进程profile仍在执行。

本地CPU16个真实Train事件的候选计时：all-visible平均0.1102396 s/event、最大0.4851325 s；modality-visible平均0.1589378 s/event、最大0.6629008 s。样本含空事件及9958-node事件，不作为全Train吞吐估计；记录为`audit/graph_cpu16.json`。正式cache须按可用CPU/RAM并行准备，避免在大样本上每pass重复构图；不保存EOS tensor中间cache。

首轮1024-event审计`1582995.0`完成内存计算后，在最终Parquet写出时失败：LCG108的Arrow未编译zstd。已统一使用项目`COMPRESSION`，同一固定样本的双policy重跑为`1582997.0`；首轮没有可交付recall，不能用progress计数当科学结果。`graph_audit.py`明确pixel trackID为G4首贡献者，不称完整ancestry。

原G2 `1582994.0`申请30GB后被站点提高到10 CPU，better-analyze显示无愿意即时运行的匹配。该idle allocation已取消。尝试Kubeflow第二MIG的server Pod dry-run被Kyverno `namespace-gpu-quota/check-total-gpu-units`拒绝，namespace总GPU单位上限1，M0已占用；未创建Job/Pod，成本0，拒绝记录保留在runtime。降低诊断申请至5 CPU/15GB后，Condor `1582996.0`成功启动并完成旧图四项真实拟合。因几何修订已由`1582998.0`接替，停止旧图cache/full-Tune profile，计入实际已用成本，不停止正式分类模型。

生成并目视核对英文PNG/PDF：`figures/r3_phase_sensitivity.*`、`figures/r3_stride_cost_profile.*`。35项R1/R2/R3相关tests通过，新检查包括interleaved bar层之间的pixel直连及真实slab中心的厚度积分；轻量probe hook的CPU实际forward也已验证四组输出shape。执行命令：

```bash
python -m unittest tests.test_flavour_v2_r3 tests.test_flavour_v2_r2 tests.test_flavour_v2_r1 -q
python -m analysis.flavour.v2.r3.plots --root "$R3_ROOT" \
  --phase "$R3_ROOT/diagnostics/phase_g1_c1/results.json" \
  --profile "$R3_ROOT/audit/profile_g1_c1/results.json"
```

## 阶段性结论

事实：只按整个探测器的相邻可见层计数会省略被bar层隔开的pixel直连；新候选策略修复这一确定的传播范围问题，并通过真实学习gate。候选构建有可测CPU成本，需要使用已获allocation的多核资源。

待检验：这是否提高recorded-label candidate recall、共同分类或τ工作点。独立probe、ablation、正式矩阵及R3最终决策仍未完成，Cal-A/Cut-B/Confirm保持封存。

## 下一步计划

先读回双policy审计及修订版profile，以实际内存选择正式cache worker数和训练并发；冻结本族控制与关系模型。M0终点配对结果形成后固定G1两侧采样，再完成正式训练、训练后可读性和重训ablation，依预定门槛决定三seed/分辨率等扩展。所有原型失败、取消及被替代allocation纳入R3终态成本。
