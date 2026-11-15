# R3：补齐强基线参照、首轮健康与新卡缓存实测

## 阶段目标

在三个 G2 正式终点仍未开放时，审查评价范围和实际学习路径，并验证部分迁移后的缓存准备。

## 工作内容

原 geometry 合同包含 control/relations 对 B1 D0 的直接配对，以及重训 global ablation 对原生族的比较，缺少 ablation 对 B1 D0 的直接配对。补充 `global_ablation_projection_evaluation.json`，使用相同100k×4的B1 D0、endpoint/best-screen两视图及all/pixel-centered/large/occupancy四域，共4视图、8配对；完整11域报告和400次physical-block配对重采样继续沿用。

冻结前检查 ablation 的完整 `predictions.parquet` 和 `best_predictions.parquet` 均未生成。补充项属于同开发Tune的比较，三配置首个4k训练screen已经存在，不能称为从未看过开发资料的独立确认。门槛不变，族间架构/参数/计算同时不同，不将差值解释为纯几何因素。

completion contract升级为revision2，保留revision1快照；原六模型、三套核心矩阵、24项证据、五组optimizer控制及所有原图均保留，补充评价和两幅英文图另加。最终科学review须覆盖全部四个评价输出。另将results中已实现的完整六模型campaign及扩展理由同步到版本控制，初始M0合同保存在archive。

## 详细数据

根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/`。

新 `pinpoint-r3-g2-k4` 的共享输入已全部构建：

| Split | Events | Array bytes | New preparation seconds | Old control preparation seconds | Exact scattered tensor checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 100,000 | 13,925,383,580 | 480.144807 | 1,517.795095 | 32 |
| Tune | 40,000 | 5,622,779,908 | 214.399648 | 586.115156 | 32 |

准备worker由旧4个增至12个；新旧输入阵列字节数和候选图完整摘要一致，loader逐条index覆盖检查及各split的32个真实散点collated tensors逐位核对通过。上述不是读取全部20GB数组做逐值旧新比较。新两split准备合计694.544455 s，低于迁移时线性估计1208.200836 s；差值是此次准备时间的比较，不是迁移总节省的实测。

两个模型现已进入同卡独立训练组，使用各自保存的100k曝光状态；消融仍在1583010.0的1g MIG训练。正式有效预算仍为每模型50,051updates、400k曝光。最终吞吐及终态成本尚未测量。

对三模型保存的100k曝光prefix做实际训练健康检查，每个有14个采样窗口：

| Model | Parameter groups | Positive gradient/update observations | Activation observations | Minimum probability std over observed windows |
| --- | ---: | --- | --- | --- |
| Control | 5 | All groups: 14/14 positive | All finite | .026515 / .025060 / .009308 / .013992 |
| Relations | 8 | All groups: 14/14 positive | All finite | .014855 / .025319 / .013396 / .023127 |
| No global | 7 | All groups: 14/14 positive | All finite | .022156 / .021088 / .020972 / .020651 |

Relations的relation gate/message/norm中位gradient RMS分别为0.00493346/0.000962351/0.000445614，对应代表tensor update RMS为4.04119e−5/4.69322e−5/5.78883e−5。No-global对应gradient RMS为0.00328004/0.000737159/0.000663253。未见这一prefix的常数预测、零上游梯度或非有限激活故障；这些是实际窗口的优化证据，不证明已经收敛或具有泛化收益。最终health仍必须覆盖各模型完整训练记录。

原24项必要证据中19项已有目标状态的产物；缺少完整正式health、native三模型probe、最终scientific review、条件扩展决定及最终图审查。正式模型只有M0/G1/C1三个完成，三个G2仍运行；核心geometry及新增强基线比较未执行，终态成本与最终accept也未完成。

主要产物：

- `audit/native_prefix_health_stage299.json`：实际首轮梯度/更新/激活及概率std。
- `groups/g2_k4.json`：新共享输入的实际准备时间和覆盖证据。
- `contracts/global_ablation_projection_evaluation.json`：新增强参照配对。
- `contracts/completion_contract_v1.json`与当前`completion_contract.json`：修订前后范围。
- 仓库`roadmap/v2/configs/r3/archive/`：初始campaign与revision1合同，当前campaign保留完整六模型。

验证：实际解析补充合同得到4视图/8配对/4指定配对域；对修订前后六模型、24项证据、probes、health路径、科学门槛路径及五组optimizer控制逐项相等核对，原评价和图列表均为新列表子集。训练代码沿用已通过43项tests的版本。

## 阶段性结论

事实：新缓存的事件/图统计与已验证设置一致，实际准备比预算估计更短，三模型首轮均存在有效的学习路径。补充参照使去除global的候选也有同训练规模强图像基线的直接配对，避免仅凭较弱原生参照上的改善宣布整体模型已提升。

解释边界：当前control/relations/no-global三配置不是完整的关系×global路径2×2矩阵。global消融估计的是“已含relations时去除global”的条件效应。若仅组合配置有效，需要补无global且无relations的匹配重训控制，或在实际胜出结构上重训同容量/打乱关系控制，才能进一步解释关系增量。没有值得扩展的收益时，不自动补完整factorial矩阵。

## 下一步计划

从新卡滚动日志核对恢复后的实际updates、曝光和速度。全部正式终点完成后，运行原geometry12视图/48配对和补充4视图/8配对，交付完整独立数值复核；再结合最终probes、训练health、重训消融及质量/成本，决定必要seed、粒度和机制实验。Cal-A/Cut-B/Confirm继续封存，最后完成全图、终态账本和严格R3验收。
