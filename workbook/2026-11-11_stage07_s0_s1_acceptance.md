# 阶段 07：用户确认样本与 detector-proxy，完成 S0/S1 验收

**S0 和 S1 已完成。**已有工程交付及实测继续有效；缺少的样本/生产认证由用户确认覆盖，现有 detector-proxy 参数直接接受。后续进入 S2，来源追溯与真实硬件标定不再是前置条件。

## 阶段目标

落实用户对验收要求的明确修改，关闭此前 source-family/生产版本 gate，并将当前工作条件统一记录到配置、路线图和 results。此次只更新验收合同与文档，不重新检验 samples、生产来源或标定，不重跑数据处理、测试和训练。

## 工作内容

用户指令原文：

> 手动覆盖缺少的这些认证。我确认这些samples是满足要求的，可以通过所有验收。 detector-proxy 可用就可以，因为我现在没有真实硬件，所以没法标定。先以已有的值作为使用即可，不要再检验了

执行决定如下：

| Item | Current decision |
| --- | --- |
| S0 sample and source-family requirements | 用户确认满足，验收通过；停止追加生产来源认证 |
| Missing production generator/macro/build records | 保留原始 null/独立核查状态，缺少这些记录不再阻塞 |
| Existing split | 接受已有 signature-grouped、seed 20261111、70/10/10/10 冻结 manifest；所有表示共用 |
| S1 response level | 接受 detector-proxy，直接使用已有参数 |
| Real hardware calibration | 当前阶段无需提供，不再重复核查 |
| Existing labels and exceptions | 沿用显式四类映射与既有异常隔离策略；用户此次确认不改变已实现的数据处理 |
| Formal test use | 仍按研究计划在模型/阈值/校准冻结后执行；无需等待来源认证 |

配置的当前 split 状态改为 `accepted_signature_grouped_user_attested`。`source_mapping_verified: false` 和 production 字段 null 继续表示独立核查事实，验收通过的依据单独记录为 `user_attested`。旧 manifest、ROOT 与模型产物无需重写；历史报告和运行日志保留当时的测量状态，由本报告及 `acceptance.json` 更新当前使用合同。

## 详细数据

沿用的实测结果：全库 **7,240 文件、7,237,572 事件**；1,150 条标签例外已隔离，空 pixel ID 修复 1,566,540 条，全空事件保留 189,009 个。S1 每类一个 chunk、共 4,000 个事件；能量/PE 守恒、变长 loader、scratch 缓存、16 张事件图、B0 小规模 train/validation、64-event 原生 GPU 往返和 15 项测试均已完成。数据及资源测量见阶段 04–06；本次未新增测量。

后续固定使用的 detector-proxy 值：

| Subsystem | Adopted values |
| --- | --- |
| Pixel | per-file ROOT geometry、原生 binary occupancy、联合位置；不使用模拟 edep 作为 pixel charge |
| Scintillator | 现有逐 bar 合并的线性 edep/PE proxy；不使用 per-track count 或 timing |
| Tracker | station 内 0.1 mm 固定网格，Gaussian σx=0.58 mm、σy=0.017 mm，efficiency=0.99，noise=false |
| Missing states | 沿用 has_response 与 unknown=-1；不新增动量拟合或接受度判定 |
| Unspecified electronics/field parameters | 继续保持 null，不补造数值 |

运行结果根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11`。新增 `acceptance.json` 保存用户确认、适用 run、验收状态和配置路径；`validation.json` 更新 S0/S1 passed，保留旧实测字段；`blocked_audit.json` 标记已由用户决定解决，原阻塞记录保留为历史。

版本化修改包括 `roadmap/README.md`、数据/执行/运行说明、两份 YAML 合同与本报告。阶段 04–06 首页添加当前验收说明。仅对本次修改的 Markdown/YAML/JSON 做语法和链接一致性检查；未再次读取样本或运行数据/模型检查。

## 阶段性结论

S0/S1 已按用户确认后的要求完成。验收依据为此前工程证据加本次用户确认，响应层级为采用现有数值的 detector-proxy。来源认证和真实硬件标定已从当前前置条件中移除。

## 下一步计划

进入 S2：复用接受的冻结 manifest 和 detector-proxy，准备共享子集与混类 shard，开展 native/1 mm × joint/projection 对照和 ParticleNet/ParT 基线。S2 的新实现、训练及性能比较按原计划执行，无需再补 S0/S1 认证。
