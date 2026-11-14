# S5 合并标签分布与真实对象smoke

## 阶段目标

确认合并sidecar的mask/分道分布，并用真实响应跑通对象构造与独立truth matching。

## 工作内容

正式合并110k事件并核对event key/label，产物`truth.parquet`约307 MiB。新增Parquet原子发布，objects入口要求truth完成summary，避免读取合并中尚未写完的footer。Arrow→pandas嵌套数组使用stack恢复二维raw truth map；测试覆盖真实读取类型。

训练前增加vertex+kinematics与相同任务的vertex-relative对照，使px/py/pT/θ/φ研究能隔离坐标路径；screen由11项变为13项，尚未观察任何S5 validation结果。

## 详细数据

τ27,500事件：hadronic17,982，muon4,755，electron4,752，unknown11。有效衰变位置27,496、有效prong27,496、有效分道27,489；其余分别mask。全部110k的bar semantics有效104,662，charm有效110,000，visible primary-system target有效109,746。

Charm presence按四类为2,829/2,543/2,658/0。NC这一特征的零计数是本生产的标签分布，后续必须在CC类别内检查charm假τ，不能把charm辅助的类别先验收益解释为可见displacement。

τ可见响应（pixel/bar/tracker任意）计数：e3,044/4,752，μ3,037/4,755，had11,632/17,982，unknown10/11。直接prong的tracker命中分别221/1,491/1,991/0。任何fromTau响应不等同于可分辨secondary vertex。

8个真实事件对象smoke耗时14.800783 s，包含sidecar加载：4,077簇、34个pixel轨段候选，其中1个通过75%单track贡献matching；2个tracker候选均匹配；1个vertex候选事件。该小样本显示pixel近邻关联可能含大量fake，不能凭8事件下总体结论；全量必须报告quality–coverage与假关联率。候选构造不读取truth。

测试已包含Arrow嵌套数组roundtrip、已知直轨matching和vertex几何。全部mask辅助测试不读真实标签作为输入。

## 阶段性结论

合并标签与显式mask可用。初步对象结果暴露pixel轨段纯度问题；下一步量化其总体分布及质量选择，而不是把这些候选当已成功重建的粒子。13项模型训练仍由真实拟合gate约束。

## 下一步计划

正式生成40个对象分块、按类别/quality统计fake和效率；随后16事件拟合检查所有新增路径，比较raw-only/object-only/raw+object净贡献。
