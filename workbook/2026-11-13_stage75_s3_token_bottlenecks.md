# 阶段目标

核对S3是否仍在module/global阶段压缩了native局部信息，并为最终四分类条件域评价准备可读图示。该检查不修改冻结训练。

# 工作内容

新增`analysis.flavour.s3.token_census`。在accepted 80,000-event train集合中，按pixel+bar+tracker channel总数稳定排序，取rank 0/39,999/79,999的同三个事件，逐一执行17种冻结screen transform及collator，直接记录active nodes、local blocks、module/type groups、module queries和global tokens。复用同一个native reader，每个事件只读一次原始observations；只在CPU转换，不执行模型forward/backward，不读取calibration/test。collator中dummy label只满足接口，未参与任何计算。

更新条件域绘图：每页最多8个模型、固定模型颜色、独立底部legend；symlog横轴将线性区域固定为FPR<=0.001，显示1%/0.1%附近结构。原始完整阈值、ties和统计计算保持原实现。

# 详细数据

三个事件分别为`(run,chunk,row)=(10082,53,804)`、`(10082,2239,15)`、`(10081,833,183)`；accepted三探测器channel数为0、557、36,549。这是原始occupancy次序的三个实测事件，不是每种转换后token数的总体min/median/max。

最高occupancy事件在pixel+bar模型中实际使用36,547 channels；另外2个为tracker，只有task14读取。逐阶段实测如下：

| task | representation / window / fusion | Active nodes | Local blocks | Module/type groups | Module query tokens | Global fusion tokens |
|---:|---|---:|---:|---:|---:|---:|
| 0 | native / 4 mm / late | 36,547 | 2,641 | 31 | 124 | 124 pooled directly |
| 1 | native / 16 pitch / late | 36,547 | 14,597 | 31 | 124 | 124 pooled directly |
| 5 | native / 16 pitch / latent | 36,547 | 14,597 | 31 | 124 | 24 |
| 6 | 1 mm / 16 pitch / latent | 8,417 | 6,169 | 31 | 124 | 24 |
| 7 | native / 4 mm / latent | 36,547 | 2,641 | 31 | 124 | 24 |
| 8 | 1 mm / 4 mm / latent | 8,417 | 1,830 | 31 | 124 | 24 |
| 9 | native / 64 pitch / latent | 36,547 | 5,097 | 31 | 124 | 24 |
| 14 | native / 16 pitch / latent, +tracker | 36,549 | 14,599 | 33 | 132 | 24 |
| 16 | native / 16 pitch / latent, 8 queries | 36,547 | 14,597 | 31 | 248 | 24 |

同一个557-channel事件，native/4 mm和native/16 pitch各为114/129 local blocks，但都变为12 module/type groups、48 query tokens；latent融合均再变为24 global tokens。Task16为96 module query tokens，global仍为24。

输入全空事件保留0 active node、1个empty placeholder block和1个placeholder module/type group；4/8 queries变为4/8 tokens，latent仍产生24个learned global tokens。Placeholder不是实际活动硬件模块，字段名明确包含这一处理。

全部51条记录保存于`roadmap/results/s3_token_census.json`及EOS的`token_census/summary.json`、`counts.parquet`。与accepted catalog交叉检查：native配置的active node数等于其配置所读取模态的channel数；1 mm配置不超过该数。没有按occupancy截断事件。

```bash
python -m analysis.flavour.s3.token_census --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12 --campaign roadmap/configs/s3_screen.json --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/token_census
python -m unittest tests.test_flavour_s3 tests.test_flavour_s3_metrics tests.test_flavour_s3_decisions
```

曲线分页仅用既有测得的baseline曲线复制9份不同layout标签来检验两页布局，路径`screen_figures/layout_validation/`，标记为layout test，不作为额外科学模型。`pdfinfo`确认2页，渲染后检查8模型legend与四类坐标完整。

# 阶段性结论

事实：细窗口保留了全部native节点，并让最高occupancy事件的local blocks由2,641增至14,597；module阶段却仍以相同124 query tokens读取，两种像素表示最终都经过24 latent tokens。Task16扩大module读出到248，global数量仍未扩大。当前结构仍存在明确的module/global压缩边界。

这说明“完整保留输入”不等于内部表示可逆。尚未通过可学习且稳定的同架构控制证明哪个压缩环节抹平了native信息；当前主模型的优化塌缩进一步妨碍该归因。三事件计数只证明结构和实际转换行为，不代替整体分类或训练统计。

# 下一步计划

继续同一个确认Job；最新已核实4个训练均为Running，late两项进入epoch2，tracker两项继续epoch1。最终结合三seed性能、分类头状态和条件域指标说明结论，不将token数量单独当成模型优劣依据。S3尚未最终验收。
