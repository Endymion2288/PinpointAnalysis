# S4 每allocation共享压缩输入副本

## 阶段目标

减少低标签重复抽样和同卡多fits对EOS的重复shard读取，保留既有native channels与内存上界。

## 工作内容

既有`CachedNativeDataset`的LRU同时淘汰decoded arrays和本地压缩文件。1k标签分布在全部shards中，反复抽样仍需复制完整shard；多个fits又各自执行同样的远程复制。

新增allocation启动阶段，以3个copy threads将全部40个shards的`observations.parquet`和`targets.parquet`复制一次到本地scratch。实际总量2144919451bytes。各child reader从共享本地压缩副本加载，原有每reader最多2个decoded shards的LRU保持；provenance、catalog和独立Rel sidecar仍使用既有路径。共享副本不包含额外ideal truth文件。

`SpatialDataset`保留原始metadata目录顺序，只通过`PINPOINT_NATIVE_SHARD_ROOT`切换native读取位置。输入数值、sampler、mask、targets和normalization不变。单次暂存开销及bytes写入`runtime/<allocation>.native_staging.json`，计入GPU allocation时间。

新增正式CPU`staging_profile`，用4个确定shards、每shard8个事件、2次交替EOS/local读取比较所有native tensors与metadata，并分列首次暂存、读取/解码/拼接耗时。配置为`roadmap/configs/s4_staging_profile.json`。

## 详细数据与验证

LCG108/Python3.12执行：

```bash
python -m unittest tests.test_flavour_s4_staging tests.test_flavour_cache tests.test_flavour_spatial
```

16项通过，11.536秒。新回归验证压缩文件逐字节相同、source/staged路径生成的完整canonical collated tensors逐位一致、原始metadata目录顺序不变、LRU容量与reader清理不删除共享输入。既有fork/spawn ownership和cache清理测试继续通过。测试中原有fork用例触发Python3.12的multithreaded-fork DeprecationWarning，生产DataLoader仍使用spawn。

暂存、profile和提交器的Python编译检查通过。真实worker的I/O耗时与端到端GPU吞吐：未测量，待正式CPU profile及后续获配worker。

## 阶段性结论

已消除未来worker在EOS上反复暂存同一压缩shard的实现需求，并有原生输入及canonical tensors等价证据。尚不声明端到端加速倍数；CPU transform、decode和GPU计算仍会产生各自耗时。

## 下一步计划

提交真实I/O profile；通过后冻结新worker源码，用共享claims领取剩余任务。运行中的模型继续完成其冻结配置。
