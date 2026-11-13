# S4 固定分类输入的进程内缓存

## 阶段目标

降低低标签训练反复解码native shards和构建相同canonical输入的CPU等待，保留原训练样本、sampler和数值。

## 工作内容

真实共享暂存profile显示decode远慢于压缩文件复制；监督100k训练也记录了较长data wait。分类的native变换和`seed=index+42`固定，因此新增仅供`LabelDataset`的process-owned事件数组LRU；SSL的逐epoch masks和`PretrainDataset`不缓存。

后续正式GPU worker通过环境变量启用每reader1024MiB的NumPy数组上限，仅对事件数不超过10k的固定分类子集生效。100k train沿用原native读取。缓存保存完整nodes/groups/modules，设为只读；返回独立浅层dict。相同事件重新访问跳过native decode与token transform，collate保持原实现。所有所选事件均已缓存时释放该reader的native缓存。spawn不发送父进程数组，fork/PID变化重置缓存owner。Python容器开销不计入该数组上限。

新增正式CPU`cache_profile`：对原1k nested subset执行两次完整顺序遍历，原始/缓存reader按pass交换调用顺序，逐batch检查所有collated tensors及metadata逐位一致。时间包含read/transform/collate，单列首次共享暂存，不冒充GPU端到端吞吐。配置为`roadmap/configs/s4_cache_profile.json`。

## 详细数据与验证

已按原样本catalog计数：1k/10k/100k子集包含1,066,094/10,775,251/109,860,401个观测通道；按每node72bytes及groups/modules各8bytes估计canonical数组上界为89.473/904.320/9220.086MiB。1k/10k预计可完整容纳于1024MiB，实际缓存bytes和性能待正式profile。最大事件通道数分别11,960/19,455/36,549。这里是现有输入的内存估算，没有重新认证samples或detector-proxy。

LCG108/Python3.12执行`python -m unittest tests.test_flavour_s4_staging tests.test_flavour_cache tests.test_flavour_spatial tests.test_flavour_s4_paths`，19项通过，11.195秒。覆盖真实Parquet输入、共享暂存等价、完整缓存后native释放、二次访问的canonical tensors逐位一致、spawn序列化清空父数组、单事件容量下的LRU淘汰和PID owner重置。既有fork测试出现Python3.12 DeprecationWarning，正式DataLoader仍使用spawn。三个新增/修改模块编译检查通过。

## 阶段性结论

固定分类事件缓存具有输入等价及进程隔离证据；真实子集的数组占用和实际加速尚未测量。运行中的模型继续使用各自冻结源码。

## 下一步计划

提交正式1k全子集profile。通过后新worker启用缓存、读取原冻结实验配置；记录实际data wait及allocation成本，再判断端到端收益。
