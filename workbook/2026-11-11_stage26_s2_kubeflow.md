# 阶段 26：Kubeflow 执行环境与资源核算

Kubeflow 的 CPU 环境探测成功，现有 EOS 输入和结果目录可访问，LCG108/Python3.12/Torch2.7.1 可加载。GPU 探测仍在排队；科学训练尚未迁移。用户明确授权在 HTCondor GPU 不足或排队时使用 Kubeflow。

## 阶段目标

在不改变 accepted samples、split、detector-proxy 与冻结科学配置的前提下，接入第二执行后端，并将实际资源使用纳入 S2 总预算。

## 工作内容

检查真实 HOME 下的现有 kubeconfig。当前 context 为 `ml-prod`、namespace 为 `xcheng`，`auth can-i create jobs.batch` 返回 yes。namespace 已有 EOS、CVMFS、shared PVC；NVIDIA 配额为1，AMD配额为2。当前训练栈使用 NVIDIA/CUDA。集群节点列表不可读取，GPU 型号以实际分配容器的运行记录为准。

新增 `analysis.flavour.s2.kubeflow`：生成/创建 Kubernetes Job、调用同一个冻结训练入口、保存 Job UID 和配置，使用 namespace 已有认证机制。Job 采用 `backoffLimit=0`，失败时先记录原因，再明确恢复；不产生未记录的自动重试。`kubectl create` 会拒绝重名存活 Job。

首次镜像 `registry.cern.ch/alma9-base:latest` 拉取返回400，改用可拉取的 `docker.io/library/almalinux:9`。最小镜像缺少 `which` 和 gnuplot 的可选 GUI 库，wrapper 用 `command -v` 提供 which，并允许完整 LCG view 中的可选工具探测失败；实际 Python imports 与任务入口仍按退出码判定成功。

直接挂载 PVC 时出现 EOS `Operation not permitted` 和 CVMFS 符号链接错误。采用 CERN 示例要求的 `mount-eos=true`、`mount-kerberos-secret=true` Pod labels，注入 OAuth2 refresh bootstrap/sidecar；CVMFS 增加 `HostToContainer` propagation。凭据由平台挂载，代码、报告和结果不写入 token。CERN 参考：[EOS workflow example](https://gitlab.cern.ch/ai-ml/examples/-/blob/master/pipelines/argo-workflows/access_eos/access_eos.yaml)。

`s2.accounting` 合并 HTCondor ClusterIds 与 Kubeflow Job UIDs，按 PodScheduled 到容器结束计入 GPU reservation，包含镜像拉取、初始化和失败尝试。排队未分配节点时为0。持久化精简 Pod lifecycle snapshots，保存终态后可继续核算；deadline 删除发生在首次观察前时，用保留的 scheduler events 与 Job 终态时间给出 allocation 上界，并明确标记。缺失生命周期证据时报告 missing，不自动算0。CPU 资源包括平台注入的 native sidecar。

## 详细数据

CPU 探测 `pinpoint-s2-cpu-probe-v4` 于 **21:58:40 UTC** 记录：

| Item | Measured value |
| --- | --- |
| Python | 3.12.11 / GCC13.1.0 |
| PyTorch | 2.7.1 |
| PyArrow | 20.0.0 |
| UID | 180107 |
| Existing plan read | successful，16 conversion tasks |
| GPU | CPU probe未申请；GPU探测仍排队 |
| Scientific full-epoch throughput | 未测量 |

产物根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/kubeflow/`。其中 `probe_cpu_environment.json` 为实际读取和写入结果，`*.job.json` 为精确 Job 配置，`execution_*.json` 记录执行 UID，`pod_accounting_snapshots.json` 保存成本证据。失败环境探测也保留记录，均未训练科学模型。

验证命令：

```bash
python -m unittest tests.test_flavour_s2_kubeflow tests.test_flavour_s2_accounting
```

6 项测试通过，覆盖未分配排队、镜像拉取 reservation、容器终态时间、deadline 前旧 init 时间不误作终点，平台注入 native sidecar 的 CPU 申请，以及既有 Condor task/name 映射。

创建环境探测的入口：

```bash
python -m analysis.flavour.s2.kubeflow prepare --root "$S2_RESULTS" \
  --name pinpoint-s2-env-probe-v4 --apply
```

## 阶段性结论

已解决容器镜像、LCG、EOS 与 CVMFS 的 CPU 执行路径。NVIDIA quota 可用不等于当前 GPU 已分配；GPU 型号、CUDA 操作与整轮训练耗时仍待实际资源。现有 Condor 35 项科学任务与两项 H100 候选继续保留，未发生两后端同时写同一 experiment。

## 下一步计划

GPU 探测成功后，依据实际型号准备同环境候选。若承接已有 H100 tasks0/6，先确认对应 Condor 副本仍 idle，再移除副本并提交。保留阶段25预先冻结的两族整轮耗时规则；通过后才扩展完整矩阵。继续 V100 四轮训练与终态评价，calibration/test 保持冻结。
