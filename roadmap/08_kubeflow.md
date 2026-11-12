# CERN Kubeflow flavor 与 S2 执行

2026-11-12核对[CERN平台总览](https://ml.docs.cern.ch/)、[命令行入门](https://ml.docs.cern.ch/getting-started/#command-line_1)、[单节点训练](https://ml.docs.cern.ch/ml/training/)、[容器环境](https://ml.docs.cern.ch/environments/)、[监控](https://ml.docs.cern.ch/monitoring/)及Kubernetes/NVIDIA上游文档。用户登录已正常。

## Flavor 选择

必须一起指定`nodeSelector.nvidia.com/gpu.product`和GPU resource。H100整卡与MIG具有相同product，资源键不同。CERN根据flavor自动注入CPU/RAM；GPU Job只填写GPU请求，手工CPU/RAM会被`validate-cpu-memory-resources`准入策略拒绝。下表CPU/RAM是文档分配值，实际Pod注入结果另存证据。

| CLI flavor | GPU product | Resource for one device | CPU | RAM | Scratch |
| --- | --- | --- | ---: | --- | --- |
| `h100-94gb` | `NVIDIA-H100-NVL` | `nvidia.com/gpu: 1` | 46 | 127 GB | 329 GB |
| `h100-47gb` | `NVIDIA-H100-NVL` | `nvidia.com/mig-3g.47gb: 1` | 22 | 63 GB | 164 GB |
| `a100-40gb` | `NVIDIA-A100-PCIE-40GB` | `nvidia.com/gpu: 1` | 14 | 115 GB | 269 GB |
| `t4-16gb` | `Tesla-T4` | `nvidia.com/gpu: 1` | 26 | 140 GB | 150 GB |
| `v100s-32gb` | `Tesla-V100S-PCIE-32GB` | `nvidia.com/gpu: 1` | 2 | 10 GB | 50 GB |
| `rtx6000-96gb` | `NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition` | `nvidia.com/gpu: 1` | 44 | 156 GB | 1000 GB |

入口`analysis.flavour.s2.kubeflow prepare --gpu-flavor ...`生成相应selector和resource；默认`h100-94gb`。保留的`--gpu-product`必须与flavor一致。S2科学合同固定整卡H100 NVL，提交入口拒绝其他产品及MIG。其他flavor可用于独立环境probe，不能直接混入当前正式科学矩阵。

H100 94GB的Pod资源片段为：

```yaml
spec:
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-NVL
  containers:
    - name: s2
      resources:
        requests:
          nvidia.com/gpu: 1
        limits:
          nvidia.com/gpu: 1
```

2026-11-12实际Pod dry-run通过，注入46 CPU/`127G` RAM；v9真实Pod具有同一注入结果。不要为缩短队列手动降低这些值，也不修改平台注入约束。个人namespace文档配额为同时1个NVIDIA或2个AMD，实际`xcheng`配额NVIDIA=1；已提交但未调度的GPU Pod也会计入请求配额。

## 环境与提交

用`ml-prod`、namespace=`xcheng`，用户OIDC登录已恢复；不要覆盖现有kubeconfig。文档说明客户端token有效10小时。单节点训练直接使用`batch/v1 Job`，不需要为一张GPU额外建立分布式PyTorchJob。

现有Alma9容器中CPU probe已实测CVMFS/EOS与LCG108可用；当前继续复用此环境，科学包由CVMFS提供，不使用文档示例PyTorch2.3.1替代冻结的Torch2.7.1。脚本设置Python路径、OMP/BLAS/MKL线程数，与既有训练保持一致；CPU配额自动注入不等于应无条件开启46个BLAS线程。实际GPU probe执行CUDA matmul并记录GPU名称、显存、BF16、Python/Torch/PyArrow与plan可读性。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.kubeflow prepare --root "$S2_RESULTS" \
  --name UNIQUE_JOB_NAME --gpu-flavor h100-94gb --deadline-seconds 21600
```

`--root`为父目录，训练脚本会读取其`h100/`；不要传入`h100/`导致双层目录。默认只生成配置，`--apply`使用create并记录Job UID。环境probe容器运行限时300秒；Job deadline包含排队时间。已提交的Job名不能复用。`pinpoint-s2-h100-94gb-probe-v9`已在12:03 UTC主动缩短deadline结束：当时全部剩余科学任务已在Condor运行，probe始终未获node；这是主动释放请求配额，不是自然排队到期。GPU runtime尚未实测。

创建Job的server dry-run只验证Job对象，不会证明Pod满足准入。应将其`spec.template`转换成临时`kind: Pod`后执行`kubectl create --dry-run=server -f ...`，检查注入的selector和resources，再提交Job。实际Pod检查`PodScheduled`、`nodeName`、`conditions`、容器状态及logs；`Pending`可能表示资源排队，Job的`FailedCreate`则可能表示Pod准入失败，两者处理不同。

## 存储、调度与成本

CERN推荐`/shared`持久复用小型代码/环境、`/scratch`暂存I/O密集产物、EOS用于跨集群访问和长期结果。当前S2沿用已经验证的EOS路径和CVMFS view，不复制大批中间ROOT/xAOD。平台支持`mount-eos`/`mount-cvmfs`/`mount-afs`等标签；本项目已有CVMFS PVC挂载及EOS注入实测，不同时改变数据路径和运行环境。

实时可用性由session创建页面的flavor色点与[监控面板](https://ml.docs.cern.ch/monitoring/)查看；红点允许排队，黄色表示配额不足。当前账号无法list cluster nodes，使用自己namespace的Pod和scheduler事件记录实际等待原因，不根据文档总量推断实时空闲卡数。

备用训练配置task33/task34均只生成、未提交；task33及task34随后均已在Condor运行。Task34在probe仍Pending时的server Pod dry-run被namespace单GPU配额拒绝，没有创建Pod。未来获得整卡H100且runtime通过后，仅迁移当时仍idle的Condor任务：先确认目标无存活writer，再结束原idle副本、核实队列状态，最后启动Kubeflow训练。已有running训练继续执行，不产生双writer。

核算从PodScheduled开始，包括image pull、init、失败和恢复分配；未调度probe计0。未创建任何Pod的准入拒绝Job，需保留终态和FailedCreate证据才记录0，不能只根据空列表推断。逐GPU resource保留hours；如独立probe使用MIG，每个分区按一张设备计入组织预算，同时单列MIG resource-hours，不作整卡性能等效换算。所有后端成本纳入200 GPUh组织上限。

上游依据：[Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)说明GPU extended resource与node selector；[NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin#configuration-option-details)说明MIG混合策略使用独立resource名称。CERN实际准入结果决定本站CPU/RAM配置方式。
