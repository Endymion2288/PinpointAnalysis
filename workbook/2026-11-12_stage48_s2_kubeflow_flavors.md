# 阶段48：按CERN flavor修正Kubeflow GPU申请

## 阶段目标

落实用户要求，依据CERN ML网站及相关上游文档，使用对应GPU flavor提交可准入的H100整卡请求，保持S2冻结环境和单writer。

## 工作内容

新增显式`--gpu-flavor`，同时选择GPU product与resource键，避免将相同product下的H100 94GB整卡和47GB MIG混淆。GPU flavor省略手写CPU/RAM，由CERN平台自动注入；保留CPU-only probe的原请求。提交前检查冻结硬件，MIG与其他产品不能用于本S2科学配置。Probe增加实际显存记录，核算增加逐resource-hours和未准入Pod的终态零成本记录。

完整文档、六种NVIDIA flavor表、提交与Pod dry-run、存储、配额及迁移方法写入[roadmap/08_kubeflow.md](../roadmap/08_kubeflow.md)。已读取CERN总览、getting-started、training、environments、monitoring页面；并核对Kubernetes GPU scheduling和NVIDIA device plugin上游资料。

## 详细数据

v7原probe只申请`nvidia.com/gpu:1`、2 CPU/4Gi，没有product selector；10:29 UTC按用户要求主动结束，以指定flavor请求替换。其全程无node、GPU allocation=0。Job的DeadlineExceeded来自主动缩短deadline，不能写成自然等到六小时期限。

v8指定H100整卡但仍保留旧CPU/RAM：Job server dry-run接受，实际Pod被`validate-cpu-memory-resources`拒绝，反馈“Resources are allocated automatically. Please remove any CPU or memory allocations.”；未创建Pod。10:31:40 UTC主动结束；保留终态与FailedCreate事件，GPU allocation=0。这不是GPU不足，也不是用户登录失败。

修正后v9先对实际Pod template执行server dry-run，通过并自动注入46 CPU、127G RAM；GPU=`nvidia.com/gpu:1`，product=`NVIDIA-H100-NVL`。10:32:44 UTC真实Pod创建并进入Pending，无node，具有相同资源和product；GPU reservation=0。Job=`pinpoint-s2-h100-94gb-probe-v9`，Pod=`pinpoint-s2-h100-94gb-probe-v9-jlqh7`，排队deadline为六小时，容器probe仍限时300秒。10:33:58 UTCscheduler反馈包括2 Insufficient GPU、1 CPU、1 memory、31 selector不匹配及其他节点状态，尚未取得GPU，不将Pending写成训练失败。

账号无法list nodes（Forbidden），用户namespace的Pod/Job/事件查询正常；不重复要求登录。10:33:46 UTC核算exit0：组织82.65420099 GPUh、missing history records=0；v7/v8终态均0，v9未调度0。正式Condor仍3 running/6 idle，未迁移任何科学writer。

产物根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/`：`kubeflow/execution_pinpoint-s2-h100-94gb-probe-v9.json`与相应Job配置记录提交UID；`h100/kubeflow/`保存`probe_v7_flavor_replacement_requested.json`、`probe_v7_flavor_terminal.json`、`probe_v8_admission_rejected.json`、`probe_v8_flavor_terminal.json`及`h100_94gb_pod_admission_check.json`。未保存注入的认证secret内容。

```bash
python -m unittest discover -s tests -p test_flavour_s2_kubeflow.py
python -m unittest discover -s tests -p test_flavour_s2_accounting.py
python -m analysis.flavour.s2.kubeflow prepare --root "$S2_RESULTS" \
  --name pinpoint-s2-h100-94gb-probe-v9 --gpu-flavor h100-94gb \
  --deadline-seconds 21600 --apply
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
```

以上提交命令已执行，不再次运行。14项Kubeflow单元测试和2项accounting测试通过，日志为`h100/kubeflow/flavor_tests.log`、`flavor_accounting_tests.log`；真实Pod dry-run与实际Pod资源一致。语法和diff检查通过。未测量GPU runtime与Kubeflow正式训练性能。

阶段提交`dc73c51`后Git曾将父提交的一个src tree读为空文件；随后该对象自行可读，`git show`及connectivity fsck通过，未修改对象或重写历史。EOS延迟可见性是待确认原因；证据为`h100/git_object_visibility_stage48.json`和connectivity日志。NVIDIA旧文档地址返回404，引用改为实际读取的官方GitHub device-plugin配置说明。

## 阶段性结论

事实：CERN flavor需要product和resource配对，GPU Pod的CPU/RAM由平台自动分配；仅Job dry-run不足以验证Pod准入。修正请求已创建真实Pod，目前是资源排队。整卡与MIG没有作性能或预算等效声明。

## 下一步计划

等待现有v9调度，取得GPU后执行已准备runtime probe；符合H100/LCG108/BF16且仍有Condor idle任务时再迁移一个任务。继续center控制、额外seeds和plane-only的原有科学训练，35项完整后执行S2严格验收。
