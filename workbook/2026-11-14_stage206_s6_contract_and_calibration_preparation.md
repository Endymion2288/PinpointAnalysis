# S6 完整合同、候选冻结与校准准备

## 阶段目标

开始Roadmap S6，完成校准、冻结独立test、paired response变体、generator对照和最终物理评价。原要求与执行顺序见`roadmap/13_s6_final_evaluation.md`。

## 工作内容

新增独立`s6`数据/提交入口，不修改S5训练结果。读取接受的inventory，保留所有usable/consistent calibration/test事件，不重分split。Test响应转换有`analysis_freeze.json`硬入口，必须在候选、校准、阈值和selection规则全部冻结后执行。

在calibration/test模型推理前固定13个checkpoint和两组三seedensemble。主方法为100k MAE3 ensemble；Scratch及native/1mm/tracker、S5 auxiliaries/six、Rel5分别保留对照。既有10k validation用于阈值，calibration仅拟合四类log-probability的scalar temperature；priors和多阈值冲突规则预先固定。

## 详细数据

| Partition | CC nu_e | CC nu_mu | CC nu_tau | NC | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| calibration | 87281 | 414559 | 64908 | 155524 | 722272 |
| test | 86905 | 416305 | 65558 | 155673 | 724441 |

共1,446,713事件、454个32-source-file转换tasks；calibration227、test227。原全库有usable/label不一致事件，沿用既有排除状态，没有重新认证或改变接受条件。数据根`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14`；manifest137MiB。最终是否保存cache以实际I/O和复用需求决定，长期只保留紧凑输出。

本地4个calibration事件ROOT→native/truth/ideal tracker smoke完成：5.265864 s，peak RSS1,019.171875 MiB，输出193,717 bytes。该smoke没有执行模型或读取test，不能代表完整转换吞吐或模型质量。全部calibration准备等待CPU batch pilot。

网络读取Guo/Lipton/Kumar/Ovadia原文页面与CERN资源文档，标题和HTTP状态已保存。误命中的无关arXiv页面已排除，不作为方法依据。

Generator搜索已找到public的`2025-11-26-genie`及Pythia6 Standard/NUX候选。后者gFaser只有旧schema，不能直接接当前generator接口；兼容性尚未证明。用户建议的papawan EOS不可读，AFS部分GENIE目录可列名但文件read权限不足；未尝试越过权限。Shared现成G4 binary位于public `Pinpoint_G4/Pinpoint/build/pinpoint`，CMakeCache为LCG107/gcc11，与ML LCG108环境分开。Field/material和generator完整对照尚未测量。

## 阶段性结论

S6已建立完整数据/冻结边界，全部calibration/test样本范围明确。实际校准、独立test、条件变体、generator结果均未测量。本阶段不会把proxy压力幅度写成真实硬件systematic，也不会用校准拟合结果代替独立test。

## 下一步计划

先提交calibration转换pilot，核对内存/吞吐与truth mask再扩227tasks；实现多checkpoint推理、回放、scalar temperature和validation阈值冻结。同时检查generator可用格式，建立最小paired resimulation并验证几何/field/material的真实变更。所有批处理复用实时空闲后端，保留终态和成本。
