# 阶段 01：仓库协作规范整理

## 阶段目标

将用户提供的运行环境、资源管理、物理分析、阶段报告和沟通要求写入 `AGENTS.md`，形成后续任务可直接执行的仓库规范。

## 工作内容

- 将原有贡献指南整理为中文，保留项目结构、开发命令、编码和测试说明。
- 补充 lxplus-gpu、CVMFS、Calypso、Kubeflow、HTCondor 与 schedd 选择要求，明确输入、模拟源码、输出及提交文件路径。
- 加入多核与 I/O 优化、EOS 产物精简、物理证据、论文检索、阶段报告和 Angular Conventional Commits 要求。
- 核对 `src/analysis/utils/utils.py` 与 `pyproject.toml`，说明当前路径默认值与新运行约定的衔接方式。

## 详细数据

输入为本轮用户要求、原有 `AGENTS.md`、上述配置和路径工具。此次修改对象为 Markdown 文档，未运行数据处理或修改分析配置。

| 检查项 | 实际结果 |
| --- | --- |
| `AGENTS.md` 行数 | 73 行 |
| `AGENTS.md` 文件大小 | 6123 bytes |
| 标题、环境、路径、调度效率、报告提交、分析表达检查 | 6 组通过，共 6 组 |
| 行尾空白 | 0 行 |
| GPU 可用性、显存、软件构建兼容性 | 未测量；环境信息由用户提供 |
| 样本事件数、物理指标、系统效应及统计误差 | 未测量 |
| schedd 负载与作业运行性能 | 未测量 |

执行命令：

```bash
cat AGENTS.md
git status --short
date '+%Y-%m-%d %Z'
git log -3 --format='%h %s'
sed -n '1,65p' src/analysis/utils/utils.py
cat pyproject.toml
```

使用 `python3` 读取 `AGENTS.md`，逐组检查标题、关键术语及指定路径，通过 `splitlines()` 和 `stat().st_size` 测量行数与字节数，并检查每行 `rstrip()` 前后相等。提交前执行 `git diff --cached --check` 检查暂存文档的空白格式。

产物路径：仓库根目录 `AGENTS.md` 与 `workbook/2026-11-11_stage01_repository_guidelines.md`。

## 阶段性结论

仓库指南已覆盖用户指定的运行、分析和协作要求。源码核对确认现有路径工具仍默认指向 `../data/` 与 `../figures/`；指南要求实际运行前接入指定样本与 results 路径。本阶段结果为规范整理，物理结论未测量。

## 下一步计划

| 具体任务 | 所需输入 | 验收标准 |
| --- | --- | --- |
| 为下一项物理分析明确问题及样本选择 | 用户指定的分析目标、样本 run/chunk 和配置 | 写明可检验的物理问题、选择条件、指标及统计依据 |
| 开始运行前核对环境与路径 | 匹配的 CVMFS 构建、指定样本目录和 results 目录 | 小样本验证通过，产物写入指定目录，并记录实际资源与测量值 |
| 扩展至批量任务时准备 HTCondor 提交 | 小样本验证结果、CPU/GPU 与 I/O 需求、schedd 负载 | 提交目录符合规范，记录 schedd 选择依据，输出经过合并或去重 |
