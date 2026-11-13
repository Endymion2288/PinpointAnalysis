# Repository Guidelines

## 项目结构

本项目使用 PyTorch 和 PyTorch Geometric 分析 PINPOINT/FASER 探测器数据，Python 包位于 `src/analysis/`。

- `preprocess_data.py`：将 ROOT 事件转换为 Parquet。
- `cnn/`、`cnn_faser/`、`gcn/`、`gravnet/`、`pointnetpp/`、`muon_hits/`：数据准备、模型、训练脚本和评估 notebooks。
- `utils/`：共享的几何、单位、数据集、绘图和路径工具；`event-displays/`：事件显示及配套资源。
- `tests/`：目前仅有 `__init__.py`；`workbook/`：中文 Markdown 阶段报告。

## 运行环境与软件

- 当前工作节点为 `lxplus-gpu`，终端可使用 GPU。交互式调试和小规模验证可直接使用 GPU，大批量数据处理必须通过 HTCondor 提交。
- CVMFS 已提供所需软件包，Calypso 等工具通常已有编译产物。优先复用与实际配置匹配的现成构建，只有源码变化时才重新构建。
- Kubeflow 服务也可用；需要使用而用户尚未登录时，提醒用户登录。

## 数据、产物与任务目录

| 用途 | 路径 |
| --- | --- |
| 输入样本 | `/eos/project-f/fasersim-bonn/public/pinpoint/data` |
| Geant4 模拟源码，探测器响应细节以此为准 | `/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4` |
| 所有运行输出，包括日志、图表和模型 | `/eos/home-x/xcheng/FASER/Pinpoint/results` |
| HTCondor 提交文件 | `~/condor/Pinpoint/<stage或任务名>/` |

home 空间有限，提交文件应尽可能小，作业日志和数据输出写入上述 results 目录。阶段报告和代码、配置保存在仓库中。现有 `utils/utils.py` 默认使用 `../data/` 和 `../figures/`；运行前通过配置或路径工具接入上述输入与输出位置。

EOS 仅保留尽量小的最终结果，避免保存 xAOD 等中间文件；合并或去重批量作业产生的 JSON 等重复输出。大型数据、checkpoints 和冗长 notebook 输出不纳入 Git。

## 开发与验证命令

项目声明 Python 3.12。优先使用匹配的现成环境；需要安装本项目时使用 Poetry。在仓库根目录执行：

- `poetry install`：安装依赖及本地 `analysis` 包。
- `poetry build`：构建本项目 Python 分发包，仅在需要打包时执行。
- `poetry run python -m analysis.preprocess_data -r 10000 -c 0`：验证单个 run/chunk，需先配置输入与输出路径。
- `poetry run python -m analysis.gravnet.train_gravnet_regression_faser --help`：查看训练参数。
- `poetry run python -m analysis.gravnet.train_gravnet_regression_faser -r 10000 -b 8 --num-epochs 1`：使用已准备的小样本验证训练，默认设备为 `cuda:0`。

评估 notebooks 使用匹配的 Python kernel。目前没有配置测试框架、覆盖率门槛、formatter 或 linter。新增自动化测试采用 `tests/test_*.py`；使用标准库 unittest 时执行 `poetry run python -m unittest discover -s tests`。数据或模型变更先用少量 run/chunk 验证，再检查对应指标和图表；记录随机种子、run ID、配置和命令。

## 编码规范与计算效率

Python 使用四空格缩进，函数和模块使用 `snake_case`，类使用 `PascalCase`，常量使用大写；使用 `analysis.*` 绝对导入。沿用 `create_*.py`、`train_*.py`、`evaluate_*.ipynb` 命名，按需补充类型标注、tensor shape 和物理单位。

代码与 HTCondor 脚本均须考虑多核、多线程、I/O 和文件读写效率。尽可能并行处理，线程数与作业申请的 CPU 资源匹配；减少重复读取和小文件写入，按任务选用 C++、Rust、ROOT RDataFrame 等合适语言或软件包加速。除非确有必要，避免 SHA 校验和大段防御性编程带来的额外 I/O 与运行开销。

提交 HTCondor 作业前，检查可用 schedd 的排队与负载情况，选择兼容任务且最空闲的 schedd，记录选择依据以减少排队时间。

## 物理分析与研究方法

每一步都应推进一个有数据证据支持的明确物理结论，同时检查物理原因与程序原因。少数离群事件不足以终止分析分支；原因未知时明确记录待查问题，不将其作为结论。报告说明样本、选择条件、系统效应和统计依据，结论的确定程度须与证据相符。

需要新思路时保持发散思考，主动检索网络论文，兼顾高能物理、计算机科学和机器学习方法；记录引用及其对当前分析的可行性判断。

模型出现常数预测、激活饱和、上游零/极小梯度或明显学习失败时，必须重新审查架构、数值尺度、训练过程与设置，修复后重新训练和测试。完成epoch、finite参数或checkpoint读回不等于有效学习；故障结果只作诊断，不据此得出架构或硬件信息无效的结论。修复方案需检索相关计算机科学论文，并记录内部健康与真实小样本拟合验证。

## 阶段报告、Commit 与 Pull Request

每个阶段完成后，先在仓库根目录 `workbook/` 写中文 Markdown 报告，再将报告与相关代码或配置一起提交；每阶段至少一次 Git commit。推荐文件名为 `YYYY-MM-DD_stageNN_topic.md`，至少包含：

- **阶段目标**：待回答的问题及验收目标。
- **工作内容**：实现、配置和分析步骤。
- **详细数据**：实际测得的数值及单位、输入与配置、执行命令、验证结果、产物路径；适用时引用数据表和图表。未测量项目明确标注“未测量”，不得用推测替代结果。
- **阶段性结论**：证据支持的结果，分别表述事实与假设。
- **下一步计划**：具体任务、所需输入和验收标准。

后续提交统一遵循 Angular Conventional Commits：`<type>(<scope>): <subject>`，例如 `feat(gravnet): add vertex distance features`、`docs(workbook): record stage01 validation`。Pull Request 说明改动、意义、验证命令和样本选择，关联相关 issue；模型或可视化变更附相应图表。

## 沟通与文档

默认使用清晰自然的中文，用户指定其他语言时按要求使用；技术术语保留 English，以表达清楚为准。所有图表使用英文，包括标题、坐标轴、图例和标注。

先给结论或实际结果，再简洁说明改动、意义和验证。事实与假设分别表述；失败时说明具体失败步骤、已知原因和下一步，按需要解释语法与基本概念。所有文档禁止防御性表达，避免套话、泛泛警告、空泛限定语、无实质内容的章节和无关扩展。
