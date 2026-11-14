# Analysis of PINPOINT data

当前研究计划与结果评估见 [模型优化路线图 v2](roadmap/README.md)；旧路线已归档，完整阶段记录见 [workbook 索引](roadmap/v2/06_workbook_index.md)。

- `preprocessing.py` modifies data in `data` directory:
  - By default this should be outside of the analysis directory, path can be changed in `get_data_path` function in `utils.py`
  - Expects common ROOT files in `data/root` directory and will create new `parquet` and `torch` directories with preprocessed data
- `cnn.py` runs neutrino flavor identification using 2d CNNs with zx and zy projections of binned pixel hits. This is meant to be run on a remote machine with a powerful GPU
- `evaluate_cnn.ipynb` takes weights from `cnn.py` and is meant to be run locally to produce performance plots
