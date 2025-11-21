# Climate-D-S: 基于深度学习的气候数据时空预测

这是一个使用深度学习模型（如 ConvLSTM 和时空 Transformer）对 ERA5 气象再分析数据进行时空预测的项目。

## 项目结构

```
.
├── config.yaml             # 主配置文件
├── data/                   # 存放原始 .nc 数据文件
├── docs/                   # 存放所有 Markdown 文档
│   ├── README.md
│   ├── QUICK_START.md
│   ├── PARAMETER_GUIDE.md
│   └── SERVER_TRAINING_GUIDE.md
├── outputs/                # 存放所有训练和预测的输出结果
├── scripts/                # 存放主要的 Python 脚本
│   └── train_eval.py       # 核心训练与评估脚本
├── src/                    # 存放源代码
│   ├── data.py             # 数据集和加载逻辑
│   └── models.py           # 模型定义
└── requirements.txt        # Python 依赖包
```

## 快速开始

详细的步骤请参考 [docs/QUICK_START.md](docs/QUICK_START.md)。

1.  **配置环境**:
    ```shell
    pip install -r requirements.txt
    ```

2.  **准备数据**:
    将您的 ERA5 NetCDF (`.nc`) 文件放入 `data/` 目录。

3.  **配置训练**:
    编辑 `config.yaml` 文件，设置数据路径、要训练的变量、模型类型等参数。详细参数说明请见 [docs/PARAMETER_GUIDE.md](docs/PARAMETER_GUIDE.md)。

4.  **开始训练**:
    ```shell
    python scripts/train_eval.py --config config.yaml
    ```

5.  **查看结果**:
    训练完成后，所有的输出（包括预测图、指标和 NetCDF 文件）都会保存在 `outputs/` 目录下，并以唯一的运行名称（`run_name`）命名的子文件夹中。

## 在服务器上训练

如果您需要在远程服务器上进行训练，请参考详细的服务器训练指南：[docs/SERVER_TRAINING_GUIDE.md](docs/SERVER_TRAINING_GUIDE.md)。

## 贡献

欢迎提交问题 (Issues) 和拉取请求 (Pull Requests)。

## 许可证

本项目采用 MIT 许可证。

