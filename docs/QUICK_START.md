# ERA5 海洋环境物理场时序预测 - 快速开始指南

## 概述
本项目使用**时空Transformer**（Spatio-Temporal Transformer）对ERA5海洋环境物理场数据进行时间序列预测。

## 主要特性
- ✅ **默认使用Transformer架构**：先进的注意力机制用于时空预测
- ✅ **自动区域裁剪**：默认裁剪到赤道太平洋小区域（-10°~10°N, 100°~160°E）以加快训练
- ✅ **自动缺失值填充**：使用前向填充（ffill）处理缺失数据
- ✅ **自动生成可视化**：训练后自动生成预测对比图并打开
- ✅ **GPU/CPU自动检测**：有GPU时自动使用，否则使用CPU
- ✅ **最小化命令行参数**：大部分参数有智能默认值

## 快速开始

### 方法1：使用批处理脚本（推荐）

#### 快速测试（1轮训练，约1-2分钟）
```cmd
quick_train.bat
```

#### 完整训练（10轮训练）
```cmd
train.bat
```

### 方法2：直接使用Python命令

#### 最简单的用法（使用所有默认值）
```cmd
python train_eval.py
```

这会自动：
- 使用时空Transformer模型
- 裁剪到赤道太平洋小区域
- 使用sst, u10, v10三个变量
- 训练2轮（epochs=2）
- 批次大小为1
- 训练后生成并打开预测对比图

#### 自定义参数示例

**快速测试**（1轮，少量batch）：
```cmd
python train_eval.py --epochs 1 --max-train-batches 10 --max-val-batches 2
```

**使用GPU训练**（自动检测）：
```cmd
python train_eval.py --epochs 20 --batch 4
```

**强制使用CPU训练**：
```cmd
set CUDA_VISIBLE_DEVICES=-1
python train_eval.py
```

**使用不同区域**（例如：全球）：
```cmd
python train_eval.py --region=-90,90,-180,180 --patch 4
```
注意：全球数据(721×1440)非常大，训练会很慢，建议使用较大的patch值和GPU。

**使用ConvLSTM而非Transformer**：
```cmd
python train_eval.py --model convLSTM
```

**禁用可视化**（仅训练）：
```cmd
python train_eval.py --no-visualize
```

**保存预测结果为NetCDF**：
```cmd
python train_eval.py --save-preds predictions.nc
```

## 命令行参数说明

### 数据参数
- `--data`：数据目录（默认：当前目录）
- `--pattern`：文件匹配模式（默认：`era5_*_extracted/*data_stream-oper_stepType-instant.nc`）
- `--vars`：使用的变量，逗号分隔（默认：`sst,u10,v10`）
- `--T`：输入序列长度/时间窗口（默认：4）
- `--region`：空间裁剪区域 `lat_min,lat_max,lon_min,lon_max`（默认：`-10,10,100,160`）
- `--fillna`：缺失值处理方法（默认：`ffill`，可选：`bfill`,`linear`,`nearest`）

### 模型参数
- `--model`：模型类型（默认：`stT` Transformer，可选：`convLSTM`）
- `--patch`：Transformer的patch大小（默认：1，不分块；对全球数据建议用4或8）
- `--embed`：嵌入维度（默认：128）
- `--depth`：Transformer层数（默认：3）
- `--heads`：注意力头数（默认：4）

### 训练参数
- `--batch`：批次大小（默认：1）
- `--epochs`：训练轮数（默认：2）
- `--lr`：学习率（默认：0.001）
- `--log-interval`：每N个batch打印一次loss（默认：10）
- `--max-train-batches`：每轮最多训练多少batch（可选，用于快速测试）
- `--max-val-batches`：验证时最多评估多少batch（可选）

### 输出参数
- `--save-preds`：保存预测结果的NetCDF路径（可选）
- `--visualize`：训练后生成可视化（默认：开启）
- `--no-visualize`：禁用自动可视化
- `--viz-path`：可视化图像保存路径（默认：`prediction_comparison.png`）

### 其他
- `--list-vars`：列出数据集中的可用变量并退出

## 输出说明

### 训练输出
训练过程会显示：
- 每个epoch的训练损失
- 验证集上的MSE、RMSE、MAE指标
- 进度提示（正在加载数据、第X个batch等）

### 可视化输出
训练完成后会自动生成 `prediction_comparison.png`（或通过`--viz-path`指定的路径），包含：
- **每个变量一行，三列**：
  - 左列：真实值（Ground Truth）
  - 中列：预测值（Prediction）
  - 右列：差异图（Pred - True），并显示RMSE和MAE
- 图像会自动用系统默认图片查看器打开

### NetCDF输出
如果指定了 `--save-preds`，会保存包含真实值和预测值的NetCDF文件，可用于：
- 后续分析
- 使用Panoply/ncview等工具查看
- 导入到其他工作流

## 在服务器上训练

### Linux服务器示例

1. **激活环境**：
```bash
conda activate climate312
```

2. **检查GPU**：
```bash
nvidia-smi
```

3. **后台训练（使用tmux）**：
```bash
tmux new -s train
python train_eval.py --epochs 50 --batch 4 --log-interval 20
# 按 Ctrl+B 然后按 D 分离会话
```

4. **查看日志**（分离后）：
```bash
tmux attach -t train
```

5. **或使用nohup后台运行**：
```bash
nohup python train_eval.py --epochs 50 --batch 4 > train.log 2>&1 &
tail -f train.log  # 查看日志
```

### 使用特定GPU
```bash
export CUDA_VISIBLE_DEVICES=0
python train_eval.py --epochs 50 --batch 4
```

## 数据要求

项目预期数据结构：
```
E:\Climate-D-S\
  era5_202407_extracted/
    data_stream-oper_stepType-instant.nc
  era5_202408_extracted/
    data_stream-oper_stepType-instant.nc
  ...
```

每个NetCDF文件应包含：
- 变量：`sst`, `u10`, `v10`（或通过`--vars`指定其他变量）
- 维度：`valid_time`, `latitude`, `longitude`

## 常见问题

**Q: 训练太慢怎么办？**
A: 
- 使用GPU（自动检测）
- 使用较小区域（默认已裁剪）
- 增加`--patch`值（例如4或8）
- 减少`--batch`或`--embed`/`--depth`

**Q: 内存不足？**
A:
- 减小batch size：`--batch 1`
- 减小模型：`--embed 64 --depth 2`
- 裁剪到更小区域
- 或使用ConvLSTM：`--model convLSTM`

**Q: 如何查看数据中有哪些变量？**
A:
```cmd
python train_eval.py --list-vars
```

**Q: 为什么第一次运行很慢？**
A: 第一次运行时需要：
- 读取并合并多个NetCDF文件
- 计算标准化的mean/std
- 建议先用`--max-train-batches 5`快速测试

**Q: 可视化没有自动打开？**
A: 
- 图像已保存到 `prediction_comparison.png`，手动打开即可
- 或使用 `--no-visualize` 禁用自动可视化

## 项目结构
```
E:\Climate-D-S\
├── train_eval.py          # 主训练脚本
├── quick_train.bat        # 快速测试批处理
├── train.bat              # 完整训练批处理
├── src/
│   ├── data.py            # 数据加载
│   ├── models.py          # 模型定义（ConvLSTM、Transformer）
│   └── utils/
│       └── metrics.py     # 评估指标
├── requirements.txt       # Python依赖
└── era5_*_extracted/      # ERA5数据目录
```

## 技术细节

### Transformer架构
- **Spatial Attention**：每一时刻内的空间相关性
- **Temporal Attention**：不同时刻之间的演化关系
- **Causal Prediction**：使用因果掩码预测下一时刻的物理场

### 数据处理
- 自动检测时间维度（`time` 或 `valid_time`）
- Per-channel标准化（z-score）
- 可选的缺失值填充（前向/后向/插值）
- 空间裁剪（减少计算量）

### 模型输出
- 输入：(Batch, Time, Channels, Height, Width)
- 输出：(Batch, Channels, Height, Width) - 预测下一时刻

## 许可
本项目用于研究目的。ERA5数据使用需遵循Copernicus Climate Change Service (C3S)的条款。

