# 🛠️ Climate-D-S 参数设置指南

## 📋 概述

本文档详细介绍了 `train_eval.py` 脚本中所有可用参数的设置方法和使用说明，帮助用户更好地配置和运行气候预测模型。

## 📊 数据相关参数

### --data
- **描述**：指定包含ERA5 NetCDF文件的数据目录路径
- **默认值**：`../data`
- **示例**：`--data="../data"`

### --pattern
- **描述**：文件匹配模式，用于筛选特定格式的数据文件
- **默认值**：`era5_*_extracted/*data_stream-oper_stepType-instant.nc`
- **示例**：`--pattern="era5_2022*.nc"`

### --vars
- **描述**：指定使用的气象变量列表，多个变量用逗号分隔
- **默认值**：`sst,u10,v10`
- **示例**：`--vars="mwd,swh"` 或 `--vars="sst,u10,v10"`

### --T
- **描述**：输入序列长度（时间窗口），表示使用连续T小时的数据预测T+1小时
- **默认值**：`4`
- **示例**：`--T=4`

## 🤖 模型相关参数

### --model
- **描述**：选择深度学习模型架构
- **选项**：`convLSTM`（卷积LSTM）或 `stT`（时空Transformer）
- **默认值**：`stT`
- **示例**：`--model="stT"`

### --patch
- **描述**：Transformer的空间分块大小，影响计算效率和特征提取粒度
- **默认值**：`1`
- **示例**：`--patch=1`

### --embed
- **描述**：特征嵌入维度，决定模型容量和表达能力
- **默认值**：`128`
- **示例**：`--embed=128`

### --depth
- **描述**：Transformer层数，影响模型深度和复杂度
- **默认值**：`3`
- **示例**：`--depth=3`

### --heads
- **描述**：注意力头数，影响多尺度特征提取能力
- **默认值**：`4`
- **示例**：`--heads=4`

## ⚙️ 训练相关参数

### --batch
- **描述**：批次大小，影响内存使用和训练稳定性
- **默认值**：`1`
- **示例**：`--batch=2`

### --epochs
- **描述**：训练轮数，决定模型训练时长和收敛程度
- **默认值**：`2`
- **示例**：`--epochs=10`

### --lr
- **描述**：学习率，控制参数更新步长，影响收敛速度和稳定性
- **默认值**：`1e-3`
- **示例**：`--lr=0.001`

### --num-workers
- **描述**：数据加载并行进程数，影响数据加载效率
- **默认值**：`0`
- **示例**：`--num-workers=0`

### --log-interval
- **描述**：每多少个训练batch打印一次loss，0为不打印
- **默认值**：`10`
- **示例**：`--log-interval=10`

### --max-train-batches
- **描述**：每个epoch最多训练多少个batch，用于快速测试
- **默认值**：`None`
- **示例**：`--max-train-batches=100`

### --max-val-batches
- **描述**：验证时最多评估多少个batch，用于快速测试
- **默认值**：`None`
- **示例**：`--max-val-batches=50`

## 🌍 数据预处理参数

### --region
- **描述**：空间裁剪区域，格式：纬度最小值,纬度最大值,经度最小值,经度最大值
- **默认值**：`-10,10,100,160`（赤道太平洋区域）
- **示例**：`--region="-5,5,100,110"` 或 `--region=""`（全球数据）

### --fillna
- **描述**：缺失值填充方法
- **选项**：`ffill`（前向填充）、`bfill`（后向填充）、`linear`（线性插值）、`nearest`（最近邻插值）
- **默认值**：`ffill`
- **示例**：`--fillna="ffill"`

## 🕐 时间选择参数

### --time-slice
- **描述**：选择特定时间范围的数据进行训练
- **格式**：`"start_time:end_time"`
- **默认值**：`None`
- **示例**：`--time-slice="2024-07-01:2024-07-15"`

### --time-points
- **描述**：选择特定时间点的数据进行训练
- **格式**：`"time1,time2,time3"`
- **默认值**：`None`
- **示例**：`--time-points="2022-01-01T00,2022-01-02T00,2022-01-03T00,2022-01-04T00"`

### --predict-time-point
- **描述**：指定用于预测的特定时间点
- **格式**：`"time"`
- **默认值**：`None`
- **示例**：`--predict-time-point="2024-08-01T00"`

## 📤 输出相关参数

### --list-vars
- **描述**：列出数据集中的可用变量并退出
- **默认值**：`False`
- **示例**：`--list-vars`

### --save-preds / --output-file
- **描述**：保存验证集预测结果的路径（NetCDF格式）
- **默认值**：`None`
- **示例**：`--save-preds="predictions.nc"`

### --visualize / --no-visualize
- **描述**：控制是否自动生成并打开预测对比图
- **默认值**：`True`（开启）
- **示例**：`--visualize` 或 `--no-visualize`

### --viz-path
- **描述**：可视化图像保存路径
- **默认值**：`prediction_comparison.png`
- **示例**：`--viz-path="results/prediction.png"`

## 🚀 使用示例

### 基本训练命令
```bash
python train_eval.py --data="../data" --epochs=10 --batch=2 --vars="mwd,swh,sst,u10,v10"
```

### 快速测试命令
```bash
python train_eval.py --data="../data" --epochs=2 --batch=1 --vars="sst,u10,v10"
```

### 自定义区域训练
```bash
python train_eval.py --data="../data" --epochs=5 --batch=1 --vars="sst,u10,v10" --region="-5,5,100,110"
```

### 指定时间点预测
```bash
python train_eval.py --data="../data" --pattern="era5_202201.nc" --time-points="2022-01-01T01,2022-01-01T02,2022-01-01T03,2022-01-01T04,2022-01-02T00" --predict-time-point="2022-01-02T00" --epochs=1 --batch=1 --vars="mwd,swh" --region="-5,5,100,110"
```

### 列出可用变量
```bash
python train_eval.py --data="../data" --pattern="era5_202201.nc" --list-vars
```

## ⚠️ 注意事项

1. **区域参数格式**：使用`--region`时，确保格式为`"lat_min,lat_max,lon_min,lon_max"`，注意使用引号包围参数值。

2. **时间点数量**：使用`--time-points`时，需要提供至少T+1个时间点（T为输入序列长度，默认为4）。

3. **变量名称**：使用`--vars`时，确保指定的变量在数据集中实际存在，可通过`--list-vars`参数查看可用变量。

4. **内存管理**：较大的`--batch`值或全球范围数据可能导致内存不足，建议从小批量和小区域开始。

5. **CUDA支持**：程序会自动检测并使用CUDA设备进行训练，如需使用CPU训练请确保CUDA不可用。

---
*文档版本：v1.0*  
*最后更新：2025年*