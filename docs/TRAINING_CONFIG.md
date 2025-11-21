# 🌤️ Climate-D-S 训练配置文档

## 📋 项目概述
Climate-D-S 是一个基于深度学习的ERA5气象数据时空预测系统，使用ConvLSTM和时空Transformer模型进行气象要素的短期预测。

## 🎯 核心预测任务
- **输入数据**：连续T小时的气象数据（T帧）
- **输出预测**：第T+1小时的气象数据（1帧）
- **默认配置**：输入4小时数据，预测第5小时

## 📖 参数设置指南
有关所有参数的详细说明和使用示例，请参阅专门的[参数设置指南](PARAMETER_GUIDE.md)。

## 📊 数据配置

### 数据文件结构
```
data/
├── era5_202201.nc    # 2022年1月数据
├── era5_202202.nc    # 2022年2月数据
├── ...
└── era5_202412.nc    # 2024年12月数据
```

### 可用气象变量
| 变量名 | 中文名称 | 物理意义 |
|--------|----------|----------|
| `mwd` | 平均波向 | 海洋波浪的平均传播方向 |
| `swh` | 有效波高 | 海洋波浪的有效高度 |
| `sst` | 海表温度 | 海洋表面的温度 |
| `u10` | 10米U风分量 | 10米高度的东西向风速 |
| `v10` | 10米V风分量 | 10米高度的南北向风速 |

### 数据配置参数
```python
# 在 train_eval.py 中的配置位置
parser.add_argument('--data', type=str, default='../data', help='数据文件目录')
parser.add_argument('--vars', type=str, default='sst,u10,v10', help='使用的变量列表')
parser.add_argument('--T', type=int, default=4, help='输入序列长度（时间窗口）')
```

## 🤖 模型配置

### 支持的模型架构
1. **ConvLSTM**：卷积长短期记忆网络，适合序列预测
2. **SpatioTemporal Transformer (stT)**：时空Transformer，适合时空特征提取

### 模型配置参数
```python
# 在 train_eval.py 中的配置位置
parser.add_argument('--model', type=str, default='stT', choices=['convLSTM', 'stT'])
parser.add_argument('--patch', type=int, default=1, help='Transformer的patch大小')
parser.add_argument('--embed', type=int, default=128, help='嵌入维度')
parser.add_argument('--depth', type=int, default=3, help='Transformer层数')
parser.add_argument('--heads', type=int, default=4, help='注意力头数')
```

## ⚙️ 训练配置

### 训练参数
```python
# 在 train_eval.py 中的配置位置
parser.add_argument('--batch', type=int, default=1, help='批次大小')
parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
parser.add_argument('--num-workers', type=int, default=0, help='数据加载工作进程数')
```

### 训练监控
```python
parser.add_argument('--log-interval', type=int, default=10, help='训练日志打印间隔')
parser.add_argument('--max-train-batches', type=int, default=None, help='最大训练批次')
parser.add_argument('--max-val-batches', type=int, default=None, help='最大验证批次')
```

## 🌍 空间配置

### 默认预测区域
- **区域范围**：赤道太平洋
- **纬度范围**：-10°S ~ 10°N
- **经度范围**：100°E ~ 160°E

### 空间配置参数
```python
# 在 train_eval.py 中的配置位置
parser.add_argument('--region', type=str, default='-10,10,100,160', 
                   help='空间裁剪区域：lat_min,lat_max,lon_min,lon_max')
```

## 🔧 数据预处理

### 缺失值处理
```python
parser.add_argument('--fillna', type=str, default='ffill', 
                   help="缺失值处理方法：'ffill','bfill','linear','nearest'")
```

## 🚀 运行配置

### 训练脚本配置
```bat
# train.bat 中的配置
C:\Users\ICEY\.conda\envs\climate312\python.exe train_eval.py \
    --data ../data \
    --epochs 10 \
    --batch 2 \
    --vars "mwd,swh,sst,u10,v10"
```

### 快速训练配置
```bat
# quick_train.bat 中的配置
C:\Users\ICEY\.conda\envs\climate312\python.exe train_eval.py \
    --data ../data \
    --epochs 2 \
    --batch 1 \
    --vars "sst,u10,v10"
```

## 📈 输出结果

### 预测文件
- **格式**：NetCDF (.nc)
- **命名规则**：`preds_模型名_年月_区域.nc`
- **示例**：`preds_convLSTM_202407_region.nc`

### 可视化结果
- **对比图片**：`prediction_comparison.png`
- **对比内容**：每行显示一个变量的三列对比
  - 左列：真实观测值
  - 中列：模型预测值
  - 右列：差异值 (Pred-True)

### 评估指标
- **RMSE**：均方根误差
- **MAE**：平均绝对误差

## 🔍 关键配置位置

### 1. 数据配置位置
- **文件**：`scripts/train_eval.py`
- **函数**：`main()` 函数开头部分
- **参数**：`--data`, `--vars`, `--T`

### 2. 模型配置位置
- **文件**：`scripts/train_eval.py`
- **函数**：`main()` 函数中模型参数部分
- **参数**：`--model`, `--patch`, `--embed`, `--depth`, `--heads`

### 3. 训练配置位置
- **文件**：`scripts/train_eval.py`
- **函数**：`main()` 函数中训练参数部分
- **参数**：`--batch`, `--epochs`, `--lr`, `--num-workers`

### 4. 空间配置位置
- **文件**：`scripts/train_eval.py`
- **函数**：`main()` 函数中数据预处理部分
- **参数**：`--region`, `--fillna`

### 5. 运行配置位置
- **文件**：`scripts/train.bat`
- **文件**：`scripts/quick_train.bat`

## 💡 使用建议

### 新手配置
```bat
# 使用默认配置快速开始
train.bat
```

### 完整变量训练
```bat
# 使用所有可用变量进行训练
C:\Users\ICEY\.conda\envs\climate312\python.exe train_eval.py \
    --data ../data \
    --epochs 10 \
    --batch 2 \
    --vars "mwd,swh,sst,u10,v10"
```

### 自定义区域训练
```bat
# 训练全球数据（不指定区域）
C:\Users\ICEY\.conda\envs\climate312\python.exe train_eval.py \
    --data ../data \
    --epochs 5 \
    --batch 1 \
    --vars "sst,u10,v10" \
    --region ""
```

---
*最后更新：2024年*  
*文档维护：Climate-D-S 项目团队*