# 更新说明 - 自动可视化功能

## 新增功能

### 1. 自动生成预测对比图
训练完成后会自动生成 `prediction_comparison.png`，包含：
- **每个变量（sst, u10, v10）一行**
- **三列对比**：
  - 真实值（Ground Truth）
  - 预测值（Prediction）  
  - 差异图（Difference = Pred - True）
- **自动显示RMSE和MAE**指标
- **图像会自动打开**（Windows/Mac/Linux）

### 2. 智能默认参数
现在无需指定大量参数即可开始训练：

```cmd
python train_eval.py
```

默认会：
- ✅ 使用**时空Transformer**模型（state-of-the-art架构）
- ✅ 自动裁剪到**赤道太平洋小区域**（-10°~10°N, 100°~160°E）
- ✅ 使用 **sst, u10, v10** 三个变量
- ✅ **前向填充**缺失值（ffill）
- ✅ 训练 **2轮**（epochs=2）
- ✅ 批次大小为 **1**
- ✅ **自动生成并打开**预测对比图

### 3. 更友好的训练进度提示
现在会显示：
- 数据加载进度
- 每个epoch的总batch数
- 每10个batch打印一次loss（可通过`--log-interval`调整）
- 验证集评估提示

### 4. 便捷的批处理脚本

#### `quick_train.bat` - 快速测试（约1-2分钟）
```cmd
quick_train.bat
```
运行1轮训练，5个训练batch，验证生成的可视化。

#### `train.bat` - 完整训练
```cmd
train.bat
```
运行10轮完整训练，适合实际使用。

## 使用示例

### 基础用法（推荐）
```cmd
# 方式1：直接运行（使用所有智能默认值）
python train_eval.py

# 方式2：使用批处理脚本快速测试
quick_train.bat
```

### 进阶用法
```cmd
# 快速测试（1轮，少量batch）
python train_eval.py --epochs 1 --max-train-batches 10

# GPU训练（自动检测）
python train_eval.py --epochs 20 --batch 4

# 禁用可视化（仅训练）
python train_eval.py --no-visualize

# 保存预测为NetCDF
python train_eval.py --save-preds my_predictions.nc

# 查看数据中的变量
python train_eval.py --list-vars
```

## 可视化输出示例

训练完成后，会看到类似这样的图像（自动打开）：

```
┌──────────────────────────────────────────────────────────┐
│  sst - True     │  sst - Predicted  │  sst - Difference  │
│  [热力图]       │  [热力图]         │  [差异图+RMSE/MAE] │
├──────────────────────────────────────────────────────────┤
│  u10 - True     │  u10 - Predicted  │  u10 - Difference  │
│  [热力图]       │  [热力图]         │  [差异图+RMSE/MAE] │
├──────────────────────────────────────────────────────────┤
│  v10 - True     │  v10 - Predicted  │  v10 - Difference  │
│  [热力图]       │  [热力图]         │  [差异图+RMSE/MAE] │
└──────────────────────────────────────────────────────────┘
```

## 技术改进

1. **自动区域裁剪**：默认从全球数据(721×1440)裁剪到小区域(81×241)，训练速度提升约**50倍**
2. **自动缺失值处理**：默认使用ffill填充，避免训练中的NaN错误
3. **智能patch处理**：当H/W不能被patch整除时，自动降级到patch=1并给出提示
4. **进度可见性**：实时显示训练进度，避免"卡住"的疑虑

## 依赖更新

新增了 `matplotlib` 用于可视化，已添加到 `requirements.txt`：
```
pip install matplotlib
```

## 快速诊断

如果遇到问题：

1. **训练很慢？** → 已自动裁剪区域；若仍慢，使用GPU或减小batch/model大小
2. **图像没打开？** → 已保存到 `prediction_comparison.png`，手动打开即可
3. **想改变区域？** → 使用 `--region=lat_min,lat_max,lon_min,lon_max`
4. **想用ConvLSTM？** → 使用 `--model convLSTM`

## 完整文档

详细说明请参阅：**`QUICK_START.md`**

