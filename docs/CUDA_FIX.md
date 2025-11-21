# CUDA兼容性问题解决方案

## 问题说明

如果你遇到以下错误：
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

这表示你安装的PyTorch版本使用的CUDA版本与你的GPU驱动/CUDA版本不匹配。

## 自动解决方案 ✅

**好消息**：代码已经更新，会自动检测CUDA兼容性问题并回退到CPU！

现在运行 `python train_eval.py` 或 `quick_train.bat` 时：
1. 程序会先尝试使用CUDA
2. 如果检测到CUDA错误，**自动切换到CPU**
3. 显示如何安装正确的PyTorch版本的提示

## 快速解决方案

### 方法1：强制使用CPU（最简单）

```cmd
quick_train_cpu.bat
```

或手动设置环境变量：
```cmd
set CUDA_VISIBLE_DEVICES=-1
python train_eval.py
```

### 方法2：检查并重装匹配的PyTorch（推荐用于长期训练）

#### 步骤1：检查你的CUDA版本

在CMD中运行：
```cmd
nvidia-smi
```

查看右上角的 "CUDA Version"，例如：`CUDA Version: 12.1`

#### 步骤2：卸载当前PyTorch

```cmd
pip uninstall torch torchvision torchaudio
```

#### 步骤3：安装匹配的PyTorch

根据你的CUDA版本选择：

**CUDA 11.8**:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4** (或更高版本):
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**如果没有GPU或想用CPU**:
```cmd
pip install torch torchvision torchaudio
```

#### 步骤4：验证安装

```cmd
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

应该看到：
```
PyTorch: 2.x.x+cu121  (或其他cu版本)
CUDA available: True
```

## 性能对比

| 设备 | 训练速度（相对） | 推荐场景 |
|------|------------------|----------|
| GPU  | 10-50x 更快      | 长时间训练、大批次、大模型 |
| CPU  | 基准速度         | 快速测试、小批次、小区域 |

## 当前优化已经很快！

由于代码已经自动裁剪到小区域（-10°~10°N, 100°~160°E），**即使用CPU训练也很快**：
- 全球数据：721×1440 像素
- 裁剪后区域：81×241 像素（约**50倍更小**）
- CPU训练时间：约几分钟（2轮训练）

## 使用建议

### 快速测试/开发
```cmd
quick_train_cpu.bat
```
或
```cmd
python train_eval.py --epochs 1 --max-train-batches 10
```

### 短期训练（2-10轮）
CPU即可，速度可接受

### 长期训练（50+轮）或大批次
建议修复CUDA兼容性或使用服务器GPU

## 在服务器上训练

如果你有访问权限的Linux服务器（有GPU）：

1. **检查服务器CUDA版本**：
```bash
nvidia-smi
```

2. **创建conda环境并安装匹配的PyTorch**：
```bash
conda create -n climate python=3.12
conda activate climate
# 根据CUDA版本选择合适的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

3. **上传代码和数据**，然后运行：
```bash
python train_eval.py --epochs 50 --batch 4
```

4. **使用tmux后台运行**：
```bash
tmux new -s train
python train_eval.py --epochs 50 --batch 4 --log-interval 20
# 按 Ctrl+B 然后 D 分离会话
```

5. **重新连接查看进度**：
```bash
tmux attach -t train
```

## 常见问题

**Q: 为什么会出现CUDA版本不匹配？**
A: 
- PyTorch预编译时绑定了特定CUDA版本
- 你的GPU驱动支持的CUDA版本可能不同
- 从pip安装的默认PyTorch可能不匹配你的系统

**Q: CPU训练慢吗？**
A: 
- 对于默认的小区域（81×241），CPU训练速度可接受（几分钟完成2轮）
- 如果使用全球数据或大批次，GPU会明显更快

**Q: 我应该用CPU还是修复CUDA？**
A:
- **快速测试/开发**：用CPU即可（已经够快）
- **短期训练（<10轮）**：CPU可接受
- **长期训练/大规模实验**：建议修复CUDA或使用服务器

**Q: 代码还会尝试使用GPU吗？**
A: 
- 是的，代码会**自动尝试GPU**
- 如果检测到CUDA错误，会**自动回退到CPU**
- 无需手动干预

**Q: 如何强制使用CPU？**
A:
```cmd
set CUDA_VISIBLE_DEVICES=-1
python train_eval.py
```
或直接运行：
```cmd
quick_train_cpu.bat
```

## 总结

✅ **问题已自动处理**：代码会自动回退到CPU  
✅ **CPU训练可用**：由于自动区域裁剪，速度可接受  
✅ **可选GPU优化**：如需长期训练，可参考本文档重装PyTorch  
✅ **强制CPU脚本**：`quick_train_cpu.bat` 可直接使用  

立即测试：`quick_train_cpu.bat` 或 `python train_eval.py` 🚀

