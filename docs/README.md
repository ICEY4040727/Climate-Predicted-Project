# Climate-D-S: ERA5 spatio-temporal prediction baselines

# Climate-D-S：ERA5 时空预测基线

This repo provides a minimal, runnable scaffold to try several forecasting backbones on ERA5-like spatio-temporal grids:
- Persistence (implied by metrics; not separately implemented)
- ConvLSTM next-frame forecast
- Lightweight Spatio-Temporal Transformer next-frame forecast

本仓库提供一个最小可运行的脚手架，用于在 ERA5 风格的时空网格上尝试多种预测骨干网络：
- Persistence（通过指标体现，未单独实现）
- ConvLSTM 下一帧预测
- 轻量级时空 Transformer 下一帧预测

It also includes data loading from local `era5_*.nc` files via xarray and basic metrics.

仓库还包含通过 `xarray` 从本地 `era5_*.nc` 文件加载数据以及基础指标计算。

## Quick start (Windows + uv + PyCharm)

## 快速开始（Windows + uv + PyCharm）

1) Install deps with uv in your project directory:

1）在项目目录用 uv 安装依赖：

```bat
uv pip install -r requirements.txt
```

If you want GPU Torch, replace the `torch` wheel with the appropriate CUDA build, e.g. (CUDA 12.1):

如果想安装 GPU 版本的 PyTorch，请用对应 CUDA 版本的 wheel：

```bat
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

2) Verify the interpreter in PyCharm matches uv

2）确认 PyCharm 中的解释器与 uv 安装的环境一致：
- In PyCharm, go to Settings > Project > Python Interpreter and select the same environment uv installed into.
- If using a uv-managed venv, point PyCharm to that venv's `python.exe`.

- 在 PyCharm 中，前往 Settings（设置）> Project > Python Interpreter，将解释器设置为 uv 安装的环境。
- 如果使用 uv 管理的虚拟环境，可直接指向该 venv 下的 `python.exe`。

3) Run a quick train on your local ERA5 files

3）在本地 ERA5 文件上运行快速训练测试：

```bat
uv run python train_eval.py --data . --vars sst,u10,v10 --T 4 --model convLSTM --epochs 2
```

Or try the Spatio-Temporal Transformer (ensure H/W divisible by patch, default 4):

也可以尝试时空 Transformer（注意 H/W 能被 patch 整除，默认 patch=4）：

```bat
uv run python train_eval.py --model stT --patch 4 --embed 128 --depth 3 --heads 4 --epochs 2
```

## Data expectations

## 数据要求

- Place monthly NetCDF files like `era5_YYYYMM.nc` under the project root (already present in your case).
- Variables must exist in these files; adjust `--vars` accordingly (you can inspect variables via xarray or `ncdump -h`).
- File naming convention: Files should match the pattern `era5_*.nc` (e.g., `era5_202401.nc`). If your files have a different naming scheme, use the `--pattern` argument to specify a custom pattern.

- 将 `era5_YYYYMM.nc` 形式的月度 NetCDF 文件放在项目根目录（或用 `--data` 指定其他目录）。
- 变量必须存在于这些文件中，请根据实际变量名调整 `--vars`（可用 xarray 或 `ncdump -h` 查看）。
- 文件命名约定：匹配 `era5_*.nc` 模式（例如 `era5_202401.nc`），如命名不同请使用 `--pattern` 指定模式。

## How it works

## 工作原理

- `src/data.py`: loads multiple NetCDF files into a single xarray Dataset, stacks chosen variables as channels, yields sliding windows `(T, C, H, W)` and targets next frame `(C, H, W)`.
- `src/models.py`: ConvLSTM baseline and a compact Spatio-Temporal Transformer (space-time factorized attention with patch embedding) predicting next frame.
- `src/utils/metrics.py`: RMSE/MAE with optional masks.
- `train_eval.py`: training and validation loop with simple MSE loss and reporting.

- `src/data.py`：将多个 NetCDF 文件合并为一个 xarray Dataset，按选定变量堆叠为通道，生成滑动窗口 `(T, C, H, W)` 及下一帧目标 `(C, H, W)`。
- `src/models.py`：实现 ConvLSTM 基线和一个紧凑的时空 Transformer（基于 patch 嵌入与时空因式注意力）用于下一帧预测。
- `src/utils/metrics.py`：包含带可选掩码的 RMSE/MAE 等指标计算。
- `train_eval.py`：训练与验证循环，使用简单的 MSE 损失并打印训练/验证报告。

## Notes

## 说明

- For multi-step forecasting, extend the head to produce K steps or roll forward autoregressively.
- For SR, pair an SR model (e.g., EDSR/SwinIR) with synthetic downsampling to create LR-HR pairs; you can reuse the data loader logic but alter targets.
- If your ERA5 files have different grid/coords or missing vars, preprocess to a common grid (e.g., xESMF) before training.

- 对于多步预测，可以扩展模型输出为 K 步或采用自回归推进。
- 对于超分辨率（SR），可用合成下采样构造低/高分辨率对，并复用数据加载逻辑来改变目标。
- 若 ERA5 文件存在不同网格/坐标或缺失变量，建议在训练前使用例如 `xESMF` 将数据重网格化到统一网格。

## Test / 环境自测

## 测试 / 环境自测

- A simple `test_env.py` script is included to verify environment dependencies (NumPy, PyTorch, xarray, netCDF4, matplotlib, etc.).
- 本项目包含 `test_env.py` 脚本，用于快速验证环境依赖（NumPy、PyTorch、xarray、netCDF4、matplotlib 等）。

## License / 许可证

## 许可证

Use the repository's original license terms.

请遵守本仓库原始许可证的使用条款。
