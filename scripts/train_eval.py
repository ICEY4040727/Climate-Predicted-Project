# ERA5气象数据深度学习训练与评估脚本
# 功能：使用ConvLSTM或时空Transformer模型对ERA5再分析数据进行时空预测
import argparse
from pathlib import Path
import sys
import os
import yaml  # 导入YAML库

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

# 导入自定义模块
from src.data import ERA5TensorDataset, find_nc_files  # 数据加载和预处理
from src.models import ConvLSTMForecast, SpatioTemporalTransformer  # 深度学习模型
from src.utils.metrics import rmse, mae  # 评估指标


def get_device():
    """自动选择计算设备：优先使用GPU，否则使用CPU"""
    if not torch.cuda.is_available():
        print("检测不到 CUDA，可用设备：CPU。")
        return 'cpu'

    try:
        # 尝试一个简单的 CUDA 操作以验证兼容性
        test_tensor = torch.zeros(1, device='cuda')
        _ = test_tensor + 1
        print(f"检测到 CUDA：{torch.cuda.get_device_name(0)}")
        return 'cuda'
    except RuntimeError as e:
        # 捕获 CUDA 运行时错误并回退到 CPU，同时给出重装 PyTorch 的提示
        print(f"警告：检测到 CUDA 错误：{e}")
        print("这通常表示 PyTorch 的 CUDA 版本与 GPU 驱动不兼容。")
        print("将回退到 CPU。若要使用 GPU，请根据你的 CUDA 版本重新安装 PyTorch：")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  （将 cu121 替换为你的 CUDA 版本：cu118、cu121、cu124 等）")
        return 'cpu'


def build_dataloaders(data_dir: Path, vars, T, batch=2, num_workers=0, region=None, fillna=None, time_slices=None, time_points=None, pattern: str = 'era5_*.nc'):
    """
    构建训练集和验证集的数据加载器

    参数:
        data_dir: 数据文件目录路径
        vars: 需要使用的变量列表
        T: 输入序列长度（时间窗口大小）
        batch: 批次大小
        num_workers: 数据加载的工作进程数
        region: 可选的空间裁剪区域
        fillna: 缺失值处理方法
        time_slices: 可选的时间切片范围列表
        time_points: 可选的特定时间点列表
        pattern: 文件匹配模式

    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        ds: 数据集对象
    """
    # 查找符合模式的数据文件
    files = find_nc_files(data_dir, pattern=pattern)
    if not files:
        raise SystemExit(f"在目录 {data_dir} 下未找到与模式 '{pattern}' 匹配的文件")

    # 创建ERA5张量数据集
    ds = ERA5TensorDataset(files, vars=vars, time_window=T, stride=1, normalize=True, region=region, fillna=fillna, time_slices=time_slices, time_points=time_points)

    n = len(ds)
    print(f"数据集总样本数: {n}")

    # 确保至少有一个样本用于训练和验证
    if n == 0:
        raise ValueError("数据集为空。请检查数据文件和选择标准。")
    elif n == 1:
        # 只有一个样本时，训练集和验证集都使用同一个样本
        train_size = 1
        val_size = 1
        print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
        # 在这种特殊情况下，我们创建一个自定义的数据集分割
        # 两个分割都包含同一个样本（索引0）
        from torch.utils.data import Subset
        train_ds = Subset(ds, [0])
        val_ds = Subset(ds, [0])
        # 注意：这种情况下训练集和验证集会有重叠
    else:
        # 根据数据集大小确定训练集/验证集分割比例
        if n < 10:
            train_size = max(1, int(n * 0.8))  # 小数据集使用80%训练，但至少保留1个样本
        else:
            train_size = int(n * 0.9)  # 大数据集使用90%训练
        val_size = n - train_size

        # 确保验证集至少有一个样本
        if val_size == 0:
            val_size = 1
            train_size = n - val_size

        print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

        # 随机分割数据集（固定随机种子以保证可重复性）
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=generator)

    print(f"实际训练集大小: {len(train_ds)}, 实际验证集大小: {len(val_ds)}")

    # 创建数据加载器（对小数据集自动调整批次大小）
    actual_batch_size = min(batch, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True, num_workers=num_workers)

    actual_val_batch_size = min(batch, len(val_ds))
    val_loader = DataLoader(val_ds, batch_size=actual_val_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, ds


def train_one_epoch(model, loader, device, optimizer, scaler=None, log_interval: int = 0, max_batches: int | None = None):
    """
    训练一个 epoch

    参数:
        model: 神经网络模型
        loader: 数据加载器
        device: 计算设备
        optimizer: 优化器
        scaler: 混合精度训练的梯度缩放器（可选）
        log_interval: 每多少个训练 batch 打印一次 loss，0 为不打印
        max_batches: 每个 epoch 最多训练多少个 batch

    返回:
        平均训练损失
    """
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    count = 0

    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        # 标准化形状：如果 y 有意外的时间维度则压缩
        if y.dim() == 5 and y.size(1) == 1:
            y = y[:, 0]

        optimizer.zero_grad()

        if scaler is not None:
            # 使用混合精度训练（CUDA）
            with torch.amp.autocast(device_type='cuda'):
                y_hat = model(x)
                if y_hat.dim() == 5 and y_hat.size(1) == 1:
                    y_hat = y_hat[:, 0]
                loss = loss_fn(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通精度训练
            y_hat = model(x)
            if y_hat.dim() == 5 and y_hat.size(1) == 1:
                y_hat = y_hat[:, 0]
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

        total += float(loss.detach().cpu())
        count += 1

        if log_interval and (i % log_interval == 0):
            print(f"  [train] step {i} loss={float(loss.detach().cpu()):.4f}")

        if max_batches is not None and i >= max_batches:
            break

    return total / max(count, 1)


def evaluate(model, loader, device, max_batches: int | None = None):
    """
    在验证集上评估模型性能，返回 MSE、RMSE、MAE

    参数:
        model: 神经网络模型
        loader: 验证数据加载器
        device: 计算设备
        max_batches: 验证时最多评估多少个 batch

    返回:
        字典，包含 'mse','rmse','mae'
    """
    model.eval()
    mses = []
    rmses = []
    maes = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)

            # 标准化形状
            if y.dim() == 5 and y.size(1) == 1:
                y = y[:, 0]

            y_hat = model(x)
            if y_hat.dim() == 5 and y_hat.size(1) == 1:
                y_hat = y_hat[:, 0]

            mses.append(nn.functional.mse_loss(y_hat, y).item())
            rmses.append(rmse(y.cpu(), y_hat.cpu()))
            maes.append(mae(y.cpu(), y_hat.cpu()))

            if max_batches is not None and i >= max_batches:
                break

    return {
        'mse': sum(mses) / len(mses) if mses else float('nan'),
        'rmse': sum(rmses) / len(rmses) if rmses else float('nan'),
        'mae': sum(maes) / len(maes) if maes else float('nan'),
    }


def save_preds_nc(out_path: Path, y_true: torch.Tensor, y_pred: torch.Tensor, ds: ERA5TensorDataset):
    """
    将预测结果保存为 NetCDF 文件

    参数:
        out_path: 输出文件路径
        y_true: 真实值张量
        y_pred: 预测值张量
        ds: 数据集对象（用于获取元数据）
    """
    import xarray as xr
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    yt = y_true.detach().cpu()
    yp = y_pred.detach().cpu()

    # 去掉多余的时间单例维度 (例如 (B,1,C,H,W) -> (B,C,H,W))
    if yt.dim() == 5 and yt.size(1) == 1:
        yt = yt[:, 0]
    if yp.dim() == 5 and yp.size(1) == 1:
        yp = yp[:, 0]

    yt_np = yt.numpy()
    yp_np = yp.numpy()

    # 确保存在批次维度
    if yt_np.ndim == 3:
        yt_np = yt_np[None]
        yp_np = yp_np[None]

    # 若维度超过4维，将前导维度合并到批次维
    if yt_np.ndim > 4:
        new_shape = (-1,) + yt_np.shape[-3:]
        yt_np = yt_np.reshape(new_shape)
        yp_np = yp_np.reshape(new_shape)

    B, C, H, W = yt_np.shape

    # 尝试反标准化
    try:
        mean, std = ds._norm
        yt_np = yt_np * std[None, :, None, None] + mean[None, :, None, None]
        yp_np = yp_np * std[None, :, None, None] + mean[None, :, None, None]
    except Exception:
        pass

    coords = {
        'sample': np.arange(B),
        'channel': list(ds.vars),
        'latitude': ds.da['latitude'].values,
        'longitude': ds.da['longitude'].values,
    }

    da_true = xr.DataArray(yt_np, dims=('sample', 'channel', 'latitude', 'longitude'), coords=coords, name='target')
    da_pred = xr.DataArray(yp_np, dims=('sample', 'channel', 'latitude', 'longitude'), coords=coords, name='prediction')
    xr.Dataset({'target': da_true, 'prediction': da_pred}).to_netcdf(out_path)


def visualize_predictions(y_true: torch.Tensor, y_pred: torch.Tensor, ds: ERA5TensorDataset, save_path: Path = None, meta: dict | None = None):
    """
    可视化预测结果与真实值的对比

    参数:
        y_true: 真实值张量
        y_pred: 预测值张量
        ds: 数据集对象（用于获取变量名和反标准化）
        save_path: 图像保存路径（可选）
        meta: 可选的元信息，例如 {'input_times': [...], 'target_time': ...}
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过可视化。可运行: pip install matplotlib")
        return

    yt = y_true.detach().cpu()
    yp = y_pred.detach().cpu()

    # 去掉多余的时间单例维度
    if yt.dim() == 5 and yt.size(1) == 1:
        yt = yt[:, 0]
    if yp.dim() == 5 and yp.size(1) == 1:
        yp = yp[:, 0]

    yt_np = yt.numpy()
    yp_np = yp.numpy()

    # 确保存在批次维度
    if yt_np.ndim == 3:
        yt_np = yt_np[None]
        yp_np = yp_np[None]

    # 若维度超过4维，将前导维度合并到批次维
    if yt_np.ndim > 4:
        new_shape = (-1,) + yt_np.shape[-3:]
        yt_np = yt_np.reshape(new_shape)
        yp_np = yp_np.reshape(new_shape)

    B, C, H, W = yt_np.shape

    # 尝试反标准化
    try:
        mean, std = ds._norm
        yt_np = yt_np * std[None, :, None, None] + mean[None, :, None, None]
        yp_np = yp_np * std[None, :, None, None] + mean[None, :, None, None]
    except Exception:
        pass

    # 选择第一个样本进行可视化
    sample_idx = 0

    # 获取经纬度范围及绘图 extent
    lat_min = lat_max = lon_min = lon_max = None
    try:
        lats = ds.da['latitude'].values
        lons = ds.da['longitude'].values
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        extent = [lon_min, lon_max, lat_min, lat_max]
    except Exception:
        lats = lons = None
        extent = None

    # 获取单位（如果可用）
    def get_units(var_name: str) -> str:
        try:
            return str(ds.ds[var_name].attrs.get('units', '') or '')
        except Exception:
            return ''

    # 创建子图：每个变量一行，三列（真实值、预测值、差异）
    fig, axes = plt.subplots(C, 3, figsize=(15, 4.8 * C), constrained_layout=True)
    if C == 1:
        axes = axes[None, :]

    var_names = ds.vars
    metrics_rows = [("variable", "rmse", "mae", "vmin", "vmax", "units")]

    for i, var_name in enumerate(var_names):
        true_field = yt_np[sample_idx, i]
        pred_field = yp_np[sample_idx, i]
        diff_field = pred_field - true_field

        # 计算全局 vmin/vmax 保持颜色尺度一致
        vmin = float(min(true_field.min(), pred_field.min()))
        vmax = float(max(true_field.max(), pred_field.max()))
        units = get_units(var_name)

        # 真实值
        im0 = axes[i, 0].imshow(
            true_field,
            cmap='RdBu_r', vmin=vmin, vmax=vmax,
            aspect='auto',
            extent=extent if extent is not None else None,
            origin='upper'
        )
        axes[i, 0].set_title(f'{var_name} - True', fontsize=11, fontweight='bold')
        axes[i, 0].set_xlabel('Longitude')
        axes[i, 0].set_ylabel('Latitude')
        cbar0 = fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        if units:
            cbar0.set_label(units)

        # 预测值
        im1 = axes[i, 1].imshow(
            pred_field,
            cmap='RdBu_r', vmin=vmin, vmax=vmax,
            aspect='auto',
            extent=extent if extent is not None else None,
            origin='upper'
        )
        axes[i, 1].set_title(f'{var_name} - Predicted', fontsize=11, fontweight='bold')
        axes[i, 1].set_xlabel('Longitude')
        axes[i, 1].set_ylabel('Latitude')
        cbar1 = fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        if units:
            cbar1.set_label(units)

        # 差异
        diff_max = float(max(abs(diff_field.min()), abs(diff_field.max())))
        im2 = axes[i, 2].imshow(
            diff_field,
            cmap='seismic', vmin=-diff_max, vmax=diff_max,
            aspect='auto',
            extent=extent if extent is not None else None,
            origin='upper'
        )
        axes[i, 2].set_title(f'{var_name} - Difference (Pred - True)', fontsize=11, fontweight='bold')
        axes[i, 2].set_xlabel('Longitude')
        axes[i, 2].set_ylabel('Latitude')
        cbar2 = fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        if units:
            cbar2.set_label(units)

        # 添加统计信息
        rmse_val = float(np.sqrt(np.mean(diff_field**2)))
        mae_val = float(np.mean(np.abs(diff_field)))
        axes[i, 2].text(
            0.02, 0.98, f'RMSE: {rmse_val:.4f}\nMAE: {mae_val:.4f}',
            transform=axes[i, 2].transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # 将数值转为字符串以便 CSV 中一致（并避免静态类型警告）
        metrics_rows.append((str(var_name), f"{rmse_val:.6f}", f"{mae_val:.6f}", f"{vmin:.6f}", f"{vmax:.6f}", str(units)))

    # 总标题与说明
    if extent is not None and None not in (lat_min, lat_max, lon_min, lon_max):
        region_str = f"lat[{lat_min:.2f},{lat_max:.2f}], lon[{lon_min:.2f},{lon_max:.2f}]"
    else:
        region_str = "region: unknown"

    subtitle = f"Prediction vs Truth (T+1) | {region_str}"
    if meta and ('target_time' in meta or 'input_times' in meta):
        try:
            tgt = meta.get('target_time', None)
            ins = meta.get('input_times', None)
            if isinstance(ins, (list, tuple)):
                ins_str = ', '.join(str(t) for t in ins)
            else:
                ins_str = None
            if tgt is not None and ins_str is not None:
                subtitle += f"\nInput: [{ins_str}] -> Target: {tgt}"
            elif tgt is not None:
                subtitle += f"\nTarget: {tgt}"
        except Exception:
            pass

    fig.suptitle(subtitle, fontsize=13, fontweight='bold')
    fig.text(
        0.5, 0.01,
        "阅读指南：左=真实值，中=预测，右=差异(Pred-True)。每行同一变量共享颜色尺度；差异图使用对称尺度。",
        ha='center', va='bottom', fontsize=10
    )

    # 保存图像
    if save_path is None:
        save_path = Path('prediction_comparison.png')
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化图像已保存到：{save_path}")

    # 另存每变量 RMSE/MAE 到 CSV（与图同目录）
    try:
        import csv
        csv_path = save_path.with_name(save_path.stem + '_metrics.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_rows)
        print(f"评估指标已保存到：{csv_path}")
    except Exception as e:
        print(f"保存指标 CSV 失败：{e}")

    # 同步保存时间元数据到 txt
    try:
        if meta:
            meta_path = save_path.with_name(save_path.stem + '_meta.txt')
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(f"region: {region_str}\n")
                if 'sample_index' in meta:
                    f.write(f"sample_index: {meta['sample_index']}\n")
                if 'input_times' in meta:
                    f.write("input_times:\n")
                    for t in meta['input_times']:
                        f.write(f"  - {t}\n")
                if 'target_time' in meta:
                    f.write(f"target_time: {meta['target_time']}\n")
            print(f"元数据已保存到：{meta_path}")
    except Exception as e:
        print(f"保存元数据失败：{e}")

    # 尝试自动打开图像
    try:
        import os
        import platform
        if platform.system() == 'Windows':
            os.startfile(save_path)
        elif platform.system() == 'Darwin':
            os.system(f'open "{save_path}"')
        else:
            os.system(f'xdg-open "{save_path}"')
        print("已在默认图像查看器中打开可视化结果...")
    except Exception as e:
        print(f"无法自动打开图像：{e}，请手动打开 {save_path}")

    plt.close(fig)


def predict_autoregressive(model, x_initial: torch.Tensor, horizon: int, device):
    """
    使用自回归方式进行多步预测

    参数:
        model: 预测模型
        x_initial: 初始输入张量 (1, T, C, H, W)
        horizon: 预测步长
        device: 计算设备

    返回:
        最终预测结果张量 (1, C, H, W)
    """
    if horizon <= 0:
        raise ValueError("预测步长必须为正整数。")

    x_current = x_initial.clone().to(device)

    print(f"开始自回归预测，步长: {horizon}...")
    for i in range(horizon):
        # 模型预测下一步
        with torch.no_grad():
            y_pred_step = model(x_current) # (1, C, H, W) or (1, 1, C, H, W)

        # 保证 y_pred_step 是 (1, 1, C, H, W) 以便拼接
        if y_pred_step.dim() == 4:
            y_pred_step = y_pred_step.unsqueeze(1)
        elif y_pred_step.dim() == 5 and y_pred_step.size(1) != 1:
             # 如果模型输出了多步，我们只取第一步
            y_pred_step = y_pred_step[:, 0:1]

        print(f"  [自回归] 步骤 {i+1}/{horizon} 完成")

        # 如果是最后一步，直接返回结果
        if i == horizon - 1:
            # 返回 (1, C, H, W)
            return y_pred_step.squeeze(1)

        # 更新输入窗口：移除最旧的一帧，添加最新预测的一帧
        x_current = torch.cat([x_current[:, 1:], y_pred_step], dim=1)

    # 理论上不会执行到这里，但在循环结束时返回最后一次的预测
    return y_pred_step.squeeze(1)


def main():
    """
    主函数：解析参数并执行训练流程

    训练配置说明：
    - 优先从 --config 加载 YAML 配置文件。
    - 命令行参数可以覆盖配置文件中的同名设置。
    - 输入数据：连续 T 小时的气象数据（T 帧）或手动输入指定时间点的数据
    - 输出预测：第 T+1 小时的气象数据（1 帧）或手动输入指定时间点的数据
    """
    parser = argparse.ArgumentParser(description='ERA5气象数据时空预测训练程序', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 配置文件路径参数
    parser.add_argument('--config', type=str, default=None, help='YAML 配置文件路径')

    # 在这里添加缺失的参数定义
    parser.add_argument('--list-vars', action='store_true', help='列出数据文件中的所有可用变量并退出')

    # 数据相关参数
    parser.add_argument('--data', type=str, help='数据文件目录')
    parser.add_argument('--pattern', type=str, help="文件匹配模式")
    parser.add_argument('--vars', type=str, help='使用的变量列表，逗号分隔')
    parser.add_argument('--T', type=int, help='输入序列长度（时间窗口）')
    parser.add_argument('--region', type=str, help='空间裁剪区域，格式：lat_min,lat_max,lon_min,lon_max')
    parser.add_argument('--fillna', type=str, help="缺失值处理方法：'ffill','bfill','linear','nearest'")
    # 更新 time_slice 的帮助文本，并添加 time_slices
    parser.add_argument('--time-slice', type=str, help='(已弃用, 请使用 --time-slices) 选择特定时间范围的数据进行训练，格式："start_time:end_time"')
    parser.add_argument('--time-slices', type=str, nargs='*', help='一个或多个时间范围，格式: "start1:end1" "start2:end2"...')
    parser.add_argument('--time-points', type=str, help='选择特定时间点的数据进行训练，格式："time1,time2,time3"')

    # 模型相关参数
    parser.add_argument('--model', type=str, choices=['convLSTM', 'stT'], help='模型类型：convLSTM 或 stT')
    parser.add_argument('--patch', type=int, help='Transformer 的 patch 大小')
    parser.add_argument('--embed', type=int, help='嵌入维度')
    parser.add_argument('--depth', type=int, help='Transformer 层数')
    parser.add_argument('--heads', type=int, help='注意力头数')

    # 训练相关参数
    parser.add_argument('--batch', type=int, help='批次大小')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--num-workers', type=int, help='数据加载工作进程数')
    parser.add_argument('--min-samples', type=int, help='训练所需的最小样本数')
    parser.add_argument('--log-interval', type=int, help='每多少个训练 batch 打印一次 loss')
    parser.add_argument('--max-train-batches', type=int, help='每个 epoch 最多训练多少个 batch')
    parser.add_argument('--max-val-batches', type=int, help='验证时最多评估多少个 batch')

    # 输出与命名相关参数
    parser.add_argument('--output-dir', type=str, help='所有输出文件的根目录')
    parser.add_argument('--run-name', type=str, help='本次运行的名称，用于创建子目录。如果留空，将自动生成')
    parser.add_argument('--save-preds', type=str, help='(已弃用) 保存预测结果的文件名，现在由脚本自动管理')
    parser.add_argument('--visualize', action='store_true', help='训练后自动生成并打开预测对比图')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize', help='禁用自动可视化')
    parser.add_argument('--viz-path', type=str, help='(已弃用) 可视化图像保存路径，现在由脚本自动管理')
    parser.add_argument('--predict-time-point', type=str, help='指定用于预测的特定时间点')
    parser.add_argument('--predict-time-ignore-year', action='store_true',
                    help='匹配时忽略年份：允许用 "MM-DD" 或 "MM-DDTHH" 等格式匹配不同年份的同一月日时')

    args = parser.parse_args()

    # 默认配置
    config = {
        'data': {
            'path': '../data',
            'pattern': 'era5_*_extracted/*data_stream-oper_stepType-instant.nc',
            'vars': 'sst,u10,v10',
            'time_window': 4,
            'region': '-10,10,100,160',
            'fillna': 'ffill',
            'time_slice': None, # 保留以便兼容旧配置
            'time_slices': None, # 新增
            'time_points': None,
        },
        'model': {
            'type': 'stT',
            'patch_size': 1,
            'embed_dim': 128,
            'depth': 3,
            'heads': 4,
        },
        'training': {
            'batch_size': 1,
            'epochs': 2,
            'learning_rate': 1e-3,
            'num_workers': 0,
            'min_samples': 2,
            'log_interval': 10,
            'max_train_batches': None,
            'max_val_batches': None,
        },
        'output': {
            'output_dir': 'outputs',
            'run_name': None,
            'visualize': True,
            'visualization_path': 'prediction_comparison.png', # 兼容旧版
            'save_predictions_path': None, # 兼容旧版
            'predict_time_point': None,
            'predict_time_ignore_year': False,
        }
    }

    # 如果提供了配置文件，则加载并合并
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        # 深层合并字典
        for key, value in yaml_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    # 将配置应用到 argparse 的命名空间，命令行参数优先
    # 数据
    args.data = args.data or config['data']['path']
    args.pattern = args.pattern or config['data']['pattern']
    args.vars = args.vars or config['data']['vars']
    args.T = args.T or config['data']['time_window']
    # region 可以是列表或字符串
    if args.region is None:
        region_val = config['data'].get('region')
        if isinstance(region_val, list):
            args.region = ','.join(map(str, region_val))
        else:
            args.region = region_val
    args.fillna = args.fillna or config['data']['fillna']
    # 优先使用 time_slices, 其次是 time_slice
    args.time_slices = args.time_slices or config['data'].get('time_slices')
    if not args.time_slices:
        # 如果 time_slices 未提供，则检查旧的 time_slice 参数
        time_slice_val = args.time_slice or config['data'].get('time_slice')
        if time_slice_val:
            args.time_slices = [time_slice_val] # 将单个 slice 包装成列表
    args.time_points = args.time_points or config['data']['time_points']

    # 模型
    args.model = args.model or config['model']['type']
    args.patch = args.patch or config['model']['patch_size']
    args.embed = args.embed or config['model']['embed_dim']
    args.depth = args.depth or config['model']['depth']
    args.heads = args.heads or config['model']['heads']

    # 训练
    args.batch = args.batch or config['training']['batch_size']
    args.epochs = args.epochs or config['training']['epochs']
    args.lr = args.lr or config['training']['learning_rate']
    args.num_workers = args.num_workers if args.num_workers is not None else config['training']['num_workers']
    args.min_samples = args.min_samples or config['training']['min_samples']
    args.log_interval = args.log_interval or config['training']['log_interval']
    args.max_train_batches = args.max_train_batches or config['training']['max_train_batches']
    args.max_val_batches = args.max_val_batches or config['training']['max_val_batches']

    # 输出
    # 对于 action='store_true'/'store_false' 的特殊处理
    if 'visualize' in config['output']:
        # 只有当命令行没有指定 --visualize 或 --no-visualize 时，才使用配置文件的值
        if 'visualize' not in sys.argv and 'no-visualize' not in sys.argv:
            args.visualize = config['output']['visualize']

    # 新的输出路径管理
    args.output_dir = args.output_dir or config['output'].get('output_dir', 'outputs')
    args.run_name = args.run_name or config['output'].get('run_name')

    # 旧的路径参数，仅用于向后兼容，但会被新逻辑覆盖
    args.viz_path = args.viz_path or config['output']['visualization_path']
    args.save_preds = args.save_preds or config['output']['save_predictions_path']

    args.predict_time_point = args.predict_time_point or config['output']['predict_time_point']
    args.predict_time_ignore_year = args.predict_time_ignore_year or config['output'].get('predict_time_ignore_year', False)


    # 设置数据目录并查找文件
    data_dir = Path(args.data)
    files = find_nc_files(data_dir, pattern=args.pattern)
    if not files:
        raise SystemExit(f"在目录 {data_dir} 下未找到与模式 '{args.pattern}' 匹配的文件")

    # 仅列出变量并退出
    if args.list_vars:
        import xarray as xr
        ds = xr.open_mfdataset([str(p) for p in files], combine='by_coords', engine='netcdf4')
        print('数据集中可用的变量:')
        for name in list(ds.data_vars):
            var = ds[name]
            dims = ','.join(map(str, var.dims))
            print(f" - {name}  dims=({dims})  shape={tuple(var.shape)}")
        return

    # 解析变量列表
    vars = [v.strip() for v in args.vars.split(',') if v.strip()]

    # --- 新的运行名称和输出目录生成逻辑 ---
    if not args.run_name:
        from datetime import datetime
        import re

        # 1. 模型和变量部分
        model_str = args.model
        vars_str = '-'.join(vars)

        # 2. 训练数据信息部分
        train_data_info = "all"
        if args.time_slices:
            if len(args.time_slices) == 1:
                # 清理时间字符串，使其适合文件名
                train_data_info = re.sub(r'[-:T]', '', args.time_slices[0])
            else:
                train_data_info = f"multi_{len(args.time_slices)}"
        elif args.time_points:
            num_points = len(args.time_points.split(','))
            train_data_info = f"points_{num_points}"

        # 3. 预测时间信息部分
        pred_time_info = "None"
        if args.predict_time_point:
            pred_time_info = re.sub(r'[-:T]', '', args.predict_time_point)

        # 4. 时间戳
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        # 组合成新的 run_name
        args.run_name = f"{model_str}_{vars_str}_train_{train_data_info}_pred_{pred_time_info}_{timestamp}"

    # 定义本次运行的专属输出目录
    run_output_dir = Path(args.output_dir) / args.run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"本次运行的所有输出将保存到: {run_output_dir}")
    # --- 逻辑结束 ---

    # 解析空间区域参数
    region = None
    if args.region:
        try:
            # 如果 region 是字符串，则分割
            if isinstance(args.region, str):
                parts = [float(x) for x in args.region.split(',')]
            else: # 否则假定是列表
                parts = args.region
            lat_min, lat_max, lon_min, lon_max = parts
            region = {'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max}
        except Exception as e:
            raise SystemExit(f"解析 --region 失败: {e}")

    # 构建数据加载器
    print(f"正在加载数据：匹配模式 '{args.pattern}' 的文件数量：{len(find_nc_files(data_dir, pattern=args.pattern))} ...")
    print(f"变量: {vars}, 区域: {region}, 缺失值处理: {args.fillna}")
    train_loader, val_loader, ds = build_dataloaders(
        data_dir, vars, args.T,
        batch=args.batch, num_workers=args.num_workers,
        region=region, fillna=args.fillna, pattern=args.pattern,
        time_slices=args.time_slices, # 传递 time_slices
        time_points=args.time_points
    )

    print(f"数据集准备就绪：样本数={len(ds)}，形状=({ds.num_channels}, {ds.H}, {ds.W})")

    # 获取计算设备
    device = get_device()
    C = ds.num_channels
    H, W = ds.H, ds.W

    # 根据模型类型创建网络实例
    if args.model == 'convLSTM':
        model = ConvLSTMForecast(in_ch=C, hidden=64, depth=2)
    else:
        chosen_patch = args.patch
        if chosen_patch < 1:
            chosen_patch = 1
        if H % chosen_patch != 0 or W % chosen_patch != 0:
            print(f"警告: H/W ({H},{W}) 不能被 patch={chosen_patch} 整除。回退到 patch=1（不进行空间分块）。")
            chosen_patch = 1
        model = SpatioTemporalTransformer(in_ch=C, out_ch=C, patch=chosen_patch, embed_dim=args.embed, depth=args.depth, num_heads=args.heads, t_max=max(args.T, 24))

    model = model.to(device)
    print(f"使用设备={device}，模型={args.model}，patch={getattr(model, 'patch', 'N/A')}，batch={args.batch}，epochs={args.epochs}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    # 训练循环
    print(f"\n开始训练：{args.epochs} 轮，{len(train_loader)} 个训练批次/轮，{len(val_loader)} 个验证批次/轮")
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== 第 {epoch}/{args.epochs} 轮 ===")
        train_loss = train_one_epoch(model, train_loader, device, optimizer, scaler, log_interval=args.log_interval, max_batches=args.max_train_batches)
        print("正在在验证集上评估...")
        metrics = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
        print(f"Epoch {epoch:02d} | 训练损失={train_loss:.4f} | 验证 MSE={metrics['mse']:.4f} | 验证 RMSE={metrics['rmse']:.4f} | 验证 MAE={metrics['mae']:.4f}")

    # 训练完成后：保存预测结果和/或可视化
    print(f"\n=== 在评估集上生成预测 ===")
    model.eval()
    with torch.no_grad():
        if not args.predict_time_point:
            raise SystemExit("错误: 未提供 --predict-time-point。请指定一个目标预测时间点。")

        import pandas as pd
        target_time_dt = pd.to_datetime(args.predict_time_point)

        all_times = ds.da[ds.time_dim].values
        all_times_dt = pd.to_datetime(all_times)

        # 寻找在 target_time 之前最近的、可以构成一个完整输入窗口 (T帧) 的结束时间点
        # 候选的输入窗口结束时间索引为 `end_idx`
        # 对应的输入窗口时间索引范围是 `[end_idx - T + 1, end_idx]`

        best_end_idx = -1
        # 我们从 `target_time` 往前找
        for i in range(len(all_times_dt) - 1, -1, -1):
            if all_times_dt[i] < target_time_dt:
                # 找到了第一个在目标时间之前的点，检查它是否能作为一个窗口的结束点
                if i - ds.T + 1 >= 0:
                    best_end_idx = i
                    break

        if best_end_idx == -1:
            raise SystemExit(f"错误: 在数据集中找不到任何可以在 {args.predict_time_point} 之前形成完整输入窗口的数据。")

        # 计算预测步长 (horizon)
        time_step = all_times_dt[1] - all_times_dt[0] # 假设时间步长是均匀的
        horizon = round((target_time_dt - all_times_dt[best_end_idx]) / time_step)

        if horizon <= 0:
            print(f"警告: 目标时间 {args.predict_time_point} 与最近的输入数据结束时间 {all_times_dt[best_end_idx]} 过于接近或已过去。")
            print("将预测紧邻的下一帧 (步长=1)。")
            horizon = 1
            # 修正目标时间为预测的实际时间
            target_time_dt = all_times_dt[best_end_idx] + time_step
            args.predict_time_point = str(target_time_dt)

        print(f"已选择输入数据的结束时间点: {all_times_dt[best_end_idx]}")
        print(f"目标预测时间点: {args.predict_time_point}")
        print(f"自动计算的预测步长 (Horizon): {horizon}")

        # 找到这个输入窗口对应的样本索引
        # ds.indices 存储的是每个样本的起始时间在 all_times 中的索引
        start_idx_needed = best_end_idx - ds.T + 1
        try:
            sample_global_idx = list(ds.indices).index(start_idx_needed)
        except (ValueError, AttributeError):
            raise SystemExit(f"内部错误: 无法在数据集中定位到起始时间索引为 {start_idx_needed} 的样本。")

        # 获取初始输入数据
        x_single, _ = ds[sample_global_idx]
        x_in = x_single.unsqueeze(0).to(device) # (1, T, C, H, W)

        # --- 进行自回归预测 ---
        y_pred = predict_autoregressive(model, x_in, horizon, device) # (1, C, H, W)

        # --- 寻找真实值 y_true (如果存在) ---
        y_true = None
        try:
            # 在时间轴上找到与目标时间最匹配的索引
            true_time_idx = np.abs(all_times_dt - target_time_dt).argmin()
            # 如果时间完全匹配
            if all_times_dt[true_time_idx] == target_time_dt:
                # 从原始 xarray 数据中提取该时间点的数据
                y_true_da = ds.da.sel({ds.time_dim: all_times[true_time_idx]})
                # 转换成 torch tensor 并处理
                y_true_np = y_true_da.transpose('channel', 'latitude', 'longitude').values
                y_true = torch.from_numpy(y_true_np).float().to(device) # (C, H, W)
                print(f"已找到目标时间的真实数据: {all_times[true_time_idx]}")
            else:
                print(f"警告: 数据集中未找到与目标时间 {args.predict_time_point} 完全匹配的真实数据。最近的时间点是 {all_times_dt[true_time_idx]}。")
        except Exception as e:
            print(f"警告: 寻找真实数据时出错: {e}。将不进行真实值对比。")

        # 如果没有找到真实值，创建一个全零的占位符
        if y_true is None:
            y_true = torch.zeros_like(y_pred.squeeze(0))
            print("将使用全零张量作为真实值占位符进行可视化。")

        # 提取时间元数据
        input_times = [str(t) for t in all_times[start_idx_needed : best_end_idx + 1]]
        target_time = args.predict_time_point

        # --- 更新输出路径 ---
        # 所有输出都将使用 run_output_dir 和标准化的文件名
        prediction_nc_path = run_output_dir / 'prediction.nc'
        viz_path = run_output_dir / 'comparison.png'
        # --- 更新结束 ---

        # 保存 NetCDF（如果指定）
        # 注意：我们现在总是保存预测，因为路径是自动管理的
        save_preds_nc(prediction_nc_path, y_true.unsqueeze(0), y_pred.unsqueeze(0), ds)
        print(f"预测结果已保存至：{prediction_nc_path}")

        # 自动可视化（默认开启）
        if args.visualize:
            print("正在生成预测对比图...")
            meta = {
                'run_name': args.run_name,
                'sample_index': sample_global_idx,
                'input_times': input_times,
                'target_time': target_time,
                'horizon': horizon,
            }
            # y_true 需要和 y_pred 维度匹配 (B, C, H, W)
            visualize_predictions(y_true.unsqueeze(0), y_pred.unsqueeze(0), ds, save_path=viz_path, meta=meta)


if __name__ == '__main__':
    main()
