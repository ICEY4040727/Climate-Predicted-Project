import argparse
from pathlib import Path
import xarray as xr
import torch
import sys
import os

# 将项目根目录添加到 Python 路径，以便导入 src 模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    # 尝试从训练脚本中导入可视化函数
    from scripts.train_eval import visualize_predictions
except ImportError as e:
    print(f"无法从 'scripts.train_eval' 导入可视化函数: {e}")
    print("请确保你在项目根目录下，并且 'scripts/train_eval.py' 文件存在。")
    sys.exit(1)

def main():
    """
    主函数：加载 NetCDF 预测结果并进行可视化。
    """
    parser = argparse.ArgumentParser(description="从 NetCDF 文件可视化预测结果")
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help="要可视化的 prediction.nc 文件路径"
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='local_comparison.png',
        help="输出的可视化图像文件名"
    )
    args = parser.parse_args()

    nc_file = Path(args.file)
    if not nc_file.exists():
        print(f"错误：文件不存在 -> {nc_file}")
        sys.exit(1)

    print(f"正在加载文件: {nc_file}")

    # 使用 xarray 加载 NetCDF 文件
    try:
        ds = xr.open_dataset(nc_file)
    except Exception as e:
        print(f"使用 xarray 打开文件时出错: {e}")
        sys.exit(1)

    print("\n--- NetCDF 文件信息 ---")
    print(ds)
    print("\n--- 变量详细信息 ---")
    for var_name in ds.data_vars:
        print(f"\n变量: {var_name}")
        print(f"  维度: {ds[var_name].dims}")
        print(f"  形状: {ds[var_name].shape}")
        # 打印一些统计数据来检查内容
        try:
            print(f"  最小值: {ds[var_name].min().item():.4f}")
            print(f"  最大值: {ds[var_name].max().item():.4f}")
            print(f"  平均值: {ds[var_name].mean().item():.4f}")
            # 检查是否存在 NaN
            if ds[var_name].isnull().any():
                print("  警告: 此变量包含 NaN 值！")
        except Exception as e:
            print(f"  无法计算统计信息: {e}")
    print("\n-----------------------\n")


    # --- 准备调用 visualize_predictions ---
    # 该函数需要 torch.Tensor 和一个模拟的 dataset 对象

    if 'target' not in ds or 'prediction' not in ds:
        print("错误: NetCDF 文件中必须包含 'target' 和 'prediction' 两个变量。")
        sys.exit(1)

    # 1. 将 xarray.DataArray 转换为 torch.Tensor
    y_true_tensor = torch.from_numpy(ds['target'].values).float()
    y_pred_tensor = torch.from_numpy(ds['prediction'].values).float()

    # 2. 创建一个模拟的 dataset 对象，为可视化函数提供必要的元数据
    class MockDataset:
        def __init__(self, nc_dataset):
            self.vars = nc_dataset.coords.get('channel', []).values.tolist()
            self.da = nc_dataset['target'] # 用于获取坐标
            self._norm = (0, 1) # 假设数据已经反归一化，设置一个无效的 norm

            # 检查变量列表是否为空
            if not self.vars:
                print("警告: 无法从 NetCDF 文件中获取变量名 ('channel' 坐标)。")
                # 尝试从变量本身的属性中恢复
                if 'long_name' in ds['target'].attrs:
                     self.vars = [ds['target'].attrs['long_name']]
                else: # 如果还没有，就用占位符
                    self.vars = [f'var_{i}' for i in range(ds.sizes.get('channel', 1))]


    mock_ds = MockDataset(ds)

    print("准备调用可视化函数...")
    print(f"检测到的变量: {mock_ds.vars}")

    # 定义输出路径
    output_path = nc_file.parent / args.output_name

    # 调用可视化函数
    visualize_predictions(
        y_true=y_true_tensor,
        y_pred=y_pred_tensor,
        ds=mock_ds,
        save_path=output_path,
        meta={'source_file': str(nc_file)} # 传递一些元信息
    )

if __name__ == '__main__':
    main()

