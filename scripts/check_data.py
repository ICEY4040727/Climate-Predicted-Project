#!/usr/bin/env python3
"""
数据文件检查脚本
用于验证数据目录中的NetCDF文件是否符合要求
"""

import os
import sys
from pathlib import Path

def check_data_files(data_dir="../data"):
    """检查数据目录中的文件"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"错误: 数据目录 '{data_path}' 不存在")
        return False
    
    # 查找所有.nc文件
    nc_files = list(data_path.glob("*.nc"))
    
    if not nc_files:
        print(f"错误: 在 '{data_path}' 目录中未找到任何.nc文件")
        return False
    
    print(f"在 '{data_path}' 目录中找到 {len(nc_files)} 个.nc文件:")
    for i, file in enumerate(sorted(nc_files), 1):
        size = file.stat().st_size
        print(f"  {i:2d}. {file.name} ({size:,} bytes)")
    
    # 检查文件命名模式
    era5_files = [f for f in nc_files if f.name.startswith("era5_") and f.name.endswith(".nc")]
    if not era5_files:
        print("\n警告: 未找到符合 'era5_*.nc' 命名模式的文件")
        print("建议将文件重命名为 'era5_YYYYMM.nc' 格式，例如 'era5_202401.nc'")
    else:
        print(f"\n找到 {len(era5_files)} 个符合 'era5_*.nc' 命名模式的文件")
    
    return True

def main():
    """主函数"""
    data_dir = "../data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    print("=" * 50)
    print("气候数据文件检查工具")
    print("=" * 50)
    
    success = check_data_files(data_dir)
    
    print("\n" + "=" * 50)
    if success:
        print("检查完成: 数据文件准备就绪")
    else:
        print("检查失败: 请检查上述错误信息")
    print("=" * 50)

if __name__ == "__main__":
    main()