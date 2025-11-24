import xarray as xr
import sys
from pathlib import Path

nc_path = Path('../data/era5_202204.nc')
if not nc_path.exists():
    print(f"文件不存在: {nc_path}，请检查 data 目录是否包含 era5_20220.nc")
    sys.exit(1)

# 打开 NetCDF 文件 并打印变量与维度信息
ds = xr.open_dataset(str(nc_path))
print("变量列表:")
for var in ds.data_vars:
    print(f"  {var}: {ds[var].sizes} {ds[var].shape}")
print("\n维度信息:")
for dim in ds.dims:
    print(f"  {dim}: {ds.sizes[dim]}")