import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


class ERA5TensorDataset(Dataset):
    """
    加载ERA5月度NetCDF文件作为滑动窗口张量数据集

    - Files: a list of paths to .nc files that contain the same variables, grid and time axis.
    Files：包含相同变量、网格和时间轴的.nc文件路径列表
    - vars: variable names to extract and stack as channels.
    vars：需提取并堆叠为通道的变量名称
    - time_window: length T of input sequence.
    time_window：输入序列的时间窗口长度T
    - stride: step between window starts.
    stride：窗口起始点之间的步长
    - region: optional dict with lat/lon min/max to crop.
    region：可选参数，用于裁剪的纬度/经度范围（格式：{"lat_min": x, "lat_max": y, "lon_min": a, "lon_max": b}）
    - normalize: per-variable mean/std normalization computed from the dataset (lazy on first epoch).
    normalize：是否进行基于数据集计算的每变量均值/标准差归一化（在第一次使用时计算）
    - fillna: optional NA handling: one of {None, 'ffill', 'bfill', 'linear', 'nearest'} applied along time (ffill/bfill) or spatiotemporal interpolate (linear/nearest).
    fillna：可选的缺失值处理方法：{None, 'ffill', 'bfill', 'linear', 'nearest'}之一，分别沿时间（ffill/bfill）或时空插值（linear/nearest）应用
    - time_slice: optional time range to filter data, format: "start_time:end_time"
    time_slice：可选的时间范围过滤，格式："start_time:end_time"
    - time_slices: optional list of time ranges to filter data, format: ["start1:end1", "start2:end2"]
    time_slices：可选的时间范围列表过滤，格式：["start1:end1", "start2:end2"]
    - time_points: optional list of specific time points to use, format: "time1,time2,time3"
    time_points：可选的特定时间点列表，格式："time1,time2,time3"


    Returns samples of shape (T, C, H, W) with target being next frame y_{T} (same channels).
    返回形状为 (T, C, H, W) 的样本，目标为下一个时间点 y_{T}（相同通道）
    """

    def __init__(
        self,
        files: List[os.PathLike],
        vars: List[str],
        time_window: int = 4,
        stride: int = 1,
        region: Optional[dict] = None,
        normalize: bool = True,
        fillna: Optional[str] = None,
        time_slices: Optional[List[str]] = None,
        time_points: Optional[str] = None,
    ):
        self.files = [Path(f) for f in files]
        self.vars = list(vars)
        self.T = int(time_window)
        self.stride = int(stride)
        self.region = region
        self.normalize = normalize
        self.fillna = fillna
        self.time_slices = time_slices
        self.time_points = time_points
        self._norm = None  # (mean, std) per-channel

        # --- 新增逻辑：预先检查文件并筛选 ---
        print(f"开始检查 {len(self.files)} 个文件是否包含所需变量: {self.vars}")
        files_with_required_vars = []
        for file_path in self.files:
            try:
                #进打开元数据以检查变量，不深拷贝加载数据到内存
                with xr.open_dataset(str(file_path), engine='netcdf4') as ds:
                    missing_vars = [v for v in self.vars if v not in ds.data_vars]
                    if not missing_vars:
                        files_with_required_vars.append(str(file_path))
                    else:
                        print(f"  - 警告: 文件 '{file_path}' 因缺少变量 {missing_vars} 而被跳过。")
                        print(f"    -> 文件 '{file_path}' 中可用的变量有: {list(ds.data_vars)}")
            except Exception as e:
                print(f"  - 错误: 无法处理文件 '{file_path}'，错误: {e}")

        if not files_with_required_vars:
            raise ValueError(f"错误: 没有任何文件包含所有必需的变量 {self.vars}。无法创建数据集。")

        print(f"\n成功加载 {len(files_with_required_vars)} 个包含所有必需变量的文件。")
        print("使用的文件列表:")
        for f_path in files_with_required_vars:
            print(f"  - {f_path}")
        # --- 检查逻辑结束 ---

        # 使用筛选后的数据集列表进行拼接
        def _preprocess(ds0):
            # 使用闭包中的 self.vars 和 self.region，避免在 preprocess 中捕获外部可变参数问题
            ds = ds0[self.vars] if self.vars is not None else ds0
            if self.region:
                lat_min = self.region.get('lat_min', None)
                lat_max = self.region.get('lat_max', None)
                lon_min = self.region.get('lon_min', None)
                lon_max = self.region.get('lon_max', None)
                try:
                    if lat_min is not None and lat_max is not None:
                        ds = ds.sel(lat=slice(lat_min, lat_max))
                except Exception:
                    # 有的文件使用 'latitude' 维/坐标名
                    try:
                        if lat_min is not None and lat_max is not None:
                            ds = ds.sel(latitude=slice(lat_max, lat_min))
                    except Exception:
                        pass
                try:
                    if lon_min is not None and lon_max is not None:
                        ds = ds.sel(lon=slice(lon_min, lon_max))
                except Exception:
                    try:
                        if lon_min is not None and lon_max is not None:
                            ds = ds.sel(longitude=slice(lon_min, lon_max))
                    except Exception:
                        pass
            return ds

        # files: 列表，vars: 列表，region: dict
        # ds = xr.open_mfdataset(
        #     files,
        #     combine='by_coords',
        #     preprocess=lambda ds0: _preprocess(ds0, vars, region),
        #     chunks={'time': 1},  # 按时间分块，使用 dask 延迟计算
        #     parallel=False  # 如安装 dask/distributed 并希望并行置为 True
        # )

        if len(files_with_required_vars) == 1:
            # 单文件也使用 chunks 返回 dask-backed arrays
            self.ds = xr.open_dataset(files_with_required_vars[0], engine='netcdf4', chunks={'time': 1})
            # 可选地只保留 vars 与裁剪 region（更安全）
            try:
                self.ds = _preprocess(self.ds)
            except Exception:
                pass
        else:
            # 多文件使用 open_mfdataset 且指定 preprocess 和 chunks（延迟加载）
            self.ds = xr.open_mfdataset(
                files_with_required_vars,
                combine='by_coords',
                preprocess=_preprocess,
                chunks={'time': 1},  # 按时间分块，使用 dask 延迟计算
                parallel=False  # 如安装 dask/distributed 并希望并行置为 True
            )

        # --- 时间过滤逻辑 ---
        # 优先使用 time_slices, 其次是 time_points
        if self.time_slices:
            print(f"应用多个时间切片: {self.time_slices}")
            datasets_to_concat = []
            for ts in self.time_slices:
                try:
                    start, end = ts.split(':')
                    datasets_to_concat.append(self.ds.sel({time_dim: slice(start, end)}))
                except Exception as e:
                    print(f"警告: 解析或应用时间切片 '{ts}' 失败: {e}")
            if datasets_to_concat:
                self.ds = xr.concat(datasets_to_concat, dim=time_dim)
            else:
                print("警告: 所有时间切片均无效，将使用全部数据。")
        elif self.time_points:
            print(f"选择特定时间点...")
            points = [p.strip() for p in self.time_points.split(',')]
            try:
                self.ds = self.ds.sel({time_dim: points}, method='nearest')
            except Exception as e:
                print(f"警告: 选择特定时间点失败: {e}。将使用全部数据。")
        # --- 时间过滤逻辑结束 ---

        time_dim = None
        candidates = ['time', 'valid_time', 't', 'date', 'datetime']
        for cand in candidates:
            if cand in self.ds.dims or cand in self.ds.coords:
                time_dim = cand
                break
        # 额外尝试：查找 dtype 为 datetime 的一维坐标
        if time_dim is None:
            for coord in self.ds.coords:
                try:
                    dtype = self.ds[coord].dtype
                    if np.issubdtype(dtype, np.datetime64):
                        time_dim = coord
                        break
                except Exception:
                    continue

        if time_dim is None:
            raise KeyError(f"无法在数据集中找到时间维度。dims={list(self.ds.dims)} coords={list(self.ds.coords)}")

        self.time_dim = time_dim
        print(f"合并后数据集的时间维: `{self.time_dim}` 大小: {self.ds.sizes.get(self.time_dim, '未知')}")

        # 2) 识别并标准化经纬度维名为 'latitude' 和 'longitude'
        lat_candidates = ['latitude', 'lat', 'y', 'nav_lat']
        lon_candidates = ['longitude', 'lon', 'x', 'nav_lon']

        lat_name = None
        lon_name = None

        for n in lat_candidates:
            if n in self.ds.dims or n in self.ds.coords:
                lat_name = n
                break
        for n in lon_candidates:
            if n in self.ds.dims or n in self.ds.coords:
                lon_name = n
                break

        # 如果没找到明确经纬度，尝试从变量维度中推断（第一个二维 coord）
        if lat_name is None or lon_name is None:
            for coord in self.ds.coords:
                if coord == self.time_dim:
                    continue
                size = self.ds[coord].sizes.get(coord, None)
                # 如果是 1D 坐标但不是时间且看起来像经度/纬度，作为兜底
                if size and size > 10 and (lat_name is None or lon_name is None):
                    if lat_name is None:
                        lat_name = coord
                    elif lon_name is None:
                        lon_name = coord

        # 最终检查
        if lat_name is None or lon_name is None:
            print(f"警告: 未能可靠识别经纬度坐标 (lat_name={lat_name}, lon_name={lon_name})，后续操作可能出错。")
        else:
            # 如果名称不是目标标准名，则重命名 dataset 中相关坐标/维名，便于后续统一处理
            rename_map = {}
            if lat_name != 'latitude':
                rename_map[lat_name] = 'latitude'
            if lon_name != 'longitude':
                rename_map[lon_name] = 'longitude'
            if rename_map:
                try:
                    self.ds = self.ds.rename(rename_map)
                    print(f"已重命名坐标: {rename_map}")
                except Exception as e:
                    print(f"警告: 重命名坐标失败: {e}")

        # 如果指定了 time_slice，则应用时间切片
        if self.time_slice:
            try:
                start_time, end_time = self.time_slice.split(":")
                if start_time and end_time:
                    self.ds = self.ds.sel({self.time_dim: slice(start_time, end_time)})
                elif start_time:
                    self.ds = self.ds.sel({self.time_dim: slice(start_time, None)})
                elif end_time:
                    self.ds = self.ds.sel({self.time_dim: slice(None, end_time)})
            except ValueError:
                # 如果分割失败，则将整个字符串视为单个时间点
                self.ds = self.ds.sel({self.time_dim: self.time_slice})

        # 如果指定了 time_points，则应用特定时间点选择
            # python
            # 如果指定了 time_points，则应用特定时间点选择
            if self.time_points:
                try:
                    tps_str = [t.strip() for t in self.time_points.split(",") if t.strip()]
                    print(f"尝试选择时间点: {tps_str}")

                    times_vals = self.ds[self.time_dim].values
                    matched_times = []

                    # 尝试将输入解析为 datetime64 对象进行矢量化匹配
                    try:
                        tps_dt = np.array(tps_str, dtype='datetime64[ns]')
                        # isin 在 xarray 中是高效的
                        matched_mask = self.ds[self.time_dim].isin(tps_dt)
                        self.ds = self.ds.sel({self.time_dim: matched_mask})

                        num_matched = self.ds.sizes.get(self.time_dim, 0)
                        if num_matched > 0:
                            print(f"通过精确时间匹配选中 {num_matched} 个时间点。")
                        if num_matched < len(tps_str):
                            print(f"警告: 部分时间点未在数据集中找到。")

                    except (ValueError, TypeError):
                        # 如果无法解析为 datetime，则回退到字符串部分匹配
                        print("警告: 无法将输入解析为 datetime，回退到字符串匹配。")
                        available_times_str = [str(t) for t in times_vals]

                        for tp_str in tps_str:
                            found = False
                            for i, at_str in enumerate(available_times_str):
                                if at_str.startswith(tp_str):
                                    matched_times.append(times_vals[i])
                                    found = True
                                    break  # 找到一个就够了
                            if not found:
                                print(f"警告: 未找到与 '{tp_str}' 匹配的时间点。")

                        # 去重并应用选择
                        unique_matched = sorted(list(set(matched_times)))
                        if unique_matched:
                            self.ds = self.ds.sel({self.time_dim: unique_matched})
                            print(f"通过字符串匹配选中 {len(unique_matched)} 个时间点。")
                        else:
                            print("警告: 字符串匹配未找到任何时间点，将使用所有可用时间。")

                except Exception as e:
                    print(f"警告: 选择特定时间点失败: {e}")



        # # 5) 保证空间裁剪使用当前 dataset 的经纬度名字（已重命名为 'latitude'/'longitude' 时生效）
        # if self.region:
        #     lat_min = self.region.get('lat_min', None)
        #     lat_max = self.region.get('lat_max', None)
        #     lon_min = self.region.get('lon_min', None)
        #     lon_max = self.region.get('lon_max', None)
        #     try:
        #         if lat_min is not None and lat_max is not None:
        #             self.ds = self.ds.sel(latitude=slice(lat_max, lat_min))
        #     except Exception:
        #         pass
        #     try:
        #         if lon_min is not None and lon_max is not None:
        #             self.ds = self.ds.sel(longitude=slice(lon_min, lon_max))
        #     except Exception:
        #         pass

        # 6) 堆叠变量为带 channel 维的 DataArray 并保证 dims 名称一致
        arrays = []
        for v in self.vars:
            if v not in self.ds:
                raise KeyError(f"变量 `{v}` 在数据集中未找到。可用变量: {list(self.ds.data_vars)}")
            arrays.append(self.ds[v])

        da = xr.concat(arrays, dim='channel')
        da = da.assign_coords(channel=('channel', self.vars))

        # 如果原始经纬度名不是 'latitude'/'longitude'，上面已尝试重命名；再一次确保存在这两个维
        if 'latitude' not in da.dims or 'longitude' not in da.dims:
            # 尝试查找可能的二维维并重命名为标准名（兜底）
            for coord in da.coords:
                if coord == self.time_dim or coord == 'channel':
                    continue
                # 若发现两个非时间的一维坐标则尝试映射
            # 如果仍然缺失，后续会在尺寸读取处抛错，提示用户检查文件

        # 7) 确保时间升序并保存到 self.da
        da = da.sortby(self.time_dim)
        self.da = da

        # 8) 缩小采样用于归一化计算（仍使用你原有 _compute_norm）
        self.num_times = int(self.da.sizes[self.time_dim])
        # 确保 channel/latitude/longitude 存在，否则抛出更友好的错误
        if 'channel' not in self.da.dims:
            raise KeyError("channel 维未找到，无法继续。")
        if 'latitude' not in self.da.dims or 'longitude' not in self.da.dims:
            raise KeyError(f"无法找到标准的空间维 (latitude/longitude)。当前 dims: {list(self.da.dims)}")

        self.num_channels = int(self.da.sizes['channel'])
        self.H = int(self.da.sizes['latitude'])
        self.W = int(self.da.sizes['longitude'])
        print(f"数据集维度 - 时间: {self.num_times}, 通道: {self.num_channels}, H: {self.H}, W: {self.W}")
        print(f"T (输入窗口大小): {self.T}, 步长: {self.stride}")

        # 9) 最少时间检查并生成索引
        min_required_times = self.T + 1
        if self.num_times < min_required_times:
            raise ValueError(
                f"数据集中时间点不足。对于 T={self.T}，至少需要 {min_required_times} 个时间点，但只有 {self.num_times} 个可用。")

        self.indices = list(range(0, self.num_times - (self.T + 1) + 1, self.stride))
        print(f"样本数量 (len(indices)): {len(self.indices)}")

        if len(self.indices) == 0:
            raise ValueError(f"时间点不足以创建样本。有 {self.num_times} 个时间点和 T={self.T}，无法创建有效的样本。")



    def __len__(self):
        return len(self.indices)

    def _compute_norm(self):
        # Compute per-channel mean/std over time and space using a subsample for speed (NaN-safe)
        x = self.da.isel({self.time_dim: slice(0, min(200, self.num_times))})  # subsample time
        x = x.transpose(self.time_dim, 'channel', 'latitude', 'longitude')
        arr = x.values  # (T, C, H, W)
        mean = np.nanmean(arr, axis=(0, 2, 3))
        std = np.nanstd(arr, axis=(0, 2, 3))
        # sanitize
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where(np.isnan(std), 1.0, std)
        std[std < 1e-6] = 1.0
        self._norm = (mean, std)

    def _normalize(self, arr: np.ndarray):
        if not self.normalize:
            return arr
        if self._norm is None:
            self._compute_norm()
        mean, std = self._norm
        out = (arr - mean[None, :, None, None]) / std[None, :, None, None]
        # ensure no NaNs/Infs propagate to the model
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t0 = self.indices[idx]
        t1 = t0 + self.T
        # x: [t0, t1), y: t1
        x = self.da.isel({self.time_dim: slice(t0, t1)}).transpose(self.time_dim, 'channel', 'latitude', 'longitude')
        y = self.da.isel({self.time_dim: t1}).transpose('channel', 'latitude', 'longitude')
        x_np = x.values.astype(np.float32)  # (T, C, H, W)
        y_np = y.values.astype(np.float32)  # (C, H, W) expected
        # Defensively squeeze any unintended leading singleton (time-like) dim
        if y_np.ndim == 4 and y_np.shape[0] == 1:
            y_np = y_np[0]
        x_np = self._normalize(x_np)
        y_np = self._normalize(y_np)
        # Final safety: replace any residual NaNs/Infs
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x_np), torch.from_numpy(y_np)


def find_nc_files(root: os.PathLike, pattern: str = 'era5_*.nc') -> List[Path]:
    root = Path(root)
    files = []
    
    # 支持逗号分隔的多个模式
    patterns = pattern.split(',')
    for pat in patterns:
        pat = pat.strip()
        files.extend(root.glob(pat))
    
    return sorted([p for p in files if p.is_file()])
