import cdsapi
import concurrent.futures
import time
import random
import os
import sys
import argparse
from pathlib import Path
import ssl
import urllib3
import warnings
import calendar
import xarray as xr



# 禁用 SSL 警告
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
# 配置 urllib3 不显示 InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 自定义 SSL 上下文，增加超时和重试
class CustomHTTPSConnectionPool(urllib3.HTTPSConnectionPool):
    def __init__(self, *args, **kwargs):
        kwargs['timeout'] = urllib3.Timeout(connect=60.0, read=300.0)
        kwargs['maxsize'] = 1
        kwargs['block'] = True
        super().__init__(*args, **kwargs)

# 替换默认的 HTTPS 连接池
urllib3.poolmanager.pool_classes_by_scheme['https'] = CustomHTTPSConnectionPool

cds_client = cdsapi.Client()
```python
def download_month(year, month, dataset):
    """下载单个月的数据；如果本地已存在且看起来完整则跳过。

    返回 'ok'/'skipped'/'failed'
    """
    filename = f"era5_{year}{month}.nc"
    filepath = Path.cwd() / filename
    temp_path = Path.cwd() / (filename + ".part")

    def days_in_month(year_int, month_int):
        return calendar.monthrange(year_int, month_int)[1]

    def check_file_variables(filepath):
        try:
            ds = xr.open_dataset(str(filepath))
            data_vars = list(ds.data_vars.keys())
            all_vars = list(ds.variables.keys())
            print(f"{filepath.name} 包含 data_vars: {data_vars}")
            print(f"{filepath.name} 包含 variables: {all_vars}")
            needed = {'sea_surface_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind'}
            missing = needed - set(data_vars) - set(all_vars)
            if missing:
                print(f"警告：缺少变量 {missing}，请确认 dataset/variable 名称是否正确或请求是否被 CDS 忽略。")
            ds.close()
        except Exception as e:
            print(f"无法打开或检查文件 {filepath}: {e}")

    try:
        if filepath.exists() and filepath.stat().st_size > 1_000_000:
            print(f"{filename} 已存在（{filepath.stat().st_size} bytes），跳过下载。")
            return 'skipped'
    except Exception:
        pass

    days = [str(d).zfill(2) for d in range(1, days_in_month(int(year), int(month)) + 1)]

    # 构建请求
    request = {
        'format': 'netcdf',
        'product_type': 'reanalysis',
        'variable': [
            'mean_wave_direction',
            'significant_height_of_combined_wind_waves_and_swell',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'sea_surface_temperature',
        ],
        'year': year,
        'month': month,
        'day': days,
        'time': [f"{h:02d}:00" for h in range(24)],
    }

    try:
        # 如果存在遗留的临时文件，先删除
        try:
            if temp_path.exists():
                print(f"发现遗留临时文件 {temp_path.name}，将删除后重新下载。")
                temp_path.unlink()
        except Exception:
            pass

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"尝试下载 {year} 年 {month} 月数据 (第{attempt+1}次尝试)...")
                # 使用模块级 cds_client（在文件顶部已创建）
                cds_client.retrieve(dataset, request).download(str(temp_path))
                # 下载成功则跳出重试循环
                break
            except Exception as e:
                msg = str(e)
                print(f"下载尝试失败：{msg}")
                if any(k in msg for k in ("SSLError", "EOF occurred", "timed out", "ConnectionError")) and attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)
                    print(f"网络相关错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("已达到最大重试次数或非重试错误，放弃下载")
                    raise

        # 确认临时文件存在后再写最终文件
        if not temp_path.exists():
            print(f"下载未生成临时文件：{temp_path}")
            return 'failed'

        try:
            temp_path.rename(filepath)
        except Exception:
            try:
                with temp_path.open('rb') as r, filepath.open('wb') as w:
                    w.write(r.read())
                temp_path.unlink()
            except Exception:
                print(f"无法将临时文件重命名为最终文件，请检查磁盘和权限：{temp_path}")
                return 'failed'

        print(f"{year}年{month}月数据下载完成！")
        check_file_variables(filepath)
        return 'ok'
    except Exception as e:
        print(f"下载 {year}年{month}月数据失败：{str(e)}")
        return 'failed'



def main():
    parser = argparse.ArgumentParser(description="批量下载 ERA5 月度数据")
    parser.add_argument('--year', type=str, help='要下载的年份（例如 2024）')
    parser.add_argument('--years', type=str, help='年份范围（例如 2020-2024）')
    parser.add_argument('--months', type=str, default='01,02,03,04,05,06,07,08,09,10,11,12', help='逗号分隔的月份列表（例如 01,02,03）')
    parser.add_argument('--dataset', type=str, default='reanalysis-era5-single-levels', help='CDS 数据集名称')
    parser.add_argument('--batch-size', type=int, default=3, help='每批并行下载的月份数')
    parser.add_argument('--delay-min', type=int, default=60, help='批次间最短延迟（秒）')
    parser.add_argument('--delay-max', type=int, default=180, help='批次间最长延迟（秒）')
    args = parser.parse_args()

    if args.years:
        try:
            start_year, end_year = map(int, args.years.split('-'))
            years = list(range(start_year, end_year + 1))
        except ValueError:
            print("错误：年份范围格式不正确，请使用 'YYYY-YYYY' 格式，例如 '2020-2024'")
            sys.exit(1)
    elif args.year:
        try:
            years = [int(args.year)]
        except ValueError:
            print("错误：年份格式不正确，请使用 'YYYY' 格式，例如 '2024'")
            sys.exit(1)
    else:
        import datetime
        years = [datetime.datetime.now().year]
        print(f"未指定年份，默认下载 {years[0]} 年的数据")

    months = [m.strip() for m in args.months.split(',') if m.strip()]
    dataset = args.dataset

    lockfile = Path.cwd() / ".download_era5.lock"
    if lockfile.exists():
        try:
            pid = int(lockfile.read_text())
            print(f"检测到锁文件，可能已有另一个实例在运行（pid={pid}）。退出以避免并发下载。")
        except Exception:
            print("检测到锁文件，可能已有另一个实例在运行。退出以避免并发下载。")
        sys.exit(0)

    try:
        lockfile.write_text(str(os.getpid()))
    except Exception:
        print("无法创建锁文件，但将继续运行（注意并发风险）。")

    try:
        for year in years:
            print(f"\n{'='*50}")
            print(f"开始下载 {year} 年的数据")
            print(f"{'='*50}\n")

            batch_size = max(1, args.batch_size)
            month_batches = [months[i:i + batch_size] for i in range(0, len(months), batch_size)]

            for i, batch in enumerate(month_batches, start=1):
                print(f"开始批次 {i}/{len(month_batches)}: {batch}")

                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    future_to_month = {executor.submit(download_month, str(year), month, dataset): month for month in batch}

                    statuses = []
                    for future in concurrent.futures.as_completed(future_to_month):
                        month = future_to_month[future]
                        try:
                            status = future.result()
                            statuses.append(status)
                            if status == 'failed':
                                print(f"{year}年{month}月：下载失败（见上方错误信息），将继续后续批次。")
                            elif status == 'skipped':
                                print(f"{year}年{month}月：已跳过（本地已有文件）。")
                            else:
                                print(f"{year}年{month}月：下载并保存成功。")
                        except Exception as e:
                            statuses.append('failed')
                            print(f"{year}年{month}月：下载任务抛出异常：{e}")

                did_action = any(s != 'skipped' for s in statuses)
                if i != len(month_batches):
                    if not did_action:
                        print("本批次所有月份均已存在，跳过等待，继续下一批次...\n")
                    else:
                        delay = random.randint(args.delay_min, args.delay_max)
                        print(f"\n等待 {delay} 秒后继续下一批次下载...\n")
                        time.sleep(delay)

            print(f"\n{year} 年数据下载完成！\n")

    except KeyboardInterrupt:
        print("用户中断：收到 Ctrl+C，正在安全退出...")
    finally:
        try:
            if lockfile.exists():
                lockfile.unlink()
        except Exception:
            pass


if __name__ == '__main__':
    # 使用示例：
    # 1. 下载2024年所有月份数据：
    #    uv run download_era5_data.py --year 2024
    #
    # 2. 下载2020-2024年所有月份数据：
    #    uv run download_era5_data.py --years 2020-2024
    #
    # 3. 下载2024年1-6月数据：
    #    uv run download_era5_data.py --year 2024 --months 01,02,03,04,05,06
    #
    # 4. 下载2020-2022年夏季数据（6-8月）：
    #    uv run download_era5_data.py --years 2020-2022 --months 06,07,08

    main()