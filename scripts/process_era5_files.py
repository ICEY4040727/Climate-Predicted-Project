"""
小工具，用于扫描目录中的 ERA5 .nc 文件，检测文件是否为 ZIP 归档，
并安全解压（防止路径穿越）。跳过部分下载的文件（例如 *.part）。

用法（Windows cmd）：
    python process_era5_files.py            # 扫描当前目录
    python process_era5_files.py <dir>     # 扫描指定目录
    python process_era5_files.py -n        # 演练模式（不进行解压）

脚本将会：
 - 跳过以 `.part` 结尾的文件
 - 检查文件魔术位和 zipfile.is_zipfile()
 - 安全解压到同一目录；如果归档仅包含一个 .nc 文件，
   它将替换被压缩的文件（可选）或并排写入。

"""
from __future__ import annotations
import os
import sys
import zipfile
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def is_zip_like(path: Path) -> bool:
    """简单检查文件是否为 ZIP（基于文件头和 zipfile 模块）"""
    try:
        with path.open('rb') as f:
            header = f.read(4)
            if len(header) < 4:
                return False
            if header[:2] == b'PK':
                return True
    except Exception:
        return False
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def safe_extract(zipf: zipfile.ZipFile, target_dir: Path) -> None:
    """安全解压，防止路径穿越"""
    target_dir = target_dir.resolve()
    for member in zipf.namelist():
        member_path = (target_dir / member).resolve()
        if not str(member_path).startswith(str(target_dir) + os.sep):
            raise RuntimeError(f"检测到不安全的归档成员路径: {member}")
    zipf.extractall(path=target_dir)


def unzip_era5_file(path: Path, dry_run: bool = False) -> None:
    """解压 ERA5 归档：若仅含单个 .nc 则移动到同目录，否则解压到 _extracted 文件夹"""
    logger.info(f"检查归档: {path}")
    try:
        if not zipfile.is_zipfile(path):
            logger.info(f"zipfile 判定非 ZIP: {path}")
            return
        with zipfile.ZipFile(path, 'r') as zf:
            members = zf.namelist()
            nc_members = [m for m in members if m.lower().endswith('.nc')]

            if len(nc_members) == 1:
                dest_file = path.with_suffix('')
                if not str(dest_file).lower().endswith('.nc'):
                    dest_file = dest_file.with_suffix('.nc')
                logger.info(f"归档包含单个 .nc: {nc_members[0]} -> {dest_file}")
                if dry_run:
                    logger.info("演练模式：不进行解压")
                    return
                tmp_dir = path.parent / (path.stem + '_tmp_extract')
                tmp_dir.mkdir(exist_ok=True)
                try:
                    safe_extract(zf, tmp_dir)
                    extracted = tmp_dir / nc_members[0]
                    if not extracted.exists():
                        extracted = tmp_dir / Path(nc_members[0]).name
                    if extracted.exists():
                        extracted.replace(dest_file)
                        logger.info(f"已解压并移动到: {dest_file}")
                    else:
                        logger.warning(f"解压后未找到预期成员: {nc_members[0]}")
                finally:
                    try:
                        for root, dirs, files in os.walk(tmp_dir, topdown=False):
                            for name in files:
                                (Path(root) / name).unlink()
                            for name in dirs:
                                (Path(root) / name).rmdir()
                        if tmp_dir.exists():
                            tmp_dir.rmdir()
                    except Exception:
                        pass
            else:
                out_dir = path.parent / (path.stem + '_extracted')
                logger.info(f"归档包含 {len(members)} 个成员；解压到: {out_dir}")
                if dry_run:
                    logger.info("演练模式：不进行解压")
                    return
                out_dir.mkdir(exist_ok=True)
                safe_extract(zf, out_dir)
                logger.info(f"解压完成: {out_dir}")
    except Exception as exc:
        logger.error(f"解压失败 {path}: {exc}")


def process_directory(directory: Path, dry_run: bool = False) -> None:
    """递归扫描目录，检测并解压伪 .nc ZIP 文件"""
    logger.info(f"扫描目录: {directory}")
    era5_files = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith('.part'):
                logger.debug(f"跳过部分下载文件: {name}")
                continue
            if name.lower().endswith('.nc'):
                era5_files.append(Path(root) / name)

    if not era5_files:
        logger.info("未找到 .nc 文件。")
        return

    logger.info(f"找到 {len(era5_files)} 个 .nc 文件；检查哪些是 ZIP ...")
    for file in era5_files:
        try:
            if is_zip_like(file):
                logger.info(f"\n{file} 看起来像 ZIP 文件，开始解压...")
                unzip_era5_file(file, dry_run=dry_run)
            else:
                logger.info(f"\n{file} 不是 ZIP 文件，跳过...")
        except Exception as e:
            logger.error(f"处理文件时出错: {file}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="扫描并处理 ERA5 .nc 文件，解压归档")
    parser.add_argument('directory', nargs='?', default='../data', help='要扫描的目录（默认: ../data）')
    parser.add_argument('-n', '--dry-run', action='store_true', help='仅演练，不实际解压')
    args = parser.parse_args(argv)

    process_directory(Path(args.directory), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
