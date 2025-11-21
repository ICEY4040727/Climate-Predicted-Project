#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试脚本：验证环境配置"""

print("=" * 60)
print("Testing Python Environment...")
print("=" * 60)

try:
    import sys
    print(f"✓ Python: {sys.version}")
    print(f"  Executable: {sys.executable}")
except Exception as e:
    print(f"✗ Python import failed: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import xarray as xr
    print(f"✓ xarray: {xr.__version__}")
except Exception as e:
    print(f"✗ xarray import failed: {e}")

try:
    import netCDF4
    print(f"✓ netCDF4: {netCDF4.__version__}")
except Exception as e:
    print(f"✗ netCDF4 import failed: {e}")

try:
    import matplotlib
    print(f"✓ matplotlib: {matplotlib.__version__}")
except Exception as e:
    print(f"✗ matplotlib import failed: {e}")

print("=" * 60)
print("Environment test completed!")
print("=" * 60)

