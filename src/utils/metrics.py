import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rmse(y_true, y_pred, mask=None, eps: float = 1e-8):
    yt = to_numpy(y_true)
    yp = to_numpy(y_pred)
    diff = yp - yt
    if mask is not None:
        m = to_numpy(mask).astype(bool)
        diff = np.where(m, diff, 0.0)
        denom = m.sum()
    else:
        denom = diff.size
    denom = max(denom, 1)
    return float(np.sqrt((diff ** 2).sum() / (denom + eps)))


def mae(y_true, y_pred, mask=None, eps: float = 1e-8):
    yt = to_numpy(y_true)
    yp = to_numpy(y_pred)
    diff = np.abs(yp - yt)
    if mask is not None:
        m = to_numpy(mask).astype(bool)
        diff = np.where(m, diff, 0.0)
        denom = m.sum()
    else:
        denom = diff.size
    denom = max(denom, 1)
    return float(diff.sum() / (denom + eps))

