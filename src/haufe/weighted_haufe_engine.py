"""Calculate inverse-scan-count weighted covariances and correlations."""

import numpy as np
import pandas as pd


def child_weights(child):
    s = pd.Series(child).astype(str)
    counts = s.value_counts()
    w = s.map(lambda x: 1.0 / counts[x]).to_numpy(float)
    return w / w.sum()


def wcov(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, ww = (x[mask], y[mask], np.asarray(w, float)[mask])
    ww = ww / ww.sum()
    return float(np.sum(ww * (x - np.sum(ww * x)) * (y - np.sum(ww * y))))


def wcorr(x, y, w):
    c = wcov(x, y, w)
    vx = wcov(x, x, w)
    vy = wcov(y, y, w)
    return c / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
