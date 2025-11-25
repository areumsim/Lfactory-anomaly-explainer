"""FeatureBank (initial): summary time-domain features.

Research context:
- Minimal, dependency-free feature extraction to support early ablations.
- Future extensions: FFT band powers, wavelet stats, periodicity, etc.

Exports:
- compute_basic(series: List[float]) -> Dict[str, float]
"""
from __future__ import annotations

from typing import Dict, List
import math


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    arr = sorted(values)
    pos = q * (len(arr) - 1)
    i = int(pos)
    if i >= len(arr) - 1:
        return arr[-1]
    frac = pos - i
    return arr[i] * (1 - frac) + arr[i + 1] * frac


def compute_basic(series: List[float]) -> Dict[str, float]:
    n = float(len(series))
    if n == 0:
        return {"count": 0.0}
    s = sum(series)
    mu = s / n
    # variance
    var = sum((x - mu) * (x - mu) for x in series) / n
    std = math.sqrt(var)
    vmin = min(series)
    vmax = max(series)
    p25 = _quantile(series, 0.25)
    p50 = _quantile(series, 0.50)
    p75 = _quantile(series, 0.75)
    # lag-1 autocorr (biased)
    if len(series) >= 2:
        num = sum((series[t] - mu) * (series[t - 1] - mu) for t in range(1, len(series))) / (len(series) - 1)
        den = var if var > 0 else 1.0
        ac1 = num / den
    else:
        ac1 = 0.0
    # mean absolute diff and energy
    madiff = sum(abs(series[t] - series[t - 1]) for t in range(1, len(series))) / (len(series) - 1)
    energy = sum(x * x for x in series) / n
    iqr = p75 - p25
    return {
        "count": float(len(series)),
        "mean": mu,
        "std": std,
        "min": vmin,
        "p25": p25,
        "median": p50,
        "p75": p75,
        "max": vmax,
        "iqr": iqr,
        "lag1_autocorr": ac1,
        "mean_abs_diff": madiff,
        "energy": energy,
    }

