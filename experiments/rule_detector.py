"""Rule-based anomaly detector.

Research context:
- Provides a transparent baseline (rolling z-score/robust z-score) for
  anomaly detection, useful to quantify Alarm Burden and serve as a
  component in the ML+Rule hybrid design.

Exports:
- zscore_detect(series, window, threshold, min_std)
- robust_zscore_detect(series, window, threshold, min_mad)
  -> dict with keys: scores, preds, params
"""
from __future__ import annotations

from typing import Dict, List


def _rolling_mean_std(series: List[float], window: int, min_std: float) -> (List[float], List[float]):
    n = len(series)
    means = [0.0] * n
    stds = [min_std] * n
    s = 0.0
    ss = 0.0
    w = window
    for i, x in enumerate(series):
        s += x
        ss += x * x
        if i >= w:
            x_old = series[i - w]
            s -= x_old
            ss -= x_old * x_old
        count = min(i + 1, w)
        mu = s / count
        var = max(ss / count - mu * mu, 0.0)
        means[i] = mu
        stds[i] = max(var ** 0.5, min_std)
    return means, stds


def zscore_detect(series: List[float], window: int = 50, threshold: float = 3.0, min_std: float = 1e-3) -> Dict:
    """Detect anomalies using rolling z-score.

    A point is anomalous if |x - mean(window)| / std(window) > threshold.
    Returns a dict suitable for logging and evaluation.
    """
    means, stds = _rolling_mean_std(series, window, min_std)
    scores: List[float] = []
    preds: List[int] = []
    for x, mu, sd in zip(series, means, stds):
        z = abs(x - mu) / sd if sd > 0 else 0.0
        scores.append(z)
        preds.append(1 if z >= threshold else 0)

    return {
        "scores": scores,
        "preds": preds,
        "params": {
            "method": "rolling_zscore",
            "window": window,
            "threshold": threshold,
            "min_std": min_std,
        },
    }


def _median(arr: List[float]) -> float:
    if not arr:
        return 0.0
    b = sorted(arr)
    n = len(b)
    m = n // 2
    if n % 2 == 1:
        return b[m]
    return 0.5 * (b[m - 1] + b[m])


def robust_zscore_detect(series: List[float], window: int = 51, threshold: float = 3.5, min_mad: float = 1e-6) -> Dict:
    """Detect anomalies using rolling robust z-score based on median/MAD.

    z = |x - median| / (1.4826 * MAD)
    Uses a naive O(w log w) per step median for clarity (sufficient for
    smoke-scale runs). "1.4826" scales MAD to match std for normal data.
    """
    n = len(series)
    w = max(3, int(window))
    scores: List[float] = []
    preds: List[int] = []
    for i in range(n):
        a = max(0, i - w + 1)
        win = series[a : i + 1]
        med = _median(win)
        absdev = [abs(x - med) for x in win]
        mad = _median(absdev)
        denom = max(1.4826 * mad, min_mad)
        z = abs(series[i] - med) / denom
        scores.append(z)
        preds.append(1 if z >= threshold else 0)
    return {
        "scores": scores,
        "preds": preds,
        "params": {
            "method": "rolling_robust_zscore",
            "window": w,
            "threshold": threshold,
            "min_mad": min_mad,
        },
    }
