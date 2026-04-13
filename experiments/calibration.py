"""Calibration utilities: Platt scaling (logistic), ECE, and helpers.

Research context:
- Provides a light-weight, dependency-free way to calibrate anomaly
  scores into probabilities and to measure Expected Calibration Error.
  Platt scaling is fit with simple gradient descent for 1D scores.

API:
- normalize_scores(scores) -> List[float] in [0,1]
- fit_platt(scores, labels, lr, epochs, l2) -> (A, B)
- apply_platt(scores, A, B) -> List[float]
- fit_temperature(scores, labels, lr=1e-2, epochs=1000, l2=1e-4) -> (mu, std, T)
- apply_temperature(scores, mu, std, T) -> List[float]
- fit_isotonic(scores, labels) -> List[tuple[float, float]] (PAV model)
- apply_isotonic(scores, model) -> List[float]
- ece(y_true, probs, bins=10) -> float
"""
from __future__ import annotations

from typing import Iterable, List, Tuple
import math


def normalize_scores(scores: Iterable[float]) -> List[float]:
    s = list(float(x) for x in scores)
    if not s:
        return []
    smin, smax = min(s), max(s)
    if smax == smin:
        return [0.0 for _ in s]
    return [(x - smin) / (smax - smin) for x in s]


def _sigmoid(x: float) -> float:
    # clamp for numerical stability
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def fit_platt(scores: Iterable[float], labels: Iterable[int], lr: float = 1e-2, epochs: int = 2000, l2: float = 1e-3) -> Tuple[float, float]:
    # Logistic regression with 1D feature: p = sigmoid(A * s + B)
    s = [float(x) for x in scores]
    y = [int(t) for t in labels]
    if not s:
        return 1.0, 0.0
    # Standardize feature for stable optimization
    mu = sum(s) / len(s)
    var = sum((v - mu) ** 2 for v in s) / len(s)
    std = math.sqrt(var) or 1.0
    xs = [(v - mu) / std for v in s]

    A = 0.0
    B = 0.0
    n = float(len(xs))
    for _ in range(max(1, epochs)):
        gA = 0.0
        gB = 0.0
        for xi, yi in zip(xs, y):
            pi = _sigmoid(A * xi + B)
            gA += (pi - yi) * xi
            gB += (pi - yi)
        # L2 regularization
        gA = gA / n + l2 * A
        gB = gB / n + l2 * B
        A -= lr * gA
        B -= lr * gB
    # Store scaling to apply on raw scores: adjust intercept to include standardization
    A_raw = A / std
    B_raw = B - A * mu / std
    return A_raw, B_raw


def apply_platt(scores: Iterable[float], A: float, B: float) -> List[float]:
    return [_sigmoid(A * float(s) + B) for s in scores]


# Isotonic calibration via pair-adjacent violators (PAV)
def fit_isotonic(scores: Iterable[float], labels: Iterable[int]) -> List[Tuple[float, float]]:
    """Fit an isotonic regression model mapping score->prob using PAV.

    Returns a list of (threshold, value) pairs representing a non-decreasing
    step function. Input is sorted by score ascending for stability.
    """
    xs = [float(s) for s in scores]
    ys = [float(int(y)) for y in labels]
    if not xs:
        return []
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    z = [ys[i] for i in order]
    w = [1.0 for _ in z]
    # PAV: merge adjacent blocks when monotonicity is violated
    v = z[:]  # block means
    wt = w[:]
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1]:
            i += 1
            continue
        # merge blocks i and i+1
        totw = wt[i] + wt[i + 1]
        mean = (v[i] * wt[i] + v[i + 1] * wt[i + 1]) / totw
        v[i] = mean
        wt[i] = totw
        del v[i + 1]
        del wt[i + 1]
        # backtrack if needed
        i = max(0, i - 1)
    # Now map block averages onto sorted xs; compress into step knots
    steps: List[Tuple[float, float]] = []
    n = len(order)
    idx = 0
    for mean, size in zip(v, wt):
        size_i = int(round(size))
        if size_i <= 0:
            continue
        end_pos = min(n - 1, idx + size_i - 1)
        thr = xs[order[end_pos]]
        steps.append((thr, mean))
        idx = end_pos + 1
    if steps:
        steps[-1] = (float("inf"), steps[-1][1])
    return steps


def apply_isotonic(scores: Iterable[float], model: List[Tuple[float, float]]) -> List[float]:
    if not model:
        return [0.0 for _ in scores]
    out: List[float] = []
    for s in scores:
        x = float(s)
        val = model[-1][1]
        for thr, m in model:
            if x <= thr:
                val = m
                break
        out.append(max(0.0, min(1.0, val)))
    return out


def ece(y_true: Iterable[int], probs: Iterable[float], bins: int = 10) -> float:
    yt = [int(v) for v in y_true]
    pr = [float(p) for p in probs]
    if not pr:
        return 0.0
    bins = max(1, int(bins))
    counts = [0] * bins
    conf_sum = [0.0] * bins
    acc_sum = [0.0] * bins
    for y, p in zip(yt, pr):
        b = min(int(p * bins), bins - 1)
        counts[b] += 1
        conf_sum[b] += p
        acc_sum[b] += y
    ece_val = 0.0
    total = float(len(pr))
    for c, s_conf, s_acc in zip(counts, conf_sum, acc_sum):
        if c == 0:
            continue
        avg_conf = s_conf / c
        avg_acc = s_acc / c
        ece_val += (c / total) * abs(avg_acc - avg_conf)
    return ece_val

# Temperature scaling (1-parameter) on standardized scores
def fit_temperature(scores: Iterable[float], labels: Iterable[int], lr: float = 1e-2, epochs: int = 1000, l2: float = 1e-4) -> tuple[float, float, float]:
    """Fit temperature T > 0 to minimize cross-entropy on standardized scores.

    We standardize scores: z = (s - mu) / std. Probability is p = sigmoid(z / T).
    Optimize u = log(T) with gradient descent for numerical stability.
    Returns (mu, std, T).
    """
    s = [float(x) for x in scores]
    y = [int(v) for v in labels]
    if not s:
        return 0.0, 1.0, 1.0
    mu = sum(s) / len(s)
    var = sum((v - mu) ** 2 for v in s) / len(s)
    std = math.sqrt(var) or 1.0
    z = [(v - mu) / std for v in s]
    u = 0.0  # log T; start at T=1
    n = float(len(z))
    for _ in range(max(1, epochs)):
        T = math.exp(u)
        grad = 0.0
        # dL/du = sum((p - y) * ( -z / T)) + l2*u  where p = sigmoid(z/T)
        invT = 1.0 / T
        for zi, yi in zip(z, y):
            p = _sigmoid(zi * invT)
            grad += (p - yi) * (-zi * invT)
        grad = grad / n + l2 * u
        u -= lr * grad
    T = math.exp(u)
    return mu, std, T


def apply_temperature(scores: Iterable[float], mu: float, std: float, T: float) -> List[float]:
    if std <= 0:
        std = 1.0
    invT = 1.0 / max(T, 1e-6)
    out: List[float] = []
    for s in scores:
        z = (float(s) - mu) / std
        out.append(_sigmoid(z * invT))
    return out
