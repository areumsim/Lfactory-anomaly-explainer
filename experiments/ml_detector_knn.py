"""ML-style anomaly detector (initial version).

Research context:
- Provide a lightweight, dependency-free detector distinct from the rule
  baseline. We approximate density via k-nearest-neighbor distance on the
  value domain (ignoring time ordering for now). Points with large average
  kNN distance receive higher anomaly scores.

Exports:
- knn_scores(series, k=10) -> List[float]
- detect_knn(series, k=10, quantile=0.99) -> dict(scores, preds, params)

Notes:
- This is an initial scaffold for B2 (ML Detector). It avoids external
  dependencies like scikit-learn. Future loops can swap this out for a
  proper IsolationForest/LSTM-AE while keeping the same interface.
"""
from __future__ import annotations

from typing import List, Dict


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


def knn_scores(series: List[float], k: int = 10) -> List[float]:
    """Compute inverse-density scores via average kNN distance in value space.

    Implementation outline:
    - Sort (value, original_index) pairs by value.
    - For each position in the sorted array, look up to k neighbors on both
      sides and compute the average absolute difference in value.
    - Map the average distance back to original index; score is proportional
      to this average distance (larger = more anomalous).
    """
    n = len(series)
    if n == 0:
        return []
    k = max(1, int(k))
    pairs = sorted([(float(v), i) for i, v in enumerate(series)], key=lambda x: x[0])
    vals = [p[0] for p in pairs]
    idxs = [p[1] for p in pairs]

    # Precompute cumulative sums to get window averages quickly
    # However, we need absolute differences to neighbors; use local sweep.
    scores_sorted: List[float] = [0.0] * n
    for j in range(n):
        # Expand to k nearest neighbors by index distance in sorted order
        left = j - 1
        right = j + 1
        taken = 0
        sdist = 0.0
        while taken < k and (left >= 0 or right < n):
            # choose the closer neighbor in value space
            dl = abs(vals[j] - vals[left]) if left >= 0 else float("inf")
            dr = abs(vals[j] - vals[right]) if right < n else float("inf")
            if dl <= dr:
                sdist += dl
                left -= 1
            else:
                sdist += dr
                right += 1
            taken += 1
        avg = sdist / max(1, taken)
        scores_sorted[j] = avg

    # Map back to original order
    scores: List[float] = [0.0] * n
    for v, i, s in zip(vals, idxs, scores_sorted):
        scores[i] = s
    return scores


def detect_knn(series: List[float], k: int = 10, quantile: float = 0.99) -> Dict:
    """Detect anomalies using kNN-distance scores with quantile thresholding.

    Returns a dict with keys: scores, preds, params.
    """
    sc = knn_scores(series, k=k)
    thr = _quantile(sc, quantile)
    preds = [1 if s >= thr else 0 for s in sc]
    return {
        "scores": sc,
        "preds": preds,
        "params": {
            "method": "knn_value_density",
            "k": int(k),
            "quantile": float(quantile),
        },
    }

