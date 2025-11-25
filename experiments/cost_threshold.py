"""Cost-sensitive threshold selection utilities.

Research context:
- Given anomaly scores and binary ground-truth labels, select a threshold
  that minimizes expected cost under a simple cost matrix.

API:
- expected_cost(y_true, y_pred, costs) -> float
- expected_cost_from_scores(y_true, scores, thr, costs) -> float
- find_optimal_threshold(y_true, scores, costs) -> (thr, cost)

Cost matrix semantics (C00, C01, C10, C11):
- C00: cost when true=0, pred=0 (TN)
- C01: cost when true=0, pred=1 (FP)
- C10: cost when true=1, pred=0 (FN)
- C11: cost when true=1, pred=1 (TP)

Defaults: (0.0, 1.0, 5.0, 0.0) i.e., FN is 5x FP.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple


def _counts(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
    return tp, tn, fp, fn


def expected_cost(y_true: Iterable[int], y_pred: Iterable[int], costs: Tuple[float, float, float, float] = (0.0, 1.0, 5.0, 0.0)) -> float:
    tp, tn, fp, fn = _counts(y_true, y_pred)
    c00, c01, c10, c11 = costs
    total = max(1, tp + tn + fp + fn)
    return (tn * c00 + fp * c01 + fn * c10 + tp * c11) / total


def expected_cost_from_scores(y_true: Iterable[int], scores: Iterable[float], thr: float, costs: Tuple[float, float, float, float] = (0.0, 1.0, 5.0, 0.0)) -> float:
    preds = [1 if float(s) >= thr else 0 for s in scores]
    return expected_cost(y_true, preds, costs)


def find_optimal_threshold(y_true: Iterable[int], scores: Iterable[float], costs: Tuple[float, float, float, float] = (0.0, 1.0, 5.0, 0.0)) -> Tuple[float, float]:
    sc = [float(s) for s in scores]
    if not sc:
        return 0.0, 0.0
    uniq = sorted(set(sc))
    # Evaluate at midpoints between unique scores and beyond extremes
    cands: List[float] = []
    cands.append(uniq[0] - 1e-9)
    for a, b in zip(uniq, uniq[1:]):
        cands.append((a + b) / 2.0)
    cands.append(uniq[-1] + 1e-9)
    best_thr = cands[0]
    best_cost = float("inf")
    for t in cands:
        c = expected_cost_from_scores(y_true, sc, t, costs)
        if c < best_cost:
            best_thr = t
            best_cost = c
    return best_thr, best_cost

