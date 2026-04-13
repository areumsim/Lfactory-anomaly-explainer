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

from typing import Iterable, List, Tuple, Optional


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


def estimate_snr(series: Iterable[float], labels: Iterable[int]) -> float:
    """Estimate signal-to-noise ratio: anomalous_mean_deviation / normal_std.

    Higher SNR means anomalies are more distinct from normal data.
    Returns 0.0 if no anomalies or no normal points exist.
    """
    s = [float(v) for v in series]
    y = [int(v) for v in labels]
    normal = [v for v, lbl in zip(s, y) if lbl == 0]
    anomalous = [v for v, lbl in zip(s, y) if lbl != 0]
    if not normal or not anomalous:
        return 0.0
    mu_n = sum(normal) / len(normal)
    var_n = sum((v - mu_n) ** 2 for v in normal) / len(normal)
    std_n = var_n ** 0.5
    if std_n < 1e-12:
        return 0.0
    mu_a = sum(anomalous) / len(anomalous)
    return abs(mu_a - mu_n) / std_n


def cost_sensitivity_sweep(
    y_true: Iterable[int],
    scores: Iterable[float],
    ratios: List[float] | None = None,
) -> List[dict]:
    """Sweep FN/FP cost ratios and find optimal threshold for each.

    Args:
        y_true: Binary ground-truth labels.
        scores: Anomaly scores.
        ratios: List of FN/FP cost ratios to evaluate (default: [1,2,5,10,20,50]).

    Returns:
        List of dicts with keys: ratio, costs, optimal_threshold, optimal_cost,
        baseline_cost (at fixed 5:1 ratio threshold), metrics.
    """
    if ratios is None:
        ratios = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    sc = [float(s) for s in scores]
    yt = [int(v) for v in y_true]

    results = []
    for ratio in ratios:
        # Cost matrix: C00=0, C01=1 (FP), C10=ratio (FN), C11=0
        costs: Tuple[float, float, float, float] = (0.0, 1.0, ratio, 0.0)
        opt_thr, opt_cost = find_optimal_threshold(yt, sc, costs)

        # Compute metrics at optimal threshold
        preds = [1 if s >= opt_thr else 0 for s in sc]
        tp, tn, fp, fn = _counts(yt, preds)
        total = max(1, tp + tn + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)

        results.append({
            "ratio": ratio,
            "costs": {"c00": 0.0, "c01": 1.0, "c10": ratio, "c11": 0.0},
            "optimal_threshold": opt_thr,
            "optimal_cost": opt_cost,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    return results


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

