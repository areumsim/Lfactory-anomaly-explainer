"""Result Manager: persistence and visualization helpers.

Research context:
- Centralizes saving artifacts (JSON/CSV) and basic plots to support
  reproducibility and quick inspection. Plotting gracefully degrades
  when matplotlib is unavailable (CSV curves are still saved).

Initial loop capabilities:
- Save predictions to CSV
- Compute ROC curve and AUC from anomaly scores
- Save ROC plot if matplotlib is installed; otherwise save ROC CSV only
- Experimental calibration plot based on min-max normalized scores
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional
import csv
import json
import os


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_predictions_csv(path: str, series: Iterable[float], labels: Iterable[int], scores: Iterable[float], preds: Iterable[int], probs: Optional[Iterable[float]] = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if probs is None:
            w.writerow(["index", "value", "label", "score", "pred"])
            for i, (v, y, s, p) in enumerate(zip(series, labels, scores, preds)):
                w.writerow([i, v, y, s, p])
        else:
            w.writerow(["index", "value", "label", "score", "pred", "prob"])
            for i, (v, y, s, p, pr) in enumerate(zip(series, labels, scores, preds, probs)):
                w.writerow([i, v, y, s, p, pr])


def save_features_csv(path: str, features: Dict[str, Any]) -> None:
    """Save a single-row feature dictionary as CSV.

    Keys become column headers; values are written as-is (JSON-serializable
    recommended). Useful for B3 FeatureBank initial verification.
    """
    ensure_dir(os.path.dirname(path) or ".")
    # Single row CSV with header
    cols = list(features.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow([features[k] for k in cols])


def _roc_from_scores(y_true: List[int], scores: List[float]) -> Tuple[List[float], List[float], List[float]]:
    # Sort by score descending
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    P = sum(1 for y in y_true if y == 1)
    N = sum(1 for y in y_true if y == 0)
    tp = fp = 0
    fpr: List[float] = [0.0]
    tpr: List[float] = [0.0]
    thresholds: List[float] = [float("inf")]
    last_score = None
    for i in order:
        s = scores[i]
        y = y_true[i]
        if last_score is not None and s != last_score:
            fpr.append(fp / N if N else 0.0)
            tpr.append(tp / P if P else 0.0)
            thresholds.append(last_score)
        if y == 1:
            tp += 1
        else:
            fp += 1
        last_score = s
    # Append final point
    fpr.append(fp / N if N else 0.0)
    tpr.append(tp / P if P else 0.0)
    thresholds.append(last_score if last_score is not None else 0.0)
    return fpr, tpr, thresholds


def _auc(x: List[float], y: List[float]) -> float:
    area = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        area += dx * (y[i] + y[i - 1]) / 2.0
    return area


# Precision-Recall utilities
@dataclass
class PrResult:
    precision: List[float]
    recall: List[float]
    thresholds: List[float]
    auc: float


def compute_pr(y_true: Iterable[int], scores: Iterable[float]) -> PrResult:
    yt = list(int(v) for v in y_true)
    sc = list(float(v) for v in scores)
    order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
    P = sum(1 for y in yt if y == 1)
    tp = fp = 0
    precision: List[float] = []
    recall: List[float] = []
    thresholds: List[float] = []
    last_score = None
    for i in order:
        s = sc[i]
        y = yt[i]
        if last_score is not None and s != last_score:
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / P if P else 0.0
            precision.append(prec)
            recall.append(rec)
            thresholds.append(last_score)
        if y == 1:
            tp += 1
        else:
            fp += 1
        last_score = s
    # final point
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / P if P else 0.0
    precision.append(prec)
    recall.append(rec)
    thresholds.append(last_score if last_score is not None else 0.0)
    # AUC-PR via trapezoid on recall axis
    # Ensure sorted by recall ascending
    pairs = sorted(zip(recall, precision))
    xs = [r for r, _ in pairs]
    ys = [p for _, p in pairs]
    auc = _auc(xs, ys)
    return PrResult(precision=ys, recall=xs, thresholds=thresholds, auc=auc)


def save_pr_csv(path: str, pr: PrResult) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["recall", "precision"])
        for r, p in zip(pr.recall, pr.precision):
            w.writerow([r, p])


def save_pr_plot(path: str, pr: PrResult) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    ensure_dir(os.path.dirname(path) or ".")
    plt.figure(figsize=(4, 4))
    plt.plot(pr.recall, pr.precision, label=f"PR AUC={pr.auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


@dataclass
class RocResult:
    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]
    auc: float


def compute_roc(y_true: Iterable[int], scores: Iterable[float]) -> RocResult:
    yt = list(int(v) for v in y_true)
    sc = list(float(v) for v in scores)
    fpr, tpr, th = _roc_from_scores(yt, sc)
    auc = _auc(fpr, tpr)
    return RocResult(fpr=fpr, tpr=tpr, thresholds=th, auc=auc)


def save_roc_csv(path: str, roc: RocResult) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for a, b, c in zip(roc.fpr, roc.tpr, roc.thresholds):
            w.writerow([a, b, c])


def save_roc_plot(path: str, roc: RocResult) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    ensure_dir(os.path.dirname(path) or ".")
    plt.figure(figsize=(4, 4))
    plt.plot(roc.fpr, roc.tpr, label=f"ROC AUC={roc.auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def save_calibration_plot(path: str, y_true: Iterable[int], probs: Iterable[float], bins: int = 10) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    yt = list(int(v) for v in y_true)
    pr = list(float(v) for v in probs)
    if not pr:
        return False
    bins = max(1, int(bins))
    sums = [0.0] * bins
    counts = [0] * bins
    obs = [0.0] * bins
    for p, y in zip(pr, yt):
        b = min(int(p * bins), bins - 1)
        sums[b] += p
        counts[b] += 1
        obs[b] += y
    xs: List[float] = []
    ys: List[float] = []
    for i in range(bins):
        if counts[i] > 0:
            xs.append(sums[i] / counts[i])
            ys.append(obs[i] / counts[i])
    ensure_dir(os.path.dirname(path) or ".")
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.scatter(xs, ys)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration (reliability diagram)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True
