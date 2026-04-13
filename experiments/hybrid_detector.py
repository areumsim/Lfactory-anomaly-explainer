"""Hybrid detector: combine Rule-based and ML-based scores.

설계:
- Rule: rolling z-score 또는 robust z-score
- ML: kNN, IsolationForest, 또는 LSTM-AE 기반 점수
- 결합 방법: linear, product, max, learned (logistic regression)
- 임계: 분위수 기반(기본), 비용 최적화는 main_experiment에서 공통 처리

의존성: 내부 모듈만 사용, 외부 패키지 없음.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import math

from . import rule_detector
from . import ml_detector_knn
from . import calibration


def _minmax(vs: List[float]) -> List[float]:
    return calibration.normalize_scores(vs)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _fit_logistic_2d(
    x1: List[float], x2: List[float], y: List[int],
    lr: float = 1e-2, epochs: int = 2000, l2: float = 1e-3,
) -> tuple[float, float, float]:
    """Fit 2D logistic regression: p = sigmoid(w1*x1 + w2*x2 + b).

    Returns (w1, w2, b).
    """
    n = len(x1)
    if n == 0:
        return 1.0, 1.0, 0.0
    s = sum(y)
    if s == 0 or s == n:
        return 1.0, 1.0, 0.0

    # Standardize inputs
    mu1 = sum(x1) / n
    mu2 = sum(x2) / n
    var1 = sum((v - mu1) ** 2 for v in x1) / n
    var2 = sum((v - mu2) ** 2 for v in x2) / n
    std1 = math.sqrt(var1) if var1 > 0 else 1.0
    std2 = math.sqrt(var2) if var2 > 0 else 1.0
    z1 = [(v - mu1) / std1 for v in x1]
    z2 = [(v - mu2) / std2 for v in x2]

    w1 = 0.0
    w2 = 0.0
    b = 0.0
    nf = float(n)

    for _ in range(max(1, epochs)):
        gw1 = 0.0
        gw2 = 0.0
        gb = 0.0
        for i in range(n):
            p = _sigmoid(w1 * z1[i] + w2 * z2[i] + b)
            err = p - y[i]
            gw1 += err * z1[i]
            gw2 += err * z2[i]
            gb += err
        gw1 = gw1 / nf + l2 * w1
        gw2 = gw2 / nf + l2 * w2
        gb = gb / nf + l2 * b
        w1 -= lr * gw1
        w2 -= lr * gw2
        b -= lr * gb

    # Convert back to raw scale
    w1_raw = w1 / std1
    w2_raw = w2 / std2
    b_raw = b - w1 * mu1 / std1 - w2 * mu2 / std2
    return w1_raw, w2_raw, b_raw


def _get_ml_scores(
    series: List[float],
    ml_backend: str,
    ml_k: int,
    quantile: float,
    seed: int = 42,
    device: str = "auto",
) -> Dict:
    """Run the specified ML backend detector."""
    if ml_backend == "isolation_forest":
        from . import ml_detector_isolation_forest
        return ml_detector_isolation_forest.detect_isolation_forest(
            series, window_size=50, contamination=0.1,
            n_estimators=100, random_state=seed,
        )
    elif ml_backend == "lstm_ae":
        from . import ml_detector_lstm_ae
        return ml_detector_lstm_ae.detect_lstm_ae(
            series, sequence_length=50, latent_dim=32,
            epochs=50, learning_rate=0.001, batch_size=32,
            quantile=0.95, random_state=seed, device=device,
        )
    else:  # default: knn
        return ml_detector_knn.detect_knn(series, k=ml_k, quantile=quantile)


def detect_hybrid(
    series: List[float],
    alpha: float = 0.5,
    rule_window: int = 50,
    rule_threshold: float = 3.0,
    rule_min_std: float = 1e-3,
    robust: bool = False,
    ml_k: int = 10,
    quantile: float = 0.99,
    method: str = "linear",
    ml_backend: str = "knn",
    labels: Optional[List[int]] = None,
    seed: int = 42,
    device: str = "auto",
) -> Dict:
    """Detect anomalies using a hybrid rule+ML ensemble.

    Args:
        method: Ensemble method - "linear", "product", "max", or "learned".
        ml_backend: ML detector - "knn", "isolation_forest", or "lstm_ae".
        labels: Ground-truth labels (required for "learned" method).
        seed: Random state for ML detectors.
        device: Device for PyTorch models.
    """
    # Compute rule scores
    if robust:
        r = rule_detector.robust_zscore_detect(series, window=rule_window, threshold=rule_threshold)
    else:
        r = rule_detector.zscore_detect(series, window=rule_window, threshold=rule_threshold, min_std=rule_min_std)

    # Compute ML scores
    m = _get_ml_scores(series, ml_backend=ml_backend, ml_k=ml_k, quantile=quantile, seed=seed, device=device)

    rs = _minmax(r["scores"]) if r["scores"] else []
    ms = _minmax(m["scores"]) if m["scores"] else []
    n = min(len(rs), len(ms))

    scores: List[float] = []

    if method == "product":
        for i in range(n):
            # Geometric mean: sqrt(ml * rule), handle zeros
            scores.append(math.sqrt(max(0.0, rs[i]) * max(0.0, ms[i])))

    elif method == "max":
        for i in range(n):
            scores.append(max(rs[i], ms[i]))

    elif method == "learned":
        if labels is not None and len(labels) >= n and sum(labels[:n]) > 0:
            # Temporal split to avoid train/test data leakage
            split_idx = max(10, int(0.6 * n))
            train_labels = labels[:split_idx]
            if sum(train_labels) == 0 or sum(train_labels) == split_idx:
                # Single-class in train split → fallback to linear
                for i in range(n):
                    scores.append((1.0 - alpha) * rs[i] + alpha * ms[i])
            else:
                w1, w2, b = _fit_logistic_2d(ms[:split_idx], rs[:split_idx], train_labels)
                for i in range(n):
                    scores.append(_sigmoid(w1 * ms[i] + w2 * rs[i] + b))
        else:
            # Fallback to linear if no labels
            for i in range(n):
                scores.append((1.0 - alpha) * rs[i] + alpha * ms[i])

    else:  # "linear" (default)
        for i in range(n):
            scores.append((1.0 - alpha) * rs[i] + alpha * ms[i])

    # Threshold by quantile of combined scores
    thr = ml_detector_knn._quantile(scores, quantile)
    preds = [1 if s >= thr else 0 for s in scores]
    params = {
        "method": f"hybrid_{method}",
        "ensemble_method": method,
        "ml_backend": ml_backend,
        "alpha": float(alpha),
        "rule_window": int(rule_window),
        "rule_threshold": float(rule_threshold),
        "ml_k": int(ml_k),
        "quantile": float(quantile),
        "robust_rule": bool(robust),
    }
    if method == "learned":
        params["learned_train_frac"] = 0.6
    return {
        "scores": scores,
        "preds": preds,
        "params": params,
    }
