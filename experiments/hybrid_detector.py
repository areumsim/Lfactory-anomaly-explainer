"""Hybrid detector: combine Rule-based and ML-based scores.

설계:
- Rule: rolling z-score 또는 robust z-score
- ML: kNN 값-공간 기반 점수
- 결합: 두 스코어를 정규화한 뒤 가중합 score = alpha * ml + (1-alpha) * rule
- 임계: 분위수 기반(기본), 비용 최적화는 main_experiment에서 공통 처리

의존성: 내부 모듈만 사용, 외부 패키지 없음.
"""
from __future__ import annotations

from typing import Dict, List

from . import rule_detector
from . import ml_detector_knn
from . import calibration


def _minmax(vs: List[float]) -> List[float]:
    return calibration.normalize_scores(vs)


def detect_hybrid(
    series: List[float],
    alpha: float = 0.5,
    rule_window: int = 50,
    rule_threshold: float = 3.0,
    rule_min_std: float = 1e-3,
    robust: bool = False,
    ml_k: int = 10,
    quantile: float = 0.99,
) -> Dict:
    # Compute rule scores
    if robust:
        r = rule_detector.robust_zscore_detect(series, window=rule_window, threshold=rule_threshold)
    else:
        r = rule_detector.zscore_detect(series, window=rule_window, threshold=rule_threshold, min_std=rule_min_std)
    # Compute ml scores
    m = ml_detector_knn.detect_knn(series, k=ml_k, quantile=quantile)

    rs = _minmax(r["scores"]) if r["scores"] else []
    ms = _minmax(m["scores"]) if m["scores"] else []
    n = min(len(rs), len(ms))
    scores: List[float] = []
    for i in range(n):
        scores.append((1.0 - alpha) * rs[i] + alpha * ms[i])
    # Threshold by quantile of combined scores
    thr = ml_detector_knn._quantile(scores, quantile)
    preds = [1 if s >= thr else 0 for s in scores]
    return {
        "scores": scores,
        "preds": preds,
        "params": {
            "method": "hybrid_rule_ml",
            "alpha": float(alpha),
            "rule_window": int(rule_window),
            "rule_threshold": float(rule_threshold),
            "ml_k": int(ml_k),
            "quantile": float(quantile),
            "robust_rule": bool(robust),
        },
    }

