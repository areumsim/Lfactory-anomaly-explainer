"""
간단 스모크 테스트: rule_detector 경계조건 확인

사용:
  python scripts/test_rule_detector.py

요구사항: 표준 라이브러리만 사용
"""
from __future__ import annotations

from experiments import rule_detector


def _assert_no_anomaly_constant() -> None:
    series = [1.0] * 100
    res = rule_detector.zscore_detect(series, window=10, threshold=3.0, min_std=1e-6)
    assert sum(res["preds"]) == 0, "Constant series should have no anomalies"


def _assert_spike_detected() -> None:
    series = [0.0] * 50 + [10.0] + [0.0] * 49
    res = rule_detector.zscore_detect(series, window=10, threshold=3.0, min_std=1e-3)
    assert any(res["preds"][40:60]), "Spike should trigger anomaly near center"


def _assert_robust_handles_outliers() -> None:
    series = [0.0] * 50 + [20.0] + [0.0] * 49
    res = rule_detector.robust_zscore_detect(series, window=11, threshold=3.5, min_mad=1e-6)
    assert any(res["preds"][45:55]), "Robust detector should flag extreme spike"


def main() -> None:
    _assert_no_anomaly_constant()
    _assert_spike_detected()
    _assert_robust_handles_outliers()
    print("OK: rule_detector smoke tests passed")


if __name__ == "__main__":
    main()

