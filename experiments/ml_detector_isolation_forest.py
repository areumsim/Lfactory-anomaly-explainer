"""IsolationForest-based anomaly detector with temporal windowing.

Research context:
- Replaces the placeholder kNN detector with a proper temporal anomaly detection
  approach. Uses sliding window features (mean, std, min, max, trend) combined
  with IsolationForest from scikit-learn.
- Addresses the limitation that kNN ignores time structure by extracting temporal
  features from rolling windows.

Exports:
- detect_isolation_forest(series, window_size=50, contamination=0.1, **kwargs) -> dict

Returns:
- dict with keys: scores, preds, params

Dependencies:
- scikit-learn (IsolationForest)
- numpy (optional, for numerical operations)

Notes:
- This implementation requires scikit-learn. If not available, it will raise
  an ImportError with a helpful message.
- Window features capture temporal patterns (trend, volatility) that pure
  value-space methods miss.
"""
from __future__ import annotations

from typing import List, Dict, Any
import math


def _extract_window_features(series: List[float], window_size: int) -> List[List[float]]:
    """Extract temporal features from sliding windows.

    Features per window:
    - mean: average value
    - std: standard deviation
    - min: minimum value
    - max: maximum value
    - trend: linear regression slope (simple approximation)

    Returns a list of feature vectors, one per time point.
    For the first (window_size-1) points, we use expanding windows.
    """
    n = len(series)
    features = []

    for i in range(n):
        # Define window bounds
        start = max(0, i - window_size + 1)
        end = i + 1
        window = series[start:end]

        if len(window) == 0:
            # Should not happen, but safety
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        # Mean
        mean_val = sum(window) / len(window)

        # Std (sample std)
        if len(window) > 1:
            variance = sum((x - mean_val) ** 2 for x in window) / (len(window) - 1)
            std_val = math.sqrt(variance)
        else:
            std_val = 0.0

        # Min, Max
        min_val = min(window)
        max_val = max(window)

        # Trend (simple linear regression slope)
        # slope = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
        # x = time indices, y = values
        if len(window) > 1:
            x_indices = list(range(len(window)))
            mean_x = sum(x_indices) / len(x_indices)
            mean_y = mean_val

            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_indices, window))
            denominator = sum((x - mean_x) ** 2 for x in x_indices)

            if denominator > 1e-10:
                trend = numerator / denominator
            else:
                trend = 0.0
        else:
            trend = 0.0

        features.append([mean_val, std_val, min_val, max_val, trend])

    return features


def detect_isolation_forest(
    series: List[float],
    window_size: int = 50,
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """Detect anomalies using IsolationForest on temporal window features.

    Args:
        series: Input time series data
        window_size: Size of rolling window for feature extraction
        contamination: Expected proportion of anomalies (0.0 to 0.5)
        n_estimators: Number of trees in IsolationForest
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments (ignored)

    Returns:
        dict with keys:
            - scores: List[float] - Anomaly scores (higher = more anomalous)
            - preds: List[int] - Binary predictions (1 = anomaly, 0 = normal)
            - params: dict - Parameters used
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise ImportError(
            "IsolationForest requires scikit-learn. "
            "Install it with: pip install scikit-learn"
        )

    if len(series) == 0:
        return {
            "scores": [],
            "preds": [],
            "params": {
                "method": "isolation_forest",
                "window_size": window_size,
                "contamination": contamination,
                "n_estimators": n_estimators,
            }
        }

    # Extract window features
    features = _extract_window_features(series, window_size)

    # Train IsolationForest
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        max_samples='auto',
        bootstrap=False
    )

    # Fit and predict
    # IsolationForest returns -1 for anomalies, 1 for normal
    predictions_raw = clf.fit_predict(features)

    # Get anomaly scores (decision_function returns negative scores for anomalies)
    # We negate to make higher = more anomalous
    scores_raw = clf.decision_function(features)

    # Normalize scores to [0, 1] range for consistency with other detectors
    if len(scores_raw) > 1:
        min_score = min(scores_raw)
        max_score = max(scores_raw)
        score_range = max_score - min_score

        if score_range > 1e-10:
            # Negate and normalize: more negative (anomalous) → higher score
            scores = [-(s - max_score) / score_range for s in scores_raw]
        else:
            # All scores are the same
            scores = [0.5] * len(scores_raw)
    else:
        scores = [0.5] * len(scores_raw)

    # Convert predictions: -1 (anomaly) → 1, 1 (normal) → 0
    preds = [1 if p == -1 else 0 for p in predictions_raw]

    return {
        "scores": scores,
        "preds": preds,
        "params": {
            "method": "isolation_forest",
            "window_size": int(window_size),
            "contamination": float(contamination),
            "n_estimators": int(n_estimators),
            "random_state": int(random_state),
        }
    }
