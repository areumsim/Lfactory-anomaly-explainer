"""
Feature Importance Analysis for Multi-Sensor Anomaly Detection

Provides SHAP-based and reconstruction-error-based feature attribution
for explaining which sensors contributed most to anomaly detection.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings


def compute_reconstruction_importance(
    multi_sensor_data: np.ndarray,
    reconstruction_errors: np.ndarray,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute feature importance based on per-sensor reconstruction errors.

    For LSTM-AE and other reconstruction-based methods, the contribution
    of each sensor is measured by its individual reconstruction error.

    Args:
        multi_sensor_data: Shape (n_samples, n_sensors) - Original sensor values
        reconstruction_errors: Shape (n_samples, n_sensors) - Per-sensor errors
        sensor_names: Optional list of sensor names

    Returns:
        dict with:
            - importance_scores: Per-sensor importance (normalized)
            - sensor_contributions: Raw contribution values
            - ranking: Sensors ranked by importance
    """
    n_sensors = multi_sensor_data.shape[1] if len(multi_sensor_data.shape) > 1 else 1

    if sensor_names is None:
        sensor_names = [f"sensor_{i}" for i in range(n_sensors)]

    # Mean reconstruction error per sensor
    if len(reconstruction_errors.shape) == 1:
        # Single sensor case
        return {
            "importance_scores": {sensor_names[0]: 1.0},
            "sensor_contributions": {sensor_names[0]: float(np.mean(reconstruction_errors))},
            "ranking": [sensor_names[0]]
        }

    mean_errors = np.mean(reconstruction_errors, axis=0)
    total_error = np.sum(mean_errors)

    if total_error > 0:
        importance = mean_errors / total_error
    else:
        importance = np.ones(n_sensors) / n_sensors

    # Create results
    importance_scores = {name: float(imp) for name, imp in zip(sensor_names, importance)}
    contributions = {name: float(err) for name, err in zip(sensor_names, mean_errors)}

    # Rank sensors by importance
    ranking = sorted(sensor_names, key=lambda x: importance_scores[x], reverse=True)

    return {
        "importance_scores": importance_scores,
        "sensor_contributions": contributions,
        "ranking": ranking
    }


def compute_shap_importance_isolation_forest(
    data: np.ndarray,
    model,
    sensor_names: Optional[List[str]] = None,
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Compute SHAP values for IsolationForest model.

    Args:
        data: Shape (n_samples, n_features) - Input data
        model: Trained IsolationForest model
        sensor_names: Optional list of sensor names
        n_samples: Number of background samples for SHAP

    Returns:
        dict with SHAP-based feature importance
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP package required. Install: pip install shap")

    n_features = data.shape[1] if len(data.shape) > 1 else 1

    if sensor_names is None:
        sensor_names = [f"sensor_{i}" for i in range(n_features)]

    # Use TreeExplainer for IsolationForest
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Sample background data
        if len(data) > n_samples:
            bg_indices = np.random.choice(len(data), n_samples, replace=False)
            background = data[bg_indices]
        else:
            background = data

        try:
            explainer = shap.TreeExplainer(model, background)
            shap_values = explainer.shap_values(data)

            # Mean absolute SHAP values per feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            total_shap = np.sum(mean_abs_shap)

            if total_shap > 0:
                importance = mean_abs_shap / total_shap
            else:
                importance = np.ones(n_features) / n_features

            importance_scores = {name: float(imp) for name, imp in zip(sensor_names, importance)}
            ranking = sorted(sensor_names, key=lambda x: importance_scores[x], reverse=True)

            return {
                "importance_scores": importance_scores,
                "shap_values": shap_values,
                "ranking": ranking,
                "method": "TreeExplainer"
            }
        except Exception as e:
            # Fallback to permutation importance
            return compute_permutation_importance(data, model, sensor_names)


def compute_permutation_importance(
    data: np.ndarray,
    model,
    sensor_names: Optional[List[str]] = None,
    n_repeats: int = 10
) -> Dict[str, Any]:
    """
    Compute permutation-based feature importance.

    Works with any model that has predict or decision_function method.

    Args:
        data: Shape (n_samples, n_features)
        model: Trained model with predict/decision_function
        sensor_names: Optional list of sensor names
        n_repeats: Number of permutation repeats

    Returns:
        dict with permutation-based importance
    """
    n_features = data.shape[1] if len(data.shape) > 1 else 1

    if sensor_names is None:
        sensor_names = [f"sensor_{i}" for i in range(n_features)]

    # Get baseline scores
    if hasattr(model, 'decision_function'):
        baseline_scores = model.decision_function(data)
    elif hasattr(model, 'predict'):
        baseline_scores = model.predict(data)
    else:
        raise ValueError("Model must have decision_function or predict method")

    baseline_metric = np.mean(np.abs(baseline_scores))

    importances = []
    for feat_idx in range(n_features):
        feat_importance = []
        for _ in range(n_repeats):
            # Permute feature
            data_permuted = data.copy()
            np.random.shuffle(data_permuted[:, feat_idx])

            # Get new scores
            if hasattr(model, 'decision_function'):
                permuted_scores = model.decision_function(data_permuted)
            else:
                permuted_scores = model.predict(data_permuted)

            permuted_metric = np.mean(np.abs(permuted_scores))
            feat_importance.append(abs(baseline_metric - permuted_metric))

        importances.append(np.mean(feat_importance))

    importances = np.array(importances)
    total = np.sum(importances)
    if total > 0:
        normalized = importances / total
    else:
        normalized = np.ones(n_features) / n_features

    importance_scores = {name: float(imp) for name, imp in zip(sensor_names, normalized)}
    ranking = sorted(sensor_names, key=lambda x: importance_scores[x], reverse=True)

    return {
        "importance_scores": importance_scores,
        "raw_importances": {name: float(imp) for name, imp in zip(sensor_names, importances)},
        "ranking": ranking,
        "method": "PermutationImportance"
    }


def compute_anomaly_point_attribution(
    anomaly_idx: int,
    multi_sensor_data: np.ndarray,
    reconstruction_errors: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    context_window: int = 20
) -> Dict[str, Any]:
    """
    Compute feature attribution for a specific anomaly point.

    Analyzes which sensors contributed most to the detection
    of a particular anomaly.

    Args:
        anomaly_idx: Index of the anomaly point
        multi_sensor_data: Shape (n_samples, n_sensors)
        reconstruction_errors: Shape (n_samples, n_sensors)
        sensor_names: Optional list of sensor names
        context_window: Window size for baseline comparison

    Returns:
        dict with point-specific attribution
    """
    n_sensors = multi_sensor_data.shape[1] if len(multi_sensor_data.shape) > 1 else 1

    if sensor_names is None:
        sensor_names = [f"sensor_{i}" for i in range(n_sensors)]

    # Handle edge cases
    start_idx = max(0, anomaly_idx - context_window)
    end_idx = min(len(multi_sensor_data), anomaly_idx + context_window + 1)

    if len(reconstruction_errors.shape) == 1:
        # Single sensor
        point_error = reconstruction_errors[anomaly_idx]
        baseline_error = np.mean(reconstruction_errors[start_idx:end_idx])

        return {
            "anomaly_index": anomaly_idx,
            "sensor_values": {sensor_names[0]: float(multi_sensor_data[anomaly_idx])},
            "sensor_errors": {sensor_names[0]: float(point_error)},
            "baseline_errors": {sensor_names[0]: float(baseline_error)},
            "deviation_ratio": {sensor_names[0]: float(point_error / baseline_error) if baseline_error > 0 else 1.0},
            "top_contributors": [sensor_names[0]],
            "contribution_scores": {sensor_names[0]: 1.0}
        }

    # Multi-sensor case
    point_errors = reconstruction_errors[anomaly_idx]
    context_errors = reconstruction_errors[start_idx:end_idx]
    baseline_errors = np.mean(context_errors, axis=0)

    # Deviation from baseline
    deviations = np.zeros(n_sensors)
    for i in range(n_sensors):
        if baseline_errors[i] > 0:
            deviations[i] = point_errors[i] / baseline_errors[i]
        else:
            deviations[i] = 1.0 if point_errors[i] == 0 else float('inf')

    # Contribution scores (normalized deviation)
    total_deviation = np.sum(deviations)
    if total_deviation > 0 and not np.isinf(total_deviation):
        contributions = deviations / total_deviation
    else:
        contributions = np.ones(n_sensors) / n_sensors

    # Top contributors (sorted by contribution)
    sensor_contrib_pairs = list(zip(sensor_names, contributions))
    sensor_contrib_pairs.sort(key=lambda x: x[1], reverse=True)
    top_contributors = [name for name, _ in sensor_contrib_pairs]

    return {
        "anomaly_index": anomaly_idx,
        "sensor_values": {name: float(multi_sensor_data[anomaly_idx, i])
                        for i, name in enumerate(sensor_names)},
        "sensor_errors": {name: float(point_errors[i])
                        for i, name in enumerate(sensor_names)},
        "baseline_errors": {name: float(baseline_errors[i])
                          for i, name in enumerate(sensor_names)},
        "deviation_ratio": {name: float(deviations[i])
                          for i, name in enumerate(sensor_names)},
        "top_contributors": top_contributors,
        "contribution_scores": {name: float(contributions[i])
                               for i, name in enumerate(sensor_names)}
    }


def format_importance_for_llm(
    importance_result: Dict[str, Any],
    top_n: int = 3
) -> str:
    """
    Format feature importance results for LLM explanation prompt.

    Args:
        importance_result: Output from compute_* functions
        top_n: Number of top features to highlight

    Returns:
        Formatted string for LLM prompt
    """
    lines = []

    if "ranking" in importance_result:
        top_sensors = importance_result["ranking"][:top_n]
        scores = importance_result.get("importance_scores", {})

        lines.append("**Top Contributing Sensors:**")
        for i, sensor in enumerate(top_sensors, 1):
            score = scores.get(sensor, 0)
            lines.append(f"  {i}. {sensor}: {score:.1%} contribution")

    if "deviation_ratio" in importance_result:
        lines.append("\n**Deviation from Normal:**")
        for sensor in importance_result.get("top_contributors", [])[:top_n]:
            ratio = importance_result["deviation_ratio"].get(sensor, 1.0)
            if ratio > 1.5:
                lines.append(f"  - {sensor}: {ratio:.1f}x higher than baseline (significant)")
            elif ratio > 1.1:
                lines.append(f"  - {sensor}: {ratio:.1f}x higher than baseline (moderate)")
            else:
                lines.append(f"  - {sensor}: within normal range")

    if "method" in importance_result:
        lines.append(f"\n(Analysis method: {importance_result['method']})")

    return "\n".join(lines)
