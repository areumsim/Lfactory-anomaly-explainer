"""Extended evaluation metrics for LLM-generated anomaly explanations.

Provides:
- BERTScore-based semantic faithfulness
- Structured extraction accuracy
- Human evaluation export/analysis utilities
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def build_reference_text(sample: Dict) -> str:
    """Build a ground-truth reference string from anomaly metadata.

    Used as the reference for BERTScore computation.
    """
    parts = []

    # Basic anomaly description
    parts.append(
        f"An anomaly was detected at time index {sample['anomaly_idx']} "
        f"with a deviation of {sample['sigma_deviation']:.1f} sigma from normal."
    )

    # Severity
    sigma = sample.get("sigma_deviation", 0)
    if sigma > 5:
        parts.append("This is a high severity anomaly requiring immediate investigation.")
    elif sigma > 2:
        parts.append("This is a moderate severity anomaly that should be monitored.")
    else:
        parts.append("This is a mild anomaly within expected variation range.")

    # Sensor information
    all_sensors = sample.get("all_sensors")
    if all_sensors:
        idx = sample["anomaly_idx"]
        sensor_z = []
        for col, vals in all_sensors.items():
            if idx < len(vals):
                start = max(0, idx - 20)
                end = min(len(vals), idx + 21)
                local = vals[start:end]
                mu = sum(local) / len(local) if local else 0
                std = (sum((v - mu) ** 2 for v in local) / len(local)) ** 0.5 if local else 1
                z = abs(vals[idx] - mu) / max(std, 1e-6)
                sensor_z.append((col, z))
        sensor_z.sort(key=lambda x: x[1], reverse=True)

        if sensor_z:
            top3 = sensor_z[:3]
            top_names = ", ".join(s[0] for s in top3)
            parts.append(
                f"The most anomalous sensors are {top_names}. "
                f"The primary anomalous sensor is {top3[0][0]} "
                f"with a z-score of {top3[0][1]:.1f}."
            )

    # Segment info
    seg_len = sample.get("segment_length", 0)
    if seg_len > 1:
        parts.append(
            f"The anomaly spans {seg_len} consecutive timesteps, "
            f"suggesting a sustained anomalous condition."
        )

    return " ".join(parts)


def compute_bertscore(
    explanations: List[str],
    references: List[str],
    model_type: str = "bert-base-uncased",
    batch_size: int = 32,
) -> Dict[str, List[float]]:
    """Compute BERTScore for a batch of explanations vs references.

    Returns dict with 'precision', 'recall', 'f1' lists.
    Requires: pip install bert-score
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError(
            "bert-score not installed. Run: pip install bert-score"
        )

    P, R, F1 = bert_score_fn(
        explanations,
        references,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
    )

    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
    }


def evaluate_semantic_faithfulness(
    explanation: str,
    sample: Dict,
    model_type: str = "bert-base-uncased",
) -> Dict[str, float]:
    """Evaluate semantic faithfulness of a single explanation using BERTScore.

    Returns dict with precision, recall, f1 scores.
    """
    reference = build_reference_text(sample)
    scores = compute_bertscore(
        [explanation], [reference], model_type=model_type
    )
    return {
        "precision": scores["precision"][0],
        "recall": scores["recall"][0],
        "f1": scores["f1"][0],
        "reference": reference,
    }


def extract_structured_claims(explanation: str) -> Dict[str, Any]:
    """Extract structured claims from an LLM explanation.

    Extracts:
    - severity_level: mentioned severity (low/medium/high/critical)
    - sensor_names: list of sensor names mentioned
    - direction: anomaly direction (increase/decrease/spike/drift)
    - has_root_cause: whether root cause analysis is present
    - has_recommendation: whether actionable recommendations are present
    """
    text = explanation.lower()

    # Severity
    severity = "unknown"
    for level in ["critical", "high", "medium", "moderate", "low", "minor"]:
        if level in text:
            severity = level
            break

    # Direction keywords
    directions = []
    direction_keywords = {
        "increase": ["increase", "rising", "elevated", "spike", "surge", "exceeded"],
        "decrease": ["decrease", "drop", "falling", "low", "below", "decline"],
        "drift": ["drift", "gradual", "slow change", "trend"],
        "oscillation": ["oscillat", "fluctuat", "unstable", "erratic"],
    }
    for direction, keywords in direction_keywords.items():
        if any(kw in text for kw in keywords):
            directions.append(direction)

    # Sensor names (uppercase patterns like FIT101, CT1, cpu_r, etc.)
    sensor_pattern = r'\b[A-Z][A-Za-z0-9_]{2,}\b'
    potential_sensors = re.findall(sensor_pattern, explanation)
    # Filter common non-sensor words
    noise = {"The", "This", "And", "For", "But", "Not", "Has", "Was", "Are",
             "Low", "High", "Medium", "Critical", "Sensor", "Normal", "Anomaly"}
    sensor_names = [s for s in potential_sensors if s not in noise]

    # Root cause
    has_root_cause = any(
        phrase in text
        for phrase in ["root cause", "cause", "reason", "due to", "because", "attributed to"]
    )

    # Recommendation
    has_recommendation = any(
        phrase in text
        for phrase in ["recommend", "action", "should", "suggest", "inspect", "check", "monitor"]
    )

    return {
        "severity_level": severity,
        "sensor_names": sensor_names,
        "directions": directions,
        "has_root_cause": has_root_cause,
        "has_recommendation": has_recommendation,
    }


def evaluate_sensor_exact_match(
    explanation: str,
    sample: Dict,
) -> Dict[str, Any]:
    """Evaluate sensor-level exact match accuracy.

    Unlike keyword faithfulness (vocabulary check), this measures whether
    the LLM correctly identifies the SPECIFIC anomalous sensors.

    Returns:
        top1_exact: True if ground-truth #1 sensor is mentioned
        top3_exact: fraction of ground-truth top-3 sensors mentioned
        wrong_sensor: True if a non-top-3 sensor is mentioned as primary
        direction_correct: True if anomaly direction matches ground truth
    """
    all_sensors = sample.get("all_sensors")
    if not all_sensors:
        return {"top1_exact": None, "top3_exact": None, "wrong_sensor": None,
                "direction_correct": None, "n_sensors": 0}

    idx = sample.get("anomaly_idx", 0)
    sensor_z = []
    sensor_direction = {}
    for col, vals in all_sensors.items():
        if idx >= len(vals):
            continue
        start = max(0, idx - 20)
        end = min(len(vals), idx + 21)
        local = vals[start:end]
        mu = sum(local) / len(local) if local else 0
        std = (sum((v - mu) ** 2 for v in local) / len(local)) ** 0.5 if local else 1
        z = abs(vals[idx] - mu) / max(std, 1e-6)
        sensor_z.append((col, z, vals[idx] - mu))
        sensor_direction[col] = "increase" if vals[idx] > mu else "decrease"

    sensor_z.sort(key=lambda x: x[1], reverse=True)
    if not sensor_z:
        return {"top1_exact": None, "top3_exact": None, "wrong_sensor": None,
                "direction_correct": None, "n_sensors": 0}

    top1_name = sensor_z[0][0]
    top3_names = {s[0] for s in sensor_z[:3]}
    all_sensor_names = set(all_sensors.keys())
    text_lower = explanation.lower()

    # Check which sensors are mentioned
    mentioned = set()
    for s_name in all_sensor_names:
        if s_name.lower() in text_lower:
            mentioned.add(s_name)

    # top1 exact match
    top1_exact = top1_name.lower() in text_lower

    # top3 overlap
    top3_mentioned = mentioned & top3_names
    top3_exact = len(top3_mentioned) / min(3, len(top3_names))

    # Wrong sensor check: is a non-top3 sensor mentioned as primary?
    non_top3_mentioned = mentioned - top3_names
    wrong_sensor = len(non_top3_mentioned) > len(top3_mentioned)

    # Direction accuracy for top-1 sensor
    direction_correct = None
    if top1_exact:
        gt_dir = sensor_direction.get(top1_name, "")
        if gt_dir == "increase":
            direction_correct = any(w in text_lower for w in
                ["increas", "ris", "elevat", "spike", "surge", "high", "above"])
        elif gt_dir == "decrease":
            direction_correct = any(w in text_lower for w in
                ["decreas", "drop", "fall", "low", "below", "declin"])

    return {
        "top1_exact": top1_exact,
        "top3_exact": top3_exact,
        "wrong_sensor": wrong_sensor,
        "direction_correct": direction_correct,
        "n_sensors": len(all_sensor_names),
        "top1_truth": top1_name,
        "mentioned_sensors": list(mentioned),
        "top3_truth": list(top3_names),
    }


def compute_structured_accuracy(
    explanation: str,
    sample: Dict,
) -> Dict[str, Any]:
    """Compare structured claims against ground truth.

    Returns accuracy metrics for each dimension.
    """
    claims = extract_structured_claims(explanation)

    # Severity accuracy
    sigma = sample.get("sigma_deviation", 0)
    expected_severity = (
        "critical" if sigma > 8
        else "high" if sigma > 5
        else "medium" if sigma > 2
        else "low"
    )
    severity_match = claims["severity_level"] in (
        expected_severity,
        {"critical": "high", "high": "medium", "medium": "moderate", "low": "minor"}.get(expected_severity, ""),
    )

    # Sensor accuracy (for multi-sensor)
    all_sensors = sample.get("all_sensors")
    sensor_accuracy = None
    if all_sensors:
        idx = sample["anomaly_idx"]
        sensor_z = []
        for col, vals in all_sensors.items():
            if idx < len(vals):
                start = max(0, idx - 20)
                end = min(len(vals), idx + 21)
                local = vals[start:end]
                mu = sum(local) / len(local) if local else 0
                std = (sum((v - mu) ** 2 for v in local) / len(local)) ** 0.5 if local else 1
                z = abs(vals[idx] - mu) / max(std, 1e-6)
                sensor_z.append((col, z))
        sensor_z.sort(key=lambda x: x[1], reverse=True)
        top3_truth = {s[0] for s in sensor_z[:3]}

        mentioned = {c for c in claims["sensor_names"] if c in all_sensors}
        overlap = len(mentioned & top3_truth)
        sensor_accuracy = overlap / min(3, len(top3_truth)) if top3_truth else 0

    return {
        "severity_match": severity_match,
        "sensor_accuracy": sensor_accuracy,
        "has_root_cause": claims["has_root_cause"],
        "has_recommendation": claims["has_recommendation"],
        "n_sensors_mentioned": len(claims["sensor_names"]),
        "n_directions": len(claims["directions"]),
        "claims": claims,
    }
