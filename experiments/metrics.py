"""Metrics for anomaly detection.

Research context:
- Point-wise 분류 지표(Precision/Recall/F1/Accuracy)와 곡선(AUC-ROC/PR)을 기본으로
  제공하며, 시계열 이벤트 기반 지표(Detection Delay, Lead Time, Point-adjust)
  를 보강합니다. 이벤트 라벨은 0/1의 연속 구간으로부터 추출합니다.

API:
- binary_metrics(y_true, y_pred) -> dict
 - segments_from_labels(labels) -> List[(start,end)]
 - event_metrics_from_segments(segments, preds) -> dict
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple


def _confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[int, int, int, int]:
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


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# --- Event-based utilities (Point-adjust, Delay/Lead Time) ---
def segments_from_labels(labels: Iterable[int]) -> list[tuple[int, int]]:
    """Extract contiguous positive segments (start,end inclusive) from 0/1 labels.

    Note: Materializes `labels` to a list to avoid iterator consumption issues.
    """
    arr = [int(x) for x in labels]
    segs: list[tuple[int, int]] = []
    start = -1
    for i, v in enumerate(arr):
        if v == 1 and start == -1:
            start = i
        elif v == 0 and start != -1:
            segs.append((start, i - 1))
            start = -1
    if start != -1:
        segs.append((start, len(arr) - 1))
    return segs


def _segments_from_preds(preds: Iterable[int]) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start = -1
    arr = [int(x) for x in preds]
    for i, v in enumerate(arr):
        if v == 1 and start == -1:
            start = i
        elif v == 0 and start != -1:
            segs.append((start, i - 1))
            start = -1
    if start != -1:
        segs.append((start, len(arr) - 1))
    return segs


def event_metrics_from_segments(segments: list[tuple[int, int]], preds: Iterable[int]) -> Dict[str, float]:
    """Compute simple event-based metrics.

    - point-adjusted detection: an event is detected if any pred==1 within the segment
    - detection delay: earliest_pred_index - segment_start (only for detected)
    - lead time: segment_end - earliest_pred_index + 1 (only for detected)
      Note: current implementation approximates the event "peak" by segment_end.
            This matches the documentation's interim policy and may be refined
            later to use domain-specific peak definitions.
    - event-precision/recall/f1 using predicted segments vs true segments overlap
    """
    pred_arr = [int(p) for p in preds]
    n_events = len(segments)
    if n_events == 0:
        return {
            "num_events": 0.0,
            "detected_events": 0.0,
            "event_recall": 0.0,
            "event_precision": 0.0,
            "event_f1": 0.0,
            "mean_detection_delay": 0.0,
            "mean_lead_time": 0.0,
        }
    detected = 0
    delays: list[float] = []  # 이벤트별 최초 탐지 시점 - 시작 시점
    leads: list[float] = []   # 이벤트 피크(여기서는 구간 끝 기준 근사)까지 남은 시간
    for (a, b) in segments:
        earliest = -1
        a0 = max(0, int(a))
        b0 = min(len(pred_arr) - 1, int(b))
        for i in range(a0, b0 + 1):
            if pred_arr[i] == 1:
                earliest = i
                break
        if earliest >= 0:
            detected += 1
            delays.append(float(earliest - a0))
            leads.append(float(max(0, b0 - earliest + 1)))

    # Event-level precision via predicted segments overlap
    pred_segs = _segments_from_preds(pred_arr)
    # Count predicted segments that overlap any true segment
    hit_pred = 0
    for (pa, pb) in pred_segs:
        for (ta, tb) in segments:
            if not (pb < ta or pa > tb):
                hit_pred += 1
                break
    event_precision = hit_pred / len(pred_segs) if pred_segs else 0.0
    event_recall = detected / n_events if n_events else 0.0
    event_f1 = (2 * event_precision * event_recall / (event_precision + event_recall)) if (event_precision + event_recall) else 0.0

    mean_delay = sum(delays) / len(delays) if delays else 0.0
    mean_lead = sum(leads) / len(leads) if leads else 0.0
    # median lead computed similarly to delay; safe for empty lists
    med_lead = (sorted(leads)[len(leads)//2] if leads else 0.0)
    return {
        "num_events": float(n_events),
        "detected_events": float(detected),
        "event_recall": float(event_recall),
        "event_precision": float(event_precision),
        "event_f1": float(event_f1),
        "mean_detection_delay": float(mean_delay),
        "median_detection_delay": float(sorted(delays)[len(delays)//2] if delays else 0.0),
        "mean_lead_time": float(mean_lead),
        "median_lead_time": float(med_lead),
    }
