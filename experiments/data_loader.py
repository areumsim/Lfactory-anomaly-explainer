"""Data Loader Module

Research context:
- Provides reproducible synthetic time-series suitable for rapid anomaly
  detection prototyping without external dependencies.
- Designed to be extended later to support public benchmarks (NAB, SMD,
  SMAP/MSL) and AI Hub datasets per the PRD.

Exports:
- load(dataset: str, **kwargs) -> dict with keys: series, labels, meta
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import random


@dataclass
class SyntheticConfig:
    length: int = 2000
    anomaly_rate: float = 0.02  # fraction of points marked anomalous
    noise_std: float = 0.2
    seed: int = 42


def _generate_sine_baseline(n: int, noise_std: float, rng: random.Random) -> List[float]:
    series: List[float] = []
    for t in range(n):
        # Smooth seasonal pattern + small trend + noise
        value = 2.0 * math.sin(2 * math.pi * t / 100.0) + 0.001 * t
        value += rng.gauss(0.0, noise_std)
        series.append(value)
    return series


def _inject_anomalies(series: List[float], anomaly_rate: float, rng: random.Random) -> Tuple[List[int], List[str]]:
    n = len(series)
    k = max(1, int(n * anomaly_rate))
    labels = [0] * n
    types = [""] * n

    # Choose disjoint anomaly events of small durations where applicable
    candidates = list(range(50, n - 50))
    rng.shuffle(candidates)

    injected = 0
    i = 0
    while injected < k and i < len(candidates):
        idx = candidates[i]
        i += 1
        atype = rng.choice(["spike", "step", "drift"])  # simple, interpretable
        if atype == "spike":
            magnitude = rng.uniform(3.0, 6.0)
            series[idx] += magnitude if rng.random() < 0.5 else -magnitude
            labels[idx] = 1
            types[idx] = atype
            injected += 1
        elif atype == "step":
            # Step change lasting ~10-30 points
            duration = rng.randint(10, 30)
            delta = rng.uniform(1.5, 3.0) * (1 if rng.random() < 0.5 else -1)
            for j in range(idx, min(n, idx + duration)):
                series[j] += delta
                labels[j] = 1
                types[j] = atype
            injected += duration
        else:  # drift
            # Slow linear drift over ~30-60 points
            duration = rng.randint(30, 60)
            slope = rng.uniform(0.03, 0.08) * (1 if rng.random() < 0.5 else -1)
            for j in range(idx, min(n, idx + duration)):
                series[j] += slope * (j - idx)
                labels[j] = 1
                types[j] = atype
            injected += duration

    return labels, types


def load(dataset: str, **kwargs) -> Dict:
    """Load dataset by name.

    Supported datasets (initial loop):
    - "synthetic": configurable sine baseline with injected anomalies.

    Returns dict with keys:
    - series: List[float]
    - labels: List[int] (0/1)
    - meta: Dict[str, any]
    """
    name = dataset.lower()
    if name == "synthetic":
        cfg = SyntheticConfig(
            length=int(kwargs.get("length", 2000)),
            anomaly_rate=float(kwargs.get("anomaly_rate", 0.02)),
            noise_std=float(kwargs.get("noise_std", 0.2)),
            seed=int(kwargs.get("seed", 42)),
        )
        rng = random.Random(cfg.seed)
        series = _generate_sine_baseline(cfg.length, cfg.noise_std, rng)
        labels, types = _inject_anomalies(series, cfg.anomaly_rate, rng)
        return {
            "series": series,
            "labels": labels,
            "meta": {
                "dataset": "synthetic",
                "config": cfg.__dict__,
                "anomaly_types": types,
            },
        }

    raise ValueError(f"Unsupported dataset: {dataset}")

