"""Data subpackage: dataset routing and loaders.

Exposes a minimal, dependency-free interface to load a single
time-series with binary labels from supported datasets so the
baseline detector can run without pandas/numpy.

Functions:
- load_timeseries(name, data_root, split, label_scheme, sample_limit)
"""
from __future__ import annotations

from .data_router import load_timeseries  # re-export for convenience

__all__ = ["load_timeseries"]

