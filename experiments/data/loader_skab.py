from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple


def _find_skab_files(root: str) -> List[str]:
    base = os.path.join(root, "SKAB")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"SKAB directory not found under {root}")
    files: List[str] = []
    for sub in ("valve1", "valve2", "other", "anomaly-free"):
        p = os.path.join(base, sub)
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith(".csv"):
                    files.append(os.path.join(p, fn))
    if not files:
        raise FileNotFoundError(f"No CSV files under {base}")
    return files


def _infer_scenario(path: str) -> str:
    """Infer SKAB scenario from parent directory name.
    Examples: valve1, valve2, other, anomaly-free
    """
    base = os.path.basename(os.path.dirname(path))
    return base.lower()


def _infer_numeric_column(header: List[str], rows: List[List[str]]) -> int:
    # pick first column that can parse as float consistently in sample rows
    ncols = len(header)
    for c in range(ncols):
        ok = 0
        total = 0
        for r in rows[:200]:
            if c >= len(r):
                continue
            val = r[c].strip()
            if not val:
                continue
            total += 1
            try:
                float(val)
                ok += 1
            except Exception:
                ok -= 10  # penalize non-numeric
        if total > 0 and ok / max(1, total) > 0.9:
            return c
    # fallback to last column
    return max(0, ncols - 1)


def _find_label_column(header: List[str]) -> Optional[int]:
    cand = ["label", "anomaly", "is_anomaly"]
    lower = [h.strip().lower() for h in header]
    for c in cand:
        if c in lower:
            return lower.index(c)
    return None


def _read_csv_timeseries(path: str, sample_limit: Optional[int] = None) -> Tuple[List[float], List[int], Dict[str, Any]]:
    # Detect delimiter (',' or ';') and read
    with open(path, "r", encoding="utf-8") as f:
        peek = f.readline()
        delim = ";" if (";" in peek and "," not in peek) else ","
        f.seek(0)
        rdr = csv.reader(f, delimiter=delim)
        header = next(rdr, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")
        rows: List[List[str]] = []
        for i, row in enumerate(rdr):
            rows.append(row)
            if sample_limit and len(rows) >= sample_limit:
                break
    c_val = _infer_numeric_column(header, rows)
    c_lab = _find_label_column(header)
    series: List[float] = []
    labels: List[int] = []
    scenario = _infer_scenario(path)
    for r in rows:
        try:
            v = float(r[c_val]) if c_val < len(r) else float("nan")
        except Exception:
            # skip unparsable rows
            continue
        series.append(v)
        if c_lab is not None and c_lab < len(r):
            try:
                y = int(float(r[c_lab]))
                labels.append(1 if y != 0 else 0)
            except Exception:
                labels.append(0)
        else:
            # Scenario-based fallback policy
            if scenario == "anomaly-free":
                labels.append(0)
            else:
                labels.append(0)  # unknown scenario → default 0 (conservative)
    meta = {
        "dataset": "SKAB",
        "file": os.path.basename(path),
        "path": path,
        "scenario": scenario,
        "value_col": header[c_val] if 0 <= c_val < len(header) else str(c_val),
        "label_col": (header[c_lab] if c_lab is not None and 0 <= c_lab < len(header) else None),
        "label_policy": ("column" if c_lab is not None else ("scenario_anomaly_free_zero" if scenario == "anomaly-free" else "fallback_all_zero")),
        "num_points": len(series),
        "label_rate": (sum(labels) / len(labels) if labels else 0.0),
    }
    return series, labels, meta


def load_one_timeseries(
    root: str,
    split: str = "test",
    sample_limit: Optional[int] = None,
    file_index: Optional[int] = None,
    min_length: int = 0,
) -> Dict[str, Any]:
    # SKAB has scenario folders; no strict split; use whatever exists.
    files = _find_skab_files(root)
    candidates: List[str] = []
    if min_length > 0:
        for p in files:
            s, y, m = _read_csv_timeseries(p, sample_limit=None)
            if len(s) >= min_length:
                candidates.append(p)
        if not candidates:
            candidates = files
    else:
        candidates = files
    idx = max(0, int(file_index)) if (file_index is not None and 0 <= int(file_index) < len(candidates)) else 0
    path = candidates[idx]
    series, labels, meta = _read_csv_timeseries(path, sample_limit=sample_limit)
    print(f"[SKAB] Loaded {meta['num_points']} points from {meta['file']} (label_rate={meta['label_rate']:.3f})")
    return {"series": series, "labels": labels, "meta": meta}
