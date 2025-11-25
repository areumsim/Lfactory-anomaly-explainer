from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional


def _join(*parts: str) -> str:
    return os.path.join(*parts)


def _list_files_recursive(root: str, exts: tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for base, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.lower().endswith(exts):
                out.append(_join(base, fn))
    if not out:
        raise FileNotFoundError(f"No files with {exts} under {root}")
    return out


def _read_csv_like(path: str, sample_limit: Optional[int] = None) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            rows.append(parts)
            if sample_limit and len(rows) >= sample_limit:
                break
    return rows


def _infer_numeric_column(rows: List[List[str]]) -> int:
    if not rows:
        return 0
    ncols = max(len(r) for r in rows)
    best_c = 0
    best_ok = -1
    for c in range(ncols):
        ok = 0
        total = 0
        for r in rows[:500]:
            if c >= len(r):
                continue
            val = r[c]
            if val == "":
                continue
            total += 1
            try:
                float(val)
                ok += 1
            except Exception:
                ok -= 5
        if total > 0 and ok > best_ok:
            best_ok = ok
            best_c = c
    return best_c


def _labels_from_json(path: str, length: int, scheme: str) -> List[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return [0] * length
    # Try common shapes: list of ints, or dict with "labels"
    arr: List[int]
    if isinstance(obj, list):
        arr = [int(x) for x in obj if isinstance(x, (int, float))]
    elif isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
        arr = [int(x) for x in obj["labels"] if isinstance(x, (int, float))]
    else:
        return [0] * length
    # Adjust to requested scheme
    if scheme == "binary":
        arr = [1 if int(v) > 0 else 0 for v in arr]
    else:  # risk4 passthrough 0..3 but clip
        arr = [max(0, min(3, int(v))) for v in arr]
    if len(arr) < length:
        arr += [0] * (length - len(arr))
    return arr[:length]


def load_one_timeseries(
    root: str,
    split: str = "Validation",
    label_scheme: str = "binary",
    sample_limit: Optional[int] = None,
    file_index: Optional[int] = None,
    min_length: int = 0,
) -> Dict[str, Any]:
    base = _join(root, "manufacturing_transport_71802")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"AIHub71802 directory not found under {root}")
    split_dir = _join(base, split if split in ("Training", "Validation") else "Validation")
    data_dir = _join(split_dir, "data")
    if not os.path.isdir(data_dir):
        # fallback: search split root
        data_dir = split_dir
    data_files = _list_files_recursive(data_dir, (".csv", ".txt"))
    if min_length > 0:
        filtered: List[str] = []
        for p in data_files:
            rows = _read_csv_like(p, sample_limit=None)
            if len(rows) >= min_length:
                filtered.append(p)
        if filtered:
            data_files = filtered
    idx = max(0, int(file_index)) if (file_index is not None and 0 <= int(file_index) < len(data_files)) else 0
    data_path = data_files[idx]
    rows = _read_csv_like(data_path, sample_limit=sample_limit)
    c_val = _infer_numeric_column(rows)
    series: List[float] = []
    for r in rows:
        try:
            v = float(r[c_val]) if c_val < len(r) else float("nan")
        except Exception:
            continue
        series.append(v)

    # Try to find a label file nearby
    labels: List[int] = [0] * len(series)
    label_dir = _join(split_dir, "label")
    candidates: List[str] = []
    if os.path.isdir(label_dir):
        candidates = _list_files_recursive(label_dir, (".json", ".csv", ".txt"))
    if candidates:
        # Prefer json if present
        json_cands = [p for p in candidates if p.lower().endswith(".json")]
        if json_cands:
            labels = _labels_from_json(json_cands[0], len(series), label_scheme)
        else:
            # read csv/txt first numeric column as risk score
            lab_rows = _read_csv_like(candidates[0])
            c_lab = _infer_numeric_column(lab_rows)
            arr: List[int] = []
            for r in lab_rows:
                try:
                    arr.append(int(float(r[c_lab])))
                except Exception:
                    arr.append(0)
            if label_scheme == "binary":
                labels = [1 if v > 0 else 0 for v in arr[: len(series)]]
            else:
                labels = [max(0, min(3, v)) for v in arr[: len(series)]]

    meta = {
        "dataset": "AIHub71802",
        "file": os.path.basename(data_path),
        "path": data_path,
        "num_points": len(series),
        "label_rate": (sum(labels) / len(labels) if labels else 0.0) if label_scheme == "binary" else None,
        "label_scheme": label_scheme,
        "split": split_dir.split(os.sep)[-1],
    }
    print(f"[AIHub71802] Loaded {meta['num_points']} points from {meta['file']} (scheme={label_scheme})")
    return {"series": series, "labels": labels, "meta": meta}
