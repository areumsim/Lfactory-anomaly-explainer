from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _dir(root: str, name: str) -> str:
    p = os.path.join(root, "SMD", name)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"SMD directory not found: {p}")
    return p


def _list_files(d: str, ext: tuple[str, ...]) -> List[str]:
    files: List[str] = []
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(ext):
            files.append(os.path.join(d, fn))
    if not files:
        raise FileNotFoundError(f"No files with {ext} under {d}")
    return files


def _read_txt_values(path: str, sample_limit: Optional[int] = None) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # SMD txt is effectively CSV with comma or whitespace
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = [p for p in line.split()]
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


def _read_labels(path: str, expected_len: int) -> List[int]:
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # labels are 0/1 separated by comma or whitespace
            tok = line.split(",") if "," in line else line.split()
            for t in tok:
                try:
                    y = int(float(t))
                except Exception:
                    y = 0
                labels.append(1 if y != 0 else 0)
            if len(labels) >= expected_len:
                break
    if len(labels) < expected_len:
        # pad with 0s if label file shorter
        labels += [0] * (expected_len - len(labels))
    elif len(labels) > expected_len:
        labels = labels[:expected_len]
    return labels


def load_one_timeseries(
    root: str,
    split: str = "test",
    sample_limit: Optional[int] = None,
    file_index: Optional[int] = None,
    min_length: int = 0,
) -> Dict[str, Any]:
    split_dir = _dir(root, split if split in ("train", "test") else "test")
    files = _list_files(split_dir, (".txt", ".csv"))
    # filter by min_length if specified
    if min_length > 0:
        filtered: List[str] = []
        for p in files:
            rows = _read_txt_values(p, sample_limit=None)
            if len(rows) >= min_length:
                filtered.append(p)
        if filtered:
            files = filtered
    idx = max(0, int(file_index)) if (file_index is not None and 0 <= int(file_index) < len(files)) else 0
    data_path = files[idx]
    rows = _read_txt_values(data_path, sample_limit=sample_limit)
    c_val = _infer_numeric_column(rows)
    series: List[float] = []
    for r in rows:
        try:
            v = float(r[c_val]) if c_val < len(r) else float("nan")
        except Exception:
            continue
        series.append(v)

    labels: List[int]
    if split.lower() == "test":
        label_dir = _dir(root, "test_label")
        label_path = os.path.join(label_dir, os.path.basename(data_path))
        if not os.path.exists(label_path):
            # fallback: any label file available
            lab_files = _list_files(label_dir, (".txt", ".csv"))
            label_path = lab_files[0]
        labels = _read_labels(label_path, expected_len=len(series))
    else:
        labels = [0] * len(series)

    meta = {
        "dataset": "SMD",
        "file": os.path.basename(data_path),
        "path": data_path,
        "num_points": len(series),
        "label_rate": (sum(labels) / len(labels) if labels else 0.0),
    }
    print(f"[SMD] Loaded {meta['num_points']} points from {meta['file']} (label_rate={meta['label_rate']:.3f})")
    return {"series": series, "labels": labels, "meta": meta}
