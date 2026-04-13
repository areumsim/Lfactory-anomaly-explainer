from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


# SMD (Server Machine Dataset) sensor names by index (0-37).
# Based on Su et al., 2019 — 38 server monitoring metrics.
# Exact metric names are not published; we use descriptive names
# following the convention: {category}_{index}.
SMD_SENSOR_NAMES: List[str] = [
    "cpu_r",          # 0: CPU rate
    "load_1",         # 1: load average 1min
    "load_5",         # 2: load average 5min
    "load_15",        # 3: load average 15min
    "mem_shmem",      # 4: shared memory
    "mem_u",          # 5: memory utilization
    "mem_u_max",      # 6: memory utilization max
    "swap_u",         # 7: swap utilization
    "in_bps",         # 8: network in bytes/s
    "in_err",         # 9: network in errors
    "in_pps",         # 10: network in packets/s
    "net_u_in",       # 11: network utilization in
    "net_u_out",      # 12: network utilization out
    "out_bps",        # 13: network out bytes/s
    "out_err",        # 14: network out errors
    "out_pps",        # 15: network out packets/s
    "tcp_tw",         # 16: TCP time-wait
    "tcp_use",        # 17: TCP connections in use
    "active_opens",   # 18: TCP active opens
    "curr_estab",     # 19: TCP currently established
    "in_errs_tcp",    # 20: TCP in errors
    "in_segs",        # 21: TCP in segments
    "listen_ovfl",    # 22: TCP listen overflows
    "out_segs",       # 23: TCP out segments
    "retrans_segs",   # 24: TCP retransmitted segments
    "io_read_bps",    # 25: disk I/O read bytes/s
    "io_read_sps",    # 26: disk I/O read ops/s
    "io_write_bps",   # 27: disk I/O write bytes/s
    "io_write_sps",   # 28: disk I/O write ops/s
    "disk_u_max",     # 29: disk utilization max
    "req_ql",         # 30: request queue length
    "pg_in",          # 31: page in
    "pg_out",         # 32: page out
    "pg_faults",      # 33: page faults
    "kb_a",           # 34: kB available
    "kb_c",           # 35: kB cached
    "kb_f",           # 36: kB free
    "kb_i",           # 37: kB inactive
]


def _dir(root: str, name: str) -> str:
    # Try with SMD subdirectory first, then without (for pre-resolved paths)
    p = os.path.join(root, "SMD", name)
    if os.path.isdir(p):
        return p
    p2 = os.path.join(root, name)
    if os.path.isdir(p2):
        return p2
    raise FileNotFoundError(f"SMD directory not found: {p}")


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

    # Build all_sensors dict: sensor_name -> list of float values
    ncols = max((len(r) for r in rows), default=0)
    num_sensors = min(ncols, len(SMD_SENSOR_NAMES))
    all_sensors: Dict[str, List[float]] = {}
    if num_sensors > 1:
        # Initialize lists
        sensor_names = SMD_SENSOR_NAMES[:num_sensors]
        for name in sensor_names:
            all_sensors[name] = []
        for r in rows:
            for ci in range(num_sensors):
                try:
                    v = float(r[ci]) if ci < len(r) else float("nan")
                except Exception:
                    v = float("nan")
                all_sensors[sensor_names[ci]].append(v)

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

    meta: Dict[str, Any] = {
        "dataset": "SMD",
        "file": os.path.basename(data_path),
        "path": data_path,
        "num_points": len(series),
        "num_sensors": num_sensors,
        "sensor_columns": SMD_SENSOR_NAMES[:num_sensors],
        "label_rate": (sum(labels) / len(labels) if labels else 0.0),
    }
    if all_sensors:
        meta["all_sensors"] = all_sensors
    import sys as _sys
    print(f"[SMD] Loaded {meta['num_points']} points × {num_sensors} sensors from {meta['file']} (label_rate={meta['label_rate']:.3f})", file=_sys.stderr)
    return {"series": series, "labels": labels, "meta": meta}
