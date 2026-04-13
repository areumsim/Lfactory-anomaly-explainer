"""AIHub 71802 Manufacturing/Transport dataset loader.

데이터 구조:
- Training/Validation 각각 data/ 와 label/ 디렉토리
- data/ 에 ZIP 파일들 (VS_agv_17_agv17_0902_1039.zip 등)
- 각 ZIP 내부: 1초 간격 센서 CSV (~300개) + thermal .bin 파일
- CSV 형식: NTC,PM1.0,PM2.5,PM10,CT1,CT2,CT3,CT4 (헤더 + 1행)
- label/ 에 대응하는 ZIP (VL_agv_17_...) 안에 JSON 라벨 파일
- JSON 라벨: annotations.tagging.state = "0" 또는 "1"

로딩 방식:
1. data ZIP 하나 = 1개 recording session (시계열)
2. ZIP 내 CSV들을 타임스탬프(파일명) 순 정렬 → concat
3. 대응하는 label ZIP에서 JSON 라벨 매칭
"""
from __future__ import annotations

import os
import json
import sys
import zipfile
import re
from typing import Any, Dict, List, Optional, Tuple


# Sensor columns in order
SENSOR_COLS = ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"]


def _extract_timestamp(filename: str) -> str:
    """Extract sortable timestamp from filename like 'oht17_0826_145422.csv'."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    # Extract digits for sorting: e.g., oht17_0826_145422 -> 0826145422
    digits = re.findall(r'\d+', stem)
    return "_".join(digits)


def _parse_csv_row(line: str) -> Optional[Dict[str, float]]:
    """Parse a single CSV data row into sensor values."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < len(SENSOR_COLS):
        return None
    values = {}
    for i, col in enumerate(SENSOR_COLS):
        try:
            values[col] = float(parts[i])
        except (ValueError, IndexError):
            values[col] = 0.0
    return values


def _parse_label_json(content: bytes) -> int:
    """Extract binary label from JSON label file."""
    try:
        obj = json.loads(content.decode("utf-8"))
        annotations = obj.get("annotations", [])
        if annotations:
            tagging = annotations[0].get("tagging", [])
            if tagging:
                state = tagging[0].get("state", "0")
                return 1 if str(state) != "0" else 0
    except Exception:
        pass
    return 0


def _session_id_from_zip(zip_path: str) -> str:
    """Extract session identifier from ZIP filename.

    VS_agv_17_agv17_0902_1039.zip -> agv_17_agv17_0902_1039
    VL_agv_17_agv17_0902_1039.zip -> agv_17_agv17_0902_1039
    """
    basename = os.path.splitext(os.path.basename(zip_path))[0]
    # Remove VS_ or VL_ prefix
    if basename.startswith(("VS_", "VL_")):
        return basename[3:]
    return basename


def _load_session_from_zip(
    data_zip_path: str,
    label_zip_path: Optional[str] = None,
    label_scheme: str = "binary",
    primary_sensor: Optional[str] = None,
) -> Tuple[List[float], List[int], Dict[str, Any]]:
    """Load one recording session from a data ZIP + optional label ZIP.

    Returns (series, labels, meta) where series is the primary sensor values.
    """
    # Read data CSVs from ZIP
    timestamped_rows: List[Tuple[str, Dict[str, float]]] = []

    with zipfile.ZipFile(data_zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for csv_name in csv_names:
            with zf.open(csv_name) as f:
                lines = f.read().decode("utf-8", errors="replace").strip().split("\n")
                # Skip header, parse data row
                for line in lines[1:]:  # skip header
                    line = line.strip()
                    if not line:
                        continue
                    row = _parse_csv_row(line)
                    if row is not None:
                        ts = _extract_timestamp(csv_name)
                        timestamped_rows.append((ts, row))

    # Sort by timestamp
    timestamped_rows.sort(key=lambda x: x[0])

    if not timestamped_rows:
        return [], [], {"error": "no_data"}

    # Build multi-sensor data
    all_sensors: Dict[str, List[float]] = {col: [] for col in SENSOR_COLS}
    for _, row in timestamped_rows:
        for col in SENSOR_COLS:
            all_sensors[col].append(row.get(col, 0.0))

    # Primary sensor for single-series interface
    if primary_sensor and primary_sensor in SENSOR_COLS:
        series = all_sensors[primary_sensor]
    else:
        # Default: CT1 (overcurrent sensor, most likely to show anomalies)
        series = all_sensors["CT1"]

    n_points = len(series)

    # Read labels from label ZIP
    labels: List[int] = [0] * n_points
    if label_zip_path and os.path.isfile(label_zip_path):
        label_map: Dict[str, int] = {}
        with zipfile.ZipFile(label_zip_path, "r") as zf:
            json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            for jn in json_names:
                with zf.open(jn) as f:
                    label_val = _parse_label_json(f.read())
                    ts = _extract_timestamp(jn)
                    label_map[ts] = label_val

        # Match labels to data by timestamp
        for i, (ts, _) in enumerate(timestamped_rows):
            if ts in label_map:
                labels[i] = label_map[ts]

    # Meta info
    session_id = _session_id_from_zip(data_zip_path)
    label_rate = sum(labels) / len(labels) if labels else 0.0
    meta = {
        "dataset": "AIHub71802",
        "session_id": session_id,
        "file": os.path.basename(data_zip_path),
        "path": data_zip_path,
        "num_points": n_points,
        "num_sensors": len(SENSOR_COLS),
        "sensor_columns": SENSOR_COLS,
        "primary_sensor": primary_sensor or "CT1",
        "label_rate": label_rate if label_scheme == "binary" else None,
        "label_scheme": label_scheme,
        "all_sensors": all_sensors,
    }

    return series, labels, meta


def _find_label_zip(data_zip_path: str, label_dir: str) -> Optional[str]:
    """Find corresponding label ZIP for a data ZIP."""
    session_id = _session_id_from_zip(data_zip_path)
    # Label ZIPs have VL_ prefix instead of VS_
    expected_name = f"VL_{session_id}.zip"
    expected_path = os.path.join(label_dir, expected_name)
    if os.path.isfile(expected_path):
        return expected_path
    # Fallback: search for matching session_id
    if os.path.isdir(label_dir):
        for fn in os.listdir(label_dir):
            if fn.endswith(".zip") and session_id in fn:
                return os.path.join(label_dir, fn)
    return None


def _list_data_zips(data_dir: str) -> List[str]:
    """List all data ZIP files in directory."""
    if not os.path.isdir(data_dir):
        return []
    zips = [
        os.path.join(data_dir, fn)
        for fn in sorted(os.listdir(data_dir))
        if fn.lower().endswith(".zip") and fn.startswith("VS_")
    ]
    return zips


def load_one_timeseries(
    root: str,
    split: str = "Validation",
    label_scheme: str = "binary",
    sample_limit: Optional[int] = None,
    file_index: Optional[int] = None,
    min_length: int = 0,
    primary_sensor: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one recording session as a time series.

    Each ZIP file in data/ represents one recording session (~300 points, 1Hz).
    CSVs inside the ZIP are sorted by timestamp and concatenated.
    """
    # Resolve base directory
    if os.path.isdir(root):
        base = root
        sub = os.path.join(root, "manufacturing_transport_71802")
        if os.path.isdir(sub):
            base = sub
    else:
        raise FileNotFoundError(f"AIHub71802 directory not found: {root}")

    split_name = split if split in ("Training", "Validation") else "Validation"
    split_dir = os.path.join(base, split_name)
    data_dir = os.path.join(split_dir, "data")
    label_dir = os.path.join(split_dir, "label")

    # List available data ZIPs
    data_zips = _list_data_zips(data_dir)
    if not data_zips:
        raise FileNotFoundError(f"No data ZIP files found in {data_dir}")

    # Filter by min_length if needed
    if min_length > 0:
        filtered = []
        for zp in data_zips:
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    csv_count = sum(1 for n in zf.namelist() if n.lower().endswith(".csv"))
                    if csv_count >= min_length:
                        filtered.append(zp)
            except Exception:
                continue
        if filtered:
            data_zips = filtered

    # Select file by index
    idx = 0
    if file_index is not None and 0 <= file_index < len(data_zips):
        idx = file_index

    data_zip_path = data_zips[idx]
    label_zip_path = _find_label_zip(data_zip_path, label_dir)

    series, labels, meta = _load_session_from_zip(
        data_zip_path, label_zip_path, label_scheme, primary_sensor
    )

    if sample_limit and sample_limit > 0 and len(series) > sample_limit:
        series = series[:sample_limit]
        labels = labels[:sample_limit]
        meta["num_points"] = len(series)

    if len(series) < 10:
        print(f"[AIHub71802] WARN: Very short series ({len(series)} points) from {meta.get('file', '?')}", file=sys.stderr)

    label_info = f"label_rate={meta.get('label_rate', 0):.3f}" if label_scheme == "binary" else f"scheme={label_scheme}"
    print(f"[AIHub71802] Loaded {meta['num_points']} points from {meta['file']} ({label_info})", file=sys.stderr)

    meta["num_sessions_available"] = len(data_zips)

    return {"series": series, "labels": labels, "meta": meta}
