"""SWaT (Secure Water Treatment) dataset loader.

Data structure (iTrust / SUTD):
- SWaT_Dataset_Normal_v1.csv: 7 days normal operation (~496,800 rows)
- SWaT_Dataset_Attack_v0.csv: 4 days with 36 attack scenarios (~449,919 rows)
- Each row: Timestamp, 51 sensor/actuator columns, Normal/Attack label
- Sensors span 6 sub-processes (P1-P6): flow, level, pressure, pH, ORP, etc.
- Sampling rate: 1 second

Sensor naming convention:
- FIT: Flow Indicator Transmitter
- LIT: Level Indicator Transmitter
- AIT: Analyzation Indicator Transmitter (pH, ORP, conductivity)
- PIT: Pressure Indicator Transmitter
- DPIT: Differential Pressure Indicator Transmitter
- MV: Motorized Valve (0/1/2 states)
- P: Pump (0/1/2 states)
- UV: UV lamp (0/1 states)

Loading:
1. Read the attack CSV (or normal CSV for training split)
2. Parse all 51 sensor columns into all_sensors dict
3. Use primary_sensor (default: first continuous sensor) for univariate series
4. Parse label column ("Normal" / "Attack" or "A ttack")
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


# All 51 SWaT sensor/actuator names in standard column order.
# Continuous sensors (transmitters) are listed first, then actuators.
SWAT_SENSOR_NAMES: List[str] = [
    # Process 1: Raw Water
    "FIT101", "LIT101", "MV101", "P101", "P102",
    # Process 2: Pre-treatment (Chemical Dosing)
    "AIT201", "AIT202", "AIT203", "FIT201", "MV201", "P201", "P202",
    "P203", "P204", "P205", "P206",
    # Process 3: Ultrafiltration
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304",
    "P301", "P302",
    # Process 4: De-Chlorination (UV + Chlorine)
    "AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402",
    "P403", "P404", "UV401",
    # Process 5: Reverse Osmosis
    "AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502",
    "FIT503", "FIT504", "MV501", "MV502", "MV503", "MV504",
    "P501", "P502", "PIT501", "PIT502", "PIT503",
    # Process 6: Backwash
    "FIT601", "P601", "P602", "P603",
]


def _find_csv(root: str, split: str) -> str:
    """Find the SWaT CSV file for the given split."""
    # Common file names used in SWaT distributions
    candidates_attack = [
        "merged.csv",  # Kaggle version: combined normal+attack
        "SWaT_Dataset_Attack_v0.csv",
        "SWaT_Dataset_Attack_v1.csv",
        "swat_attack.csv",
        "SWaT_A1_A2.csv",
        "Physical_A1_A2.csv",
        "attack.csv",
        "test.csv",
    ]
    candidates_normal = [
        "SWaT_Dataset_Normal_v1.csv",
        "SWaT_Dataset_Normal_v0.csv",
        "swat_normal.csv",
        "SWaT_Normal.csv",
        "Physical_Normal.csv",
        "normal.csv",
        "train.csv",
    ]
    candidates = candidates_attack if split in ("test", "attack") else candidates_normal

    # Direct file match
    for fn in candidates:
        p = os.path.join(root, fn)
        if os.path.isfile(p):
            return p

    # Recursive search (one level)
    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        if os.path.isdir(sub_path):
            for fn in candidates:
                p = os.path.join(sub_path, fn)
                if os.path.isfile(p):
                    return p

    # Fallback: any CSV in root
    csvs = [f for f in os.listdir(root) if f.lower().endswith(".csv")]
    if csvs:
        # Prefer files with 'attack' or 'test' in name for test split
        keyword = "attack" if split in ("test", "attack") else "normal"
        for c in csvs:
            if keyword in c.lower():
                return os.path.join(root, c)
        return os.path.join(root, csvs[0])

    raise FileNotFoundError(
        f"No SWaT CSV found in {root} for split={split}. "
        f"Expected one of: {candidates}"
    )


def _parse_label(val: str) -> int:
    """Parse SWaT label: 'Normal' -> 0, 'Attack' or 'A ttack' -> 1."""
    v = val.strip().lower().replace(" ", "")
    if v in ("attack", "attck", "a ttack"):
        return 1
    return 0


def _match_columns(header: List[str]) -> Dict[str, int]:
    """Match header columns to known SWaT sensor names.

    Returns a dict mapping sensor_name -> column_index.
    Handles whitespace and case variations in headers.
    """
    # Normalize header
    normalized = [h.strip().replace(" ", "") for h in header]
    col_map: Dict[str, int] = {}
    for sensor in SWAT_SENSOR_NAMES:
        sensor_clean = sensor.strip().replace(" ", "")
        for i, h in enumerate(normalized):
            if h.upper() == sensor_clean.upper():
                col_map[sensor] = i
                break
    return col_map


def load_one_timeseries(
    root: str,
    split: str = "test",
    sample_limit: Optional[int] = None,
    file_index: Optional[int] = None,
    min_length: int = 0,
) -> Dict[str, Any]:
    """Load SWaT time-series data.

    Args:
        root: Path to SWaT data directory.
        split: "test" (attack) or "train" (normal).
        sample_limit: Max rows to read.
        file_index: Ignored (SWaT has a single file per split).
        min_length: Minimum series length (rows).

    Returns:
        {"series": List[float], "labels": List[int], "meta": dict}
        meta includes "all_sensors" dict mapping sensor names to value lists.
    """
    csv_path = _find_csv(root, split)

    # Read CSV
    header: List[str] = []
    rows_data: List[List[str]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]

            # Detect header row (contains known sensor names)
            if not header:
                # Check if this row has sensor-like names
                upper_parts = [p.upper().replace(" ", "") for p in parts]
                if any(s.upper() in upper_parts for s in SWAT_SENSOR_NAMES[:5]):
                    header = parts
                    continue
                # Also detect if first row has "Timestamp" header
                if any("timestamp" in p.lower() for p in parts):
                    header = parts
                    continue
                # Skip non-data rows (e.g., blank or metadata)
                # Try to parse as data if no header yet — but first need header
                continue

            rows_data.append(parts)
            if sample_limit and len(rows_data) >= sample_limit:
                break

    if not header:
        raise ValueError(f"Could not find header row in {csv_path}")

    # Match sensor columns
    col_map = _match_columns(header)
    if not col_map:
        raise ValueError(
            f"No SWaT sensor columns matched in header: {header[:10]}..."
        )

    # Find label column
    label_col = -1
    for i, h in enumerate(header):
        h_clean = h.strip().lower().replace(" ", "")
        if h_clean in ("normal/attack", "label", "attack", "normal_attack", "class"):
            label_col = i
            break
    # Fallback: last column
    if label_col < 0:
        label_col = len(header) - 1

    # Build sensor data and labels
    all_sensors: Dict[str, List[float]] = {s: [] for s in col_map}
    labels: List[int] = []
    primary_sensor_name = next(
        (s for s in SWAT_SENSOR_NAMES if s in col_map and s.startswith("FIT")),
        list(col_map.keys())[0] if col_map else "FIT101",
    )

    for parts in rows_data:
        # Parse sensor values
        for sensor, ci in col_map.items():
            try:
                v = float(parts[ci]) if ci < len(parts) else float("nan")
            except (ValueError, IndexError):
                v = float("nan")
            all_sensors[sensor].append(v)

        # Parse label
        try:
            lbl = _parse_label(parts[label_col]) if label_col < len(parts) else 0
        except (ValueError, IndexError):
            lbl = 0
        labels.append(lbl)

    # Primary series (for univariate compatibility)
    series: List[float] = all_sensors.get(primary_sensor_name, [])
    n_points = len(series)

    if min_length > 0 and n_points < min_length:
        raise ValueError(
            f"SWaT series too short: {n_points} < {min_length}"
        )

    label_rate = sum(labels) / len(labels) if labels else 0.0

    meta: Dict[str, Any] = {
        "dataset": "SWaT",
        "file": os.path.basename(csv_path),
        "path": csv_path,
        "num_points": n_points,
        "num_sensors": len(col_map),
        "sensor_columns": list(col_map.keys()),
        "primary_sensor": primary_sensor_name,
        "label_rate": label_rate,
        "all_sensors": all_sensors,
    }

    import sys as _sys
    print(
        f"[SWaT] Loaded {n_points} points × {len(col_map)} sensors "
        f"from {meta['file']} (label_rate={label_rate:.3f})",
        file=_sys.stderr,
    )
    return {"series": series, "labels": labels, "meta": meta}
