"""Dataset normalization utilities with Parquet (CSV fallback).

연구 맥락:
- 각 데이터셋(SKAB, SMD)을 공통 스키마로 정규화하여 캐시(Parquet)로 저장합니다.
- 의존성 최소화를 위해 pandas/pyarrow가 없으면 동일 경로에 CSV로 저장합니다.

공통 스키마(열):
- timestamp: int (0..N-1, 원본 타임스탬프 부재 시 인덱스 사용)
- value: float (단일 센서 값; 멀티채널은 추후 확장)
- label: int (0/1)
- dataset: str (예: "SKAB", "SMD")
- file: str (원본 파일명)
- machine_id: str (기계/시나리오 식별; 불명확 시 파일명 기반)

출력 경로:
- CODE_ROOT 기준 상대 경로 `parquet/<ds>/...`
- Parquet 선호, 미지원 시 같은 이름으로 `.csv` 저장

사용 예시:
  from experiments.data.normalize import normalize_skab, normalize_smd
  normalize_skab(data_root="/workspace/data1_arsim/LFactory_d")
  normalize_smd(data_root="/workspace/data1_arsim/LFactory_d")
"""
from __future__ import annotations

import csv
import os
from typing import List, Tuple

from . import loader_skab, loader_smd, loader_aihub_71802


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _try_write_parquet(path: str, rows: List[Tuple[int, float, int, str, str, str]]) -> str:
    """rows: (timestamp, value, label, dataset, file, machine_id)

    Returns the final written path (parquet or csv).
    """
    # Prefer Parquet if pandas available; else CSV
    dirpath = os.path.dirname(path)
    _ensure_dir(dirpath or ".")
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows, columns=["timestamp", "value", "label", "dataset", "file", "machine_id"])
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv_path = os.path.splitext(path)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "value", "label", "dataset", "file", "machine_id"])
            for r in rows:
                w.writerow(list(r))
        return csv_path


def normalize_skab(data_root: str, out_root: str = "parquet") -> List[str]:
    """SKAB CSV들을 공통 스키마로 정규화하여 저장합니다.

    반환: 생성된 파일 경로 리스트(parquet 또는 csv)
    """
    files = loader_skab._find_skab_files(data_root)
    written: List[str] = []
    for path in files:
        series, labels, meta = loader_skab._read_csv_timeseries(path)
        orig_n = int(meta.get("num_points", len(series)))
        rows: List[Tuple[int, float, int, str, str, str]] = []
        machine_id = (meta.get("file") or os.path.basename(path)).split(".")[0]
        for i, (v, y) in enumerate(zip(series, labels)):
            rows.append((i, float(v), int(y), "SKAB", os.path.basename(path), machine_id))
        rel = os.path.join(out_root, "skab")
        _ensure_dir(rel)
        out_path = os.path.join(rel, f"{machine_id}.parquet")
        outp = _try_write_parquet(out_path, rows)
        written.append(outp)
        # Integrity check: sample count ratio
        ratio = (len(rows) / max(1, orig_n)) if orig_n else 1.0
        if ratio < 0.95:
            print(f"[WARN] SKAB normalize: kept {ratio:.3%} of rows for {os.path.basename(path)} → {os.path.basename(outp)}")
    return written


def normalize_smd(data_root: str, split: str = "test", out_root: str = "parquet") -> List[str]:
    """SMD TXT/CSV를 공통 스키마로 정규화하여 저장합니다.

    - split: "train" 또는 "test" (train은 라벨 0으로 채움)
    반환: 생성된 파일 경로 리스트(parquet 또는 csv)
    """
    # 내부 유틸을 활용해 파일 목록을 가져온 후 순회
    split_dir = os.path.join(data_root, "SMD", split if split in ("train", "test") else "test")
    files: List[str] = []
    for fn in sorted(os.listdir(split_dir)):
        if fn.lower().endswith((".txt", ".csv")):
            files.append(os.path.join(split_dir, fn))
    if not files:
        raise FileNotFoundError(f"No SMD files under {split_dir}")

    written: List[str] = []
    for data_path in files:
        # 각 파일별로 직접 파싱하여 시계열/라벨을 구성
        raw_rows = loader_smd._read_txt_values(data_path, sample_limit=None)
        c_val = loader_smd._infer_numeric_column(raw_rows)
        series = []
        for r in raw_rows:
            try:
                v = float(r[c_val]) if c_val < len(r) else float("nan")
            except Exception:
                continue
            series.append(v)

        if split == "test":
            label_dir = os.path.join(data_root, "SMD", "test_label")
            label_path = os.path.join(label_dir, os.path.basename(data_path))
            if not os.path.exists(label_path):
                # fallback: first available label file
                cand = [p for p in sorted(os.listdir(label_dir)) if p.lower().endswith((".txt", ".csv"))]
                label_path = os.path.join(label_dir, cand[0]) if cand else ""
            labels = loader_smd._read_labels(label_path, expected_len=len(series)) if label_path else [0] * len(series)
        else:
            labels = [0] * len(series)
        # machine_id를 파일명 기준으로 설정
        machine_id = os.path.basename(data_path).split(".")[0]
        rows: List[Tuple[int, float, int, str, str, str]] = []
        for i, (v, y) in enumerate(zip(series, labels)):
            rows.append((i, float(v), int(y), "SMD", os.path.basename(data_path), machine_id))
        rel = os.path.join(out_root, "smd")
        _ensure_dir(rel)
        out_path = os.path.join(rel, f"{machine_id}.parquet")
        outp = _try_write_parquet(out_path, rows)
        written.append(outp)
        # Integrity check
        ratio = (len(rows) / max(1, len(raw_rows))) if raw_rows else 1.0
        if ratio < 0.95:
            print(f"[WARN] SMD normalize: kept {ratio:.3%} of rows for {os.path.basename(data_path)} → {os.path.basename(outp)}")
    return written


def normalize_aihub71802(data_root: str, split: str = "Validation", label_scheme: str = "binary", out_root: str = "parquet") -> List[str]:
    """AIHub 71802 JSON/CSV 라벨/데이터를 공통 스키마로 정규화하여 저장합니다.

    - split: "Training" 또는 "Validation"
    - label_scheme: "binary"(>0→1) 또는 "risk4"
    반환: 생성된 파일 경로 리스트(parquet 또는 csv)
    """
    base = os.path.join(data_root, "manufacturing_transport_71802")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"AIHub71802 directory not found under {data_root}")
    split_dir = os.path.join(base, split if split in ("Training", "Validation") else "Validation")
    data_dir = os.path.join(split_dir, "data")
    if not os.path.isdir(data_dir):
        data_dir = split_dir

    data_files = loader_aihub_71802._list_files_recursive(data_dir, (".csv", ".txt"))
    label_dir = os.path.join(split_dir, "label")
    label_candidates = []
    if os.path.isdir(label_dir):
        label_candidates = loader_aihub_71802._list_files_recursive(label_dir, (".json", ".csv", ".txt"))

    written: List[str] = []
    for data_path in data_files:
        rows = loader_aihub_71802._read_csv_like(data_path, sample_limit=None)
        c_val = loader_aihub_71802._infer_numeric_column(rows)
        series: List[float] = []
        for r in rows:
            try:
                series.append(float(r[c_val]) if c_val < len(r) else float("nan"))
            except Exception:
                series.append(float("nan"))
        # labels: try nearest label file if present
        labels: List[int] = [0] * len(series)
        if label_candidates:
            json_cands = [p for p in label_candidates if p.lower().endswith(".json")]
            if json_cands:
                labels = loader_aihub_71802._labels_from_json(json_cands[0], len(series), label_scheme)
            else:
                lab_rows = loader_aihub_71802._read_csv_like(label_candidates[0])
                c_lab = loader_aihub_71802._infer_numeric_column(lab_rows)
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

        # device_id from filename (best-effort)
        device_id = os.path.basename(data_path).split(".")[0]
        common_rows: List[Tuple[int, float, int, str, str, str]] = []
        if label_scheme == "binary":
            for i, (v, y) in enumerate(zip(series, labels)):
                common_rows.append((i, float(v), int(y), "AIHub71802", os.path.basename(data_path), device_id))
        else:
            # risk4: store non-binary label as-is in the same 'label' column
            for i, (v, y) in enumerate(zip(series, labels)):
                common_rows.append((i, float(v), int(y), "AIHub71802", os.path.basename(data_path), device_id))

        rel = os.path.join(out_root, "aihub71802")
        _ensure_dir(rel)
        out_path = os.path.join(rel, f"{device_id}_{split}_{label_scheme}.parquet")
        outp = _try_write_parquet(out_path, common_rows)
        written.append(outp)
        # Integrity check
        ratio = (len(common_rows) / max(1, len(rows))) if rows else 1.0
        if ratio < 0.95:
            print(f"[WARN] AIHub normalize: kept {ratio:.3%} of rows for {os.path.basename(data_path)} → {os.path.basename(outp)}")

    return written


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Normalize datasets to common schema (Parquet/CSV)")
    p.add_argument("--dataset", choices=["SKAB", "SMD", "AIHub71802"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--label-scheme", choices=["binary", "risk4"], default="binary")
    p.add_argument("--out-root", default="parquet")
    args = p.parse_args()

    if args.dataset == "SKAB":
        paths = normalize_skab(args.data_root, out_root=args.out_root)
    elif args.dataset == "SMD":
        paths = normalize_smd(args.data_root, split=args.split, out_root=args.out_root)
    else:
        split = args.split if args.split in ("Training", "Validation") else "Validation"
        paths = normalize_aihub71802(args.data_root, split=split, label_scheme=args.label_scheme, out_root=args.out_root)
    for pth in paths[:5]:
        print(pth)
