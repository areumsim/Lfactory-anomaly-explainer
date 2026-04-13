"""
배치 보고 스크립트

지정된 데이터셋 집합에 대해 calibration 비교 및 비용 임계 A/B 요약을 실행하고,
결과 CSV들을 하나의 디렉토리에 수집합니다.

예)
  python scripts/batch_report.py --datasets synthetic --out-dir runs/reports/batch1
  python scripts/batch_report.py --datasets SKAB SMD --data-root /path/to/ROOT --out-dir runs/reports
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run batch calibration and cost A/B reports")
    p.add_argument("--datasets", nargs="+", required=True, choices=["synthetic", "SKAB", "SMD", "AIHub71802"], help="Datasets to process")
    p.add_argument("--data-root", default="", help="Root path for real datasets")
    p.add_argument("--out-dir", required=True, help="Output directory for collected CSVs")
    p.add_argument("--detector", default="ml", choices=["rule", "ml", "hybrid", "speccnn"], help="Detector for cost A/B runs")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    base = os.path.abspath(a.out_dir)
    os.makedirs(base, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, ts)
    os.makedirs(out, exist_ok=True)
    for ds in a.datasets:
        if ds != "synthetic" and not a.data_root:
            print(f"[WARN] Skipping {ds}: --data-root required for real datasets")
            continue
        # Calibration eval
        cal_csv = os.path.join(out, f"{ds}_cal.csv")
        cmd_cal: List[str] = [
            "python", "scripts/calibration_eval.py", "--dataset", ds, "--out", cal_csv
        ]
        if ds != "synthetic":
            cmd_cal += ["--data-root", a.data_root]
        subprocess.call(cmd_cal)
        # Cost A/B
        cost_csv = os.path.join(out, f"{ds}_cost_ab.csv")
        cmd_cost: List[str] = [
            "python", "scripts/cost_ab_report.py", "--dataset", ds, "--detector", a.detector, "--out", cost_csv
        ]
        if ds != "synthetic":
            cmd_cost += ["--data-root", a.data_root]
        subprocess.call(cmd_cost)
    print(f"[batch_report] Collected reports under {out}")


if __name__ == "__main__":
    main()

