"""
Calibration 비교 스크립트

지정한 데이터셋/옵션으로 calibrate=[none, platt, isotonic]을 각각 실행하여
ECE와 AUC를 비교 요약 CSV로 저장합니다.

예)
  python scripts/calibration_eval.py --dataset synthetic --out runs/cal_eval.csv
  python scripts/calibration_eval.py --dataset SKAB --data-root /path/to/ROOT --out runs/skab_cal_eval.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate calibration methods and summarize ECE/AUC")
    p.add_argument("--dataset", required=True, choices=["synthetic", "SKAB", "SMD", "AIHub71802"])
    p.add_argument("--data-root", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--label-scheme", default="binary", choices=["binary", "risk4"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detector", default="rule", choices=["rule", "ml", "hybrid", "speccnn"])
    p.add_argument("--out", required=True, help="Output CSV path for summary")
    return p.parse_args()


def _run(method: str, args: argparse.Namespace, outdir: str) -> Dict[str, Any]:
    run_json = os.path.join(outdir, f"run_{method}.json")
    cmd: List[str] = [
        "python", "-m", "experiments.main_experiment",
        "--dataset", args.dataset,
        "--mode", "detect",
        "--detector", args.detector,
        "--seed", str(args.seed),
        "--calibrate", method,
        "--out-json", run_json,
    ]
    if args.dataset != "synthetic":
        if not args.data_root:
            raise SystemExit("--data-root is required for real datasets")
        cmd += ["--data-root", args.data_root, "--split", args.split, "--label-scheme", args.label_scheme]
    subprocess.check_call(cmd)
    with open(run_json, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    tmpdir = os.path.join(outdir, "_cal_runs")
    os.makedirs(tmpdir, exist_ok=True)
    methods = ["none", "platt", "isotonic", "temperature"]
    rows: List[List[str]] = [["method", "f1", "accuracy", "auc_roc", "auc_pr", "ece"]]
    for m in methods:
        try:
            res = _run(m, args, tmpdir)
        except subprocess.CalledProcessError:
            continue
        met = res.get("metrics", {})
        rows.append([
            m,
            f"{met.get('f1', 0.0):.6f}",
            f"{met.get('accuracy', 0.0):.6f}",
            f"{met.get('auc_roc', 0.0):.6f}",
            f"{met.get('auc_pr', 0.0):.6f}",
            f"{met.get('ece', 0.0):.6f}",
        ])
    # Save CSV
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"[calibration_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
