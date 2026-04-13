"""
비용 민감 임계 A/B 리포트 스크립트

단일 실행으로 기대 비용 고정/최적 값을 비교하여 요약 CSV를 생성합니다.

예)
  python scripts/cost_ab_report.py --dataset synthetic --detector ml --out runs/cost_ab.csv
  python scripts/cost_ab_report.py --dataset SKAB --data-root /path/to/ROOT --out runs/skab_cost_ab.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run cost-optimal thresholding and summarize A/B expected cost")
    p.add_argument("--dataset", required=True, choices=["synthetic", "SKAB", "SMD", "AIHub71802"])
    p.add_argument("--data-root", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--label-scheme", default="binary", choices=["binary", "risk4"])
    p.add_argument("--detector", default="rule", choices=["rule", "ml", "hybrid", "speccnn"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--costs", default="0,1,5,0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    outdir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    run_json = os.path.join(outdir, "cost_run.json")
    cmd: List[str] = [
        "python", "-m", "experiments.main_experiment",
        "--dataset", a.dataset,
        "--mode", "detect",
        "--detector", a.detector,
        "--seed", str(a.seed),
        "--cost-optimize",
        "--costs", a.costs,
        "--apply-cost-threshold",
        "--out-json", run_json,
    ]
    if a.dataset != "synthetic":
        if not a.data_root:
            raise SystemExit("--data-root is required for real datasets")
        cmd += ["--data-root", a.data_root, "--split", a.split, "--label-scheme", a.label_scheme]
    subprocess.check_call(cmd)
    with open(run_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    dec = obj.get("decision", {}) or {}
    m = obj.get("metrics", {}) or {}
    tm = (obj.get("thresholded", {}) or {}).get("metrics", {})
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("metric,original,cost_opt\n")
        f.write(f"expected_cost,{dec.get('fixed_expected_cost',0):.6f},{dec.get('optimal_expected_cost',0):.6f}\n")
        f.write(f"f1,{m.get('f1',0):.6f},{tm.get('f1',0):.6f}\n")
        f.write(f"accuracy,{m.get('accuracy',0):.6f},{tm.get('accuracy',0):.6f}\n")
    print(f"[cost_ab] wrote {a.out}")


if __name__ == "__main__":
    main()

