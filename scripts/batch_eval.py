#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
from typing import List, Dict


def run_one(cmd: List[str]) -> Dict:
    out = subprocess.check_output(cmd)
    try:
        return json.loads(out.decode("utf-8"))
    except Exception:
        return {}


def summarize(dataset: str, results: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows: List[List] = [[
        "run_id","detector","start_ts","precision","recall","f1","accuracy","auc_roc","auc_pr","ece","fixed_cost","optimal_cost","gain"
    ]]
    for r in results:
        m = r.get("metrics", {})
        d = r.get("decision", {}) or {}
        rows.append([
            r.get("run",{}).get("run_id",""),
            r.get("detector",{}).get("method",""),
            r.get("run",{}).get("start_ts",""),
            m.get("precision",0.0), m.get("recall",0.0), m.get("f1",0.0), m.get("accuracy",0.0), m.get("auc_roc",0.0), m.get("auc_pr",0.0), m.get("ece",0.0),
            d.get("fixed_expected_cost",0.0), d.get("optimal_expected_cost",0.0), d.get("expected_cost_gain",0.0)
        ])
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    # markdown report
    lines = [f"# Batch Summary – {dataset}", "", f"Total runs: {len(results)}", "", "Columns:",
             "run_id, detector, start_ts, precision, recall, f1, accuracy, auc_roc, auc_pr, ece, fixed_cost, optimal_cost, gain"]
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    # Config: adjust as needed
    data_root = os.environ.get("DATA_ROOT", "/workspace/data1_arsim/LFactory_d")
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    out_root = os.path.join(base, "runs", "reports", ts)

    # SKAB batch (rule robust), first 2 files meeting min_length
    skab_res: List[Dict] = []
    for i in range(2):
        cmd = [
            "python","-m","experiments.main_experiment",
            "--dataset","SKAB","--mode","detect","--detector","rule","--z-robust",
            "--data-root", data_root, "--split","test","--seed","42",
            "--file-index", str(i), "--min-length", "200",
            "--run-id", f"skab_batch_{i}",
            "--calibrate","none","--ece-bins","10","--cost-optimize","--costs","0,1,5,0"
        ]
        skab_res.append(run_one(cmd))
    summarize("SKAB", skab_res, os.path.join(out_root, "SKAB"))

    # SMD batch (ml knn), first 2 files
    smd_res: List[Dict] = []
    for i in range(2):
        cmd = [
            "python","-m","experiments.main_experiment",
            "--dataset","SMD","--mode","detect","--detector","ml",
            "--data-root", data_root, "--split","test","--seed","42",
            "--file-index", str(i), "--min-length","200",
            "--ml-k","10","--ml-quantile","0.99",
            "--run-id", f"smd_batch_{i}",
            "--calibrate","none","--ece-bins","10","--cost-optimize","--costs","0,1,5,0"
        ]
        smd_res.append(run_one(cmd))
    summarize("SMD", smd_res, os.path.join(out_root, "SMD"))

    print(f"Batch summaries written under {out_root}")


if __name__ == "__main__":
    main()
