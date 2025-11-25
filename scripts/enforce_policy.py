#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple
import hashlib


def load_result(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def collect_runs(runs_dir: str) -> List[Tuple[str, Dict]]:
    out: List[Tuple[str, Dict]] = []
    for name in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, name)
        if not os.path.isdir(p):
            continue
        rj = os.path.join(p, "run.json")
        if not os.path.exists(rj):
            continue
        res = load_result(rj)
        out.append((p, res))
    return out


def run_signature(r: Dict) -> str:
    """Compute a stable signature to detect duplicate runs.

    Keys considered: dataset, file/path(if any), split/label_scheme, detector method+params,
    seed, git_sha (optional), and series length if available in meta.
    """
    meta = r.get("meta", {})
    run = r.get("run", {})
    det = r.get("detector", {})
    parts = {
        "dataset": meta.get("dataset", ""),
        "file": meta.get("file", ""),
        "path": meta.get("path", ""),
        "split": meta.get("split", ""),
        "label_scheme": meta.get("label_scheme", ""),
        "detector_method": det.get("method", ""),
        "detector_params": {k: det.get(k) for k in sorted(det.keys()) if k != "method"},
        "seed": run.get("seed", ""),
        "git_sha": run.get("git_sha", ""),
        "num_points": meta.get("num_points", ""),
    }
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def archive_duplicates(runs: List[Tuple[str, Dict]], archive_root: str) -> List[str]:
    seen: Dict[str, Tuple[str, Dict]] = {}
    moved: List[str] = []
    # Sort by start_ts so that later (newer) ones overwrite earlier
    ordered = sorted(runs, key=lambda x: x[1].get("run", {}).get("start_ts", ""))
    for p, r in ordered:
        sig = run_signature(r)
        if sig in seen:
            # duplicate: archive this path
            dst = os.path.join(archive_root, os.path.basename(p))
            os.makedirs(archive_root, exist_ok=True)
            shutil.move(p, dst)
            moved.append(p)
        else:
            seen[sig] = (p, r)
    return moved


def ensure_index(runs: List[Tuple[str, Dict]], runs_dir: str) -> None:
    rows = [[
        "dataset","start_ts","run_id","detector","precision","recall","f1","accuracy","auc_roc","auc_pr","ece","fixed_cost","optimal_cost","gain","path"
    ]]
    for p, r in runs:
        m = r.get("metrics", {})
        d = r.get("decision", {}) or {}
        rows.append([
            r.get("meta",{}).get("dataset",""),
            r.get("run",{}).get("start_ts",""),
            r.get("run",{}).get("run_id",""),
            r.get("detector",{}).get("method",""),
            m.get("precision",0.0), m.get("recall",0.0), m.get("f1",0.0), m.get("accuracy",0.0), m.get("auc_roc",0.0), m.get("auc_pr",0.0), m.get("ece",0.0),
            d.get("fixed_expected_cost",0.0), d.get("optimal_expected_cost",0.0), d.get("expected_cost_gain",0.0),
            p
        ])
    os.makedirs(runs_dir, exist_ok=True)
    with open(os.path.join(runs_dir, "index.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    # minimal markdown index
    with open(os.path.join(runs_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write(f"# Runs Index\n\nTotal: {len(runs)}\n\nSee index.csv for details.\n")


def archive_excess(runs: List[Tuple[str, Dict]], retain: int, archive_root: str) -> List[str]:
    # Group by (dataset, detector)
    groups: Dict[Tuple[str, str], List[Tuple[str, Dict]]] = defaultdict(list)
    for p, r in runs:
        ds = str(r.get("meta",{}).get("dataset","unknown"))
        det = str(r.get("detector",{}).get("method","unknown"))
        groups[(ds, det)].append((p, r))
    moved: List[str] = []
    for key, items in groups.items():
        # sort by start_ts descending
        items.sort(key=lambda x: x[1].get("run",{}).get("start_ts",""), reverse=True)
        excess = items[retain:]
        for p, r in excess:
            dst = os.path.join(archive_root, os.path.basename(p))
            os.makedirs(archive_root, exist_ok=True)
            shutil.move(p, dst)
            moved.append(p)
    return moved


def remove_unknown(runs_dir: str) -> List[str]:
    removed: List[str] = []
    for name in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, name)
        if not os.path.isdir(p):
            continue
        rj = os.path.join(p, "run.json")
        if not os.path.exists(rj):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description="Enforce runs/ policy: organize, retain, archive, and index")
    ap.add_argument("--retain", type=int, default=5, help="Keep N most recent per (dataset,detector)")
    ap.add_argument("--archive-root", type=str, default="runs_archive", help="Archive directory root")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    runs_dir = os.path.join(root, "runs")
    archive_dir = os.path.join(root, args.archive_root)
    os.makedirs(archive_dir, exist_ok=True)

    # 1) remove unknown (no run.json)
    remove_unknown(runs_dir)
    # 2) collect
    runs = collect_runs(runs_dir)
    # 3) archive duplicates (keep latest)
    archive_duplicates(runs, archive_root=archive_dir)
    # 4) collect again and archive excess by policy
    runs = collect_runs(runs_dir)
    moved = archive_excess(runs, retain=args.retain, archive_root=os.path.join(archive_dir))
    # 5) rebuild index
    runs = collect_runs(runs_dir)
    ensure_index(runs, runs_dir)
    print(f"Archived {len(moved)} old runs. Index written to runs/index.csv")


if __name__ == "__main__":
    main()
