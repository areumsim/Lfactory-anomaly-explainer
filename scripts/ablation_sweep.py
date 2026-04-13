"""
하이퍼파라미터 스윕 스크립트 (Ablation & 민감도)

주요 하이퍼파라미터(alpha, z-window, ml-k, quantile)를 스윕하며
`experiments.main_experiment`를 호출해 핵심 지표를 CSV로 요약합니다.

예)
  python scripts/ablation_sweep.py --dataset synthetic --out runs/ablate.csv \
    --detectors rule ml hybrid --alphas 0.3 0.5 0.7 --z-windows 30 50 80 \
    --ml-ks 5 10 20 --quantiles 0.95 0.99

실데이터의 경우 `--data-root` 지정이 필요합니다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation/sensitivity sweep over key hyperparameters")
    p.add_argument("--dataset", required=True, choices=["synthetic", "SKAB", "SMD", "AIHub71802"])
    p.add_argument("--data-root", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--label-scheme", default="binary", choices=["binary", "risk4"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="Output CSV path for summary")
    p.add_argument("--detectors", nargs="+", default=["rule", "ml", "hybrid"], choices=["rule", "ml", "hybrid", "speccnn"], help="Detectors to include")
    p.add_argument("--alphas", nargs="*", type=float, default=[0.5], help="Hybrid alpha list")
    p.add_argument("--z-windows", nargs="*", type=int, default=[50], help="Rule z-window list")
    p.add_argument("--ml-ks", nargs="*", type=int, default=[10], help="ML k list")
    p.add_argument("--quantiles", nargs="*", type=float, default=[0.99], help="Quantile thresholds list")
    p.add_argument("--calibrate", default="none", choices=["none", "platt", "isotonic", "temperature"], help="Calibration method")
    return p.parse_args()


def run_one(ds: str, det: str, seed: int, params: dict, common: argparse.Namespace, outdir: str) -> dict:
    run_json = os.path.join(outdir, f"run_{det}_a{params.get('alpha','')}_w{params.get('z_window','')}_k{params.get('ml_k','')}_q{params.get('quantile','')}.json")
    cmd: List[str] = [
        "python3", "-m", "experiments.main_experiment",
        "--dataset", ds,
        "--mode", "detect",
        "--detector", det,
        "--seed", str(seed),
        "--calibrate", common.calibrate,
        "--out-json", run_json,
    ]
    if ds != "synthetic":
        if not common.data_root:
            raise SystemExit("--data-root is required for real datasets")
        cmd += ["--data-root", common.data_root, "--split", common.split, "--label-scheme", common.label_scheme]
    # detector-specific params
    if det in ("rule", "hybrid") and "z_window" in params:
        cmd += ["--z-window", str(params["z_window"])]
    if det in ("ml", "hybrid") and "ml_k" in params:
        cmd += ["--ml-k", str(params["ml_k"])]
    if det == "hybrid" and "alpha" in params:
        cmd += ["--hybrid-alpha", str(params["alpha"]) ]
    # quantile for ml/hybrid/speccnn
    if det == "ml":
        cmd += ["--ml-quantile", str(params.get("quantile", 0.99))]
    if det == "hybrid":
        cmd += ["--hybrid-quantile", str(params.get("quantile", 0.99))]
    if det == "speccnn":
        cmd += ["--sc-quantile", str(params.get("quantile", 0.99))]
    subprocess.check_call(cmd)
    with open(run_json, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    a = parse_args()
    outdir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, "_ablate_runs")
    os.makedirs(tmp, exist_ok=True)

    combos = []
    for det in a.detectors:
        if det == "rule":
            for w in a.z_windows:
                for q in a.quantiles:
                    combos.append((det, {"z_window": w, "quantile": q}))
        elif det == "ml":
            for k in a.ml_ks:
                for q in a.quantiles:
                    combos.append((det, {"ml_k": k, "quantile": q}))
        elif det == "hybrid":
            for al, w, k, q in itertools.product(a.alphas, a.z_windows, a.ml_ks, a.quantiles):
                combos.append((det, {"alpha": al, "z_window": w, "ml_k": k, "quantile": q}))
        else:  # speccnn
            for q in a.quantiles:
                combos.append((det, {"quantile": q}))

    rows: List[List[str]] = [[
        "detector", "alpha", "z_window", "ml_k", "quantile",
        "f1", "accuracy", "auc_roc", "auc_pr", "ece"
    ]]
    for det, params in combos:
        try:
            res = run_one(a.dataset, det, a.seed, params, a, tmp)
        except subprocess.CalledProcessError:
            continue
        met = res.get("metrics", {})
        rows.append([
            det,
            str(params.get("alpha", "")),
            str(params.get("z_window", "")),
            str(params.get("ml_k", "")),
            str(params.get("quantile", "")),
            f"{met.get('f1',0):.6f}",
            f"{met.get('accuracy',0):.6f}",
            f"{met.get('auc_roc',0):.6f}",
            f"{met.get('auc_pr',0):.6f}",
            f"{met.get('ece',0):.6f}",
        ])

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows[1:]:
            f.write(",".join(r) + "\n")
    print(f"[ablation_sweep] Wrote summary to {a.out}")


if __name__ == "__main__":
    main()

