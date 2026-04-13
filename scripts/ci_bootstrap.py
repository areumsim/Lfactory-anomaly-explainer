"""
부트스트랩 CI 스크립트

입력 예시:
  python scripts/ci_bootstrap.py --preds runs/tmp_preds.csv --n 1000 --out runs/ci.csv

설명:
- preds CSV는 result_manager.save_predictions_csv의 포맷을 가정(index,value,label,score,pred[,prob])
- AUC-PR은 score 기반으로 계산, ECE는 prob 컬럼이 있으면 사용하고 없으면 min-max 정규화(score) 사용
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from typing import List, Tuple

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from experiments import result_manager, calibration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap CI for AUC-PR and ECE")
    p.add_argument("--preds", required=True, help="Path to predictions CSV (index,value,label,score,pred[,prob])")
    p.add_argument("--n", type=int, default=1000, help="Number of bootstrap resamples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="Output CSV path for CI")
    p.add_argument("--bins", type=int, default=10, help="ECE bins")
    return p.parse_args()


def _read_preds(path: str) -> Tuple[List[int], List[float], List[float]]:
    ys: List[int] = []
    scores: List[float] = []
    probs: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        has_prob = False
        if header and "prob" in [h.strip().lower() for h in header]:
            has_prob = True
        for row in r:
            if not row:
                continue
            try:
                y = int(float(row[2]))
                s = float(row[3])
                ys.append(1 if y > 0 else 0)
                scores.append(s)
                if has_prob and len(row) >= 6:
                    probs.append(float(row[5]))
            except Exception:
                continue
    if not probs:
        probs = calibration.normalize_scores(scores)
    return ys, scores, probs


def _bootstrap_indices(n: int, rng: random.Random) -> List[int]:
    return [rng.randrange(n) for _ in range(n)]


def main() -> None:
    a = parse_args()
    ys, scores, probs = _read_preds(a.preds)
    if not ys:
        raise SystemExit("No rows found in preds CSV")
    rng = random.Random(a.seed)
    aucs: List[float] = []
    eces: List[float] = []
    for _ in range(max(1, a.n)):
        idx = _bootstrap_indices(len(ys), rng)
        yb = [ys[i] for i in idx]
        sb = [scores[i] for i in idx]
        pb = [probs[i] for i in idx]
        pr = result_manager.compute_pr(yb, sb)
        aucs.append(pr.auc)
        eces.append(calibration.ece(yb, pb, bins=a.bins))
    aucs.sort(); eces.sort()
    def ci(vals: List[float]) -> Tuple[float, float, float]:
        if not vals: return 0.0, 0.0, 0.0
        mean = sum(vals)/len(vals)
        lo = vals[int(0.025*(len(vals)-1))]
        hi = vals[int(0.975*(len(vals)-1))]
        return mean, lo, hi
    auc_mean, auc_lo, auc_hi = ci(aucs)
    ece_mean, ece_lo, ece_hi = ci(eces)
    outdir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("metric,mean,ci_lo,ci_hi\n")
        f.write(f"auc_pr,{auc_mean:.6f},{auc_lo:.6f},{auc_hi:.6f}\n")
        f.write(f"ece,{ece_mean:.6f},{ece_lo:.6f},{ece_hi:.6f}\n")
    print(f"[ci_bootstrap] Wrote {a.out}")


if __name__ == "__main__":
    main()
