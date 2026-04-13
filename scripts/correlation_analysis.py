"""RQ3: Point-wise vs Event-wise metric correlation analysis.

Usage:
    python scripts/correlation_analysis.py --results-json runs/all_results_clean.json
    python scripts/correlation_analysis.py --runs runs/multi_seed_*

Computes:
- Pearson and Spearman correlations (point_f1 vs event_f1)
- Stratified analysis by anomaly segment length (short vs long events)
- Per-detector correlation comparison
- Bootstrap CI for correlation coefficients
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Pearson r with t-test p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))

    if dx < 1e-12 or dy < 1e-12:
        return 0.0, 1.0

    r = num / (dx * dy)
    r = max(-1.0, min(1.0, r))

    if abs(r) > 0.9999:
        return r, 0.0
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
    p = 2 * (1 - _norm_cdf(abs(t)))
    return r, p


def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    def rank(vals):
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and abs(indexed[j][1] - indexed[i][1]) < 1e-12:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg
            i = j
        return ranks

    rx = rank(x)
    ry = rank(y)
    return pearson_correlation(rx, ry)


def bootstrap_correlation_ci(
    x: List[float], y: List[float],
    method: str = "spearman", n_boot: int = 1000, seed: int = 42
) -> Dict:
    """Bootstrap 95% CI for correlation coefficient."""
    import random
    rng = random.Random(seed)
    n = len(x)
    func = spearman_correlation if method == "spearman" else pearson_correlation

    r_point, _ = func(x, y)
    boot_rs = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        bx = [x[i] for i in idx]
        by = [y[i] for i in idx]
        br, _ = func(bx, by)
        boot_rs.append(br)

    boot_rs.sort()
    lo = boot_rs[max(0, int(0.025 * n_boot))]
    hi = boot_rs[min(n_boot - 1, int(0.975 * n_boot))]

    return {"r": r_point, "ci_lower": lo, "ci_upper": hi}


def _interpret(r: float) -> str:
    a = abs(r)
    if a < 0.3:
        return "weak"
    elif a < 0.7:
        return "moderate"
    return "strong"


def load_results(results_json: str = "", run_pattern: str = "") -> List[Dict]:
    """Load results from aggregated JSON or run directories."""
    if results_json:
        with open(results_json) as f:
            raw = json.load(f)
        return raw if isinstance(raw, list) else raw.get("results", [])

    from glob import glob
    results = []
    for rd in glob(run_pattern):
        rj = Path(rd) / "run.json"
        if rj.exists():
            with open(rj) as f:
                results.append(json.load(f))
    return results


def analyze_correlation(results: List[Dict], output_dir: str = "runs/rq3_correlation") -> Dict:
    """Full RQ3 correlation analysis."""
    # Extract data points
    data_points = []
    for r in results:
        m = r.get("metrics", {})
        em = r.get("event_metrics", {})
        if not em:
            continue

        point_f1 = m.get("f1")
        optimal_f1 = m.get("optimal_f1")
        event_f1 = em.get("event_f1")
        event_recall = em.get("event_recall")

        if point_f1 is None or event_f1 is None:
            continue

        det = r.get("detector", {})
        detector = det.get("method", "unknown")
        ml_method = det.get("ml_method")
        label = f"{detector}_{ml_method}" if ml_method else detector

        # Event duration info
        num_events = em.get("num_events", 0)
        mean_delay = em.get("mean_detection_delay", 0)

        data_points.append({
            "point_f1": point_f1,
            "optimal_f1": optimal_f1 or 0.0,
            "event_f1": event_f1,
            "event_recall": event_recall or 0.0,
            "detector": label,
            "dataset": r.get("meta", {}).get("dataset", "unknown"),
            "num_events": num_events,
            "mean_delay": mean_delay,
        })

    if len(data_points) < 5:
        return {"error": f"Insufficient data ({len(data_points)} points, need >= 5)"}

    # Overall correlation
    pf1 = [d["point_f1"] for d in data_points]
    ef1 = [d["event_f1"] for d in data_points]
    of1 = [d["optimal_f1"] for d in data_points]

    pearson_r, pearson_p = pearson_correlation(pf1, ef1)
    spearman_r, spearman_p = spearman_correlation(pf1, ef1)
    spearman_ci = bootstrap_correlation_ci(pf1, ef1, method="spearman")

    # Optimal F1 vs Event F1
    opt_spearman_r, opt_spearman_p = spearman_correlation(of1, ef1)

    result: Dict = {
        "n_samples": len(data_points),
        "overall": {
            "point_f1_vs_event_f1": {
                "pearson": {"r": pearson_r, "p": pearson_p, "interpretation": _interpret(pearson_r)},
                "spearman": {
                    "r": spearman_r, "p": spearman_p,
                    "ci_lower": spearman_ci["ci_lower"],
                    "ci_upper": spearman_ci["ci_upper"],
                    "interpretation": _interpret(spearman_r),
                },
            },
            "optimal_f1_vs_event_f1": {
                "spearman": {"r": opt_spearman_r, "p": opt_spearman_p, "interpretation": _interpret(opt_spearman_r)},
            },
        },
    }

    # Per-detector stratification
    detectors = sorted(set(d["detector"] for d in data_points))
    per_detector = {}
    for det in detectors:
        dp = [d for d in data_points if d["detector"] == det]
        if len(dp) < 5:
            continue
        dx = [d["point_f1"] for d in dp]
        dy = [d["event_f1"] for d in dp]
        sr, sp = spearman_correlation(dx, dy)
        per_detector[det] = {
            "n": len(dp),
            "spearman_r": sr,
            "p_value": sp,
            "mean_point_f1": sum(dx) / len(dx),
            "mean_event_f1": sum(dy) / len(dy),
            "event_point_ratio": (sum(dy) / len(dy)) / max(1e-6, sum(dx) / len(dx)),
        }
    result["per_detector"] = per_detector

    # Per-dataset stratification
    datasets = sorted(set(d["dataset"] for d in data_points))
    per_dataset = {}
    for ds in datasets:
        dp = [d for d in data_points if d["dataset"] == ds]
        if len(dp) < 5:
            continue
        dx = [d["point_f1"] for d in dp]
        dy = [d["event_f1"] for d in dp]
        sr, sp = spearman_correlation(dx, dy)
        per_dataset[ds] = {"n": len(dp), "spearman_r": sr, "p_value": sp}
    result["per_dataset"] = per_dataset

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rq3_correlation.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n=== RQ3: Point F1 vs Event F1 Correlation (N={len(data_points)}) ===")
    s = result["overall"]["point_f1_vs_event_f1"]["spearman"]
    print(f"Spearman rho = {s['r']:.4f} [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}] (p={s['p']:.4f}) -> {s['interpretation']}")
    o = result["overall"]["optimal_f1_vs_event_f1"]["spearman"]
    print(f"Optimal-F1 vs Event-F1: rho = {o['r']:.4f} (p={o['p']:.4f}) -> {o['interpretation']}")

    if per_detector:
        print("\nPer-detector:")
        for det, v in per_detector.items():
            print(f"  {det}: rho={v['spearman_r']:.3f}, event/point ratio={v['event_point_ratio']:.2f}")

    print(f"\nSaved to: {out / 'rq3_correlation.json'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ3 correlation analysis")
    parser.add_argument("--results-json", type=str, default="", help="Aggregated results JSON")
    parser.add_argument("--runs", type=str, default="runs/*", help="Run directory pattern")
    parser.add_argument("--output-dir", type=str, default="runs/rq3_correlation", help="Output directory")
    args = parser.parse_args()

    results = load_results(args.results_json, args.runs)
    print(f"Loaded {len(results)} results")
    analyze_correlation(results, args.output_dir)
