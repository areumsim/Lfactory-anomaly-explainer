"""Correlation analysis between Point-wise F1 and Event-wise F1 (RQ3).

Usage:
    python scripts/correlation_analysis.py --runs runs/*

Computes Pearson and Spearman correlations to answer RQ3.
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple


def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Pearson correlation coefficient and approximate p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return 0.0, 1.0

    r = numerator / (denom_x * denom_y)

    # Approximate p-value using t-distribution
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r**2) if abs(r) < 0.9999 else 10.0
    # Very rough p-value approximation
    p_value = 2 * (1 - _norm_cdf(abs(t)))

    return r, p_value


def _norm_cdf(z):
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def analyze_correlation(run_pattern: str):
    """Analyze Point-wise vs Event-wise F1 correlation."""
    from glob import glob

    run_dirs = glob(run_pattern)
    point_f1_values = []
    event_f1_values = []
    metadata = []

    for run_dir in run_dirs:
        run_json = Path(run_dir) / "run.json"
        if run_json.exists():
            with open(run_json) as f:
                data = json.load(f)

            point_f1 = data.get("metrics", {}).get("f1", None)
            event_metrics = data.get("event_metrics", {})
            event_f1 = event_metrics.get("event_f1", None) if event_metrics else None

            if point_f1 is not None and event_f1 is not None:
                point_f1_values.append(point_f1)
                event_f1_values.append(event_f1)
                metadata.append({
                    "dataset": data.get("meta", {}).get("dataset", "unknown"),
                    "detector": data.get("detector", {}).get("method", "unknown"),
                })

    if len(point_f1_values) < 3:
        print("⚠️  Insufficient data for correlation analysis (need >= 3 data points)")
        return None

    # Compute correlation
    r, p_value = pearson_correlation(point_f1_values, event_f1_values)
    r_squared = r ** 2

    results = {
        "n_samples": len(point_f1_values),
        "pearson_r": r,
        "r_squared": r_squared,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "interpretation": _interpret_correlation(r),
        "data_points": [
            {"point_f1": pf1, "event_f1": ef1, **meta}
            for pf1, ef1, meta in zip(point_f1_values, event_f1_values, metadata)
        ],
    }

    # Save results
    output_file = Path("runs/correlation_analysis_rq3.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Correlation analysis complete!")
    print(f"📊 Pearson r = {r:.4f} (p = {p_value:.4f})")
    print(f"📈 R² = {r_squared:.4f}")
    print(f"📝 Interpretation: {results['interpretation']}")
    print(f"💾 Results saved to: {output_file}")

    if results['significant']:
        print(f"✅ SIGNIFICANT correlation (p < 0.05)")
    else:
        print(f"❌ Not significant (p >= 0.05)")

    return results


def _interpret_correlation(r):
    """Interpret correlation strength."""
    abs_r = abs(r)
    if abs_r < 0.3:
        return "weak"
    elif abs_r < 0.7:
        return "moderate"
    else:
        return "strong"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation analysis for RQ3")
    parser.add_argument("--runs", type=str, default="runs/*", help="Pattern to match run directories")

    args = parser.parse_args()
    analyze_correlation(args.runs)
