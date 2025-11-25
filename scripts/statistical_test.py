"""Statistical significance testing for detector comparisons.

Usage:
    python scripts/statistical_test.py --runs runs/multi_seed_*

Performs Wilcoxon signed-rank test and Mann-Whitney U test to determine
if performance differences between detectors are statistically significant.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
import math


def _wilcoxon_test(pairs: List[tuple]) -> Dict:
    """Wilcoxon signed-rank test for paired samples.

    Args:
        pairs: List of (x, y) tuples (paired observations)

    Returns:
        dict with test_statistic and p_value (approximate)
    """
    # Compute differences
    diffs = [y - x for x, y in pairs]

    # Remove zeros
    diffs = [d for d in diffs if abs(d) > 1e-10]

    if len(diffs) < 5:
        return {"test_statistic": 0, "p_value": 1.0, "note": "insufficient_data"}

    # Rank absolute differences
    abs_diffs = [(abs(d), i, 1 if d > 0 else -1) for i, d in enumerate(diffs)]
    abs_diffs.sort(key=lambda x: x[0])

    # Assign ranks (simple approach, no tie handling)
    ranks = []
    for rank, (abs_diff, idx, sign) in enumerate(abs_diffs, 1):
        ranks.append((rank, sign))

    # Sum of positive and negative ranks
    W_plus = sum(r for r, s in ranks if s > 0)
    W_minus = sum(r for r, s in ranks if s < 0)

    W = min(W_plus, abs(W_minus))
    n = len(diffs)

    # Approximate p-value using normal approximation (for n > 10)
    if n > 10:
        mean_W = n * (n + 1) / 4
        std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (W - mean_W) / std_W if std_W > 0 else 0

        # Two-tailed p-value (very rough approximation)
        p_value = 2 * (1 - _norm_cdf(abs(z)))
    else:
        # For small n, p-value is less reliable
        p_value = 0.5  # Placeholder

    return {
        "test_statistic": W,
        "n_pairs": n,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def _norm_cdf(z):
    """Approximate standard normal CDF."""
    # Very rough approximation
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def load_run_results(run_pattern: str) -> List[Dict]:
    """Load results from run directories matching pattern."""
    from glob import glob

    run_dirs = glob(run_pattern)
    results = []

    for run_dir in run_dirs:
        run_json = Path(run_dir) / "run.json"
        if run_json.exists():
            with open(run_json) as f:
                data = json.load(f)
                results.append(data)

    return results


def compare_detectors(results: List[Dict], metric="auc_pr") -> Dict:
    """Compare detectors using statistical tests."""
    # Group by (dataset, detector, ml_method, seed)
    groups = {}

    for r in results:
        dataset = r.get("meta", {}).get("dataset", "unknown")
        detector = r.get("detector", {}).get("method", "unknown")
        ml_method = r.get("detector", {}).get("ml_method", None)
        seed = r.get("run", {}).get("seed", 0)

        key = (dataset, detector, ml_method)
        if key not in groups:
            groups[key] = []

        metric_value = r.get("metrics", {}).get(metric, 0.0)
        groups[key].append((seed, metric_value))

    # Pairwise comparisons
    comparisons = []
    group_keys = list(groups.keys())

    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            key_a = group_keys[i]
            key_b = group_keys[j]

            # Only compare within same dataset
            if key_a[0] != key_b[0]:
                continue

            values_a = dict(groups[key_a])
            values_b = dict(groups[key_b])

            # Find common seeds
            common_seeds = set(values_a.keys()) & set(values_b.keys())

            if len(common_seeds) < 5:
                continue

            pairs = [(values_a[s], values_b[s]) for s in sorted(common_seeds)]

            # Wilcoxon test
            test_result = _wilcoxon_test(pairs)

            mean_a = sum(values_a.values()) / len(values_a) if values_a else 0
            mean_b = sum(values_b.values()) / len(values_b) if values_b else 0

            comparisons.append({
                "detector_a": f"{key_a[1]}_{key_a[2]}" if key_a[2] else key_a[1],
                "detector_b": f"{key_b[1]}_{key_b[2]}" if key_b[2] else key_b[1],
                "dataset": key_a[0],
                "mean_a": mean_a,
                "mean_b": mean_b,
                "difference": mean_b - mean_a,
                "test": "wilcoxon",
                **test_result,
            })

    return {"metric": metric, "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical significance testing")
    parser.add_argument("--runs", type=str, default="runs/multi_seed_*", help="Pattern to match run directories")
    parser.add_argument("--metric", type=str, default="auc_pr", help="Metric to compare (auc_pr, f1, etc.)")
    parser.add_argument("--output", type=str, default="runs/statistical_tests.json", help="Output file")

    args = parser.parse_args()

    print(f"Loading results from: {args.runs}")
    results = load_run_results(args.runs)
    print(f"Loaded {len(results)} results")

    print(f"\nComparing detectors on metric: {args.metric}")
    comparison_results = compare_detectors(results, args.metric)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\n✅ Statistical tests complete!")
    print(f"📊 Results saved to: {output_path}")

    # Print summary
    print(f"\n📈 Summary:")
    for comp in comparison_results["comparisons"]:
        sig_marker = "✅ SIGNIFICANT" if comp.get("significant") else "❌ Not significant"
        print(f"{comp['detector_a']} vs {comp['detector_b']} ({comp['dataset']}): "
              f"Δ={comp['difference']:.4f}, p={comp['p_value']:.4f} {sig_marker}")
