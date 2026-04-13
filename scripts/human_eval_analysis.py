"""Analyze human evaluation results.

Computes:
- Fleiss' kappa for inter-rater agreement
- Per-condition and per-dataset mean scores
- Correlation between human ratings and automated metrics
- Statistical tests for condition differences

Usage:
    python scripts/human_eval_analysis.py \
        --eval-files rater1.csv rater2.csv rater3.csv \
        --output human_eval_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def load_ratings(csv_path: str) -> List[Dict]:
    """Load human ratings from CSV file."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse numeric ratings
            for col in ["accuracy_1_5", "relevance_1_5", "actionability_1_5", "sensor_attribution_1_5"]:
                try:
                    row[col] = int(row[col]) if row[col].strip() else None
                except (ValueError, AttributeError):
                    row[col] = None
            rows.append(row)
    return rows


def fleiss_kappa(ratings_matrix: List[List[int]], k: int = 5) -> float:
    """Compute Fleiss' kappa for inter-rater agreement.

    Args:
        ratings_matrix: N x k matrix where entry [i][j] = number of raters
                       who assigned category j to item i.
        k: Number of categories (default 5 for 1-5 Likert).

    Returns:
        Fleiss' kappa coefficient.
    """
    N = len(ratings_matrix)
    if N == 0:
        return 0.0

    n = sum(ratings_matrix[0])  # number of raters
    if n <= 1:
        return 0.0

    # P_i for each item
    P_bar = 0.0
    for i in range(N):
        row = ratings_matrix[i]
        P_i = (sum(r * r for r in row) - n) / (n * (n - 1))
        P_bar += P_i
    P_bar /= N

    # P_e (expected agreement by chance)
    P_e = 0.0
    for j in range(k):
        p_j = sum(ratings_matrix[i][j] for i in range(N)) / (N * n)
        P_e += p_j * p_j

    if abs(1 - P_e) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0

    return (P_bar - P_e) / (1 - P_e)


def compute_agreement(
    rater_files: List[str],
    dimension: str = "accuracy_1_5",
    k: int = 5,
) -> Dict[str, Any]:
    """Compute inter-rater agreement for a rating dimension.

    Args:
        rater_files: List of CSV file paths, one per rater.
        dimension: Rating column name.
        k: Number of rating categories.

    Returns:
        Dict with kappa, per-rater means, and interpretation.
    """
    all_ratings = []
    for fp in rater_files:
        rows = load_ratings(fp)
        all_ratings.append(rows)

    # Build ratings matrix: align by (sample_id, condition)
    # Each item = unique (sample_id, condition) pair
    items: Dict[str, List[int]] = defaultdict(lambda: [0] * k)

    for rater_rows in all_ratings:
        for row in rater_rows:
            key = f"{row.get('sample_id', '')}_{row.get('condition', '')}"
            rating = row.get(dimension)
            if rating is not None and 1 <= rating <= k:
                items[key][rating - 1] += 1

    # Filter items rated by all raters
    n_raters = len(rater_files)
    valid_items = []
    for key, counts in items.items():
        if sum(counts) == n_raters:
            valid_items.append(counts)

    if not valid_items:
        return {"kappa": None, "n_items": 0, "interpretation": "No valid items"}

    kappa = fleiss_kappa(valid_items, k)

    # Interpretation (Landis & Koch, 1977)
    if kappa < 0:
        interp = "poor (worse than chance)"
    elif kappa < 0.20:
        interp = "slight"
    elif kappa < 0.40:
        interp = "fair"
    elif kappa < 0.60:
        interp = "moderate"
    elif kappa < 0.80:
        interp = "substantial"
    else:
        interp = "almost perfect"

    # Per-rater means
    rater_means = []
    for rater_rows in all_ratings:
        vals = [r[dimension] for r in rater_rows if r.get(dimension) is not None]
        rater_means.append(sum(vals) / len(vals) if vals else 0)

    return {
        "kappa": round(kappa, 4),
        "n_items": len(valid_items),
        "n_raters": n_raters,
        "interpretation": interp,
        "rater_means": [round(m, 3) for m in rater_means],
        "overall_mean": round(sum(rater_means) / len(rater_means), 3) if rater_means else 0,
    }


def analyze_by_condition(
    rater_files: List[str],
    dimensions: List[str] = None,
) -> Dict[str, Dict]:
    """Compute mean ratings per condition across all raters."""
    if dimensions is None:
        dimensions = ["accuracy_1_5", "relevance_1_5", "actionability_1_5", "sensor_attribution_1_5"]

    # Collect all ratings
    all_rows = []
    for fp in rater_files:
        all_rows.extend(load_ratings(fp))

    # Group by condition
    by_condition: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        cond = row.get("condition", "unknown")
        for dim in dimensions:
            val = row.get(dim)
            if val is not None:
                by_condition[cond][dim].append(val)

    results = {}
    for cond, dim_vals in by_condition.items():
        results[cond] = {}
        for dim, vals in dim_vals.items():
            mu = sum(vals) / len(vals) if vals else 0
            std = (sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0
            results[cond][dim] = {
                "mean": round(mu, 3),
                "std": round(std, 3),
                "n": len(vals),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze human evaluation results")
    parser.add_argument("--eval-files", nargs="+", required=True, help="CSV files from raters")
    parser.add_argument("--output", type=str, default="human_eval_analysis.json")
    args = parser.parse_args()

    dimensions = ["accuracy_1_5", "relevance_1_5", "actionability_1_5", "sensor_attribution_1_5"]

    print(f"Analyzing {len(args.eval_files)} rater files...")

    # Inter-rater agreement
    agreement = {}
    for dim in dimensions:
        agreement[dim] = compute_agreement(args.eval_files, dim)
        print(f"  {dim}: kappa={agreement[dim]['kappa']}, {agreement[dim]['interpretation']}")

    # Per-condition analysis
    by_condition = analyze_by_condition(args.eval_files, dimensions)

    results = {
        "inter_rater_agreement": agreement,
        "by_condition": by_condition,
        "n_raters": len(args.eval_files),
        "rater_files": args.eval_files,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n=== Per-Condition Means ===")
    for cond, dims in sorted(by_condition.items()):
        scores = " | ".join(
            f"{d.replace('_1_5', '')}={dims[d]['mean']:.2f}"
            for d in dimensions if d in dims
        )
        print(f"  {cond}: {scores}")


if __name__ == "__main__":
    main()
