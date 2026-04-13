"""Export LLM explanations for human evaluation.

Generates a CSV spreadsheet with:
- Sample metadata (dataset, sensor, anomaly details)
- LLM explanations from different conditions
- Rating columns for human evaluators

Usage:
    python scripts/human_eval_export.py \
        --results-dir runs/llm_explanation_v3 \
        --output human_eval_sheet.csv \
        --n-samples 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_explanation_results(results_dir: str) -> List[Dict]:
    """Load results from LLM explanation experiment."""
    results_file = os.path.join(results_dir, "llm_explanation_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("results", [])


def select_stratified_samples(
    results: List[Dict],
    n_samples: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """Select stratified samples across datasets and conditions."""
    random.seed(seed)

    # Group by dataset
    by_dataset: Dict[str, List[Dict]] = {}
    for r in results:
        ds = r.get("sample", {}).get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    # Proportional sampling
    selected = []
    per_dataset = max(1, n_samples // max(1, len(by_dataset)))
    for ds, ds_results in by_dataset.items():
        n = min(per_dataset, len(ds_results))
        selected.extend(random.sample(ds_results, n))

    # Fill remaining quota
    remaining = n_samples - len(selected)
    if remaining > 0:
        pool = [r for r in results if r not in selected]
        if pool:
            selected.extend(random.sample(pool, min(remaining, len(pool))))

    return selected[:n_samples]


def export_csv(
    samples: List[Dict],
    output_path: str,
    conditions: List[str] = None,
):
    """Export samples to CSV for human evaluation.

    Each row = one (sample, condition) pair.
    Columns: sample_id, dataset, file, anomaly_idx, sigma, condition, explanation,
             accuracy(1-5), relevance(1-5), actionability(1-5), sensor_attribution(1-5), notes
    """
    if conditions is None:
        conditions = ["baseline", "feature_only", "domain_only", "full"]

    rows = []
    for i, result in enumerate(samples):
        sample_info = result.get("sample", {})
        for cond in conditions:
            cond_data = result.get(cond, {})
            explanation = cond_data.get("explanation", "")
            if not explanation:
                continue

            rows.append({
                "sample_id": i + 1,
                "dataset": sample_info.get("dataset", ""),
                "file": sample_info.get("file", ""),
                "anomaly_idx": sample_info.get("anomaly_idx", ""),
                "sigma_deviation": f"{sample_info.get('sigma_deviation', 0):.2f}",
                "n_sensors": sample_info.get("n_sensors", ""),
                "condition": cond,
                "explanation": explanation.replace("\n", " "),
                "accuracy_1_5": "",
                "relevance_1_5": "",
                "actionability_1_5": "",
                "sensor_attribution_1_5": "",
                "evaluator_notes": "",
            })

    # Shuffle to avoid ordering bias
    random.shuffle(rows)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {output_path}")
    print(f"  Samples: {len(samples)}, Conditions: {len(conditions)}")
    print(f"  Columns to fill: accuracy_1_5, relevance_1_5, actionability_1_5, sensor_attribution_1_5")


def main():
    parser = argparse.ArgumentParser(description="Export LLM explanations for human evaluation")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with llm_explanation_results.json")
    parser.add_argument("--output", type=str, default="human_eval_sheet.csv")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="+", default=["baseline", "feature_only", "domain_only", "full"])
    args = parser.parse_args()

    results = load_explanation_results(args.results_dir)
    print(f"Loaded {len(results)} results from {args.results_dir}")

    samples = select_stratified_samples(results, args.n_samples, args.seed)
    print(f"Selected {len(samples)} stratified samples")

    export_csv(samples, args.output, args.conditions)


if __name__ == "__main__":
    main()
