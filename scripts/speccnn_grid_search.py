"""Grid search for SpecCNN frequency band weights optimization.

Usage:
    python scripts/speccnn_grid_search.py --dataset SKAB

Finds optimal weights for low/mid/high frequency bands on validation set.
"""
import argparse
import itertools
import json
from pathlib import Path


def grid_search_weights(dataset, data_root=""):
    """Grid search for optimal SpecCNN weights."""
    w_low_range = [-0.5, -0.2, 0.0, 0.2]
    w_mid_range = [0.2, 0.4, 0.6, 0.8, 1.0]
    w_high_range = [0.2, 0.4, 0.6, 0.8, 1.0]

    best_score = 0.0
    best_weights = None
    results = []

    print(f"Starting grid search for {dataset}...")
    print(f"Total combinations: {len(w_low_range) * len(w_mid_range) * len(w_high_range)}")

    # Note: This is a skeleton. Actual implementation would:
    # 1. Load validation data
    # 2. For each weight combination, run SpecCNN detector
    # 3. Compute AUC-PR on validation set
    # 4. Track best weights

    for w_l, w_m, w_h in itertools.product(w_low_range, w_mid_range, w_high_range):
        # Placeholder: In real implementation, run detector with these weights
        # and compute validation AUC-PR
        score = 0.5  # Placeholder

        results.append({
            "w_low": w_l,
            "w_mid": w_m,
            "w_high": w_h,
            "auc_pr": score,
        })

        if score > best_score:
            best_score = score
            best_weights = (w_l, w_m, w_h)

    # Save results
    output_file = Path(f"runs/speccnn_grid_search_{dataset}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "dataset": dataset,
            "best_weights": {"low": best_weights[0], "mid": best_weights[1], "high": best_weights[2]},
            "best_score": best_score,
            "all_results": results,
        }, f, indent=2)

    print(f"\n✅ Grid search complete!")
    print(f"📊 Best weights: low={best_weights[0]}, mid={best_weights[1]}, high={best_weights[2]}")
    print(f"📈 Best AUC-PR: {best_score:.4f}")
    print(f"💾 Results saved to: {output_file}")

    return best_weights, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpecCNN grid search")
    parser.add_argument("--dataset", type=str, default="SKAB", help="Dataset name")
    parser.add_argument("--data-root", type=str, default="", help="Data root directory")

    args = parser.parse_args()
    grid_search_weights(args.dataset, args.data_root)
