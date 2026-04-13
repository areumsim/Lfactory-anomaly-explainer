"""Grid search for SpecCNN frequency band weights optimization.

Usage:
    python scripts/speccnn_grid_search.py --dataset SKAB --data-root /workspace/data1/arsim/LFactory_d

Finds optimal weights for low/mid/high frequency bands using temporal 3-fold CV.
"""
import argparse
import itertools
import json
import sys
import os
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.data import data_router
from experiments import spec_cnn, result_manager


def _auc_pr_score(y_true, scores):
    """Compute AUC-PR from scores and labels."""
    pr = result_manager.compute_pr(y_true, scores)
    return pr.auc


def _temporal_3fold_indices(n, num_folds=3):
    """Generate temporal (non-shuffled) fold indices."""
    fold_size = n // num_folds
    folds = []
    for i in range(num_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < num_folds - 1 else n
        train_idx = list(range(0, val_start)) + list(range(val_end, n))
        val_idx = list(range(val_start, val_end))
        folds.append((train_idx, val_idx))
    return folds


def grid_search_weights(dataset, data_root="", datasets_cfg=""):
    """Grid search for optimal SpecCNN weights with temporal cross-validation."""
    w_low_range = [-0.5, 0.0, 0.5]
    w_mid_range = [0.0, 0.5, 1.0]
    w_high_range = [0.0, 0.5, 1.0]

    # Resolve data root
    if not data_root and datasets_cfg:
        data_root = data_router.resolve_data_root(dataset, datasets_cfg) or ""
    if not data_root:
        raise ValueError("--data-root or --datasets-cfg required")

    # Load data
    data = data_router.load_timeseries(name=dataset, data_root=data_root)
    series = data["series"]
    labels = data["labels"]
    n = len(series)

    if n < 100:
        raise ValueError(f"Series too short for grid search: {n} points")

    folds = _temporal_3fold_indices(n, num_folds=3)

    best_score = -1.0
    best_weights = None
    results = []

    total = len(w_low_range) * len(w_mid_range) * len(w_high_range)
    print(f"Starting grid search for {dataset} ({n} points, {total} combinations, 3-fold temporal CV)...")

    for idx, (w_l, w_m, w_h) in enumerate(itertools.product(w_low_range, w_mid_range, w_high_range)):
        fold_scores = []
        for train_idx, val_idx in folds:
            val_series = [series[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            if len(val_series) < 16 or sum(val_labels) == 0:
                continue

            det = spec_cnn.detect_speccnn(
                val_series, window=128, hop=16, quantile=0.99,
                w_low=w_l, w_mid=w_m, w_high=w_h,
            )
            auc = _auc_pr_score(val_labels, det["scores"])
            fold_scores.append(auc)

        mean_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0

        results.append({
            "w_low": w_l,
            "w_mid": w_m,
            "w_high": w_h,
            "auc_pr": mean_score,
            "fold_scores": fold_scores,
        })

        if mean_score > best_score:
            best_score = mean_score
            best_weights = (w_l, w_m, w_h)

        if (idx + 1) % 9 == 0:
            print(f"  [{idx + 1}/{total}] current best AUC-PR={best_score:.4f}")

    # Save results
    output_file = Path(f"runs/speccnn_grid_search_{dataset}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "dataset": dataset,
            "num_points": n,
            "best_weights": {"low": best_weights[0], "mid": best_weights[1], "high": best_weights[2]} if best_weights else None,
            "best_score": best_score,
            "all_results": results,
        }, f, indent=2)

    print(f"\nGrid search complete!")
    if best_weights:
        print(f"Best weights: low={best_weights[0]}, mid={best_weights[1]}, high={best_weights[2]}")
    print(f"Best AUC-PR: {best_score:.4f}")
    print(f"Results saved to: {output_file}")

    return best_weights, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpecCNN grid search")
    parser.add_argument("--dataset", type=str, default="SKAB", help="Dataset name")
    parser.add_argument("--data-root", type=str, default="", help="Data root directory")
    parser.add_argument("--datasets-cfg", type=str, default="experiments/data/datasets.yaml", help="datasets.yaml path")

    args = parser.parse_args()
    grid_search_weights(args.dataset, args.data_root, args.datasets_cfg)
