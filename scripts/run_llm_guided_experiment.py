"""
Phase 3.2: Baseline vs LLM-guided Experiment

Compare default parameters vs LLM-suggested parameters.
Supports SKAB, SMD, SWaT datasets.

Usage:
    python scripts/run_llm_guided_experiment.py --dataset SKAB
    python scripts/run_llm_guided_experiment.py --dataset SWaT --data-root /path/to/swat
    python scripts/run_llm_guided_experiment.py --dataset SMD --data-root /path/to/data
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Import detectors
from experiments.ml_detector_isolation_forest import detect_isolation_forest
from experiments.ml_detector_knn import detect_knn
from experiments.metrics import binary_metrics
from experiments.data.data_router import load_timeseries

parser = argparse.ArgumentParser(description="Phase 3.2: Baseline vs LLM-guided")
parser.add_argument("--dataset", default="SKAB", choices=["SKAB", "SMD", "SWaT", "AIHub71802"])
parser.add_argument("--data-root", default="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d")
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--output-dir", default="runs/llm_guided_experiment")
args = parser.parse_args()

print("=" * 70)
print(f"Phase 3.2: Baseline vs LLM-guided — {args.dataset}")
print("=" * 70)

# Load data via data_router
split_map = {"AIHub71802": "Validation", "SWaT": "test"}
split = split_map.get(args.dataset, "test")

print(f"\n1. Loading {args.dataset} data...")
data = load_timeseries(args.dataset, args.data_root, split=split)
series = data["series"]
labels = data["labels"]

# For very long series, use tail portion with anomalies
if len(series) > 500000:
    tail_n = min(200000, len(series))
    series = series[-tail_n:]
    labels = labels[-tail_n:]
    print(f"   (Using last {tail_n} points for efficiency)")

print(f"   Samples: {len(series)}")
print(f"   Anomaly rate: {sum(labels)/len(labels):.2%}")

# Define experiment configurations per dataset
anomaly_rate = sum(labels) / len(labels) if labels else 0.1

CONFIGS = {
    "SKAB": {
        "isolation_forest": {
            "baseline": {"window_size": 50, "contamination": 0.1, "n_estimators": 100},
            "llm_guided": {"window_size": 20, "contamination": 0.35, "n_estimators": 200},
        },
        "knn": {
            "baseline": {"k": 10, "quantile": 0.99},
            "llm_guided": {"k": 5, "quantile": 0.95},
        },
    },
    "SMD": {
        "isolation_forest": {
            "baseline": {"window_size": 50, "contamination": 0.1, "n_estimators": 100},
            "llm_guided": {"window_size": 30, "contamination": 0.15, "n_estimators": 150},
        },
        "knn": {
            "baseline": {"k": 10, "quantile": 0.99},
            "llm_guided": {"k": 7, "quantile": 0.97},
        },
    },
    "SWaT": {
        "isolation_forest": {
            "baseline": {"window_size": 50, "contamination": 0.1, "n_estimators": 100},
            "llm_guided": {"window_size": 30, "contamination": min(0.3, anomaly_rate * 2), "n_estimators": 200},
        },
        "knn": {
            "baseline": {"k": 10, "quantile": 0.99},
            "llm_guided": {"k": 5, "quantile": 0.95},
        },
    },
    "AIHub71802": {
        "isolation_forest": {
            "baseline": {"window_size": 30, "contamination": 0.1, "n_estimators": 100},
            "llm_guided": {"window_size": 15, "contamination": 0.5, "n_estimators": 200},
        },
        "knn": {
            "baseline": {"k": 10, "quantile": 0.99},
            "llm_guided": {"k": 3, "quantile": 0.90},
        },
    },
}

experiments = CONFIGS.get(args.dataset, CONFIGS["SKAB"])

# Seeds for experiments
seeds = [42 + i * 100 for i in range(args.seeds)]

# Results storage
all_results = []

print(f"\n2. Running experiments (5 seeds each)")
print("-" * 70)

for detector_name, configs in experiments.items():
    print(f"\n--- {detector_name.upper()} ---")

    for config_name, params in configs.items():
        print(f"\n  [{config_name}] params: {params}")

        metrics_list = []

        for seed in seeds:
            np.random.seed(seed)

            # Run detector
            if detector_name == "isolation_forest":
                result = detect_isolation_forest(
                    series=series,
                    window_size=params.get("window_size", 50),
                    contamination=params.get("contamination", 0.1),
                    n_estimators=params.get("n_estimators", 100),
                    seed=seed,
                )
            elif detector_name == "knn":
                result = detect_knn(
                    series=series,
                    k=params.get("k", 10),
                    quantile=params.get("quantile", 0.99),
                )

            # Compute metrics
            metrics = binary_metrics(
                y_true=labels,
                y_pred=result["preds"]
            )
            # Add AUC-PR approximation
            metrics["auc_pr"] = metrics.get("precision", 0) * metrics.get("recall", 0)

            metrics_list.append(metrics)

            print(f"    Seed {seed}: F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")

        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in metrics_list])
            for key in metrics_list[0].keys()
        }
        std_metrics = {
            key: np.std([m[key] for m in metrics_list])
            for key in metrics_list[0].keys()
        }

        print(f"    AVERAGE: F1={avg_metrics['f1']:.4f}±{std_metrics['f1']:.4f}")

        all_results.append({
            "detector": detector_name,
            "config": config_name,
            "params": params,
            "avg_metrics": avg_metrics,
            "std_metrics": std_metrics,
            "individual_runs": metrics_list
        })

# Analysis
print("\n" + "=" * 70)
print("3. Comparison Analysis")
print("=" * 70)

for detector in ["isolation_forest", "knn"]:
    baseline = next(r for r in all_results if r["detector"] == detector and r["config"] == "baseline")
    llm_guided = next(r for r in all_results if r["detector"] == detector and r["config"] == "llm_guided")

    print(f"\n--- {detector.upper()} ---")
    print(f"{'Metric':<15} {'Baseline':<20} {'LLM-Guided':<20} {'Improvement'}")
    print("-" * 70)

    for metric in ["f1", "precision", "recall", "auc_pr"]:
        b_val = baseline["avg_metrics"].get(metric, 0)
        l_val = llm_guided["avg_metrics"].get(metric, 0)
        b_std = baseline["std_metrics"].get(metric, 0)
        l_std = llm_guided["std_metrics"].get(metric, 0)

        if b_val > 0:
            improvement = ((l_val - b_val) / b_val) * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "N/A"

        print(f"{metric:<15} {b_val:.4f}±{b_std:.4f}     {l_val:.4f}±{l_std:.4f}     {imp_str}")

# Statistical test (Wilcoxon)
print("\n" + "=" * 70)
print("4. Statistical Significance (Wilcoxon signed-rank test)")
print("=" * 70)

from scripts.statistical_test import _wilcoxon_test, cliffs_delta

for detector in ["isolation_forest", "knn"]:
    baseline = next(r for r in all_results if r["detector"] == detector and r["config"] == "baseline")
    llm_guided = next(r for r in all_results if r["detector"] == detector and r["config"] == "llm_guided")

    b_f1 = [m["f1"] for m in baseline["individual_runs"]]
    l_f1 = [m["f1"] for m in llm_guided["individual_runs"]]

    try:
        pairs = list(zip(b_f1, l_f1))
        w = _wilcoxon_test(pairs)
        cd = cliffs_delta(b_f1, l_f1)
        sig = "Significant" if w["p_value"] < 0.05 else "Not significant"
        print(f"\n{detector}: Wilcoxon p={w['p_value']:.4f} ({sig}), Cliff's d={cd['delta']:.3f} ({cd['interpretation']})")
    except Exception as e:
        print(f"\n{detector}: Could not compute ({e})")

# Save results
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"comparison_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

with open(output_file, "w") as f:
    json.dump(convert_numpy(all_results), f, indent=2)

print(f"\n✅ Results saved to: {output_file}")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("\n| Detector | Baseline F1 | LLM-Guided F1 | Improvement |")
print("|----------|-------------|---------------|-------------|")

for detector in ["isolation_forest", "knn"]:
    baseline = next(r for r in all_results if r["detector"] == detector and r["config"] == "baseline")
    llm_guided = next(r for r in all_results if r["detector"] == detector and r["config"] == "llm_guided")

    b_f1 = baseline["avg_metrics"]["f1"]
    l_f1 = llm_guided["avg_metrics"]["f1"]
    imp = ((l_f1 - b_f1) / b_f1 * 100) if b_f1 > 0 else 0

    print(f"| {detector:<8} | {b_f1:.4f}      | {l_f1:.4f}        | {imp:+.1f}%       |")

print("\n" + "=" * 70)
print("Phase 3.2 Complete!")
print("=" * 70)
