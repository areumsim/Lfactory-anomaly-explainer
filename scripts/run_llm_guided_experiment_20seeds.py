"""
Phase 3.2 Extended: Baseline vs LLM-guided Experiment (20 seeds)

Compare default parameters vs LLM-suggested parameters with statistical significance
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Import detectors
from experiments.ml_detector_isolation_forest import detect_isolation_forest
from experiments.ml_detector_knn import detect_knn
from experiments.metrics import binary_metrics

print("=" * 70)
print("Phase 3.2 Extended: Baseline vs LLM-guided Experiment (20 seeds)")
print("=" * 70)

# Load SKAB data
DATA_ROOT = "/workspace/data1/arsim/LFactory_d"
skab_file = f"{DATA_ROOT}/SKAB/valve1/0.csv"

print(f"\n1. Loading data: {skab_file}")
df = pd.read_csv(skab_file, sep=';')

# Use one sensor for experiments (Accelerometer1RMS)
sensor_col = "Accelerometer1RMS"
series = df[sensor_col].values.tolist()
labels = df['anomaly'].values.astype(int).tolist()

print(f"   Samples: {len(series)}")
print(f"   Anomaly rate: {sum(labels)/len(labels):.2%}")

# Define experiment configurations
experiments = {
    "isolation_forest": {
        "baseline": {
            "window": 50,
            "contamination": 0.1,
            "n_estimators": 100
        },
        "llm_guided": {
            "window": 20,
            "contamination": 0.35,
            "n_estimators": 200
        }
    },
    "knn": {
        "baseline": {
            "k": 10,
            "quantile": 0.99
        },
        "llm_guided": {
            "k": 5,
            "quantile": 0.95
        }
    }
}

# 20 seeds for statistical significance
seeds = [42 + i*100 for i in range(20)]  # [42, 142, 242, ..., 1942]

# Results storage
all_results = []

print(f"\n2. Running experiments ({len(seeds)} seeds each)")
print(f"   Seeds: {seeds[:5]}...{seeds[-1]}")
print("-" * 70)

for detector_name, configs in experiments.items():
    print(f"\n--- {detector_name.upper()} ---")

    for config_name, params in configs.items():
        print(f"\n  [{config_name}] params: {params}")

        metrics_list = []

        for i, seed in enumerate(seeds):
            np.random.seed(seed)

            # Run detector
            if detector_name == "isolation_forest":
                result = detect_isolation_forest(
                    series=series,
                    window=params["window"],
                    contamination=params["contamination"],
                    n_estimators=params["n_estimators"],
                    random_state=seed
                )
            elif detector_name == "knn":
                result = detect_knn(
                    series=series,
                    k=params["k"],
                    quantile=params["quantile"]
                )

            # Compute metrics
            metrics = binary_metrics(
                y_true=labels,
                y_pred=result["preds"]
            )
            # Add AUC-PR approximation
            metrics["auc_pr"] = metrics.get("precision", 0) * metrics.get("recall", 0)

            metrics_list.append(metrics)

            # Print progress every 5 seeds
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i+1}/{len(seeds)} seeds completed")

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
            "n_seeds": len(seeds),
            "avg_metrics": avg_metrics,
            "std_metrics": std_metrics,
            "individual_runs": metrics_list
        })

# Analysis
print("\n" + "=" * 70)
print("3. Comparison Analysis")
print("=" * 70)

comparison_results = {}

for detector in ["isolation_forest", "knn"]:
    baseline = next(r for r in all_results if r["detector"] == detector and r["config"] == "baseline")
    llm_guided = next(r for r in all_results if r["detector"] == detector and r["config"] == "llm_guided")

    print(f"\n--- {detector.upper()} ---")
    print(f"{'Metric':<15} {'Baseline':<20} {'LLM-Guided':<20} {'Improvement'}")
    print("-" * 70)

    detector_comparison = {}

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
            improvement = 0

        print(f"{metric:<15} {b_val:.4f}±{b_std:.4f}     {l_val:.4f}±{l_std:.4f}     {imp_str}")

        detector_comparison[metric] = {
            "baseline": {"mean": b_val, "std": b_std},
            "llm_guided": {"mean": l_val, "std": l_std},
            "improvement_pct": improvement
        }

    comparison_results[detector] = detector_comparison

# Statistical test (Wilcoxon)
print("\n" + "=" * 70)
print("4. Statistical Significance (Wilcoxon signed-rank test)")
print("=" * 70)

from scipy.stats import wilcoxon

statistical_tests = {}

for detector in ["isolation_forest", "knn"]:
    baseline = next(r for r in all_results if r["detector"] == detector and r["config"] == "baseline")
    llm_guided = next(r for r in all_results if r["detector"] == detector and r["config"] == "llm_guided")

    b_f1 = [m["f1"] for m in baseline["individual_runs"]]
    l_f1 = [m["f1"] for m in llm_guided["individual_runs"]]

    try:
        stat, p_value = wilcoxon(b_f1, l_f1)
        sig = "✓ Significant" if p_value < 0.05 else "✗ Not significant"
        print(f"\n{detector}: p-value={p_value:.6f} {sig}")

        statistical_tests[detector] = {
            "test": "wilcoxon_signed_rank",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_samples": len(b_f1)
        }
    except Exception as e:
        print(f"\n{detector}: Error - {str(e)}")
        statistical_tests[detector] = {"error": str(e)}

# Effect size (Cohen's d)
print("\n" + "=" * 70)
print("5. Effect Size (Cohen's d)")
print("=" * 70)

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group2) - np.mean(group1)) / pooled_std if pooled_std > 0 else 0

for detector in ["isolation_forest", "knn"]:
    baseline = next(r for r in all_results if r["detector"] == detector and r["config"] == "baseline")
    llm_guided = next(r for r in all_results if r["detector"] == detector and r["config"] == "llm_guided")

    b_f1 = [m["f1"] for m in baseline["individual_runs"]]
    l_f1 = [m["f1"] for m in llm_guided["individual_runs"]]

    d = cohens_d(b_f1, l_f1)

    if abs(d) < 0.2:
        effect = "negligible"
    elif abs(d) < 0.5:
        effect = "small"
    elif abs(d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    print(f"{detector}: Cohen's d = {d:.4f} ({effect})")
    statistical_tests[detector]["cohens_d"] = float(d)
    statistical_tests[detector]["effect_size"] = effect

# Save results
output_dir = Path("./runs/llm_guided_experiment_20seeds")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Full results
output_file = output_dir / f"full_results_{timestamp}.json"

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

full_output = {
    "experiment": "LLM-guided Parameter Optimization (20 seeds)",
    "timestamp": timestamp,
    "dataset": "SKAB valve1",
    "n_seeds": len(seeds),
    "seeds": seeds,
    "results": convert_numpy(all_results),
    "comparison": convert_numpy(comparison_results),
    "statistical_tests": convert_numpy(statistical_tests)
}

with open(output_file, "w") as f:
    json.dump(full_output, f, indent=2)

print(f"\n✅ Full results saved to: {output_file}")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\n| Detector | Baseline F1 | LLM-Guided F1 | Improvement | p-value | Significant |")
print("|----------|-------------|---------------|-------------|---------|-------------|")

for detector in ["isolation_forest", "knn"]:
    baseline = next(r for r in all_results if r["detector"] == detector and r["config"] == "baseline")
    llm_guided = next(r for r in all_results if r["detector"] == detector and r["config"] == "llm_guided")

    b_f1 = baseline["avg_metrics"]["f1"]
    b_std = baseline["std_metrics"]["f1"]
    l_f1 = llm_guided["avg_metrics"]["f1"]
    l_std = llm_guided["std_metrics"]["f1"]
    imp = ((l_f1 - b_f1) / b_f1 * 100) if b_f1 > 0 else 0

    p_val = statistical_tests[detector].get("p_value", 1.0)
    sig = "✓" if statistical_tests[detector].get("significant", False) else "✗"

    print(f"| {detector:<8} | {b_f1:.4f}±{b_std:.4f} | {l_f1:.4f}±{l_std:.4f} | {imp:+.1f}% | {p_val:.4f} | {sig} |")

print("\n" + "=" * 70)
print("Phase 3.2 Extended Complete!")
print("=" * 70)
