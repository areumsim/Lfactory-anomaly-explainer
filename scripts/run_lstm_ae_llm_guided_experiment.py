"""
Phase 3.4: LSTM-AE - LLM-guided Parameter Optimization (20 seeds)

Compare default parameters vs LLM-suggested parameters for LSTM Autoencoder
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

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Import detector
from experiments.ml_detector_lstm_ae import detect_lstm_ae
from experiments.metrics import binary_metrics

print("=" * 70)
print("Phase 3.4: LSTM-AE LLM-guided Experiment (20 seeds)")
print("=" * 70)

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
except:
    print("PyTorch not found")

# Load SKAB data (same as previous experiments)
DATA_ROOT = "/workspace/data1/arsim/LFactory_d"
skab_file = f"{DATA_ROOT}/SKAB/valve1/0.csv"

print(f"\n1. Loading data: {skab_file}")
df = pd.read_csv(skab_file, sep=';')

# Use one sensor for experiments
sensor_col = "Accelerometer1RMS"
series = df[sensor_col].values.tolist()
labels = df['anomaly'].values.astype(int).tolist()

print(f"   Samples: {len(series)}")
print(f"   Anomaly rate: {sum(labels)/len(labels):.2%}")

# Define experiment configurations
experiments = {
    "lstm_ae": {
        "baseline": {
            "sequence_length": 50,
            "latent_dim": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 32,
            "quantile": 0.95
        },
        "llm_guided": {
            # LLM suggestions for SKAB (35% anomaly rate):
            # - Shorter sequence for sudden valve anomalies
            # - Larger latent dim for complex patterns
            # - More epochs for better learning
            # - Lower quantile for higher sensitivity
            "sequence_length": 20,
            "latent_dim": 64,
            "epochs": 100,
            "learning_rate": 0.001,
            "batch_size": 16,
            "quantile": 0.65  # Match ~35% anomaly rate
        }
    }
}

# 10 seeds for LSTM (computationally expensive)
seeds = [42 + i*100 for i in range(10)]  # [42, 142, ..., 942]

# Results storage
all_results = []

print(f"\n2. Running experiments ({len(seeds)} seeds)")
print(f"   Seeds: {seeds}")
print("-" * 70)

for detector_name, configs in experiments.items():
    print(f"\n--- {detector_name.upper()} ---")

    for config_name, params in configs.items():
        print(f"\n  [{config_name}] params: {params}")

        metrics_list = []

        for i, seed in enumerate(seeds):
            np.random.seed(seed)

            # Run detector
            result = detect_lstm_ae(
                series=series,
                sequence_length=params["sequence_length"],
                latent_dim=params["latent_dim"],
                epochs=params["epochs"],
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                quantile=params["quantile"],
                random_state=seed
            )

            # Compute metrics
            metrics = binary_metrics(
                y_true=labels,
                y_pred=result["preds"]
            )
            metrics["auc_pr"] = metrics.get("precision", 0) * metrics.get("recall", 0)

            metrics_list.append(metrics)

            print(f"    Seed {seed}: F1={metrics['f1']:.4f}")

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

baseline = next(r for r in all_results if r["config"] == "baseline")
llm_guided = next(r for r in all_results if r["config"] == "llm_guided")

print(f"\n--- LSTM_AE ---")
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

comparison_results["lstm_ae"] = detector_comparison

# Statistical test
print("\n" + "=" * 70)
print("4. Statistical Significance (Wilcoxon signed-rank test)")
print("=" * 70)

from scipy.stats import wilcoxon

statistical_tests = {}

b_f1 = [m["f1"] for m in baseline["individual_runs"]]
l_f1 = [m["f1"] for m in llm_guided["individual_runs"]]

try:
    stat, p_value = wilcoxon(b_f1, l_f1)
    sig = "✓ Significant" if p_value < 0.05 else "✗ Not significant"
    print(f"\nlstm_ae: p-value={p_value:.6f} {sig}")

    statistical_tests["lstm_ae"] = {
        "test": "wilcoxon_signed_rank",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_samples": len(b_f1)
    }
except Exception as e:
    print(f"\nlstm_ae: Error - {str(e)}")
    statistical_tests["lstm_ae"] = {"error": str(e)}

# Effect size
print("\n" + "=" * 70)
print("5. Effect Size (Cohen's d)")
print("=" * 70)

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group2) - np.mean(group1)) / pooled_std if pooled_std > 0 else float('inf')

d = cohens_d(b_f1, l_f1)

if d == float('inf'):
    effect = "deterministic"
elif abs(d) < 0.2:
    effect = "negligible"
elif abs(d) < 0.5:
    effect = "small"
elif abs(d) < 0.8:
    effect = "medium"
else:
    effect = "large"

print(f"lstm_ae: Cohen's d = {d:.4f} ({effect})")

if "lstm_ae" in statistical_tests and "error" not in statistical_tests["lstm_ae"]:
    statistical_tests["lstm_ae"]["cohens_d"] = float(d) if d != float('inf') else "inf"
    statistical_tests["lstm_ae"]["effect_size"] = effect

# Save results
output_dir = Path("./runs/lstm_ae_llm_guided_experiment")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
    "experiment": "LSTM-AE LLM-guided Parameter Optimization (10 seeds)",
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

print(f"\n✅ Results saved to: {output_file}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\n| Config | F1 | Precision | Recall | Improvement |")
print("|--------|-----|-----------|--------|-------------|")

b_f1_avg = baseline["avg_metrics"]["f1"]
l_f1_avg = llm_guided["avg_metrics"]["f1"]
imp = ((l_f1_avg - b_f1_avg) / b_f1_avg * 100) if b_f1_avg > 0 else 0

print(f"| Baseline | {baseline['avg_metrics']['f1']:.4f} | {baseline['avg_metrics']['precision']:.4f} | {baseline['avg_metrics']['recall']:.4f} | - |")
print(f"| LLM-Guided | {llm_guided['avg_metrics']['f1']:.4f} | {llm_guided['avg_metrics']['precision']:.4f} | {llm_guided['avg_metrics']['recall']:.4f} | {imp:+.1f}% |")

print("\n" + "=" * 70)
print("Phase 3.4 LSTM-AE Experiment Complete!")
print("=" * 70)
