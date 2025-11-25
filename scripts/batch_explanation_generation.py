"""
Phase 2.5: Batch Explanation Generation

Generates explanations for 10+ diverse anomaly samples across:
- Different datasets (SKAB)
- Different detectors (LSTM-AE, IsolationForest)
- Different anomaly types
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API key from environment
# export OPENAI_API_KEY="your-key" before running

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from experiments.llm_explainer import AnomalyExplainer

print("=" * 70)
print("Phase 2.5: Batch Explanation Generation")
print("=" * 70)

# Configuration
DATA_ROOT = "/workspace/data1/arsim/LFactory_d"
OUTPUT_DIR = Path("./runs/batch_explanations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load SKAB data
skab_file = f"{DATA_ROOT}/SKAB/valve1/0.csv"
print(f"\n1. Loading SKAB data...")
df = pd.read_csv(skab_file, sep=';')

sensor_cols = [c for c in df.columns if c not in ['datetime', 'anomaly', 'changepoint']]
multi_sensor_data = df[sensor_cols].values
labels = df['anomaly'].values.astype(int)

# Simulate reconstruction errors
mean_vals = np.mean(multi_sensor_data, axis=0)
std_vals = np.std(multi_sensor_data, axis=0) + 1e-6
reconstruction_errors = np.abs(multi_sensor_data - mean_vals) / std_vals

print(f"   {len(df)} samples, {len(sensor_cols)} sensors")

# Find experiment runs
runs_dir = Path("./runs")

# Get different types of runs
lstm_runs = sorted([d for d in runs_dir.iterdir()
                   if d.is_dir() and "SKAB" in d.name and "lstm" in d.name.lower()
                   and (d / "run.json").exists()])[:3]

iforest_runs = sorted([d for d in runs_dir.iterdir()
                      if d.is_dir() and "SKAB" in d.name and "isolation" in d.name.lower()
                      and (d / "run.json").exists()])[:3]

knn_runs = sorted([d for d in runs_dir.iterdir()
                  if d.is_dir() and "SKAB" in d.name and "knn" in d.name.lower()
                  and (d / "run.json").exists()])[:3]

all_runs = lstm_runs + iforest_runs + knn_runs
print(f"\n2. Found {len(all_runs)} experiment runs")
for r in all_runs:
    print(f"   - {r.name}")

if not all_runs:
    print("No experiment runs found!")
    sys.exit(1)

# Initialize explainer
print("\n3. Initializing explainer...")
explainer = AnomalyExplainer(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=2000
)

# Generate explanations
all_explanations = []
target_count = 12  # Generate 12 explanations

print(f"\n4. Generating {target_count} explanations...")
print("-" * 70)

explanation_count = 0
for run_dir in all_runs:
    if explanation_count >= target_count:
        break

    print(f"\nProcessing: {run_dir.name}")

    # Load predictions
    preds_df = pd.read_csv(run_dir / "preds.csv")
    detected = preds_df[preds_df['pred'] == 1].index.tolist()

    if not detected:
        print("  No anomalies detected, skipping...")
        continue

    # Align data
    data_len = min(len(multi_sensor_data), len(preds_df))

    # Select diverse anomaly indices
    sample_indices = []
    if len(detected) >= 4:
        # Select from different positions: start, middle, end
        sample_indices = [
            detected[0],
            detected[len(detected)//3],
            detected[2*len(detected)//3],
            detected[-1]
        ]
    else:
        sample_indices = detected[:4]

    for idx in sample_indices:
        if explanation_count >= target_count:
            break

        if idx >= data_len:
            continue

        print(f"  Generating explanation for index {idx}...", end=" ", flush=True)

        try:
            result = explainer.explain_anomaly(
                run_dir=run_dir,
                anomaly_idx=idx,
                context_window=20,
                include_model_interpretation=True,
                include_domain_knowledge=True,
                include_feature_importance=True,
                multi_sensor_data=multi_sensor_data[:data_len],
                reconstruction_errors=reconstruction_errors[:data_len],
                sensor_names=sensor_cols
            )

            # Add metadata
            result["batch_metadata"] = {
                "run_name": run_dir.name,
                "anomaly_idx": idx,
                "generated_at": datetime.now().isoformat(),
                "explanation_id": explanation_count + 1
            }

            all_explanations.append(result)
            explanation_count += 1
            print("✓")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

# Summary
print("\n" + "=" * 70)
print(f"Generated {len(all_explanations)} explanations")
print("=" * 70)

# Save all explanations
def convert_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types(v) for v in obj]
    return obj

output_file = OUTPUT_DIR / f"batch_explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w") as f:
    json.dump(convert_types(all_explanations), f, indent=2)

print(f"\n✅ Saved to: {output_file}")

# Print summary statistics
print("\n" + "-" * 70)
print("Summary by Detector Type:")
print("-" * 70)

by_detector = {}
for exp in all_explanations:
    method = exp["metadata"].get("method", "unknown")
    by_detector[method] = by_detector.get(method, 0) + 1

for method, count in by_detector.items():
    print(f"  {method}: {count} explanations")

# Print sample explanation
if all_explanations:
    print("\n" + "-" * 70)
    print("Sample Explanation (first one):")
    print("-" * 70)
    sample = all_explanations[0]
    print(f"\nRun: {sample['batch_metadata']['run_name']}")
    print(f"Index: {sample['batch_metadata']['anomaly_idx']}")
    print(f"\n{sample['explanation'][:1500]}...")

print("\n" + "=" * 70)
print("Phase 2.5 Complete!")
print("=" * 70)
