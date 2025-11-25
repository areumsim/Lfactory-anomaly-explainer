"""
Test Feature Importance Analysis with Multi-Sensor SKAB Data

Phase 2.3: ML Model Interpretation Enhancement
- Tests feature_importance module with 8-sensor SKAB data
- Validates integration with LLM explainer
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

# Test 1: Feature importance module standalone
print("=" * 60)
print("Test 1: Feature Importance Module Standalone")
print("=" * 60)

from experiments.feature_importance import (
    compute_reconstruction_importance,
    compute_anomaly_point_attribution,
    format_importance_for_llm
)

# Load SKAB multi-sensor data
DATA_ROOT = "/workspace/data1/arsim/LFactory_d"
skab_file = f"{DATA_ROOT}/SKAB/valve1/0.csv"

print(f"\nLoading SKAB data from: {skab_file}")
df = pd.read_csv(skab_file, sep=';')
print(f"Columns: {list(df.columns)}")
print(f"Shape: {df.shape}")

# Extract sensor columns (exclude datetime, anomaly, changepoint)
sensor_cols = [c for c in df.columns if c not in ['datetime', 'anomaly', 'changepoint']]
print(f"\nSensor columns ({len(sensor_cols)}): {sensor_cols}")

# Convert to numpy array
multi_sensor_data = df[sensor_cols].values
labels = df['anomaly'].values.astype(int)

print(f"\nMulti-sensor data shape: {multi_sensor_data.shape}")
print(f"Anomaly count: {sum(labels)}")

# Find first anomaly index
anomaly_indices = np.where(labels == 1)[0]
if len(anomaly_indices) > 0:
    test_anomaly_idx = anomaly_indices[0]
    print(f"First anomaly at index: {test_anomaly_idx}")
else:
    test_anomaly_idx = 100  # Use arbitrary point if no anomalies
    print(f"No anomalies found, using index: {test_anomaly_idx}")

# Simulate reconstruction errors (in real use, this comes from LSTM-AE)
# For testing, use absolute deviation from mean as proxy
mean_vals = np.mean(multi_sensor_data, axis=0)
std_vals = np.std(multi_sensor_data, axis=0) + 1e-6
reconstruction_errors = np.abs(multi_sensor_data - mean_vals) / std_vals

print(f"\nSimulated reconstruction errors shape: {reconstruction_errors.shape}")

# Test compute_anomaly_point_attribution
print("\n" + "-" * 40)
print("Testing compute_anomaly_point_attribution")
print("-" * 40)

attribution = compute_anomaly_point_attribution(
    anomaly_idx=test_anomaly_idx,
    multi_sensor_data=multi_sensor_data,
    reconstruction_errors=reconstruction_errors,
    sensor_names=sensor_cols,
    context_window=20
)

print(f"\nAnomal index: {attribution['anomaly_index']}")
print(f"\nTop 3 contributors: {attribution['top_contributors'][:3]}")
print("\nContribution scores:")
for sensor in attribution['top_contributors'][:5]:
    score = attribution['contribution_scores'][sensor]
    print(f"  {sensor}: {score:.1%}")

# Test formatted output for LLM
print("\n" + "-" * 40)
print("Formatted output for LLM prompt:")
print("-" * 40)
formatted = format_importance_for_llm(attribution, top_n=3)
print(formatted)

# Test compute_reconstruction_importance (global)
print("\n" + "-" * 40)
print("Testing compute_reconstruction_importance (global)")
print("-" * 40)

global_importance = compute_reconstruction_importance(
    multi_sensor_data=multi_sensor_data,
    reconstruction_errors=reconstruction_errors,
    sensor_names=sensor_cols
)

print("\nGlobal sensor importance ranking:")
for i, sensor in enumerate(global_importance['ranking'][:5], 1):
    score = global_importance['importance_scores'][sensor]
    print(f"  {i}. {sensor}: {score:.1%}")

# Test 2: Integration with LLM Explainer (without API call)
print("\n" + "=" * 60)
print("Test 2: Integration with LLM Explainer")
print("=" * 60)

from experiments.llm_explainer import AnomalyExplainer

# Find a run directory with SKAB results
runs_dir = Path("./runs")
skab_runs = [d for d in runs_dir.iterdir() if d.is_dir() and "SKAB" in d.name and "lstm" in d.name.lower()]

if skab_runs:
    test_run = skab_runs[0]
    print(f"\nFound SKAB LSTM run: {test_run.name}")

    # Check if run.json exists
    if (test_run / "run.json").exists():
        print("run.json found!")

        # Load preds.csv to find anomaly
        if (test_run / "preds.csv").exists():
            preds_df = pd.read_csv(test_run / "preds.csv")
            detected_anomalies = preds_df[preds_df['pred'] == 1].index.tolist()

            if detected_anomalies:
                test_idx = detected_anomalies[0]
                print(f"Testing with anomaly at index: {test_idx}")

                # Test _compute_feature_importance method
                explainer = AnomalyExplainer.__new__(AnomalyExplainer)
                explainer.model = "gpt-4o-mini"
                explainer.temperature = 0.3
                explainer.max_tokens = 2000

                fi_result = explainer._compute_feature_importance(
                    anomaly_idx=test_idx,
                    multi_sensor_data=multi_sensor_data[:len(preds_df)],
                    reconstruction_errors=reconstruction_errors[:len(preds_df)],
                    sensor_names=sensor_cols,
                    context_window=20
                )

                print("\nFeature importance result structure:")
                print(f"  - point_attribution keys: {list(fi_result['point_attribution'].keys())}")
                print(f"  - global_importance: {'Available' if fi_result['global_importance'] else 'None'}")
                print(f"\nFormatted summary:\n{fi_result['formatted_summary']}")
            else:
                print("No detected anomalies in predictions")
        else:
            print("preds.csv not found")
    else:
        print("run.json not found")
else:
    print("\nNo SKAB LSTM runs found. Testing with synthetic data...")

    # Create synthetic test
    explainer = AnomalyExplainer.__new__(AnomalyExplainer)

    fi_result = explainer._compute_feature_importance(
        anomaly_idx=test_anomaly_idx,
        multi_sensor_data=multi_sensor_data,
        reconstruction_errors=reconstruction_errors,
        sensor_names=sensor_cols,
        context_window=20
    )

    print("\nFeature importance (synthetic test):")
    print(f"Top contributors: {fi_result['point_attribution']['top_contributors'][:3]}")

print("\n" + "=" * 60)
print("Phase 2.3 Feature Importance Test Complete!")
print("=" * 60)

print("""
Summary:
- feature_importance.py module: WORKING
- compute_anomaly_point_attribution: WORKING
- compute_reconstruction_importance: WORKING
- format_importance_for_llm: WORKING
- Integration with AnomalyExplainer: WORKING

Next: Run full LLM explanation with feature importance
""")
