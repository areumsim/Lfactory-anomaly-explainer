"""
Full Integration Test: LLM Explanation with Feature Importance

Phase 2.3: Complete test of anomaly explanation with multi-sensor attribution
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set API key from environment variable
# export OPENAI_API_KEY="your-api-key" before running this script

import numpy as np
import pandas as pd
import json
from pathlib import Path

from experiments.llm_explainer import AnomalyExplainer

print("=" * 70)
print("Full Integration Test: LLM + Feature Importance")
print("=" * 70)

# Load SKAB multi-sensor data
DATA_ROOT = "/workspace/data1/arsim/LFactory_d"
skab_file = f"{DATA_ROOT}/SKAB/valve1/0.csv"

print(f"\n1. Loading SKAB data from: {skab_file}")
df = pd.read_csv(skab_file, sep=';')

# Extract sensor columns
sensor_cols = [c for c in df.columns if c not in ['datetime', 'anomaly', 'changepoint']]
multi_sensor_data = df[sensor_cols].values
labels = df['anomaly'].values.astype(int)

print(f"   - {len(sensor_cols)} sensors: {sensor_cols}")
print(f"   - {len(df)} samples, {sum(labels)} anomalies")

# Simulate reconstruction errors
mean_vals = np.mean(multi_sensor_data, axis=0)
std_vals = np.std(multi_sensor_data, axis=0) + 1e-6
reconstruction_errors = np.abs(multi_sensor_data - mean_vals) / std_vals

# Find experiment run
runs_dir = Path("./runs")
skab_lstm_runs = sorted([d for d in runs_dir.iterdir()
                        if d.is_dir() and "SKAB" in d.name and "lstm" in d.name.lower()])

if not skab_lstm_runs:
    print("\nNo SKAB LSTM runs found!")
    sys.exit(1)

test_run = skab_lstm_runs[0]
print(f"\n2. Using experiment run: {test_run.name}")

# Load predictions
preds_df = pd.read_csv(test_run / "preds.csv")
detected_anomalies = preds_df[preds_df['pred'] == 1].index.tolist()

if not detected_anomalies:
    print("   No detected anomalies in this run!")
    sys.exit(1)

# Select test anomaly
test_anomaly_idx = detected_anomalies[min(5, len(detected_anomalies)-1)]  # 6th anomaly or last
print(f"   - Testing anomaly at index: {test_anomaly_idx}")

# Align data length
data_len = min(len(multi_sensor_data), len(preds_df))
multi_sensor_aligned = multi_sensor_data[:data_len]
recon_errors_aligned = reconstruction_errors[:data_len]

# Initialize explainer
print("\n3. Initializing AnomalyExplainer...")
explainer = AnomalyExplainer(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=2000
)

# Generate explanation with feature importance
print("\n4. Generating explanation with feature importance...")
print("   (This will call OpenAI API...)")

try:
    result = explainer.explain_anomaly(
        run_dir=test_run,
        anomaly_idx=test_anomaly_idx,
        context_window=20,
        include_model_interpretation=True,
        include_domain_knowledge=True,
        include_feature_importance=True,
        multi_sensor_data=multi_sensor_aligned,
        reconstruction_errors=recon_errors_aligned,
        sensor_names=sensor_cols
    )

    print("\n" + "=" * 70)
    print("LLM-Generated Explanation with Feature Importance")
    print("=" * 70)
    print(result["explanation"])

    # Print feature importance summary
    if result.get("feature_importance"):
        print("\n" + "-" * 70)
        print("Feature Importance Details")
        print("-" * 70)
        fi = result["feature_importance"]
        if "point_attribution" in fi:
            pa = fi["point_attribution"]
            print("\nTop contributing sensors at anomaly point:")
            for i, sensor in enumerate(pa["top_contributors"][:5], 1):
                score = pa["contribution_scores"].get(sensor, 0)
                print(f"  {i}. {sensor}: {score:.1%}")

    # Save result
    output_file = test_run / "llm_explanation_with_fi.json"
    with open(output_file, "w") as f:
        # Convert numpy types for JSON
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        json.dump(convert_types(result), f, indent=2)

    print(f"\n✅ Result saved to: {output_file}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Phase 2.3 Complete!")
print("=" * 70)
