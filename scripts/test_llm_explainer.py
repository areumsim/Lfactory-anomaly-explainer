"""
Test script for LLM-based anomaly explanation system

Tests the explanation system on real experimental results
"""
import sys
from pathlib import Path
import pandas as pd
import json

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.llm_explainer import AnomalyExplainer
from experiments.llm_config import test_api_connection


def find_true_anomalies(preds_csv: Path, top_n: int = 5) -> list:
    """
    Find indices of true anomalies (label=1) with highest anomaly scores
    """
    df = pd.read_csv(preds_csv)
    true_anomalies = df[df["label"] == 1].copy()

    if len(true_anomalies) == 0:
        print("Warning: No true anomalies found in this file!")
        # Fall back to top N predicted anomalies
        return df.nlargest(top_n, "score").index.tolist()

    # Get top N true anomalies by score
    top_anomalies = true_anomalies.nlargest(top_n, "score")
    return top_anomalies.index.tolist()


def main():
    print("=" * 80)
    print("LLM Anomaly Explanation System - Test")
    print("=" * 80)

    # Test API connection
    print("\n1. Testing OpenAI API connection...")
    if test_api_connection():
        print("   ✓ API connection successful!")
    else:
        print("   ✗ API connection failed!")
        return

    # Select a run to explain
    run_dir = Path("/workspace/arsim/LFactory/runs/SKAB_20251124_053427_multi_seed_ml_lstm_ae_SKAB_seed42")

    print(f"\n2. Loading run data from: {run_dir.name}")
    preds_csv = run_dir / "preds.csv"

    if not preds_csv.exists():
        print(f"   ✗ preds.csv not found in {run_dir}")
        return

    # Load run metadata
    with open(run_dir / "run.json", "r") as f:
        run_data = json.load(f)

    print(f"   Dataset: {run_data['meta']['dataset']}")
    print(f"   Scenario: {run_data['meta'].get('scenario', 'N/A')}")
    print(f"   Sensor: {run_data['meta']['value_col']}")
    print(f"   Detector: {run_data['detector']['method']}")
    print(f"   F1 Score: {run_data['metrics']['f1']:.4f}")

    # Find true anomalies to explain
    print("\n3. Finding anomalies to explain...")
    anomaly_indices = find_true_anomalies(preds_csv, top_n=3)
    print(f"   Found {len(anomaly_indices)} anomalies: {anomaly_indices}")

    # Initialize explainer
    print("\n4. Initializing LLM Explainer...")
    explainer = AnomalyExplainer()

    # Generate explanation for first anomaly
    print(f"\n5. Generating explanation for anomaly at index {anomaly_indices[0]}...")
    print("   (This may take 10-30 seconds...)\n")

    result = explainer.explain_anomaly(
        run_dir=run_dir,
        anomaly_idx=anomaly_indices[0],
        context_window=20,
        include_model_interpretation=True,
        include_domain_knowledge=True
    )

    # Display results
    print("=" * 80)
    print("EXPLANATION RESULT")
    print("=" * 80)

    def format_val(v, fmt=".6f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    print("\n### Anomaly Information ###")
    info = result["anomaly_info"]
    print(f"Dataset: {info['dataset']}")
    print(f"Sensor: {info['sensor']}")
    print(f"Index: {info['index']}")
    print(f"Value: {info['value']:.6f} (mean: {format_val(info['normal_value_mean'])}, std: {format_val(info['normal_value_std'])})")
    print(f"Anomaly Score: {info['score']:.6f} (mean: {format_val(info['normal_score_mean'])})")
    print(f"Deviation: {format_val(info['value_deviation_sigma'], '.2f')} sigma")
    print(f"True Label: {'Anomaly' if info['true_label'] == 1 else 'Normal'}")
    print(f"Predicted: {'Anomaly' if info['predicted_label'] == 1 else 'Normal'}")

    print("\n### LLM Explanation ###")
    print(result["explanation"])

    # Save to file
    output_file = run_dir / "llm_explanation_sample.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Test completed successfully!")
    print(f"✓ Full result saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
