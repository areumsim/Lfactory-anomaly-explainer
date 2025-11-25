"""
Test LLM Parameter Advisor

Phase 3.1: Test parameter suggestion system
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key from .env file
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

import json
from pathlib import Path
from experiments.llm_parameter_advisor import LLMParameterAdvisor, load_baseline_results

print("=" * 70)
print("Phase 3.1: LLM Parameter Advisor Test")
print("=" * 70)

# Initialize advisor
advisor = LLMParameterAdvisor(
    model="gpt-4o-mini",
    temperature=0.3
)

# Define test cases
test_cases = [
    {
        "dataset": "SKAB",
        "detector": "lstm_autoencoder",
        "params": {
            "sequence_length": 50,
            "latent_dim": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "quantile": 0.95
        },
        "metrics": {
            "f1": 0.087,
            "precision": 0.082,
            "recall": 0.412,
            "auc_pr": 0.338
        },
        "data_stats": {
            "n_samples": 1147,
            "anomaly_rate": 0.35,
            "n_features": 8
        }
    },
    {
        "dataset": "SKAB",
        "detector": "isolation_forest",
        "params": {
            "window": 50,
            "contamination": 0.1,
            "n_estimators": 100
        },
        "metrics": {
            "f1": 0.033,
            "precision": 0.345,
            "recall": 0.017,
            "auc_pr": 0.201
        },
        "data_stats": {
            "n_samples": 1147,
            "anomaly_rate": 0.35,
            "n_features": 8
        }
    },
    {
        "dataset": "SKAB",
        "detector": "knn",
        "params": {
            "k": 10,
            "window": 50,
            "quantile": 0.99
        },
        "metrics": {
            "f1": 0.052,
            "precision": 0.420,
            "recall": 0.028,
            "auc_pr": 0.229
        },
        "data_stats": {
            "n_samples": 1147,
            "anomaly_rate": 0.35,
            "n_features": 8
        }
    }
]

# Generate suggestions
all_suggestions = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"Test {i}: {test['dataset']} / {test['detector']}")
    print("=" * 70)

    print(f"\nCurrent Performance:")
    print(f"  F1: {test['metrics']['f1']:.4f}")
    print(f"  Precision: {test['metrics']['precision']:.4f}")
    print(f"  Recall: {test['metrics']['recall']:.4f}")

    print(f"\nGenerating suggestions (calling LLM)...")

    suggestion = advisor.suggest_parameters(
        dataset=test["dataset"],
        detector=test["detector"],
        current_params=test["params"],
        current_metrics=test["metrics"],
        data_stats=test["data_stats"]
    )

    all_suggestions.append(suggestion)

    if "error" not in suggestion:
        print(f"\n--- Suggested Parameters ---")
        for key, val in suggestion["suggested_params"].items():
            current = test["params"].get(key, "N/A")
            if current != val:
                print(f"  {key}: {current} → {val}")
            else:
                print(f"  {key}: {val} (unchanged)")

        print(f"\n--- LLM Response ---")
        print(suggestion["raw_response"][:1500])
        if len(suggestion["raw_response"]) > 1500:
            print("...")
    else:
        print(f"Error: {suggestion['error']}")

# Save results
output_dir = Path("./runs/parameter_suggestions")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "llm_suggestions.json"
with open(output_file, "w") as f:
    json.dump(all_suggestions, f, indent=2)

print(f"\n{'='*70}")
print("Summary")
print("=" * 70)

print(f"\nGenerated {len(all_suggestions)} parameter suggestions")
print(f"Saved to: {output_file}")

# Summary table
print(f"\n{'Dataset':<10} {'Detector':<20} {'Current F1':<12} {'Status'}")
print("-" * 60)
for sug in all_suggestions:
    dataset = sug.get("dataset", "?")
    detector = sug.get("detector", "?")
    f1 = sug.get("current_metrics", {}).get("f1", 0)
    status = "✓ Suggested" if "error" not in sug else "✗ Error"
    print(f"{dataset:<10} {detector:<20} {f1:<12.4f} {status}")

print(f"\n{'='*70}")
print("Phase 3.1 Complete!")
print("=" * 70)
