"""Multi-seed experiment framework for statistical validation.

Usage:
    python scripts/multi_seed_experiment.py \
        --datasets synthetic SKAB SMD AIHub71802 \
        --detectors rule ml hybrid speccnn \
        --seeds 10

This script runs experiments with multiple random seeds to ensure reproducibility
and compute confidence intervals.
"""
import subprocess
import argparse
import json
from pathlib import Path


def run_multi_seed(datasets, detectors, num_seeds=10, ml_methods=None):
    """Run experiments with multiple seeds."""
    seeds = [42 + i * 100 for i in range(num_seeds)]
    ml_methods = ml_methods or ["knn", "isolation_forest", "lstm_ae"]

    results = []

    for dataset in datasets:
        for detector in detectors:
            if detector == "ml":
                # Run for each ML method
                for ml_method in ml_methods:
                    for seed in seeds:
                        run_id = f"multi_seed_{detector}_{ml_method}_{dataset}_seed{seed}"
                        print(f"Running: {run_id}")

                        cmd = [
                            "python3", "-m", "experiments.main_experiment",
                            "--dataset", dataset,
                            "--detector", detector,
                            "--ml-method", ml_method,
                            "--seed", str(seed),
                            "--run-id", run_id,
                            "--calibrate", "platt",
                            "--cost-optimize",
                        ]

                        if dataset != "synthetic":
                            cmd.extend(["--data-root", "/workspace/data1/arsim/LFactory_d"])

                        try:
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                            results.append({
                                "dataset": dataset,
                                "detector": detector,
                                "ml_method": ml_method if detector == "ml" else None,
                                "seed": seed,
                                "run_id": run_id,
                                "success": result.returncode == 0,
                                "stdout": result.stdout[:500],  # Truncate
                            })
                        except Exception as e:
                            results.append({
                                "dataset": dataset,
                                "detector": detector,
                                "ml_method": ml_method if detector == "ml" else None,
                                "seed": seed,
                                "run_id": run_id,
                                "success": False,
                                "error": str(e),
                            })
            else:
                # Non-ML detectors
                for seed in seeds:
                    run_id = f"multi_seed_{detector}_{dataset}_seed{seed}"
                    print(f"Running: {run_id}")

                    cmd = [
                        "python3", "-m", "experiments.main_experiment",
                        "--dataset", dataset,
                        "--detector", detector,
                        "--seed", str(seed),
                        "--run-id", run_id,
                        "--calibrate", "platt",
                        "--cost-optimize",
                    ]

                    if dataset != "synthetic":
                        cmd.extend(["--data-root", "/workspace/data1/arsim/LFactory_d"])

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        results.append({
                            "dataset": dataset,
                            "detector": detector,
                            "ml_method": None,
                            "seed": seed,
                            "run_id": run_id,
                            "success": result.returncode == 0,
                        })
                    except Exception as e:
                        results.append({
                            "dataset": dataset,
                            "detector": detector,
                            "ml_method": None,
                            "seed": seed,
                            "run_id": run_id,
                            "success": False,
                            "error": str(e),
                        })

    # Save summary
    output_file = Path("runs/multi_seed_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Multi-seed experiments complete!")
    print(f"📊 Results saved to: {output_file}")
    print(f"📈 Total runs: {len(results)}")
    print(f"✅ Successful: {sum(1 for r in results if r['success'])}")
    print(f"❌ Failed: {sum(1 for r in results if not r['success'])}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--datasets", nargs="+", default=["synthetic"], help="Datasets to run")
    parser.add_argument("--detectors", nargs="+", default=["rule", "ml"], help="Detectors to run")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--ml-methods", nargs="+", default=["knn", "isolation_forest"], help="ML methods (if detector=ml)")

    args = parser.parse_args()
    run_multi_seed(args.datasets, args.detectors, args.seeds, args.ml_methods)
