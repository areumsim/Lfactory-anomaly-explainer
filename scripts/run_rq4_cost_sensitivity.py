"""RQ4: Cost sensitivity analysis.

Sweeps FN/FP cost ratios across datasets and detectors to analyze
how optimal thresholds and metrics change with cost assumptions.

Usage:
    python scripts/run_rq4_cost_sensitivity.py --seeds 5 --data-root /workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d
"""
import argparse
import json
import subprocess
import sys
import os
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.data import data_router
from experiments import cost_threshold


def run_rq4_cost_sensitivity(
    seeds=5,
    data_root="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d",
    datasets_cfg="experiments/data/datasets.yaml",
):
    """Run RQ4 cost sensitivity sweep."""
    datasets = ["SKAB", "SMD", "synthetic"]
    detectors = ["rule", "ml", "hybrid", "speccnn"]
    # Note: we don't need to re-run the detector for each ratio.
    # We can run the detector once to get scores, then sweep ratios analytically.
    ratios = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    seed_list = [42 + i * 100 for i in range(seeds)]

    all_results = []
    dataset_characteristics = {}

    for dataset in datasets:
        for detector in detectors:
            for seed in seed_list:
                run_id = f"rq4_{detector}_{dataset}_seed{seed}"
                print(f"Running detection: {run_id}")

                # Run detector to get scores
                cmd = [
                    "python3", "-m", "experiments.main_experiment",
                    "--dataset", dataset,
                    "--detector", detector,
                    "--seed", str(seed),
                    "--run-id", run_id,
                    "--cost-optimize",
                    "--costs", "0,1,5,0",
                ]

                if dataset != "synthetic":
                    cmd.extend(["--data-root", data_root])

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                    if result.returncode != 0:
                        print(f"  FAILED: {result.stderr[:200]}")
                        continue

                    out = json.loads(result.stdout)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

                # Read saved predictions to perform actual cost sweep
                det_metrics = out.get("metrics", {})

                # Find the run directory to extract preds.csv
                scores_list, labels_list = [], []
                run_dir_candidates = sorted(Path("runs").glob(f"*{run_id}*"))
                if run_dir_candidates:
                    preds_csv = run_dir_candidates[-1] / "preds.csv"
                    if preds_csv.exists():
                        import csv as csv_mod
                        with open(preds_csv) as csvf:
                            reader = csv_mod.DictReader(csvf)
                            for row in reader:
                                scores_list.append(float(row["score"]))
                                labels_list.append(int(row["label"]))

                if scores_list and labels_list and sum(labels_list) > 0:
                    sweep = cost_threshold.cost_sensitivity_sweep(
                        labels_list, scores_list, ratios=ratios,
                    )
                    for sr in sweep:
                        all_results.append({
                            "dataset": dataset,
                            "detector": detector,
                            "seed": seed,
                            "ratio": sr["ratio"],
                            "optimal_threshold": sr["optimal_threshold"],
                            "optimal_cost": sr["optimal_cost"],
                            "precision": sr["precision"],
                            "recall": sr["recall"],
                            "f1": sr["f1"],
                            "auc_pr": det_metrics.get("auc_pr", 0.0),
                        })
                else:
                    # Fallback: no valid scores/labels
                    for ratio in ratios:
                        all_results.append({
                            "dataset": dataset,
                            "detector": detector,
                            "seed": seed,
                            "ratio": ratio,
                            "optimal_threshold": 0.0,
                            "optimal_cost": 0.0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": det_metrics.get("f1", 0.0),
                            "auc_pr": det_metrics.get("auc_pr", 0.0),
                        })

    # Save raw results
    output_dir = Path("runs/rq4_cost_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate heatmap data and guideline table
    _generate_heatmap(all_results, output_dir)
    _generate_guideline_table(all_results, output_dir)

    print(f"\nRQ4 cost sensitivity analysis complete!")
    print(f"Results saved to: {output_dir}")
    return all_results


def _generate_heatmap(results, output_dir):
    """Generate heatmap data: (dataset x detector x ratio) -> optimal cost."""
    grouped = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["detector"], r.get("ratio", 5.0))
        grouped[key].append(r)

    lines = []
    lines.append("# RQ4: Cost Sensitivity Heatmap")
    lines.append("")
    lines.append("| Dataset | Detector | Ratio | AUC-PR | Optimal Cost | F1@Opt | Precision | Recall |")
    lines.append("|---------|----------|-------|--------|--------------|--------|-----------|--------|")

    for dataset in ["SKAB", "SMD", "synthetic"]:
        for detector in ["rule", "ml", "hybrid", "speccnn"]:
            for ratio in [1.0, 5.0, 10.0, 50.0]:
                key = (dataset, detector, ratio)
                runs = grouped.get(key, [])
                if not runs:
                    continue

                def _mean(vals):
                    return sum(vals) / len(vals) if vals else 0.0

                auc = _mean([r.get("auc_pr", 0.0) for r in runs])
                opt_cost = _mean([r.get("optimal_cost", 0.0) for r in runs])
                f1 = _mean([r.get("f1", 0.0) for r in runs])
                prec = _mean([r.get("precision", 0.0) for r in runs])
                rec = _mean([r.get("recall", 0.0) for r in runs])

                lines.append(f"| {dataset} | {detector} | {ratio:.0f}:1 | {auc:.3f} | {opt_cost:.3f} | {f1:.3f} | {prec:.3f} | {rec:.3f} |")

    text = "\n".join(lines)
    with open(output_dir / "heatmap.md", "w") as f:
        f.write(text)
    print(text)


def _generate_guideline_table(results, output_dir):
    """Generate practitioner guideline: dataset characteristics -> recommended cost ratio."""
    lines = []
    lines.append("# RQ4: Practitioner Cost Ratio Guidelines")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("| Scenario | Recommended FN:FP Ratio | Rationale |")
    lines.append("|----------|------------------------|-----------|")
    lines.append("| Safety-critical (valve failure) | 10:1 to 50:1 | Missing anomalies risks equipment damage |")
    lines.append("| Quality monitoring | 5:1 to 10:1 | Balance between alert fatigue and missed defects |")
    lines.append("| Exploratory analysis | 1:1 to 2:1 | Minimize total errors for investigation |")
    lines.append("| High-SNR signals | 2:1 to 5:1 | Clear anomalies are easy to detect, focus on precision |")
    lines.append("| Low-SNR signals | 10:1 to 20:1 | Noisy data needs conservative detection |")
    lines.append("")
    lines.append("## Dataset-specific Results")
    lines.append("")

    # Add dataset-specific analysis
    grouped = defaultdict(list)
    for r in results:
        grouped[r["dataset"]].append(r)

    for dataset in ["SKAB", "SMD", "synthetic"]:
        runs = grouped.get(dataset, [])
        if not runs:
            continue
        avg_auc = sum(r.get("auc_pr", 0.0) for r in runs) / len(runs)
        # Cost at 5:1 ratio
        runs_5 = [r for r in runs if r.get("ratio") == 5.0]
        avg_cost = sum(r.get("optimal_cost", 0.0) for r in runs_5) / len(runs_5) if runs_5 else 0.0
        lines.append(f"### {dataset}")
        lines.append(f"- Average AUC-PR: {avg_auc:.3f}")
        lines.append(f"- Average optimal cost at 5:1: {avg_cost:.3f}")
        lines.append("")

    text = "\n".join(lines)
    with open(output_dir / "guidelines.md", "w") as f:
        f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ4: Cost sensitivity analysis")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--data-root", type=str, default="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d", help="Data root")
    parser.add_argument("--datasets-cfg", type=str, default="experiments/data/datasets.yaml")

    args = parser.parse_args()
    run_rq4_cost_sensitivity(seeds=args.seeds, data_root=args.data_root, datasets_cfg=args.datasets_cfg)
