"""RQ2: Ensemble method comparison experiment.

Compares 4 ensemble methods (linear, product, max, learned) across datasets
with statistical significance testing.

Usage:
    python scripts/run_rq2_ensemble.py --seeds 20 --data-root /workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d
"""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_rq2_ensemble(seeds=20, data_root="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d"):
    """Run RQ2 ensemble comparison experiment."""
    methods = ["linear", "product", "max", "learned"]
    datasets = ["SKAB", "SMD", "synthetic"]
    seed_list = [42 + i * 100 for i in range(seeds)]

    results = []

    total = len(methods) * len(datasets) * len(seed_list)
    count = 0

    for dataset in datasets:
        for method in methods:
            for seed in seed_list:
                count += 1
                run_id = f"rq2_{method}_{dataset}_seed{seed}"
                print(f"[{count}/{total}] {run_id}")

                cmd = [
                    "python3", "-m", "experiments.main_experiment",
                    "--dataset", dataset,
                    "--detector", "hybrid",
                    "--ensemble-method", method,
                    "--seed", str(seed),
                    "--run-id", run_id,
                    "--calibrate", "platt",
                    "--cost-optimize",
                    "--apply-cost-threshold",
                ]

                if dataset != "synthetic":
                    cmd.extend(["--data-root", data_root])

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    # Try to parse output JSON for metrics
                    metrics = {}
                    if result.returncode == 0:
                        try:
                            out = json.loads(result.stdout)
                            metrics = out.get("metrics", {})
                            decision = out.get("decision", {}) or {}
                            metrics["expected_cost"] = decision.get("optimal_expected_cost", 0.0)
                        except (json.JSONDecodeError, Exception):
                            pass

                    results.append({
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "run_id": run_id,
                        "success": result.returncode == 0,
                        "auc_pr": metrics.get("auc_pr", 0.0),
                        "optimal_f1": metrics.get("optimal_f1", 0.0),
                        "ece": metrics.get("ece", 0.0),
                        "expected_cost": metrics.get("expected_cost", 0.0),
                        "f1": metrics.get("f1", 0.0),
                    })
                except Exception as e:
                    results.append({
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "run_id": run_id,
                        "success": False,
                        "error": str(e),
                    })

    # Save raw results
    output_dir = Path("runs/rq2_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate comparison table
    _generate_comparison_table(results, output_dir)
    # Statistical tests
    _run_statistical_tests(results, output_dir)

    print(f"\nRQ2 ensemble experiment complete!")
    print(f"Results saved to: {output_dir}")
    return results


def _generate_comparison_table(results, output_dir):
    """Generate comparison table: method x dataset -> mean +/- std for each metric."""
    import math

    grouped = defaultdict(list)
    for r in results:
        if r.get("success"):
            key = (r["dataset"], r["method"])
            grouped[key].append(r)

    lines = []
    lines.append("# RQ2: Ensemble Method Comparison")
    lines.append("")
    lines.append("| Dataset | Method | AUC-PR | ECE | Expected Cost | F1 |")
    lines.append("|---------|--------|--------|-----|---------------|-----|")

    for dataset in ["SKAB", "SMD", "synthetic"]:
        for method in ["linear", "product", "max", "learned"]:
            key = (dataset, method)
            runs = grouped.get(key, [])
            if not runs:
                lines.append(f"| {dataset} | {method} | - | - | - | - |")
                continue

            def _mean_std(vals):
                if not vals:
                    return 0.0, 0.0
                mu = sum(vals) / len(vals)
                if len(vals) > 1:
                    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
                    return mu, math.sqrt(var)
                return mu, 0.0

            auc_m, auc_s = _mean_std([r["auc_pr"] for r in runs])
            ece_m, ece_s = _mean_std([r["ece"] for r in runs])
            cost_m, cost_s = _mean_std([r.get("expected_cost", 0) for r in runs])
            f1_m, f1_s = _mean_std([r["f1"] for r in runs])

            lines.append(
                f"| {dataset} | {method} | "
                f"{auc_m:.3f}+/-{auc_s:.3f} | "
                f"{ece_m:.3f}+/-{ece_s:.3f} | "
                f"{cost_m:.3f}+/-{cost_s:.3f} | "
                f"{f1_m:.3f}+/-{f1_s:.3f} |"
            )

    table_text = "\n".join(lines)
    with open(output_dir / "comparison_table.md", "w") as f:
        f.write(table_text)
    print(table_text)


def _run_statistical_tests(results, output_dir):
    """Run Wilcoxon signed-rank test between method pairs."""
    from collections import defaultdict
    import math

    # Group by (dataset, seed) -> method -> metric
    by_seed = defaultdict(dict)
    for r in results:
        if r.get("success"):
            key = (r["dataset"], r["seed"])
            by_seed[key][r["method"]] = r

    methods = ["linear", "product", "max", "learned"]
    datasets = ["SKAB", "SMD", "synthetic"]
    metric = "auc_pr"

    test_results = []

    for dataset in datasets:
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                # Collect paired observations
                paired = []
                for seed in range(100):
                    key = (dataset, 42 + seed * 100)
                    if key in by_seed and m1 in by_seed[key] and m2 in by_seed[key]:
                        v1 = by_seed[key][m1].get(metric, 0.0)
                        v2 = by_seed[key][m2].get(metric, 0.0)
                        paired.append((v1, v2))

                if len(paired) < 5:
                    continue

                # Simple sign test (Wilcoxon approximation without scipy)
                diffs = [a - b for a, b in paired if abs(a - b) > 1e-10]
                if not diffs:
                    p_approx = 1.0
                else:
                    n_pos = sum(1 for d in diffs if d > 0)
                    n_total = len(diffs)
                    # Normal approximation to sign test
                    expected = n_total / 2.0
                    std_dev = math.sqrt(n_total) / 2.0
                    if std_dev > 0:
                        z = (n_pos - expected) / std_dev
                        # Two-tailed p-value approximation
                        p_approx = 2.0 * (1.0 - _norm_cdf(abs(z)))
                    else:
                        p_approx = 1.0

                mean_diff = sum(a - b for a, b in paired) / len(paired)
                test_results.append({
                    "dataset": dataset,
                    "method_1": m1,
                    "method_2": m2,
                    "metric": metric,
                    "n_pairs": len(paired),
                    "mean_diff": mean_diff,
                    "p_value_approx": p_approx,
                    "significant_005": p_approx < 0.05,
                })

    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Print significant results
    sig = [t for t in test_results if t["significant_005"]]
    if sig:
        print(f"\nStatistically significant pairs (p < 0.05):")
        for t in sig:
            print(f"  {t['dataset']}: {t['method_1']} vs {t['method_2']} "
                  f"(diff={t['mean_diff']:.4f}, p={t['p_value_approx']:.4f})")


def _norm_cdf(x):
    """Approximate normal CDF using error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


if __name__ == "__main__":
    import math

    parser = argparse.ArgumentParser(description="RQ2: Ensemble method comparison")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--data-root", type=str, default="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d", help="Data root")

    args = parser.parse_args()
    run_rq2_ensemble(seeds=args.seeds, data_root=args.data_root)
