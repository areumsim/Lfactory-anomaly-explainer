"""Multi-seed experiment framework for statistical validation.

Usage:
    python scripts/multi_seed_experiment.py \
        --datasets synthetic SKAB SMD AIHub71802 \
        --detectors rule ml hybrid speccnn \
        --seeds 10

    # Run on all SKAB/SMD files with parallelism:
    python scripts/multi_seed_experiment.py \
        --datasets SKAB SMD --detectors rule ml --seeds 3 --all-files --parallel 4

    # Limit data size for faster runs:
    python scripts/multi_seed_experiment.py \
        --datasets SMD --detectors rule ml --seeds 5 --sample-limit 5000

This script runs experiments with multiple random seeds to ensure reproducibility
and compute confidence intervals.
"""
import subprocess
import argparse
import json
import csv
import sys
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_skab_file_count(data_root):
    """Get number of SKAB files for --all-files mode."""
    try:
        from experiments.data.loader_skab import list_all_files
        files = list_all_files(data_root)
        return len(files)
    except Exception as e:
        print(f"[WARN] Could not list SKAB files: {e}")
        return 0


def _get_smd_file_count(data_root):
    """Get number of SMD test files for --all-files mode."""
    test_dir = os.path.join(data_root, "SMD", "test")
    if not os.path.isdir(test_dir):
        return 0
    return len([f for f in os.listdir(test_dir) if f.endswith((".txt", ".csv"))])


def _extract_metrics(result):
    """Extract metrics from subprocess stdout (JSON)."""
    extracted = {}
    if result.returncode == 0:
        try:
            out = json.loads(result.stdout)
            em = out.get("metrics", {})
            extracted = {
                "f1": em.get("f1", 0.0),
                "auc_pr": em.get("auc_pr", 0.0),
                "auc_roc": em.get("auc_roc", 0.0),
                "optimal_f1": em.get("optimal_f1", 0.0),
                "precision": em.get("precision", 0.0),
                "recall": em.get("recall", 0.0),
            }
        except (json.JSONDecodeError, Exception):
            pass
    return extracted


def _run_single_experiment(cmd, run_info):
    """Run a single experiment subprocess and return result dict."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        extracted = _extract_metrics(result)
        return {**run_info, "success": result.returncode == 0, **extracted}
    except subprocess.TimeoutExpired:
        return {**run_info, "success": False, "error": "timeout"}
    except Exception as e:
        return {**run_info, "success": False, "error": str(e)}


def _aggregate_metrics(results):
    """Group by (dataset, detector, ml_method), compute mean+-std for each metric."""
    groups = defaultdict(list)
    for r in results:
        if r.get("success") and r.get("auc_pr") is not None:
            key = (r["dataset"], r["detector"], r.get("ml_method") or "")
            groups[key].append(r)

    summary = []
    for (ds, det, ml), runs in sorted(groups.items()):
        row = {"dataset": ds, "detector": det, "ml_method": ml, "n_runs": len(runs)}
        for metric in ["f1", "auc_pr", "auc_roc", "optimal_f1"]:
            vals = [r.get(metric, 0.0) for r in runs]
            mu = sum(vals) / len(vals) if vals else 0.0
            var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1) if len(vals) > 1 else 0.0
            row[f"{metric}_mean"] = round(mu, 4)
            row[f"{metric}_std"] = round(var ** 0.5, 4)
        summary.append(row)

    # Save CSV + JSON
    Path("runs").mkdir(parents=True, exist_ok=True)
    csv_path = Path("runs/multi_seed_metric_summary.csv")
    if summary:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    with open(Path("runs/multi_seed_metric_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metric summary: {csv_path}")
    return summary


def run_multi_seed(datasets, detectors, num_seeds=10, ml_methods=None,
                   all_files=False, data_root="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d",
                   sample_limit=0, parallel=1, device="auto"):
    """Run experiments with multiple seeds."""
    seeds = [42 + i * 100 for i in range(num_seeds)]
    ml_methods = ml_methods or ["knn", "isolation_forest", "lstm_ae"]

    # Build all jobs first
    jobs = []  # list of (cmd, run_info)

    for dataset in datasets:
        # Determine file indices to iterate over
        if all_files and dataset == "SKAB":
            file_count = _get_skab_file_count(data_root)
            file_indices = list(range(file_count)) if file_count > 0 else [-1]
            print(f"[SKAB] Running on all {file_count} files")
        elif all_files and dataset == "SMD":
            file_count = _get_smd_file_count(data_root)
            file_indices = list(range(file_count)) if file_count > 0 else [-1]
            print(f"[SMD] Running on all {file_count} files")
        elif all_files and dataset == "AIHub71802":
            from experiments.data.loader_aihub_71802 import _list_data_zips
            aihub_dir = os.path.join(data_root, "manufacturing_transport_71802", "Validation", "data")
            file_count = len(_list_data_zips(aihub_dir))
            file_indices = list(range(file_count)) if file_count > 0 else [-1]
            print(f"[AIHub71802] Running on all {file_count} sessions")
        elif all_files and dataset == "SWaT":
            # SWaT has a single test file; no file iteration needed
            file_indices = [-1]
            print(f"[SWaT] Running on single test file (all 51 sensors)")
        else:
            file_indices = [-1]  # default: single file

        for file_index in file_indices:
            for detector in detectors:
                methods_to_run = ml_methods if detector == "ml" else [None]
                for ml_method in methods_to_run:
                    for seed in seeds:
                        fi_tag = f"_file{file_index}" if file_index >= 0 else ""
                        if ml_method:
                            run_id = f"multi_seed_{detector}_{ml_method}_{dataset}{fi_tag}_seed{seed}"
                        else:
                            run_id = f"multi_seed_{detector}_{dataset}{fi_tag}_seed{seed}"

                        cmd = [
                            "python3", "-m", "experiments.main_experiment",
                            "--dataset", dataset,
                            "--detector", detector,
                            "--seed", str(seed),
                            "--run-id", run_id,
                            "--calibrate", "platt",
                            "--cost-optimize",
                        ]

                        if ml_method:
                            cmd.extend(["--ml-method", ml_method])
                        if dataset != "synthetic":
                            cmd.extend(["--data-root", data_root])
                        if file_index >= 0:
                            cmd.extend(["--file-index", str(file_index)])
                        if sample_limit > 0:
                            cmd.extend(["--sample-limit", str(sample_limit)])
                        if device != "auto":
                            cmd.extend(["--device", device])

                        run_info = {
                            "dataset": dataset,
                            "detector": detector,
                            "ml_method": ml_method,
                            "file_index": file_index if file_index >= 0 else None,
                            "seed": seed,
                            "run_id": run_id,
                        }
                        jobs.append((cmd, run_info))

    print(f"\nTotal jobs: {len(jobs)}, parallel workers: {parallel}")

    # Execute jobs
    results = []
    completed = 0

    if parallel <= 1:
        # Sequential execution
        for cmd, run_info in jobs:
            completed += 1
            print(f"[{completed}/{len(jobs)}] {run_info['run_id']}")
            result = _run_single_experiment(cmd, run_info)
            results.append(result)
            if not result.get("success"):
                print(f"  FAILED: {result.get('error', 'unknown')}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for cmd, run_info in jobs:
                future = executor.submit(_run_single_experiment, cmd, run_info)
                futures[future] = run_info

            for future in as_completed(futures):
                completed += 1
                result = future.result()
                results.append(result)
                status = "OK" if result.get("success") else "FAIL"
                if completed % 50 == 0 or completed == len(jobs):
                    print(f"  Progress: {completed}/{len(jobs)} ({status}: {result['run_id']})")

    # Save summary
    output_file = Path("runs/multi_seed_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # If --all-files was used, also generate per-file CSV summary
    if all_files:
        _save_per_file_csv(results)

    # Aggregate metrics (mean+-std per detector configuration)
    metric_summary = _aggregate_metrics(results)

    print(f"\nMulti-seed experiments complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")

    if metric_summary:
        print(f"\n{'Dataset':<12} {'Detector':<10} {'ML':<16} {'N':>4} {'AUC-PR':>12} {'Opt-F1':>12} {'F1':>12}")
        print("-" * 80)
        for row in metric_summary:
            print(f"{row['dataset']:<12} {row['detector']:<10} {row['ml_method']:<16} {row['n_runs']:>4} "
                  f"{row['auc_pr_mean']:.4f}±{row['auc_pr_std']:.4f} "
                  f"{row['optimal_f1_mean']:.4f}±{row['optimal_f1_std']:.4f} "
                  f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}")

    return results


def _save_per_file_csv(results):
    """Aggregate per-file results into a CSV summary."""
    csv_path = Path("runs/multi_seed_per_file_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by (dataset, detector, ml_method, file_index)
    groups = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["detector"], r.get("ml_method"), r.get("file_index"))
        groups[key].append(r)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "detector", "ml_method", "file_index", "num_seeds", "success_rate"])
        for (ds, det, ml, fi), runs in sorted(groups.items()):
            n_success = sum(1 for r in runs if r["success"])
            writer.writerow([ds, det, ml or "", fi if fi is not None else "", len(runs), f"{n_success/len(runs):.2f}"])

    print(f"Per-file summary saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--datasets", nargs="+", default=["synthetic"], help="Datasets to run")
    parser.add_argument("--detectors", nargs="+", default=["rule", "ml"], help="Detectors to run")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--ml-methods", nargs="+", default=["knn", "isolation_forest"], help="ML methods (if detector=ml)")
    parser.add_argument("--all-files", action="store_true", help="For SKAB/SMD: iterate over all files instead of just the first")
    parser.add_argument("--data-root", type=str, default="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d", help="Data root directory")
    parser.add_argument("--sample-limit", type=int, default=0, help="If >0, limit number of data points per file (speeds up large datasets)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (default: 1 = sequential)")
    parser.add_argument("--device", type=str, default="auto", help="Device for PyTorch models (auto/cuda:0/cuda:1/cpu)")

    args = parser.parse_args()
    run_multi_seed(args.datasets, args.detectors, args.seeds, args.ml_methods,
                   all_files=args.all_files, data_root=args.data_root,
                   sample_limit=args.sample_limit, parallel=args.parallel,
                   device=args.device)
