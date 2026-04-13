"""Aggregate all run.json files into a deduplicated clean results file.

Reads all runs/*/run.json, filters for valid results (optimal_f1 present,
label_rate > 0 unless synthetic), deduplicates by (dataset, detector, seed,
meta.path) keeping the newest run, and outputs:
  - runs/all_results_clean.json  (flat list of result dicts)
  - runs/results_summary.csv     (per-dataset/detector aggregation)

Usage:
    python scripts/aggregate_results.py [--runs-dir runs]
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _parse_run_dir_timestamp(dirname: str) -> str:
    """Extract timestamp portion from run directory name for sorting.

    Directory names follow pattern: DATASET_YYYYMMDD_HHMMSS_...
    Returns the YYYYMMDD_HHMMSS portion, or empty string if unparseable.
    """
    parts = dirname.split("_")
    if len(parts) >= 3:
        # Try to find YYYYMMDD pattern
        for i, p in enumerate(parts):
            if len(p) == 8 and p.isdigit():
                ts = p
                if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                    ts += "_" + parts[i + 1]
                return ts
    return ""


def aggregate(runs_dir: str = "runs") -> list[dict]:
    """Aggregate and deduplicate all run.json results."""
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        print(f"ERROR: runs directory not found: {runs_dir}", file=sys.stderr)
        return []

    # Collect all run.json files with filter statistics
    raw_entries = []
    filter_stats = {
        "total_files": 0,
        "skipped_parse_error": 0,
        "skipped_no_optimal_f1": 0,
        "skipped_label_rate_zero": 0,
        "skipped_too_short": 0,
        "passed_filters": 0,
    }
    for run_json in sorted(runs_path.glob("*/run.json")):
        filter_stats["total_files"] += 1
        try:
            with open(run_json) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            filter_stats["skipped_parse_error"] += 1
            print(f"WARN: skipping {run_json}: {e}", file=sys.stderr)
            continue

        meta = data.get("meta", {})
        metrics = data.get("metrics", {})

        # Skip if no optimal_f1 (pre-3rd-review results)
        if "optimal_f1" not in metrics:
            filter_stats["skipped_no_optimal_f1"] += 1
            continue

        dataset = meta.get("dataset", "unknown")
        label_rate = meta.get("label_rate", 0.0)

        # Skip label_rate=0 files (except synthetic which generates labels)
        if label_rate == 0 and dataset != "synthetic":
            filter_stats["skipped_label_rate_zero"] += 1
            continue

        # Skip extremely short series (likely broken data loads)
        num_points = meta.get("num_points", 0)
        if num_points < 10:
            filter_stats["skipped_too_short"] += 1
            continue

        filter_stats["passed_filters"] += 1

        # Determine detector method
        det_info = data.get("detector", {})
        detector_method = det_info.get("method", "unknown")

        # Extract seed from run_id or directory name
        run_id = data.get("run_id", run_json.parent.name)
        seed = data.get("seed", None)
        if seed is None:
            # Try to parse from run_id
            for part in run_id.split("_"):
                if part.startswith("seed") and part[4:].isdigit():
                    seed = int(part[4:])
                    break
            if seed is None:
                seed = 0

        # Use full file path for dedup (not basename)
        file_path = meta.get("path", meta.get("file", "unknown"))

        entry = {
            "dataset": dataset,
            "detector": detector_method,
            "seed": seed,
            "file_path": file_path,
            "file": meta.get("file", ""),
            "run_dir": str(run_json.parent.name),
            "run_dir_ts": _parse_run_dir_timestamp(run_json.parent.name),
            "num_points": meta.get("num_points", 0),
            "label_rate": label_rate,
            "f1": metrics.get("f1", 0.0),
            "optimal_f1": metrics.get("optimal_f1", 0.0),
            "auc_pr": metrics.get("auc_pr", 0.0),
            "auc_roc": metrics.get("auc_roc", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "optimal_f1_threshold": metrics.get("optimal_f1_threshold", 0.0),
            "optimal_f1_precision": metrics.get("optimal_f1_precision", 0.0),
            "optimal_f1_recall": metrics.get("optimal_f1_recall", 0.0),
        }
        raw_entries.append(entry)

    print(f"\n--- Filter Statistics ---", file=sys.stderr)
    for k, v in filter_stats.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"Found {len(raw_entries)} valid run.json files (with optimal_f1, label_rate>0)", file=sys.stderr)

    # Deduplicate: keep newest run per (dataset, detector, seed, file_path)
    dedup: dict[tuple, dict] = {}
    for entry in raw_entries:
        key = (entry["dataset"], entry["detector"], entry["seed"], entry["file_path"])
        existing = dedup.get(key)
        if existing is None or entry["run_dir_ts"] > existing["run_dir_ts"]:
            dedup[key] = entry

    results = sorted(dedup.values(), key=lambda r: (r["dataset"], r["detector"], r["seed"], r["file_path"]))
    n_duplicates = len(raw_entries) - len(results)
    filter_stats["duplicates_removed"] = n_duplicates
    filter_stats["final_unique"] = len(results)

    print(f"After dedup: {len(results)} unique entries (removed {n_duplicates} duplicates)", file=sys.stderr)

    # Remove internal fields
    for r in results:
        r.pop("run_dir_ts", None)

    # Save clean results with filter statistics
    clean_path = runs_path / "all_results_clean.json"
    output_data = {
        "filter_statistics": filter_stats,
        "results": results,
    }
    with open(clean_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved: {clean_path}", file=sys.stderr)

    # Generate summary CSV
    _generate_summary(results, runs_path / "results_summary.csv")

    return results


def _generate_summary(results: list[dict], output_path: Path):
    """Generate per-dataset/detector summary statistics."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        grouped[(r["dataset"], r["detector"])].append(r)

    rows = []
    for (dataset, detector), entries in sorted(grouped.items()):
        n = len(entries)
        auc_prs = [e["auc_pr"] for e in entries]
        opt_f1s = [e["optimal_f1"] for e in entries]
        f1s = [e["f1"] for e in entries]

        row = {
            "dataset": dataset,
            "detector": detector,
            "n": n,
            "auc_pr_mean": f"{mean(auc_prs):.4f}",
            "auc_pr_std": f"{stdev(auc_prs):.4f}" if n > 1 else "N/A",
            "optimal_f1_mean": f"{mean(opt_f1s):.4f}",
            "optimal_f1_std": f"{stdev(opt_f1s):.4f}" if n > 1 else "N/A",
            "f1_mean": f"{mean(f1s):.4f}",
            "f1_std": f"{stdev(f1s):.4f}" if n > 1 else "N/A",
        }
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and deduplicate run results")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subdirectories")
    args = parser.parse_args()
    aggregate(args.runs_dir)
