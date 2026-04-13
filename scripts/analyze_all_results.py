"""Comprehensive analysis of all LLM explanation experiments.

Combines results from:
1. SWaT nosensor ablation (domain_only vs domain_nosensor)
2. FP samples (TP vs FN vs FP comparison)
3. Cross-model judge agreement (GPT-4o-mini judge vs GPT-4 judge)
4. Direction accuracy (overall and by condition)

Usage:
    python scripts/analyze_all_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def wilcoxon_signed_rank(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Simplified Wilcoxon signed-rank test (two-sided).
    Returns (statistic, approximate p-value).
    """
    diffs = [(xi - yi) for xi, yi in zip(x, y) if xi != yi]
    if not diffs:
        return 0, 1.0

    abs_diffs = [abs(d) for d in diffs]
    ranks = list(range(1, len(abs_diffs) + 1))
    # Simple ranking (no ties handling)
    sorted_indices = sorted(range(len(abs_diffs)), key=lambda i: abs_diffs[i])
    rank_map = [0] * len(abs_diffs)
    for rank_val, idx in enumerate(sorted_indices, 1):
        rank_map[idx] = rank_val

    w_plus = sum(rank_map[i] for i in range(len(diffs)) if diffs[i] > 0)
    w_minus = sum(rank_map[i] for i in range(len(diffs)) if diffs[i] < 0)
    w = min(w_plus, w_minus)

    n = len(diffs)
    # Normal approximation for p-value
    mean_w = n * (n + 1) / 4
    std_w = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if std_w == 0:
        return w, 1.0
    z = (w - mean_w) / std_w
    # Approximate two-sided p-value from z
    import math
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return w, p


def kruskal_wallis(groups: List[List[float]]) -> Tuple[float, float]:
    """Simplified Kruskal-Wallis H test."""
    all_vals = []
    for gi, group in enumerate(groups):
        for v in group:
            all_vals.append((v, gi))

    n = len(all_vals)
    if n < 3:
        return 0, 1.0

    sorted_vals = sorted(all_vals, key=lambda x: x[0])
    ranks = [0.0] * n
    for i, (v, gi) in enumerate(sorted_vals):
        ranks[i] = i + 1

    # Map ranks back
    group_ranks: Dict[int, List[float]] = {}
    for rank, (v, gi) in zip(ranks, sorted_vals):
        group_ranks.setdefault(gi, []).append(rank)

    k = len(groups)
    h = (12 / (n * (n + 1))) * sum(
        len(r) * (sum(r) / len(r) - (n + 1) / 2) ** 2
        for r in group_ranks.values()
    )

    # Approximate p-value (chi-squared with k-1 df)
    import math
    df = k - 1
    # Very rough chi-squared approximation
    if h <= 0:
        return 0, 1.0
    # Use incomplete gamma for better approximation
    p = math.exp(-h / 2)  # rough
    return h, p


def cliffs_delta(x: List[float], y: List[float]) -> float:
    """Cliff's delta effect size."""
    n_more = sum(1 for xi in x for yi in y if xi > yi)
    n_less = sum(1 for xi in x for yi in y if xi < yi)
    n = len(x) * len(y)
    return (n_more - n_less) / n if n > 0 else 0


def analyze_nosensor(results_path: str):
    """Analyze nosensor ablation results."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: SWaT Nosensor Ablation (Domain Confound Test)")
    print("=" * 70)

    data = load_results(results_path)
    results = data.get("results", [])

    conditions = ["baseline", "domain_only", "feature_only", "full", "domain_nosensor", "full_nosensor"]

    # Collect paired data
    pairs = {"domain_only": [], "domain_nosensor": [], "feature_only": [], "baseline": []}
    for r in results:
        for cond in pairs:
            if cond in r:
                pairs[cond].append(r[cond].get("faithfulness_keyword", 0))

    # Wilcoxon: domain_only vs domain_nosensor
    x = pairs["domain_only"]
    y = pairs["domain_nosensor"]
    w, p = wilcoxon_signed_rank(x, y)
    d = cliffs_delta(x, y)
    print(f"\ndomain_only vs domain_nosensor (N={len(x)}):")
    print(f"  domain_only mean:    {sum(x)/len(x):.3f}")
    print(f"  domain_nosensor mean: {sum(y)/len(y):.3f}")
    print(f"  Difference:          {(sum(x)/len(x) - sum(y)/len(y)):.3f}")
    print(f"  Wilcoxon W={w:.0f}, p={p:.6f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    print(f"  Cliff's delta: {d:.3f} ({'large' if abs(d) > 0.474 else 'medium' if abs(d) > 0.33 else 'small'})")

    # By sample mode
    by_mode = {}
    for r in results:
        mode = r["sample"].get("sample_mode", "tp")
        by_mode.setdefault(mode, {"domain_only": [], "domain_nosensor": []})
        if "domain_only" in r:
            by_mode[mode]["domain_only"].append(r["domain_only"].get("faithfulness_keyword", 0))
        if "domain_nosensor" in r:
            by_mode[mode]["domain_nosensor"].append(r["domain_nosensor"].get("faithfulness_keyword", 0))

    print(f"\nBy sample mode:")
    for mode, vals in sorted(by_mode.items()):
        do = vals["domain_only"]
        dn = vals["domain_nosensor"]
        if do and dn:
            diff = sum(do) / len(do) - sum(dn) / len(dn)
            print(f"  {mode.upper()}: domain_only={sum(do)/len(do):.3f}, domain_nosensor={sum(dn)/len(dn):.3f}, diff={diff:.3f}")

    # Direction accuracy
    dir_counts = {}
    for r in results:
        for cond in conditions:
            if cond not in r:
                continue
            strict = r[cond].get("faithfulness_strict", {})
            dc = strict.get("direction_correct")
            if cond not in dir_counts:
                dir_counts[cond] = {"Y": 0, "N": 0, "?": 0}
            if dc is True:
                dir_counts[cond]["Y"] += 1
            elif dc is False:
                dir_counts[cond]["N"] += 1
            else:
                dir_counts[cond]["?"] += 1

    print(f"\nDirection accuracy by condition:")
    for cond in conditions:
        if cond in dir_counts:
            c = dir_counts[cond]
            total = c["Y"] + c["N"]
            pct = c["Y"] / total * 100 if total > 0 else 0
            print(f"  {cond:20s}: {c['Y']}/{total} = {pct:.1f}% ({c['?']} ambiguous)")


def analyze_fp(fp_path: str, tp_fn_path: str):
    """Analyze FP samples vs TP/FN."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: TP vs FN vs FP Comparison")
    print("=" * 70)

    fp_data = load_results(fp_path)
    tp_fn_data = load_results(tp_fn_path)

    # Collect KW by sample mode
    by_mode = {}
    for data in [fp_data, tp_fn_data]:
        for r in data.get("results", []):
            mode = r["sample"].get("sample_mode", "tp")
            by_mode.setdefault(mode, [])
            for cond in ["baseline", "feature_only"]:
                if cond in r:
                    by_mode[mode].append(r[cond].get("faithfulness_keyword", 0))

    print(f"\nKeyword faithfulness by sample mode:")
    for mode in ["tp", "fn", "fp"]:
        vals = by_mode.get(mode, [])
        if vals:
            print(f"  {mode.upper()}: mean={sum(vals)/len(vals):.3f}, n={len(vals)}")

    # Kruskal-Wallis
    groups = [by_mode.get(m, []) for m in ["tp", "fn", "fp"]]
    groups = [g for g in groups if g]
    if len(groups) >= 2:
        h, p = kruskal_wallis(groups)
        print(f"\n  Kruskal-Wallis H={h:.3f}, p={p:.4f}")


def analyze_cross_judge(cross_path: str):
    """Analyze cross-model judge agreement."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Cross-Model Judge Agreement")
    print("=" * 70)

    data = load_results(cross_path)
    agreement = data.get("agreement", {})
    config = data.get("config", {})

    print(f"Judge: {config.get('judge_provider')}/{config.get('judge_model')}")
    print(f"Source: {config.get('results_dir')}")

    print(f"\n{'Condition':<20} {'N':>4} {'Orig Mean':>10} {'Cross Mean':>11} {'Spearman ρ':>11}")
    print("-" * 60)
    for cond, ag in sorted(agreement.items()):
        print(f"{cond:<20} {ag['n']:>4} {ag['original_mean']:>10.3f} {ag['cross_mean']:>11.3f} {ag['spearman_rho']:>11.3f}")


def main():
    base = Path("/workspace/arsim/LFactory/runs")

    # 1. Nosensor ablation
    nosensor_path = base / "llm_explanation_swat_nosensor_ablation" / "llm_explanation_results.json"
    if nosensor_path.exists():
        analyze_nosensor(str(nosensor_path))
    else:
        print(f"[SKIP] Nosensor results not found: {nosensor_path}")

    # 2. FP analysis
    fp_path = base / "llm_explanation_fp_samples" / "llm_explanation_results.json"
    tp_fn_path = base / "llm_explanation_swat_nosensor_ablation" / "llm_explanation_results.json"
    if fp_path.exists() and tp_fn_path.exists():
        analyze_fp(str(fp_path), str(tp_fn_path))
    else:
        print(f"\n[SKIP] FP results not found: {fp_path}")

    # 3. Cross-model judge
    for judge_dir in ["cross_model_judge_swat", "cross_model_judge_smd_aihub"]:
        cross_path = base / judge_dir / "cross_model_judge_results.json"
        if cross_path.exists():
            analyze_cross_judge(str(cross_path))
        else:
            print(f"\n[SKIP] Cross-judge results not found: {cross_path}")

    print("\n" + "=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
