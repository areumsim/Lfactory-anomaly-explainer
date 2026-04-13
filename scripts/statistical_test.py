"""Statistical significance testing for detector comparisons.

Usage:
    python scripts/statistical_test.py --runs runs/multi_seed_*

Implements:
- Wilcoxon signed-rank test (pairwise, with tie handling)
- Friedman test (multi-group non-parametric ANOVA)
- Nemenyi post-hoc test (after significant Friedman)
- Holm-Bonferroni correction for multiple comparisons
- Cliff's delta effect size
- Bootstrap confidence intervals
- McNemar's test (paired binary outcomes)
- Kendall's W (coefficient of concordance)
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _assign_ranks_with_ties(values: List[float]) -> List[float]:
    """Assign ranks with averaged tie handling."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and abs(indexed[j][1] - indexed[i][1]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _wilcoxon_test(pairs: List[tuple]) -> Dict:
    """Wilcoxon signed-rank test with tie handling."""
    diffs = [y - x for x, y in pairs]
    diffs = [d for d in diffs if abs(d) > 1e-10]

    if len(diffs) < 5:
        return {"test_statistic": 0, "p_value": 1.0, "note": "insufficient_data"}

    abs_diffs = [abs(d) for d in diffs]
    signs = [1 if d > 0 else -1 for d in diffs]
    ranks = _assign_ranks_with_ties(abs_diffs)

    W_plus = sum(r for r, s in zip(ranks, signs) if s > 0)
    W_minus = sum(r for r, s in zip(ranks, signs) if s < 0)
    W = min(W_plus, abs(W_minus))
    n = len(diffs)

    # Tie correction for variance
    from collections import Counter
    rank_counts = Counter(ranks)
    tie_correction = sum(t * (t * t - 1) for t in rank_counts.values() if t > 1) / 48.0

    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    std_W = math.sqrt(max(var_W, 1e-12))
    z = (W - mean_W) / std_W if std_W > 0 else 0
    p_value = 2 * (1 - _norm_cdf(abs(z)))

    return {
        "test_statistic": W,
        "n_pairs": n,
        "z_score": z,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


# ---- Friedman Test ----

def friedman_test(data: Dict[str, List[float]]) -> Dict:
    """Friedman test for comparing k related groups over n subjects.

    Args:
        data: {detector_name: [metric values per subject]}
              All lists must have the same length (n subjects).

    Returns:
        dict with chi_squared, p_value, mean_ranks, k, n
    """
    detectors = list(data.keys())
    k = len(detectors)
    if k < 3:
        return {"error": "Need at least 3 groups for Friedman test"}

    n = len(data[detectors[0]])
    for d in detectors:
        if len(data[d]) != n:
            return {"error": f"Unequal lengths: {d} has {len(data[d])}, expected {n}"}

    if n < 5:
        return {"error": f"Insufficient subjects (n={n}, need >=5)"}

    # Rank within each subject (row)
    rank_sums = {d: 0.0 for d in detectors}
    for i in range(n):
        row = [data[d][i] for d in detectors]
        ranks = _assign_ranks_with_ties(row)
        for j, d in enumerate(detectors):
            rank_sums[d] += ranks[j]

    mean_ranks = {d: rank_sums[d] / n for d in detectors}

    # Friedman chi-squared statistic
    chi_sq = (12.0 * n / (k * (k + 1))) * sum(
        (mean_ranks[d] - (k + 1) / 2.0) ** 2 for d in detectors
    )

    # Approximate p-value via chi-squared distribution (k-1 df)
    df = k - 1
    p_value = _chi2_sf(chi_sq, df)

    return {
        "test": "friedman",
        "chi_squared": chi_sq,
        "df": df,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "k_groups": k,
        "n_subjects": n,
        "mean_ranks": mean_ranks,
    }


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared distribution (approximate)."""
    if x <= 0:
        return 1.0
    # Use Wilson-Hilferty approximation
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 1.0 - _norm_cdf(z)


# ---- Nemenyi Post-Hoc ----

def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference for Nemenyi test.

    CD = q_alpha * sqrt(k*(k+1)/(6*n))
    q_alpha values from studentized range distribution (approximate).
    """
    # Approximate q_alpha values for alpha=0.05
    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q = q_table.get(k, 2.343 + 0.15 * (k - 3))  # rough extrapolation
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def nemenyi_posthoc(friedman_result: Dict) -> List[Dict]:
    """Pairwise Nemenyi post-hoc comparisons after significant Friedman test."""
    mean_ranks = friedman_result.get("mean_ranks", {})
    k = friedman_result.get("k_groups", 0)
    n = friedman_result.get("n_subjects", 0)

    if k < 3 or n < 5:
        return []

    cd = nemenyi_cd(k, n)
    detectors = list(mean_ranks.keys())
    comparisons = []

    for i in range(len(detectors)):
        for j in range(i + 1, len(detectors)):
            d_a, d_b = detectors[i], detectors[j]
            rank_diff = abs(mean_ranks[d_a] - mean_ranks[d_b])
            comparisons.append({
                "detector_a": d_a,
                "detector_b": d_b,
                "rank_a": mean_ranks[d_a],
                "rank_b": mean_ranks[d_b],
                "rank_difference": rank_diff,
                "critical_difference": cd,
                "significant": rank_diff > cd,
            })

    return comparisons


# ---- Holm-Bonferroni Correction ----

def holm_bonferroni(p_values: List[Tuple[str, float]]) -> List[Dict]:
    """Apply Holm-Bonferroni correction to a list of (label, p_value) tuples.

    Returns list of dicts with original p, adjusted p, and significance.
    """
    m = len(p_values)
    if m == 0:
        return []

    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    results = []

    for i, (label, p) in enumerate(sorted_pvals):
        adjusted_p = min(1.0, p * (m - i))
        results.append({
            "comparison": label,
            "p_value_raw": p,
            "p_value_adjusted": adjusted_p,
            "significant": adjusted_p < 0.05,
            "rank": i + 1,
        })

    # Enforce monotonicity (adjusted p must be non-decreasing)
    for i in range(1, len(results)):
        results[i]["p_value_adjusted"] = max(
            results[i]["p_value_adjusted"],
            results[i - 1]["p_value_adjusted"]
        )
        results[i]["significant"] = results[i]["p_value_adjusted"] < 0.05

    return results


# ---- Cliff's Delta Effect Size ----

def cliffs_delta(x: List[float], y: List[float]) -> Dict:
    """Compute Cliff's delta effect size (non-parametric).

    delta = (#{x_i > y_j} - #{x_i < y_j}) / (n_x * n_y)
    |delta| interpretation: <0.147 negligible, <0.33 small, <0.474 medium, else large
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return {"delta": 0.0, "interpretation": "undefined"}

    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    delta = (more - less) / (nx * ny)

    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"

    return {"delta": delta, "abs_delta": abs_d, "interpretation": interp}


# ---- Bootstrap Confidence Interval ----

def bootstrap_ci(
    values: List[float],
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Bootstrap confidence interval for a statistic.

    Args:
        values: Sample values
        statistic: "mean" or "median"
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95)
        seed: Random seed

    Returns:
        dict with point_estimate, ci_lower, ci_upper, se
    """
    if not values:
        return {"point_estimate": 0, "ci_lower": 0, "ci_upper": 0, "se": 0}

    rng = random.Random(seed)
    n = len(values)

    def calc_stat(sample):
        if statistic == "median":
            s = sorted(sample)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2
        return sum(sample) / len(sample)

    point_est = calc_stat(values)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(calc_stat(sample))

    boot_stats.sort()
    alpha = 1 - confidence
    lo_idx = max(0, int(math.floor(alpha / 2 * n_bootstrap)))
    hi_idx = min(n_bootstrap - 1, int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1)

    se = math.sqrt(sum((b - point_est) ** 2 for b in boot_stats) / max(1, n_bootstrap - 1))

    return {
        "point_estimate": point_est,
        "ci_lower": boot_stats[lo_idx],
        "ci_upper": boot_stats[hi_idx],
        "se": se,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


# ---- Post-hoc Power Analysis ----

def posthoc_power(
    pairs: List[tuple],
    alpha: float = 0.05,
) -> Dict:
    """Post-hoc power analysis for paired samples (Wilcoxon signed-rank test).

    Estimates statistical power using the observed effect size and sample size.
    Uses the normal approximation: power = P(Z > z_alpha - delta*sqrt(N))
    where delta is the standardized effect size (mean_diff / std_diff).

    Args:
        pairs: List of (x, y) paired observations
        alpha: Significance level (default 0.05)

    Returns:
        Dict with effect_size_d (Cohen's d for paired), power, n_pairs,
        min_n_80 (minimum N for 80% power), and min_n_90.
    """
    diffs = [y - x for x, y in pairs]
    diffs = [d for d in diffs if abs(d) > 1e-10]
    n = len(diffs)

    if n < 3:
        return {
            "effect_size_d": 0.0,
            "power": 0.0,
            "n_pairs": n,
            "min_n_80": float("inf"),
            "min_n_90": float("inf"),
            "note": "insufficient_data",
        }

    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)
    std_d = math.sqrt(var_d) if var_d > 0 else 1e-10

    # Cohen's d for paired samples
    d = abs(mean_d) / std_d

    # Power via normal approximation for Wilcoxon (asymptotic)
    # Uses ARE (Asymptotic Relative Efficiency) of Wilcoxon vs t-test ≈ 0.955
    are_factor = 0.955
    z_alpha = abs(_norm_ppf(alpha / 2))
    noncentrality = d * math.sqrt(n) * math.sqrt(are_factor)
    power = 1.0 - _norm_cdf(z_alpha - noncentrality)

    # Minimum N for target power levels
    def _min_n_for_power(target_power: float) -> float:
        if d < 1e-10:
            return float("inf")
        z_beta = abs(_norm_ppf(1.0 - target_power))
        return math.ceil(((z_alpha + z_beta) / (d * math.sqrt(are_factor))) ** 2)

    return {
        "effect_size_d": round(d, 4),
        "power": round(power, 4),
        "n_pairs": n,
        "min_n_80": _min_n_for_power(0.80),
        "min_n_90": _min_n_for_power(0.90),
    }


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (percent point function).

    Uses rational approximation (Abramowitz & Stegun 26.2.23).
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    if p > 0.5:
        return -_norm_ppf(1 - p)

    # Rational approximation for p < 0.5
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


# ---- McNemar's Test (paired binary outcomes) ----

def mcnemar_test(
    a_correct: List[bool],
    b_correct: List[bool],
) -> Dict:
    """McNemar's test for paired binary outcomes.

    Tests whether two conditions have the same proportion of correct results.
    Useful for comparing sensor identification accuracy between conditions.

    Args:
        a_correct: Boolean list indicating correct/incorrect for condition A.
        b_correct: Boolean list indicating correct/incorrect for condition B.

    Returns:
        Dict with chi2, p_value, n_discordant, interpretation.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("Both lists must have the same length")

    n = len(a_correct)
    # Contingency table
    # b=1 (correct): a=1 -> both correct, a=0 -> only b correct
    # b=0 (wrong):   a=1 -> only a correct, a=0 -> both wrong
    b_right_a_wrong = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    b_wrong_a_right = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)

    n_discordant = b_right_a_wrong + b_wrong_a_right

    if n_discordant == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "b_right_a_wrong": b_right_a_wrong,
            "b_wrong_a_right": b_wrong_a_right,
            "n_discordant": 0,
            "significant": False,
            "interpretation": "No discordant pairs",
        }

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b_right_a_wrong - b_wrong_a_right) - 1) ** 2 / n_discordant
    # p-value from chi-squared distribution with 1 df
    # Using normal approximation: chi2 with 1 df -> |z|^2
    z = math.sqrt(chi2)
    p_value = 2 * (1 - _norm_cdf(z))

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "b_right_a_wrong": b_right_a_wrong,
        "b_wrong_a_right": b_wrong_a_right,
        "n_discordant": n_discordant,
        "n_total": n,
        "significant": p_value < 0.05,
        "interpretation": "significant" if p_value < 0.05 else "not significant",
    }


# ---- Kendall's W (Coefficient of Concordance) ----

def kendall_w(
    rankings: List[List[float]],
) -> Dict:
    """Kendall's W coefficient of concordance.

    Measures agreement among m raters (datasets) ranking k items (detectors).
    W=0: no agreement, W=1: perfect agreement.

    Args:
        rankings: m x k matrix. rankings[i][j] = rank of detector j
                  according to rater/dataset i.

    Returns:
        Dict with W, chi2, p_value, interpretation.
    """
    m = len(rankings)  # number of raters (datasets)
    if m < 2:
        return {"W": None, "error": "Need at least 2 raters"}

    k = len(rankings[0])  # number of items (detectors)
    if k < 2:
        return {"W": None, "error": "Need at least 2 items"}

    # Sum of ranks for each item across all raters
    R = [0.0] * k
    for j in range(k):
        for i in range(m):
            R[j] += rankings[i][j]

    R_mean = sum(R) / k
    S = sum((r - R_mean) ** 2 for r in R)

    # Kendall's W
    W = (12 * S) / (m ** 2 * (k ** 3 - k))

    # Chi-squared approximation
    chi2 = m * (k - 1) * W
    # p-value using chi-squared with (k-1) df
    # Approximate using normal for large df
    df = k - 1
    z = math.sqrt(2 * chi2) - math.sqrt(2 * df - 1) if chi2 > 0 else 0
    p_value = 1 - _norm_cdf(z) if z > 0 else 1.0

    if W < 0.1:
        interp = "no agreement"
    elif W < 0.3:
        interp = "weak agreement"
    elif W < 0.5:
        interp = "moderate agreement"
    elif W < 0.7:
        interp = "strong agreement"
    else:
        interp = "very strong agreement"

    return {
        "W": round(W, 4),
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "m_raters": m,
        "k_items": k,
        "rank_sums": [round(r, 2) for r in R],
        "significant": p_value < 0.05,
        "interpretation": interp,
    }


def load_run_results(run_pattern: str) -> List[Dict]:
    """Load results from run directories matching pattern."""
    from glob import glob

    run_dirs = glob(run_pattern)
    results = []

    for run_dir in run_dirs:
        run_json = Path(run_dir) / "run.json"
        if run_json.exists():
            with open(run_json) as f:
                data = json.load(f)
                results.append(data)

    return results


def compare_detectors(results: List[Dict], metric: str = "auc_pr") -> Dict:
    """Compare detectors using Wilcoxon, Friedman, Nemenyi, Cliff's delta, and Bootstrap CI."""
    # Group by (dataset, detector_label) -> {seed: metric_value}
    groups: Dict[Tuple[str, str], Dict[int, float]] = {}

    for r in results:
        dataset = r.get("meta", {}).get("dataset", "unknown")
        detector = r.get("detector", {}).get("method", "unknown")
        ml_method = r.get("detector", {}).get("ml_method", None)
        seed = r.get("run", {}).get("seed", 0)

        label = f"{detector}_{ml_method}" if ml_method else detector
        key = (dataset, label)
        if key not in groups:
            groups[key] = {}
        groups[key][seed] = r.get("metrics", {}).get(metric, 0.0)

    # Organize by dataset
    datasets: Dict[str, Dict[str, Dict[int, float]]] = {}
    for (ds, label), seed_vals in groups.items():
        if ds not in datasets:
            datasets[ds] = {}
        datasets[ds][label] = seed_vals

    all_results = {}

    for ds, detector_data in datasets.items():
        ds_result: Dict[str, Any] = {"dataset": ds, "metric": metric}

        # --- Bootstrap CI per detector ---
        ci_results = {}
        for label, seed_vals in detector_data.items():
            vals = list(seed_vals.values())
            ci_results[label] = bootstrap_ci(vals, statistic="mean", n_bootstrap=1000)
        ds_result["bootstrap_ci"] = ci_results

        # --- Pairwise Wilcoxon + Cliff's delta ---
        detector_labels = sorted(detector_data.keys())
        pairwise = []
        raw_pvalues = []

        for i in range(len(detector_labels)):
            for j in range(i + 1, len(detector_labels)):
                la, lb = detector_labels[i], detector_labels[j]
                va, vb = detector_data[la], detector_data[lb]
                common = sorted(set(va.keys()) & set(vb.keys()))

                if len(common) < 5:
                    continue

                pairs = [(va[s], vb[s]) for s in common]
                wilcox = _wilcoxon_test(pairs)
                cliff = cliffs_delta([va[s] for s in common], [vb[s] for s in common])

                mean_a = sum(va[s] for s in common) / len(common)
                mean_b = sum(vb[s] for s in common) / len(common)

                power = posthoc_power(pairs)

                comp_label = f"{la} vs {lb}"
                raw_pvalues.append((comp_label, wilcox["p_value"]))

                pairwise.append({
                    "detector_a": la,
                    "detector_b": lb,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "difference": mean_b - mean_a,
                    "wilcoxon": wilcox,
                    "cliffs_delta": cliff,
                    "power_analysis": power,
                })

        ds_result["pairwise"] = pairwise

        # --- Holm-Bonferroni correction ---
        if raw_pvalues:
            ds_result["holm_bonferroni"] = holm_bonferroni(raw_pvalues)

        # --- Friedman test (if >= 3 detectors) ---
        if len(detector_labels) >= 3:
            all_seeds = set()
            for sv in detector_data.values():
                all_seeds |= set(sv.keys())
            common_seeds = sorted(all_seeds)
            for sv in detector_data.values():
                common_seeds = [s for s in common_seeds if s in sv]

            if len(common_seeds) >= 5:
                friedman_data = {
                    label: [detector_data[label][s] for s in common_seeds]
                    for label in detector_labels
                }
                fr = friedman_test(friedman_data)
                ds_result["friedman"] = fr

                if fr.get("significant"):
                    ds_result["nemenyi"] = nemenyi_posthoc(fr)

        all_results[ds] = ds_result

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical significance testing")
    parser.add_argument("--runs", type=str, default="runs/multi_seed_*", help="Pattern to match run directories")
    parser.add_argument("--results-json", type=str, default="", help="Path to aggregated results JSON (alternative to --runs)")
    parser.add_argument("--metric", type=str, default="auc_pr", help="Metric to compare")
    parser.add_argument("--output", type=str, default="runs/statistical_tests.json", help="Output file")

    args = parser.parse_args()

    if args.results_json:
        with open(args.results_json) as f:
            raw = json.load(f)
        results = raw if isinstance(raw, list) else raw.get("results", [])
    else:
        print(f"Loading results from: {args.runs}")
        results = load_run_results(args.runs)

    print(f"Loaded {len(results)} results")
    print(f"Comparing detectors on metric: {args.metric}")

    comparison_results = compare_detectors(results, args.metric)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Print summary per dataset
    for ds, ds_result in comparison_results.items():
        print(f"\n=== {ds} ({ds_result.get('metric', '')}) ===")

        # Bootstrap CI
        ci = ds_result.get("bootstrap_ci", {})
        for det, c in ci.items():
            print(f"  {det}: {c['point_estimate']:.4f} [{c['ci_lower']:.4f}, {c['ci_upper']:.4f}]")

        # Friedman
        fr = ds_result.get("friedman", {})
        if fr:
            sig = "SIGNIFICANT" if fr.get("significant") else "not significant"
            print(f"  Friedman: chi2={fr.get('chi_squared', 0):.2f}, p={fr.get('p_value', 1):.4f} ({sig})")

        # Holm-Bonferroni
        hb = ds_result.get("holm_bonferroni", [])
        for h in hb:
            sig = "SIG" if h["significant"] else "ns"
            print(f"  {h['comparison']}: p_raw={h['p_value_raw']:.4f} p_adj={h['p_value_adjusted']:.4f} [{sig}]")

        # Nemenyi
        nem = ds_result.get("nemenyi", [])
        if nem:
            cd = nem[0].get("critical_difference", 0)
            print(f"  Nemenyi CD={cd:.3f}")
            for n in nem:
                sig = "SIG" if n["significant"] else "ns"
                print(f"    {n['detector_a']} vs {n['detector_b']}: "
                      f"rank_diff={n['rank_difference']:.3f} [{sig}]")
