"""Phase 2 LLM Explanation Experiment — Minimum Viable Experiment.

Generates anomaly explanations using API-based LLMs (Claude/OpenAI) and evaluates:
- Faithfulness: Does the LLM correctly identify anomalous sensor behavior?
- Consistency: Are explanations stable across repeated generations?
- Domain knowledge impact: 2x2 ablation (domain knowledge x feature importance)

Usage:
    # With Claude API
    export ANTHROPIC_API_KEY=sk-...
    python scripts/run_llm_explanation_experiment.py --provider claude --model claude-haiku-4-5-20251001

    # With OpenAI API
    export OPENAI_API_KEY=sk-...
    python scripts/run_llm_explanation_experiment.py --provider openai --model gpt-4o-mini
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.data.data_router import load_timeseries
from experiments.metrics import binary_metrics, find_optimal_f1_threshold, segments_from_labels
from experiments.domain_knowledge.knowledge_retriever import DomainKnowledgeRetriever


# ---- LLM Providers ----

class LLMProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1500):
        import anthropic
        # Support CLAUDE_API_KEY as alias for ANTHROPIC_API_KEY
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3,
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1500):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content


# ---- Anomaly Sample Selection ----

def select_anomaly_samples(
    dataset: str,
    data_root: str,
    n_samples: int = 10,
    detector_func=None,
    sample_mode: str = "tp",
) -> List[Dict]:
    """Select anomaly samples for LLM explanation evaluation.

    Args:
        sample_mode: One of "tp" (true positives — detector correct),
                     "fn" (false negatives — detector missed anomaly),
                     "fp" (false positives — detector false alarm on normal point).
                     Default "tp" for backward compatibility.
    """
    from experiments.ml_detector_isolation_forest import detect_isolation_forest
    from experiments.spec_cnn import detect_speccnn

    samples = []
    max_files_map = {"SKAB": 34, "AIHub71802": 38, "SMD": 28, "SWaT": 1}
    max_files = max_files_map.get(dataset, 28)
    split_map = {"AIHub71802": "Validation", "SWaT": "test"}
    split = split_map.get(dataset, "test")

    for file_idx in range(max_files):
        if len(samples) >= n_samples:
            break
        try:
            data = load_timeseries(dataset, data_root, split=split,
                                   file_index=file_idx)
            series, labels = data["series"], data["labels"]
            meta = data.get("meta", {})

            if sum(labels) == 0:
                continue

            # For very long series (>500K), use tail portion containing anomalies
            # to speed up detection and improve TP yield
            if len(series) > 500000:
                tail_n = min(200000, len(series))
                series_det = series[-tail_n:]
                labels_det = labels[-tail_n:]
                offset = len(series) - tail_n
            else:
                series_det = series
                labels_det = labels
                offset = 0

            # Use IF as default detector
            det = detect_isolation_forest(series_det, window_size=30, contamination=0.2)
            scores, preds = det["scores"], det["preds"]

            # Find sample indices based on sample_mode
            segments = segments_from_labels(labels_det)

            if sample_mode == "fp":
                # False positives: normal points (label=0) where detector fires (pred=1)
                fp_indices = [i for i in range(len(preds))
                              if labels_det[i] == 0 and preds[i] == 1]
                if not fp_indices:
                    continue
                # Limit per-file to spread FP across multiple files
                remaining = n_samples - len(samples)
                remaining_files = max(1, max_files - file_idx)
                per_file = max(1, min(remaining, remaining // remaining_files + 1))
                needed = min(per_file, len(fp_indices))
                step = max(1, len(fp_indices) // needed)
                all_selected = fp_indices[::step][:needed]
            else:
                # TP or FN: iterate over anomaly segments
                all_selected = []
                for seg_start, seg_end in segments:
                    if len(all_selected) + len(samples) >= n_samples:
                        break
                    seg_len = seg_end - seg_start + 1

                    if sample_mode == "fn":
                        # False negatives: anomaly points (label=1) detector missed (pred=0)
                        candidate_indices = [idx for idx in range(seg_start, min(seg_end + 1, len(preds)))
                                             if preds[idx] == 0]
                    else:
                        # True positives: anomaly points (label=1) detector caught (pred=1)
                        candidate_indices = [idx for idx in range(seg_start, min(seg_end + 1, len(preds)))
                                             if preds[idx] == 1]

                    if not candidate_indices:
                        continue

                    if seg_len > 1000:
                        needed = min(n_samples - len(samples) - len(all_selected), len(candidate_indices))
                        step = max(1, len(candidate_indices) // needed)
                        all_selected.extend(candidate_indices[::step][:needed])
                    else:
                        all_selected.append(candidate_indices[0])

            for idx in all_selected:
                if len(samples) >= n_samples:
                    break
                # Map back to full series indices
                full_idx = idx + offset
                ctx_start = max(0, full_idx - 20)
                ctx_end = min(len(series), full_idx + 21)
                ctx_values = series[ctx_start:ctx_end]
                ctx_labels = labels[ctx_start:ctx_end]

                normal_vals = [v for v, l in zip(ctx_values, ctx_labels) if l == 0]
                normal_mean = sum(normal_vals) / len(normal_vals) if normal_vals else 0
                normal_std = (sum((v - normal_mean)**2 for v in normal_vals) / max(1, len(normal_vals)))**0.5 if normal_vals else 1

                # For all_sensors, slice to tail if offset applied
                sample_sensors = None
                if meta.get("all_sensors"):
                    sample_sensors = {k: v[-len(series_det):] if offset > 0 else v
                                      for k, v in meta["all_sensors"].items()}

                samples.append({
                    "dataset": dataset,
                    "file": meta.get("file", f"file_{file_idx}"),
                    "file_index": file_idx,
                    "anomaly_idx": idx,  # index within detection slice
                    "value": series_det[idx],
                    "score": scores[idx],
                    "true_label": labels_det[idx],
                    "pred_label": preds[idx],
                    "sample_mode": sample_mode,
                    "normal_mean": normal_mean,
                    "normal_std": normal_std,
                    "sigma_deviation": abs(series_det[idx] - normal_mean) / max(normal_std, 1e-6),
                    "segment": (seg_start, seg_end) if sample_mode != "fp" else None,
                    "segment_length": (seg_end - seg_start + 1) if sample_mode != "fp" else 0,
                    "context_values": ctx_values,
                    "all_sensors": sample_sensors,
                })
        except Exception as e:
            print(f"  Skip file {file_idx}: {e}", file=sys.stderr)
            continue

    return samples


# ---- Prompt Construction ----

SYSTEM_PROMPT = """You are an expert in manufacturing process monitoring and anomaly detection.
Analyze the given anomaly data and provide a structured explanation with:
1. **Anomaly Description**: What abnormal behavior was observed
2. **Severity Assessment**: Low/Medium/High/Critical with reasoning
3. **Possible Root Causes**: 2-3 most likely causes
4. **Recommended Actions**: Immediate and follow-up steps
5. **Sensor Analysis**: Which sensor(s) show the most abnormal readings (if multi-sensor data provided)

Be concise and actionable. Format each section clearly."""


def build_anomaly_prompt(
    sample: Dict,
    include_domain_knowledge: bool = True,
    include_feature_importance: bool = True,
    knowledge_retriever: Optional[DomainKnowledgeRetriever] = None,
) -> str:
    """Build explanation prompt for an anomaly sample."""
    prompt = f"""## Anomaly Detection Report

**Dataset**: {sample['dataset']}
**File**: {sample['file']}
**Time Index**: {sample['anomaly_idx']}

### Anomaly Details
- Sensor value: {sample['value']:.4f}
- Anomaly score: {sample['score']:.4f}
- Deviation from normal: {sample['sigma_deviation']:.2f} sigma
- Normal baseline: mean={sample['normal_mean']:.4f}, std={sample['normal_std']:.4f}
- Anomaly segment length: {sample['segment_length']} timesteps
"""

    # Add multi-sensor data if available
    if include_feature_importance and sample.get("all_sensors"):
        sensors = sample["all_sensors"]
        idx = sample["anomaly_idx"]
        prompt += "\n### Multi-Sensor Values at Anomaly Point\n"
        for col, vals in sensors.items():
            if idx < len(vals):
                # Compute local baseline
                start = max(0, idx - 20)
                end = min(len(vals), idx + 21)
                local_vals = vals[start:end]
                local_mean = sum(local_vals) / len(local_vals)
                local_std = (sum((v - local_mean)**2 for v in local_vals) / len(local_vals))**0.5
                z = abs(vals[idx] - local_mean) / max(local_std, 1e-6)
                prompt += f"- **{col}**: {vals[idx]:.3f} (z-score: {z:.2f})\n"

    # Add domain knowledge
    if include_domain_knowledge and knowledge_retriever:
        try:
            dk = knowledge_retriever.format_knowledge_for_llm(
                dataset=sample["dataset"],
                sensor_name=sample.get("primary_sensor", ""),
                anomaly_score=sample["score"],
            )
            if dk:
                prompt += f"\n### Domain Knowledge\n{dk}\n"
        except Exception:
            pass

    prompt += "\nPlease analyze this anomaly and provide your structured explanation."
    return prompt


# ---- Evaluation Metrics ----

def evaluate_keyword_coverage(explanation: str, sample: Dict) -> float:
    """Keyword coverage check: does explanation use anomaly-related terms?

    NOTE: Renamed from evaluate_faithfulness to clarify that this metric only
    checks keyword presence, not factual accuracy. Use evaluate_faithfulness()
    for the stricter metric that verifies direction and sensor identification.
    """
    score = 0.0
    total = 0.0

    # Check if high deviation is mentioned
    if sample["sigma_deviation"] > 2:
        total += 1
        if any(w in explanation.lower() for w in ["abnormal", "deviation", "anomal", "unusual", "spike", "exceed"]):
            score += 1

    # Check severity assessment
    total += 1
    if sample["sigma_deviation"] > 5:
        if any(w in explanation.lower() for w in ["high", "critical", "severe", "significant"]):
            score += 1
    elif sample["sigma_deviation"] > 2:
        if any(w in explanation.lower() for w in ["medium", "moderate", "notable"]):
            score += 1
    else:
        score += 0.5  # mild anomaly, any assessment is acceptable

    # Check if sensor analysis is present when multi-sensor
    if sample.get("all_sensors"):
        total += 1
        if any(col.lower() in explanation.lower() for col in (sample["all_sensors"] or {}).keys()):
            score += 1

    return score / max(total, 1)


# Backward compatibility alias
evaluate_faithfulness = evaluate_keyword_coverage


def evaluate_faithfulness_strict(explanation: str, sample: Dict) -> Dict:
    """Strict faithfulness evaluation with directional accuracy and sensor identification.

    Returns dict with sub-scores:
    - keyword_coverage: same as evaluate_keyword_coverage (0-1)
    - direction_correct: whether the explanation correctly identifies anomaly direction (bool/None)
    - sensor_top1_correct: whether the top anomalous sensor is correctly identified (bool/None)
    - sensor_top3_overlap: fraction of top-3 sensors mentioned (0-1 or None)
    - overall: weighted composite score (0-1)
    """
    explanation_lower = explanation.lower()
    result = {}

    # 1. Keyword coverage (existing metric)
    result["keyword_coverage"] = evaluate_keyword_coverage(explanation, sample)

    # 2. Directional accuracy: check if explanation correctly identifies increase vs decrease
    anomaly_value = sample.get("value", 0)
    normal_mean = sample.get("normal_mean", 0)
    deviation = anomaly_value - normal_mean

    # Direction keywords — exclude ambiguous words ("high"/"low") that appear
    # in non-directional contexts like "high severity", "low baseline"
    increase_phrases = ["increase", "rise", "rising", "spike", "surge", "elevated",
                        "jump", "peak", "above normal", "higher than", "exceeded"]
    decrease_phrases = ["decrease", "drop", "falling", "dip", "decline", "plunge",
                        "reduced", "below normal", "lower than"]

    has_increase = any(p in explanation_lower for p in increase_phrases)
    has_decrease = any(p in explanation_lower for p in decrease_phrases)

    if abs(deviation) > sample.get("normal_std", 1) * 0.5:
        if deviation > 0:
            # For increase: correct if increase mentioned, tolerate decrease co-occurrence
            # (LLM may say "increased from low baseline")
            result["direction_correct"] = has_increase
            result["direction_ground_truth"] = "increase"
        else:
            result["direction_correct"] = has_decrease
            result["direction_ground_truth"] = "decrease"
    else:
        result["direction_correct"] = None  # deviation too small to judge direction
        result["direction_ground_truth"] = "ambiguous"

    # 3. Sensor identification accuracy (multi-sensor only)
    sensors = sample.get("all_sensors")
    if sensors and len(sensors) > 1:
        idx = sample.get("anomaly_idx", 0)
        sensor_z = {}
        for col, vals in sensors.items():
            if idx < len(vals):
                start = max(0, idx - 20)
                end = min(len(vals), idx + 21)
                local = vals[start:end]
                mu = sum(local) / len(local) if local else 0
                std = (sum((v - mu) ** 2 for v in local) / max(len(local), 1)) ** 0.5
                sensor_z[col] = abs(vals[idx] - mu) / max(std, 1e-6)

        sorted_sensors = sorted(sensor_z, key=sensor_z.get, reverse=True)
        top1 = sorted_sensors[0] if sorted_sensors else ""
        top3 = set(sorted_sensors[:3])

        result["sensor_top1_correct"] = top1.lower() in explanation_lower if top1 else None
        result["sensor_top1_name"] = top1
        top3_found = sum(1 for s in top3 if s.lower() in explanation_lower)
        result["sensor_top3_overlap"] = top3_found / min(3, len(sorted_sensors)) if sorted_sensors else None
        result["sensor_top3_names"] = list(top3)
    else:
        result["sensor_top1_correct"] = None
        result["sensor_top3_overlap"] = None

    # 4. Composite overall score
    components = [result["keyword_coverage"]]
    weights = [1.0]

    if result["direction_correct"] is not None:
        components.append(1.0 if result["direction_correct"] else 0.0)
        weights.append(1.5)  # direction accuracy is important

    if result["sensor_top1_correct"] is not None:
        components.append(1.0 if result["sensor_top1_correct"] else 0.0)
        weights.append(1.5)

    if result["sensor_top3_overlap"] is not None:
        components.append(result["sensor_top3_overlap"])
        weights.append(1.0)

    result["overall"] = sum(c * w for c, w in zip(components, weights)) / sum(weights)

    return result


JUDGE_SYSTEM_PROMPT = """You are a strict expert evaluator of anomaly detection explanations.
You will receive ground truth anomaly data and an explanation to evaluate.
Rate the explanation strictly based on how well it matches the provided data.
USE THE FULL 1-5 SCALE. Most explanations should NOT get 4 or 5.
A score of 3 means "acceptable but with notable issues". Reserve 5 for near-perfect explanations.
Respond ONLY with a JSON object, no other text."""


def evaluate_faithfulness_llm(
    explanation: str,
    sample: Dict,
    provider: LLMProvider,
    max_retries: int = 3,
) -> Dict:
    """LLM-as-judge evaluation of explanation faithfulness.

    Returns dict with per-dimension scores (1-5) and overall (0-1).
    """
    import re

    sensor_names = list((sample.get("all_sensors") or {}).keys())
    # Compute top anomalous sensors by z-score
    top_sensors = []
    if sample.get("all_sensors"):
        idx = sample["anomaly_idx"]
        sensor_z = {}
        for col, vals in sample["all_sensors"].items():
            if idx < len(vals):
                start = max(0, idx - 20)
                end = min(len(vals), idx + 21)
                local = vals[start:end]
                mu = sum(local) / len(local)
                std = (sum((v - mu) ** 2 for v in local) / len(local)) ** 0.5
                sensor_z[col] = abs(vals[idx] - mu) / max(std, 1e-6)
        top_sensors = sorted(sensor_z, key=sensor_z.get, reverse=True)[:3]

    judge_prompt = f"""## Ground Truth Data
- Dataset: {sample['dataset']}
- Sensor value at anomaly: {sample['value']:.4f}
- Deviation from normal: {sample['sigma_deviation']:.2f} sigma
- Normal baseline: mean={sample['normal_mean']:.4f}, std={sample['normal_std']:.4f}
- Anomaly segment length: {sample['segment_length']} timesteps
- Sensors present: {', '.join(sensor_names) if sensor_names else 'single sensor'}
- Top anomalous sensors (by z-score): {', '.join(top_sensors) if top_sensors else 'N/A'}

## Explanation to Evaluate
{explanation[:2000]}

Rate this explanation on each criterion using the FULL 1-5 scale:
1. DATA_ACCURACY: Does it correctly reference the actual sensor values and deviation magnitude?
   1=fabricated values, 2=wrong values, 3=approximate, 4=mostly correct, 5=exact match
2. SENSOR_ID: Does it correctly identify which sensor(s) are most anomalous?
   1=wrong sensors, 2=vague/no identification, 3=partially correct (single-sensor=3), 4=correct but incomplete reasoning, 5=exact match with reasoning
3. SEVERITY: Is the severity assessment appropriate for a {sample['sigma_deviation']:.1f}-sigma deviation?
   1=completely wrong level, 2=overreacts or underreacts significantly, 3=roughly appropriate, 4=well-calibrated, 5=perfectly matched
4. ACTIONABILITY: Are the recommended actions practical and relevant?
   1=irrelevant/generic, 2=vague, 3=somewhat useful, 4=specific and practical, 5=immediately actionable with clear priority

Respond ONLY in this exact JSON format:
{{"data_accuracy": N, "sensor_id": N, "severity": N, "actionability": N}}"""

    for attempt in range(max_retries):
        try:
            response = provider.generate(JUDGE_SYSTEM_PROMPT, judge_prompt)
            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores = json.loads(json_match.group())
                # Validate
                dims = ["data_accuracy", "sensor_id", "severity", "actionability"]
                for d in dims:
                    if d not in scores or not isinstance(scores[d], (int, float)):
                        scores[d] = 3  # neutral fallback
                    scores[d] = max(1, min(5, scores[d]))
                scores["overall"] = sum(scores[d] for d in dims) / (len(dims) * 5)
                return scores
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return {"data_accuracy": 0, "sensor_id": 0, "severity": 0, "actionability": 0, "overall": 0, "error": str(e)}
    return {"data_accuracy": 0, "sensor_id": 0, "severity": 0, "actionability": 0, "overall": 0, "error": "max_retries"}


def evaluate_consistency(explanations: List[str]) -> float:
    """Measure consistency between repeated explanations via keyword overlap."""
    if len(explanations) < 2:
        return 1.0

    def extract_keywords(text):
        words = set(w.lower() for w in text.split() if len(w) > 3)
        return words

    pairs = []
    for i in range(len(explanations)):
        for j in range(i + 1, len(explanations)):
            kw_i = extract_keywords(explanations[i])
            kw_j = extract_keywords(explanations[j])
            if kw_i and kw_j:
                jaccard = len(kw_i & kw_j) / len(kw_i | kw_j)
                pairs.append(jaccard)

    return sum(pairs) / len(pairs) if pairs else 0.0


# ---- Main Experiment ----

def run_experiment(args):
    data_root = args.data_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM provider
    if args.provider == "claude":
        provider = ClaudeProvider(model=args.model, max_tokens=args.max_tokens)
    elif args.provider == "openai":
        provider = OpenAIProvider(model=args.model, max_tokens=args.max_tokens)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    # Initialize domain knowledge retrievers (with and without sensor names)
    knowledge_dir = str(Path(__file__).resolve().parent.parent / "experiments" / "domain_knowledge")
    kr = DomainKnowledgeRetriever(knowledge_dir)
    kr_nosensor = DomainKnowledgeRetriever(knowledge_dir, use_nosensor=True)

    # Select anomaly samples — stratified by TP/FN/FP for unbiased evaluation
    sample_modes = getattr(args, "sample_modes", ["tp"])
    print(f"Selecting anomaly samples from {args.datasets} (modes: {sample_modes})...")
    all_samples = []

    # Resolve per-dataset data roots from datasets.yaml
    datasets_cfg = str(Path(__file__).resolve().parent.parent / "experiments" / "data" / "datasets.yaml")
    from experiments.data.data_router import resolve_data_root

    for dataset in args.datasets:
        ds_root = resolve_data_root(dataset, datasets_cfg) or data_root
        for mode in sample_modes:
            samples = select_anomaly_samples(
                dataset, ds_root,
                n_samples=args.samples_per_dataset,
                sample_mode=mode,
            )
            all_samples.extend(samples)
            print(f"  {dataset} [{mode.upper()}]: {len(samples)} samples selected")

    if not all_samples:
        print("ERROR: No anomaly samples found!")
        return

    # 2x2 Ablation conditions (+ optional nosensor for confound control)
    conditions = [
        {"name": "baseline", "domain": False, "feature": False, "nosensor": False},
        {"name": "domain_only", "domain": True, "feature": False, "nosensor": False},
        {"name": "feature_only", "domain": False, "feature": True, "nosensor": False},
        {"name": "full", "domain": True, "feature": True, "nosensor": False},
    ]
    if getattr(args, "include_nosensor", False):
        conditions.extend([
            {"name": "domain_nosensor", "domain": True, "feature": False, "nosensor": True},
            {"name": "full_nosensor", "domain": True, "feature": True, "nosensor": True},
        ])

    # Resume support: load existing results if available
    results = []
    completed_keys = set()
    result_path = output_dir / "llm_explanation_results.json"
    if args.resume and result_path.exists():
        try:
            with open(result_path) as f:
                existing = json.load(f)
            results = existing.get("results", [])
            for r in results:
                s = r.get("sample", {})
                completed_keys.add((s.get("dataset"), s.get("file"), s.get("anomaly_idx")))
            print(f"Resumed: {len(results)} completed samples loaded")
        except Exception:
            pass

    total_calls = len(all_samples) * len(conditions)
    call_count = len(completed_keys) * len(conditions)

    for si, sample in enumerate(all_samples):
        sample_key = (sample["dataset"], sample["file"], sample["anomaly_idx"])
        if sample_key in completed_keys:
            continue

        sample_results = {"sample": {k: v for k, v in sample.items() if k not in ("context_values", "all_sensors")}}

        for condition in conditions:
            call_count += 1
            print(f"[{call_count}/{total_calls}] {sample['dataset']}:{sample['file']}@{sample['anomaly_idx']} "
                  f"({condition['name']})", end=" ", flush=True)

            active_kr = kr_nosensor if condition.get("nosensor") else kr
            prompt = build_anomaly_prompt(
                sample,
                include_domain_knowledge=condition["domain"],
                include_feature_importance=condition["feature"],
                knowledge_retriever=active_kr if condition["domain"] else None,
            )

            for attempt in range(3):
                try:
                    t0 = time.time()
                    explanation = provider.generate(SYSTEM_PROMPT, prompt)
                    latency = time.time() - t0

                    faith_kw = evaluate_keyword_coverage(explanation, sample)
                    faith_strict = evaluate_faithfulness_strict(explanation, sample)

                    # LLM-as-judge evaluation
                    judge_scores = evaluate_faithfulness_llm(explanation, sample, provider)
                    time.sleep(0.3)

                    dir_ok = faith_strict.get("direction_correct")
                    dir_str = "?" if dir_ok is None else ("Y" if dir_ok else "N")
                    s1_ok = faith_strict.get("sensor_top1_correct")
                    s1_str = "?" if s1_ok is None else ("Y" if s1_ok else "N")
                    print(f"kw={faith_kw:.2f} strict={faith_strict['overall']:.2f} "
                          f"dir={dir_str} s1={s1_str} llm={judge_scores.get('overall', 0):.2f} {latency:.1f}s")

                    sample_results[condition["name"]] = {
                        "explanation": explanation,
                        "faithfulness_keyword": faith_kw,
                        "faithfulness_strict": faith_strict,
                        "faithfulness_llm": judge_scores,
                        "latency": latency,
                        "prompt_length": len(prompt),
                        "response_length": len(explanation),
                        # Backward compat
                        "faithfulness": faith_kw,
                    }
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"retry({attempt+1})...", end=" ", flush=True)
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"ERROR: {e}")
                        sample_results[condition["name"]] = {"error": str(e)}

            # Rate limit
            time.sleep(0.5)

        # Consistency check: generate 3x for "full" condition
        if args.consistency_repeats > 1:
            repeat_explanations = []
            prompt = build_anomaly_prompt(sample, True, True, kr)
            for rep in range(args.consistency_repeats):
                try:
                    exp = provider.generate(SYSTEM_PROMPT, prompt)
                    repeat_explanations.append(exp)
                    time.sleep(0.3)
                except Exception:
                    pass
            if repeat_explanations:
                sample_results["consistency"] = {
                    "n_repeats": len(repeat_explanations),
                    "jaccard_similarity": evaluate_consistency(repeat_explanations),
                }

        results.append(sample_results)

        # Periodic save (every 10 samples)
        if (si + 1) % 10 == 0:
            _save_results(results, conditions, args, output_dir)
            print(f"  [checkpoint] {len(results)} samples saved")

    # Final save
    _save_results(results, conditions, args, output_dir)


def _save_results(results, conditions, args, output_dir):
    """Save results with summary."""
    summary = _compute_summary(results, conditions)
    with open(output_dir / "llm_explanation_results.json", "w") as f:
        json.dump({"results": results, "summary": summary, "config": vars(args)}, f, indent=2, ensure_ascii=False)
    _print_summary(summary, conditions)
    print(f"\nResults saved to: {output_dir / 'llm_explanation_results.json'}")


def _compute_summary(results: List[Dict], conditions: List[Dict]) -> Dict:
    summary = {}
    for cond in conditions:
        name = cond["name"]
        faiths_kw = [r[name]["faithfulness"] for r in results if name in r and "faithfulness" in r.get(name, {})]
        latencies = [r[name]["latency"] for r in results if name in r and "latency" in r.get(name, {})]

        # LLM-judge scores
        llm_overalls = []
        llm_dims = {"data_accuracy": [], "sensor_id": [], "severity": [], "actionability": []}
        for r in results:
            if name in r and "faithfulness_llm" in r.get(name, {}):
                llm = r[name]["faithfulness_llm"]
                if "overall" in llm and llm["overall"] > 0:
                    llm_overalls.append(llm["overall"])
                    for d in llm_dims:
                        if d in llm:
                            llm_dims[d].append(llm[d])

        cond_summary = {
            "n": len(faiths_kw),
            "mean_faithfulness_keyword": sum(faiths_kw) / len(faiths_kw) if faiths_kw else 0,
            "mean_faithfulness_llm": sum(llm_overalls) / len(llm_overalls) if llm_overalls else 0,
            "n_llm_judge": len(llm_overalls),
            "mean_latency": sum(latencies) / len(latencies) if latencies else 0,
        }
        # Per-dimension LLM scores
        for d, vals in llm_dims.items():
            if vals:
                cond_summary[f"llm_{d}_mean"] = sum(vals) / len(vals)

        # Backward compat
        cond_summary["mean_faithfulness"] = cond_summary["mean_faithfulness_keyword"]
        summary[name] = cond_summary

    # Consistency
    consistencies = [r["consistency"]["jaccard_similarity"] for r in results if "consistency" in r]
    summary["consistency"] = {
        "n": len(consistencies),
        "mean_jaccard": sum(consistencies) / len(consistencies) if consistencies else 0,
    }
    return summary


def _print_summary(summary: Dict, conditions: List[Dict]):
    print(f"\n{'='*70}")
    print("LLM Explanation Experiment Summary")
    print(f"{'='*70}")
    print(f"{'Condition':<18} {'N':>4} {'Keyword':>10} {'LLM-Judge':>10} {'Latency':>10}")
    print("-" * 56)
    for cond in conditions:
        s = summary[cond["name"]]
        kw = s.get("mean_faithfulness_keyword", s.get("mean_faithfulness", 0))
        llm = s.get("mean_faithfulness_llm", 0)
        print(f"{cond['name']:<18} {s['n']:>4} {kw:>10.3f} {llm:>10.3f} {s['mean_latency']:>9.1f}s")
    c = summary.get("consistency", {})
    if c.get("n", 0) > 0:
        print(f"\nConsistency (Jaccard): {c['mean_jaccard']:.3f} (n={c['n']})")


def run_cross_model_judge(args):
    """Re-evaluate existing explanations using a different LLM as judge.

    Reads results from --results-dir, scores each explanation with the specified
    judge provider/model, and saves cross-judge results to --output-dir.
    """
    results_dir = Path(args.results_dir)
    result_file = results_dir / "llm_explanation_results.json"
    if not result_file.exists():
        print(f"ERROR: {result_file} not found")
        return

    with open(result_file) as f:
        data = json.load(f)

    original_results = data.get("results", [])
    print(f"Loaded {len(original_results)} samples from {result_file}")

    # Initialize judge provider (different from generation provider)
    if args.judge_provider == "claude":
        judge = ClaudeProvider(model=args.judge_model, max_tokens=500)
    else:
        judge = OpenAIProvider(model=args.judge_model, max_tokens=500)

    # Identify conditions to re-judge
    conditions_found = set()
    for r in original_results:
        for k in r:
            if k not in ("sample", "consistency"):
                conditions_found.add(k)
    conditions_found = sorted(conditions_found)
    print(f"Conditions to re-judge: {conditions_found}")

    cross_results = []
    total_evals = len(original_results) * len(conditions_found)
    eval_count = 0

    for si, r in enumerate(original_results):
        sample = r.get("sample", {})
        sample_cross = {"sample": sample}

        for cond_name in conditions_found:
            cond_data = r.get(cond_name, {})
            explanation = cond_data.get("explanation", "")
            if not explanation or "error" in cond_data:
                continue

            eval_count += 1
            print(f"[{eval_count}/{total_evals}] {sample.get('dataset')}:{sample.get('file')}@{sample.get('anomaly_idx')} "
                  f"({cond_name}) ", end="", flush=True)

            for attempt in range(3):
                try:
                    judge_scores = evaluate_faithfulness_llm(explanation, sample, judge)
                    print(f"llm={judge_scores.get('overall', 0):.2f}")
                    sample_cross[cond_name] = {
                        "original_llm_judge": cond_data.get("faithfulness_llm", {}),
                        "cross_llm_judge": judge_scores,
                        "original_kw": cond_data.get("faithfulness_keyword", 0),
                        "original_strict": cond_data.get("faithfulness_strict", {}),
                    }
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"retry({attempt+1})...", end=" ", flush=True)
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"ERROR: {e}")
                        sample_cross[cond_name] = {"error": str(e)}

            time.sleep(0.5)

        cross_results.append(sample_cross)

        # Periodic save
        if (si + 1) % 10 == 0:
            _save_cross_results(cross_results, conditions_found, args)
            print(f"  [checkpoint] {len(cross_results)} samples saved")

    _save_cross_results(cross_results, conditions_found, args)


def _save_cross_results(results, conditions, args):
    """Save cross-model judge results with agreement analysis."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute agreement metrics
    agreement = {}
    for cond in conditions:
        orig_scores = []
        cross_scores = []
        for r in results:
            if cond not in r or "error" in r[cond]:
                continue
            orig = r[cond].get("original_llm_judge", {}).get("overall", 0)
            cross = r[cond].get("cross_llm_judge", {}).get("overall", 0)
            if orig > 0 and cross > 0:
                orig_scores.append(orig)
                cross_scores.append(cross)

        if len(orig_scores) >= 3:
            n = len(orig_scores)
            mean_orig = sum(orig_scores) / n
            mean_cross = sum(cross_scores) / n
            # Spearman rank correlation (simplified)
            rank_orig = [sorted(orig_scores).index(v) for v in orig_scores]
            rank_cross = [sorted(cross_scores).index(v) for v in cross_scores]
            d_sq = sum((ro - rc) ** 2 for ro, rc in zip(rank_orig, rank_cross))
            spearman = 1 - (6 * d_sq) / (n * (n**2 - 1)) if n > 1 else 0
            agreement[cond] = {
                "n": n,
                "original_mean": round(mean_orig, 4),
                "cross_mean": round(mean_cross, 4),
                "spearman_rho": round(spearman, 4),
            }

    output = {
        "results": results,
        "agreement": agreement,
        "config": {
            "results_dir": str(args.results_dir),
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
        },
    }
    with open(output_dir / "cross_model_judge_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Cross-Model Judge Agreement ({args.judge_provider}/{args.judge_model})")
    print(f"{'='*70}")
    print(f"{'Condition':<18} {'N':>4} {'Orig Mean':>10} {'Cross Mean':>11} {'Spearman ρ':>11}")
    print("-" * 58)
    for cond, ag in agreement.items():
        print(f"{cond:<18} {ag['n']:>4} {ag['original_mean']:>10.3f} {ag['cross_mean']:>11.3f} {ag['spearman_rho']:>11.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 LLM Explanation Experiment")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Main experiment sub-command
    exp_parser = subparsers.add_parser("run", help="Run LLM explanation experiment")
    exp_parser.add_argument("--provider", choices=["claude", "openai"], default="claude")
    exp_parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    exp_parser.add_argument("--max-tokens", type=int, default=1500)
    exp_parser.add_argument("--datasets", nargs="+", default=["SKAB", "SMD", "AIHub71802", "SWaT"])
    exp_parser.add_argument("--samples-per-dataset", type=int, default=40)
    exp_parser.add_argument("--sample-modes", nargs="+", default=["tp"],
                            choices=["tp", "fn", "fp"],
                            help="Sample selection modes: tp (true positives), fn (false negatives), fp (false positives)")
    exp_parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    exp_parser.add_argument("--include-nosensor", action="store_true",
                            help="Add domain_nosensor/full_nosensor conditions to disentangle sensor name effect")
    exp_parser.add_argument("--consistency-repeats", type=int, default=3)
    exp_parser.add_argument("--data-root", type=str, default="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d")
    exp_parser.add_argument("--output-dir", type=str, default="runs/llm_explanation")

    # Cross-model judge sub-command
    judge_parser = subparsers.add_parser("cross-judge", help="Re-evaluate explanations with a different LLM judge")
    judge_parser.add_argument("--judge-provider", choices=["claude", "openai"], required=True)
    judge_parser.add_argument("--judge-model", type=str, required=True)
    judge_parser.add_argument("--results-dir", type=str, required=True,
                              help="Directory containing llm_explanation_results.json to re-judge")
    judge_parser.add_argument("--output-dir", type=str, required=True,
                              help="Directory to save cross-judge results")

    # Backward compatibility: insert 'run' if first positional arg is not a sub-command
    known_commands = {"run", "cross-judge"}
    argv = sys.argv[1:]
    if not argv or argv[0] not in known_commands:
        argv = ["run"] + argv

    args = parser.parse_args(argv)

    if args.command == "cross-judge":
        run_cross_model_judge(args)
    else:
        run_experiment(args)
