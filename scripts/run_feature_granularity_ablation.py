"""Feature Attribution Granularity Ablation Experiment.

Tests what level of feature attribution information LLM needs:
- names_only: sensor names without scores
- top1: only the top-1 anomalous sensor with z-score
- top3: top-3 sensors with z-scores
- full_ranking: all sensors with z-scores (current default)
- zscore_only: z-scores without sensor names (ablation control)

This experiment answers: "What is the minimum attribution info LLM needs?"
— key to demonstrating non-trivial insight beyond "more info = better".

Usage:
    export $(cat .env | xargs)
    python scripts/run_feature_granularity_ablation.py \
        --provider claude --model claude-haiku-4-5-20251001 \
        --samples-per-dataset 40
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_llm_explanation_experiment import (
    ClaudeProvider, OpenAIProvider, LLMProvider,
    select_anomaly_samples, evaluate_faithfulness, evaluate_faithfulness_llm,
    SYSTEM_PROMPT,
)
from experiments.domain_knowledge.knowledge_retriever import DomainKnowledgeRetriever


def _compute_sensor_zscores(sample: Dict) -> List[Dict]:
    """Compute per-sensor z-scores at anomaly point, sorted by z-score desc."""
    sensors = sample.get("all_sensors")
    if not sensors:
        return []
    idx = sample["anomaly_idx"]
    results = []
    for col, vals in sensors.items():
        if idx >= len(vals):
            continue
        start = max(0, idx - 20)
        end = min(len(vals), idx + 21)
        local = vals[start:end]
        mu = sum(local) / len(local)
        std = (sum((v - mu) ** 2 for v in local) / len(local)) ** 0.5
        z = abs(vals[idx] - mu) / max(std, 1e-6)
        results.append({"name": col, "value": vals[idx], "z_score": z})
    results.sort(key=lambda x: x["z_score"], reverse=True)
    return results


def build_granularity_prompt(
    sample: Dict,
    granularity: str,
) -> str:
    """Build prompt with varying levels of feature attribution."""
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
    sensor_data = _compute_sensor_zscores(sample)

    if granularity == "no_feature":
        # Baseline: no sensor info at all
        pass

    elif granularity == "names_only":
        # Sensor names without any scores
        if sensor_data:
            prompt += "\n### Available Sensors\n"
            for s in sensor_data:
                prompt += f"- {s['name']}\n"

    elif granularity == "top1":
        # Only the most anomalous sensor
        if sensor_data:
            top = sensor_data[0]
            prompt += f"\n### Top Anomalous Sensor\n"
            prompt += f"- **{top['name']}**: value={top['value']:.3f}, z-score={top['z_score']:.2f}\n"

    elif granularity == "top3":
        # Top-3 anomalous sensors
        if sensor_data:
            prompt += "\n### Top-3 Anomalous Sensors\n"
            for s in sensor_data[:3]:
                prompt += f"- **{s['name']}**: value={s['value']:.3f}, z-score={s['z_score']:.2f}\n"

    elif granularity == "full_ranking":
        # All sensors with z-scores (current default)
        if sensor_data:
            prompt += "\n### Multi-Sensor Values at Anomaly Point\n"
            for s in sensor_data:
                prompt += f"- **{s['name']}**: {s['value']:.3f} (z-score: {s['z_score']:.2f})\n"

    elif granularity == "zscore_only":
        # Z-scores without sensor names (control)
        if sensor_data:
            prompt += "\n### Sensor Deviation Scores (anonymized)\n"
            for i, s in enumerate(sensor_data):
                prompt += f"- Sensor_{i+1}: z-score={s['z_score']:.2f}\n"

    elif granularity == "adversarial":
        # Adversarial: shuffle top-3 sensor names (assign wrong names to wrong z-scores)
        # If LLM reports these shuffled names, it proves reliance on provided attribution.
        import random as _rng
        if sensor_data and len(sensor_data) >= 3:
            top3 = sensor_data[:3]
            names = [s["name"] for s in top3]
            # Rotate names by 1 position (deterministic, always wrong)
            shuffled_names = names[1:] + names[:1]
            prompt += "\n### Top-3 Anomalous Sensors\n"
            for i, s in enumerate(top3):
                prompt += f"- **{shuffled_names[i]}**: value={s['value']:.3f}, z-score={s['z_score']:.2f}\n"
        elif sensor_data:
            # Less than 3 sensors: just swap first two
            prompt += "\n### Top Anomalous Sensors\n"
            if len(sensor_data) >= 2:
                prompt += f"- **{sensor_data[1]['name']}**: value={sensor_data[0]['value']:.3f}, z-score={sensor_data[0]['z_score']:.2f}\n"
                prompt += f"- **{sensor_data[0]['name']}**: value={sensor_data[1]['value']:.3f}, z-score={sensor_data[1]['z_score']:.2f}\n"

    prompt += "\nPlease analyze this anomaly and provide your structured explanation."
    return prompt


def evaluate_sensor_identification(explanation: str, sensor_data: List[Dict]) -> Dict:
    """Check if LLM correctly identifies the top anomalous sensors."""
    if not sensor_data:
        return {"top1_correct": None, "top3_overlap": None, "sensors_mentioned": 0}

    top1_name = sensor_data[0]["name"].lower()
    top3_names = {s["name"].lower() for s in sensor_data[:3]}
    explanation_lower = explanation.lower()

    # Count how many sensors are mentioned
    mentioned = [s for s in sensor_data if s["name"].lower() in explanation_lower]

    # Top-1 accuracy
    top1_correct = top1_name in explanation_lower

    # Top-3 overlap
    top3_found = sum(1 for name in top3_names if name in explanation_lower)

    return {
        "top1_correct": top1_correct,
        "top3_overlap": top3_found / min(3, len(sensor_data)),
        "sensors_mentioned": len(mentioned),
        "total_sensors": len(sensor_data),
    }


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize provider
    if args.provider == "claude":
        provider = ClaudeProvider(model=args.model, max_tokens=args.max_tokens)
    else:
        provider = OpenAIProvider(model=args.model, max_tokens=args.max_tokens)

    # Only AIHub has multi-sensor (effect only exists there)
    # But include SKAB/SMD as controls
    datasets = args.datasets
    print(f"Selecting samples from {datasets}...")
    all_samples = []
    for ds in datasets:
        samples = select_anomaly_samples(ds, args.data_root, n_samples=args.samples_per_dataset)
        all_samples.extend(samples)
        print(f"  {ds}: {len(samples)} samples")

    # Granularity conditions
    conditions = [
        "no_feature",      # baseline
        "names_only",      # sensor names without scores
        "top1",            # top-1 sensor with z-score
        "top3",            # top-3 sensors with z-scores
        "full_ranking",    # all sensors (current default)
        "zscore_only",     # z-scores without names (control)
        "adversarial",     # shuffled sensor names (proves LLM uses attribution)
    ]

    results = []
    total = len(all_samples) * len(conditions)
    call_n = 0

    for si, sample in enumerate(all_samples):
        sensor_data = _compute_sensor_zscores(sample)
        sample_result = {
            "sample": {k: v for k, v in sample.items() if k not in ("context_values", "all_sensors")},
            "n_sensors": len(sensor_data),
        }

        for cond in conditions:
            call_n += 1
            print(f"[{call_n}/{total}] {sample['dataset']}:{sample['file']}@{sample['anomaly_idx']} ({cond})",
                  end=" ", flush=True)

            prompt = build_granularity_prompt(sample, cond)

            for attempt in range(3):
                try:
                    t0 = time.time()
                    explanation = provider.generate(SYSTEM_PROMPT, prompt)
                    latency = time.time() - t0

                    faith_kw = evaluate_faithfulness(explanation, sample)
                    sensor_eval = evaluate_sensor_identification(explanation, sensor_data)

                    # LLM-as-judge
                    judge = evaluate_faithfulness_llm(explanation, sample, provider)
                    time.sleep(0.3)

                    print(f"kw={faith_kw:.2f} llm={judge.get('overall',0):.2f} "
                          f"top1={'Y' if sensor_eval.get('top1_correct') else 'N'} {latency:.1f}s")

                    sample_result[cond] = {
                        "explanation": explanation,
                        "faithfulness_kw": faith_kw,
                        "faithfulness_llm": judge,
                        "sensor_eval": sensor_eval,
                        "latency": latency,
                    }
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"retry({attempt+1})...", end=" ", flush=True)
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"ERROR: {e}")
                        sample_result[cond] = {"error": str(e)}

            time.sleep(0.5)

        results.append(sample_result)

        # Checkpoint
        if (si + 1) % 10 == 0:
            _save(results, conditions, output_dir)
            print(f"  [checkpoint] {len(results)} samples saved")

    _save(results, conditions, output_dir)
    _print_analysis(results, conditions)


def _save(results, conditions, output_dir):
    summary = {}
    for cond in conditions:
        kw_scores = [r[cond]["faithfulness_kw"] for r in results if cond in r and "faithfulness_kw" in r.get(cond, {})]
        llm_scores = [r[cond]["faithfulness_llm"]["overall"] for r in results
                      if cond in r and "faithfulness_llm" in r.get(cond, {}) and r[cond]["faithfulness_llm"].get("overall", 0) > 0]
        top1_scores = [r[cond]["sensor_eval"]["top1_correct"] for r in results
                       if cond in r and "sensor_eval" in r.get(cond, {}) and r[cond]["sensor_eval"].get("top1_correct") is not None]
        top3_scores = [r[cond]["sensor_eval"]["top3_overlap"] for r in results
                       if cond in r and "sensor_eval" in r.get(cond, {}) and r[cond]["sensor_eval"].get("top3_overlap") is not None]
        sid_scores = [r[cond]["faithfulness_llm"].get("sensor_id", 0) for r in results
                      if cond in r and "faithfulness_llm" in r.get(cond, {}) and r[cond]["faithfulness_llm"].get("sensor_id", 0) > 0]

        summary[cond] = {
            "n": len(kw_scores),
            "mean_kw": sum(kw_scores) / len(kw_scores) if kw_scores else 0,
            "mean_llm": sum(llm_scores) / len(llm_scores) if llm_scores else 0,
            "top1_accuracy": sum(top1_scores) / len(top1_scores) if top1_scores else 0,
            "top3_overlap": sum(top3_scores) / len(top3_scores) if top3_scores else 0,
            "mean_sensor_id": sum(sid_scores) / len(sid_scores) if sid_scores else 0,
        }

    with open(output_dir / "granularity_ablation_results.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2, ensure_ascii=False)


def _print_analysis(results, conditions):
    print(f"\n{'='*80}")
    print("Feature Attribution Granularity Ablation Results")
    print(f"{'='*80}")

    # Split by multi-sensor vs single-sensor
    multi = [r for r in results if r.get("n_sensors", 0) > 1]
    single = [r for r in results if r.get("n_sensors", 0) <= 1]

    for label, subset in [("ALL", results), ("Multi-sensor", multi), ("Single-sensor", single)]:
        if not subset:
            continue
        print(f"\n--- {label} (N={len(subset)}) ---")
        print(f"{'Condition':<18} {'KW Faith':>10} {'LLM Judge':>10} {'Top1 Acc':>10} {'Top3 Ovlp':>10} {'sensor_id':>10}")
        print("-" * 72)
        for cond in conditions:
            kw = [r[cond]["faithfulness_kw"] for r in subset if cond in r and "faithfulness_kw" in r.get(cond, {})]
            llm = [r[cond]["faithfulness_llm"]["overall"] for r in subset
                   if cond in r and "faithfulness_llm" in r.get(cond, {}) and r[cond]["faithfulness_llm"].get("overall", 0) > 0]
            t1 = [r[cond]["sensor_eval"]["top1_correct"] for r in subset
                  if cond in r and "sensor_eval" in r.get(cond, {}) and r[cond]["sensor_eval"].get("top1_correct") is not None]
            t3 = [r[cond]["sensor_eval"]["top3_overlap"] for r in subset
                  if cond in r and "sensor_eval" in r.get(cond, {}) and r[cond]["sensor_eval"].get("top3_overlap") is not None]
            sid = [r[cond]["faithfulness_llm"].get("sensor_id", 0) for r in subset
                   if cond in r and "faithfulness_llm" in r.get(cond, {}) and r[cond]["faithfulness_llm"].get("sensor_id", 0) > 0]

            kw_avg = sum(kw) / len(kw) if kw else 0
            llm_avg = sum(llm) / len(llm) if llm else 0
            t1_avg = sum(t1) / len(t1) if t1 else 0
            t3_avg = sum(t3) / len(t3) if t3 else 0
            sid_avg = sum(sid) / len(sid) if sid else 0

            print(f"{cond:<18} {kw_avg:>10.3f} {llm_avg:>10.3f} {t1_avg:>10.1%} {t3_avg:>10.3f} {sid_avg:>10.2f}/5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Attribution Granularity Ablation")
    parser.add_argument("--provider", choices=["claude", "openai"], default="claude")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--datasets", nargs="+", default=["AIHub71802", "SKAB", "SMD", "SWaT"])
    parser.add_argument("--samples-per-dataset", type=int, default=40)
    parser.add_argument("--data-root", type=str, default="/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d")
    parser.add_argument("--output-dir", type=str, default="runs/feature_granularity_ablation")
    args = parser.parse_args()
    run_experiment(args)
