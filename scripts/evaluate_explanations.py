"""
Phase 2.6: Explanation Quality Evaluation

Evaluates generated explanations based on:
1. Completeness - Are all required sections present?
2. Technical Accuracy - Does it reference correct sensor/model info?
3. Actionability - Does it provide clear recommendations?
4. Domain Relevance - Does it use domain-specific knowledge?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from pathlib import Path
from typing import Dict, List, Any

print("=" * 70)
print("Phase 2.6: Explanation Quality Evaluation")
print("=" * 70)

# Find the latest batch explanations file
batch_dir = Path("./runs/batch_explanations")
batch_files = sorted(batch_dir.glob("batch_explanations_*.json"))

if not batch_files:
    print("No batch explanation files found!")
    sys.exit(1)

latest_file = batch_files[-1]
print(f"\nEvaluating: {latest_file}")

with open(latest_file, "r") as f:
    explanations = json.load(f)

print(f"Total explanations: {len(explanations)}")


def evaluate_explanation(exp: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single explanation."""
    text = exp.get("explanation", "")
    anomaly_info = exp.get("anomaly_info", {})
    feature_importance = exp.get("feature_importance", {})

    scores = {}

    # 1. Completeness (0-100)
    completeness_checks = {
        "has_anomaly_description": bool(re.search(r"anomal|deviation|observ", text, re.I)),
        "has_ml_explanation": bool(re.search(r"model|LSTM|autoencoder|reconstruct|threshold", text, re.I)),
        "has_severity_assessment": bool(re.search(r"severity|critical|high|medium|low|action|recommend", text, re.I)),
        "has_sensor_info": bool(re.search(r"accelerometer|pressure|temperature|sensor", text, re.I)),
        "has_numeric_values": bool(re.search(r"\d+\.\d+", text)),
    }
    scores["completeness"] = sum(completeness_checks.values()) / len(completeness_checks) * 100
    scores["completeness_details"] = completeness_checks

    # 2. Technical Accuracy (0-100)
    accuracy_checks = {
        "correct_dataset": anomaly_info.get("dataset", "").upper() in text.upper(),
        "correct_sensor": anomaly_info.get("sensor", "") in text,
        "mentions_threshold": "threshold" in text.lower(),
        "mentions_score": "score" in text.lower(),
        "has_sigma_reference": "sigma" in text.lower() or "deviation" in text.lower(),
    }
    scores["technical_accuracy"] = sum(accuracy_checks.values()) / len(accuracy_checks) * 100
    scores["accuracy_details"] = accuracy_checks

    # 3. Actionability (0-100)
    action_patterns = [
        r"should|recommend|suggest|check|inspect|monitor|investigate",
        r"next step|action|immediate|schedule|plan",
        r"operator|maintenance|team|personnel",
    ]
    action_count = sum(1 for p in action_patterns if re.search(p, text, re.I))
    scores["actionability"] = min(action_count / len(action_patterns) * 100, 100)

    # 4. Domain Relevance (0-100)
    domain_terms = [
        "valve", "circulation", "water", "vibration", "mechanical",
        "bearing", "pump", "flow", "pressure", "industrial",
        "manufacturing", "process", "equipment", "system"
    ]
    domain_count = sum(1 for term in domain_terms if term.lower() in text.lower())
    scores["domain_relevance"] = min(domain_count / 5 * 100, 100)  # 5+ terms = 100%

    # 5. Feature Importance Integration (0-100)
    fi_checks = {
        "has_fi_data": feature_importance is not None,
        "mentions_sensors": bool(re.search(r"contribut|sensor|important|impact", text, re.I)),
        "has_percentages": bool(re.search(r"\d+\.?\d*%", text)),
        "ranks_sensors": bool(re.search(r"(first|second|third|1\.|2\.|3\.)", text, re.I)),
    }
    scores["feature_importance_integration"] = sum(fi_checks.values()) / len(fi_checks) * 100

    # Overall score
    weights = {
        "completeness": 0.25,
        "technical_accuracy": 0.25,
        "actionability": 0.20,
        "domain_relevance": 0.15,
        "feature_importance_integration": 0.15,
    }
    scores["overall"] = sum(scores[k] * weights[k] for k in weights.keys())

    # Length metrics
    scores["word_count"] = len(text.split())
    scores["paragraph_count"] = text.count("\n\n") + 1

    return scores


# Evaluate all explanations
print("\n" + "-" * 70)
print("Evaluating each explanation...")
print("-" * 70)

all_scores = []
for i, exp in enumerate(explanations, 1):
    scores = evaluate_explanation(exp)
    all_scores.append(scores)

    run_name = exp.get("batch_metadata", {}).get("run_name", "unknown")
    idx = exp.get("batch_metadata", {}).get("anomaly_idx", "?")

    print(f"\n{i}. {run_name[:40]}... (idx={idx})")
    print(f"   Overall: {scores['overall']:.1f}/100")
    print(f"   - Completeness: {scores['completeness']:.0f}%")
    print(f"   - Technical: {scores['technical_accuracy']:.0f}%")
    print(f"   - Actionable: {scores['actionability']:.0f}%")
    print(f"   - Domain: {scores['domain_relevance']:.0f}%")
    print(f"   - Feature Imp: {scores['feature_importance_integration']:.0f}%")
    print(f"   Words: {scores['word_count']}")

# Aggregate statistics
print("\n" + "=" * 70)
print("Aggregate Quality Metrics")
print("=" * 70)

metrics = ["overall", "completeness", "technical_accuracy", "actionability",
           "domain_relevance", "feature_importance_integration"]

for metric in metrics:
    values = [s[metric] for s in all_scores]
    avg = sum(values) / len(values)
    min_val = min(values)
    max_val = max(values)
    print(f"\n{metric.replace('_', ' ').title()}:")
    print(f"  Average: {avg:.1f}")
    print(f"  Range: {min_val:.1f} - {max_val:.1f}")

# Word count stats
word_counts = [s["word_count"] for s in all_scores]
print(f"\nWord Count:")
print(f"  Average: {sum(word_counts)/len(word_counts):.0f}")
print(f"  Range: {min(word_counts)} - {max(word_counts)}")

# Quality summary
avg_overall = sum(s["overall"] for s in all_scores) / len(all_scores)

print("\n" + "=" * 70)
print("Quality Summary")
print("=" * 70)

if avg_overall >= 80:
    quality_grade = "EXCELLENT"
elif avg_overall >= 70:
    quality_grade = "GOOD"
elif avg_overall >= 60:
    quality_grade = "SATISFACTORY"
else:
    quality_grade = "NEEDS IMPROVEMENT"

print(f"\nOverall Quality Grade: {quality_grade}")
print(f"Average Score: {avg_overall:.1f}/100")
print(f"Explanations Meeting Threshold (≥70): {sum(1 for s in all_scores if s['overall'] >= 70)}/{len(all_scores)}")

# Save evaluation results
eval_output = batch_dir / f"evaluation_results_{latest_file.stem.split('_', 2)[-1]}.json"
eval_data = {
    "source_file": str(latest_file),
    "total_explanations": len(explanations),
    "average_overall_score": avg_overall,
    "quality_grade": quality_grade,
    "individual_scores": all_scores,
    "aggregate_metrics": {
        metric: {
            "average": sum(s[metric] for s in all_scores) / len(all_scores),
            "min": min(s[metric] for s in all_scores),
            "max": max(s[metric] for s in all_scores),
        }
        for metric in metrics
    }
}

with open(eval_output, "w") as f:
    json.dump(eval_data, f, indent=2)

print(f"\n✅ Evaluation saved to: {eval_output}")

print("\n" + "=" * 70)
print("Phase 2.6 Complete!")
print("=" * 70)
