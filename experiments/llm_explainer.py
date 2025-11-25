"""
LLM-based Anomaly Explanation System

This module provides 3-in-1 explanation capabilities:
1. Anomaly Explanation - What is the anomaly and why is it anomalous?
2. ML Model Explanation - Why did the ML model flag this point?
3. Domain Knowledge - What does this mean in the manufacturing context?
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from .llm_config import get_openai_client, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class AnomalyExplainer:
    """
    Main class for generating LLM-based explanations of anomaly detection results
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        self.client = get_openai_client()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def explain_anomaly(
        self,
        run_dir: Path,
        anomaly_idx: int,
        context_window: int = 20,
        include_model_interpretation: bool = True,
        include_domain_knowledge: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a detected anomaly

        Args:
            run_dir: Path to experiment run directory containing run.json and preds.csv
            anomaly_idx: Index of the anomaly point to explain
            context_window: Number of points before/after anomaly to include as context
            include_model_interpretation: Whether to include ML model interpretation
            include_domain_knowledge: Whether to include domain knowledge

        Returns:
            Dictionary containing the full explanation
        """
        # Load experiment data
        run_data = self._load_run_data(run_dir)
        preds_df = pd.read_csv(run_dir / "preds.csv")

        # Extract anomaly information
        anomaly_info = self._extract_anomaly_info(run_data, preds_df, anomaly_idx, context_window)

        # Generate LLM explanation
        explanation = self._generate_explanation(
            anomaly_info,
            include_model_interpretation=include_model_interpretation,
            include_domain_knowledge=include_domain_knowledge
        )

        return {
            "anomaly_info": anomaly_info,
            "explanation": explanation,
            "metadata": {
                "run_id": run_data["run"]["run_id"],
                "dataset": run_data["meta"]["dataset"],
                "method": run_data["detector"]["method"],
                "anomaly_idx": anomaly_idx
            }
        }

    def _load_run_data(self, run_dir: Path) -> Dict:
        """Load run.json metadata"""
        with open(run_dir / "run.json", "r") as f:
            return json.load(f)

    def _to_python_type(self, value):
        """Convert numpy types to native Python types for JSON serialization"""
        if value is None:
            return None
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _extract_anomaly_info(
        self,
        run_data: Dict,
        preds_df: pd.DataFrame,
        anomaly_idx: int,
        context_window: int
    ) -> Dict[str, Any]:
        """
        Extract relevant information about the anomaly for explanation
        """
        # Get the anomaly point
        anomaly_point = preds_df.iloc[anomaly_idx]

        # Get context window
        start_idx = max(0, anomaly_idx - context_window)
        end_idx = min(len(preds_df), anomaly_idx + context_window + 1)
        context_df = preds_df.iloc[start_idx:end_idx]

        # Calculate statistics
        normal_mask = context_df["label"] == 0
        normal_values = context_df[normal_mask]["value"].values
        normal_scores = context_df[normal_mask]["score"].values

        # Helper to convert to Python types
        to_py = self._to_python_type

        return {
            "dataset": run_data["meta"]["dataset"],
            "scenario": run_data["meta"].get("scenario", "unknown"),
            "sensor": run_data["meta"]["value_col"],
            "detector_method": run_data["detector"]["method"],
            "detector_config": run_data["detector"],

            # Anomaly point details
            "index": int(anomaly_idx),
            "value": to_py(anomaly_point["value"]),
            "score": to_py(anomaly_point["score"]),
            "prob": to_py(anomaly_point["prob"]),
            "true_label": to_py(anomaly_point["label"]),
            "predicted_label": to_py(anomaly_point["pred"]),

            # Context statistics
            "normal_value_mean": to_py(np.mean(normal_values)) if len(normal_values) > 0 else None,
            "normal_value_std": to_py(np.std(normal_values)) if len(normal_values) > 0 else None,
            "normal_score_mean": to_py(np.mean(normal_scores)) if len(normal_scores) > 0 else None,
            "normal_score_std": to_py(np.std(normal_scores)) if len(normal_scores) > 0 else None,

            # Deviation metrics
            "value_deviation_sigma": to_py(self._calculate_sigma_deviation(
                anomaly_point["value"], normal_values
            )) if len(normal_values) > 0 else None,
            "score_deviation_sigma": to_py(self._calculate_sigma_deviation(
                anomaly_point["score"], normal_scores
            )) if len(normal_scores) > 0 else None,

            # Detection threshold
            "threshold": to_py(run_data["decision"]["fixed_threshold"]),
            "threshold_exceeded": to_py(anomaly_point["score"] > run_data["decision"]["fixed_threshold"]),

            # Model performance on this file
            "overall_metrics": run_data["metrics"]
        }

    def _calculate_sigma_deviation(self, value: float, baseline: np.ndarray) -> float:
        """Calculate how many standard deviations away from baseline"""
        if len(baseline) == 0:
            return 0.0
        mean = np.mean(baseline)
        std = np.std(baseline)
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _generate_explanation(
        self,
        anomaly_info: Dict[str, Any],
        include_model_interpretation: bool,
        include_domain_knowledge: bool
    ) -> str:
        """
        Generate LLM-based explanation using OpenAI API
        """
        prompt = self._build_explanation_prompt(
            anomaly_info,
            include_model_interpretation,
            include_domain_knowledge
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating explanation: {str(e)}"

    def _get_system_prompt(self) -> str:
        """System prompt defining the LLM's role"""
        return """You are an expert in manufacturing process monitoring and anomaly detection.
Your role is to explain anomalies detected in industrial sensor data to engineers and operators.

Your explanations should:
1. Be clear, concise, and actionable
2. Explain what the anomaly is (sensor behavior)
3. Explain why the ML model flagged it (technical reasoning)
4. Connect to domain knowledge (what it means for the manufacturing process)
5. Avoid jargon when possible, but be technically accurate
6. Provide context and severity assessment

Focus on helping operators understand and respond to the anomaly."""

    def _format_value(self, value: Optional[float], format_spec: str = ".6f") -> str:
        """Safely format a value that might be None"""
        if value is None:
            return "N/A"
        return f"{value:{format_spec}}"

    def _build_explanation_prompt(
        self,
        anomaly_info: Dict[str, Any],
        include_model_interpretation: bool,
        include_domain_knowledge: bool
    ) -> str:
        """Build the user prompt with anomaly information"""

        # Basic anomaly information
        prompt = f"""Explain the following anomaly detected in manufacturing sensor data:

## Dataset and Context
- Dataset: {anomaly_info['dataset']}
- Scenario: {anomaly_info['scenario']}
- Sensor: {anomaly_info['sensor']}
- Time index: {anomaly_info['index']}

## Anomaly Details
- Sensor value: {anomaly_info['value']:.6f}
- Anomaly score: {anomaly_info['score']:.6f}
- Calibrated probability: {anomaly_info['prob']:.4f}
- True label: {'Anomaly' if anomaly_info['true_label'] == 1 else 'Normal'}
- Model prediction: {'Anomaly' if anomaly_info['predicted_label'] == 1 else 'Normal'}

## Baseline (Normal Behavior)
- Normal value mean: {self._format_value(anomaly_info['normal_value_mean'])}
- Normal value std: {self._format_value(anomaly_info['normal_value_std'])}
- Normal score mean: {self._format_value(anomaly_info['normal_score_mean'])}
- Normal score std: {self._format_value(anomaly_info['normal_score_std'])}

## Deviation Metrics
- Value deviation: {self._format_value(anomaly_info['value_deviation_sigma'], '.2f')} sigma from normal
- Score deviation: {self._format_value(anomaly_info['score_deviation_sigma'], '.2f')} sigma from normal baseline

"""

        # Add ML model interpretation section
        if include_model_interpretation:
            prompt += f"""
## ML Model Information
- Detection method: {anomaly_info['detector_method']}
- Detector configuration: {json.dumps(anomaly_info['detector_config'], indent=2)}
- Decision threshold: {anomaly_info['threshold']:.6f}
- Threshold exceeded: {'Yes' if anomaly_info['threshold_exceeded'] else 'No'}

### Overall Model Performance on this file:
- Precision: {anomaly_info['overall_metrics']['precision']:.4f}
- Recall: {anomaly_info['overall_metrics']['recall']:.4f}
- F1 Score: {anomaly_info['overall_metrics']['f1']:.4f}

"""

        # Add domain knowledge request
        if include_domain_knowledge:
            prompt += f"""
## Domain Context
Please explain what this anomaly might mean in the context of:
- Dataset: {anomaly_info['dataset']} ({self._get_dataset_description(anomaly_info['dataset'])})
- Sensor type: {anomaly_info['sensor']}
- What could cause such deviations?
- What should operators check or do?

"""

        prompt += """
Please provide a comprehensive explanation that:
1. Describes what anomalous behavior was observed
2. Explains why the ML model flagged it (based on the reconstruction error/score)
3. Interprets what this means for the manufacturing process
4. Assesses the severity and suggests next steps
"""

        return prompt

    def _get_dataset_description(self, dataset: str) -> str:
        """Get brief description of the dataset"""
        descriptions = {
            "SKAB": "Skoltech Anomaly Benchmark - valve monitoring in water circulation system",
            "SMD": "Server Machine Dataset - server monitoring metrics",
            "synthetic": "Synthetic time series with injected anomalies",
            "AIHub71802": "Korean manufacturing facility sensor data"
        }
        return descriptions.get(dataset, "Unknown dataset")

    def batch_explain(
        self,
        run_dir: Path,
        anomaly_indices: List[int],
        output_file: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple anomalies

        Args:
            run_dir: Path to experiment run directory
            anomaly_indices: List of anomaly indices to explain
            output_file: Optional path to save explanations as JSON

        Returns:
            List of explanation dictionaries
        """
        explanations = []

        for idx in anomaly_indices:
            print(f"Generating explanation for anomaly at index {idx}...")
            explanation = self.explain_anomaly(run_dir, idx)
            explanations.append(explanation)

        if output_file:
            with open(output_file, "w") as f:
                json.dump(explanations, f, indent=2)
            print(f"Saved {len(explanations)} explanations to {output_file}")

        return explanations
