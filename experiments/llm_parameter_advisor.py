"""
LLM Parameter Advisor for Anomaly Detection Optimization

Phase 3: LLM-guided parameter tuning based on domain knowledge
and current performance metrics.
"""
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from .llm_config import get_openai_client, DEFAULT_MODEL, DEFAULT_TEMPERATURE
from .domain_knowledge import DomainKnowledgeRetriever


class LLMParameterAdvisor:
    """
    Uses LLM to suggest optimal ML parameters based on:
    - Current performance metrics
    - Dataset characteristics
    - Domain knowledge
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3
    ):
        self.client = get_openai_client()
        self.model = model
        self.temperature = temperature
        self.knowledge_retriever = DomainKnowledgeRetriever()

        # Default parameters for each detector
        self.default_params = {
            "lstm_autoencoder": {
                "sequence_length": 50,
                "latent_dim": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 32,
                "quantile": 0.95
            },
            "isolation_forest": {
                "window": 50,
                "contamination": 0.1,
                "n_estimators": 100,
                "max_samples": "auto",
                "max_features": 1.0
            },
            "knn": {
                "k": 10,
                "window": 50,
                "quantile": 0.99,
                "algorithm": "ball_tree"
            }
        }

    def suggest_parameters(
        self,
        dataset: str,
        detector: str,
        current_params: Dict[str, Any],
        current_metrics: Dict[str, float],
        data_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get LLM suggestions for parameter optimization.

        Args:
            dataset: Dataset name (SKAB, SMD, etc.)
            detector: Detector type (lstm_autoencoder, isolation_forest, knn)
            current_params: Current parameter values
            current_metrics: Current performance (f1, precision, recall, etc.)
            data_stats: Optional data statistics (length, anomaly_rate, etc.)

        Returns:
            Dictionary with suggested parameters and reasoning
        """
        # Get domain knowledge
        domain_context = self.knowledge_retriever.get_dataset_context(dataset)

        # Build prompt
        prompt = self._build_suggestion_prompt(
            dataset=dataset,
            detector=detector,
            current_params=current_params,
            current_metrics=current_metrics,
            data_stats=data_stats,
            domain_context=domain_context
        )

        # Get LLM response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )

            llm_response = response.choices[0].message.content

            # Parse suggestions
            suggestions = self._parse_suggestions(llm_response, detector)

            return {
                "dataset": dataset,
                "detector": detector,
                "current_params": current_params,
                "current_metrics": current_metrics,
                "suggested_params": suggestions["params"],
                "reasoning": suggestions["reasoning"],
                "expected_improvement": suggestions["expected_improvement"],
                "raw_response": llm_response
            }

        except Exception as e:
            return {
                "error": str(e),
                "dataset": dataset,
                "detector": detector,
                "current_params": current_params,
                "suggested_params": current_params  # Keep current if error
            }

    def _get_system_prompt(self) -> str:
        """System prompt for parameter advisor."""
        return """You are an expert ML engineer specializing in anomaly detection for manufacturing systems.

Your task is to suggest optimal hyperparameters based on:
1. Current performance metrics
2. Dataset characteristics
3. Domain knowledge about the manufacturing process

For each parameter suggestion:
- Explain WHY this change will help
- Ground reasoning in domain knowledge
- Estimate expected improvement

Be specific and quantitative. Avoid generic advice."""

    def _build_suggestion_prompt(
        self,
        dataset: str,
        detector: str,
        current_params: Dict[str, Any],
        current_metrics: Dict[str, float],
        data_stats: Optional[Dict[str, Any]],
        domain_context: str
    ) -> str:
        """Build the prompt for parameter suggestions."""

        prompt = f"""Suggest optimal parameters for anomaly detection.

## Dataset: {dataset}
{domain_context}

## Data Statistics
"""
        if data_stats:
            prompt += f"""- Total samples: {data_stats.get('n_samples', 'unknown')}
- Anomaly rate: {data_stats.get('anomaly_rate', 'unknown'):.2%}
- Number of features: {data_stats.get('n_features', 'unknown')}
"""
        else:
            prompt += "- Not provided\n"

        prompt += f"""
## Current Detector: {detector}

### Current Parameters
```json
{json.dumps(current_params, indent=2)}
```

### Current Performance
- F1 Score: {current_metrics.get('f1', 0):.4f}
- Precision: {current_metrics.get('precision', 0):.4f}
- Recall: {current_metrics.get('recall', 0):.4f}
- AUC-PR: {current_metrics.get('auc_pr', 0):.4f}

## Task
Suggest improved parameters for {detector} on {dataset}.

For EACH parameter you want to change:
1. Current value → Suggested value
2. Reason for change (domain-specific)
3. Expected impact on metrics

Format your response as:

### Suggested Parameters
[List each parameter change]

### Reasoning
[Detailed explanation for each change]

### Expected Improvement
- F1: [current] → [expected]
- Precision: [current] → [expected]
- Recall: [current] → [expected]

### Confidence Level
[High/Medium/Low] - [Why]
"""

        return prompt

    def _parse_suggestions(
        self,
        llm_response: str,
        detector: str
    ) -> Dict[str, Any]:
        """Parse LLM response to extract parameter suggestions."""

        # Get default params as base
        suggested_params = self.default_params.get(detector, {}).copy()

        reasoning = []
        expected_improvement = {}

        # Parse parameter changes from response
        lines = llm_response.split('\n')

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # LSTM-AE parameters
            if 'sequence_length' in line_lower or 'window' in line_lower:
                val = self._extract_number(line)
                if val and detector == "lstm_autoencoder":
                    suggested_params["sequence_length"] = int(val)
                elif val:
                    suggested_params["window"] = int(val)

            if 'latent_dim' in line_lower:
                val = self._extract_number(line)
                if val:
                    suggested_params["latent_dim"] = int(val)

            if 'epoch' in line_lower:
                val = self._extract_number(line)
                if val:
                    suggested_params["epochs"] = int(val)

            if 'learning_rate' in line_lower or 'lr' in line_lower:
                val = self._extract_float(line)
                if val:
                    suggested_params["learning_rate"] = val

            if 'batch_size' in line_lower:
                val = self._extract_number(line)
                if val:
                    suggested_params["batch_size"] = int(val)

            # IsolationForest parameters
            if 'contamination' in line_lower:
                val = self._extract_float(line)
                if val and 0 < val < 1:
                    suggested_params["contamination"] = val

            if 'n_estimators' in line_lower or 'estimators' in line_lower:
                val = self._extract_number(line)
                if val:
                    suggested_params["n_estimators"] = int(val)

            # kNN parameters
            if line_lower.startswith('k ') or 'k=' in line_lower or 'k:' in line_lower:
                val = self._extract_number(line)
                if val and val > 0 and val < 100:
                    suggested_params["k"] = int(val)

            if 'quantile' in line_lower:
                val = self._extract_float(line)
                if val and 0 < val < 1:
                    suggested_params["quantile"] = val

            # Extract reasoning sections
            if 'reason' in line_lower or 'because' in line_lower:
                reasoning.append(line.strip())

        # Extract expected improvement
        for line in lines:
            if 'f1' in line.lower() and '→' in line:
                try:
                    parts = line.split('→')
                    if len(parts) == 2:
                        expected_improvement["f1"] = self._extract_float(parts[1])
                except:
                    pass

        return {
            "params": suggested_params,
            "reasoning": reasoning,
            "expected_improvement": expected_improvement
        }

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract integer from text."""
        import re
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            return int(numbers[-1])
        return None

    def _extract_float(self, text: str) -> Optional[float]:
        """Extract float from text."""
        import re
        numbers = re.findall(r'(\d+\.?\d*)', text)
        if numbers:
            return float(numbers[-1])
        return None

    def batch_suggest(
        self,
        experiments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate suggestions for multiple experiments.

        Args:
            experiments: List of dicts with dataset, detector, params, metrics

        Returns:
            List of suggestion results
        """
        results = []

        for exp in experiments:
            print(f"Generating suggestions for {exp['dataset']}/{exp['detector']}...")

            suggestion = self.suggest_parameters(
                dataset=exp["dataset"],
                detector=exp["detector"],
                current_params=exp.get("params", {}),
                current_metrics=exp.get("metrics", {}),
                data_stats=exp.get("data_stats")
            )

            results.append(suggestion)

        return results


def load_baseline_results(runs_dir: Path, dataset: str, detector: str) -> Dict[str, Any]:
    """Load baseline experiment results for a dataset/detector combination."""

    # Find matching runs
    pattern = f"{dataset}*{detector}*"
    matching_runs = list(runs_dir.glob(pattern))

    if not matching_runs:
        return {}

    # Load first matching run's results
    run_dir = matching_runs[0]

    results = {
        "dataset": dataset,
        "detector": detector,
        "params": {},
        "metrics": {}
    }

    # Load run.json
    run_json = run_dir / "run.json"
    if run_json.exists():
        with open(run_json) as f:
            data = json.load(f)
            results["params"] = data.get("detector", {})
            results["metrics"] = data.get("metrics", {})

    return results
