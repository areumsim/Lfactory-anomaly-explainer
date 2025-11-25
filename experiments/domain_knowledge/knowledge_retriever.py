"""
Domain Knowledge Retriever

Retrieves relevant domain knowledge for anomaly explanation
based on dataset, sensor, and anomaly characteristics.
"""
import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path


class DomainKnowledgeRetriever:
    """
    Retrieves and formats domain knowledge for LLM explanations.

    Supports SKAB (industrial valve monitoring) and SMD (server monitoring).
    """

    def __init__(self, knowledge_dir: Optional[str] = None):
        """
        Initialize the knowledge retriever.

        Args:
            knowledge_dir: Directory containing knowledge YAML files.
                          Defaults to same directory as this module.
        """
        if knowledge_dir is None:
            knowledge_dir = os.path.dirname(os.path.abspath(__file__))

        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_cache: Dict[str, Dict] = {}

        # Load available knowledge bases
        self._load_knowledge_bases()

    def _load_knowledge_bases(self):
        """Load all YAML knowledge files."""
        for yaml_file in self.knowledge_dir.glob("*_knowledge.yaml"):
            dataset_name = yaml_file.stem.replace("_knowledge", "").upper()
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    self.knowledge_cache[dataset_name] = yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not load {yaml_file}: {e}")

    def get_dataset_context(self, dataset: str) -> str:
        """
        Get general context about a dataset.

        Args:
            dataset: Dataset name (e.g., "SKAB", "SMD")

        Returns:
            Formatted string with dataset context
        """
        kb = self.knowledge_cache.get(dataset.upper(), {})
        if not kb:
            return f"Dataset: {dataset} (no domain knowledge available)"

        ds_info = kb.get("dataset", {})
        lines = [
            f"**Dataset**: {ds_info.get('full_name', dataset)}",
            f"**Domain**: {ds_info.get('domain', 'Unknown')}",
            f"**Description**: {ds_info.get('description', 'No description available').strip()}"
        ]

        return "\n".join(lines)

    def get_sensor_knowledge(
        self,
        dataset: str,
        sensor_name: str
    ) -> Dict[str, Any]:
        """
        Get detailed knowledge about a specific sensor.

        Args:
            dataset: Dataset name
            sensor_name: Name of the sensor

        Returns:
            Dictionary with sensor knowledge
        """
        kb = self.knowledge_cache.get(dataset.upper(), {})
        if not kb:
            return {"sensor": sensor_name, "knowledge_available": False}

        # Check sensors section
        sensors = kb.get("sensors", {})
        sensor_info = sensors.get(sensor_name, {})

        if not sensor_info:
            # Try to find partial match
            for key in sensors:
                if sensor_name.lower() in key.lower() or key.lower() in sensor_name.lower():
                    sensor_info = sensors[key]
                    break

        return {
            "sensor": sensor_name,
            "knowledge_available": bool(sensor_info),
            "type": sensor_info.get("type", "unknown"),
            "unit": sensor_info.get("unit", ""),
            "normal_range": sensor_info.get("normal_range", []),
            "description": sensor_info.get("description", ""),
            "anomaly_indicators": sensor_info.get("anomaly_indicators", {}),
            "failure_modes": sensor_info.get("failure_modes", [])
        }

    def get_relevant_failure_modes(
        self,
        dataset: str,
        sensor_name: str,
        anomaly_pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get failure modes relevant to the sensor and anomaly pattern.

        Args:
            dataset: Dataset name
            sensor_name: Name of the sensor
            anomaly_pattern: Description of anomaly pattern (e.g., "high", "spike", "drift")

        Returns:
            List of relevant failure mode dictionaries
        """
        sensor_knowledge = self.get_sensor_knowledge(dataset, sensor_name)
        failure_modes = sensor_knowledge.get("failure_modes", [])

        if not anomaly_pattern:
            return failure_modes

        # Filter by pattern if provided
        pattern_lower = anomaly_pattern.lower()
        relevant = []
        for fm in failure_modes:
            fm_pattern = fm.get("pattern", "").lower()
            if any(word in fm_pattern for word in pattern_lower.split()):
                relevant.append(fm)

        return relevant if relevant else failure_modes

    def get_correlation_rules(
        self,
        dataset: str,
        sensors: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get correlation rules involving the specified sensors.

        Args:
            dataset: Dataset name
            sensors: List of sensor names to check

        Returns:
            List of relevant correlation rules
        """
        kb = self.knowledge_cache.get(dataset.upper(), {})
        if not kb:
            return []

        rules = kb.get("correlation_rules", [])
        sensors_lower = [s.lower() for s in sensors]

        relevant = []
        for rule in rules:
            rule_sensors = rule.get("sensors", [])
            rule_sensors_lower = [s.lower() for s in rule_sensors]

            # Check if any sensor matches
            if any(s in rule_sensors_lower for s in sensors_lower):
                relevant.append(rule)

        return relevant

    def get_severity_assessment(
        self,
        dataset: str,
        anomaly_score: float,
        sensors_affected: List[str]
    ) -> Dict[str, Any]:
        """
        Get severity assessment based on anomaly characteristics.

        Args:
            dataset: Dataset name
            anomaly_score: Anomaly score (0-1)
            sensors_affected: List of sensors showing anomaly

        Returns:
            Severity assessment dictionary
        """
        kb = self.knowledge_cache.get(dataset.upper(), {})
        severity_levels = kb.get("severity_levels", {})

        # Determine severity based on score and number of sensors
        if anomaly_score > 0.9 or len(sensors_affected) > 3:
            level = "critical"
        elif anomaly_score > 0.7 or len(sensors_affected) > 2:
            level = "high"
        elif anomaly_score > 0.5:
            level = "medium"
        else:
            level = "low"

        level_info = severity_levels.get(level, {})

        return {
            "level": level,
            "description": level_info.get("description", f"{level.capitalize()} severity"),
            "recommended_actions": level_info.get("actions", [])
        }

    def format_knowledge_for_llm(
        self,
        dataset: str,
        sensor_name: str,
        anomaly_score: float,
        top_contributors: Optional[List[str]] = None,
        anomaly_pattern: Optional[str] = None
    ) -> str:
        """
        Format all relevant knowledge for LLM prompt.

        Args:
            dataset: Dataset name
            sensor_name: Primary sensor name
            anomaly_score: Anomaly score (0-1)
            top_contributors: List of top contributing sensors
            anomaly_pattern: Description of anomaly pattern

        Returns:
            Formatted string for LLM prompt
        """
        lines = []

        # Dataset context
        lines.append("## Domain Knowledge")
        lines.append(self.get_dataset_context(dataset))
        lines.append("")

        # Primary sensor knowledge
        sensor_kb = self.get_sensor_knowledge(dataset, sensor_name)
        if sensor_kb.get("knowledge_available"):
            lines.append(f"### Primary Sensor: {sensor_name}")
            lines.append(f"- **Type**: {sensor_kb.get('type', 'unknown')}")
            lines.append(f"- **Unit**: {sensor_kb.get('unit', 'N/A')}")
            if sensor_kb.get('normal_range'):
                lines.append(f"- **Normal Range**: {sensor_kb['normal_range']}")
            lines.append(f"- **Description**: {sensor_kb.get('description', '')}")

            # Anomaly indicators
            indicators = sensor_kb.get("anomaly_indicators", {})
            if indicators:
                lines.append("\n**Anomaly Indicators:**")
                for key, value in indicators.items():
                    if isinstance(value, str):
                        lines.append(f"  - {key}: {value}")
            lines.append("")

        # Failure modes
        failure_modes = self.get_relevant_failure_modes(dataset, sensor_name, anomaly_pattern)
        if failure_modes:
            lines.append("### Possible Failure Modes")
            for i, fm in enumerate(failure_modes[:3], 1):
                lines.append(f"{i}. **Pattern**: {fm.get('pattern', 'N/A')}")
                lines.append(f"   - Cause: {fm.get('cause', 'Unknown')}")
                lines.append(f"   - Severity: {fm.get('severity', 'Unknown')}")
                lines.append(f"   - Action: {fm.get('action', 'Investigate')}")
            lines.append("")

        # Correlation rules if multiple sensors
        if top_contributors and len(top_contributors) > 1:
            correlations = self.get_correlation_rules(dataset, top_contributors)
            if correlations:
                lines.append("### Sensor Correlations")
                for rule in correlations[:2]:
                    lines.append(f"- Sensors: {rule.get('sensors', [])}")
                    lines.append(f"  - Relationship: {rule.get('relationship', '')}")
                    lines.append(f"  - Deviation meaning: {rule.get('deviation_meaning', '')}")
                lines.append("")

        # Severity assessment
        severity = self.get_severity_assessment(
            dataset,
            anomaly_score,
            top_contributors or [sensor_name]
        )
        lines.append("### Severity Assessment")
        lines.append(f"- **Level**: {severity['level'].upper()}")
        lines.append(f"- **Description**: {severity['description']}")
        if severity.get('recommended_actions'):
            lines.append("- **Recommended Actions**:")
            for action in severity['recommended_actions'][:3]:
                lines.append(f"  - {action}")

        return "\n".join(lines)

    def get_scenario_knowledge(
        self,
        dataset: str,
        scenario: str
    ) -> Dict[str, Any]:
        """
        Get knowledge about a specific scenario (e.g., valve1, valve2).

        Args:
            dataset: Dataset name
            scenario: Scenario name

        Returns:
            Scenario knowledge dictionary
        """
        kb = self.knowledge_cache.get(dataset.upper(), {})
        scenarios = kb.get("scenarios", {})

        scenario_info = scenarios.get(scenario, {})

        return {
            "scenario": scenario,
            "description": scenario_info.get("description", ""),
            "anomaly_types": scenario_info.get("anomaly_types", []),
            "typical_causes": scenario_info.get("typical_causes", [])
        }
