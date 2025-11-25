"""
Test Domain Knowledge Retrieval System

Phase 2.4: Domain Knowledge Base
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.domain_knowledge import DomainKnowledgeRetriever

print("=" * 60)
print("Test Domain Knowledge Retrieval")
print("=" * 60)

# Initialize retriever
retriever = DomainKnowledgeRetriever()

print("\n1. Testing SKAB Knowledge")
print("-" * 40)

# Test dataset context
context = retriever.get_dataset_context("SKAB")
print("\nDataset Context:")
print(context)

# Test sensor knowledge
print("\n\n2. Sensor Knowledge: Accelerometer1RMS")
print("-" * 40)
sensor_kb = retriever.get_sensor_knowledge("SKAB", "Accelerometer1RMS")
print(f"Type: {sensor_kb['type']}")
print(f"Unit: {sensor_kb['unit']}")
print(f"Normal Range: {sensor_kb['normal_range']}")
print(f"Description: {sensor_kb['description']}")
print(f"\nAnomaly Indicators:")
for key, val in sensor_kb.get('anomaly_indicators', {}).items():
    print(f"  - {key}: {val}")

# Test failure modes
print("\n\n3. Failure Modes")
print("-" * 40)
failure_modes = retriever.get_relevant_failure_modes("SKAB", "Accelerometer1RMS")
for i, fm in enumerate(failure_modes[:2], 1):
    print(f"\n{i}. {fm['pattern']}")
    print(f"   Cause: {fm['cause']}")
    print(f"   Severity: {fm['severity']}")
    print(f"   Action: {fm['action']}")

# Test correlation rules
print("\n\n4. Correlation Rules")
print("-" * 40)
correlations = retriever.get_correlation_rules("SKAB", ["Accelerometer1RMS", "Pressure"])
for rule in correlations:
    print(f"\nSensors: {rule['sensors']}")
    print(f"Relationship: {rule['relationship']}")
    print(f"Meaning: {rule['deviation_meaning']}")

# Test severity assessment
print("\n\n5. Severity Assessment")
print("-" * 40)
severity = retriever.get_severity_assessment("SKAB", anomaly_score=0.75, sensors_affected=["Accelerometer1RMS", "Pressure"])
print(f"Level: {severity['level']}")
print(f"Description: {severity['description']}")
print(f"Actions: {severity['recommended_actions']}")

# Test full formatted output
print("\n\n6. Full Formatted Output for LLM")
print("-" * 40)
formatted = retriever.format_knowledge_for_llm(
    dataset="SKAB",
    sensor_name="Accelerometer1RMS",
    anomaly_score=0.75,
    top_contributors=["Accelerometer1RMS", "Pressure", "Current"]
)
print(formatted)

# Test SMD knowledge
print("\n\n" + "=" * 60)
print("7. Testing SMD Knowledge")
print("=" * 60)

smd_context = retriever.get_dataset_context("SMD")
print("\nSMD Dataset Context:")
print(smd_context)

print("\n" + "=" * 60)
print("Domain Knowledge Tests Complete!")
print("=" * 60)
