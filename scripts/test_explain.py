#!/usr/bin/env python3
"""Test Script for RAG-Bayes Explainer (Phase 2 Prototype)

This script tests the explain_rag module with various queries and contexts.
Used for validating the RAG explainer before integration into main_experiment.py.

Usage:
    # Test with TF-IDF only (no LLM)
    python scripts/test_explain.py

    # Test with specific LLM provider
    python scripts/test_explain.py --llm-provider openai_gpt35
    python scripts/test_explain.py --llm-provider local_exaone_35_78b

    # Test with custom config
    python scripts/test_explain.py --config experiments/llm_config.yaml

Dependencies:
    - experiments/explain_rag.py (always required)
    - experiments/llm_config.yaml (config file)
    - Optional: openai, transformers, torch (depending on provider)

Expected output:
    - Query results with retrieved documents
    - LLM-generated explanations (if provider specified)
    - Bayes prior recommendations
    - Formatted markdown explanation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.explain_rag import RAGExplainer


def test_basic_retrieval():
    """Test 1: TF-IDF retrieval without LLM."""
    print("=" * 80)
    print("Test 1: TF-IDF Retrieval Only (No LLM)")
    print("=" * 80)

    explainer = RAGExplainer(
        config_path="experiments/llm_config.yaml",
        llm_provider=None  # No LLM
    )

    query = "Why should I use calibration in anomaly detection?"
    context = {
        "dataset": "SKAB",
        "imbalance": 0.05,  # 5% anomalies
        "snr": 4.2,
    }

    result = explainer.explain(query=query, context=context)

    print(f"\n[Query]: {result['query']}")
    print(f"\n[Context]: {context}")
    print(f"\n[Retrieved Documents]: {len(result['retrieved_docs'])} chunks")
    for i, doc in enumerate(result['retrieved_docs']):
        print(f"\n--- Doc {i+1} (Citation: {result['citations'][i]}) ---")
        print(doc[:300] + "..." if len(doc) > 300 else doc)

    print(f"\n[Bayes Recommendations]: {len(result['bayes_recommendations'])}")
    for rec in result['bayes_recommendations']:
        print(f"  - {rec['adjustment']}")
        print(f"    Reason: {rec['reason']}")

    print("\n[Full Explanation]:")
    print(result['explanation'])

    return result


def test_llm_generation(llm_provider: str):
    """Test 2: Full RAG with LLM generation."""
    print("\n" + "=" * 80)
    print(f"Test 2: RAG with LLM Provider '{llm_provider}'")
    print("=" * 80)

    try:
        explainer = RAGExplainer(
            config_path="experiments/llm_config.yaml",
            llm_provider=llm_provider
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load LLM provider '{llm_provider}': {e}")
        print("Skipping LLM test.")
        return None

    query = "What is the difference between point-wise F1 and event-wise F1?"
    context = {
        "dataset": "SMD",
        "detector": "Hybrid",
        "point_f1": 0.72,
        "event_f1": 0.68,
    }

    print(f"\n[Query]: {query}")
    print(f"[Context]: {context}")
    print("\n[Generating explanation... this may take 10-30 seconds]")

    result = explainer.explain(query=query, context=context)

    print(f"\n[LLM Response]:")
    print(result['llm_response'])

    print(f"\n[Full Explanation]:")
    print(result['explanation'][:500] + "..." if len(result['explanation']) > 500 else result['explanation'])

    return result


def test_bayes_rules():
    """Test 3: Bayes prior adjustment rules."""
    print("\n" + "=" * 80)
    print("Test 3: Bayes Prior Adjustment Rules")
    print("=" * 80)

    explainer = RAGExplainer(
        config_path="experiments/llm_config.yaml",
        llm_provider=None
    )

    # Test case 1: High imbalance
    print("\n[Test Case 1: High Imbalance Dataset]")
    query1 = "How should I set the cost matrix?"
    context1 = {"imbalance": 0.15, "snr": 5.0}
    result1 = explainer.explain(query=query1, context=context1)

    print(f"Context: {context1}")
    print(f"Recommendations: {len(result1['bayes_recommendations'])}")
    for rec in result1['bayes_recommendations']:
        print(f"  - {rec['adjustment']}")

    # Test case 2: Low SNR
    print("\n[Test Case 2: Low SNR Dataset]")
    query2 = "Which detector should I use?"
    context2 = {"imbalance": 0.03, "snr": 2.5}
    result2 = explainer.explain(query=query2, context=context2)

    print(f"Context: {context2}")
    print(f"Recommendations: {len(result2['bayes_recommendations'])}")
    for rec in result2['bayes_recommendations']:
        print(f"  - {rec['adjustment']}")

    # Test case 3: Combined conditions
    print("\n[Test Case 3: High Imbalance + Low SNR]")
    context3 = {"imbalance": 0.12, "snr": 2.8}
    result3 = explainer.explain(query="What precautions should I take?", context=context3)

    print(f"Context: {context3}")
    print(f"Recommendations: {len(result3['bayes_recommendations'])}")
    for rec in result3['bayes_recommendations']:
        print(f"  - {rec['adjustment']}")

    return result1, result2, result3


def test_document_sources():
    """Test 4: Verify document sources are loaded correctly."""
    print("\n" + "=" * 80)
    print("Test 4: Document Source Verification")
    print("=" * 80)

    explainer = RAGExplainer(
        config_path="experiments/llm_config.yaml",
        llm_provider=None
    )

    print(f"\n[Total chunks indexed]: {len(explainer.retriever.chunks)}")
    print(f"\n[Document sources]:")

    sources = set(chunk['source'] for chunk in explainer.retriever.chunks)
    for source in sorted(sources):
        count = sum(1 for chunk in explainer.retriever.chunks if chunk['source'] == source)
        print(f"  - {source}: {count} chunks")

    # Test retrieval quality
    print("\n[Retrieval Quality Test]:")
    test_queries = [
        "calibration",
        "cost-sensitive threshold",
        "event detection delay",
        "local anomaly",
        "frequency features",
    ]

    for query in test_queries:
        retrieved = explainer.retriever.retrieve(query, top_k=1)
        if retrieved:
            top_chunk = retrieved[0]
            print(f"\n  Query: '{query}'")
            print(f"  Top source: {top_chunk['source']}")
            print(f"  Snippet: {top_chunk['text'][:100]}...")


def test_error_handling():
    """Test 5: Error handling and edge cases."""
    print("\n" + "=" * 80)
    print("Test 5: Error Handling")
    print("=" * 80)

    # Test 5.1: Invalid provider
    print("\n[Test 5.1: Invalid LLM Provider]")
    try:
        explainer = RAGExplainer(
            config_path="experiments/llm_config.yaml",
            llm_provider="nonexistent_provider"
        )
        print("ERROR: Should have failed with invalid provider")
    except Exception as e:
        print(f"✓ Correctly caught error: {type(e).__name__}")

    # Test 5.2: Empty query
    print("\n[Test 5.2: Empty Query]")
    explainer = RAGExplainer(
        config_path="experiments/llm_config.yaml",
        llm_provider=None
    )
    result = explainer.explain(query="", context={})
    print(f"✓ Handled empty query: {len(result['retrieved_docs'])} docs retrieved")

    # Test 5.3: Large context
    print("\n[Test 5.3: Large Context Dictionary]")
    large_context = {f"param_{i}": i for i in range(100)}
    result = explainer.explain(query="test", context=large_context)
    print(f"✓ Handled large context: {len(result['bayes_recommendations'])} recommendations")


def main():
    parser = argparse.ArgumentParser(
        description="Test RAG-Bayes Explainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/llm_config.yaml",
        help="Path to LLM config file (default: experiments/llm_config.yaml)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="LLM provider to test (e.g., openai_gpt35, local_exaone_35_78b). If not specified, tests TF-IDF only."
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "retrieval", "llm", "bayes", "sources", "errors"],
        default="all",
        help="Which test to run (default: all)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("RAG-Bayes Explainer Test Suite")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"LLM Provider: {args.llm_provider if args.llm_provider else 'None (TF-IDF only)'}")
    print(f"Test: {args.test}")

    # Run tests
    try:
        if args.test in ["all", "retrieval"]:
            test_basic_retrieval()

        if args.test in ["all", "llm"] and args.llm_provider:
            test_llm_generation(args.llm_provider)

        if args.test in ["all", "bayes"]:
            test_bayes_rules()

        if args.test in ["all", "sources"]:
            test_document_sources()

        if args.test in ["all", "errors"]:
            test_error_handling()

        print("\n" + "=" * 80)
        print("✓ All tests completed successfully!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Test failed with error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
