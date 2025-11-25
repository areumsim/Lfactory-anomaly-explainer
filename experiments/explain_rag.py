"""RAG-Bayes Explanation Module (Phase 2 Prototype)

Research context:
- Phase 2 experimental feature for LLM-guided explanations
- Retrieves relevant documentation via TF-IDF
- Generates explanations using LLM (OpenAI or local EXAONE models)
- Suggests Bayesian prior adjustments for cost/threshold tuning

Dependencies (all optional):
- openai: For OpenAI providers (GPT-3.5, GPT-4)
- transformers, torch: For local EXAONE models
- If missing: TF-IDF retrieval works standalone

Usage:
    from experiments.explain_rag import RAGExplainer

    explainer = RAGExplainer(config_path="experiments/llm_config.yaml")
    result = explainer.explain(
        query="Why should I use calibration?",
        context={"dataset": "SKAB", "imbalance": 0.05}
    )
    print(result["explanation"])
"""
from __future__ import annotations

import math
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TFIDFRetriever:
    """Simple TF-IDF document retrieval (dependency-free)."""

    def __init__(self, documents: List[Tuple[str, str]], chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            documents: List of (source_name, content) tuples
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunks: List[Dict[str, Any]] = []
        self._build_index(documents, chunk_size, chunk_overlap)

    def _build_index(self, documents: List[Tuple[str, str]], chunk_size: int, chunk_overlap: int):
        """Split documents into chunks and build TF-IDF index."""
        for source, content in documents:
            # Split into chunks
            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk_text = content[start:end]
                self.chunks.append({
                    "source": source,
                    "text": chunk_text,
                    "start": start,
                    "end": end,
                })
                start += (chunk_size - chunk_overlap)

        # Compute IDF
        self.idf: Dict[str, float] = {}
        n_docs = len(self.chunks)
        term_doc_count: Counter = Counter()

        for chunk in self.chunks:
            terms = set(self._tokenize(chunk["text"]))
            for term in terms:
                term_doc_count[term] += 1

        for term, count in term_doc_count.items():
            self.idf[term] = math.log(n_docs / count)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())

    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF vector for text."""
        terms = self._tokenize(text)
        term_freq = Counter(terms)
        n_terms = len(terms)

        vec: Dict[str, float] = {}
        for term, freq in term_freq.items():
            tf = freq / n_terms if n_terms > 0 else 0.0
            idf = self.idf.get(term, 0.0)
            vec[term] = tf * idf
        return vec

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Cosine similarity between two TF-IDF vectors."""
        all_terms = set(vec1.keys()) | set(vec2.keys())
        dot = sum(vec1.get(t, 0.0) * vec2.get(t, 0.0) for t in all_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks."""
        query_vec = self._tfidf_vector(query)
        scores: List[Tuple[float, Dict[str, Any]]] = []

        for chunk in self.chunks:
            chunk_vec = self._tfidf_vector(chunk["text"])
            sim = self._cosine_similarity(query_vec, chunk_vec)
            scores.append((sim, chunk))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scores[:top_k]]


class LLMProvider:
    """Unified interface for OpenAI and local LLM providers."""

    def __init__(self, provider_config: Dict[str, Any]):
        self.config = provider_config
        self.provider_type = provider_config["type"]

        if self.provider_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI provider requires 'openai' package. Install with: pip install openai")
            self.client = self._init_openai()
        elif self.provider_type == "local":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Local provider requires 'transformers' and 'torch'. Install with: pip install transformers torch")
            self.model, self.tokenizer = self._init_local()
        else:
            raise ValueError(f"Unknown provider type: {self.provider_type}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        openai.api_key = self.config["api_key"]
        if self.config.get("organization"):
            openai.organization = self.config["organization"]
        return openai

    def _init_local(self) -> Tuple[Any, Any]:
        """Initialize local transformers model."""
        model_path = self.config["model_path"]
        device = self.config.get("device", "cuda:0")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Local model not found: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        )
        model.to(device)
        model.eval()

        return model, tokenizer

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        if self.provider_type == "openai":
            return self._generate_openai(prompt)
        elif self.provider_type == "local":
            return self._generate_local(prompt)
        else:
            return ""

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        try:
            response = self.client.ChatCompletion.create(
                model=self.config["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.get("max_tokens", 512),
                temperature=self.config.get("temperature", 0.7),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            warnings.warn(f"OpenAI generation failed: {e}")
            return "[LLM generation failed]"

    def _generate_local(self, prompt: str) -> str:
        """Generate using local model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_new_tokens", 512),
                    temperature=self.config.get("temperature", 0.7),
                    top_p=self.config.get("top_p", 0.9),
                    do_sample=True,
                )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from output
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            return generated
        except Exception as e:
            warnings.warn(f"Local LLM generation failed: {e}")
            return "[LLM generation failed]"


class RAGExplainer:
    """RAG-Bayes explainer for anomaly detection results."""

    def __init__(self, config_path: str = "experiments/llm_config.yaml", llm_provider: Optional[str] = None):
        """
        Args:
            config_path: Path to llm_config.yaml
            llm_provider: Provider name (e.g., "openai_gpt35"); if None, uses default or no LLM
        """
        self.config = self._load_config(config_path)
        self.retriever = self._init_retriever()
        self.llm: Optional[LLMProvider] = None

        if llm_provider:
            try:
                provider_config = self.config["providers"][llm_provider]
                self.llm = LLMProvider(provider_config)
                print(f"[RAGExplainer] Loaded LLM provider: {llm_provider}")
            except Exception as e:
                warnings.warn(f"Failed to load LLM provider '{llm_provider}': {e}. Using TF-IDF only.")
        else:
            print("[RAGExplainer] No LLM provider specified. Using TF-IDF retrieval only.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML config (simple parser, no PyYAML required)."""
        # For simplicity, use the existing _load_simple_yaml from main_experiment
        # Or implement basic YAML parsing here
        # For now, we'll import from main_experiment
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from main_experiment import _load_simple_yaml
        return _load_simple_yaml(config_path)

    def _init_retriever(self) -> TFIDFRetriever:
        """Initialize document retriever."""
        code_root = Path(__file__).parent.parent
        documents: List[Tuple[str, str]] = []

        for doc_path in self.config.get("document_sources", []):
            full_path = code_root / doc_path
            if full_path.exists():
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append((doc_path, content))
                except Exception as e:
                    warnings.warn(f"Failed to load document {doc_path}: {e}")

        chunk_size = self.config.get("retrieval", {}).get("chunk_size", 500)
        chunk_overlap = self.config.get("retrieval", {}).get("chunk_overlap", 100)

        return TFIDFRetriever(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _apply_bayes_rules(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Apply Bayesian prior adjustment rules based on context."""
        recommendations: List[Dict[str, str]] = []

        imbalance = context.get("imbalance", 0.0)
        snr = context.get("snr", float("inf"))
        anomaly_types = context.get("anomaly_types", [])

        for rule in self.config.get("bayes_rules", []):
            condition = rule["condition"]
            # Simple eval (unsafe in production, OK for prototype)
            try:
                if eval(condition, {"imbalance": imbalance, "snr": snr, "anomaly_types": anomaly_types}):
                    recommendations.append({
                        "adjustment": rule["adjustment"],
                        "reason": rule["reason"],
                    })
            except Exception:
                pass

        return recommendations

    def explain(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a query.

        Args:
            query: User question (e.g., "Why use calibration?")
            context: Detection context (dataset, imbalance, snr, anomaly_types, etc.)

        Returns:
            dict with keys: query, retrieved_docs, llm_response, bayes_recommendations, explanation
        """
        if context is None:
            context = {}

        # 1. TF-IDF retrieval
        top_k = self.config.get("retrieval", {}).get("top_k", 3)
        retrieved = self.retriever.retrieve(query, top_k=top_k)

        retrieved_texts = []
        citations = []
        for chunk in retrieved:
            retrieved_texts.append(chunk["text"])
            citations.append(f"{chunk['source']}:{chunk['start']}-{chunk['end']}")

        # 2. LLM generation (if available)
        llm_response = None
        if self.llm:
            prompt = self._build_prompt(query, retrieved_texts, context)
            llm_response = self.llm.generate(prompt)

        # 3. Bayes prior adjustments
        bayes_recs = self._apply_bayes_rules(context)

        # 4. Combine into explanation
        explanation = self._format_explanation(query, retrieved_texts, llm_response, bayes_recs, citations)

        return {
            "query": query,
            "retrieved_docs": retrieved_texts,
            "citations": citations,
            "llm_response": llm_response,
            "bayes_recommendations": bayes_recs,
            "explanation": explanation,
        }

    def _build_prompt(self, query: str, retrieved: List[str], context: Dict[str, Any]) -> str:
        """Build LLM prompt from query, retrieved docs, and context."""
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        docs_str = "\n\n".join(f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(retrieved))

        prompt = f"""You are an expert in anomaly detection for manufacturing time-series.

**User Question:**
{query}

**Detection Context:**
{context_str}

**Retrieved Documentation:**
{docs_str}

**Instructions:**
1. Answer the user's question based on the retrieved documentation.
2. Be concise and precise.
3. Cite specific recommendations when applicable.
4. If the documentation doesn't cover the question, state that clearly.

**Answer:**
"""
        return prompt

    def _format_explanation(
        self,
        query: str,
        retrieved: List[str],
        llm_response: Optional[str],
        bayes_recs: List[Dict[str, str]],
        citations: List[str],
    ) -> str:
        """Format final explanation."""
        lines = [
            f"# Explanation for: {query}",
            "",
            "## Retrieved Evidence",
        ]

        for i, (doc, cite) in enumerate(zip(retrieved, citations)):
            lines.append(f"### Source {i+1}: {cite}")
            lines.append(f"```\n{doc[:200]}...\n```")  # Truncate for readability
            lines.append("")

        if llm_response:
            lines.append("## LLM Analysis")
            lines.append(llm_response)
            lines.append("")

        if bayes_recs:
            lines.append("## Recommended Adjustments (Bayes Priors)")
            for rec in bayes_recs:
                lines.append(f"- **{rec['adjustment']}**")
                lines.append(f"  - Reason: {rec['reason']}")
            lines.append("")

        lines.append("## References")
        for cite in citations:
            lines.append(f"- {cite}")

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import sys

    # Test TF-IDF only
    print("=" * 60)
    print("RAG-Bayes Explainer Test (TF-IDF only)")
    print("=" * 60)

    explainer = RAGExplainer(config_path="experiments/llm_config.yaml", llm_provider=None)

    result = explainer.explain(
        query="Why should I use calibration in anomaly detection?",
        context={"dataset": "SKAB", "imbalance": 0.05, "snr": 4.2},
    )

    print("\n[Query]")
    print(result["query"])

    print("\n[Retrieved Docs]")
    for i, doc in enumerate(result["retrieved_docs"]):
        print(f"\nDoc {i+1}: {doc[:150]}...")

    print("\n[Bayes Recommendations]")
    for rec in result["bayes_recommendations"]:
        print(f"- {rec['adjustment']}: {rec['reason']}")

    print("\n[Full Explanation]")
    print(result["explanation"])

    # If LLM provider specified via CLI, test with LLM
    if len(sys.argv) > 1:
        provider = sys.argv[1]
        print("\n" + "=" * 60)
        print(f"Testing with LLM provider: {provider}")
        print("=" * 60)

        explainer_llm = RAGExplainer(config_path="experiments/llm_config.yaml", llm_provider=provider)
        result_llm = explainer_llm.explain(
            query="What is the difference between point-wise and event-wise metrics?",
            context={"dataset": "SMD"},
        )

        print("\n[LLM Response]")
        print(result_llm["llm_response"])
