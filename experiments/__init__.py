"""
Experiments package for LLM-Guided Local Anomaly Detection.

This package implements a two-phase pipeline for manufacturing time-series anomaly detection
with transparent rule baselines, ML detectors, proper calibration, and cost-sensitive thresholds.

## Project Structure

### Phase 1: Detect (Complete)
Core anomaly detection with minimal dependencies (Python 3.9+ standard library only).

**Detectors** (4 methods):
- Rule: Rolling z-score with robust (median/MAD) option
- ML: k-NN value-density detector
- Hybrid: Normalized ensemble of Rule + ML
- SpecCNN: Frequency-domain heuristic (STFT bands)

**Calibration** (3 methods):
- Platt scaling (logistic regression)
- Isotonic regression (non-parametric)
- Temperature scaling (single parameter)

**Metrics**:
- Point-wise: Precision, Recall, F1, Accuracy, AUC-ROC, AUC-PR
- Event-wise: Detection Delay, Lead Time, Event F1
- Calibration: Expected Calibration Error (ECE)
- Cost-sensitive: Expected cost with optimal threshold search

**Datasets**:
- Synthetic: Configurable spike/step/drift anomalies
- SKAB: Benchmark valve monitoring (real manufacturing)
- SMD: Server Machine Dataset (multi-channel)
- AIHub71802: Korean manufacturing transport dataset

### Phase 2: Explain + Act (Prototype)
LLM-guided explanations for detection results (optional, requires external dependencies).

**RAG (Retrieval-Augmented Generation)**:
- TF-IDF document retrieval from project documentation
- Multi-provider LLM support: OpenAI API + Local EXAONE models
- Query-based explanations for calibration, cost matrix, detector selection

**Bayes Prior Adjustment**:
- Hard-coded rules based on dataset characteristics (imbalance, SNR)
- Cost ratio recommendations for high imbalance / low SNR scenarios

**Status**: Prototype implementation demonstrating LLM integration.
Not required for core detection. See `explain_rag.py` and `scripts/test_explain.py`.

## Key Modules

### Core Detection
- `main_experiment.py`: Main CLI entry point for all experiments
- `data/`: Dataset loaders, router, normalization with cache
- `rule_detector.py`, `ml_detector.py`, `hybrid_detector.py`, `spec_cnn.py`: Detector implementations
- `metrics.py`: Point/event metrics, segments extraction, delay/lead time
- `calibration.py`: Platt, Isotonic, Temperature scaling
- `cost_threshold.py`: Expected cost minimization, optimal threshold search
- `result_manager.py`: JSON/CSV output, REPORT.md generation, artifact management
- `feature_bank.py`: Extensible feature extraction for detectors

### Phase 2 (Optional)
- `explain_rag.py`: RAG explainer with TF-IDF retrieval and multi-provider LLM
- `llm_config.yaml`: Configuration for 7 LLM providers (OpenAI + EXAONE variants)

### Scripts (in ../scripts/)
- Calibration comparison: `calibration_eval.py`
- Cost A/B testing: `cost_ab_report.py`
- Ablation sweeps: `ablation_sweep.py`
- Batch evaluation: `batch_eval.py`
- Run management: `organize_runs.py`, `enforce_policy.py`
- Testing: `test_rule_detector.py`, `test_explain.py`

## Quick Start

### Phase 1 (Detection)
```bash
# Synthetic smoke test with ML detector + calibration + cost optimization
PYTHONPATH=. python -m experiments.main_experiment \\
  --dataset synthetic --mode detect --detector ml \\
  --length 300 --seed 123 \\
  --out-json runs/quick/run.json --out-csv runs/quick/preds.csv \\
  --plots-dir runs/quick/plots --calibrate isotonic --ece-bins 10 \\
  --cost-optimize --costs 0,1,5,0 --apply-cost-threshold

# Real dataset (SKAB) with Rule detector
PYTHONPATH=. python -m experiments.main_experiment \\
  --dataset SKAB --mode detect --detector rule --z-robust \\
  --data-root $DATA_ROOT --split test --seed 42 \\
  --run-id skab_smoke --plots-dir runs/skab_smoke/plots
```

### Phase 2 (Explanation)
```bash
# Test RAG explainer without LLM (TF-IDF only)
python scripts/test_explain.py

# Test with local EXAONE model
python scripts/test_explain.py --llm-provider local_exaone_35_78b

# Run experiment with explanation
PYTHONPATH=. python -m experiments.main_experiment \\
  --dataset SKAB --mode detect --detector hybrid \\
  --data-root $DATA_ROOT --split test --seed 42 \\
  --run-id skab_explain --explain --llm-provider local_exaone_35_78b
```

## Research Questions

This package is designed to answer 4 interconnected non-trivial research questions:

1. **RQ1: Frequency vs Time Features**: Do frequency-domain features (SpecCNN)
   outperform time-domain features (Rule) for manufacturing anomalies?

2. **RQ2: Ensemble Methods**: Which ensemble method (linear, product, max, learned)
   achieves best calibration-cost trade-off?

3. **RQ3: Point vs Event Metrics**: What is the correlation between point-wise F1
   and event-wise F1 across detectors?

4. **RQ4: Cost Sensitivity**: How should FN/FP cost ratio vary with dataset
   imbalance and SNR?

See `docs/HANDBOOK.md` Section 4 for detailed hypotheses and acceptance criteria.

## Dependencies

**Phase 1 (Core Detection)**: Zero external dependencies beyond Python 3.9+ standard library
- Optional: `matplotlib` (plots), `pandas` (CSV), `pyarrow` (Parquet cache)

**Phase 2 (Explanation)**: Optional, only if using LLM features
- OpenAI API: `pip install openai` (requires API key)
- Local LLM: `pip install torch transformers` (requires GPU, ~15GB VRAM)

## Documentation

- **[docs/HANDBOOK.md](../docs/HANDBOOK.md)**: Comprehensive onboarding & research guide (START HERE)
- **[README.md](../README.md)**: Project overview and quick start
- **[EVALUATION_PROTOCOL.md](../EVALUATION_PROTOCOL.md)**: Metrics definitions and evaluation standards
- **[TODO.md](../TODO.md)**: Research improvement plan and task tracking
- **[docs/LOCAL_DEFINITION.md](../docs/LOCAL_DEFINITION.md)**: Three dimensions of "Local" anomaly detection
- **[docs/RQ_JUSTIFICATION.md](../docs/RQ_JUSTIFICATION.md)**: Empirical evidence for research questions
- **[docs/RQ_DEPENDENCIES.md](../docs/RQ_DEPENDENCIES.md)**: Research question interdependencies
- **[docs/RELATED_WORK.md](../docs/RELATED_WORK.md)**: Literature survey (29 papers)

## Contributing

See `docs/HANDBOOK.md` Section 12 for contribution guidelines.
All changes should reference research hypotheses or acceptance criteria from Section 4.
"""

