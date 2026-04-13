LLM-Guided Local Anomaly Detection – Onboarding & Research Handbook
==================================================================

This handbook is the single “read this first” for new contributors. It consolidates onboarding, research method, datasets, experiments, models, testing, troubleshooting, reproducibility, and contributing into one document.

- Who this is for: AI researchers and software engineers joining the project.
- What you can do after reading: run the core experiments, understand metrics and artifacts, extend detectors/features, debug issues, and contribute changes reproducibly.

---

1. Goals and Project Map
------------------------

- Research motivation
  - Blend transparent rule baselines with ML detectors for manufacturing time series to improve AUC-PR and reduce operational cost (expected cost) with proper calibration (ECE).
  - Mature the Detect stage first, then incrementally add Explain (RAG-Bayes, evidence) and Act (policies), keeping the system reproducible and dependency-light.

- Engineering goals
  - Minimal-dependency pipeline (Python 3.9+, optional matplotlib/pandas/pyarrow) with a single CLI to run experiments across synthetic and public datasets.
  - Standardized runs/ outputs: JSON/CSV/plots + per-run REPORT.md and global index.

- Repository map (key paths)
  - experiments/ (main package)
    - main_experiment.py: detect CLI and run orchestration
    - data/: dataset loaders, router, normalization
    - rule_detector.py, ml_detector.py, hybrid_detector.py, spec_cnn.py
    - metrics.py, calibration.py, cost_threshold.py, result_manager.py, feature_bank.py
  - scripts/: reporting, ablations, indices, bootstrap CI, utilities
  - Top-level docs: README.md, EVALUATION_PROTOCOL.md, RESULTS_POLICY.md, EXPERIMENT_REPORT.md, UPDATE_LOG.md, this docs/HANDBOOK.md

---

2. 10-Minute Quickstart (Read Me → Do This)
-------------------------------------------

- Environment
  - Python 3.9+
  - Optional: pip install matplotlib pandas pyarrow
  - Code root: CODE_ROOT (this repo)
  - Data root example: DATA_ROOT=/workspace/data1_arsim/LFactory_d (you provide)

- Synthetic smoke (ML with calibration + cost)
PYTHONPATH=. python -m experiments.main_experiment \
  --dataset synthetic --mode detect --detector ml \
  --length 300 --seed 123 \
  --out-json runs/quick/run.json --out-csv runs/quick/preds.csv \
  --plots-dir runs/quick/plots --calibrate isotonic --ece-bins 10 \
  --cost-optimize --costs 0,1,5,0 --apply-cost-threshold

- Real dataset smoke (SKAB rule robust)
PYTHONPATH=. python -m experiments.main_experiment \
  --dataset SKAB --mode detect --detector rule --z-robust \
  --data-root $DATA_ROOT --split test --seed 42 \
  --run-id skab_smoke --plots-dir runs/skab_smoke/plots

- Useful scripts
  - Calibration comparison: python scripts/calibration_eval.py --dataset synthetic --out runs/cal_eval.csv
  - Cost A/B summary: python scripts/cost_ab_report.py --dataset synthetic --detector ml --out runs/cost_ab.csv
  - Ablation sweep: python scripts/ablation_sweep.py --dataset synthetic --out runs/ablate.csv --detectors rule ml hybrid
  - Batch eval (per-dataset summaries): DATA_ROOT=... python scripts/batch_eval.py
  - Organize/index runs: python scripts/organize_runs.py && python scripts/enforce_policy.py --retain 20

---

3. Glossary (Short and Practical)
---------------------------------

- Point-wise metrics: Precision, Recall, F1, Accuracy computed per timestamp.
- Event metrics: contiguous positive label segments; event recall/precision/F1; detection delay (first hit minus start); lead time (peak minus first hit). Current implementation approximates peak by segment end; a domain peak definition can replace this later.
- ROC/PR AUC: computed from score order; insensitive to chosen decision threshold.
- Calibration & ECE: probability output quality; ECE is the weighted gap between predicted confidence and observed frequency across bins.
- Cost matrix (C00,C01,C10,C11): expected cost = (TN*C00 + FP*C01 + FN*C10 + TP*C11)/N; we minimize over thresholds.

---

4. Research Questions and Hypotheses
------------------------------------

This project investigates four non-trivial research questions (RQs) that together form a unified anomaly detection framework.

### 4.1 RQ1: Frequency vs Time-Domain Features

**Question**: Do frequency-domain features (DFT bands via SpecCNN) outperform time-domain features (rolling stats via Rule detector) for manufacturing spike/step/drift anomalies?

**Why non-trivial**:
- Spikes are transient (duration << window) → DFT may miss them (leakage effect)
- Step changes are discontinuous → DFT assumes periodicity (Gibbs phenomenon)
- Answer depends on anomaly type, SNR, and sampling rate

**Testable Hypotheses**:
- H1.1: SpecCNN achieves higher AUC-PR than Rule on periodic datasets (e.g., SKAB valve)
- H1.2: Rule achieves higher Event Recall on transient spikes (< 10 samples)
- H1.3: Hybrid (Rule + SpecCNN) improves ≥10% over best single detector

**Acceptance Criteria**:
- Frequency analysis completed for SKAB, AIHub (PSD, discriminative bands)
- Ablation study: Rule-only, SpecCNN-only, Hybrid with statistical significance (p < 0.05, n ≥ 5 seeds)
- Documented in `experiments/ABLATION_FREQUENCY_VS_TIME.md`

**Dependencies**: → RQ2 (ensemble method depends on which features perform better)

**Reference**: See `docs/RQ_JUSTIFICATION.md` Section 1

---

### 4.2 RQ2: Ensemble Methods

**Question**: Which ensemble method (linear, product, max, learned) achieves best calibration-cost trade-off?

**Why non-trivial**:
- Linear is simplest but may not be optimal
- Product requires consensus (conservative, may miss events)
- Max is aggressive (recall-biased, may inflate false alarms)
- Learned may overfit on small validation sets

**Testable Hypotheses**:
- H2.1: Linear achieves lowest ECE (< 0.05)
- H2.2: Max achieves lowest expected cost on high-imbalance (> 1:50) with FN >> FP
- H2.3: Learned improves ≥5% AUC-PR if validation size ≥ 500

**Acceptance Criteria**:
- All 4 methods implemented and tested on ≥3 datasets
- Multi-metric evaluation: AUC-PR, ECE, Expected Cost, Point F1, Event F1
- Pareto frontier plot: ECE (x) vs Expected Cost (y)
- Statistical tests (paired t-test, n ≥ 5 seeds)
- Documented in `experiments/ABLATION_ENSEMBLE_METHODS.md`

**Dependencies**: RQ1 → RQ2 (feature choice affects ensemble), RQ2 → RQ3 (ensemble affects metrics)

**Reference**: See `docs/RQ_DEPENDENCIES.md` Section 2.1

---

### 4.3 RQ3: Point vs Event Metrics Correlation

**Question**: What is the correlation between point-wise F1 and event-wise F1 across detectors?

**Why non-trivial**:
- Point F1 penalizes every FP timestamp → favors precision
- Event F1 only cares if event detected → favors recall
- Example: Conservative detector may have low Point F1 but high Event F1

**Testable Hypotheses**:
- H3.1: Correlation is moderate (0.5 < ρ < 0.8)
- H3.2: Correlation higher for long events (> 50 samples) than short events (< 10 samples)
- H3.3: Rule detectors have higher Event/Point F1 ratio than ML detectors

**Acceptance Criteria**:
- Full batch evaluation (≥ 50 data points for correlation)
- Scatter plot: Point F1 (x) vs Event F1 (y), color-coded by detector type
- Pearson + Spearman correlation with 95% CI (bootstrap n=1000)
- Stratified by anomaly length, documented in `docs/METRICS_TRADEOFF.md`

**Dependencies**: RQ2 → RQ3 (ensemble choice affects metric trade-off), RQ3 → RQ4 (metric choice affects cost design)

**Reference**: See `docs/RQ_JUSTIFICATION.md` Section 3

---

### 4.4 RQ4: Cost Sensitivity to Dataset Characteristics

**Question**: How should FN/FP cost ratio vary with dataset imbalance and SNR?

**Why non-trivial**:
- High imbalance + low SNR: Increasing FN cost may collapse threshold → more FPs
- Optimal ratio is a function of (imbalance, SNR, detector quality)

**Testable Hypotheses**:
- H4.1: Cost ratio increases linearly with imbalance for moderate ranges (1:10 to 1:50)
  - Formula: `cost_ratio ≈ 0.5 × imbalance`
- H4.2: Ratio saturates at ~20 for extreme imbalance (> 1:100)
- H4.3: Optimal ratio decreases as SNR decreases (more conservative at low SNR)
- H4.4: Well-calibrated detectors (ECE < 0.05) show stronger cost-threshold correlation

**Acceptance Criteria**:
- Grid search: Imbalance × Cost ratio → Expected cost heatmap
- SNR computed for each dataset (formula documented)
- Fitted model: `cost_ratio = f(imbalance, SNR)` with R² > 0.7
- Practitioner guideline table for different scenarios
- Documented in `experiments/COST_SENSITIVITY_ANALYSIS.md`

**Dependencies**: RQ3 → RQ4 (metric correlation affects cost matrix design), RQ4 → RQ1 (cost depends on anomaly type detection)

**Reference**: See `docs/RQ_DEPENDENCIES.md` Section 3

---

### 4.5 Acceptance Thresholds (Overall)

**For publication-quality results**:
- ≥2 non-trivial RQs answered rigorously (statistical significance, p < 0.05)
- All comparisons include 95% confidence intervals (n ≥ 5 seeds)
- Honest limitations documented (no overclaims)
- Baseline comparisons show method's strengths AND weaknesses

**Standard Experiment Battery**:
- Single run → ROC/PR/ECE + REPORT.md
- Calibration comparison → ECE reduction validated
- Cost A/B → Expected cost reduction quantified
- Ablations → Feature/ensemble/parameter sensitivity
- Batch eval → Cross-dataset generalization

**Reporting Format**:
- Per-run REPORT.md: metadata, detector config, metrics (point + event), calibration, cost A/B, artifact links
- Batch summaries: Aggregated CSV + statistical tests
- Cross-references: Link to detailed analysis docs (RQ_JUSTIFICATION.md, etc.)

---

5. Datasets – Cards and Validation
----------------------------------

- Common rules
  - Keep CODE_ROOT separate from DATA_ROOT.
  - Use --data-root or --datasets-cfg to locate data.
  - Normalization caches (Parquet preferred; CSV fallback) should retain ≥95% of rows.
- SKAB: ${ROOT}/SKAB/{valve1,valve2,other,anomaly-free}/*.csv; label column if present, else conservative fallback; watch delimiters and missing labels.
- SMD: ${ROOT}/SMD/{train,test,test_label}/*.txt; align test with test_label; trim/pad as needed.
- AIHub71802: ${ROOT}/manufacturing_transport_71802/{Training,Validation}/{data,label}/...; label schemes: binary (>0→1) or risk4 (0..3) stored; metrics binarize >0→1.
- Quick validation: python scripts/inspect_dataset.py --dataset SKAB --data-root $DATA_ROOT --split test; check head, label counts, meta.

---

6. Experiment Playbook
----------------------

- Single detect run: set --out-json/--out-csv/--plots-dir to capture artifacts; optionally --features-csv.
- Calibration comparison: scripts/calibration_eval.py; compare ECE/AUC; include reliability bins.
- Cost A/B study: scripts/cost_ab_report.py; enable --apply-cost-threshold to see A/B metrics.
- Ablations: scripts/ablation_sweep.py; vary alpha/window/k/quantile; start small.
- Batch & housekeeping: scripts/batch_eval.py; scripts/organize_runs.py; scripts/enforce_policy.py.

---

7. Models & Extensibility
-------------------------

- Built-in detectors: rule (rolling z), robust (median/MAD), ML kNN value-density, hybrid (normalized blend), SpecCNN-lite (frequency heuristic).
- Add a detector: return dict with keys scores/preds/params; add CLI flags; ensure compatibility with calibration/cost/report.
- FeatureBank: add functions returning flat dict; saved as single-row CSV.
- Coding patterns: keep deps optional; record params in outputs.

---

8. Phase 2 – LLM-Guided Explanation (Prototype)
------------------------------------------------

Phase 2 adds RAG (Retrieval-Augmented Generation) and Bayes prior adjustment for explaining anomaly detection results. This is a prototype demonstrating LLM integration; not required for core detection (Phase 1).

### 8.1 Quick Start

- Test without LLM (TF-IDF retrieval only):
python scripts/test_explain.py

- Test with OpenAI GPT-3.5:
python scripts/test_explain.py --llm-provider openai_gpt35

- Test with local EXAONE model:
python scripts/test_explain.py --llm-provider local_exaone_35_78b

- Run experiment with explanation:
PYTHONPATH=. python -m experiments.main_experiment \
  --dataset SKAB --mode detect --detector hybrid \
  --data-root $DATA_ROOT --split test --seed 42 \
  --run-id skab_explain --explain --llm-provider local_exaone_35_78b

### 8.2 LLM Provider Selection

| Provider | Type | Use Case | Dependencies | Speed |
|----------|------|----------|--------------|-------|
| None | TF-IDF only | Minimal setup, no LLM | Standard library | Instant |
| openai_gpt35 | OpenAI API | Quick prototyping | openai | ~5-10s |
| openai_gpt4o | OpenAI API | High quality | openai | ~10-20s |
| local_exaone_35_78b | Local LLM | On-premise, no API cost | torch, transformers | ~20-40s |
| local_exaone_35_32b | Local LLM | Faster local | torch, transformers | ~10-20s |
| local_exaone_30_78b | Local LLM | Stable baseline | torch, transformers | ~20-40s |

Configuration in `experiments/llm_config.yaml`:
- API keys for OpenAI providers
- Model paths for local providers (EXAONE models on cuda:3)
- Document sources: README.md, HANDBOOK.md, EVALUATION_PROTOCOL.md
- Retrieval settings: top_k=3, chunk_size=500

### 8.3 Generated Explanations

When --explain is enabled, main_experiment.py generates EXPLANATIONS.md with:

1. **Query-based explanations**: Answers to questions like:
   - "Why should I use calibration in anomaly detection?"
   - "What is the difference between point-wise F1 and event-wise F1?"
   - "How should I set the cost matrix for this dataset?"

2. **Retrieved evidence**: Top-3 most relevant document chunks with citations

3. **Bayes prior recommendations**: Hard-coded rules based on dataset characteristics:
   - High imbalance (>0.1) → Increase FN cost relative to FP cost
   - Low SNR (<3.0) → Use conservative threshold, prefer Rule or Hybrid
   - High imbalance + Low SNR → Strong calibration recommended

4. **LLM synthesis** (if provider specified): Natural language explanation combining retrieved docs and Bayes rules

Example output structure:
```markdown
# LLM-Guided Explanations

## 1. Calibration
**Query**: Why should I use calibration in anomaly detection?
**Retrieved Evidence**:
- [EVALUATION_PROTOCOL.md] "Calibration ensures predicted probabilities match actual frequencies..."
**Bayes Recommendations**:
- Your dataset has imbalance=0.12 → Calibration critical for cost-sensitive threshold
**LLM Response**: [If available] "Calibration is essential because..."
```

### 8.4 Testing Phase 2

The test suite `scripts/test_explain.py` includes 5 test cases:
1. **Basic retrieval**: TF-IDF without LLM
2. **LLM generation**: Full RAG with specified provider
3. **Bayes rules**: Prior adjustment logic for different contexts
4. **Document sources**: Verification of loaded documents
5. **Error handling**: Edge cases (invalid provider, empty query, large context)

Run all tests:
python scripts/test_explain.py --test all

Run specific test:
python scripts/test_explain.py --test bayes

### 8.5 Known Limitations (Phase 2)

- **Bayes rules are hard-coded**: Not learned from data; fixed thresholds for imbalance (0.1) and SNR (3.0)
- **No hypothesis verification**: Rules not yet validated against RQ4 experimental results
- **Single retrieval strategy**: TF-IDF only; no semantic embeddings or reranking
- **Local LLM inference time**: 20-40s per query on cuda:3; consider batching for production
- **No feedback loop**: Explanations don't update detector parameters (Act stage not implemented)

Improvements planned:
- Week 3: Validate Bayes rules against RQ4 cost sensitivity results
- Week 4: Add semantic retrieval option (sentence-transformers)
- Week 5-6: Implement Act stage (policy recommendations)

### 8.6 Phase 2 Dependencies

Optional (install only if using Phase 2):
- OpenAI API: `pip install openai` (requires API key)
- Local LLM: `pip install torch transformers` (requires GPU, ~15GB VRAM for 7.8B model)

Core detection (Phase 1) has zero external dependencies beyond Python 3.9+ standard library.

---

9. Known Issues and Limitations (Phase 1)
------------------------------------------

This section documents current implementation limitations and workarounds. All issues are tracked in TODO.md with fix priorities.

### 9.1 ML Detector: Ignores Temporal Structure (CRITICAL)

**Issue**: `experiments/ml_detector.py` lines 50-51 sorts training data by value only:
```python
X_sorted = X[np.argsort(X)]  # Sort by value, loses temporal order
```

**Impact**:
- Cannot detect temporal patterns (step changes, drifts over time)
- May misclassify sudden shifts as anomalies even if they're part of normal temporal evolution
- Reduces effectiveness on datasets with non-stationary processes

**Quantified Impact** (will be measured in upcoming experiments):
- Expected: 10-20% reduction in Event F1 for step/drift anomalies vs temporal-aware baseline
- [TODO] Measure on SKAB valve datasets after implementing fix

**Workaround**: Use Rule or Hybrid detector for step/drift anomalies (Rule detector uses rolling windows, preserves temporal context)

**Fix Priority**: CRITICAL (Week 2)
- Planned fix: Implement sliding-window kNN or Matrix Profile-based distance
- Reference: TODO.md Part 2.1

### 9.2 SpecCNN: Score Discriminability Failure (CRITICAL)

**Issue**: **ALL anomaly scores are 0.0** (zero variance) on SKAB and SMD datasets.

**Root Cause** (`experiments/spec_cnn.py:43-66`):
- Negative low-frequency weight (`w_low = -0.2`) causes scores to become negative on low-frequency signals
- `max(0, score)` clipping forces all negative scores to 0
- Result: All scores = 0.0, AUC-PR = 0.0

**Measured Impact** (Phase 1 experiments, 2025-10-01):
- **SKAB**: Point F1=0.518 ✅, Event F1=1.0 ✅, AUC-PR=0.0 ❌
- **SMD**: Point F1=0.173 ✅, Event F1=1.0 ✅, AUC-PR=0.0 ❌
- Binary predictions work, but **NO score distribution** for ranking or cost-sensitive learning

**Research Impact**:
- ❌ **RQ1** (frequency features): Cannot validate - scores don't discriminate
- ❌ **RQ4** (cost-sensitive): Cannot validate on SpecCNN - needs score distribution
- ⚠️ **RQ2** (ensemble): Partial - binary predictions work, scores don't

**Workaround**:
- Use binary predictions for Event F1 evaluation (still valid)
- Use alternative frequency-domain detectors (Wavelet, Spectral Residual) for RQ1 validation

**Fix Options**:
- **Option A** (2-3 weeks): Implement adaptive frequency band selection, remove clipping
- **Option B** (1-2 days, recommended): Accept limitation, document in paper, focus on cost optimization (62.81% reduction) as primary contribution

**Detailed Analysis**: See `runs/status_report_2025-10-01/SPECCNN_DISCRIMINABILITY_ANALYSIS.md`

**Fix Priority**: CRITICAL - Decision needed (Week 1)
- Reference: TODO.md Section 1.5.1 (SpecCNN Decision)
- Alternative detectors: TODO.md Section 1.5.2 (Wavelet, Spectral Residual, IForest)

### 9.3 Single Seed: No Statistical Confidence (REQUIRED)

**Issue**: Most experiments use single seed (42 or 123); no confidence intervals reported

**Impact**:
- Cannot claim statistical significance for comparisons (e.g., "Hybrid beats Rule by 5% F1")
- Results may be due to random variation, not method quality
- Violates publication-quality standards (need p < 0.05, n ≥ 5)

**Quantified Impact**:
- Bootstrap CI estimation suggests ±3-5% F1 variation across seeds for current methods
- [TODO] Validate with multi-seed experiments

**Workaround**: Current results are directional; avoid strong claims like "X is better than Y" without CI

**Fix Priority**: REQUIRED (Week 2)
- Planned fix: Run all RQ experiments with n=5 seeds (42, 123, 456, 789, 2024)
- Compute mean ± 95% CI for all metrics
- Add paired t-test for detector comparisons
- Reference: TODO.md Part 3.3

### 9.4 Event Peak Definition (MINOR)

**Issue**: `experiments/metrics.py` lines 95-96 approximates event peak by segment end:
```python
# TODO: Allow domain-specific peak definition
peak_t = event_end  # Approximation: use end of segment
```

**Impact**:
- Lead time calculation may be inaccurate if true critical point occurs mid-segment
- For manufacturing, true peak often corresponds to max deviation or threshold violation

**Quantified Impact**: Low (affects Lead Time metric only; Detection Delay and F1 unaffected)

**Workaround**: Current approximation acceptable for initial experiments; Lead Time still directionally correct

**Fix Priority**: MINOR (Week 4)
- Planned fix: Add --peak-definition option (choices: end, max, median, domain-specific)
- Allow users to provide peak timestamps via optional column
- Reference: TODO.md Part 4.3

### 9.5 Dataset-Specific Issues

**SKAB**:
- Some files missing explicit label column → falls back to conservative heuristic (may undercount anomalies)
- Delimiter inconsistency (some CSV use ';' instead of ',') → handled by loader with fallback

**SMD**:
- Test/label length mismatches → trimmed or zero-padded by loader (documented in EVALUATION_PROTOCOL.md Section 7)
- High dimensionality (38 channels) → current detectors univariate only

**AIHub71802**:
- Multi-class labels (risk levels 0-3) → binarized to >0 for consistency with other datasets
- Large file sizes → normalization cache required (Parquet preferred)

**Workaround**: All dataset loaders include fallback logic and warnings; verify with `scripts/inspect_dataset.py`

### 9.6 Dependency Management

**Current state**: Phase 1 has zero external dependencies (matplotlib/pandas optional for plots/CSV)

**Limitation**: Restricts advanced methods requiring scipy, sklearn, etc.

**Tradeoff**: Reproducibility and ease of deployment vs method sophistication

**Fix Priority**: LOW (Week 5-6)
- Consider adding optional dependencies for advanced methods (e.g., Matrix Profile, Isolation Forest baselines)
- Keep core detectors dependency-free
- Reference: TODO.md Part 5.2

---

10. Testing & Troubleshooting
----------------------------

- Add tests: calibration ECE decrease; cost threshold sanity; metrics segments & delay/lead med/mean; loader alignment/delimiters/retention.
- Existing smoke: PYTHONPATH=. python scripts/test_rule_detector.py
- Golden files: ROC/PR CSV and REPORT.md snippets for stable seeds.
- Troubleshooting: data paths; label length mismatches; matplotlib missing → CSV only; normalization warnings.

---

11. Reproducibility
-------------------

- Environment: Python 3.9+; optional matplotlib/pandas/pyarrow; PYTHONPATH=. during dev.
- Run metadata: run.json records run_id/seed/git_sha/start_ts and metrics; config snapshot saved if provided.
- Verification: optimal cost ≤ fixed; ECE with calibration ≤ without; AUC-PR stable under reseed; bootstrap CI via scripts/ci_bootstrap.py.

---

12. Contributing Guidelines
---------------------------

- Style: small, pure functions; clear params; record params in outputs.
- PRs: small, reviewed, docs/tests updated; reference hypotheses or acceptance criteria.
- Branch/commits: descriptive branches; imperative, concise commit messages with rationale when non-obvious.

---

This single handbook is the onboarding guide, research method reference, and engineering playbook. Propose edits here to keep it the source of truth.

