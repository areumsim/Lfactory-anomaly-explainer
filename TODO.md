# LFactory Research Improvement Plan (연구 개선 계획)

**Version**: 2.0 (Complete Revision - Research-Driven)
**Last Updated**: 2025-10-01
**Owner**: Research Planning Lead

---

## 📌 Executive Summary

This TODO supersedes the previous version with a **research-first approach**. The project currently faces critical theoretical gaps:
1. **Identity crisis**: "LLM-Guided" with no LLM implementation
2. **Methodological flaws**: ML detector ignores temporal structure; SpecCNN lacks theoretical grounding
3. **Trivial hypotheses**: All current hypotheses are self-evident
4. **Missing baselines**: No comparison with standard methods (IsolationForest, LSTM-AE)
5. **Undefined terms**: "Local" anomaly detection undefined; event metric "peak" approximated incorrectly

This plan prioritizes **theoretical validity**, **rigorous experiments**, and **honest documentation** over premature claims.

---

## 🔬 Part 1: Research Identity & Theory (CRITICAL - Week 1-2)

**STATUS (2025-10-01)**: Week 1 tasks COMPLETED ✅
- Option B selected: LLM prototype implemented
- 10/13 tasks completed (see details below)
- Remaining: ML detector fix, SpecCNN analysis, multi-seed experiments (Week 2)

### 1.1 Project Scope Clarification [COMPLETED ✅]

#### Problem
- **Current claim**: "LLM-Guided Local Anomaly Detection"
- **Reality**: Zero LLM implementation; Detect-only pipeline
- **Impact**: Misleading title; overselling capabilities

#### Tasks
- [x] **Decide on project scope** ✅ COMPLETED
  - **Decision**: Option B selected - Implemented minimal LLM prototype
  - **Deliverables**:
    - `experiments/llm_config.yaml`: 7 LLM providers (OpenAI API + 6 local EXAONE variants)
    - `experiments/explain_rag.py`: RAG-Bayes explainer (455 lines)
    - `scripts/test_explain.py`: Comprehensive test suite (5 test cases)
    - Updated `main_experiment.py` with --explain, --llm-provider, --llm-config flags
  - **Owner**: Research Lead
  - **Completed**: 2025-10-01

- [x] **Remove or justify "LLM-Guided" in all documents** ✅ COMPLETED
  - **Approach**: Justified with Phase 2 prototype implementation
  - **Files updated**:
    - README.md: Added Phase 1/2 distinction, documentation map
    - HANDBOOK.md: Added Section 8 (Phase 2 usage), Section 9 (Known Issues)
    - HANDBOOK_ko.md: Synchronized with English version
    - experiments/__init__.py: Comprehensive docstring with Phase 1/2 info
    - main_experiment.py: Added explanation generation logic (lines 508-594)

#### Success Criteria
- [x] No misleading claims in any document ✅
- [x] Clear phase separation ✅ (Phase 1: Detect, Phase 2: Explain + Act)
- [x] Functional LLM prototype ✅ (RAG with TF-IDF + multi-provider LLM support)

---

### 1.2 Define "Local" or Remove It [COMPLETED ✅]

#### Problem
- **Current**: "Local Anomaly Detection" used throughout, but "Local" is **never defined**
- **Ambiguity**: Time-local? Space-local? Context-local?
- **Impact**: Non-reproducible research; reviewers will reject

#### Tasks
- [x] **Literature review on "local" in anomaly detection** ✅ COMPLETED
  - **Deliverable**: `docs/LOCAL_DEFINITION.md` (256 lines, comprehensive 3-dimensional definition)
  - **Content**:
    - Dimension 1: Data/Process-Specific Adaptation (primary meaning)
    - Dimension 2: On-Premise Deployment Capability (secondary meaning)
    - Dimension 3: Context-Local Detection (technical meaning)
  - **Completed**: 2025-10-01

- [x] **Choose definition or remove term** ✅ COMPLETED
  - **Decision**: All three dimensions are relevant and interconnected
  - **Deliverable**: EVALUATION_PROTOCOL.md Section 0 "용어 정의 (Terminology)"
  - **Content**: Added Section 0.1 explaining all three dimensions with:
    - Definitions, rationale, implementation details
    - Research implications (connects to RQ4)
    - Operational table with parameter examples
  - **Completed**: 2025-10-01

#### Success Criteria
- [x] Unambiguous definition in ≥2 documents ✅ (LOCAL_DEFINITION.md + EVALUATION_PROTOCOL.md)
- [x] Reproducible: two researchers interpret identically ✅ (Comprehensive with examples)

---

### 1.3 Reformulate Research Questions [COMPLETED ✅]

#### Problem
**Current hypotheses** (from old TODO.md):
1. "ML/Hybrid improve AUC-PR vs Rule" → **Trivial** (Rule is too simple)
2. "Calibration reduces ECE" → **Tautological** (ECE definition)
3. "Cost-optimal < fixed" → **Tautological** (optimization definition)

**Impact**: No research contribution; any competent reviewer will reject

#### Tasks
- [x] **Formulate 4 non-trivial research questions** ✅ COMPLETED
  - **Deliverables**:
    - `docs/HANDBOOK.md` Section 4 (comprehensive RQ1-4 with hypotheses, acceptance criteria)
    - `docs/HANDBOOK_ko.md` Section 4 (Korean translation)
    - `docs/RQ_JUSTIFICATION.md` (480 lines, empirical evidence for each RQ)
    - `docs/RQ_DEPENDENCIES.md` (470 lines, interdependency analysis)

  - **RQ1** (Feature comparison): "Do frequency-domain features (DFT bands) outperform time-domain features (rolling stats) for manufacturing spike/step/drift anomalies?"
    - **Hypotheses**: H1.1 (SpecCNN > Rule on periodic), H1.2 (Rule > SpecCNN on transient), H1.3 (Hybrid ≥10% improvement)
    - **Acceptance**: Frequency analysis (PSD), ablation study (p<0.05, n≥5), documented in ABLATION_FREQUENCY_VS_TIME.md
    - **Non-trivial**: Answer depends on anomaly type, SNR, sampling rate

  - **RQ2** (Hybrid method): "Which ensemble method (linear, product, max, learned) achieves best calibration-cost trade-off?"
    - **Hypotheses**: H2.1 (Linear ECE<0.05), H2.2 (Max best cost on high imbalance), H2.3 (Learned ≥5% AUC-PR gain)
    - **Acceptance**: 4 methods on ≥3 datasets, Pareto frontier plot, paired t-test, documented in ABLATION_ENSEMBLE_METHODS.md
    - **Non-trivial**: Linear is simplest but not necessarily optimal

  - **RQ3** (Metric correlation): "What is the correlation between point-wise F1 and event-wise F1 across detectors?"
    - **Hypotheses**: H3.1 (0.5<ρ<0.8), H3.2 (correlation higher for long events), H3.3 (Rule > ML in Event/Point F1 ratio)
    - **Acceptance**: ≥50 data points, scatter plot, Pearson+Spearman with 95% CI (bootstrap n=1000), documented in METRICS_TRADEOFF.md
    - **Non-trivial**: Point F1 favors precision, Event F1 favors recall - trade-off expected

  - **RQ4** (Cost sensitivity): "How should the FN/FP cost ratio vary with dataset imbalance and SNR?"
    - **Hypotheses**: H4.1 (cost_ratio ≈ 0.5×imbalance), H4.2 (saturation at ~20 for >1:100), H4.3 (ratio↓ as SNR↓), H4.4 (ECE<0.05 → stronger correlation)
    - **Acceptance**: Grid search heatmap, SNR calculation, fitted model (R²>0.7), practitioner guideline table, documented in COST_SENSITIVITY_ANALYSIS.md
    - **Non-trivial**: Practical guidance for different scenarios

  - **Completed**: 2025-10-01

#### Success Criteria
- [x] Each RQ has clear experimental protocol ✅ (Detailed acceptance criteria in HANDBOOK Section 4)
- [x] Hypotheses are testable and non-trivial ✅ (12 testable hypotheses with statistical significance requirements)
- [x] RQ interdependencies documented ✅ (RQ_DEPENDENCIES.md)
- [ ] Results can be "yes/no" or quantitative comparison
- [ ] At least 2 RQs publishable if answered rigorously

---

### 1.4 Write Theoretical Background Documents [REQUIRED]

#### 1.4.1 Related Work Survey [COMPLETED ✅]
- [x] **Create `docs/RELATED_WORK.md`** ✅ COMPLETED
  - **Deliverable**: 430 lines, 29 papers across 4 sections (initial draft)
  - **Completed**: 2025-10-01

  - **Section 1**: Time-Series Anomaly Detection (13 papers) ✅
    - Classical: ARIMA, STL decomposition, Holt-Winters
    - ML: Isolation Forest (Liu et al. 2008), LOF (Breunig et al. 2000), One-Class SVM (Schölkopf et al. 2001)
    - DL: LSTM-AE (Malhotra et al. 2016), VAE (An & Cho 2015), Transformer (Li et al. 2019)
    - Matrix Profile (Yeh et al. 2016) for motif discovery
    - Each cited with limitations (e.g., LSTM-AE: hard to interpret, needs GPU)

  - **Section 2**: Calibration (6 papers) ✅
    - Guo et al. 2017 "On Calibration of Modern Neural Networks" (foundational)
    - Platt 1999 (Platt scaling), Zadrozny & Elkan 2001 (Isotonic regression)
    - Guo et al. 2017 (Temperature scaling)
    - Nixon et al. 2019 (measuring calibration quality)
    - Gap: Calibration rarely applied to manufacturing time-series AD

  - **Section 3**: Cost-Sensitive Learning (5 papers) ✅
    - Elkan 2001 "The Foundations of Cost-Sensitive Learning"
    - Zhou & Liu 2006 (training cost-sensitive neural networks)
    - Gap: Static cost matrices, ignore detection delay

  - **Section 4**: Manufacturing-Specific (5 papers) ✅
    - SKAB dataset (Katser & Kozitsin 2020)
    - Susto et al. 2015 (manufacturing PHM)
    - Gap: Most are rule-based or single-method approaches

  - **Status**: Initial draft complete (29 papers)
  - **Expansion planned**: Week 2 expansion to 40-50 papers (TODO.md Part 1.4.1)
  - **Format**: Each paper has 2-4 sentences: method, result, limitation, relevance to our work

#### 1.4.2 Research Background & Motivation
- [ ] **Create `docs/RESEARCH_BACKGROUND.md`**
  - **Section 1**: Manufacturing Anomaly Detection Requirements
    - Low false alarm rate (production disruption cost)
    - Early detection (minimize damage cost)
    - Interpretability (operator trust)
    - **Evidence**: Industry reports or expert interviews (if available)

  - **Section 2**: Limitations of Existing Approaches
    - **Rule-only**: Low recall, manual threshold tuning
    - **ML-only**: Black-box, high false alarm, needs labeled data
    - **Uncalibrated**: Probabilities unreliable → hard to set thresholds
    - **Each claim**: Cite paper or provide empirical evidence from SKAB/SMD

  - **Section 3**: Proposed Approach (Hybrid + Calibration + Cost)
    - **Hybrid**: Combine rule transparency + ML expressiveness
    - **Calibration**: Reliable probabilities for threshold setting
    - **Cost**: Align thresholds with operational cost structure
    - **Diagram**: Flowchart of pipeline

  - **Section 4**: Expected Contributions
    - Empirical comparison of ensemble methods
    - Analysis of point vs event metrics
    - Cost matrix guidance for manufacturing

  - **Length**: 5-6 pages

#### Success Criteria
- [ ] Every claim has reference or empirical support
- [ ] Clear positioning: what's new vs existing work
- [ ] Honest about limitations (e.g., "exploratory study")

---

### 1.5 Status Report Follow-up [URGENT - Week 1-2]

#### Problem
Phase 1 experiments (2025-10-01) revealed critical issues and key findings:
1. **SpecCNN discriminability**: ALL scores are 0 (AUC-PR=0 on both SKAB and SMD)
   - Root cause: Negative low-frequency weight + max(0) clipping → all scores clipped to 0
   - Impact: RQ1 (frequency features) and RQ4 (cost-sensitive on SpecCNN) cannot be validated
2. **Cost optimization SUCCESS**: 62.81% cost reduction on SKAB (bug fixed, validated ✅)
3. **Point-Event F1 divergence**: Up to Δ=0.80 (strong validation of RQ3)
4. **Research direction**: Need to decide on SpecCNN fix vs accept limitation

**Status**: Analysis complete; decision and action needed

#### Tasks

##### 1.5.1 SpecCNN Decision [URGENT - Week 1]
- [ ] **Advisor meeting: Decide on SpecCNN approach**

  **Option A**: Fix SpecCNN (2-3 weeks)
  - Implement adaptive frequency band selection (data-driven)
  - Remove max(0, score) clipping + add normalization
  - Re-run all 8 experiments (SKAB + SMD × 4 detectors)
  - **Pros**:
    - Validates RQ1 (frequency features effectiveness)
    - Completes RQ4 (cost-sensitive learning) for all detectors
    - Stronger methodological contribution
  - **Cons**:
    - Delays Phase 2 by 2-3 weeks
    - May still not achieve high AUC-PR (fundamental limitation?)

  **Option B**: Accept limitation + Document (recommended, 1-2 days)
  - Document SpecCNN limitation in HANDBOOK.md Section 9 (Known Issues)
  - Update SPECCNN_DISCRIMINABILITY_ANALYSIS.md with future work
  - Focus on binary predictions (Event F1=1.0 still valid)
  - Pivot research narrative to: **cost optimization (63%) + Point-Event analysis (Δ=0.80)**
  - **Pros**:
    - Fast, scientifically honest approach
    - Allows Phase 2 start immediately
    - Cost optimization is strong standalone contribution
  - **Cons**:
    - Weakens RQ1 (cannot validate frequency features)
    - Blocks RQ4 validation on SpecCNN specifically

  - **Decision deadline**: End of Week 1
  - **Owner**: Research Lead + Advisor
  - **Deliverable**: Decision documented in HANDBOOK.md Section 4 update

##### 1.5.1.1 긴급 결정 체크리스트 [Week 1]

**Advisor 논의 필수**:
- [ ] **SpecCNN 문제 해결 방안 결정**
  - Option A: 수정 (2-3주 소요, 적응형 주파수 밴드 선택 구현)
  - Option B: 한계 인정 + 문서화 → Phase 2로 빠르게 진행 (권장)

- [ ] **논문 핵심 기여 방향 확정**
  - 비용 최적화 중심 (62.81% 절감)
  - Point-Event 평가 방법론 중심 (Δ=0.80)

**이번 주 작업 (Week 1)**:
- [ ] HANDBOOK.md Section 9 업데이트 (SpecCNN 한계 문서화)
- [ ] TODO.md 진행 상황 업데이트 (완료 항목 체크)
- [ ] Phase 2 준비 (RAG-Bayes 리뷰, LLM threshold 실험 설계)

##### 1.5.2 Add Alternative Frequency-Domain Detectors [Week 1-2]
To address SpecCNN limitations and strengthen RQ1, add 2-3 alternatives with better theoretical grounding:

- [ ] **Wavelet Transform Detector** (Priority 1, 2-3 days)
  - **File**: `experiments/wavelet_detector.py`
  - **Method**:
    - Discrete Wavelet Transform (DWT) using scipy.signal
    - Detail coefficients at multiple scales → statistical threshold
    - Score = max deviation across scales
  - **Rationale**:
    - Better time-frequency localization than STFT (SpecCNN)
    - Adaptive to signal characteristics
    - Well-established in signal processing literature
  - **Expected AUC-PR**: 0.3-0.5 (better than SpecCNN's 0.0)
  - **Dependencies**: scipy.signal (already installed ✅)
  - **Implementation complexity**: ★★☆☆☆ (150 lines)
  - **Owner**: Signal processing researcher
  - **Validation**: Run on SKAB + SMD, compare with SpecCNN

- [ ] **Spectral Residual (SR) Method** (Priority 2, 1-2 days)
  - **File**: `experiments/spectral_residual.py`
  - **Method**:
    - FFT → Log-amplitude spectrum → Residual = actual - smoothed
    - Anomaly = windows with high residual
    - Based on Microsoft 2019 "Time-Series Anomaly Detection Service"
  - **Rationale**:
    - Simpler than SpecCNN, theoretically grounded
    - Detects deviations from average frequency pattern
    - Zero external dependencies (numpy only)
  - **Expected AUC-PR**: 0.2-0.4
  - **Dependencies**: numpy only ✅
  - **Implementation complexity**: ★☆☆☆☆ (120 lines)
  - **Owner**: ML researcher
  - **Reference**: Ren et al. "Time-Series Anomaly Detection Service at Microsoft" (KDD 2019)

- [ ] **Isolation Forest Baseline** (from TODO Part 4.1, accelerated, 1-2 days)
  - **File**: `experiments/baseline_iforest.py`
  - **Method**:
    - Sliding window embedding: [v_{t-w}, ..., v_t]
    - Apply scikit-learn IsolationForest to embeddings
    - Score = anomaly score from ensemble
  - **Rationale**:
    - Industry-standard baseline (already planned in Part 4.1)
    - Non-parametric, handles multimodal distributions
    - Fast (O(n log n))
  - **Expected AUC-PR**: 0.4-0.6 (likely best among alternatives)
  - **Dependencies**: scikit-learn 1.4.2 (already installed ✅)
  - **Implementation complexity**: ★☆☆☆☆ (100 lines)
  - **Owner**: ML researcher
  - **Integration**: Add `--detector iforest` to main_experiment.py

##### 1.5.3 Research Narrative Pivot [Week 1-2]
- [ ] **Confirm core contribution priority**

  **Option A**: Cost optimization as primary (recommended)
  - Main claim: "63% cost reduction through adaptive thresholding"
  - Supporting: Point-Event F1 analysis (Δ=0.80)
  - Secondary: LLM integration for threshold guidance (Phase 2)
  - **Rationale**: Strongest empirical result, practical impact

  **Option B**: Point-Event evaluation methodology as primary
  - Main claim: "Point-Event F1 divergence reveals evaluation trade-offs"
  - Supporting: Cost optimization (63%)
  - **Rationale**: Fills gap in evaluation literature

  **Option C**: LLM integration as primary (requires Phase 2 completion)
  - Main claim: "LLM-guided threshold optimization"
  - Supporting: All Phase 1 results
  - **Rationale**: Most innovative, but needs Phase 2 results

  - **Decision**: Document in HANDBOOK.md Section 4 update
  - **Owner**: Research Lead

- [ ] **Update paper story**
  - **Old narrative**:
    > "LFactory: Frequency features + Ensemble + Cost optimization for manufacturing anomaly detection"

  - **New narrative (Option A)**:
    > "LFactory: Achieving **63% cost reduction** in manufacturing anomaly detection through adaptive threshold optimization. We demonstrate Point-Event F1 divergence (Δ up to 0.80) and provide practical guidance for detector selection."

  - **Files to update**:
    - README.md Section 1 (Project Overview)
    - HANDBOOK.md Section 1.1 (Goals)
    - CURRENT_STATUS_REPORT.md Section 7 (Updated Research Direction)
  - **Owner**: Documentation Lead
  - **Deadline**: End of Week 1

##### 1.5.4 Document Updates [Week 1-2]
- [ ] **HANDBOOK.md updates**
  - **Section 4.1 (RQ1)**: Update validation status
    - Note SpecCNN limitation, reference alternative detectors
    - Mark as "Partially Validated" if alternatives show success
  - **Section 4.4 (RQ4)**: Update validation status
    - Note SpecCNN cannot validate RQ4 (no score distribution)
    - Mark as "Validated on Rule, ML, Hybrid" (5/6 detectors achieve >30% reduction)
  - **Section 9 (Known Issues)**: Add SpecCNN discriminability problem
    - Reference: SPECCNN_DISCRIMINABILITY_ANALYSIS.md
    - Status: Known limitation, future work planned
  - **Section 10 (Future Work)**: Add research directions
    - Short-term: Alternative frequency detectors
    - Long-term: Adaptive band selection, learned features
  - **Owner**: Documentation Lead

- [ ] **TODO.md updates** (this file)
  - [x] Add Section 1.5 (Status Report Follow-up) ✅
  - [ ] Mark Part 1 Week 1 tasks as COMPLETED
  - [ ] Update timeline estimates in Part 8
  - [ ] Add checklist for Week 2 deliverables
  - **Owner**: Project Manager

- [ ] **Update runs/status_report_2025-10-01/COMPLETION_SUMMARY.md**
  - Add Section "Next Steps: Alternative Detectors"
  - Reference this TODO section
  - **Owner**: Documentation Lead

#### Success Criteria
- [ ] SpecCNN decision made and documented (Option A or B)
- [ ] ≥1 alternative frequency-domain detector implemented and tested
- [ ] Research narrative confirmed and updated across all documents
- [ ] All status report findings documented in HANDBOOK.md

#### Timeline
- **Week 1 End** (by 2025-10-08):
  - SpecCNN decision finalized
  - Alternative detectors started (Wavelet + IForest)
  - HANDBOOK.md updated with research narrative
- **Week 2 End** (by 2025-10-15):
  - Alternative detectors complete + tested
  - Full re-run with new detectors (if Option B chosen)
  - Updated comparison tables and visualizations

---

## 🧪 Part 2: Fix Methodological Flaws (CRITICAL - Week 2-3)

### 2.1 ML Detector: Temporal Structure Violation [REQUIRED]

#### Problem
**Current implementation** (`experiments/ml_detector.py:50-51`):
```python
pairs = sorted([(float(v), i) for i, v in enumerate(series)], key=lambda x: x[0])
# Sorts by VALUE only → time information completely lost
```

**Why this is CRITICAL**:
- This is **not time-series anomaly detection**; it's statistical outlier detection
- A spike at t=10 vs t=1000 gets same score if value is same
- **Violates fundamental assumption** of time-series analysis
- Undermines entire "manufacturing time-series specialized" claim
- **Any reviewer will reject this as invalid**

#### Tasks
- [ ] **Implement time-aware alternatives (choose one)**
  - **Option A**: Sliding window embedding
    ```python
    # Embed each point as [v_{t-w}, ..., v_t] vector
    # kNN in embedding space
    ```
    - Pro: Pure Python, interpretable
    - Con: O(n²w) complexity

  - **Option B**: STOMP-based (Matrix Profile)
    - Pro: State-of-art for TS anomaly detection
    - Con: Complex, may need `stumpy` library

  - **Option C**: Replace with Isolation Forest
    - Pro: Standard baseline, well-studied
    - Con: Needs `scikit-learn` (optional dependency OK per project policy)

  - **Owner**: ML researcher
  - **Deadline**: Week 2
  - **File**: `experiments/ml_detector.py` (refactor `knn_scores` function)

- [ ] **Validation test: temporal shuffle invariance**
  - Create test: shuffle time series → score should change significantly
  - Current detector will **fail** (score unchanged)
  - New detector must **pass**
  - **File**: `tests/test_ml_detector.py` (new)

- [ ] **Ablation study: time-aware vs time-agnostic**
  - Keep old detector as `ml_detector_valueonly.py` (baseline)
  - Compare on SKAB/SMD:
    - AUC-PR difference
    - Event recall difference
  - **Hypothesis**: Time-aware ≥ 10% AUC-PR improvement
  - **Deliverable**: `experiments/ABLATION_TIME_AWARE.md`

#### Success Criteria
- [ ] Temporal shuffle test passes
- [ ] Statistically significant improvement over value-only baseline
- [ ] Documented in EXPERIMENT_REPORT.md

---

### 2.2 SpecCNN: Theory Vacuum [REQUIRED]

#### Problem
**Current** (`experiments/spec_cnn.py:52, 61-64`):
```python
w_low, w_mid, w_high = -0.2, 0.6, 0.6  # Why these weights?
# Bands: [0, 0.1], (0.1, 0.3], (0.3, 0.5]  # Why these ranges?
```

**Issues**:
- **Arbitrary hyperparameters**: No justification for 3 bands, these ranges, these weights
- **Misleading name**: "SpecCNN" suggests CNN; this is just DFT + weighted sum
- **No domain analysis**: Manufacturing data frequency characteristics unknown
- **Untestable**: No way to know if this is good/bad without theory

#### Tasks
- [ ] **Frequency analysis of manufacturing data**
  - **Step 1**: Compute power spectral density (PSD) for SKAB/AIHub
    - Normal segments: median PSD
    - Anomaly segments: median PSD
    - Plot overlay
  - **Step 2**: Identify discriminative bands
    - Use KL-divergence or t-test per frequency bin
    - Find top-3 most discriminative ranges
  - **Step 3**: Justify band selection
    - Do [0, 0.1], [0.1, 0.3], [0.3, 0.5] align with discriminative bands?
    - If not, update bands
  - **Deliverable**: `docs/FREQUENCY_ANALYSIS.md` + plots in `docs/figures/`
  - **Owner**: Signal processing researcher
  - **Deadline**: Week 3

- [ ] **Rename detector**
  - **Old**: `SpecCNN`
  - **New**: `Freq3Band` or `SpectralHeuristic`
  - **Update**: All files, docs, CLI

- [ ] **Document limitations**
  - Add to `experiments/spec_cnn.py` docstring:
    > **Limitations**: Heuristic weights; not learnable; domain-specific; requires window ≥64.
  - Add to HANDBOOK:
    > SpecCNN-lite is an exploratory heuristic detector. Band selection is based on SKAB frequency analysis (see docs/FREQUENCY_ANALYSIS.md). For production use, consider learned feature extractors.

- [ ] **Optional: Learnable version**
  - Implement `spec_cnn_learned.py`:
    - Bands and weights as trainable parameters
    - Use validation set for tuning
  - Compare learned vs heuristic
  - **Deliverable**: `ABLATION_SPECCNN.md`

#### Success Criteria
- [ ] Frequency analysis shows clear discriminative bands
- [ ] Band selection justified by data
- [ ] Honest documentation of limitations

---

### 2.3 Hybrid Ensemble: Trivial Method [MEDIUM]

#### Problem
**Current** (`experiments/hybrid_detector.py:47`):
```python
scores.append((1.0 - alpha) * rs[i] + alpha * ms[i])
# Simple weighted average
```

**Issue**: This is the **most basic ensemble method**; minimal research contribution

#### Tasks
- [ ] **Implement 4 ensemble variants**
  - **Linear** (current): `(1-α)×R + α×M`
  - **Product**: `R^(1-α) × M^α` (geometric mean)
  - **Max**: `max(R, M)` (conservative)
  - **Learned**: Logistic regression on `[R, M]` features
    - Train on validation set
    - Test on test set
  - **File**: `experiments/hybrid_detector.py` (add methods)

- [ ] **Compare on RQ2 (from Section 1.3)**
  - **Metrics**: ECE, Expected Cost, AUC-PR, Event F1
  - **Datasets**: Synthetic, SKAB, SMD
  - **Script**: `scripts/hybrid_comparison.py`
  - **Output**: `runs/hybrid_comparison/summary.csv`

- [ ] **Select and justify default method**
  - If Linear wins: OK, but state it was tested against alternatives
  - If Learned wins: Update default in main_experiment.py
  - **Deliverable**: EXPERIMENT_REPORT.md Section "Ensemble Method Comparison"

#### Success Criteria
- [ ] All 4 methods implemented and tested
- [ ] Statistical test shows winner (if any)
- [ ] Default choice documented with rationale

---

### 2.4 Cost Matrix: Domain Analysis [IMPORTANT]

#### Problem
**Current default** (`experiments/cost_threshold.py:18`):
```python
costs = (0.0, 1.0, 5.0, 0.0)  # C_TN, C_FP, C_FN, C_TP
# FN cost = 5 × FP cost → WHY?
```

**Issue**: No manufacturing domain analysis; arbitrary multiplier

#### Tasks
- [ ] **Literature review: Manufacturing false alarm & miss costs**
  - Survey 3-5 papers on production line disruption costs
  - Survey 2-3 papers on defect escape costs
  - Extract cost ranges: C_FP ∈ [?, ?], C_FN ∈ [?, ?]
  - **Deliverable**: `docs/COST_ANALYSIS.md`

- [ ] **Cost sensitivity analysis**
  - **Grid**: FN/FP ratio ∈ {1, 3, 5, 10, 20}
  - **Datasets**: Synthetic (controlled), SKAB, SMD
  - **Vary**: Dataset imbalance, SNR
  - **Measure**: Optimal threshold, resulting F1, Event Recall
  - **Hypothesis (RQ4)**: Optimal ratio correlates with imbalance
  - **Script**: `scripts/cost_sensitivity.py`
  - **Output**: Heatmap plot + CSV

- [ ] **Detection delay cost model [OPTIONAL]**
  - Extend cost to include delay: `C_FN(delay) = C_FN_base × (1 + β × delay)`
  - Test if early detection reduces expected cost
  - **Deliverable**: `experiments/cost_threshold_delay.py`

#### Success Criteria
- [ ] Cost ranges justified by literature
- [ ] Sensitivity analysis reveals patterns
- [ ] Practitioner guidance: "For imbalance >1:50, use ratio ≥10"

---

## 📊 Part 3: Evaluation Metrics Rigor (HIGH - Week 3-4)

### 3.1 Event Metric "Peak" Definition [REQUIRED]

#### Problem
**Current** (`experiments/metrics.py:95-96, 114, 126`):
```python
# Note: current implementation approximates the event "peak" by segment_end.
leads.append(float(max(0, b0 - earliest + 1)))
```

**Why this is WRONG**:
- In manufacturing, "peak" = maximum severity (e.g., max temperature, max vibration)
- Approximating peak as **end of segment** makes lead time meaningless
- Example: Anomaly at [100, 200], detected at 120
  - Current: lead = 200 - 120 + 1 = 81 (always positive!)
  - Correct (max): If peak at 150, lead = 150 - 120 = 30

**Impact**: Invalid event metrics → invalid experimental conclusions

#### Tasks
- [ ] **Implement 4 peak definitions**
  - `peak_end` (current, for comparison)
  - `peak_max`: `argmax(values[start:end])`
  - `peak_threshold`: First index where `|value - baseline| > threshold`
  - `peak_mid`: `(start + end) // 2`
  - **File**: `experiments/metrics.py` (refactor `event_metrics_from_segments`)
  - **API**: Add parameter `peak_method='max'`

- [ ] **Compare all 4 definitions**
  - Compute lead time & detection delay for each
  - **Datasets**: Synthetic (where true peak is known), SKAB
  - **Analysis**: Which aligns best with domain intuition?
  - **Deliverable**: `experiments/PEAK_DEFINITION_STUDY.md`

- [ ] **Update evaluation protocol**
  - **File**: `EVALUATION_PROTOCOL.md` Section 4.2
  - Add subsection:
    > **Peak Definition**: We define event peak as the timestamp of maximum absolute deviation from baseline within the anomaly segment. Rationale: In manufacturing, anomaly severity is typically highest at this point. Alternative definitions (segment end, midpoint, threshold crossing) are implemented for comparison (see PEAK_DEFINITION_STUDY.md).

#### Success Criteria
- [ ] All 4 definitions implemented
- [ ] Default changed to `peak_max`
- [ ] Domain rationale documented

---

### 3.2 Point vs Event Metrics Study [IMPORTANT]

#### Problem
- Both point-wise F1 and event-wise F1 are reported
- **Unclear**: Which is primary? Can they conflict? Is there a trade-off?
- **Impact**: Cannot decide how to optimize detector

#### Tasks
- [ ] **Correlation analysis (RQ3 from Section 1.3)**
  - Run all detectors (Rule, ML, Hybrid, SpecCNN) on all datasets
  - Scatter plot: Point F1 (x-axis) vs Event F1 (y-axis)
  - Compute Pearson & Spearman correlation
  - **Hypothesis**: Weak correlation (≤0.6) → metrics capture different aspects
  - **Script**: `scripts/point_vs_event_study.py`

- [ ] **Detector bias analysis**
  - Identify detectors high on Point F1 but low on Event F1 (precise but misses events)
  - Identify detectors low on Point F1 but high on Event F1 (detects events but noisy)
  - **Deliverable**: `docs/METRICS_TRADEOFF.md`

- [ ] **Manufacturing domain perspective**
  - Interview or literature: Which metric matters more for production?
  - **Hypothesis**: Event recall > Point precision (missing an event is worse than false alarms)
  - **Document**: HANDBOOK Section "Choosing Metrics"

#### Success Criteria
- [ ] Correlation coefficient reported with 95% CI
- [ ] Trade-off curve plotted
- [ ] Domain-grounded recommendation

---

### 3.3 Statistical Significance Testing [REQUIRED]

#### Problem
- Current experiments: single seed (e.g., seed=42)
- **Weakness**: Results might be due to luck
- **Impact**: Unreproducible; reviewers will ask for statistical tests

#### Tasks
- [ ] **Multi-seed experiments**
  - **Seeds**: {42, 123, 456, 789, 2024} (n=5 minimum)
  - **Apply to**: All detector × dataset combinations
  - **Automate**: `scripts/multi_seed_eval.py`
    - Input: detector, dataset, hyperparams
    - Output: `runs/multi_seed/<detector>_<dataset>/seed_<N>/run.json`
  - **Aggregate**: `scripts/aggregate_multi_seed.py`
    - Compute mean ± std for AUC-PR, F1, ECE, etc.
    - Output: `runs/multi_seed/summary.csv`

- [ ] **Implement statistical tests**
  - **File**: `experiments/statistics.py` (new)
  - **Tests**:
    - Paired t-test (if Gaussian)
    - Wilcoxon signed-rank test (if non-Gaussian)
    - Bonferroni correction for multiple comparisons
  - **Function**: `compare_detectors(results_A, results_B, metric='auc_pr') -> (statistic, p_value, effect_size)`

- [ ] **Integrate into reports**
  - Update `result_manager.py`:
    - If multiple seeds exist, compute 95% CI
    - Add to REPORT.md: "AUC-PR: 0.78 ± 0.03 (95% CI: [0.72, 0.84])"
  - Update EXPERIMENT_REPORT.md:
    - Table with mean ± std for each detector
    - Asterisks for significant differences: "Hybrid* vs Rule (p<0.05)"

#### Success Criteria
- [ ] All main results include ≥5 seeds
- [ ] Significance tests reported (p-values)
- [ ] Effect sizes reported (Cohen's d or similar)

---

## 🧬 Part 4: Experimental Design Enhancement (HIGH - Week 4-5)

### 4.1 Add Standard Baselines [REQUIRED]

#### Problem
- **Current**: Only "Rule" baseline (too simple)
- **Missing**: Industry-standard methods (IsolationForest, LSTM-AE, etc.)
- **Impact**: Cannot claim "our method is better" without comparison

#### Tasks
- [ ] **Implement Isolation Forest baseline**
  - **Dependency**: `scikit-learn` (optional, already in project policy)
  - **File**: `experiments/baseline_iforest.py`
  - **Interface**: Same as other detectors → `detect(series, **kwargs) -> {scores, preds, params}`
  - **Hyperparameters**: `n_estimators=100`, `contamination=auto`
  - **Integrate**: Add `--detector iforest` to main_experiment.py

- [ ] **Implement LSTM-AE baseline [OPTIONAL]**
  - **Dependency**: PyTorch or TensorFlow (optional)
  - **File**: `experiments/baseline_lstmae.py`
  - **Architecture**: Simple 1-layer LSTM encoder-decoder
  - **Training**: Use train split (if available) or sliding window on test
  - **Score**: Reconstruction error
  - **Note**: If too complex, defer to future work

- [ ] **Baseline comparison experiments**
  - **Run**: All baselines (Rule, IForest, LSTM-AE) + proposed methods (ML, Hybrid, SpecCNN)
  - **Datasets**: Synthetic, SKAB, SMD
  - **Metrics**: AUC-PR, Event F1, ECE, Expected Cost
  - **Script**: `scripts/baseline_comparison.py`
  - **Output**: `runs/baseline_comparison/REPORT.md` with comparison table

#### Success Criteria
- [ ] ≥2 standard baselines implemented
- [ ] Fair comparison (same data, same evaluation)
- [ ] If proposed method loses: Honest discussion of why

---

### 4.2 Dataset Profiling [REQUIRED]

#### Problem
- Claim: "Manufacturing time-series specialized"
- **Issue**: SMD is **server monitoring**, not manufacturing!
- **Missing**: Quantitative dataset characteristics

#### Tasks
- [ ] **Create `docs/DATA_CHARACTERISTICS.md`**
  - **For each dataset** (Synthetic, SKAB, SMD, AIHub71802):
    - **Domain**: Manufacturing? IT infrastructure? Other?
    - **Sampling rate**: Hz or samples/day
    - **Length**: Total points, train/test split
    - **Label distribution**: P(anomaly), imbalance ratio
    - **Anomaly types**: Spike, step, drift, other (manual inspection or metadata)
    - **SNR estimate**: `signal_power / noise_power` (approximate)
    - **Frequency characteristics**: Dominant frequencies from FFT
  - **Table format**: Easy to compare

- [ ] **Justify or remove SMD**
  - **Option A**: Remove SMD, add another manufacturing dataset
  - **Option B**: Change claim to "Manufacturing **and infrastructure** time-series"
  - **Option C**: Analyze SMD separately, don't mix with manufacturing results
  - **Deliverable**: Updated README.md Section "Datasets"

- [ ] **Dataset-specific analysis**
  - Why does Detector X work well on SKAB but poorly on SMD?
  - Correlate detector properties with dataset characteristics
  - **Example**: "SpecCNN excels on datasets with clear frequency separation (SKAB valve), struggles on broadband noise (SMD machine-1-1)."
  - **Deliverable**: EXPERIMENT_REPORT.md Section "Dataset Sensitivity"

#### Success Criteria
- [ ] Quantitative profile for each dataset
- [ ] "Manufacturing specialized" claim revised or supported
- [ ] Dataset-detector fit analyzed

---

### 4.3 Scale Up to Full Datasets [IMPORTANT]

#### Problem
- **Current**: Smoke tests on 1 file per dataset (`--file-index 0`)
- **Missing**: Full evaluation on all files

#### Tasks
- [ ] **Batch evaluation script**
  - **Script**: `scripts/full_batch_eval.py` (extend existing batch_eval.py)
  - **Datasets**: SKAB (all scenarios), SMD (all machines), AIHub (all sensors)
  - **Detectors**: All (Rule, ML, Hybrid, SpecCNN, IForest)
  - **Output**: `runs/full_batch/<dataset>_<UTC>/`
    - Per-file results
    - Aggregate statistics (mean, std, CI)
    - Failure cases (files where AUC-PR < 0.5)

- [ ] **Failure analysis**
  - Identify files where all detectors fail
  - Inspect: Is label quality poor? Anomaly too subtle?
  - Document: `docs/FAILURE_CASES.md`
  - **Honest reporting**: "On SKAB file valve1_03, all methods achieve AUC-PR ≤ 0.3. Manual inspection reveals label ambiguity (see discussion)."

- [ ] **Update EXPERIMENT_REPORT.md**
  - Replace single-file results with full-dataset results
  - Add distribution plots (histogram of AUC-PR across files)

#### Success Criteria
- [ ] ≥80% of files evaluated
- [ ] Aggregate statistics reported
- [ ] Failure cases documented honestly

---

## 📝 Part 5: Documentation Overhaul (MEDIUM - Week 5-6)

### 5.1 Core Documents Update

- [ ] **README.md**
  - **Section 1**: Project Scope (Phase 1 vs Phase 2, remove/qualify "LLM-guided")
  - **Section 2**: "Local" definition (add or remove term)
  - **Section 3**: Research Questions (4 RQs from Section 1.3)
  - **Section 4**: Quick Start (unchanged)
  - **Section 5**: Documentation Map
    ```
    - For research background → RESEARCH_BACKGROUND.md
    - For related work → RELATED_WORK.md
    - For evaluation details → EVALUATION_PROTOCOL.md
    - For onboarding → HANDBOOK.md
    ```

- [ ] **HANDBOOK.md (English)**
  - **Section 1**: Goals and Project Map (update scope)
  - **Section 4**: Research Method and Acceptance
    - Replace trivial hypotheses with RQ1-RQ4
    - Add acceptance thresholds: "RQ1 accepted if frequency features show ≥15% AUC-PR improvement over time features on ≥2 datasets with p<0.05."
  - **Section 5**: Datasets (add characteristics table)
  - **Section 7**: Models & Extensibility (add baseline info)

- [ ] **HANDBOOK_ko.md (Korean)**
  - Sync with English version
  - Translate new RQs and acceptance criteria

---

### 5.2 Evaluation Protocol Enhancement

- [ ] **EVALUATION_PROTOCOL.md major revision**
  - **Section 1**: Terminology
    - Define "Local" (if retained)
    - Define "Point-wise" vs "Event-wise"

  - **Section 4**: Time-Series Specific Metrics
    - **4.2 Detection Delay**: (current, OK)
    - **4.3 Lead Time**: Replace with 4 peak definitions
      - Formulas for each
      - When to use which
    - **4.4 Point-Adjust**: (current, OK)

  - **Section 5**: Calibration & ECE
    - **NEW**: Why calibration matters in AD
    - How to interpret reliability diagrams
    - ECE < 0.05 guideline rationale

  - **Section 6**: Cost-Sensitive Thresholding
    - **NEW**: How to set cost matrix
    - Sensitivity analysis guidance
    - Example cost ranges from literature

  - **Section 7**: Statistical Testing
    - **NEW**: Multi-seed protocol
    - Significance tests
    - Effect size reporting

  - **Section 8**: Metric Selection Guide
    - When to use Point F1 vs Event F1
    - Manufacturing domain recommendations

  - **Length**: Expand from current 2 pages to 8-10 pages

---

### 5.3 Experiment Report Restructure

- [ ] **EXPERIMENT_REPORT.md complete rewrite**
  - **Current issues**: Focuses on smoke tests; lacks rigor

  - **New structure**:
    - **Section 1**: Baseline Comparison
      - Table: All detectors × datasets, AUC-PR mean ± std
      - Statistical tests
      - Ranking with significance markers

    - **Section 2**: Research Question Answers
      - **RQ1**: Frequency vs time features → Results + interpretation
      - **RQ2**: Ensemble methods → Winner + rationale
      - **RQ3**: Point vs event correlation → Scatter plot + discussion
      - **RQ4**: Cost sensitivity → Heatmap + guidelines

    - **Section 3**: Ablation Studies
      - Time-aware ML vs value-only
      - SpecCNN learned vs heuristic
      - Calibration method comparison
      - Hybrid combination methods

    - **Section 4**: Failure Analysis
      - Datasets/files where all methods fail
      - Hypothesis for why (label quality, anomaly subtlety)
      - Recommendations for data improvement

    - **Section 5**: Limitations & Future Work
      - **Honest**: Current ML detector still simple; need DL
      - SpecCNN is heuristic; need learned features
      - Cost matrix is static; need dynamic/delay-dependent
      - No LLM explanation yet (Phase 2)

    - **Section 6**: Reproducibility
      - All experiments with seed, git SHA
      - Statistical tests for all comparisons
      - Link to runs/ artifacts

  - **Length**: 15-20 pages

---

## 🛠️ Part 6: Engineering Improvements (MEDIUM - Week 6-7)

### 6.1 Testing [Existing TODO items, reprioritized]

- [ ] **Unit tests** (from old TODO, still valid)
  - Calibration: Platt/Isotonic/Temperature convergence
  - Cost thresholds: sanity checks
  - Metrics: segments extraction, delay/lead with edge cases
  - Data loaders: alignment, label length handling
  - **Target**: 80% code coverage
  - **File**: `tests/test_*.py`

- [ ] **Integration tests**
  - End-to-end: synthetic data → detect → calibrate → cost → report
  - Golden files: ROC/PR CSV for seed=42 should match exactly

---

### 6.2 Code Quality

- [ ] **Logging framework** (from old TODO)
  - Replace `print()` with `logging.info/debug/warning`
  - Add `--log-level` CLI argument
  - **File**: `experiments/logging_config.py`

- [ ] **Packaging** (from old TODO)
  - Create `pyproject.toml`
  - Make pip-installable: `pip install -e .`
  - Console entry point: `lfactory-detect`

- [ ] **Type hints**
  - Add to all new code
  - Gradually add to existing code
  - Run `mypy` in CI

---

### 6.3 Reproducibility

- [ ] **Environment specification**
  - Pin versions in `requirements.txt`
  - Test installation in clean venv
  - Document Python version requirement (3.9+)

- [ ] **Docker image [OPTIONAL]**
  - `Dockerfile` for reproducible environment
  - Pre-load datasets (if permissible)

---

## 🗑️ Part 7: Remove Unnecessary / Misleading Content

### 7.1 Remove or Consolidate

- [ ] **UPDATE_LOG.md**: Too verbose
  - **Action**: Create `UPDATE_LOG_SUMMARY.md` (1-2 pages)
  - Keep detailed log in `archive/UPDATE_LOG_FULL.md`

- [ ] **Duplicate content**
  - README vs HANDBOOK overlap → Clarify roles:
    - README: Quick overview, setup, navigation
    - HANDBOOK: Comprehensive onboarding, research method

---

### 7.2 Revise Overclaims

- [ ] **"LLM-Guided"** → See Section 1.1
- [ ] **"Manufacturing specialized"** → See Section 4.2
- [ ] **"SpecCNN"** → Rename to "Freq3Band" or "SpectralHeuristic"
- [ ] **"Local"** → Define or remove

---

### 7.3 Defer to Phase 2

- [ ] **RAG-Bayes, LLM, Act** (currently in TODO as Low)
  - Move to separate `TODO_PHASE2.md`
  - Clear separation: Phase 1 is Detect-only, Phase 2 is Explain+Act

---

## ✅ Success Criteria (Project-Level)

### As Research
- [ ] ≥2 non-trivial research questions answered rigorously
- [ ] All comparisons statistically validated (p-values, CIs)
- [ ] Honest about limitations (no overclaims)
- [ ] Baseline comparisons show our method's strength/weakness

### As Documentation
- [ ] New researcher can onboard in ≤3 days
- [ ] Every method has theoretical justification or limitation noted
- [ ] 100% reproducibility (same seed → same result ±1e-6)

### As Code
- [ ] ≥80% test coverage
- [ ] Pip-installable
- [ ] Logging framework complete
- [ ] Optional dependencies handled gracefully

---

## 📅 Timeline Summary

| Week  | Focus                          | Key Deliverables                              |
|-------|--------------------------------|-----------------------------------------------|
| 1-2   | Research Identity & Theory     | Scope decision, RQs, RELATED_WORK.md          |
| 2-3   | Fix Methodological Flaws       | ML detector refactor, SpecCNN justification   |
| 3-4   | Evaluation Metrics Rigor       | Peak definition, statistical tests            |
| 4-5   | Experimental Design            | Baselines, dataset profiling, full batch eval |
| 5-6   | Documentation Overhaul         | HANDBOOK, EVALUATION_PROTOCOL, EXPERIMENT_REPORT revisions |
| 6-7   | Engineering Improvements       | Tests, logging, packaging                     |

**Total**: 6-7 weeks (1 FTE researcher + 1 FTE engineer, or 2 FTE researchers doing both)

---

## 🔄 Maintenance & Updates

- **Weekly**: Update checkbox status in this TODO
- **After major experiments**: Update EXPERIMENT_REPORT.md
- **After paper submission**: Archive this TODO → `archive/TODO_v2.0.md`

---

**Last Updated**: 2025-10-01
**Version**: 2.0
**Next Review**: After Week 2 (check if RQs are answerable)
