# LFactory Update Log Summary

**Version**: 1.0
**Period**: 2025-09-11 ~ 2025-10-01 (Loop 1-17)
**Purpose**: 주요 변경사항 요약 (상세 기록은 archive/UPDATE_LOG_FULL.md 참조)

---

## Phase 1: Core Detection Pipeline (Loop 1-10)

### Loop 1-2: Foundation (2025-09-11)
- **Detect-only minimal**: Rule detector (rolling z-score), metrics (P/R/F1), synthetic data
- **Result manager**: ROC/PR curves, calibration plots, CSV export

### Loop 3-4: Data & ML (2025-09-11~12)
- **Dataset routing**: SKAB, SMD, AIHub71802 loaders
- **ML detector**: kNN-based (value-space, no temporal structure)
- **Feature bank**: Time-domain statistics

### Loop 5-6: Calibration & Cost (2025-09-12)
- **Calibration**: Platt, Isotonic (PAV), Temperature scaling
- **Cost-sensitive threshold**: Expected cost minimization
- **Robust z-score**: Rolling median/MAD option
- **Run metadata**: seed, git_sha, run_id tracking

### Loop 7-8: Hybrid & Frequency (2025-09-15)
- **Hybrid detector**: Rule+ML weighted ensemble
- **SpecCNN-lite**: Frequency-domain heuristic (3-band DFT)
- **AIHub normalization**: Parquet/CSV caching

### Loop 9-10: Metrics & Scripts (2025-09-15~16)
- **Time-series metrics**: Detection delay, lead time, event-level F1
- **Utility scripts**: batch_eval, organize_runs, enforce_policy
- **Point-adjust metric**: Event-based evaluation tolerance

---

## Phase 2: LLM Integration Prototype (Loop 11-14)

### Loop 11-13: RAG-Bayes Explainer (2025-10-01)
- **explain_rag.py**: TF-IDF + LLM explanation generation
- **llm_config.yaml**: 7 providers (OpenAI + 6 local EXAONE variants)
- **Bayes prior tuning**: Cost matrix recommendations based on imbalance/SNR
- **test_explain.py**: 5 comprehensive test cases

### Loop 14: CLI Integration (2025-10-01)
- **--explain flag**: Optional Phase 2 activation
- **--llm-provider**: OpenAI API vs local EXAONE selection
- **EXPLANATIONS.md**: LLM-generated insights output

---

## Phase 3: Research Documentation (Loop 15-17)

### Loop 15: Research Identity (2025-10-01)
- **LOCAL_DEFINITION.md**: 3-dimensional "Local" definition
- **RQ_JUSTIFICATION.md**: Empirical evidence for RQ1-4
- **RQ_DEPENDENCIES.md**: Interdependency analysis (470 lines)
- **RELATED_WORK.md**: 29 papers across 4 categories

### Loop 16: Handbooks (2025-10-01)
- **HANDBOOK.md**: Onboarding + research methodology (English)
- **HANDBOOK_ko.md**: Korean translation
- **Section 4**: RQ1-4 with acceptance criteria
- **Section 9**: Known Issues (ML temporal, SpecCNN, single seed)

### Loop 17: Phase 1 Experiments (2025-10-01)
- **8/8 experiments completed**: SKAB + SMD × 4 detectors
- **Key findings**:
  - Cost optimization: 62.81% reduction (kNN on SKAB) ✅
  - Point-Event F1 divergence: Δ=0.80 (strong validation) ✅
  - SpecCNN discriminability: AUC-PR=0 (all scores zero) ❌
- **Status report**: CURRENT_STATUS_REPORT.md (403 lines)
- **Analysis**: SPECCNN_DISCRIMINABILITY_ANALYSIS.md (root cause + 4 fix options)

---

## Key Architectural Decisions

### Detector Design
- **Rule**: Rolling robust z-score (median/MAD), transparent baseline
- **ML**: kNN value-space (KNOWN ISSUE: ignores temporal structure)
- **Hybrid**: Linear combination (α=0.5), simple but sub-optimal
- **SpecCNN**: Heuristic 3-band DFT (CRITICAL ISSUE: zero scores)

### Calibration
- **Methods**: Platt (logistic), Isotonic (PAV), Temperature scaling
- **Metric**: ECE (Expected Calibration Error)
- **Target**: ECE < 0.10 (SKAB: 0.14-0.15, SMD: 0.39-0.41)

### Cost-Sensitive Learning
- **Approach**: Threshold optimization via expected cost minimization
- **Default**: `(c_TN=0, c_FP=1, c_FN=5, c_TP=0)` (FP 5× worse than FN)
- **Result**: 50-63% cost reduction on 5/6 detector-dataset combinations

### Phase 1 vs Phase 2
- **Phase 1**: Detect-only (Rule/ML/Hybrid/SpecCNN + Calibration + Cost)
- **Phase 2**: Explain + Act (LLM-guided threshold optimization, optional)

---

## Research Questions Status (2025-10-01)

| RQ | Description | Status | Key Metric |
|----|-------------|--------|------------|
| RQ1 | Frequency features effectiveness | ❌ Cannot validate | SpecCNN AUC-PR=0 |
| RQ2 | Ensemble vs single methods | ⚠️ Mixed | Hybrid < Best Single |
| RQ3 | Point-Event F1 correlation | ✅ Validated | Δ=0.80 |
| RQ4 | Cost-sensitive thresholding | ✅ Validated | 62.81% reduction |

---

## Current Priorities (Week 1-2)

### Urgent Decisions
1. **SpecCNN fix**: Option A (2-3 weeks) vs Option B (accept limitation, 1-2 days)
2. **Core contribution**: Cost optimization vs Point-Event analysis

### Week 1 Tasks
1. ✅ HANDBOOK.md Section 9 update (SpecCNN limitation documented)
2. ✅ TODO.md update (Section 1.5.1.1 urgent checklist added)
3. ⏳ Advisor discussion (SpecCNN decision, paper narrative)
4. ⏳ Phase 2 preparation (RAG-Bayes review, LLM threshold experiments)

### Alternative Detectors (TODO Section 1.5.2)
- **Wavelet Transform** (Priority 1, 2-3 days, AUC-PR 0.3-0.5)
- **Spectral Residual** (Priority 2, 1-2 days, AUC-PR 0.2-0.4)
- **Isolation Forest** (Priority 3, 1-2 days, AUC-PR 0.4-0.6)

---

## Dependencies

### Core
- Python 3.9+ standard library only
- Optional: matplotlib (plots), pandas/pyarrow (Parquet caching)

### Phase 2 (Optional)
- openai (API providers)
- transformers + torch (local EXAONE models)

---

## File Structure Evolution

### Week 1 (Loop 1-6)
```
LFactory/
├── experiments/
│   ├── data_loader.py, rule_detector.py, ml_detector.py
│   ├── calibration.py, cost_threshold.py, metrics.py
│   └── main_experiment.py
├── README.md, TODO.md
└── runs/ (synthetic experiments)
```

### Week 2 (Loop 7-10)
```
+ experiments/hybrid_detector.py, spec_cnn.py, feature_bank.py
+ experiments/data/ (loaders, normalize.py)
+ scripts/ (batch_eval, organize_runs, enforce_policy)
+ EXPERIMENT_REPORT.md, RESULTS_POLICY.md
```

### Week 3-4 (Loop 11-17)
```
+ experiments/explain_rag.py, llm_config.yaml
+ scripts/test_explain.py
+ docs/ (HANDBOOK.md, LOCAL_DEFINITION.md, RQ_*.md, RELATED_WORK.md)
+ EVALUATION_PROTOCOL.md
+ runs/status_report_2025-10-01/ (8 experiments + analysis)
```

---

**For detailed loop-by-loop changes, see**: `archive/UPDATE_LOG_FULL.md`

**Last Updated**: 2025-10-02
**Total Loops**: 17
**Total Lines (UPDATE_LOG.md)**: ~400 lines
