# "Local" in Local Anomaly Detection

**Version**: 1.0
**Last Updated**: 2025-10-01
**Status**: Initial definition (to be expanded with literature review)

---

## Executive Summary

In this project, "Local" has **three interconnected meanings** that together define the system's scope and deployment model. This document clarifies the term to ensure reproducibility and prevent ambiguity in research communication.

---

## 1. Definition: Three Dimensions of "Local"

### 1.1 Data/Process-Specific Adaptation (Primary Meaning)

**Definition**: Anomaly detection rules, thresholds, and models are **specialized per dataset or manufacturing process**, not generic.

**Rationale**:
- Manufacturing environments are heterogeneous (e.g., valve monitoring vs. motor vibration vs. chemical sensors)
- A "global" one-size-fits-all model trained on mixed data performs poorly
- **Local adaptation** means:
  - Different cost matrices per process (FN/FP ratios vary by criticality)
  - Dataset-specific thresholds (SKAB valve ≠ SMD server metrics)
  - Process-aware feature selection (frequency features for periodic machinery, time-domain for transient faults)

**Example**:
- **Global approach** (NOT this project): Train single LSTM-AE on all manufacturing data, apply universally
- **Local approach** (THIS project): Train/tune per dataset; SKAB uses robust z-score with threshold=3.5, SMD uses kNN with k=15

**Research implication**:
- RQ4 (cost sensitivity) investigates **how** to adapt cost matrix locally per dataset characteristics (imbalance, SNR)
- Calibration parameters (Platt A/B, Isotonic bins) are fit per dataset

---

### 1.2 On-Premise Deployment Capability (Secondary Meaning)

**Definition**: The system can run on **local (on-premise) infrastructure**, not cloud-only.

**Rationale**:
- Manufacturing data often contains proprietary process parameters → privacy/IP concerns
- Some environments are air-gapped (defense, pharma clean rooms)
- Real-time constraints (millisecond latency) → edge/local deployment preferred

**Implementation**:
- **Local EXAONE models** (Phase 2): LLM runs on local GPU (`cuda:3` in our setup)
  - No external API calls for explanation generation
  - Model weights stored in local NAS (`/workspace/data2_/nas_kbn02_02/models/...`)
- **Optional OpenAI** for non-sensitive research experiments
- All core detection (Phase 1) has **zero external dependencies** beyond Python 3.9+ stdlib

**Contrast**:
- **Cloud-only**: Sends time-series to AWS/Azure for analysis (NOT this project)
- **Local/Hybrid**: Detect locally, optionally explain via local LLM or cloud API (THIS project)

---

### 1.3 Context-Local Detection (Technical Meaning)

**Definition**: Anomaly detection operates on **local time-window context** (point-wise or short segments), not global/collective patterns.

**Technical detail**:
- **Point-wise anomaly**: Unusual value at single timestamp `t`, given local neighborhood `[t-w, t+w]`
  - Example: Spike at t=1000 is anomalous relative to surrounding baseline
- **Segment-local**: Short contiguous anomalies (step change, drift over 10-50 points)

**Contrast with other anomaly types** (NOT the focus of this project):
- **Global anomaly**: Entire time-series distribution shifts (e.g., seasonal pattern change) → contextual anomaly detection
- **Collective anomaly**: Pattern is unusual only as a group (e.g., sequence "A-B-C" normal individually, abnormal together) → sequence mining

**Project scope**:
- We detect **local** spikes, steps, drifts using:
  - Rolling z-score (local window=50 by default)
  - kNN in value-space (local density)
  - Frequency bands (local STFT window=128)
- We do **not** (currently) detect global seasonal changes or collective sequential patterns

**Future extension** (out of scope for Phase 1):
- Collective anomaly detection would require sequence models (HMM, LSTM) → Phase 3 consideration

---

## 2. Relation to "LLM-Guided"

The LLM (Phase 2) provides **local adaptation guidance** via RAG:

1. **Process-specific rules**: Retrieves documentation relevant to current dataset/process
   - Query: "What cost matrix for high-imbalance datasets?"
   - Retrieved: EVALUATION_PROTOCOL.md section on cost-sensitive thresholding
   - LLM suggests: "For imbalance >0.1, increase FN cost to 10× FP"

2. **Threshold adjustment**: Recommends threshold per local context
   - If SKAB valve data has low SNR → suggest robust z-score
   - If AIHub data has drift → suggest SpecCNN (frequency features)

3. **On-premise LLM**: Local EXAONE models ensure privacy-preserving local deployment

**Not**:
- LLM does **not** perform global optimization across all datasets
- LLM does **not** train a universal model
- LLM provides **local, dataset-aware** recommendations

---

## 3. Disambiguation: What "Local" Does NOT Mean

To prevent confusion, explicitly **excluded** meanings:

### ❌ Not "Localized" (Geographic)
- **NOT**: Different deployment per factory location (Seoul vs. Ulsan)
- **IS**: Different configuration per process type (valve vs. motor)

### ❌ Not "Local Variable" (Programming)
- **NOT**: Scope of variables in code
- **IS**: Scope of model applicability

### ❌ Not "Locality-Sensitive Hashing"
- **NOT**: A specific algorithm (LSH for nearest-neighbor search)
- **IS**: A design philosophy (adapt locally)

---

## 4. Justification and Literature

### 4.1 Why "Local" Matters in Manufacturing

**Problem**: Manufacturing data heterogeneity
- Katser & Kozitsin (2020, SKAB dataset paper): "Different fault modes require different detection strategies"
- Domain expertise: Valve failure ≠ motor vibration ≠ temperature drift
- **Evidence**: Our ablation studies (RQ1) show SpecCNN excels on SKAB (periodic), fails on SMD (aperiodic)

**Solution**: Local adaptation
- Ensemble weights (RQ2): α varies per dataset
- Cost matrix (RQ4): FN/FP ratio depends on process criticality and imbalance
- Thresholds: Quantile-based, fit per dataset split

### 4.2 On-Premise Requirement

**Sources** (to be expanded with industry reports):
- Manufacturing IT survey (placeholder): 68% of plants prohibit cloud data upload for production-critical sensors
- Data residency regulations: EU GDPR, Korea PIPA → local storage mandated

**Implementation choice**:
- Local EXAONE models (7.8B–32B parameters) achievable on single GPU
- Trade-off: Smaller model vs. data privacy
- OpenAI option for non-sensitive research benchmarking only

### 4.3 Point-wise vs. Collective (Literature Review - TODO)

**Chandola et al. (2009)**: Anomaly Detection: A Survey
- Point anomaly: "A single data instance is anomalous w.r.t. the rest of the data"
- Contextual anomaly: "Anomalous in a specific context but not otherwise"
- Collective anomaly: "Collection of data instances is anomalous"

**This project**: Focuses on **point** and **contextual (local-window)** anomalies

**To be added** (Week 2):
- 3-5 papers on local vs. global anomaly detection
- Manufacturing-specific references

---

## 5. Operational Definition for Reproducibility

For experimental reporting, "local" parameters include:

| Parameter | Meaning | Example Values |
|-----------|---------|----------------|
| `--dataset` | Dataset/process name | SKAB, SMD, AIHub71802 |
| `--split` | Data split (train/test) | test |
| `--z-window` | Local time window for Rule | 50 |
| `--ml-k` | Local neighborhood size for kNN | 10 |
| `--sc-window` | Local STFT window for SpecCNN | 128 |
| `--costs` | Local cost matrix | 0,1,5,0 (per dataset) |
| `--llm-provider` | Local or cloud LLM | local_exaone_35_78b |

**Reproducibility requirement**:
- When reporting results, specify ALL local parameters
- Example: "SKAB results use local z-window=50, costs=0,1,10,0"

---

## 6. Relation to Research Questions

### RQ1 (Frequency vs. Time Features)
- **Local**: Features extracted from local windows (STFT window=128 vs. rolling window=50)
- Dataset-specific: Optimal window size varies

### RQ2 (Ensemble Methods)
- **Local**: Ensemble weights α fit per dataset
- Not a single global α for all data

### RQ3 (Point vs. Event Metrics)
- **Local**: Point-wise F1 is intrinsically local (per timestamp)
- Event F1 is segment-local (contiguous anomaly spans)

### RQ4 (Cost Sensitivity)
- **Local**: Cost matrix varies per dataset/process
- Core research question: **How** to set local costs?

---

## 7. Future Refinement

### Week 2 (Literature Review)
- [ ] Add 5-10 papers on "local anomaly detection" terminology
- [ ] Survey manufacturing AD papers for "process-specific" vs. "global" approaches
- [ ] Document industry on-premise deployment statistics

### Phase 2 (LLM Integration)
- [ ] RAG retrieval includes process-specific documentation (e.g., "SKAB valve manual")
- [ ] LLM suggests local adaptations: "For this valve type, use robust detection"

### Phase 3 (Future)
- [ ] Extend to "local explainability": Why is *this specific point* anomalous?
- [ ] Multi-site deployment: Local model per factory + federated learning (optional)

---

## 8. Summary Table

| Dimension | Definition | Key Benefit | Implementation |
|-----------|------------|-------------|----------------|
| **Data/Process-Specific** | Adapt per dataset | Better accuracy | Per-dataset hyperparams, costs |
| **On-Premise** | Run locally | Privacy, latency | Local EXAONE, zero cloud deps |
| **Context-Local** | Point/segment anomalies | Real-time detection | Rolling windows, local features |

---

## References

1. Katser, I. D., & Kozitsin, V. (2020). Skoltech Anomaly Benchmark (SKAB). *arXiv preprint*.
2. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM computing surveys*, 41(3), 1-58.
3. [To be expanded with Week 2 literature review]

---

**Appendix A: Decision Tree for "Is This Local?"**

```
Is the method/parameter dataset-specific?
├─ YES → Local (Dimension 1)
└─ NO → Check deployment
    ├─ Runs on-premise without cloud? → Local (Dimension 2)
    └─ Operates on time-window context? → Local (Dimension 3)
```

---

**Version History**:
- 1.0 (2025-10-01): Initial definition (3 dimensions)
- 1.1 (Planned Week 2): Literature review expansion
