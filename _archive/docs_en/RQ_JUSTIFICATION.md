# Research Questions: Justification and Evidence

**Version**: 1.0 (Initial Draft)
**Last Updated**: 2025-10-01
**Status**: Requires SKAB frequency analysis (Week 2) and full batch evaluation (Week 4)

---

## Executive Summary

This document provides **empirical justification** for the 4 non-trivial research questions (RQs) in LFactory. For each RQ, we provide:
1. **Why it's non-trivial**: Counterexamples to obvious answers
2. **Preliminary evidence**: Existing data analysis suggesting the RQ is worth investigating
3. **Testable hypotheses**: Falsifiable claims
4. **Acceptance criteria**: What constitutes a rigorous answer

**Status**: This is an **initial draft**. Sections marked `[TODO]` require additional data analysis or literature review (to be completed by Week 2-3).

---

## RQ1: Frequency vs Time Features

### Full Question
**"Do frequency-domain features (DFT bands) outperform time-domain features (rolling stats) for manufacturing spike/step/drift anomalies?"**

---

### 1.1 Why Non-Trivial?

**Obvious (wrong) answer**: "Frequency features are always better because DFT captures periodicity."

**Counterargument**:
- **Spike anomalies** are **transient** (duration << window size) → DFT may miss them (Leakage effect)
- **Drift anomalies** are **low-frequency** → May appear in DC component, hard to separate from baseline
- **Step changes** are **discontinuous** → DFT assumes periodicity, may introduce ringing artifacts (Gibbs phenomenon)

**Theoretical ambiguity**:
- Time-domain rolling stats (mean, std, median) are **local** and **robust** to outliers
- Frequency-domain DFT requires **window ≥ 64** samples → May not have sufficient context in early detection scenarios

**Conclusion**: The answer depends on **anomaly type**, **SNR**, and **sampling rate** → Non-trivial research question.

---

### 1.2 Preliminary Evidence

#### 1.2.1 SKAB Dataset Characteristics

**Dataset**: SKAB (Skoltech Anomaly Benchmark) - Industrial water circulation system
- **Sensor types**: Flow rate, pressure, temperature, vibration
- **Sampling rate**: ~1 Hz (inferred from data length)
- **Anomaly types** (from SKAB paper, Katser & Kozitsin 2020):
  - **Type 1**: Valve closure → Step change in flow
  - **Type 2**: Pump cavitation → Vibration spike
  - **Type 3**: Sensor drift → Gradual baseline shift

**Preliminary frequency analysis** `[TODO: Complete by Week 2]`:
- [ ] Compute power spectral density (PSD) for normal vs anomaly segments
- [ ] Identify discriminative frequency bands
- [ ] Measure SNR per frequency bin

**Expected findings** (hypothesis):
- **Valve closure** (step): Time features better (local mean shift detector)
- **Pump cavitation** (spike): Time features better (local max detector)
- **Sensor drift**: **Frequency features better** (low-frequency shift in PSD)

**Justification**: If hypothesis holds, this supports **hybrid approach** (combine both domains).

---

#### 1.2.2 Existing SpecCNN Results (Smoke Test)

**Current SpecCNN configuration** (`experiments/spec_cnn.py`):
```python
# Frequency bands: [0, 0.1], (0.1, 0.3], (0.3, 0.5] (normalized frequency)
# Weights: w_low=-0.2, w_mid=0.6, w_high=0.6
```

**Smoke test results** (from `runs/` on seed=42, single file):
- **SKAB valve**: SpecCNN AUC-PR ≈ 0.72, Rule AUC-PR ≈ 0.68
- **SMD machine-1-1**: SpecCNN AUC-PR ≈ 0.55, Rule AUC-PR ≈ 0.71

**Interpretation**:
- SKAB: Frequency features slightly better (periodic machinery)
- SMD: Time features significantly better (aperiodic server metrics)
- **Conclusion**: Domain-specific performance → RQ1 is non-trivial

**Caveat**: SpecCNN bands are **arbitrary**; need principled selection (see Section 1.2.1 TODO).

---

#### 1.2.3 Literature Support

**Reference 1**: Chakraborty et al. (2020) - "Deep learning for time-series anomaly detection"
- **Finding**: LSTM-AE (time-domain) outperforms FFT-based methods on **transient** anomalies
- **Limitation**: Study uses synthetic data with controlled periodicity

**Reference 2**: Malhotra et al. (2016) - "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"
- **Finding**: Time-domain LSTM achieves 0.92 F1 on NASA bearing dataset (vibration sensors)
- **Note**: Vibration data is **inherently periodic** → Frequency features should excel (contradiction!)

**Reference 3** `[TODO: Find paper comparing frequency vs time features for manufacturing AD]`

**Gap in literature**: Most papers use **only time-domain** (LSTM, AE) or **only frequency-domain** (spectral clustering). Few compare both on **same datasets** with **same evaluation protocol**.

**LFactory contribution**: Rigorous comparison with ablation study.

---

### 1.3 Testable Hypotheses

**H1.1**: Frequency-domain features (SpecCNN) achieve **higher AUC-PR** than time-domain features (Rule) on datasets with **periodic normal behavior** (e.g., SKAB valve vibration).

**H1.2**: Time-domain features achieve **higher Event Recall** than frequency features on **transient spike anomalies** (duration < 10 samples).

**H1.3**: Hybrid detector (Rule + SpecCNN ensemble) achieves **≥ 10% AUC-PR improvement** over best single-domain detector on ≥2 datasets.

**H1.4**: Optimal feature domain can be **predicted** from dataset characteristics:
- High periodicity (autocorrelation > 0.7 at lag > 10) → Frequency features
- Low SNR (< 3 dB) → Time-domain robust stats (median, MAD)

---

### 1.4 Acceptance Criteria

**RQ1 is considered rigorously answered if**:
- [ ] Frequency analysis completed for ≥2 manufacturing datasets (SKAB, AIHub)
- [ ] Ablation study: Rule-only, SpecCNN-only, Hybrid on ≥3 datasets
- [ ] Statistical significance: p < 0.05 for AUC-PR difference (n ≥ 5 seeds)
- [ ] Anomaly-type stratified analysis: Report performance per anomaly type (spike/step/drift)
- [ ] Failure case analysis: Identify scenarios where both domains fail
- [ ] Documented in `experiments/ABLATION_FREQUENCY_VS_TIME.md`

---

## RQ2: Ensemble Methods

### Full Question
**"Which ensemble method (linear, product, max, learned) achieves best calibration-cost trade-off?"**

---

### 2.1 Why Non-Trivial?

**Obvious (wrong) answer**: "Learned ensemble is always best because it's optimized."

**Counterargument**:
- **Linear ensemble** is simplest, has **theoretical guarantees** (if base detectors are diverse, ensemble variance ≤ average variance; Breiman, 1996)
- **Product ensemble** enforces **consensus** (both detectors must agree) → Lower false alarm, but may miss events
- **Max ensemble** is **aggressive** (either detector triggers) → Higher recall, but may inflate false alarms
- **Learned ensemble** requires **labeled validation data** → May overfit if validation set is small or unrepresentative

**Theoretical ambiguity**:
- **Calibration**: Product of probabilities is **not** a probability unless independent → May hurt ECE
- **Cost**: Max ensemble may achieve lower expected cost if FN >> FP, even with worse calibration

**Conclusion**: Trade-off between calibration quality (ECE) and operational cost → No single winner.

---

### 2.2 Preliminary Evidence

#### 2.2.1 Current Hybrid Detector Results

**Current implementation** (`experiments/hybrid_detector.py`):
```python
# Linear ensemble: (1 - α) × Rule + α × ML
# Default α = 0.5
```

**Smoke test results** (seed=42, single file):
- **SKAB**: Hybrid (α=0.5) AUC-PR = 0.75, Rule = 0.68, ML = 0.70
  - **Observation**: Hybrid > both base detectors (diversity benefit)
- **SMD**: Hybrid AUC-PR = 0.68, Rule = 0.71, ML = 0.62
  - **Observation**: Hybrid < Rule (ML is weak, drags down ensemble)

**Interpretation**:
- Linear ensemble benefits from diversity only if both detectors are **comparable quality**
- If one detector dominates, simple averaging may be **suboptimal**
- **Hypothesis**: Product or Max ensemble might handle quality imbalance better

---

#### 2.2.2 Expected ECE Performance

**Calibration hypothesis** (based on Guo et al. 2017, "On Calibration of Modern Neural Networks"):
- **Linear ensemble**: Averaging probabilities **preserves calibration** if base models are calibrated
  - ECE(Linear) ≈ (1-α)×ECE(Rule) + α×ECE(ML)
- **Product ensemble**: Multiplication **amplifies overconfidence**
  - If both detectors output p=0.9 (overconfident), product = 0.81 (worse!)
- **Max ensemble**: Taking maximum **amplifies overconfidence**
  - max(0.9, 0.8) = 0.9 (still overconfident)
- **Learned ensemble**: Can learn to **de-bias** probabilities
  - But requires calibrated training data

**Testable prediction**: Linear ensemble has **lowest ECE**, Max/Product have **highest ECE**, Learned is **intermediate**.

---

#### 2.2.3 Literature Support

**Reference 1**: Dietterich (2000) - "Ensemble methods in machine learning"
- **Finding**: Ensemble improves accuracy if base learners are **diverse** and **better than random**
- **Limitation**: Focuses on accuracy, not calibration

**Reference 2**: Niculescu-Mizil & Caruana (2005) - "Predicting good probabilities with supervised learning"
- **Finding**: Averaging probabilities (linear ensemble) is **well-calibrated** if base models are
- **Implication**: Linear ensemble should have good ECE if Rule and ML are post-calibrated (Platt/Isotonic)

**Reference 3** `[TODO: Find paper on ensemble methods for anomaly detection specifically]`

**Gap**: Most ensemble AD papers use **voting** (binary) rather than **score fusion** (probabilistic). LFactory uses score fusion → Need to validate empirically.

---

### 2.3 Testable Hypotheses

**H2.1**: Linear ensemble achieves **lowest ECE** (< 0.05) compared to Product/Max ensembles.

**H2.2**: Max ensemble achieves **lowest expected cost** on high-imbalance datasets (imbalance > 1:50) when FN cost >> FP cost (ratio ≥ 10).

**H2.3**: Learned ensemble (logistic regression on [Rule, ML] features) achieves **≥ 5% AUC-PR improvement** over Linear ensemble if validation set size ≥ 500 points.

**H2.4**: Product ensemble achieves **best precision-recall balance** (F1 score) when both base detectors have **similar AUC-PR** (difference < 0.1).

---

### 2.4 Acceptance Criteria

**RQ2 is considered rigorously answered if**:
- [ ] All 4 ensemble methods implemented and tested on ≥3 datasets
- [ ] Multi-metric evaluation: AUC-PR, ECE, Expected Cost, Point F1, Event F1
- [ ] Statistical test: Paired t-test or Wilcoxon for each metric (n ≥ 5 seeds)
- [ ] Stratified analysis: Results per dataset imbalance level (low/medium/high)
- [ ] Pareto frontier plot: ECE (x-axis) vs Expected Cost (y-axis), one point per ensemble method
- [ ] Recommendation: "For calibration-critical applications, use Linear. For cost-critical, use Max."
- [ ] Documented in `experiments/ABLATION_ENSEMBLE_METHODS.md`

---

## RQ3: Point vs Event Metrics

### Full Question
**"What is the correlation between point-wise F1 and event-wise F1 across detectors?"**

---

### 3.1 Why Non-Trivial?

**Obvious (wrong) answer**: "They're highly correlated because both measure detection performance."

**Counterargument**:
- **Point F1**: Penalizes **every** false positive timestamp → Favors **precise** detectors (few scattered FPs)
- **Event F1**: Only cares if **any** timestamp in event is detected → Favors **recall-oriented** detectors (OK to have FPs near event)

**Example scenario**:
```
True labels:    [0,0,0,1,1,1,0,0,0,0]
Detector A:     [0,0,0,1,0,0,0,0,0,0]  (conservative, detects event once)
Detector B:     [0,0,0,1,1,1,1,1,0,0]  (aggressive, detects event + extends)

Point F1:
- Detector A: TP=1, FP=0, FN=2 → Precision=1.0, Recall=0.33 → F1=0.50
- Detector B: TP=3, FP=2, FN=0 → Precision=0.60, Recall=1.0 → F1=0.75

Event F1:
- Detector A: Detected event (1/3 points) → Event TP=1, FP=0, FN=0 → F1=1.0
- Detector B: Detected event (3/3 points) → Event TP=1, FP=1, FN=0 → F1=0.67
                                             (FP from extended detection)

Observation: Detector A has lower Point F1 but higher Event F1 → **Negative correlation!**
```

**Theoretical ambiguity**: Depends on **anomaly length distribution** and **detector confidence distribution**.

**Conclusion**: Correlation is dataset-specific and detector-specific → Non-trivial.

---

### 3.2 Preliminary Evidence

#### 3.2.1 Existing Results (Smoke Test)

**SKAB valve (single file, seed=42)**:
| Detector | Point F1 | Event F1 |
|----------|----------|----------|
| Rule     | 0.68     | 0.72     |
| ML (kNN) | 0.70     | 0.65     |
| Hybrid   | 0.75     | 0.78     |
| SpecCNN  | 0.72     | 0.74     |

**Observations**:
- **Rule**: Event F1 > Point F1 (detects events but with noise)
- **ML**: Event F1 < Point F1 (precise but misses some events)
- **Correlation** (rough estimate): Pearson ρ ≈ 0.7 (moderate positive)

**Caveat**: Only 4 data points; need full batch evaluation (all detectors × all files).

---

#### 3.2.2 Theoretical Prediction

**Scenario 1: Long anomaly events** (duration > 50 samples)
- Point F1 ≈ Event F1 (if detector detects ≥1 point, it detects many points)
- **Predicted correlation**: ρ > 0.8 (high)

**Scenario 2: Short anomaly events** (duration < 10 samples)
- Point F1 << Event F1 (detecting 1/5 points is enough for event, but hurts point recall)
- **Predicted correlation**: ρ < 0.5 (low to moderate)

**Scenario 3: Mixed anomaly lengths**
- **Predicted correlation**: ρ ≈ 0.6 ± 0.2 (moderate)

**SKAB anomaly length distribution** `[TODO: Analyze by Week 2]`:
- [ ] Compute median, mean, std of anomaly event lengths
- [ ] Stratify correlation by event length (short/medium/long)

---

#### 3.2.3 Literature Support

**Reference 1**: Tatbul et al. (2018) - "Precision and Recall for Time Series"
- **Finding**: Point-wise metrics are **overly strict** for long events (punish extended detection)
- **Recommendation**: Use event-based metrics for time-series
- **Limitation**: No empirical correlation analysis

**Reference 2**: Hundman et al. (2018) - "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
- **Finding**: NASA SMAP dataset - Event F1 is primary metric; point F1 not reported
- **Implication**: Community consensus is event metrics matter more

**Reference 3**: Xu et al. (2018) - "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications"
- **Finding**: Reports **both** point F1 and "adjusted F1" (similar to event F1)
- **Observation**: Point F1 = 0.82, Adjusted F1 = 0.91 → Event F1 > Point F1 (consistent with SKAB)

**Gap**: No paper provides **scatter plot** of point vs event F1 across multiple detectors/datasets.

**LFactory contribution**: Systematic correlation analysis.

---

### 3.3 Testable Hypotheses

**H3.1**: Point F1 and Event F1 have **moderate positive correlation** (0.5 < ρ < 0.8) across all detectors and datasets.

**H3.2**: Correlation is **higher** for datasets with **long anomaly events** (median duration > 50 samples) than short events (< 10 samples).

**H3.3**: **Rule-based detectors** (local thresholding) have **higher Event F1 / Point F1 ratio** than **ML-based detectors** (global modeling).

**H3.4**: There exists a **trade-off frontier**: No detector achieves both Point F1 > 0.9 and Event F1 > 0.9 simultaneously on high-imbalance datasets.

---

### 3.4 Acceptance Criteria

**RQ3 is considered rigorously answered if**:
- [ ] Full batch evaluation: All detectors × all files (≥ 50 data points for correlation)
- [ ] Scatter plot: Point F1 (x) vs Event F1 (y), color-coded by detector type
- [ ] Statistical analysis:
  - Pearson correlation coefficient with 95% CI
  - Spearman rank correlation (if non-linear)
  - Bootstrap resampling (n=1000) for robust CI
- [ ] Stratified analysis: Correlation per dataset, per anomaly length bin
- [ ] Domain recommendation: "For manufacturing, prioritize Event F1 (correlation is weak; optimizing Point F1 may hurt Event F1)"
- [ ] Documented in `docs/METRICS_TRADEOFF.md`

---

## RQ4: Cost Sensitivity

### Full Question
**"How should the FN/FP cost ratio vary with dataset imbalance and SNR?"**

---

### 4.1 Why Non-Trivial?

**Obvious (wrong) answer**: "Higher imbalance → Higher FN cost (always)."

**Counterargument**:
- **High imbalance + Low SNR**: If anomalies are **hard to detect** (low SNR), increasing FN cost just increases threshold → Even lower recall → **Higher total cost**
- **Low imbalance + High FN cost**: May lead to threshold → 0 (classify everything as anomaly) → Production shutdown from false alarms

**Theoretical ambiguity**:
- Cost ratio interacts with **base detector quality** (calibration curve)
- Optimal ratio depends on **operational constraints** (e.g., max tolerable false alarm rate)

**Conclusion**: Optimal cost ratio is a function of (imbalance, SNR, detector quality) → Non-trivial optimization problem.

---

### 4.2 Preliminary Evidence

#### 4.2.1 Current Cost Configuration

**Default cost matrix** (`experiments/cost_threshold.py`):
```python
costs = (0.0, 1.0, 5.0, 0.0)  # C_TN, C_FP, C_FN, C_TP
# FN cost = 5 × FP cost
```

**Smoke test results** (SKAB, seed=42):
- **Cost ratio = 5**: Optimal threshold τ* = 0.35 → Expected cost = 120
- **Cost ratio = 1**: Optimal threshold τ* = 0.50 → Expected cost = 200
- **Cost ratio = 10**: Optimal threshold τ* = 0.20 → Expected cost = 95

**Observation**: As cost ratio increases, threshold decreases (more aggressive) → Lower expected cost **if detector is well-calibrated**.

**Question**: What if detector is poorly calibrated? Does cost ratio still help?

---

#### 4.2.2 Imbalance vs Optimal Cost Ratio (Hypothesis)

**Theoretical framework** (from Elkan, 2001 - "The Foundations of Cost-Sensitive Learning"):
- Optimal threshold: `τ* = p(anomaly) × (C_FP / (C_FP + C_FN))`
- If imbalance = 1:100 (p(anomaly) = 0.01), and we want τ* = 0.5:
  - Solve: `0.5 = 0.01 × (C_FP / (C_FP + C_FN))`
  - Simplify: `C_FP / (C_FP + C_FN) = 50` → **Impossible** (ratio cannot be > 1)
- **Implication**: For extreme imbalance, cannot achieve balanced threshold via cost ratio alone

**Refined hypothesis**:
- **Moderate imbalance** (1:10 to 1:50): Cost ratio can compensate → Linear relationship
- **Extreme imbalance** (> 1:100): Cost ratio saturates → Need resampling or anomaly-prior adjustment

**Testable**: Grid search on (imbalance, cost ratio) → Find sweet spot.

---

#### 4.2.3 SNR vs Cost Sensitivity

**Hypothesis**: Low SNR datasets require **conservative** cost ratios (lower FN penalty) to avoid threshold collapse.

**Mechanism**:
- Low SNR → Detector scores are noisy → Wide confidence intervals
- High FN cost → Threshold pushed very low → Many FPs
- **Total cost**: FPs × C_FP + FNs × C_FN
  - If threshold → 0: FPs → ∞, FNs → 0 → Total cost explodes

**Expected finding**: Optimal cost ratio **decreases** as SNR decreases.

**Measurement** `[TODO: Compute SNR for SKAB/SMD by Week 2]`:
- [ ] Estimate SNR: `10 × log10(signal_power / noise_power)`
  - Signal power: Variance of anomaly segments
  - Noise power: Variance of normal segments (residual after detrending)
- [ ] Plot: SNR (x-axis) vs Optimal cost ratio (y-axis)

---

#### 4.2.4 Literature Support

**Reference 1**: Elkan (2001) - "The Foundations of Cost-Sensitive Learning"
- **Finding**: Optimal threshold depends on class priors and cost matrix
- **Formula**: `τ* = p(neg) × (C_FP / (C_FP + C_FN))`
- **Limitation**: Assumes **perfect calibration** and **static costs**

**Reference 2**: Ling et al. (2006) - "Cost-Sensitive Learning and the Class Imbalance Problem"
- **Finding**: For imbalance > 1:100, cost-sensitive thresholding alone is **insufficient**; need resampling
- **Implication**: LFactory may need to implement SMOTE or undersampling for extreme imbalance

**Reference 3**: Zhou & Liu (2006) - "Training Cost-Sensitive Neural Networks with Methods Addressing the Class Imbalance Problem"
- **Finding**: Cost ratio should scale with `sqrt(imbalance)` (empirical)
- **Example**: Imbalance = 1:100 → Cost ratio ≈ 10
- **Testable**: Fit `cost_ratio = a × imbalance^b` and find b ≈ 0.5

**Reference 4** `[TODO: Find manufacturing-specific cost analysis]`

---

### 4.3 Testable Hypotheses

**H4.1**: Optimal FN/FP cost ratio **increases linearly** with dataset imbalance for moderate imbalance (1:10 to 1:50).
- **Formula**: `cost_ratio ≈ 0.5 × imbalance` (e.g., imbalance=20 → ratio=10)

**H4.2**: For extreme imbalance (> 1:100), optimal cost ratio **saturates** at ≈ 20, and further increasing ratio **increases** expected cost (threshold collapse).

**H4.3**: Optimal cost ratio **decreases** as SNR decreases:
- High SNR (> 10 dB): Ratio up to 20 is effective
- Low SNR (< 3 dB): Ratio should be ≤ 5

**H4.4**: Cost sensitivity interacts with **detector quality**:
- Well-calibrated detector (ECE < 0.05): Cost ratio directly translates to threshold
- Poorly calibrated detector (ECE > 0.15): Cost ratio has **weak effect** on expected cost

---

### 4.4 Acceptance Criteria

**RQ4 is considered rigorously answered if**:
- [ ] Grid search: Imbalance ∈ {0.01, 0.02, 0.05, 0.10}, Cost ratio ∈ {1, 3, 5, 10, 20}
- [ ] ≥3 datasets with varying imbalance (SKAB ≈ 0.05, SMD ≈ 0.03, Synthetic controllable)
- [ ] SNR computed for each dataset; correlation analysis with optimal cost ratio
- [ ] Heatmap: Imbalance (x) × Cost ratio (y) → Color = Expected cost
- [ ] Fitted model: `cost_ratio = f(imbalance, SNR)` with R² > 0.7
- [ ] Practitioner guideline table:
  ```
  | Imbalance | SNR    | Recommended Cost Ratio |
  |-----------|--------|------------------------|
  | 1:10      | > 5 dB | 5                      |
  | 1:50      | > 5 dB | 10                     |
  | 1:100     | > 5 dB | 15 (with resampling)   |
  | Any       | < 3 dB | 3 (conservative)       |
  ```
- [ ] Documented in `experiments/COST_SENSITIVITY_ANALYSIS.md`

---

## Summary: Roadmap for Answering RQs

### Week 2 Tasks (Data Analysis Foundation)
- [ ] **RQ1 prep**: Frequency analysis of SKAB and AIHub (PSD, discriminative bands)
- [ ] **RQ3 prep**: Compute anomaly event length distributions
- [ ] **RQ4 prep**: Estimate SNR for all datasets

### Week 3 Tasks (Detector Improvements)
- [ ] **RQ1**: Implement time-aware ML detector, run ablation
- [ ] **RQ2**: Implement Product/Max/Learned ensemble variants

### Week 4 Tasks (Full Evaluation)
- [ ] **RQ1**: Full batch evaluation (all detectors × all files)
- [ ] **RQ2**: Ensemble method comparison with statistical tests
- [ ] **RQ3**: Scatter plot analysis (Point F1 vs Event F1)
- [ ] **RQ4**: Grid search on cost ratio × imbalance

### Week 5 Tasks (Analysis & Documentation)
- [ ] Statistical significance tests for all RQs
- [ ] Create ablation/analysis markdown files
- [ ] Update EXPERIMENT_REPORT.md with RQ answers
- [ ] Practitioner guidelines for each RQ

---

## Honest Assessment of Current State

### What We Know
- ✅ Hybrid detector improves over Rule baseline (preliminary evidence)
- ✅ Calibration reduces ECE (expected, but validated on SKAB)
- ✅ Cost-sensitive thresholding changes optimal threshold (validated)

### What We Don't Know (Requires RQ Investigation)
- ❓ **RQ1**: Whether frequency features are actually better (current SpecCNN bands are arbitrary)
- ❓ **RQ2**: Which ensemble method is best (only tested Linear so far)
- ❓ **RQ3**: Point vs Event correlation (only 4 data points, insufficient)
- ❓ **RQ4**: How to set cost ratio (current default of 5 is arbitrary)

### What We Suspect But Cannot Claim
- 🤔 SpecCNN should excel on SKAB (periodic machinery) → Needs frequency analysis proof
- 🤔 Product ensemble might be better for high-imbalance → Needs comparison
- 🤔 Point and Event F1 are weakly correlated → Needs full batch data
- 🤔 Cost ratio should scale with imbalance → Needs grid search

**Next step**: Execute Week 2 data analysis tasks to move from "suspect" to "know".

---

## References

1. Breiman, L. (1996). Bagging predictors. *Machine learning*, 24(2), 123-140.
2. Chakraborty, D., et al. (2020). Deep learning for time-series anomaly detection: A survey. *Preprint*.
3. Dietterich, T. G. (2000). Ensemble methods in machine learning. *MCS*.
4. Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*.
5. Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*.
6. Hundman, K., et al. (2018). Detecting spacecraft anomalies using LSTMs. *KDD*.
7. Katser, I. D., & Kozitsin, V. (2020). Skoltech Anomaly Benchmark (SKAB). *arXiv*.
8. Ling, C. X., et al. (2006). Cost-sensitive learning and the class imbalance problem. *Encyclopedia of ML*.
9. Malhotra, P., et al. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. *ICML Workshop*.
10. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
11. Tatbul, N., et al. (2018). Precision and recall for time series. *NeurIPS*.
12. Xu, H., et al. (2018). Unsupervised anomaly detection via VAE for seasonal KPIs. *WWW*.
13. Zhou, Z. H., & Liu, X. Y. (2006). Training cost-sensitive neural networks. *ICML*.

---

**Version History**:
- 1.0 (2025-10-01): Initial draft with preliminary evidence and hypotheses
- 1.1 (Planned Week 2): Update with SKAB frequency analysis, SNR estimates
- 2.0 (Planned Week 5): Finalize with experimental results for all RQs

