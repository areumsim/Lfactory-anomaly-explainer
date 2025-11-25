# Research Questions Interdependency Analysis

**Version**: 1.0
**Last Updated**: 2025-10-01
**Status**: Initial draft (to be refined after Week 2 experiments)

---

## Executive Summary

This document analyzes how the 4 research questions (RQs) in LFactory are **organically interconnected**, forming a unified research narrative rather than independent contributions. Understanding these dependencies is critical for:
1. **Experimental design**: RQ answers depend on each other's results
2. **Interpretation**: Cannot interpret RQ2 without RQ1 context
3. **Practical application**: Real-world deployment requires all 4 RQs answered together

---

## 1. The Four Research Questions (Recap)

From `TODO.md` Section 1.3:

- **RQ1** (Feature Domain): "Do frequency-domain features (DFT bands) outperform time-domain features (rolling stats) for manufacturing spike/step/drift anomalies?"

- **RQ2** (Ensemble Method): "Which ensemble method (linear, product, max, learned) achieves best calibration-cost trade-off?"

- **RQ3** (Metric Correlation): "What is the correlation between point-wise F1 and event-wise F1 across detectors?"

- **RQ4** (Cost Sensitivity): "How should the FN/FP cost ratio vary with dataset imbalance and SNR?"

---

## 2. Pairwise Dependencies

### 2.1 RQ1 → RQ2: Feature Domain Affects Ensemble Design

**Dependency**: The optimal ensemble method (RQ2) depends on which feature domain performs better (RQ1).

**Mechanism**:
- If frequency features (SpecCNN) dominate time features (Rule) on a dataset (RQ1 answer: "frequency wins"), then:
  - **Linear ensemble** should assign higher weight α to SpecCNN
  - **Product ensemble** might over-amplify SpecCNN's high confidence
  - **Max ensemble** might ignore Rule entirely
  - **Learned ensemble** should learn to trust SpecCNN more

**Example Scenario**:
```
Dataset: SKAB valve (periodic vibration)
RQ1 Result: SpecCNN AUC-PR = 0.85, Rule AUC-PR = 0.65 → Frequency wins

RQ2 Implication:
- Linear (α=0.7): 0.3×Rule + 0.7×SpecCNN → AUC-PR ≈ 0.80
- Product: Rule^0.3 × SpecCNN^0.7 → Might be ≈ 0.82 (less penalty from Rule)
- Learned: Automatically assigns SpecCNN weight ≈ 0.8
```

**Experimental Protocol Consequence**:
- Cannot finalize RQ2 ensemble method choice until RQ1 is answered per dataset
- RQ2 experiments should be stratified by RQ1 result (frequency-dominant vs time-dominant datasets)

**Citation**: This is related to **stacked generalization** (Wolpert, 1992), where level-1 model performance depends on level-0 model diversity.

---

### 2.2 RQ2 → RQ3: Ensemble Method Affects Point/Event Metric Trade-off

**Dependency**: Different ensemble methods (RQ2) may bias toward point-wise precision vs event-wise recall (RQ3).

**Mechanism**:
- **Linear ensemble**: Smooth score averaging → Fewer isolated false positives → Higher point precision, lower event recall
- **Max ensemble**: Takes maximum score → More aggressive → Higher event recall, lower point precision
- **Product ensemble**: Requires both detectors to agree → Conservative → Higher point precision, lower event recall
- **Learned ensemble**: Depends on training objective (if trained on point-wise loss, biases toward point F1)

**Example Scenario**:
```
Detector: Hybrid (Rule + ML)
Ensemble: Linear α=0.5

Point F1: 0.72 (precise but conservative)
Event F1: 0.65 (misses some events due to averaged low scores)

Ensemble: Max

Point F1: 0.68 (more false positives)
Event F1: 0.78 (catches more events)
```

**Experimental Protocol Consequence**:
- RQ3 correlation analysis must be repeated for each RQ2 ensemble variant
- RQ3 scatter plot should have different colors per ensemble method
- Cannot claim "point and event F1 are correlated" without specifying which ensemble

**Citation**: Analogous to precision-recall trade-off in threshold selection (Davis & Goadrich, 2006).

---

### 2.3 RQ3 → RQ4: Metric Correlation Affects Cost Matrix Design

**Dependency**: If point F1 and event F1 are uncorrelated (RQ3 answer: "weak correlation"), then cost matrix (RQ4) must account for both metrics.

**Mechanism**:
- **High correlation (ρ > 0.8)**: Optimizing for point F1 also optimizes for event F1 → Single cost matrix suffices
- **Low correlation (ρ < 0.6)**: Optimizing for one metric hurts the other → Need to decide which metric cost matrix prioritizes

**Example Scenario**:
```
RQ3 Result: Point F1 vs Event F1 correlation = 0.45 (weak)

RQ4 Implication:
- If manufacturing domain prioritizes event detection (missing an event = critical):
  → Cost matrix should use **event-based FN cost**
  → C_FN = 10 × C_FP (high penalty for missed events)
  → Threshold optimized for event recall, accept lower point precision

- If manufacturing domain prioritizes low false alarm rate:
  → Cost matrix uses **point-based FP cost**
  → C_FP = 5 × C_FN
  → Threshold optimized for point precision, accept missed events
```

**Experimental Protocol Consequence**:
- RQ4 cost sensitivity analysis must report both point-optimal and event-optimal thresholds
- If correlation is weak, recommend two operating points: "high-recall mode" and "high-precision mode"
- Document in `EVALUATION_PROTOCOL.md`: "For processes where event detection is critical (e.g., valve failure), use event-optimized cost matrix (C_FN=10). For processes where false alarms are expensive (e.g., production line stoppage), use point-optimized matrix (C_FP=5)."

**Citation**: Multi-objective optimization in imbalanced learning (Galar et al., 2011).

---

### 2.4 RQ4 → RQ1: Cost Sensitivity Depends on Anomaly Type Detection

**Dependency**: The optimal cost ratio (RQ4) depends on which anomaly types are detected better by which features (RQ1).

**Mechanism**:
- **RQ1 finding**: Frequency features detect **drift** well (SNR-robust), time features detect **spikes** well (local sensitivity)
- **RQ4 cost matrix**: If **drift** is more costly to miss (e.g., gradual equipment degradation → expensive failure), then:
  - Increase FN cost for drift anomalies
  - Weight ensemble toward frequency features (SpecCNN)

**Example Scenario**:
```
Dataset: AIHub chemical sensor
Anomaly types: 60% drift, 40% spike

RQ1 Result:
- Drift detection: SpecCNN F1 = 0.80, Rule F1 = 0.55
- Spike detection: SpecCNN F1 = 0.60, Rule F1 = 0.75

Domain knowledge: Drift → equipment failure (C_FN = $10,000)
                   Spike → transient noise (C_FN = $100)

RQ4 Cost Matrix (weighted by frequency × severity):
- Effective C_FN = 0.6 × 10,000 + 0.4 × 100 = $6,040
- Optimize for drift recall → Use SpecCNN-heavy ensemble
```

**Experimental Protocol Consequence**:
- RQ4 analysis should be **stratified by anomaly type** (if labels include type metadata)
- If RQ1 shows feature domain preference varies by anomaly type, then RQ4 should recommend **anomaly-type-specific cost matrices**

**Citation**: Context-dependent cost matrices in active learning (Lowell et al., 2019).

---

## 3. Three-Way Dependencies

### 3.1 RQ1 + RQ2 + RQ4: Unified Cost-Optimal Detector

**Connection**: All three RQs converge on the final question: **"What is the best detector configuration (features + ensemble + threshold) for minimizing expected cost?"**

**Workflow**:
1. **RQ1**: Determine which features perform best per dataset/anomaly type
   - Output: Feature preference matrix `F[dataset, anomaly_type] → {time, freq, hybrid}`

2. **RQ2**: Select ensemble method that balances calibration and cost
   - Input: Feature scores from RQ1
   - Output: Combined score via optimal ensemble method

3. **RQ4**: Set threshold using dataset-specific cost matrix
   - Input: Combined score from RQ2
   - Output: Binary prediction at cost-optimal threshold

**Diagram**:
```
RQ1: Feature Selection         RQ2: Ensemble         RQ4: Thresholding
    ↓                              ↓                      ↓
Time features (Rule) ──┐       Linear: 0.3×R + 0.7×S     Cost matrix:
Freq features (SpecCNN) ─→  →  Product: R^0.3 × S^0.7  → (0, 1, 10, 0)
    ↑                          Max: max(R, S)             Threshold: τ*
RQ1 result per dataset         Learned: f(R, S)           Expected cost ↓
```

**Integration**:
- Final detector: `HybridCostOptimal(feature_domain=RQ1_answer, ensemble_method=RQ2_answer, cost_matrix=RQ4_answer)`
- **Cannot decouple**: Changing any one RQ answer requires re-evaluating the others

---

### 3.2 RQ2 + RQ3 + RQ4: Metric-Aware Cost Optimization

**Connection**: Cost optimization (RQ4) must account for which metric matters (RQ3), and ensemble method (RQ2) affects metric trade-off.

**Workflow**:
1. **RQ3**: Determine if point F1 and event F1 are correlated
   - If **high correlation**: Optimize for either metric (same threshold)
   - If **low correlation**: Need to choose primary metric

2. **RQ4**: Design cost matrix aligned with chosen metric
   - If **point-primary**: Cost = Σ (point-wise FP + point-wise FN × cost_ratio)
   - If **event-primary**: Cost = Σ (event-wise FP + event-wise FN × cost_ratio)

3. **RQ2**: Select ensemble method that best optimizes the chosen metric under cost constraint
   - Evaluate: ECE (calibration quality) + Expected cost (primary metric)

**Example**:
```
RQ3 Result: Correlation = 0.50 (weak) → Need to choose

Manufacturing domain analysis: Event recall is critical
→ Primary metric: Event F1

RQ4 Cost Matrix: (0, 1, 15, 0) [high FN penalty]
→ Optimize threshold for event recall

RQ2 Ensemble Comparison:
- Linear (α=0.5): Event F1 = 0.72, ECE = 0.04, Cost = 120
- Max:            Event F1 = 0.78, ECE = 0.06, Cost = 95  ← Winner
- Product:        Event F1 = 0.68, ECE = 0.03, Cost = 140

Choose: Max ensemble (sacrifices calibration slightly for better event recall and lower cost)
```

---

## 4. Experimental Design Implications

### 4.1 Sequential vs Parallel Execution

**Option A: Sequential** (Recommended for initial exploration)
1. Week 2: Answer RQ1 (feature comparison) → Identify frequency-dominant vs time-dominant datasets
2. Week 3: Answer RQ2 (ensemble methods) → Stratify by RQ1 result
3. Week 3: Answer RQ3 (metric correlation) → Per ensemble method from RQ2
4. Week 4: Answer RQ4 (cost sensitivity) → Using best ensemble from RQ2, accounting for RQ3 metric choice

**Option B: Parallel** (For comprehensive analysis)
- Run all RQ experiments in full factorial design:
  - RQ1: 2 feature domains × 3 datasets
  - RQ2: 4 ensemble methods × 3 datasets
  - RQ3: (4 ensemble methods × 3 datasets) → correlation per combo
  - RQ4: 5 cost ratios × 3 datasets × 4 ensemble methods
- **Total**: ~180 experiment runs
- **Time**: ~1 week with batch scripts

**Recommendation**: Start with Sequential (Option A) to build understanding, then use Option B for final validation.

---

### 4.2 Reporting Structure

**EXPERIMENT_REPORT.md** should reflect dependencies:

```markdown
## Section 2: Research Question Answers

### RQ1: Frequency vs Time Features (Foundation)
- **Result**: SpecCNN outperforms Rule on SKAB (0.85 vs 0.65 AUC-PR, p<0.01)
- **Implication for RQ2**: Ensemble should favor SpecCNN on SKAB

### RQ2: Ensemble Methods (Depends on RQ1)
- **Stratified Analysis**:
  - Frequency-dominant datasets (SKAB): Product ensemble wins
  - Time-dominant datasets (AIHub): Linear ensemble wins
- **Implication for RQ3**: Need separate metric analysis per dataset type

### RQ3: Point vs Event Correlation (Depends on RQ2)
- **Result**: Correlation = 0.52 ± 0.08 (weak to moderate)
- **Implication for RQ4**: Cost matrix should be event-primary for manufacturing

### RQ4: Cost Sensitivity (Integrates RQ1-3)
- **Result**: Optimal cost ratio = 10× FN/FP for high-imbalance datasets (>1:50)
- **Configuration**: Use Product ensemble + SpecCNN-heavy + event-optimized threshold
```

---

## 5. Theoretical Justification for Interconnectedness

### 5.1 Detection as a Pipeline (Not Independent Stages)

The 4 RQs correspond to 4 stages of the anomaly detection pipeline:

```
Raw Data → Feature Extraction (RQ1) → Score Fusion (RQ2) → Thresholding (RQ4) → Evaluation (RQ3)
              ↑                          ↑                       ↑                    ↑
           Time vs Freq           Linear/Product/Max      Cost-sensitive τ*    Point vs Event
```

**Theoretical claim**: In a pipeline, optimizing each stage independently is **suboptimal** compared to joint optimization (Bishop, 2006, Pattern Recognition).

**Evidence**: If RQ4 cost matrix changes, it may change the optimal threshold, which may change which ensemble method (RQ2) is best, which may change which features (RQ1) should be weighted higher.

---

### 5.2 Multi-Objective Optimization Framework

The 4 RQs are actually constraints/objectives in a single optimization problem:

**Objective**: Minimize expected cost (RQ4)

**Subject to**:
- Feature domain choice (RQ1): `θ_features ∈ {time, freq, hybrid}`
- Ensemble method (RQ2): `θ_ensemble ∈ {linear, product, max, learned}`
- Calibration quality (RQ2): `ECE < 0.05`
- Metric constraint (RQ3): If `corr(F1_point, F1_event) < 0.6`, choose primary metric

**Solution**: Pareto frontier of (expected cost, ECE, event recall)

**Citation**: Multi-objective optimization in machine learning (Jin & Sendhoff, 2008).

---

## 6. Practical Guidance for Practitioners

### 6.1 Decision Tree for Detector Configuration

```
START: New manufacturing dataset

Step 1 (RQ1): Analyze frequency characteristics
├─ High frequency separation (PSD has clear peaks)?
│  → Use SpecCNN-heavy ensemble (α ≥ 0.7)
└─ Broadband noise?
   → Use Rule-heavy ensemble (α ≤ 0.3)

Step 2 (RQ2): Choose ensemble method
├─ Need calibrated probabilities (for downstream decision)?
│  → Use Linear or Learned ensemble (better ECE)
└─ Only need binary detection?
   → Use Max or Product (better recall)

Step 3 (RQ3): Choose primary metric
├─ False alarms very expensive?
│  → Point F1 primary
└─ Missing events critical?
   → Event F1 primary

Step 4 (RQ4): Set cost matrix
├─ Imbalance > 1:50 AND Event primary?
│  → C_FN/C_FP ratio ≥ 10
├─ Imbalance < 1:20 AND Point primary?
│  → C_FN/C_FP ratio ≤ 5
└─ Otherwise: Start with ratio = 5, tune via validation
```

---

## 7. Open Questions for Future Work

### 7.1 Unresolved Dependencies

- **RQ1 temporal dependency**: Does optimal feature domain change over time as equipment ages? (Concept drift)
- **RQ4 delay-dependent cost**: Should cost matrix include detection delay? `C_FN(delay) = C_FN_base × (1 + β × delay)`

### 7.2 Dataset-Specific vs Universal Answers

- Can we find **universal** answers (e.g., "Product ensemble always best"), or are all answers dataset-specific?
- If dataset-specific, what are the **meta-features** (imbalance, SNR, frequency characteristics) that predict optimal configuration?

**Proposed future RQ**: "Can we build a meta-model that predicts optimal (RQ1, RQ2, RQ4) configuration from dataset profiling alone?"

---

## 8. Summary Table

| RQ Pair | Dependency Strength | Mechanism | Experimental Consequence |
|---------|---------------------|-----------|--------------------------|
| RQ1 → RQ2 | **Strong** | Feature domain affects ensemble weight | Stratify RQ2 by RQ1 result |
| RQ2 → RQ3 | **Moderate** | Ensemble method affects metric trade-off | Report RQ3 per ensemble |
| RQ3 → RQ4 | **Strong** | Metric choice affects cost design | RQ4 uses metric from RQ3 |
| RQ4 → RQ1 | **Moderate** | Cost depends on anomaly type detection | RQ4 stratified by anomaly type |
| RQ1+RQ2+RQ4 | **Very Strong** | All converge on cost-optimal config | Cannot answer one without others |
| RQ2+RQ3+RQ4 | **Strong** | Metric-aware cost optimization | Joint optimization needed |

---

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
2. Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*.
3. Galar, M., et al. (2011). A review on ensembles for the class imbalance problem. *IEEE TKDE*.
4. Jin, Y., & Sendhoff, B. (2008). Pareto-based multiobjective machine learning. *IEEE TNNLS*.
5. Lowell, D., et al. (2019). Practical obstacles to deploying active learning. *EMNLP*.
6. Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259.

---

**Version History**:
- 1.0 (2025-10-01): Initial draft based on TODO.md RQ1-RQ4
- 1.1 (Planned Week 2): Update with preliminary RQ1 results
- 2.0 (Planned Week 5): Finalize with all RQ answers and empirical dependency confirmation

