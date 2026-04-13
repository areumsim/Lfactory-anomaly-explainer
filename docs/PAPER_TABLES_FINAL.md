# 논문 최종 테이블 (2026-04-08)

모든 수치는 6차+확대 실험(N=50 per cell) 결과 기준.

---

## Table 1: Feature Attribution Effect on Keyword Faithfulness

*Wilcoxon signed-rank test, baseline vs feature_only*

| Dataset | Sensors | Mode | N | Baseline | Feature | Δ% | p-value | Cliff's d |
|---------|:-------:|:----:|:-:|:--------:|:-------:|:--:|:-------:|:---------:|
| SMD | 38 | TP | 50 | 0.503 | 0.883 | +75 | <10⁻⁶ | 0.952 |
| SMD | 38 | FN | 50 | 0.430 | 0.850 | +98 | <10⁻⁶ | 0.975 |
| SMD | 38 | FP | 50 | 0.252 | 0.755 | +200 | <10⁻⁶ | 1.000 |
| SWaT | 51 | TP | 50 | 0.488 | 0.932 | +91 | <10⁻⁶ | 0.950 |
| SWaT | 51 | FN | 50 | 0.282 | 0.798 | +183 | <10⁻⁶ | 0.999 |
| SWaT | 51 | FP | 50 | 0.250 | 0.750 | +200 | <10⁻⁶ | 1.000 |
| AIHub | 8 | TP | 20 | 0.312 | 0.787 | +152 | <10⁻⁶ | — |
| SKAB | 1 | TP | 20 | 0.575 | 0.575 | 0 | 1.000 | 0.000 |

---

## Table 2: Nosensor Confound Test (Domain Knowledge에서 센서 이름 효과 분리)

*Wilcoxon signed-rank test, domain_only vs domain_nosensor*

| Dataset | N | domain_only KW | domain_nosensor KW | Δ% | p-value | Cliff's d |
|---------|:-:|:--------------:|:------------------:|:--:|:-------:|:---------:|
| SWaT | 150 | 0.751 | 0.340 | +121 | <10⁻⁶ | 0.907 |
| SMD | 150 | 0.382 | 0.379 | +1 | 0.496 (ns) | 0.003 |

**해석**: SWaT domain YAML에 센서 이름 포함 → 제거 시 KW 급락. SMD domain YAML에는 센서 이름 없음 → 효과 없음 (대조군 일치).

---

## Table 3: Sensor Top1 Identification Accuracy

*N=150 per condition (TP 50 + FN 50 + FP 50)*

| Condition | SMD (38 sensors) | SWaT (51 sensors) |
|-----------|:----------------:|:------------------:|
| baseline | 0/150 (0%) | 0/150 (0%) |
| feature_only | 105/150 (**70%**) | 85/150 (**57%**) |
| domain_only | 0/150 (0%) | 18/150 (12%) |
| full | 96/150 (64%) | 95/150 (63%) |
| domain_nosensor | 0/150 (0%) | 0/150 (0%) |
| full_nosensor | 98/150 (65%) | 83/150 (55%) |

---

## Table 4: Detector Rank Reversal (AUC-PR)

| Rank | SKAB (1) | SMD (38) | AIHub (8) | SWaT (51) |
|:----:|:--------:|:--------:|:---------:|:---------:|
| 1 | SpecCNN 0.446 | kNN 0.390 | SpecCNN 0.739 | IF 0.046 |
| 2 | IF 0.433 | IF 0.349 | Rule 0.659 | Rule 0.027 |
| 3 | kNN 0.420 | SpecCNN 0.202 | kNN 0.649 | SpecCNN 0.024 |
| 4 | Rule 0.367 | Rule 0.057 | IF 0.583 | kNN 0.019 |

Kendall's W = 0.125 (very low concordance).

---

## Table 5: Calibration ECE Distortion (Temporal Split vs Leakage)

| Detector | ECE (leakage) | ECE (temporal split) | Distortion |
|----------|:-------------:|:--------------------:|:----------:|
| Rule | 0.011 | 0.187 | 16.5× |
| kNN | 0.026 | 0.186 | 7.2× |
| IF | 0.090 | 0.203 | 2.3× |
| SpecCNN | 0.106 | 0.194 | 1.8× |

---

## Table 6: Feature Granularity Ablation (SWaT 51 sensors, N=15, 5차 결과)

| Condition | KW Faith | Top1 Acc | sensor_id |
|-----------|:--------:|:--------:|:---------:|
| no_feature | 0.500 | 0% | 3.40/5 |
| names_only | 0.722 | 87% | 3.73/5 |
| top1 | 0.833 | 100% | 4.20/5 |
| **top3** | **0.811** | **100%** | **4.47/5** |
| full_ranking | 0.878 | 87% | 3.60/5 |
| zscore_only | 0.500 | 0% | 3.33/5 |
| adversarial | 0.856 | 100% | 4.20/5 |

**names_only(87%) vs zscore_only(0%)**: 센서 이름이 핵심, z-score는 무의미.
**top3(4.47) > full_ranking(3.60)**: 정보 과부하 효과.
**adversarial(100%)**: LLM은 제공된 이름에 100% 의존 (인과적 증거).

---

## Supplementary Table S1: TP vs FN vs FP Comparison

| Dataset | TP KW | FN KW | FP KW | Kruskal-Wallis H |
|---------|:-----:|:-----:|:-----:|:----------------:|
| SWaT | 0.932 | 0.798 | 0.750 | **20.39 (p<0.001)** |
| SMD | 0.883 | 0.850 | 0.755 | 1.03 (ns) |

SMD 비유의: keyword faithfulness 메트릭이 3단계 이산값으로 분산 제한.

## Supplementary Table S2: Direction Accuracy

| Dataset | Correct | Total | % | Ambiguous |
|---------|:-------:|:-----:|:---:|:---------:|
| SMD | 503 | 576 | 87.3 | 324 |
| SWaT | 423 | 552 | 76.6 | 348 |

방향이 명확한 이상(sigma > 0.5*std)에서만 평가. Ambiguous의 95.5%가 sigma < 0.05.

## Supplementary Table S3: Cross-Model Judge Agreement

*GPT-4o-mini (생성+1차 평가) vs GPT-4 (2차 평가)*

| Condition | N | Orig Mean | Cross Mean | Spearman ρ |
|-----------|:-:|:---------:|:----------:|:----------:|
| baseline | 100 | 0.863 | 0.768 | 0.299 |
| feature_only | 100 | 0.811 | 0.638 | 0.527 |
| full | 100 | 0.774 | 0.761 | 0.315 |

LLM-as-judge는 절대값보다 조건 간 상대 비교에 적합.
