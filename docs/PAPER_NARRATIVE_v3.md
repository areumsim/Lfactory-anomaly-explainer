# 논문 프레임 v3: 6차+확대 실험 반영

**작성일**: 2026-04-08
**이전 버전**: PAPER_NARRATIVE_v2.md (v2: C1=Rank Reversal 중심)
**변경 이유**: 6차 실험 + N=50 확대로 C2(Feature Attribution) 통계적 유의성 대폭 강화. C1/C2 비중 재조정.

---

## 제목 (안)

**"Sensor Names Matter: What Feature Attribution Actually Provides to LLM Anomaly Explanations in Multi-Sensor Time Series"**

한국어:
**"센서 이름이 핵심이다: 다중 센서 시계열 이상탐지에서 Feature Attribution이 LLM 설명에 실제로 제공하는 것"**

부제안:
**"An Empirical Study Across 4 Datasets, 300 Samples, and 1,800 LLM Evaluations"**

---

## 연구 동기

LLM 기반 이상 설명이 확산되고 있으나, **LLM에 무엇을 제공해야 설명이 좋아지는지**에 대한 실증적 연구가 부족하다.

핵심 질문:
1. Feature attribution(어떤 센서가 이상한지)을 제공하면 설명이 좋아지는가?
2. 좋아진다면, **구체적으로 어떤 정보**(센서 이름? z-score? 순위?)가 핵심인가?
3. 이 효과는 센서 수, 탐지 결과(TP/FN/FP), 데이터셋에 따라 어떻게 달라지는가?

---

## 3대 기여 (6차+확대 결과 반영)

### C1 (주력, 50%): Feature Attribution → LLM 설명 충실성

**센서 이름 제공이 LLM 설명 품질의 핵심 요인이다.**

#### Table 1: Feature Attribution Effect (N=50 per cell, Wilcoxon signed-rank)

| Dataset | Sensors | Mode | N | Baseline KW | Feature KW | Δ | p-value | Cliff's d |
|---------|---------|------|---|-------------|-----------|---|---------|-----------|
| SMD | 38 | TP | 50 | 0.503 | 0.883 | +75% | <0.000001 | 0.952 |
| SMD | 38 | FN | 50 | 0.430 | 0.850 | +98% | <0.000001 | 0.975 |
| SMD | 38 | FP | 50 | 0.252 | 0.755 | +200% | <0.000001 | 1.000 |
| SWaT | 51 | TP | 50 | 0.488 | 0.932 | +91% | <0.000001 | 0.950 |
| SWaT | 51 | FN | 50 | 0.282 | 0.798 | +183% | <0.000001 | 0.999 |
| SWaT | 51 | FP | 50 | 0.250 | 0.750 | +200% | <0.000001 | 1.000 |

**전 조건 p<0.000001, Cliff's d>0.95 (very large effect)**

#### Table 2: Nosensor Confound Test (domain knowledge에서 센서 이름 효과 분리)

| Dataset | N | domain_only | domain_nosensor | Δ | p-value | Cliff's d |
|---------|---|-------------|-----------------|---|---------|-----------|
| SWaT | 150 | 0.751 | 0.340 | +121% | <0.000001 | 0.907 |
| SMD | 150 | 0.382 | 0.379 | +1% | 0.496 (ns) | 0.003 |

→ SWaT domain YAML에 센서 이름(FIT101 등)이 포함 → 제거 시 KW 급락
→ SMD domain YAML에는 센서 이름 없음 → nosensor 효과 없음 (대조군 일치)
→ **센서 이름이 domain knowledge 효과의 핵심 요인임을 인과적으로 입증**

#### Table 3: Sensor Top1 Identification Accuracy (N=150 per condition)

| Condition | SMD (38 sensors) | SWaT (51 sensors) |
|-----------|:----------------:|:------------------:|
| baseline | 0% | 0% |
| **feature_only** | **70%** | **57%** |
| domain_only | 0% | 12% |
| full | 64% | 63% |
| domain_nosensor | 0% | 0% |
| full_nosensor | 65% | 55% |

→ Feature attribution 없이는 센서 식별 **불가능** (0%)
→ 제공 시 SMD 70%, SWaT 57%로 극적 향상

#### 비자명한 통찰 4가지

1. **센서 이름 > z-score**: names_only 86.7% vs zscore_only 0% (5차 granularity 결과)
2. **Top-3 > Full ranking**: sensor_id 4.47(top3) > 3.60(full) — 정보 과부하 효과
3. **FP에서 효과 가장 큼**: +200% (오경보에서도 attribution이 도움)
4. **Adversarial 100% 의존**: 셔플된 잘못된 이름을 LLM이 그대로 보고 (5차)

---

### C2 (보조, 30%): 탐지기 선택은 데이터 의존적, 임계값이 3-29× 더 중요

#### Table 4: Rank Reversal (AUC-PR, 4개 데이터셋)

| Rank | SKAB (1센서) | SMD (38센서) | AIHub (8센서) | SWaT (51센서) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | SpecCNN (0.446) | kNN (0.390) | SpecCNN (0.739) | IF (0.046) |
| 2 | IF (0.433) | IF (0.349) | Rule (0.659) | Rule (0.027) |
| 3 | kNN (0.420) | SpecCNN (0.202) | kNN (0.649) | SpecCNN (0.024) |
| 4 | Rule (0.367) | Rule (0.057) | IF (0.583) | kNN (0.019) |

- Kendall's W = 0.125 (very low concordance) → **4개 데이터셋 모두 1위가 다름**
- Friedman: SMD p<0.000001, AIHub p<0.0001, SKAB p=0.036

#### Threshold Sensitivity

| Detector | Fixed-F1 | Optimal-F1 | Ratio |
|----------|:--------:|:----------:|:-----:|
| Rule | 0.018 | 0.519 | 29.3× |
| kNN | 0.025 | 0.561 | 22.2× |
| SpecCNN | 0.027 | 0.570 | 21.2× |
| IF | 0.160 | 0.615 | 3.8× |

→ **임계값 최적화가 탐지기 선택보다 3.8-29.3× 더 효과적**

---

### C3 (방법론, 20%): Calibration Temporal Split 필수

#### Table 5: ECE 왜곡

| Detector | ECE (leakage) | ECE (temporal split) | 왜곡 배율 |
|----------|:---:|:---:|:---:|
| Rule | 0.011 | 0.187 | 16.5× |
| kNN | 0.026 | 0.186 | 7.2× |
| IF | 0.090 | 0.203 | 2.3× |
| SpecCNN | 0.106 | 0.194 | 1.8× |

→ Temporal split 없이 캘리브레이션하면 ECE가 최대 16.5배 과소평가됨

---

## 보조 분석 결과

### TP vs FN vs FP 비교 (feature_only KW)

| Dataset | TP (N=50) | FN (N=50) | FP (N=50) | Kruskal-Wallis |
|---------|-----------|-----------|-----------|----------------|
| SWaT | 0.932 | 0.798 | 0.750 | **H=20.39 (유의)** |
| SMD | 0.883 | 0.850 | 0.755 | H=1.03 (ns) |

→ SWaT에서 TP>FN>FP 유의한 차이. SMD에서는 경향 동일하나 유의하지 않음.
→ SMD 비유의 원인: keyword faithfulness 메트릭이 이산값(3단계)이라 분산 본질적으로 작음.

### Direction Accuracy

| Dataset | Correct/Total | % | Ambiguous |
|---------|:------------:|:---:|:---------:|
| SMD | 503/576 | 87.3% | 324 |
| SWaT | 423/552 | 76.6% | 348 |

→ 방향이 명확한 이상(sigma > 0.5*std)에서만 평가.
→ Ambiguous의 95.5%가 sigma_deviation < 0.05 (사실상 정상값) — 데이터 특성.

### Cross-Model Judge Agreement (GPT-4o-mini 생성 → GPT-4 평가)

| Condition | N | Orig Mean | Cross Mean | Spearman ρ |
|-----------|---|:---------:|:----------:|:----------:|
| baseline | 100 | 0.863 | 0.768 | 0.299 |
| feature_only | 100 | 0.811 | 0.638 | 0.527 |
| full | 100 | 0.774 | 0.761 | 0.315 |

→ Spearman ρ = 0.21~0.53 — judge 간 순위 일관성 약~중간
→ GPT-4가 더 엄격 (orig > cross 대부분)
→ **LLM-as-judge는 절대값보다 조건 간 상대 비교에 적합. Keyword faithfulness를 주 메트릭으로 사용.**

---

## Negative Results (정식 보고)

1. **앙상블 방법 간 차이 없음** (RQ2): Linear, Product, Max, Learned 기대 비용 0.324-0.344
2. **주파수 특징의 보편적 우위 없음** (RQ1): SpecCNN은 밸브 이상에서만 1위
3. **ECE < 0.05 달성 불가**: Calibration leakage 수정 후 ECE ≈ 0.19

---

## Limitations

### 실험 한계
1. **Human evaluation 미실시**: 자동 메트릭만 사용. 평가 시트 준비 완료, 평가자 배포 대기 중.
2. **Keyword faithfulness 해상도 제한**: 3-4단계 이산값으로 세밀한 차이 포착 어려움. SMD FP 비유의의 원인.
3. **Cross-model judge 일관성 낮음**: Spearman ρ < 0.6. LLM-judge는 절대값보다 상대 비교에만 신뢰.
4. **Direction accuracy ambiguous 36-44%**: sigma_deviation < 0.05인 샘플이 대부분. 방향이 명확한 이상에서만 77-87% 정확.

### 데이터 한계
5. **SWaT 절대 성능 낮음**: AUC-PR 0.02-0.05. 단일 센서(FIT101) 탐지의 한계.
6. **AIHub 비공개 + 높은 label rate**: 54% → random baseline ≈ 0.54. 재현 불가.
7. **SKAB power 부족**: Pairwise 6쌍 중 1쌍만 adequate power.
8. **Phase 3(LLM 파라미터 최적화) 제한적**: SMD +11.6%, SWaT -12.5% — 데이터 의존적.

### 방법론 한계
9. **단변량 분석**: 38/51차원을 단일 센서 기준 탐지. 다변량 상관 미활용.
10. **LLM 모델 한정**: GPT-4o-mini 위주. Claude Haiku 교차 검증은 N=10만.

---

## 실험 규모 요약

| 항목 | 값 |
|------|-----|
| Detection runs | 7,600+ |
| LLM evaluations (6차+확대) | 3,600+ |
| Datasets | 4 (SKAB, SMD, AIHub, SWaT) |
| Detectors | 5 (Rule, kNN, IF, SpecCNN, AT) |
| LLM explanation samples | 300 (TP 100 + FN 100 + FP 100) |
| Ablation conditions | 6 (baseline, domain, feature, full, domain_nosensor, full_nosensor) |
| Cross-model judge evals | 720+ |
| Human eval sheets | 360 rows (SWaT 180 + SMD+AIHub 180) 준비 완료 |

---

## 논문 구조 (예상 11p)

1. **Introduction** (1.5p): LLM 설명의 입력 정보 문제 제기
2. **Related Work** (1.5p): 시계열 AD, XAI, LLM 기반 설명
3. **Method** (2p): Detect-Explain 파이프라인, 탐지기, attribution, LLM 평가
4. **Experiments** (1p): 4 datasets, 실험 설계, 메트릭
5. **Results** (3p):
   - 5.1 Feature Attribution Effect (C1) — Table 1, 2, 3
   - 5.2 Detector Rank Reversal (C2) — Table 4
   - 5.3 Calibration Impact (C3) — Table 5
   - 5.4 Supplementary: TP/FN/FP, Direction, Cross-judge
6. **Discussion** (1p): 센서 이름 vs z-score 메커니즘, 정보 과부하, 실무 함의
7. **Limitations & Future Work** (0.5p)
8. **Conclusion** (0.5p)

---

## 타겟 학회

| 타겟 | 판정 | 근거 |
|------|:---:|------|
| **KSC/KIISE (국내)** | **✅ 충분** | 현재 상태로 투고 가능 |
| **IEEE Access/Sensors** | **⚠️ Human eval 필요** | 시트 준비됨, 평가자만 확보하면 됨 |
| CIKM/PAKDD | ⚠️ | Human eval + 프레임 강화 필요 |
| NeurIPS/ICML | ❌ | Novelty 부족 |
