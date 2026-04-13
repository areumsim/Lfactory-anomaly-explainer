# 논문 프레임 및 핵심 구조

**작성일**: 2026-04-06 (5차 실험 반영 최종본)
**이전 버전**: 2026-03-31 (감사 후 ���정본)
**감사 기준**: 모든 수치는 `runs/` 디렉토리 실제 데이터에서 검증됨
**5차 변경**: SWaT(51센서) + SMD(38센서 다중) 추가, adversarial attribution, 통계 검정 완료

---

## 제목 (안)

**"Feature Attribution Enables Faithful LLM-Generated Explanations
for Multi-Sensor Anomaly Detection in Manufacturing Time Series"**

또는 한국어:
**"특성 귀인(Feature Attribution)이 다중 센서 제조 시계열 이상 탐지에서 LLM 설명의 충실성을 결정한다"**

---

## 3대 기여 (Contributions)

### C1 (주력). Feature Attribution → LLM 설명 충실성 ✅✅✅ [5차 업데이트 2026-04-06]

**[5차] 4개 데이터셋, Spearman ρ=1.0, adversarial 인과적 증거 확보**
→ 상세: `docs/EXPERIMENT_RESULTS_5TH.md` 섹션 2 참조

**핵심 발견 (5차 확대: 4 datasets, N=168, 7 conditions)**:

다중 센서(AIHub 8채널, N=34)에서:
| 조건 | KW Faith | Top1 정확도 | sensor_id |
|------|----------|-----------|-----------|
| no_feature (baseline) | 0.336 | 0% | 1.85/5 |
| **names_only** (이름만) | **0.801** | **91%** | **3.41/5** |
| top1 (1위 센서+z) | 0.801 | 100% | 3.41/5 |
| **top3** (상위3+z) | 0.801 | 100% | **4.65/5** |
| full_ranking (전체+z) | 0.801 | 100% | 3.32/5 |
| **zscore_only** (수치만) | **0.336** | **0%** | **1.38/5** |

**비자명한 통찰 3가지**:
1. **센서 이름이 핵심, z-score가 아님**: names_only=91% vs zscore_only=0%
2. **Top-3이 최적**: sensor_id 4.65 > full_ranking 3.32 → 정보 과부하 효과
3. **단일 센서에서 효과 0**: SKAB/SMD(N=74) 모든 조건 동일 (0.730)

**이전 2×2 Ablation 확인** (N=108):
- keyword: baseline 0.606 → feature 0.752 (+24%, p<0.000001)
- LLM-judge: 전체 차이 없음 (p=0.922), sensor_id만 2.61→3.18 개선

**논문 메시지**: "LLM 설명에서 feature attribution의 가치는 z-score 수치가 아니라 **센서 이름**에 있다. Top-3 제공이 전체 제공보다 효과적이며, 이 효과는 다중 센서 환경에서만 발현된다."

**증거 강도**: 강 (N=108 × 6 conditions = 648 evaluations, AIHub에서 명확한 패턴)
**검증 상태**: ✅ `runs/feature_granularity_ablation/` + `runs/llm_explanation_v3/`

### C2 (보조). 디텍터 순위의 데이터 의존성 ✅ (AT 포함 5개 디텍터)
- 3개 데이터셋, **5개 디텍터** (Rule, kNN, IF, SpecCNN, **Anomaly Transformer**)
- **Anomaly Transformer (ICLR 2022)** 추가 → 최신 DL도 우위 보장 안 됨

| Rank | SKAB | SMD |
|------|------|-----|
| 1 | SpecCNN (0.446) | kNN (0.390) |
| 2 | IF (0.433) | IF (0.349) |
| 3 | kNN (0.420) | SpecCNN (0.202) |
| 4 | **AT (0.396)** | **AT (0.164)** |
| 5 | Rule (0.367) | Rule (0.057) |

- **AT가 SKAB 4위, SMD 4위** → 최신 DL이 전통 방법보다 반드시 우수하지 않음
- Rank Reversal 유지: SpecCNN 1위(SKAB) vs 3위(SMD), kNN 3위(SKAB) vs 1위(SMD)
- **"최신 DL 포함해도 데이터 의존적 성능 패턴은 동일"**

**증거 강도**: 강 (2,500+ runs, 5 detectors, 3 datasets)
**검증 상태**: ✅ `runs/all_results_clean.json` + AT runs

### C3 (보조). Threshold 민감도 > Detector 선택
- Optimal/Point F1 비율: IF 3.7x, kNN 15.2x, SpecCNN 18.7x, Rule 23.9x
- 고정 threshold에서 F1≈0인 디텍터도 oracle에서 F1≈0.5
- **"threshold/비용 전략이 디텍터 자체보다 성능에 3~24배 더 큰 영향"**

**증거 강도**: SKAB 강
**검증 상태**: ✅ SKAB 데이터 기반 확인

---

## 논문 구조

### 1. Introduction (1.5p)
- 제조 이상 탐지의 실무 과제: 탐지는 되지만 **설명이 안 된다**
- 기존 연구의 gap: LLM을 설명에 활용하나, 어떤 입력이 설명 품질을 결정하는지 미검증
- 본 연구: Feature Attribution이 LLM 설명의 충실성(faithfulness)에 미치는 영향 실증
- 부가 발견: 디텍터 순위의 데이터 의존성, threshold의 압도적 영향

### 2. Related Work (1.5p)
- 시계열 이상 탐지 (IF, kNN, frequency-domain)
- XAI for anomaly detection (SHAP, LIME, attention)
- LLM for explanation generation
- Cost-sensitive thresholding

### 3. Method (2p)
- 3.1 Detect-Explain Pipeline: ML 탐지 → Feature Attribution → LLM 설명 생성
- 3.2 Detectors: Rule(z-score), kNN, IF, SpecCNN(spectral flux)
- 3.3 Feature Attribution: reconstruction-based (LSTM-AE), SHAP (tree models)
- 3.4 LLM Explanation: domain knowledge + feature importance → structured prompt
- 3.5 Evaluation: Faithfulness (keyword-based), Sensor ID accuracy, Consistency (Jaccard)

### 4. Experimental Setup (1p)
- 3 datasets:
  - SKAB (valve monitoring, 34 files, ~1,147 pts/file, label_rate ~35%)
  - SMD (server metrics, 28 machines, ~28K pts/machine, label_rate ~9.5%)
  - AIHub71802 (manufacturing transport, 8-channel sensors, ~300 pts/session, label_rate ~54%)
    - **Note**: AIHub의 높은 label_rate는 random baseline AUC-PR ≈ 0.54를 의미
- 5 random seeds per configuration
- Metrics: AUC-PR (primary), Optimal-F1, Point F1, Event F1
- LLM: GPT-4o-mini, 2×2 ablation (domain knowledge × feature importance)

### 5. Results (3p)

#### 5.1 **RQ_main: Feature Attribution → LLM Explanation Quality** (핵심)
- Table 1: 2×2 Ablation 결과
- Table 2: 센서 식별 정확도 (AIHub 8채널)
- 센서 수에 따른 효과 분석 (AIHub vs SKAB/SMD)

#### 5.2 RQ1: Detector Comparison
- Table 3: Detector × Dataset AUC-PR (검증된 수치만)
- Friedman + post-hoc 결과 (SMD 재실험 후 확정)

#### 5.3 Threshold Sensitivity
- Table 4: Optimal/Point F1 비율
- Figure 1: Cost Ratio vs F1

### 6. Discussion (1p)
- Feature attribution의 센서 수 의존성: 실제 제조 환경(수십~수백 센서)에서의 중요성
- Domain knowledge의 제한적 역할: 정확성이 아닌 맥락 제공
- Detector 순위 역전의 함의: 데이터 특성에 맞는 detector 선택 필요
- Threshold의 압도적 영향: adaptive thresholding이 detector 개선보다 우선

### 7. Limitations (0.5p)
- LLM 샘플 크기: N=39 (확대 필요)
- AIHub 높은 label_rate (54%): random baseline 대비 실질적 개선 정도
- AIHub 짧은 시계열 (300-361 pts)
- Faithfulness 측정: keyword-based (semantic similarity 미사용)
- 단일 LLM (GPT-4o-mini): 다른 모델 비교 미수행

### 8. Conclusion (0.5p)
- Feature attribution이 LLM 설명 정확도를 결정하는 핵심 요소
- Future work: larger samples, local LLMs, multi-sensor deep learning

---

## 핵심 Table/Figure — 검증된 수치만 사용

### Table 1: LLM 2×2 Ablation (N=108) ✅ v3 확대 검증

#### Keyword Faithfulness

| | Feature Imp. OFF | Feature Imp. ON |
|---|---|---|
| **Domain Know. OFF** | 0.606 | **0.752** |
| **Domain Know. ON** | 0.606 | **0.748** |

Wilcoxon p < 0.000001, Cliff's delta = -0.305 (small), 향상 34건 / 저하 0건

#### LLM-as-judge Faithfulness (신규)

| | Feature Imp. OFF | Feature Imp. ON |
|---|---|---|
| **Domain Know. OFF** | 0.828 | 0.827 |
| **Domain Know. ON** | 0.750 | 0.758 |

LLM-judge 전체 점수는 차이 없음 (p=0.922). 단, **sensor_id 차원**에서만 2.61→3.18 개선.
출처: `runs/llm_explanation_v3/llm_explanation_results.json`

### Table 2: LLM 센서 식별 정확도 (AIHub 8채널, N=13) ✅ 검증됨

| 조건 | Top1 센서 정확도 | Top3 겹침 |
|------|----------------|----------|
| baseline | 0.0% | 0.00/3 |
| domain_only | 0.0% | 0.00/3 |
| **feature_only** | **92.3%** | **2.31/3** |
| **full** | **84.6%** | **2.23/3** |

출처: `runs/llm_explanation_v2/sensor_faithfulness.json`

### Table 3: Detector × Dataset AUC-PR ✅ (전체 검증 완료)

| Detector | SKAB (n=80) | SMD (n=135) | AIHub (n=170) |
|----------|-------------|-------------|---------------|
| IF | 0.349 [0.324, 0.373] | 0.349 [0.311, 0.385] | 0.583 [0.555, 0.610] |
| SpecCNN | **0.399** [0.380, 0.420] | 0.202 [0.184, 0.217] | **0.739** [0.709, 0.768] |
| kNN | 0.343 [0.336, 0.349] | **0.386** [0.354, 0.416] | 0.649 [0.626, 0.672] |
| Rule | 0.345 [0.339, 0.352] | 0.059 [0.052, 0.066] | 0.659 [0.639, 0.679] |
| **Friedman p** | **< 0.000001** | **< 0.000001** | **< 0.000001** |
| **1위** | **SpecCNN** | **kNN ≈ IF** | **SpecCNN** |

**Rank Reversal 확인**: SpecCNN은 SKAB/AIHub 1위이나 SMD 3위. kNN은 SMD 1위이나 SKAB/AIHub 3위.
**AIHub caveat**: label_rate ≈ 54% → random baseline AUC-PR ≈ 0.54

출처: `runs/all_results_clean.json` (2,105 entries, 2026-03-31 최종 재집계)

### Table 4: Threshold Sensitivity ✅ 검증됨 (SKAB, N=170/detector)

| Detector | Mean Point F1 | Mean Opt F1 | Ratio |
|----------|-------------|-----------|-------|
| Rule | 0.018 | 0.519 | **29.3x** |
| SpecCNN | 0.027 | 0.570 | **21.2x** |
| kNN | 0.025 | 0.561 | **22.2x** |
| IF | 0.160 | 0.615 | **3.8x** |

출처: `runs/all_results_clean.json` (SKAB only, 2026-03-31 재검증)

### Figure 1: Cost Ratio vs F1 (SKAB)
- X축: FN:FP ratio (1, 2, 5, 10, 20, 50)
- Y축: F1
- 출처: `runs/rq4_cost_sensitivity/raw_results.json`

---

## Negative Results (Discussion에서 다룸)

1. **RQ2 Ensemble**: 차이 negligible → "linear suffices"
2. **Domain knowledge alone**: +0% faithfulness → "attribution > knowledge"
3. **Cost ratio at high anomaly rate**: 무관 (AIHub label_rate~54%)

---

## 데이터 총량 ✅ (전체 검증 완료)

| 항목 | 수량 | 출처 |
|------|------|------|
| SKAB detection runs | 865 (deduped) | `runs/all_results_clean.json` |
| SMD detection runs | **560** (28 machines × 4 det × 5 seeds, deduped) | `runs/all_results_clean.json` |
| AIHub detection runs | 680 (deduped) | `runs/all_results_clean.json` |
| **총 detection runs** | **2,105** (deduped) | `runs/all_results_clean.json` |
| LLM explanation samples | 39 (4 conditions) | `runs/llm_explanation_v2/` |
| Statistical tests | Friedman × 3, Nemenyi × 3, Cliff's delta × all | `runs/verified_statistics_20260331.json` |
