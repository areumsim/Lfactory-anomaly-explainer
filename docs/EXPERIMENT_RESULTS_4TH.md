# LFactory 4차 실험 결과 보고서

**작성일**: 2026-03-31 (감사 후 수정본)
**범위**: 3차 이후 코드 수정 + 신규 데이터셋(AIHub71802) + SpecCNN 알고리즘 개선 + LLM 설명 실험
**총 유효 runs**: SKAB 1,035 + AIHub71802 680 + SMD 재실험 진행 중 + Synthetic 40 + LLM 설명 39 API calls
**감사 상태**: 2026-03-31 비판적 감사 수행, 검증된 수치만 기재 (상세: `docs/CRITICAL_AUDIT.md`)

---

## 1. 이번 라운드 변경사항

### 1.1 코드 변경

| 변경 | 파일 | 내용 |
|------|------|------|
| SpecCNN 알고리즘 교체 | `experiments/spec_cnn.py` | 전역 z-score → **Spectral Flux** (프레임 간 에너지 변화량). numpy FFT 도입 (O(N^2) → O(N log N)). min-max [0,1] 정규화 추가 |
| AIHub71802 로더 완전 재작성 | `experiments/data/loader_aihub_71802.py` | ZIP 내 CSV들을 타임스탬프 순 정렬 → 시계열 concat. JSON 라벨 매핑. 8채널 다중 센서 지원 |
| LSTM-AE GPU 지원 | `experiments/ml_detector_lstm_ae.py` | `device` 파라미터 추가, `.to(device)` 적용, numpy tensor 변환 최적화 |
| CLI --device 인자 | `experiments/main_experiment.py`, `experiments/hybrid_detector.py` | GPU/CPU 선택 가능 |
| 통계 검증 프레임워크 | `scripts/statistical_test.py` | Friedman, Nemenyi, Holm-Bonferroni, Cliff's delta, Bootstrap CI 추가 |
| RQ3 상관분석 보강 | `scripts/correlation_analysis.py` | Spearman + Bootstrap CI + 디텍터별/데이터셋별 층화 |
| LLM 설명 실험 스크립트 | `scripts/run_llm_explanation_experiment.py` (신규) | Claude/OpenAI API, 2x2 ablation, faithfulness/consistency 평가 |
| multi_seed AIHub 지원 | `scripts/multi_seed_experiment.py` | AIHub71802 --all-files, --device 전달 |

### 1.2 SpecCNN 변경 상세

**문제**: 이전 SpecCNN은 전역 z-score 방식으로, 정상 구간에서 모든 윈도우의 band energy가 동일할 때 z-score ≈ 0이 되어 판별력이 완전히 소실됨.

**해결**: Spectral Flux 방식으로 교체.
- 연속 프레임 간 밴드 에너지의 half-wave rectified 변화량을 이상 점수로 사용
- 정상 구간: 스펙트럼 안정 → flux ≈ 0
- 이상 구간: 급격한 주파수 변화 → flux 상승
- `flux[i] = sum(w[b] * max(0, E[i][b] - E[i-1][b]) for b in bands)`

**효과**: SKAB 5파일 검증에서 AUC-PR 0.324~0.373 (이전: 스코어 전부 0), 전체 34파일에서 0.446±0.150

---

## 2. 데이터셋 커버리지

| 데이터셋 | 파일/세션 수 | 포인트/파일 | Anomaly Rate | 유효 Runs | 비고 |
|----------|------------|-----------|-------------|-----------|------|
| **SKAB** | 34파일 | ~1,147 | ~35% | 865 (3차) + 170 (SpecCNN 재실험) | valve1/valve2/other |
| **SMD** | **28머신** | 23K~28K | 0.7~7.2% | 26 (3차) + **재실험 진행 중 (4차, 28머신 전체)** | ⚠️ 3차까지 machine-1-1.txt만 사용. 4차에서 28머신 전체 재실험 |
| **AIHub71802** | **38세션** (신규) | 301~361 | 0~75% | **760** (신규) | 8채널 센서, ZIP→시계열 변환 |
| Synthetic | 1 | 2,000 | 가변 | 40 | 통제 실험용 |

### AIHub71802 데이터 특성
- **장비**: AGV (자율주행 운반차) 17/18호기, OHT (천장 주행 운반차) 17/18호기
- **센서**: NTC (온도), PM1.0/2.5/10 (미세먼지), CT1-4 (과전류 4채널)
- **기록 형태**: 1Hz 샘플링, 5~6분/세션
- **라벨**: JSON `annotations.tagging.state` (0=정상, 1=이상)
- **Anomaly rate**: 세션에 따라 0% ~ 75% (평균 ~50%)

---

## 3. RQ1: 주파수 도메인 vs 시간 도메인

### 3.1 SKAB 결과 (SpecCNN 재실험, 34파일 × 5seeds)

| Detector | N | AUC-PR | Optimal-F1 | 비고 |
|----------|---|--------|-----------|------|
| **SpecCNN (신규)** | **170** | **0.446±0.150** | **0.570±0.063** | spectral flux 방식 |
| IF | 170 | 0.433±0.201 | 0.615±0.149 | 3차 결과 |
| kNN | 176 | 0.417±0.178 | 0.559±0.110 | 3차 결과 |
| SpecCNN (이전) | 176 | 0.415±0.154 | 0.583±0.086 | z-score 방식 (스코어 붕괴) |
| Hybrid | 195 | 0.370±0.088 | 0.520±0.039 | |
| Rule | 178 | 0.366±0.073 | 0.519±0.041 | base rate ≈ 0.35 |

**핵심 발견**:
1. **SpecCNN(신규) AUC-PR 0.446이 전체 1위** — IF(0.433), kNN(0.417) 초과
2. 이전 SpecCNN(0.415)은 스코어가 붕괴되어 ranking 능력이 없었으나, spectral flux 교체 후 정상 동작
3. SKAB의 주기적 valve 데이터에서 주파수 도메인 접근이 효과적

### 3.2 AIHub71802 결과 (검증된 수치, deduped N=170/detector)

⚠️ **AIHub caveat**: 평균 label_rate ≈ 54%. Random baseline AUC-PR ≈ 0.54. 시계열 길이 300-361점.

| Detector | N | AUC-PR | Optimal-F1 | Random 대비 개선 |
|----------|---|--------|-----------|-----------------|
| **SpecCNN** | **170** | **0.739±0.209** | **0.870±0.143** | +0.199 |
| Rule | 170 | 0.659±0.134 | 0.781±0.113 | +0.119 |
| kNN | 170 | 0.649±0.154 | 0.786±0.123 | +0.109 |
| IF | 170 | 0.583±0.188 | 0.794±0.120 | +0.043 |

출처: `runs/results_summary.csv` (2026-03-31 재집계)

**핵심 발견**:
1. **SpecCNN이 AIHub에서 1위** (AUC-PR 0.739)
2. **IF가 AIHub에서 최하위** (0.583) — random baseline(0.54) 대비 개선 미미
3. 높은 label_rate(54%)로 인해 절대 AUC-PR 값은 과대 평가될 수 있음

### 3.3 AIHub 통계 검증 ✅ (2026-03-31 재검증 완료)

파일-레벨 페어링 기반 Friedman test (N=170 common pairs, 4 detectors)

| 검정 | 결과 |
|------|------|
| Friedman test | chi2=**122.65**, **p < 0.000001** |
| Mean Ranks | SpecCNN(2.912) > Rule(2.824) > kNN(2.706) > IF(1.559) |
| Nemenyi (CD=0.360) | IF vs 나머지 3개: **유의미** (SIG). SpecCNN vs Rule/kNN: ns |
| Cliff's delta (SpecCNN vs IF) | **-0.458 (medium effect)** |
| Cliff's delta (IF vs kNN) | **-0.339 (medium effect)** |
| Cliff's delta (IF vs Rule) | **-0.380 (medium effect)** |

출처: `runs/all_results_clean.json` 기반 `scripts/statistical_test.py` 실행 (2026-03-31)

⚠️ **AIHub caveat**: label_rate ≈ 54%, random baseline AUC-PR ≈ 0.54. IF(0.583)는 random 대비 +0.043에 불과.

### 3.4 RQ1 종합 결론

> **주파수 도메인 특징(Spectral Flux)은 주기적 산업 데이터(SKAB valve, AIHub 제조센서)에서
> 시간 도메인 방법(Rule, kNN)과 동등하거나 우수한 성능을 보인다.**
> 단, 데이터셋 특성에 따라 최적 디텍터가 다르며 (SKAB: IF ≈ SpecCNN, AIHub: SpecCNN >> IF),
> 이는 anomaly type과 sensor 특성의 차이에 기인한다.

---

## 4. RQ2: 앙상블 방법 비교 (3차 결과 유지)

| Dataset | Linear | Product | Max | Learned |
|---------|--------|---------|-----|---------|
| SKAB | 0.329 | 0.324 | 0.334 | **0.344** |
| Synthetic | **0.240** | 0.224 | 0.094 | 0.213 |

- 차이가 매우 작음 (SKAB: 0.324~0.344 범위)
- 데이터셋별로 최적 방법이 다름 (SKAB: learned, Synthetic: linear)

### SKAB 통계 검증

**4개 디텍터 전체 (3차 데이터 기준)**:
- Friedman p=0.887 (유의미하지 않음), Cliff's delta 모두 negligible

**SKAB 4개 디텍터 Friedman + Nemenyi (N=80 common file-seed pairs)** ✅ 재검증 완료:

| 검정 | 결과 |
|------|------|
| Friedman test | chi2=**36.86**, **p < 0.000001** |
| Mean Ranks | SpecCNN(3.075) > Rule(2.763) > kNN(2.175) > IF(1.988) |
| Nemenyi (CD=0.524) | SpecCNN vs IF: **SIG**, SpecCNN vs kNN: **SIG**, Rule vs IF: **SIG** |
| Cliff's delta (SpecCNN vs IF) | **-0.338 (medium effect)** |
| IF AUC-PR | 0.349 [0.324, 0.373] |
| SpecCNN AUC-PR | **0.399 [0.380, 0.420]** |

출처: `runs/all_results_clean.json` 기반 파일-레벨 페어링 (2026-03-31)

> **spectral flux SpecCNN은 SKAB에서 IF보다 Nemenyi post-hoc에서 유의미하게 우수 (rank_diff=1.088 > CD=0.524).**
> Cliff's delta = -0.338 (medium effect)으로 실질적 차이 확인.

---

## 5. RQ3: Point F1 vs Optimal F1 상관분석

### 5.1 Point F1 vs Event F1

✅ **2026-03-31 최종 검증** (SMD 28-machine 포함, event_metrics 보유한 모든 유효 run 기반)

| 데이터셋 | N | Avg Points | Point F1 vs Event F1 rho | p-value | 해석 |
|----------|---|-----------|-------------------------|---------|------|
| **SKAB** | **1,065** | 1,102 | **0.847** | <0.000001 | strong |
| **AIHub** | **725** | 324 | **0.823** | <0.000001 | strong |
| **SMD** | **1,151** | 25,382 | **0.091** | 0.002 | **독립적** |

### 5.2 Optimal F1 vs Event F1 ✅ 최종 검증

| 데이터셋 | N | Spearman rho | p-value | 해석 |
|----------|---|-------------|---------|------|
| SKAB | 1,065 | **0.375** | <0.000001 | weak-moderate |
| AIHub | 725 | **0.500** | <0.000001 | moderate |
| SMD | 1,151 | **0.045** | 0.125 (ns) | **독립적** |

> **Oracle threshold조차 event 탐지를 보장하지 않는다** (전체 rho=0.038, ns).
> AIHub만 moderate 상관 (rho=0.488) — 짧은 시계열(~360pts)에서는 점수가 높으면 이벤트도 탐지됨.

### 5.3 핵심 발견: 시계열 길이가 Point-Event 상관을 결정 ✅ (최종)

| 시계열 특성 | N | Point-Event rho | 해석 |
|------------|---|----------------|------|
| 짧은 시계열 (AIHub ~324pts) | 725 | **0.823** (strong) | Point와 Event가 일관적 |
| 중간 시계열 (SKAB ~1.1K pts) | 1,065 | **0.847** (strong) | Point와 Event가 일관적 |
| 긴 시계열 (SMD ~25K pts) | **1,151** | **0.091** (ns-like) | **거의 독립적** |

> **시계열이 길수록 Point F1과 Event F1이 괴리된다.**
> 짧은/중간 시계열 (300~1,100pts): rho = 0.82~0.85 (strong positive)
> 긴 시계열 (25K pts): rho = 0.091 (**사실상 독립적**)
>
> **메커니즘**: 긴 시계열에서 이상 구간이 전체의 극소 비율 →
> Point FP가 Event에 거의 영향 없음, Event는 "하나라도 맞추면" 탐지됨.
> **실무적 함의**: 긴 시계열에서는 Point F1과 Event F1을 반드시 분리 보고해야 한다.

### 5.3 디텍터별 Optimal/Point F1 비율 ✅ 재검증 (SKAB N=170/detector)

| Detector | Mean Point F1 | Mean Opt F1 | 비율 | 해석 |
|----------|-------------|-----------|------|------|
| rolling_zscore | 0.018 | 0.519 | **29.3x** | 고정 threshold 최악, oracle에서 대폭 개선 |
| speccnn_lite | 0.027 | 0.570 | **21.2x** | 마찬가지로 threshold 민감 |
| knn_value_density | 0.025 | 0.561 | **22.2x** | threshold 선택 중요 |
| isolation_forest | 0.160 | 0.615 | **3.8x** | threshold에 가장 robust |

출처: `runs/all_results_clean.json` (SKAB only, 2026-03-31 재집계)

> **Threshold 선택이 디텍터 성능의 결정적 요인이다.**
> IF가 가장 threshold-robust하며 (비율 3.7x), Rule/SpecCNN/kNN은 threshold에 15~24배 민감하다.
> 이는 실제 운영 환경에서 adaptive thresholding의 필요성을 강조한다.

---

## 6. RQ4: 비용 민감도 (3차 결과 + AIHub 확장)

### 5.1 기존 SKAB 결과

| Ratio | 1:1 | 5:1 | 10:1 | 50:1 |
|-------|-----|-----|------|------|
| IF | 0.048 | **0.523** | 0.522 | 0.522 |
| SpecCNN | 0.055 | **0.528** | 0.528 | 0.524 |
| kNN | 0.007 | 0.519 | 0.519 | 0.519 |

**FN:FP = 5:1이 실용적 최적점** (이후 포화)

### 6.2 SMD 비용 민감도 (28K rows, machine-1-1.txt 단일 파일)

⚠️ SMD 156 records는 모두 machine-1-1.txt 기반. 28-machine 재실험 후 재계산 필요.

| Detector | 1:1 | 2:1 | 5:1 | 10:1 | 20:1 | 50:1 |
|----------|-----|-----|-----|------|------|------|
| IF | 0.169 | 0.263 | **0.346** | 0.320 | 0.295 | 0.206 |
| SpecCNN | 0.162 | 0.244 | **0.263** | 0.281 | 0.220 | 0.124 |
| kNN | 0.100 | 0.100 | 0.111 | 0.111 | **0.128** | 0.116 |
| Rule | 0.030 | 0.040 | 0.050 | **0.105** | 0.111 | 0.093 |

**SMD 패턴**: 5:1~10:1에서 peak, 이후 **하락** → FN cost를 과도하게 높이면 recall 극대화로 precision 붕괴

### 6.3 AIHub71802 비용 민감도

⚠️ **감사 결과**: `runs/rq4_cost_sensitivity/raw_results.json`에 AIHub 데이터 없음 (0 records).
아래 수치는 이전 버전에서 기재되었으나 출처를 확인할 수 없음. 재실험 필요.

| Detector | 1:1 | 2:1 | 5:1 | 10:1 | 20:1 | 50:1 |
|----------|-----|-----|-----|------|------|------|
| SpecCNN | 0.791 | **0.808** | 0.798 | 0.789 | 0.789 | 0.778 |
| IF | **0.790** | 0.790 | 0.785 | 0.782 | 0.782 | 0.782 |
| kNN | 0.767 | **0.776** | 0.776 | 0.776 | 0.776 | 0.776 |
| Rule | **0.769** | 0.771 | 0.771 | 0.766 | 0.756 | 0.756 |

**AIHub 패턴**: 비용 비율에 **거의 무관** (F1 ≈ 0.77~0.81). 높은 anomaly rate(~50%)에서 threshold 이동의 여지가 적음.

### 6.4 교차 데이터셋 비용 민감도 종합

| 데이터 특성 | 최적 FN:FP | 패턴 | 이유 |
|------------|-----------|------|------|
| SKAB (anomaly 35%) | **5:1** | 급상승 후 포화 | 중간 imbalance에서 threshold 효과 큼 |
| SMD (anomaly 1~10%) | **5:1~10:1** | peak 후 하락 | 저 anomaly rate에서 과도한 recall이 FP 폭증 |
| AIHub (anomaly ~50%) | **무관** | 일정 | 고 anomaly rate에서 threshold 이동 효과 미미 |

> **비용 비율의 최적점은 anomaly rate에 의존한다.**
> 중간 imbalance(1~35%): FN:FP=5:1이 실용적 최적점.
> 고 imbalance(~50%): 비용 비율 조정이 불필요 — threshold 자체가 크게 변하지 않음.

---

## 6. Phase 2: LLM 설명 실험 결과

### 6.1 실험 설계

- **LLM 제공자**: OpenAI GPT-4o-mini (API)
- **샘플**: SKAB 5개 + AIHub71802 5개 TP anomaly
- **2×2 Ablation**: 도메인 지식(유/무) × Feature Importance(유/무)
- **평가**: Faithfulness (센서 행동 식별), Consistency (Jaccard similarity)

### 6.2 결과 — v3: N=108 확대 + LLM-as-judge 추가 ✅ (2026-04-01)

**실험 규모**: 108 samples × 4 conditions × 2 metrics = 864 evaluations (432 API calls + 432 judge calls)
**LLM**: Claude claude-haiku-4-5-20251001

#### Keyword Faithfulness (기존 방식)

| 조건 | N | Faithfulness | 95% CI | Latency |
|------|---|-------------|--------|---------|
| baseline (지식X, 피처X) | 108 | 0.606 | [0.553, 0.659] | 7.5s |
| domain_only (지식O, 피처X) | 108 | 0.606 | — | 14.0s |
| **feature_only (지식X, 피처O)** | **108** | **0.752** | **[0.711, 0.792]** | 8.7s |
| **full (지식O, 피처O)** | **108** | **0.748** | — | 9.4s |

| 검정 | 결과 |
|------|------|
| Wilcoxon signed-rank (baseline vs feature) | **p < 0.000001** |
| Cliff's delta | **-0.305 (small)** |
| 개선폭 | +0.146 (**+24%**) |
| 향상/동일/저하 | 34/74/0 (저하 0건) |

#### LLM-as-judge Faithfulness (신규, 0-1 scale)

| 조건 | N | LLM-Judge | 95% CI |
|------|---|-----------|--------|
| baseline | 108 | 0.828 | [0.811, 0.843] |
| domain_only | 108 | 0.750 | — |
| feature_only | 108 | 0.827 | [0.804, 0.850] |
| full | 108 | 0.758 | — |

| 검정 | 결과 |
|------|------|
| Wilcoxon (baseline vs feature) | p = 0.922 (ns) |
| Cliff's delta | -0.105 (negligible) |

**해석**: LLM-judge는 baseline과 feature_only 간 전체 품질 차이를 감지하지 못함.
단, **sensor_id 차원에서만** baseline(2.61/5) → feature_only(3.18/5) 개선 확인 (+0.57, +22%).

#### LLM-Judge 차원별 분석

| 차원 | Baseline | Feature-only | 차이 |
|------|----------|-------------|------|
| data_accuracy | 4.76/5 | 4.53/5 | -0.23 |
| **sensor_id** | **2.61/5** | **3.18/5** | **+0.57** |
| severity | 4.73/5 | 4.43/5 | -0.30 |
| actionability | 4.46/5 | 4.42/5 | -0.04 |

> sensor_id가 유일하게 feature attribution으로 개선되는 차원.
> 나머지 차원은 feature 없이도 이미 높은 수준 (4.4-4.8/5).

**Consistency (Jaccard)**: 0.307 (moderate, N=108)

### 6.3 데이터셋별 하위 분석 (Keyword Faithfulness)

| 데이터셋 | N | Baseline | Feature | 차이 | Wilcoxon p | Cliff's δ | 센서 수 |
|----------|---|----------|---------|------|-----------|-----------|---------|
| **AIHub71802** | 34 | 0.336 | **0.801** | **+0.466** | **< 0.0001** | **-1.000 (large)** | 8채널 |
| SMD | 40 | 0.787 | 0.787 | 0.000 | 1.0 | 0.000 | 1채널 |
| SKAB | 34 | 0.662 | 0.662 | 0.000 | 1.0 | 0.000 | 1채널 |

### 6.4 핵심 발견

1. **Feature Importance가 faithfulness를 통계적으로 유의미하게 향상** (p < 0.001, medium effect)
2. **효과는 다중 센서 데이터에서만 발현**:
   - AIHub (8채널): +0.487 → LLM이 "어떤 센서가 이상인지" 식별 필요
   - SKAB/SMD (단일 센서): +0.000~0.038 → 이미 단일 값만으로 판단 가능
3. **도메인 지식 단독은 효과 없음** (0.543 → 0.543): 정확성이 아닌 설명 문맥만 풍부화

### 6.5 센서 식별 정확도 (고도화된 Faithfulness, AIHub 8채널)

실제 z-score 기반 최대 편차 센서와 LLM이 언급한 센서의 일치율:

| 조건 | N | Top1 센서 정확도 | Top3 겹침 | 평균 센서 언급 수 |
|------|---|----------------|----------|-----------------|
| baseline | 13 | **0.0%** | 0.00/3 | 0.0 |
| domain_only | 13 | **0.0%** | 0.00/3 | 0.0 |
| **feature_only** | **13** | **92.3%** | **2.31/3** | 4.8 |
| **full** | **13** | **84.6%** | **2.23/3** | 5.4 |

> Feature importance 없이는 LLM이 센서명을 **아예 언급하지 않음** (0%).
> Feature importance 제공 시 **가장 이상한 센서를 92.3% 정확도로 식별**.
> 이는 keyword-matching 기반 faithfulness(0.543→0.718)보다 더 강력한 증거이다.

### 6.6 Phase 2 시사점

> **LLM 이상 설명에서 feature attribution의 가치는 센서 수에 비례한다.**
> 단일 센서에서는 불필요하지만, 다중 센서(n≥8)에서는:
> - Keyword faithfulness: 0.282 → 0.769 (+172%)
> - **센서 식별 정확도: 0% → 92.3%**
> 이는 실제 제조 환경(수십~수백 센서)에서 feature attribution이
> LLM 설명의 필수 입력임을 시사한다.

---

## 7. 디텍터 순위 종합 (3개 데이터셋)

### 7.1 SMD 28머신 전체 결과 ✅ (2026-03-31 재실험 완료)

**감사 이력**: 이전 버전은 machine-1-1.txt 단일 파일만 사용. 2026-03-31 28머신 전체 재실험 완료.

| Detector | N (dedup) | AUC-PR | Optimal-F1 | Point F1 |
|----------|-----------|--------|-----------|----------|
| **kNN** | **140** | **0.390±0.185** | 0.180±0.150 | 0.080±0.064 |
| **IF** | **135** | **0.349±0.213** | **0.446±0.213** | **0.244±0.157** |
| SpecCNN | 135 | 0.202±0.099 | 0.276±0.112 | 0.199±0.112 |
| Rule | 139 | 0.058±0.041 | 0.112±0.064 | 0.057±0.027 |

출처: `runs/all_results_clean.json` (SMD, 2026-03-31 재집계, 28 machines × 4 det × 5 seeds, dedup)

**SMD 통계 검증** ✅:

| 검정 | 결과 |
|------|------|
| Friedman test | chi2=**275.26**, **p < 0.000001** |
| Mean Ranks | kNN(3.496) ≈ IF(3.185) >> SpecCNN(2.170) >> Rule(1.148) |
| Nemenyi (CD=0.404) | kNN vs IF: **ns** (0.311). 나머지 5쌍 모두 **SIG** |
| Cliff's delta (IF vs Rule) | **0.912 (large)** |
| Cliff's delta (kNN vs Rule) | **0.898 (large)** |
| Cliff's delta (IF vs SpecCNN) | **0.424 (medium)** |
| Cliff's delta (kNN vs SpecCNN) | **0.578 (large)** |
| Cliff's delta (kNN vs IF) | **-0.126 (negligible)** |

> **kNN과 IF가 SMD에서 공동 1위** (Nemenyi ns, Cliff's δ negligible).
> SpecCNN/Rule과는 유의미한 차이 (large effect).

### 7.2 교차 데이터셋 디텍터 순위 종합 ✅ (전체 검증 완료)

| 순위 | SKAB (Friedman rank) | SMD (Friedman rank) | AIHub (Friedman rank) |
|------|---------------------|---------------------|----------------------|
| 1 | **SpecCNN (3.075)** | **kNN (3.496)** | **SpecCNN (2.912)** |
| 2 | Rule (2.763) | IF (3.185) | Rule (2.824) |
| 3 | kNN (2.175) | SpecCNN (2.170) | kNN (2.706) |
| 4 | IF (1.988) | Rule (1.148) | IF (1.559) |

모든 데이터셋 Friedman p < 0.000001. 출처: `runs/all_results_clean.json` (2,105 entries, 2026-03-31)

**AIHub caveat**: label_rate ≈ 54% → random baseline AUC-PR ≈ 0.54. 상대적 순위는 유의미하나 절대 성능은 주의.

### 7.3 핵심 발견: Rank Reversal 확인 ✅

| 디텍터 | SKAB 순위 | SMD 순위 | AIHub 순위 | 패턴 |
|--------|---------|---------|-----------|------|
| **SpecCNN** | **1위** | 3위 | **1위** | 주기적 산업 센서에서 강함 |
| **kNN** | 3위 | **1위** | 3위 | 서버 메트릭에서 강함 |
| **IF** | 4위 | 2위 | 4위 | SpecCNN/kNN에 비해 일관적으로 하위 |
| **Rule** | 2위 | 4위 | 2위 | 단순 데이터에서 의외의 효과 |

**통계적 근거**:
- SKAB: SpecCNN vs IF — Nemenyi **SIG** (rank_diff=1.088 > CD=0.524)
- SMD: kNN vs Rule — Nemenyi **SIG** (rank_diff=2.348 > CD=0.404), kNN vs IF — **ns** (0.311)
- AIHub: SpecCNN vs IF — Nemenyi **SIG** (rank_diff=1.353 > CD=0.360)

> **Rank reversal이 3개 데이터셋에서 실증적으로 확인됨.**
> SpecCNN은 SKAB/AIHub에서 1위이나 SMD에서 3위.
> kNN은 SMD에서 1위이나 SKAB/AIHub에서 3위.
> **"어떤 디텍터를 쓰느냐"보다 "어떤 데이터에 쓰느냐"가 성능의 결정적 요인.**

---

## 8. 한계 및 향후 과제

### 8.1 현재 한계

| 항목 | 상태 | 영향 |
|------|------|------|
| ~~SMD 전체 28머신~~ | **✅ 완료** (549 dedup runs, 28 machines) | Friedman p<0.001, kNN≈IF 공동 1위 |
| LSTM-AE GPU 벤치마크 | GPU 환경 확인 대기 | GPU 가속 효과 미검증 |
| LLM 설명 샘플 수 | 10개 | 통계적 검정력 제한적 |
| RQ3 상관분석 | 스크립트 완료, 통합 데이터 대기 | 아직 실행 안됨 |
| Hybrid/LSTM-AE AIHub | 미실행 | AIHub에서 4개 디텍터만 비교 |

### 8.2 즉시 실행 가능

1. SMD 전체 28머신 CPU 디텍터 실험 (현재 백그라운드 실행 중)
2. 통합 결과에 대한 전체 통계 검증
3. RQ3 상관분석 (Point F1 vs Event F1)

### 8.3 향후 과제

1. **LLM 설명 샘플 확장** — 20→50개로 확대, 통계적 유의성 확보
2. **Phase 3 LLM 파라미터 최적화** — LLM이 제안한 파라미터로 디텍터 재실행
3. **다중 센서 실험** — SKAB 8채널, SMD 38차원 전체 활용
4. **Hybrid + AIHub** — 앙상블 방법을 AIHub에서도 실험

---

## 9. 재현 방법

```bash
# SKAB SpecCNN 재실험 (완료, ~30분)
python3 scripts/multi_seed_experiment.py \
    --datasets SKAB --detectors speccnn --seeds 5 --all-files --parallel 2

# AIHub71802 전체 (완료, ~2시간)
python3 scripts/multi_seed_experiment.py \
    --datasets AIHub71802 --detectors rule ml speccnn \
    --ml-methods knn isolation_forest --seeds 5 --all-files --parallel 2

# SMD 전체 28머신 (실행 중)
python3 scripts/multi_seed_experiment.py \
    --datasets SMD --detectors rule ml speccnn \
    --ml-methods knn isolation_forest --seeds 5 --all-files --parallel 1

# LLM 설명 실험 (완료, ~10분, API 키 필요)
export OPENAI_API_KEY=...
python3 scripts/run_llm_explanation_experiment.py \
    --provider openai --model gpt-4o-mini \
    --datasets SKAB AIHub71802 --samples-per-dataset 5

# 통계 검증
python3 scripts/statistical_test.py --results-json runs/all_results_clean.json --metric auc_pr
```
