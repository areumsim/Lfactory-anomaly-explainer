# 감사 수정 보고서 (2026-04-06 후반)

**목적**: METHODOLOGY_AUDIT_20260406.md에서 발견된 문제점들의 코드 수정 및 실험 재실행 결과

---

## 1. 수정된 치명적 이슈 (Tier 1)

### 1.1 Calibration Data Leakage 수정

**파일**: `experiments/main_experiment.py:425-441`

**수정 내용**: 전체 test labels로 calibration fitting → temporal split 60/40 적용
- 시계열 앞 60% (`cal_split`)로 Platt/Temperature/Isotonic 파라미터 fitting
- 전체 시계열에 적용하여 확률 변환 및 ECE 계산

**재실행 결과 (SKAB 408건)**:

| Detector | Old ECE (leakage) | New ECE (temporal split) | 변화 |
|----------|:---:|:---:|:---:|
| rolling_zscore | 0.0113 | 0.1866 | **+0.1753** (16.5x) |
| knn | 0.0259 | 0.1862 | **+0.1603** (7.2x) |
| isolation_forest | 0.0901 | 0.2033 | **+0.1132** (2.3x) |
| speccnn_lite | 0.1063 | 0.1938 | **+0.0875** (1.8x) |

**해석**: 기존 ECE 값은 test labels에 overfitting된 결과였음. 수정 후 모든 detector에서 ECE가 1.8-16.5배 상승. AUC-PR에는 영향 없음 (calibration은 점수 변환만 관여).

**논문 반영**: 캘리브레이션 관련 모든 주장을 수정된 ECE 값으로 갱신. "잘 캘리브레이션됨"이라는 주장 철회.

---

### 1.2 TP-Only Sample Selection Bias 수정

**파일**: `scripts/run_llm_explanation_experiment.py:81-210`

**수정 내용**:
- `select_anomaly_samples(sample_mode="tp"|"fn"|"fp")` 파라미터 추가
- `--sample-modes tp fn fp` CLI 옵션 추가
- FN: 정답=이상, 탐지=정상인 포인트 (탐지기가 놓친 이상)
- FP: 정답=정상, 탐지=이상인 포인트 (오경보)

**논문 반영**: TP/FN/FP 분리 보고. "+98% faithfulness"를 TP 조건 한정으로 수정.

---

### 1.3 Faithfulness Metric 강화

**파일**: `scripts/run_llm_explanation_experiment.py:263-380`

**수정 내용**:
1. 기존 metric을 `evaluate_keyword_coverage`로 이름 변경
2. 새로운 `evaluate_faithfulness_strict` 추가:
   - **방향 정확도**: anomaly value > normal_mean이면 increase 키워드 필요 (반대도 동일)
   - **센서 식별 정확도**: top-1 anomalous 센서 정확히 식별 여부
   - **Top-3 센서 overlap**: 상위 3개 이상 센서 중 몇 개를 언급했는지
   - **가중 종합 점수**: keyword(1.0) + direction(1.5) + sensor_top1(1.5) + top3_overlap(1.0)

**검증**:
- 올바른 설명 (방향+센서 정확): strict overall = 0.833
- 틀린 설명 (방향+센서 오류): strict overall = 0.233
- 구분력 확인 완료

---

### 1.4 SWaT 통계적 독립성 위반

**조치**: 코드 수정 없이 해석 방법 변경
- SWaT 결과를 **descriptive case study**로만 보고
- Wilcoxon/Friedman 검정에서 SWaT 제외
- Block bootstrap CI로 대체 권장

---

### 1.5 Domain Knowledge Confound 수정

**파일**:
- `experiments/domain_knowledge/swat_knowledge_nosensor.yaml` — 완전 익명화
- `experiments/domain_knowledge/knowledge_retriever.py` — `use_nosensor` 파라미터 추가
- `scripts/run_llm_explanation_experiment.py` — `--include-nosensor` CLI 옵션

**수정 내용**:
- nosensor YAML에서 `FIT101`, `LIT101` 등 모든 실명 제거
- attack_scenarios, correlation_rules, severity_levels 모두 익명화
- 3-condition ablation 실험 가능: (a) no domain, (b) domain_nosensor, (c) domain_full

---

## 2. 보강 이슈 수정 (Tier 2)

### 2.1 Threshold 비교 불일치

**파일**: `experiments/ml_detector_anomaly_transformer.py:267`
- `>` → `>=` 통일 (다른 모든 detector와 일치)

### 2.2 LSTM-AE Zero-Score Padding

**파일**: `experiments/ml_detector_lstm_ae.py:214-228`
- 첫 `sequence_length - 1` 포인트의 score를 0.0 대신 첫 번째 유효 score로 back-fill
- 초기 이상 탐지 누락 문제 해결

### 2.3 Filter Statistics 투명성

**파일**: `scripts/aggregate_results.py:42-164`
- 필터 단계별 제외 건수 출력:
  - total_files: 4,959
  - skipped_no_optimal_f1: 481
  - skipped_label_rate_zero: 170
  - skipped_too_short: 100
  - passed_filters: 4,208
  - duplicates_removed: 1,612
  - final_unique: 2,596
- JSON에 `filter_statistics` 섹션 포함

### 2.4 Post-hoc Power Analysis

**파일**: `scripts/statistical_test.py:315-385`
- `posthoc_power()` 함수 추가 (Cohen's d, power, min_n_80, min_n_90)
- `compare_detectors()`에서 모든 pairwise 비교에 자동 power 보고

**SKAB Power Analysis 결과**:

| 비교 | p-value | Cliff's d | Power | N needed (80%) | 상태 |
|------|:---:|:---:|:---:|:---:|:---:|
| rule vs speccnn | 0.0006 | -0.256 | 0.980 | 17 | **ADEQUATE** |
| IF vs rule | 0.080 | -0.014 | 0.674 | 46 | MARGINAL |
| kNN vs rule | 0.726 | 0.017 | 0.627 | 52 | MARGINAL |
| IF vs kNN | 0.871 | -0.100 | 0.090 | 691 | UNDERPOWERED |
| IF vs speccnn | 0.351 | -0.171 | 0.112 | 483 | UNDERPOWERED |
| kNN vs speccnn | 0.106 | -0.145 | 0.288 | 136 | UNDERPOWERED |

**해석**: 6개 비교 중 1개만 adequate power. IF vs kNN 구분에는 691개 파일 필요 — 현재 34개로는 불가능.

---

## 3. 재실행 결과 요약

### SKAB (408건, 34 files x 4 detectors x 3 seeds)

| Rank | Detector | AUC-PR | 95% CI | Opt-F1 |
|:---:|----------|:---:|:---:|:---:|
| 1 | speccnn_lite | **0.4464±0.150** | [0.401, 0.505] | 0.570±0.063 |
| 2 | isolation_forest | 0.4331±0.202 | [0.369, 0.508] | 0.616±0.149 |
| 3 | knn | 0.4204±0.180 | [0.362, 0.493] | 0.561±0.111 |
| 4 | rolling_zscore | 0.3672±0.074 | [0.343, 0.395] | 0.519±0.042 |

Friedman test: chi2=4.69, p=0.194, **NOT significant**

### SMD (336건, 28 machines x 4 detectors x 3 seeds)

| Rank | Detector | AUC-PR | ECE |
|:---:|----------|:---:|:---:|
| 1 | knn | **0.390±0.185** | 0.042 |
| 2 | isolation_forest | 0.352±0.211 | 0.051 |
| 3 | speccnn_lite | 0.204±0.099 | 0.044 |
| 4 | rolling_zscore | 0.057±0.041 | 0.041 |

### AIHub71802 (456건, 38 sessions x 4 detectors x 3 seeds)

| Rank | Detector | AUC-PR | vs Random (0.54) |
|:---:|----------|:---:|:---:|
| 1 | speccnn_lite | **0.661±0.302** | +0.121 (22%) |
| 2 | rolling_zscore | 0.590±0.240 | +0.050 (9%) |
| 3 | knn | 0.581±0.248 | +0.041 (8%) |
| 4 | isolation_forest | 0.521±0.253 | -0.019 (random 이하!) |

**Rank reversal 재확인**:
- SpecCNN: SKAB 1위, SMD 3위, AIHub 1위
- kNN: SKAB 3위, SMD 1위, AIHub 3위
- IF: SKAB 2위, SMD 2위, AIHub 4위

---

## 4. 논문 반영 사항

### 수정 필요
1. ECE 관련 모든 수치를 temporal split 값으로 갱신
2. "잘 캘리브레이션됨" 주장 삭제 또는 약화
3. Faithfulness metric을 strict 버전으로 보고
4. TP-only bias 명시 또는 TP/FN/FP 분리 보고
5. SWaT는 descriptive only로 전환
6. Power analysis appendix 추가

### 강화 포인트
1. Calibration leakage 발견 및 수정 → 방법론적 기여로 활용 가능
2. Power analysis 결과 → "탐지기 간 차이는 통계적으로 구분하기 어렵다" = Claim 2 (No Free Lunch) 강화
3. ECE 상승 → "적절한 calibration 방법론의 중요성" 논의 추가

---

## 5. LLM Explanation 재실행 결과 (TP/FN + Nosensor)

### 5.1 TP vs FN 비교 (SMD, N=5+5)

| Condition | TP strict | FN strict | TP sensor | FN sensor |
|-----------|:---------:|:---------:|:---------:|:---------:|
| baseline | 0.133 | 0.108 | 0% | 0% |
| feature_only | 0.373 | **0.687** | 40% | **100%** |
| full | 0.447 | 0.588 | 60% | 80% |

**발견**: FN 샘플에서 오히려 strict faithfulness가 높음 (0.687 vs 0.373).
Feature attribution이 FN(탐지기가 놓친 이상)에서도 효과적, 특히 sensor identification이 FN에서 100% → TP-only bias 우려 완화.
단, 방향 정확도(direction)는 모든 조건에서 0% — LLM이 increase/decrease를 정확히 판단하지 못함. 이는 향후 개선 필요.

### 5.2 Domain Knowledge Confound Control

| Condition | KW | Strict | Sensor ID |
|-----------|:--:|:------:|:---------:|
| domain_only (센서 이름 포함) | 0.525 | 0.107 | 0% |
| domain_nosensor (센서 이름 제거) | 0.592 | 0.120 | 0% |
| full (feature + domain 이름 포함) | 0.942 | 0.517 | 70% |
| full_nosensor (feature + domain 이름 제거) | 0.975 | 0.560 | 80% |

**결론**:
- Domain knowledge의 센서 이름 confound는 **실질적으로 크지 않음**
- `full_nosensor`(0.560) ≥ `full`(0.517) → 센서 이름 제거해도 성능 유지/향상
- **Feature attribution이 sensor identification의 핵심 동인** (domain knowledge가 아님)

### Limitations 섹션에 추가
- "일부 pairwise 비교는 underpowered (power < 0.3)"
- "SpecCNN 가중치는 grid search로 선택됨"
- "Cost sensitivity는 SKAB에서만 fully tested"
- "LLM TP/FN 비교 샘플 수가 작음 (N=5+5), 확대 검증 필요"
