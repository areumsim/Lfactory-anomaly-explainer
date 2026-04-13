# Phase 2: LLM 해석 시스템 완료 보고서

**완료일**: 2025-11-25
**상태**: ✅ 완료

---

## 1. 개요

Phase 2에서는 ML 이상 탐지 결과를 LLM이 해석하는 시스템을 구축했습니다.

### 1.1 사용 LLM 모델

| 항목 | 설정 |
|------|------|
| **모델** | GPT-4o-mini (OpenAI) |
| **Temperature** | 0.3 (사실적 응답을 위해 낮게 설정) |
| **Max Tokens** | 2000 |
| **API** | OpenAI Chat Completions API |

### 1.2 핵심 성과
- **4-in-1 설명 시스템** 구현
- **12개 설명** 배치 생성
- **평균 품질 93.9/100** (EXCELLENT)

---

## 2. 구현된 모듈

### 2.1 Feature Importance (Phase 2.3)
**파일**: `experiments/feature_importance.py`

```python
# 주요 함수
compute_reconstruction_importance()   # 전역 센서 중요도
compute_anomaly_point_attribution()   # 특정 이상점 기여도
format_importance_for_llm()           # LLM 프롬프트 형식화
```

**기능**:
- 8개 센서별 기여도 계산
- Reconstruction error 기반 attribution
- Z-score 기반 fallback 분석

### 2.2 Domain Knowledge (Phase 2.4)
**파일**: `experiments/domain_knowledge/`

```
domain_knowledge/
├── __init__.py
├── knowledge_retriever.py    # 지식 검색 시스템
├── skab_knowledge.yaml       # SKAB 도메인 지식
└── smd_knowledge.yaml        # SMD 도메인 지식
```

**SKAB 지식 내용**:
- 8개 센서 상세 스펙 (정상 범위, 이상 지표)
- 센서별 failure modes (3개 패턴/센서)
- 센서 간 correlation rules (4개)
- 심각도 레벨 정의 (Critical/High/Medium/Low)

### 2.3 통합 Explainer
**파일**: `experiments/llm_explainer.py`

```python
class AnomalyExplainer:
    def explain_anomaly(
        run_dir,
        anomaly_idx,
        include_model_interpretation=True,
        include_domain_knowledge=True,
        include_feature_importance=True,
        multi_sensor_data=None,
        reconstruction_errors=None,
        sensor_names=None
    )
```

---

## 3. 설명 생성 결과

### 3.1 배치 생성 (Phase 2.5)
- **총 12개** 설명 생성
- **데이터셋**: SKAB valve1
- **탐지기**: LSTM Autoencoder (3 seeds × 4 anomalies)

### 3.2 품질 평가 (Phase 2.6)

| 메트릭 | 평균 | 범위 |
|--------|------|------|
| **Overall** | **93.9** | 90.0-95.0 |
| Completeness | 100.0 | 100-100 |
| Technical Accuracy | 76.7 | 60-80 |
| Actionability | 100.0 | 100-100 |
| Domain Relevance | 100.0 | 100-100 |
| Feature Importance | 97.9 | 75-100 |

**품질 등급**: EXCELLENT

---

## 4. 설명 예시

### 예시 1: SKAB valve1 (Index 467)

```
### Anomaly Explanation for Valve1

#### 1. Anomalous Behavior Observed
At time index 467, the sensor data from the accelerometer (Accelerometer1RMS)
recorded a value of 0.024167 g, which is below the normal mean of 0.026204 g.
The anomaly detection model flagged this with a score of 0.998112, indicating
a significant deviation from expected behavior.

#### 2. Why the ML Model Flagged It
The LSTM autoencoder model detected this anomaly based on reconstruction error.
The value deviation is -8.17 sigma from the normal mean, and the score deviation
is 9.32 sigma from the normal score baseline.

#### 3. Manufacturing Perspective
In the context of the industrial water circulation system, a low accelerometer
reading could indicate:
- Sensor malfunction or calibration drift
- Sudden damping in the valve mechanism
- Loss of mechanical connection

#### 4. Severity Assessment: CRITICAL
Recommended Actions:
1. Alert operator immediately
2. Consider emergency shutdown if safety risk
3. Dispatch maintenance team

#### 5. Contributing Sensors
1. Accelerometer1RMS: 39.3% contribution
2. Accelerometer2RMS: 15.2%
3. Pressure: 15.0%
4. Thermocouple: 8.7%
5. Temperature: 8.7%
```

### 예시 2: High Severity Anomaly (Index 1143)

```
### Anomaly Explanation

#### Observed Behavior
Multiple sensors showing coordinated anomaly at timestamp 1143:
- Accelerometer1RMS: 45.2% contribution (2.8x baseline)
- Pressure: 22.1% contribution (1.9x baseline)
- Flow Rate: 18.3% contribution (1.6x baseline)

#### ML Model Analysis
LSTM Autoencoder score: 0.987 (threshold: 0.600)
This multi-sensor pattern indicates systematic process deviation.

#### Domain Interpretation
The combination of high vibration, elevated pressure, and reduced flow
is a classic indicator of valve restriction or partial blockage in
water circulation systems.

#### Recommended Actions
1. Immediate: Inspect valve for foreign material
2. Short-term: Check upstream filter condition
3. Long-term: Review maintenance schedule
```

---

## 5. 생성된 파일

```
experiments/
├── feature_importance.py          # 325 lines
├── llm_explainer.py               # 480 lines (updated)
└── domain_knowledge/
    ├── __init__.py
    ├── knowledge_retriever.py     # 280 lines
    ├── skab_knowledge.yaml        # 250 lines
    └── smd_knowledge.yaml         # 200 lines

scripts/
├── test_feature_importance.py
├── test_domain_knowledge.py
├── test_full_integration.py
├── batch_explanation_generation.py
└── evaluate_explanations.py

runs/batch_explanations/
├── batch_explanations_20251125_043620.json   # 12 explanations
└── evaluation_results_20251125_043620.json   # Quality metrics
```

---

## 6. 기술적 결정

### 6.1 모델 선택
- **GPT-4o-mini**: 비용 효율적, 빠른 응답 (5-10초)
- Temperature: 0.3 (사실적, 일관된 설명)
- Max tokens: 2000-2500

### 6.2 도메인 지식 구조
- YAML 형식으로 구조화
- 센서별 상세 스펙 포함
- Failure mode와 action 매핑

### 6.3 Feature Importance 방식
- Reconstruction error 기반 (LSTM-AE)
- Z-score fallback (데이터만 있는 경우)
- 상위 5개 센서 ranking

---

## 7. 한계 및 향후 개선

### 현재 한계
1. SKAB 데이터셋에 집중 (SMD 테스트 미완료)
2. LSTM-AE 외 탐지기 SHAP 미적용
3. 실시간 설명 생성 미지원

### 향후 개선
1. IsolationForest SHAP 통합
2. 실시간 스트리밍 지원
3. 다국어 설명 (한국어)

---

## 8. 결론

Phase 2에서 **4-in-1 LLM 설명 시스템**을 성공적으로 구축했습니다:

1. ✅ 이상치 설명 (What happened)
2. ✅ ML 모델 해석 (Why flagged)
3. ✅ 도메인 지식 연결 (Manufacturing context)
4. ✅ Feature Importance (Which sensors)

**품질 평가 결과 93.9/100 (EXCELLENT)**로, 연구 목표를 달성했습니다.

---

## 다음 단계: Phase 3

LLM 기반 파라미터 최적화:
- LLM Parameter Advisor 구현
- Baseline vs LLM-guided 비교 실험
- 통계적 유의성 검증
