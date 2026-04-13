# LFactory 연구 개선 계획

**버전**: 2.0 (전면 개정 - 연구 중심 접근)
**최종 업데이트**: 2025-10-01
**담당**: 연구 기획 리드

---

## 📌 요약

본 TODO는 **연구 우선 접근법**으로 이전 버전을 대체합니다. 현재 프로젝트는 다음과 같은 치명적인 이론적 공백을 가지고 있습니다:

1. **정체성 위기**: "LLM-Guided"를 표방하나 LLM 구현 전무
2. **방법론적 결함**: ML detector가 시간 구조 무시; SpecCNN의 이론적 근거 부재
3. **자명한 가설**: 현재 가설들이 모두 너무 당연함 (tautological)
4. **Baseline 부재**: 표준 방법(IsolationForest, LSTM-AE)과 비교 없음
5. **용어 미정의**: "Local" anomaly detection 정의 없음; 이벤트 지표의 "peak" 부정확하게 근사

본 계획은 **이론적 타당성**, **엄밀한 실험**, **정직한 문서화**를 조기 주장보다 우선합니다.

---

## 🔬 Part 1: 연구 정체성 및 이론 확립 (CRITICAL - Week 1-2)

### 1.1 프로젝트 범위 명확화 [필수]

#### 문제
- **현재 주장**: "LLM-Guided Local Anomaly Detection"
- **실제**: LLM 구현 제로; Detect만 있는 파이프라인
- **영향**: 오해를 불러일으키는 제목; 과대 광고

#### 작업
- [ ] **프로젝트 범위 결정**
  - **Option A (권장)**: 명시적으로 2단계로 분리
    - Phase 1 (현재): "Calibrated Cost-Sensitive Anomaly Detection for Manufacturing Time-Series"
    - Phase 2 (미래): "LLM-Guided Explanation and Action Layer"
  - **Option B**: 최소 LLM 프로토타입 구현 (TF-IDF 기반 RAG-Bayes stub)
  - **담당자**: 연구 리드
  - **마감**: Week 1
  - **산출물**: README.md Section 1, HANDBOOK.md Section 1 업데이트

- [ ] **모든 문서에서 "LLM-Guided" 제거 또는 정당화**
  - Option A: 프로젝트명을 Phase 1 이름으로 변경; Phase 2 로드맵 추가
  - Option B: `experiments/explain_rag.py` stub + 문서화 추가
  - **수정 파일**: README.md, HANDBOOK (EN/KO), experiments/__init__.py, main_experiment.py

#### 성공 기준
- [ ] 어떤 문서에도 오해의 소지가 없음
- [ ] Option A 선택 시 명확한 단계 분리
- [ ] Option B 선택 시 작동하는 LLM 프로토타입

---

### 1.2 "Local" 정의 또는 제거 [필수]

#### 문제
- **현재**: "Local Anomaly Detection"을 사용하나 "Local"이 **전혀 정의되지 않음**
- **모호성**: Time-local? Space-local? Context-local?
- **영향**: 재현 불가능한 연구; 심사자가 거부할 것

#### 작업
- [ ] **"local" 관련 문헌 조사**
  - "local anomaly"를 사용하는 논문 5-10편 조사
  - 공통 정의 파악
  - **산출물**: `docs/LOCAL_DEFINITION.md` (2-3 페이지)

- [ ] **정의 선택 또는 용어 제거**
  - **Option A**: "point-wise anomaly"로 정의 (vs global/collective)
  - **Option B**: "time-window local"로 정의 (슬라이딩 윈도우 맥락)
  - **Option C**: "Local" 용어 완전 제거
  - **산출물**: EVALUATION_PROTOCOL.md Section "Terminology"

#### 성공 기준
- [ ] ≥2개 문서에 명확한 정의
- [ ] 재현 가능: 두 연구자가 동일하게 해석

---

### 1.3 연구 질문 재정립 [필수]

#### 문제
**현재 가설** (이전 TODO.md):
1. "ML/Hybrid가 Rule보다 AUC-PR 높다" → **자명함** (Rule이 너무 단순)
2. "Calibration이 ECE 감소" → **동어반복** (ECE 정의상)
3. "Cost-optimal < fixed" → **동어반복** (최적화 정의상)

**영향**: 연구 기여도 없음; 심사자가 거부할 것

#### 작업
- [ ] **4개 비자명 연구 질문 정식화**
  - **RQ1** (특징 비교): "제조 시계열의 spike/step/drift 이상에서 주파수 영역 특징(DFT 대역)이 시간 영역 특징(롤링 통계)보다 우수한가?"
    - **검증 가능**: SpecCNN 변형 vs Rule 변형 ablation 비교
    - **비자명**: 어느 것이 나은지 명확하지 않음; 이상 유형에 따라 다름

  - **RQ2** (Hybrid 방법): "어떤 앙상블 방법(선형, 곱셈, max, 학습)이 보정-비용 trade-off에서 최선인가?"
    - **검증 가능**: 4개 변형 구현; ECE와 기대 비용 동시 측정
    - **비자명**: 선형이 가장 단순하지만 반드시 최선은 아님

  - **RQ3** (지표 상관관계): "탐지기들에 대해 포인트 단위 F1과 이벤트 단위 F1의 상관관계는?"
    - **검증 가능**: 산점도, Pearson/Spearman 상관계수
    - **비자명**: Trade-off 존재 가능; 제조 AD에서 잘 연구되지 않음

  - **RQ4** (비용 민감도): "데이터셋 불균형과 SNR에 따라 FN/FP 비용 비율을 어떻게 조정해야 하는가?"
    - **검증 가능**: (불균형, SNR, cost_ratio)에 대한 그리드 탐색; 최적값 찾기
    - **비자명**: 실무자를 위한 실용적 가이드

  - **산출물**: TODO.md Section 1.3 (이 섹션), HANDBOOK Section 4 (Research Method)

#### 성공 기준
- [ ] 각 RQ에 명확한 실험 프로토콜
- [ ] 결과가 "yes/no" 또는 정량적 비교 가능
- [ ] 최소 2개 RQ가 엄밀하게 답하면 출판 가능

---

### 1.4 이론적 배경 문서 작성 [필수]

#### 1.4.1 선행 연구 조사
- [ ] **`docs/RELATED_WORK.md` 작성**
  - **Section 1**: 시계열 이상탐지 (10-15편 논문)
    - Classical: ARIMA 기반, STL decomposition
    - ML: Isolation Forest, LOF, One-Class SVM
    - DL: LSTM-AE, VAE, GAN 기반
    - 각 방법의 한계 인용 (예: LSTM-AE는 해석 어려움)

  - **Section 2**: 이상탐지에서의 Calibration (5-7편)
    - Calibration이 중요한 이유 (Guo et al. 2017을 분류에서 AD로 확장)
    - 기존 방법: Platt, Isotonic, Temperature
    - Gap: 제조 시계열에 거의 적용 안 됨

  - **Section 3**: 비용 민감 학습 (5편)
    - 불균형 분류에서의 비용 행렬
    - 임계값 최적화
    - Gap: 정적 비용 행렬, 지연 무시

  - **Section 4**: 제조 특화 방법 (3-5편)
    - SKAB 데이터셋 논문들
    - 도메인 특화 피처 엔지니어링
    - Gap: 대부분 규칙 기반 또는 단일 방법

  - **분량**: 8-10 페이지
  - **형식**: 각 논문당 2-3문장: 방법, 결과, 한계

#### 1.4.2 연구 배경 및 동기
- [ ] **`docs/RESEARCH_BACKGROUND.md` 작성**
  - **Section 1**: 제조 이상탐지 요구사항
    - 낮은 false alarm률 (생산 중단 비용)
    - 조기 탐지 (손상 비용 최소화)
    - 해석 가능성 (운영자 신뢰)
    - **근거**: 산업 보고서 또는 전문가 인터뷰 (가능 시)

  - **Section 2**: 기존 접근법의 한계
    - **Rule-only**: 낮은 recall, 수동 임계 조정
    - **ML-only**: 블랙박스, 높은 false alarm, 라벨 데이터 필요
    - **Uncalibrated**: 확률 부정확 → 임계 설정 어려움
    - **각 주장**: 논문 인용 또는 SKAB/SMD에서 경험적 증거

  - **Section 3**: 제안 접근법 (Hybrid + Calibration + Cost)
    - **Hybrid**: 규칙 투명성 + ML 표현력 결합
    - **Calibration**: 임계 설정을 위한 신뢰할 수 있는 확률
    - **Cost**: 운영 비용 구조에 임계 정렬
    - **다이어그램**: 파이프라인 흐름도

  - **Section 4**: 기대 기여
    - 앙상블 방법의 경험적 비교
    - Point vs event 지표 분석
    - 제조를 위한 비용 행렬 가이드

  - **분량**: 5-6 페이지

#### 성공 기준
- [ ] 모든 주장에 참조 또는 경험적 근거
- [ ] 명확한 포지셔닝: 기존 연구 대비 새로운 점
- [ ] 한계에 대한 정직 (예: "탐색적 연구")

---

## 🧪 Part 2: 방법론적 결함 수정 (CRITICAL - Week 2-3)

### 2.1 ML Detector: 시간 구조 위반 [필수]

#### 문제
**현재 구현** (`experiments/ml_detector.py:50-51`):
```python
pairs = sorted([(float(v), i) for i, v in enumerate(series)], key=lambda x: x[0])
# 값만으로 정렬 → 시간 정보 완전 손실
```

**왜 CRITICAL인가**:
- 이것은 **시계열 이상탐지가 아님**; 통계적 outlier detection
- t=10의 스파이크 vs t=1000의 스파이크가 값이 같으면 동일 점수
- **시계열 분석의 기본 가정 위반**
- 전체 "제조 시계열 특화" 주장 훼손
- **심사자가 무효로 거부할 것**

#### 작업
- [ ] **시간 고려 대안 구현 (하나 선택)**
  - **Option A**: 슬라이딩 윈도우 임베딩
    ```python
    # 각 포인트를 [v_{t-w}, ..., v_t] 벡터로 임베딩
    # 임베딩 공간에서 kNN
    ```
    - 장점: 순수 Python, 해석 가능
    - 단점: O(n²w) 복잡도

  - **Option B**: STOMP 기반 (Matrix Profile)
    - 장점: 시계열 AD의 최신 기법
    - 단점: 복잡, `stumpy` 라이브러리 필요 가능

  - **Option C**: Isolation Forest로 교체
    - 장점: 표준 baseline, 잘 연구됨
    - 단점: `scikit-learn` 필요 (프로젝트 정책상 선택 의존성 OK)

  - **담당**: ML 연구자
  - **마감**: Week 2
  - **파일**: `experiments/ml_detector.py` (`knn_scores` 함수 리팩터)

- [ ] **검증 테스트: 시간 셔플 불변성**
  - 테스트 생성: 시계열 셔플 → 점수가 크게 변해야 함
  - 현재 detector는 **실패** (점수 불변)
  - 새 detector는 **통과**해야 함
  - **파일**: `tests/test_ml_detector.py` (신규)

- [ ] **Ablation 연구: time-aware vs time-agnostic**
  - 이전 detector를 `ml_detector_valueonly.py`로 보존 (baseline)
  - SKAB/SMD에서 비교:
    - AUC-PR 차이
    - Event recall 차이
  - **가설**: Time-aware가 ≥10% AUC-PR 개선
  - **산출물**: `experiments/ABLATION_TIME_AWARE.md`

#### 성공 기준
- [ ] 시간 셔플 테스트 통과
- [ ] value-only baseline 대비 통계적으로 유의미한 개선
- [ ] EXPERIMENT_REPORT.md에 문서화

---

### 2.2 SpecCNN: 이론적 공백 [필수]

#### 문제
**현재** (`experiments/spec_cnn.py:52, 61-64`):
```python
w_low, w_mid, w_high = -0.2, 0.6, 0.6  # 왜 이 가중치?
# Bands: [0, 0.1], (0.1, 0.3], (0.3, 0.5]  # 왜 이 범위?
```

**이슈**:
- **임의의 하이퍼파라미터**: 3개 밴드, 이 범위, 이 가중치에 대한 정당화 없음
- **오해의 소지 있는 이름**: "SpecCNN"은 CNN 암시; 실제로는 DFT + 가중합
- **도메인 분석 없음**: 제조 데이터 주파수 특성 미지
- **검증 불가**: 이론 없이는 좋고 나쁨을 알 수 없음

#### 작업
- [ ] **제조 데이터 주파수 분석**
  - **Step 1**: SKAB/AIHub의 PSD(Power Spectral Density) 계산
    - 정상 구간: 중앙값 PSD
    - 이상 구간: 중앙값 PSD
    - 오버레이 플롯
  - **Step 2**: 판별력 있는 대역 식별
    - 주파수 빈마다 KL-divergence 또는 t-test
    - 상위 3개 판별 범위 찾기
  - **Step 3**: 대역 선택 정당화
    - [0, 0.1], [0.1, 0.3], [0.3, 0.5]가 판별 대역과 일치하는가?
    - 아니면 대역 업데이트
  - **산출물**: `docs/FREQUENCY_ANALYSIS.md` + `docs/figures/`의 플롯
  - **담당**: 신호처리 연구자
  - **마감**: Week 3

- [ ] **Detector 이름 변경**
  - **기존**: `SpecCNN`
  - **신규**: `Freq3Band` 또는 `SpectralHeuristic`
  - **업데이트**: 모든 파일, 문서, CLI

- [ ] **한계 문서화**
  - `experiments/spec_cnn.py` docstring에 추가:
    > **한계**: 휴리스틱 가중치; 학습 불가; 도메인 특화; 윈도우 ≥64 필요.
  - HANDBOOK에 추가:
    > SpecCNN-lite는 탐색적 휴리스틱 detector입니다. 대역 선택은 SKAB 주파수 분석 기반(docs/FREQUENCY_ANALYSIS.md 참조). 실제 사용에는 학습 가능한 특징 추출기 권장.

- [ ] **선택: 학습 가능 버전**
  - `spec_cnn_learned.py` 구현:
    - 대역 및 가중치를 학습 가능 파라미터로
    - 검증 세트로 튜닝
  - 학습 vs 휴리스틱 비교
  - **산출물**: `ABLATION_SPECCNN.md`

#### 성공 기준
- [ ] 주파수 분석이 명확한 판별 대역 표시
- [ ] 데이터로 대역 선택 정당화
- [ ] 한계에 대한 정직한 문서화

---

### 2.3 Hybrid 앙상블: 자명한 방법 [MEDIUM]

#### 문제
**현재** (`experiments/hybrid_detector.py:47`):
```python
scores.append((1.0 - alpha) * rs[i] + alpha * ms[i])
# 단순 가중 평균
```

**이슈**: 이것은 **가장 기본적인 앙상블 방법**; 최소한의 연구 기여도

#### 작업
- [ ] **4개 앙상블 변형 구현**
  - **Linear** (현재): `(1-α)×R + α×M`
  - **Product**: `R^(1-α) × M^α` (기하 평균)
  - **Max**: `max(R, M)` (보수적)
  - **Learned**: `[R, M]` 특징에 대한 로지스틱 회귀
    - 검증 세트로 학습
    - 테스트 세트로 평가
  - **파일**: `experiments/hybrid_detector.py` (메서드 추가)

- [ ] **RQ2 비교 (Section 1.3)**
  - **지표**: ECE, Expected Cost, AUC-PR, Event F1
  - **데이터셋**: Synthetic, SKAB, SMD
  - **스크립트**: `scripts/hybrid_comparison.py`
  - **출력**: `runs/hybrid_comparison/summary.csv`

- [ ] **기본 방법 선택 및 정당화**
  - Linear가 이기면: OK, 대안 테스트했음을 명시
  - Learned가 이기면: main_experiment.py의 기본값 업데이트
  - **산출물**: EXPERIMENT_REPORT.md Section "Ensemble Method Comparison"

#### 성공 기준
- [ ] 4개 방법 모두 구현 및 테스트
- [ ] 통계 검정으로 승자 표시 (있으면)
- [ ] 근거와 함께 기본 선택 문서화

---

### 2.4 비용 행렬: 도메인 분석 [IMPORTANT]

#### 문제
**현재 기본값** (`experiments/cost_threshold.py:18`):
```python
costs = (0.0, 1.0, 5.0, 0.0)  # C_TN, C_FP, C_FN, C_TP
# FN 비용 = 5 × FP 비용 → 왜?
```

**이슈**: 제조 도메인 분석 없음; 임의의 배수

#### 작업
- [ ] **문헌 조사: 제조 false alarm & miss 비용**
  - 생산 라인 중단 비용 논문 3-5편 조사
  - 불량품 유출 비용 논문 2-3편 조사
  - 비용 범위 추출: C_FP ∈ [?, ?], C_FN ∈ [?, ?]
  - **산출물**: `docs/COST_ANALYSIS.md`

- [ ] **비용 민감도 분석**
  - **그리드**: FN/FP 비율 ∈ {1, 3, 5, 10, 20}
  - **데이터셋**: Synthetic (통제), SKAB, SMD
  - **변화**: 데이터셋 불균형, SNR
  - **측정**: 최적 임계, 결과 F1, Event Recall
  - **가설 (RQ4)**: 최적 비율이 불균형과 상관
  - **스크립트**: `scripts/cost_sensitivity.py`
  - **출력**: Heatmap 플롯 + CSV

- [ ] **탐지 지연 비용 모델 [선택]**
  - 비용에 지연 포함: `C_FN(delay) = C_FN_base × (1 + β × delay)`
  - 조기 탐지가 기대 비용 감소하는지 테스트
  - **산출물**: `experiments/cost_threshold_delay.py`

#### 성공 기준
- [ ] 문헌으로 비용 범위 정당화
- [ ] 민감도 분석이 패턴 드러냄
- [ ] 실무자 가이드: "불균형 >1:50이면 비율 ≥10 사용"

---

## 📊 Part 3: 평가 지표 엄밀성 (HIGH - Week 3-4)

### 3.1 이벤트 지표 "Peak" 정의 [필수]

#### 문제
**현재** (`experiments/metrics.py:95-96, 114, 126`):
```python
# Note: current implementation approximates the event "peak" by segment_end.
leads.append(float(max(0, b0 - earliest + 1)))
```

**왜 틀렸는가**:
- 제조에서 "peak" = 최대 심각도 (예: 최대 온도, 최대 진동)
- Peak를 **구간 끝**으로 근사하면 lead time이 무의미
- 예: [100, 200]의 이상, 120에서 탐지
  - 현재: lead = 200 - 120 + 1 = 81 (항상 양수!)
  - 올바름(max): Peak가 150이면 lead = 150 - 120 = 30

**영향**: 무효한 이벤트 지표 → 무효한 실험 결론

#### 작업
- [ ] **4개 peak 정의 구현**
  - `peak_end` (현재, 비교용)
  - `peak_max`: `argmax(values[start:end])`
  - `peak_threshold`: `|value - baseline| > threshold`인 첫 인덱스
  - `peak_mid`: `(start + end) // 2`
  - **파일**: `experiments/metrics.py` (`event_metrics_from_segments` 리팩터)
  - **API**: 파라미터 `peak_method='max'` 추가

- [ ] **4개 정의 모두 비교**
  - 각각에 대해 lead time & detection delay 계산
  - **데이터셋**: Synthetic (true peak 알려짐), SKAB
  - **분석**: 어느 것이 도메인 직관과 가장 일치하는가?
  - **산출물**: `experiments/PEAK_DEFINITION_STUDY.md`

- [ ] **평가 프로토콜 업데이트**
  - **파일**: `EVALUATION_PROTOCOL.md` Section 4.2
  - 하위 섹션 추가:
    > **Peak 정의**: 이벤트 peak를 이상 구간 내 베이스라인으로부터의 절댓값 편차 최대인 시점으로 정의. 근거: 제조에서 이상 심각도는 보통 이 시점에 최고. 대안 정의(구간 끝, 중간, 임계 교차)도 비교를 위해 구현(PEAK_DEFINITION_STUDY.md 참조).

#### 성공 기준
- [ ] 4개 정의 모두 구현
- [ ] 기본값을 `peak_max`로 변경
- [ ] 도메인 근거 문서화

---

### 3.2 Point vs Event 지표 연구 [IMPORTANT]

#### 문제
- Point-wise F1과 event-wise F1 모두 보고
- **불명확**: 어느 것이 주요? 충돌 가능? Trade-off 존재?
- **영향**: Detector 최적화 방법 결정 불가

#### 작업
- [ ] **상관관계 분석 (Section 1.3의 RQ3)**
  - 모든 detector (Rule, ML, Hybrid, SpecCNN)를 모든 데이터셋에서 실행
  - 산점도: Point F1 (x축) vs Event F1 (y축)
  - Pearson & Spearman 상관계수 계산
  - **가설**: 약한 상관 (≤0.6) → 지표가 다른 측면 포착
  - **스크립트**: `scripts/point_vs_event_study.py`

- [ ] **Detector 편향 분석**
  - Point F1 높지만 Event F1 낮은 detector 식별 (정밀하나 이벤트 놓침)
  - Point F1 낮지만 Event F1 높은 detector 식별 (이벤트 탐지하나 노이지)
  - **산출물**: `docs/METRICS_TRADEOFF.md`

- [ ] **제조 도메인 관점**
  - 인터뷰 또는 문헌: 생산에서 어느 지표가 더 중요?
  - **가설**: Event recall > Point precision (이벤트 놓치는 것이 false alarm보다 나쁨)
  - **문서**: HANDBOOK Section "Choosing Metrics"

#### 성공 기준
- [ ] 95% CI와 함께 상관계수 보고
- [ ] Trade-off 곡선 플롯
- [ ] 도메인 기반 권장사항

---

### 3.3 통계적 유의성 검정 [필수]

#### 문제
- 현재 실험: 단일 seed (예: seed=42)
- **약점**: 결과가 우연일 수 있음
- **영향**: 재현 불가; 심사자가 통계 검정 요구할 것

#### 작업
- [ ] **다중 seed 실험**
  - **Seeds**: {42, 123, 456, 789, 2024} (최소 n=5)
  - **적용**: 모든 detector × dataset 조합
  - **자동화**: `scripts/multi_seed_eval.py`
    - 입력: detector, dataset, hyperparams
    - 출력: `runs/multi_seed/<detector>_<dataset>/seed_<N>/run.json`
  - **집계**: `scripts/aggregate_multi_seed.py`
    - AUC-PR, F1, ECE 등의 평균 ± 표준편차 계산
    - 출력: `runs/multi_seed/summary.csv`

- [ ] **통계 검정 구현**
  - **파일**: `experiments/statistics.py` (신규)
  - **검정**:
    - Paired t-test (정규 분포인 경우)
    - Wilcoxon signed-rank test (비정규인 경우)
    - 다중 비교를 위한 Bonferroni 보정
  - **함수**: `compare_detectors(results_A, results_B, metric='auc_pr') -> (statistic, p_value, effect_size)`

- [ ] **리포트에 통합**
  - `result_manager.py` 업데이트:
    - 다중 seed 존재 시 95% CI 계산
    - REPORT.md에 추가: "AUC-PR: 0.78 ± 0.03 (95% CI: [0.72, 0.84])"
  - EXPERIMENT_REPORT.md 업데이트:
    - 각 detector의 평균 ± 표준편차 표
    - 유의미한 차이에 별표: "Hybrid* vs Rule (p<0.05)"

#### 성공 기준
- [ ] 모든 주요 결과에 ≥5 seeds
- [ ] 유의성 검정 보고 (p-values)
- [ ] 효과 크기 보고 (Cohen's d 등)

---

## 🧬 Part 4: 실험 설계 강화 (HIGH - Week 4-5)

### 4.1 표준 Baseline 추가 [필수]

#### 문제
- **현재**: "Rule" baseline만 (너무 단순)
- **누락**: 산업 표준 방법 (IsolationForest, LSTM-AE 등)
- **영향**: Baseline 비교 없이 "우리 방법이 더 낫다" 주장 불가

#### 작업
- [ ] **Isolation Forest baseline 구현**
  - **의존성**: `scikit-learn` (선택, 프로젝트 정책상 허용)
  - **파일**: `experiments/baseline_iforest.py`
  - **인터페이스**: 다른 detector와 동일 → `detect(series, **kwargs) -> {scores, preds, params}`
  - **하이퍼파라미터**: `n_estimators=100`, `contamination=auto`
  - **통합**: main_experiment.py에 `--detector iforest` 추가

- [ ] **LSTM-AE baseline 구현 [선택]**
  - **의존성**: PyTorch 또는 TensorFlow (선택)
  - **파일**: `experiments/baseline_lstmae.py`
  - **아키텍처**: 단순 1-layer LSTM encoder-decoder
  - **학습**: train split 사용 (가능 시) 또는 test의 슬라이딩 윈도우
  - **점수**: 재구성 오류
  - **참고**: 너무 복잡하면 향후 작업으로 연기

- [ ] **Baseline 비교 실험**
  - **실행**: 모든 baseline (Rule, IForest, LSTM-AE) + 제안 방법 (ML, Hybrid, SpecCNN)
  - **데이터셋**: Synthetic, SKAB, SMD
  - **지표**: AUC-PR, Event F1, ECE, Expected Cost
  - **스크립트**: `scripts/baseline_comparison.py`
  - **출력**: 비교 표와 함께 `runs/baseline_comparison/REPORT.md`

#### 성공 기준
- [ ] ≥2개 표준 baseline 구현
- [ ] 공정한 비교 (동일 데이터, 동일 평가)
- [ ] 제안 방법이 지면: 이유에 대한 정직한 논의

---

### 4.2 데이터셋 프로파일링 [필수]

#### 문제
- 주장: "제조 시계열 특화"
- **이슈**: SMD는 **서버 모니터링**이지 제조가 아님!
- **누락**: 정량적 데이터셋 특성

#### 작업
- [ ] **`docs/DATA_CHARACTERISTICS.md` 작성**
  - **각 데이터셋** (Synthetic, SKAB, SMD, AIHub71802):
    - **도메인**: 제조? IT 인프라? 기타?
    - **샘플링 레이트**: Hz 또는 samples/day
    - **길이**: 총 포인트, train/test split
    - **라벨 분포**: P(anomaly), 불균형 비율
    - **이상 유형**: Spike, step, drift, 기타 (수동 검사 또는 메타데이터)
    - **SNR 추정**: `signal_power / noise_power` (근사)
    - **주파수 특성**: FFT로부터 지배 주파수
  - **표 형식**: 비교하기 쉽게

- [ ] **SMD 정당화 또는 제거**
  - **Option A**: SMD 제거, 다른 제조 데이터셋 추가
  - **Option B**: 주장을 "제조 **및 인프라** 시계열"로 변경
  - **Option C**: SMD를 별도 분석, 제조 결과와 섞지 않음
  - **산출물**: README.md Section "Datasets" 업데이트

- [ ] **데이터셋별 분석**
  - 왜 Detector X가 SKAB에서는 잘 작동하나 SMD에서는 나쁜가?
  - Detector 속성과 데이터셋 특성 상관
  - **예**: "SpecCNN은 명확한 주파수 분리가 있는 데이터셋(SKAB valve)에서 탁월, 광대역 노이즈(SMD machine-1-1)에서는 고전."
  - **산출물**: EXPERIMENT_REPORT.md Section "Dataset Sensitivity"

#### 성공 기준
- [ ] 각 데이터셋의 정량적 프로파일
- [ ] "제조 특화" 주장 수정 또는 뒷받침
- [ ] 데이터셋-detector 적합성 분석

---

### 4.3 전체 데이터셋으로 확장 [IMPORTANT]

#### 문제
- **현재**: 데이터셋당 1개 파일로 스모크 테스트 (`--file-index 0`)
- **누락**: 모든 파일에 대한 전체 평가

#### 작업
- [ ] **배치 평가 스크립트**
  - **스크립트**: `scripts/full_batch_eval.py` (기존 batch_eval.py 확장)
  - **데이터셋**: SKAB (모든 시나리오), SMD (모든 머신), AIHub (모든 센서)
  - **Detectors**: 전부 (Rule, ML, Hybrid, SpecCNN, IForest)
  - **출력**: `runs/full_batch/<dataset>_<UTC>/`
    - 파일별 결과
    - 집계 통계 (평균, 표준편차, CI)
    - 실패 케이스 (AUC-PR < 0.5인 파일)

- [ ] **실패 분석**
  - 모든 detector가 실패하는 파일 식별
  - 검사: 라벨 품질 나쁜가? 이상이 너무 미묘한가?
  - 문서화: `docs/FAILURE_CASES.md`
  - **정직한 보고**: "SKAB 파일 valve1_03에서 모든 방법이 AUC-PR ≤ 0.3 달성. 수동 검사 결과 라벨 모호성 드러남 (논의 참조)."

- [ ] **EXPERIMENT_REPORT.md 업데이트**
  - 단일 파일 결과를 전체 데이터셋 결과로 교체
  - 분포 플롯 추가 (파일 간 AUC-PR 히스토그램)

#### 성공 기준
- [ ] ≥80% 파일 평가
- [ ] 집계 통계 보고
- [ ] 실패 케이스 정직하게 문서화

---

## 📝 Part 5: 문서 대폭 수정 (MEDIUM - Week 5-6)

### 5.1 핵심 문서 업데이트

- [ ] **README.md**
  - **Section 1**: 프로젝트 범위 (Phase 1 vs Phase 2, "LLM-guided" 제거/수정)
  - **Section 2**: "Local" 정의 (추가 또는 용어 제거)
  - **Section 3**: 연구 질문 (Section 1.3의 4개 RQ)
  - **Section 4**: Quick Start (불변)
  - **Section 5**: 문서 맵
    ```
    - 연구 배경 → RESEARCH_BACKGROUND.md
    - 선행 연구 → RELATED_WORK.md
    - 평가 세부사항 → EVALUATION_PROTOCOL.md
    - 온보딩 → HANDBOOK.md
    ```

- [ ] **HANDBOOK.md (영문)**
  - **Section 1**: Goals and Project Map (범위 업데이트)
  - **Section 4**: Research Method and Acceptance
    - 자명한 가설을 RQ1-RQ4로 교체
    - 수용 임계 추가: "RQ1은 주파수 특징이 시간 특징 대비 ≥15% AUC-PR 개선을 ≥2 데이터셋에서 p<0.05로 보이면 수용."
  - **Section 5**: Datasets (특성 표 추가)
  - **Section 7**: Models & Extensibility (baseline 정보 추가)

- [ ] **HANDBOOK_ko.md (한글)**
  - 영문 버전과 동기화
  - 새 RQ 및 수용 기준 번역

---

### 5.2 평가 프로토콜 강화

- [ ] **EVALUATION_PROTOCOL.md 대폭 개정**
  - **Section 1**: 용어
    - "Local" 정의 (유지 시)
    - "Point-wise" vs "Event-wise" 정의

  - **Section 4**: 시계열 특화 지표
    - **4.2 Detection Delay**: (현재, OK)
    - **4.3 Lead Time**: 4개 peak 정의로 교체
      - 각각의 공식
      - 어떤 것을 언제 사용하는지
    - **4.4 Point-Adjust**: (현재, OK)

  - **Section 5**: Calibration & ECE
    - **NEW**: AD에서 calibration이 중요한 이유
    - Reliability diagram 해석 방법
    - ECE < 0.05 가이드라인 근거

  - **Section 6**: 비용 민감 임계
    - **NEW**: 비용 행렬 설정 방법
    - 민감도 분석 가이드
    - 문헌의 비용 범위 예시

  - **Section 7**: 통계 검정
    - **NEW**: 다중 seed 프로토콜
    - 유의성 검정
    - 효과 크기 보고

  - **Section 8**: 지표 선택 가이드
    - Point F1 vs Event F1 언제 사용
    - 제조 도메인 권장사항

  - **분량**: 현재 2페이지에서 8-10페이지로 확장

---

### 5.3 실험 보고서 재구성

- [ ] **EXPERIMENT_REPORT.md 완전 재작성**
  - **현재 이슈**: 스모크 테스트 집중; 엄밀성 부족

  - **신규 구조**:
    - **Section 1**: Baseline 비교
      - 표: 모든 detector × dataset, AUC-PR 평균 ± 표준편차
      - 통계 검정
      - 유의성 마커와 함께 랭킹

    - **Section 2**: 연구 질문 답변
      - **RQ1**: 주파수 vs 시간 특징 → 결과 + 해석
      - **RQ2**: 앙상블 방법 → 승자 + 근거
      - **RQ3**: Point vs event 상관 → 산점도 + 논의
      - **RQ4**: 비용 민감도 → Heatmap + 가이드라인

    - **Section 3**: Ablation Studies
      - Time-aware ML vs value-only
      - SpecCNN 학습 vs 휴리스틱
      - Calibration 방법 비교
      - Hybrid 결합 방법

    - **Section 4**: 실패 분석
      - 모든 방법이 실패하는 데이터셋/파일
      - 이유 가설 (라벨 품질, 이상 미묘함)
      - 데이터 개선 권장사항

    - **Section 5**: 한계 및 향후 작업
      - **정직**: 현재 ML detector 여전히 단순; DL 필요
      - SpecCNN은 휴리스틱; 학습 특징 필요
      - 비용 행렬 정적; 동적/지연 의존 필요
      - LLM 설명 아직 없음 (Phase 2)

    - **Section 6**: 재현성
      - 모든 실험에 seed, git SHA
      - 모든 비교에 통계 검정
      - runs/ 아티팩트 링크

  - **분량**: 15-20 페이지

---

## 🛠️ Part 6: 엔지니어링 개선 (MEDIUM - Week 6-7)

### 6.1 테스트 [기존 TODO 항목, 우선순위 재조정]

- [ ] **Unit tests** (이전 TODO에서, 여전히 유효)
  - Calibration: Platt/Isotonic/Temperature 수렴
  - Cost thresholds: 정합성 체크
  - Metrics: segments 추출, 엣지 케이스의 delay/lead
  - Data loaders: 정렬, 라벨 길이 처리
  - **목표**: 80% 코드 커버리지
  - **파일**: `tests/test_*.py`

- [ ] **Integration tests**
  - End-to-end: synthetic data → detect → calibrate → cost → report
  - Golden files: seed=42의 ROC/PR CSV가 정확히 일치해야

---

### 6.2 코드 품질

- [ ] **로깅 프레임워크** (이전 TODO)
  - `print()`를 `logging.info/debug/warning`으로 교체
  - `--log-level` CLI 인자 추가
  - **파일**: `experiments/logging_config.py`

- [ ] **패키징** (이전 TODO)
  - `pyproject.toml` 생성
  - pip 설치 가능: `pip install -e .`
  - Console entry point: `lfactory-detect`

- [ ] **Type hints**
  - 모든 신규 코드에 추가
  - 기존 코드에 점진적으로 추가
  - CI에서 `mypy` 실행

---

### 6.3 재현성

- [ ] **환경 명세**
  - `requirements.txt`에 버전 고정
  - 깨끗한 venv에서 설치 테스트
  - Python 버전 요구사항 문서화 (3.9+)

- [ ] **Docker 이미지 [선택]**
  - 재현 가능한 환경을 위한 `Dockerfile`
  - 데이터셋 사전 로드 (허용 시)

---

## 🗑️ Part 7: 불필요/오해의 소지 제거

### 7.1 제거 또는 통합

- [ ] **UPDATE_LOG.md**: 너무 장황
  - **조치**: `UPDATE_LOG_SUMMARY.md` 생성 (1-2 페이지)
  - 상세 로그는 `archive/UPDATE_LOG_FULL.md`에 보관

- [ ] **중복 내용**
  - README vs HANDBOOK 중복 → 역할 명확화:
    - README: 빠른 개요, 설정, 내비게이션
    - HANDBOOK: 포괄적 온보딩, 연구 방법

---

### 7.2 과장된 주장 수정

- [ ] **"LLM-Guided"** → Section 1.1 참조
- [ ] **"제조 특화"** → Section 4.2 참조
- [ ] **"SpecCNN"** → "Freq3Band" 또는 "SpectralHeuristic"으로 변경
- [ ] **"Local"** → 정의 또는 제거

---

### 7.3 Phase 2로 연기

- [ ] **RAG-Bayes, LLM, Act** (현재 TODO에서 Low)
  - 별도의 `TODO_PHASE2.md`로 이동
  - 명확한 분리: Phase 1은 Detect만, Phase 2는 Explain+Act

---

## ✅ 성공 기준 (프로젝트 수준)

### 연구로서
- [ ] ≥2개 비자명 연구 질문에 엄밀하게 답변
- [ ] 모든 비교가 통계적으로 검증 (p-values, CIs)
- [ ] 한계에 대한 정직 (과장 없음)
- [ ] Baseline 비교로 우리 방법의 강점/약점 표시

### 문서로서
- [ ] 신규 연구자가 ≤3일 내 온보딩 가능
- [ ] 모든 방법에 이론적 정당화 또는 한계 명시
- [ ] 100% 재현성 (동일 seed → 동일 결과 ±1e-6)

### 코드로서
- [ ] ≥80% 테스트 커버리지
- [ ] Pip 설치 가능
- [ ] 로깅 프레임워크 완비
- [ ] 선택 의존성 우아하게 처리

---

## 📅 타임라인 요약

| 주차  | 집중 영역                      | 주요 산출물                                |
|-------|--------------------------------|--------------------------------------------|
| 1-2   | 연구 정체성 & 이론             | 범위 결정, RQs, RELATED_WORK.md            |
| 2-3   | 방법론적 결함 수정             | ML detector 리팩터, SpecCNN 정당화         |
| 3-4   | 평가 지표 엄밀성               | Peak 정의, 통계 검정                       |
| 4-5   | 실험 설계 강화                 | Baselines, 데이터셋 프로파일링, 전체 배치 |
| 5-6   | 문서 대폭 수정                 | HANDBOOK, EVALUATION_PROTOCOL, EXPERIMENT_REPORT 개정 |
| 6-7   | 엔지니어링 개선                | 테스트, 로깅, 패키징                       |

**총**: 6-7주 (연구자 1 FTE + 엔지니어 1 FTE, 또는 두 역할 모두 하는 연구자 2 FTE)

---

## 🔄 유지보수 & 업데이트

- **주간**: 본 TODO의 체크박스 상태 업데이트
- **주요 실험 후**: EXPERIMENT_REPORT.md 업데이트
- **논문 제출 후**: 본 TODO를 `archive/TODO_v2.0.md`로 아카이브

---

**최종 업데이트**: 2025-10-01
**버전**: 2.0
**다음 리뷰**: Week 2 이후 (RQs 답변 가능한지 확인)
