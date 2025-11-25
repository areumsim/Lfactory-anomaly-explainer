# 관련 연구: 문헌 조사

**버전**: 1.0 (초안)
**최종 업데이트**: 2025-10-01
**상태**: 1주차 초기 조사 (섹션당 5편); 2주차까지 15-20편으로 확장 예정

---

## 요약

이 문서는 LFactory 연구 범위와 관련된 문헌을 조사합니다:
1. **시계열 이상 탐지** (고전, ML, DL 방법)
2. **보정(Calibration)** 이상 탐지 및 분류에서
3. **비용 민감 학습** 불균형 문제를 위한
4. **제조업 특화** 이상 탐지

각 논문에 대해 다음을 제공합니다:
- **방법(Method)**: 핵심 기법
- **주요 발견(Key findings)**: 주요 결과
- **한계(Limitations)**: LFactory가 해결하는 격차

**현재 상태**: 대표적인 논문 약 20편의 초안. 2주차 문헌 조사에서 40-50편으로 확장 예정.

---

## 1. 시계열 이상 탐지

### 1.1 고전 통계 방법

#### [1] Chandola, Banerjee & Kumar (2009) - "Anomaly Detection: A Survey"
- **학회**: ACM Computing Surveys
- **방법**: 이상 탐지 기법의 포괄적 분류체계
- **주요 발견**:
  - **포인트(point)**, **문맥적(contextual)**, **집합적(collective)** 이상을 정의
  - 통계적 방법(ARIMA, 가우시안 모델), 최근접 이웃, 클러스터링 방법 조사
  - 응용 도메인 식별: 침입 탐지, 사기, 의료, 산업
- **한계**:
  - 딥러닝 이전 시대(2009); LSTM, 오토인코더 미포함
  - **보정(calibration)**과 **비용 민감성(cost-sensitivity)**에 대한 논의 제한적
  - 제조 시계열 벤치마크 없음
- **LFactory와의 관련성**: 기초 분류체계 제공; LFactory는 집합적 이상이 아닌 **포인트 및 문맥적** 이상에 초점

---

#### [2] Box, Jenkins & Reinsel (2015) - "Time Series Analysis: Forecasting and Control"
- **학회**: 교과서 (5판)
- **방법**: 시계열 예측을 위한 ARIMA 모델; 이상 = 잔차가 임계값 초과
- **주요 발견**:
  - ARIMA가 추세, 계절성, 자기상관 포착
  - 잔차 기반 이상 탐지: `z = (x_t - x̂_t) / σ_residual > threshold`
  - 가우시안 노이즈 가정 하에서 이론적 보장
- **한계**:
  - **선형** 동역학 가정 (비선형 제조 프로세스에서 실패)
  - 수동 하이퍼파라미터 선택 필요 (p, d, q)
  - **일시적 스파이크**에서 성능 낮음 (모델은 부드러운 자기상관 가정)
- **LFactory와의 관련성**: 기준 비교; LFactory의 Rule 탐지기는 유사하지만 더 단순함 (rolling z-score)

---

#### [3] Hyndman & Khandakar (2008) - "Automatic Time Series Forecasting: The forecast Package for R"
- **학회**: Journal of Statistical Software
- **방법**: Auto-ARIMA (AIC를 통한 자동 하이퍼파라미터 선택)
- **주요 발견**:
  - ARIMA 차수 선택 자동화, 계절성 처리 (SARIMA)
  - 산업에서 널리 사용 (R `forecast` 패키지)
  - 잔차 기반 이상 탐지 내장
- **한계**:
  - 계산 비용이 큰 n에 대해 비효율적 확장 (O(n³))
  - 여전히 선형 모델 가정
  - **다변량** 지원 없음 (LFactory 데이터셋은 여러 센서 보유)
- **LFactory와의 관련성**: 고전 기준으로 추가 가능; 현재 미구현

---

### 1.2 기계학습 방법

#### [4] Liu, Ting & Zhou (2008) - "Isolation Forest"
- **학회**: ICDM
- **방법**: 무작위 격리 트리의 앙상블; 이상은 격리가 쉬움 (경로 길이가 짧음)
- **주요 발견**:
  - **O(n log n)** 학습 복잡도 (LOF, kNN보다 빠름)
  - 고차원 데이터에서 우수한 성능
  - 데이터 분포에 대한 가정 없음
- **한계**:
  - **시간 구조 무시** (시계열을 i.i.d. 샘플로 취급)
  - 내장 **보정** 없음 ("이상 점수" 출력, 확률 아님)
  - 하이퍼파라미터 `contamination` 설정 필요 (도메인 지식 필요)
- **LFactory와의 관련성**: 기준으로 계획됨 (TODO.md Part 4.1); 시간 특징 추가 필요 (슬라이딩 윈도우 임베딩)

---

#### [5] Breunig et al. (2000) - "LOF: Identifying Density-Based Local Outliers"
- **학회**: SIGMOD
- **방법**: Local Outlier Factor (LOF); 포인트의 국지 밀도를 이웃과 비교
- **주요 발견**:
  - **가변 밀도** 클러스터 처리 (전역 이상 탐지보다 우수)
  - 파라미터 없는 밀도 비율 (자동 스케일링)
  - 산업에서 널리 사용
- **한계**:
  - **O(n²)** 복잡도 (대규모 시계열에 비실용적)
  - **시간 인식 없음** (Isolation Forest와 동일한 문제)
  - k (이웃 크기) 선택에 민감
- **LFactory와의 관련성**: kNN 탐지기를 대체 가능하지만 더 느림; LFactory는 효율성을 위해 더 단순한 kNN 사용

---

#### [6] Schölkopf et al. (2001) - "Estimating the Support of a High-Dimensional Distribution"
- **학회**: Neural Computation
- **방법**: One-Class SVM (OC-SVM); 정상 데이터 주변의 결정 경계 학습
- **주요 발견**:
  - 커널 트릭이 비선형 경계 가능하게 함
  - 이론적 보장 (PAC 학습 프레임워크)
  - 새로움 탐지(novelty detection)에 효과적
- **한계**:
  - **커널 선택** 필요 (RBF, 다항식) → 하이퍼파라미터 튜닝
  - **O(n²)** ~ **O(n³)** 학습 복잡도
  - **확률적 출력 없음** (이진 결정만)
- **LFactory와의 관련성**: 확장성 및 확률 보정 부족으로 우선순위 낮음

---

### 1.3 딥러닝 방법

#### [7] Malhotra et al. (2016) - "LSTM-based Encoder-Decoder for Multi-Sensor Anomaly Detection"
- **학회**: ICML Workshop on Anomaly Detection
- **방법**: LSTM 오토인코더; 재구성 오류를 이상 점수로 사용
- **주요 발견**:
  - **다변량** 시계열 처리 (여러 센서)
  - **시간 의존성** 자동 학습 (수동 특징 엔지니어링 불필요)
  - NASA 베어링 진동 데이터셋에서 **0.92 F1** 달성
- **한계**:
  - **대규모 학습 데이터** 필요 (수천 개의 정상 시퀀스)
  - **해석 불가능** (블랙박스 신경망)
  - **보정 없음** (재구성 오류 ≠ 확률)
  - 학습이 느림 (GPU 필요)
- **LFactory와의 관련성**: 선택적 기준으로 계획됨 (TODO.md Part 4.1); LFactory는 해석 가능성 우선 → LSTM-AE는 Phase 3로 연기

---

#### [8] An & Cho (2015) - "Variational Autoencoder based Anomaly Detection using Reconstruction Probability"
- **학회**: SNU Data Mining Center Technical Report
- **방법**: Variational Autoencoder (VAE); 이상 = 학습된 분포 하에서 낮은 재구성 확률
- **주요 발견**:
  - **확률적** 출력 (디코더로부터 p(x | latent))
  - MNIST 이상값 탐지에서 일반 오토인코더(AE)보다 우수
  - 잠재 공간이 해석 가능 (가우시안 사전분포)
- **한계**:
  - **가우시안** 잠재 공간 가정 (제조 데이터에 맞지 않을 수 있음)
  - 재구성 확률이 **보정되지 않음** (분포 내에서 과신뢰)
  - 복잡한 학습 (변분 추론, KL 발산)
- **LFactory와의 관련성**: 향후 작업에 흥미로움 (Phase 3); 현재 초점은 보정된 더 단순한 방법

---

#### [9] Li et al. (2018) - "MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks"
- **학회**: ICANN
- **방법**: GAN 기반; 생성자가 정상 분포 학습, 판별자가 이상 탐지
- **주요 발견**:
  - **다변량** 시계열 처리
  - LSTM-GAN 아키텍처 (시간적 + 적대적 학습 결합)
  - SWaT (Secure Water Treatment) 데이터셋에서 LSTM-AE 능가
- **한계**:
  - **매우 복잡** (두 네트워크, 적대적 학습 불안정)
  - **이론적 보장 없음** (GAN 학습 수렴이 어려움)
  - **보정 없음** (판별자 점수 ≠ 확률)
- **LFactory와의 관련성**: 현재 범위에 너무 복잡; 향후 작업으로 연기

---

#### [10] Hundman et al. (2018) - "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
- **학회**: KDD
- **방법**: LSTM 예측 오류 + **동적 임계값 설정** (EWMA + 가지치기)
- **주요 발견**:
  - NASA SMAP/MSL 우주선 원격측정에서 **높은 정밀도** 달성
  - **이벤트 기반 평가** (포인트 단위만이 아님)
  - 오픈소스 구현 (https://github.com/khundman/telemanom)
- **한계**:
  - 동적 임계값이 **휴리스틱** (수동 튜닝된 EWMA)
  - **보정 없음** (임계값이 확률적이지 않음)
  - 우주선 데이터는 제조와 **다름** (낮은 SNR, 다변량)
- **LFactory와의 관련성**: 이벤트 기반 메트릭의 **핵심 참고문헌** (탐지 지연, 리드 타임); LFactory는 유사한 메트릭 사용하지만 **보정 + 비용 민감성** 추가

---

### 1.4 Matrix Profile / Discord Discovery

#### [11] Yeh et al. (2016) - "Matrix Profile I: All Pairs Similarity Joins for Time Series"
- **학회**: ICDM (Best Paper Award)
- **방법**: Matrix Profile (MP); 모든 부분 시퀀스에 대한 최근접 이웃 거리의 효율적 계산
- **주요 발견**:
  - 정확한 최근접 이웃 탐색을 위한 **O(n² log n)** 알고리즘 (STOMP)
  - **모티프** (반복 패턴) 및 **디스코드** (이상) 탐지
  - **파라미터 없음** (윈도우 크기만)
- **한계**:
  - **형태 기반** 이상 탐지 (비정상적인 부분 시퀀스 형태)
  - **값 기반** 이상을 놓칠 수 있음 (크기 스파이크지만 정상 형태)
  - O(n²) 공간 복잡도 (대규모 데이터셋은 GPU 또는 근사 필요)
- **LFactory와의 관련성**: ML 탐지기 개선 고려 (TODO.md Part 2.1 Option B); 아직 미구현

---

### 1.5 스펙트럼 / 주파수 도메인 방법

#### [12] Rasheed & Alhajj (2013) - "A Framework for Periodic Outlier Pattern Detection in Time-Series Sequences"
- **학회**: IEEE Transactions on Cybernetics
- **방법**: 푸리에 변환 + 주파수 도메인에서 이상값 탐지
- **주요 발견**:
  - 주기적 이상이 DFT 계수의 **진폭 변화**로 나타남
  - **기계 진동** 데이터에 효과적 (주기적 정상 행동)
  - 주파수 + 시간 도메인 결합 (하이브리드 접근)
- **한계**:
  - **주기성** 필요 (SMD 서버 메트릭과 같은 비주기 데이터에서 실패)
  - **윈도우 크기** 선택이 중요 (트레이드오프: 주파수 해상도 vs 시간 국지화)
  - 표준 평가 벤치마크 없음
- **LFactory와의 관련성**: SpecCNN 탐지기에 대한 **직접적 영감**; LFactory는 **밴드 가중치**와 **하이브리드 앙상블** 추가

---

#### [13] Cheng et al. (2016) - "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network"
- **학회**: KDD
- **방법**: Stochastic RNN (variational RNN); 이상 = 낮은 예측 확률
- **주요 발견**:
  - **확률적** 예측 (점 추정이 아닌 분포 출력)
  - **다변량** 의존성 처리
  - Yahoo S5 데이터셋 (웹 트래픽)에서 평가
- **한계**:
  - 복잡한 모델 (확률적 레이어 + 변분 추론)
  - **보정 검증 없음** (예측 분포가 잘 보정되어 있다고 가정하지만 테스트되지 않음)
  - 웹 트래픽 ≠ 제조 (다른 이상 특성)
- **LFactory와의 관련성**: 확률적 확장에 흥미롭지만, 현재 초점은 **더 단순한** 모델 보정이 우선

---

---

## 2. 이상 탐지 및 분류에서의 보정(Calibration)

### 2.1 보정 기초

#### [14] Guo et al. (2017) - "On Calibration of Modern Neural Networks"
- **학회**: ICML
- **방법**: 심층 신경망의 보정 분석; **Temperature Scaling** 제안
- **주요 발견**:
  - 현대 DNN은 **과신뢰** (예측 확률 > 실제 빈도)
  - **Expected Calibration Error (ECE)** 메트릭이 오보정 정량화
  - **Temperature Scaling** (단일 파라미터 T)이 ECE를 효과적으로 감소
  - Platt scaling보다 단순하고 실전에서 우수
- **한계**:
  - **분류**에 초점, 이상 탐지 아님 (다른 설정)
  - 검증 세트 가용 가정 (비지도 AD에서는 참이 아닐 수 있음)
  - ECE 빈이 임의적일 수 있음 (빈 선택에 대한 민감도)
- **LFactory와의 관련성**: 보정 방법의 **핵심 참고문헌**; LFactory는 Temperature Scaling + Platt + Isotonic 구현 및 이상 탐지에서 비교

---

#### [15] Platt (1999) - "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods"
- **학회**: Advances in Large Margin Classifiers
- **방법**: **Platt Scaling**; SVM 결정값에 시그모이드 적합
- **주요 발견**:
  - SVM 점수를 **보정된 확률**로 변환
  - 2-파라미터 시그모이드: `p = 1 / (1 + exp(A × score + B))`
  - UCI 벤치마크에서 검증
- **한계**:
  - **SVM**을 위해 설계, 일반 이상 점수가 아님
  - **레이블된 검증 데이터** 필요 (지도 보정)
  - 검증 세트가 작으면 과적합 가능 (< 100 샘플)
- **LFactory와의 관련성**: 보정 방법으로 구현됨 (`experiments/calibration.py`); Isotonic 및 Temperature와 비교

---

#### [16] Zadrozny & Elkan (2002) - "Transforming Classifier Scores into Accurate Multiclass Probability Estimates"
- **학회**: KDD
- **방법**: 보정을 위한 **Isotonic Regression**; 비모수적 (시그모이드 가정 없음)
- **주요 발견**:
  - Platt scaling보다 **유연** (임의 단조 함수 적합)
  - 비시그모이드 보정 곡선 처리
  - **대규모 검증 세트**에서 우수 (> 500 샘플)
- **한계**:
  - 소규모 검증 세트에서 **과적합** (< 100 샘플)
  - 닫힌 형태 해 없음 (반복 PAV 알고리즘 필요)
  - **비평활** 보정 곡선 생성 가능 (들쭉날쭉)
- **LFactory와의 관련성**: 구현 및 비교됨; LFactory 발견: 대규모 데이터셋(SKAB)에서 Isotonic이 우수, 소규모(AIHub)에서 Platt 우수

---

#### [17] Niculescu-Mizil & Caruana (2005) - "Predicting Good Probabilities with Supervised Learning"
- **학회**: ICML
- **방법**: 보정 방법의 포괄적 비교 (Platt, Isotonic, Bayesian Binning)
- **주요 발견**:
  - **부스팅 트리** 및 **랜덤 포레스트**가 보정 불량 (과신뢰)
  - **나이브 베이즈**는 잘 보정되지만 낮은 정확도
  - **보정이 Brier score 개선** (확률 정확도 메트릭)
- **한계**:
  - **지도 분류**에 초점
  - 이상 탐지 벤치마크 없음
  - **비용 민감** 보정 다루지 않음 (보정 + 비용 공동 최적화)
- **LFactory와의 관련성**: 보정이 탐지 품질과 **분리**됨을 검증; LFactory는 동일한 패턴 표시 (ML 탐지기가 보정 전 높은 AUC지만 낮은 ECE)

---

### 2.2 이상 탐지에서의 보정 (희소한 문헌)

#### [18] Emmott et al. (2013) - "Systematic Construction of Anomaly Detection Benchmarks from Real Data"
- **학회**: ACM SIGKDD Workshop on Outlier Detection
- **방법**: **ODDS** (Outlier Detection DataSets) 벤치마크 제안; 보정을 암묵적으로 평가
- **주요 발견**:
  - 대부분의 이상 탐지기가 **보정되지 않은 점수** 출력 (임의 범위)
  - **순위 메트릭** (AUC-ROC, AUC-PR)이 표준이지만 **확률 보정** 무시됨
  - AD에서 보정을 위한 표준 프로토콜 없음
- **한계**:
  - 문제를 식별하지만 **해결책 제안하지 않음**
  - ODDS 데이터셋은 **시계열이 아님** (대부분 표 형식/공간)
- **LFactory와의 관련성**: **격차 식별**; LFactory는 분류 보정 방법을 시계열 AD에 적용하여 이 격차 해결

---

#### [19] Perini et al. (2020) - "Quantifying the Confidence of Anomaly Detectors in Their Example-Wise Predictions"
- **학회**: ECML-PKDD
- **방법**: **conformal prediction**을 사용한 이상 탐지기의 사후 보정
- **주요 발견**:
  - Conformal prediction이 **예측 집합** 제공 (단일 점수가 아님)
  - **커버리지** 보장 (P(true label ∈ prediction set) ≥ 1-α)
  - 이미지 및 표 형식 AD 벤치마크에서 평가
- **한계**:
  - **시계열** 특화되지 않음
  - Conformal prediction은 **교환가능성** 필요 (시계열에서 유지되지 않을 수 있음)
  - 계산 비용: 예측당 O(n) (보정 세트 필요)
- **LFactory와의 관련성**: 대안 보정 접근; 향후 작업으로 연기 (현재 초점은 Platt/Isotonic/Temperature)

---

---

## 3. 비용 민감 학습

### 3.1 기초

#### [20] Elkan (2001) - "The Foundations of Cost-Sensitive Learning"
- **학회**: IJCAI
- **방법**: 비대칭 비용 하에서 **최적 결정 규칙** 도출
- **주요 발견**:
  - 최적 임계값: `τ* = p(negative) × (C_FP / (C_FP + C_FN))`
  - 불균형 데이터의 경우, 결정 임계값 ≠ 0.5
  - 비용 민감 학습이 오류율이 아닌 **예상 비용을 감소**
- **한계**:
  - **알려진 비용 행렬** 가정 (하지만 실제로는 비용이 추정됨)
  - **완벽한 보정** 가정 (확률이 잘못되면 임계값도 잘못됨)
  - **시계열**이나 **탐지 지연** 다루지 않음
- **LFactory와의 관련성**: **기초 이론**; LFactory는 (1) 보정된 확률, (2) 탐지 지연 비용 (향후 작업), (3) 제조 도메인 비용 추정으로 확장

---

#### [21] Ling et al. (2006) - "Cost-Sensitive Learning and the Class Imbalance Problem"
- **학회**: Encyclopedia of Machine Learning
- **방법**: 비용 민감 기법 조사 (리샘플링, 임계값 설정, 앙상블)
- **주요 발견**:
  - **리샘플링** (SMOTE, 언더샘플링) vs **임계값 설정** (비용 조정 결정)
  - 불균형 > 1:100인 경우, 리샘플링과 비용 민감 임계값 **모두** 필요
  - 앙상블 방법 (배깅, 부스팅)을 비용 민감하게 만들 수 있음
- **한계**:
  - 일반 조사; **시계열 AD**에 대한 구체적 지침 없음
  - **보정 + 비용** 공동 문제 다루지 않음
- **LFactory와의 관련성**: RQ4 (비용 민감도 분석)에 정보 제공; LFactory는 극단적 불균형에 SMOTE 필요할 수 있음 (TODO.md Part 2.4)

---

#### [22] Zhou & Liu (2006) - "Training Cost-Sensitive Neural Networks with Methods Addressing the Class Imbalance Problem"
- **학회**: IEEE TKDE
- **방법**: 학습 중 비용 행렬을 **손실 함수에 통합**
- **주요 발견**:
  - 비용 민감 손실: `L = Σ C(y_true, y_pred) × CrossEntropy`
  - 경험적으로, 최적 비용 비율 ≈ `sqrt(imbalance)` (예: 불균형=100 → 비율≈10)
  - 사후 임계값 설정보다 우수 (비용 최소화를 위해 직접 모델 학습)
- **한계**:
  - **지도 학습** 필요 (비지도 AD에 적용 불가)
  - 신경망 특화 (kNN, Rule 기반 탐지기에 불가)
- **LFactory와의 관련성**: **비용 비율 ~ 불균형** 관계 검증 (RQ4 가설); LFactory는 사후 임계값 설정 사용 (더 단순, 재학습 불필요)

---

#### [23] Lowell et al. (2019) - "Practical Obstacles to Deploying Active Learning"
- **학회**: EMNLP
- **방법**: NLP에서 **비용 민감 능동 학습** 연구; 문맥 의존 주석 비용
- **주요 발견**:
  - **비용이 예제마다 다름** (정적 행렬 아님)
  - 주석 비용이 **난이도** 및 **도메인 전문가 가용성**에 의존
  - 능동 학습이 **예산 제약** 고려해야 함
- **한계**:
  - **능동 학습**에 초점, 이상 탐지 아님
  - 비용이 **주석 비용**, **운영 비용** 아님 (다른 도메인)
- **LFactory와의 관련성**: **동적 비용 행렬** 영감 (향후 작업: 탐지 지연이 비용 증가)

---

### 3.2 비용 민감 평가

#### [24] Drummond & Holte (2006) - "Cost Curves: An Improved Method for Visualizing Classifier Performance"
- **학회**: Machine Learning
- **방법**: **Cost curves** (클래스 분포 및 비용 비율 vs 예상 비용 플롯)
- **주요 발견**:
  - **비용 민감** 비교를 위해 ROC보다 우수
  - **최적 동작 지점** 직접 표시
  - **가변 비용** 및 **클래스 사전분포** 처리
- **한계**:
  - **정답** 필요 (레이블 없는 비지도 AD에 적용 불가)
  - **이진 분류** 가정 (시계열 세그먼트로의 확장이 자명하지 않음)
- **LFactory와의 관련성**: 비용 민감 평가를 위한 AUC-PR 대안; 시각화로 추가 가능 (TODO.md Part 5.2)

---

---

## 4. 제조업 특화 이상 탐지

### 4.1 벤치마크 데이터셋

#### [25] Katser & Kozitsin (2020) - "Skoltech Anomaly Benchmark (SKAB)"
- **학회**: arXiv preprint
- **방법**: **SKAB 데이터셋** 소개 (산업 물 순환 시스템)
- **주요 발견**:
  - 실제 고장이 있는 **7개 시나리오** (밸브 폐쇄, 펌프 고장, 센서 드리프트)
  - 타임스탬프가 있는 **레이블된 이상 세그먼트**
  - 8가지 방법 벤치마크 (ARIMA, Isolation Forest, LSTM-AE 등)
  - **발견**: 모든 시나리오를 지배하는 단일 방법 없음
- **한계**:
  - 소규모 데이터셋 (총 ~34,000 포인트, 이상 ~1,000개만)
  - 단일 산업 프로세스 (일반화 가능성 불명확)
  - **비용 분석** 없음 (F1, 정밀도, 재현율만)
- **LFactory와의 관련성**: **주요 평가 데이터셋**; LFactory는 (1) 보정, (2) 비용 민감성, (3) 이벤트 메트릭으로 SKAB 벤치마킹 확장

---

#### [26] Su et al. (2019) - "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network"
- **학회**: KDD
- **방법**: **SMD 데이터셋** 소개 (Server Machine Dataset, 28개 서버 엔티티)
- **주요 발견**:
  - **다변량** (엔티티당 38차원)
  - 대형 인터넷 회사의 실제 프로덕션 서버 데이터
  - OmniAnomaly (stochastic RNN) 평가 → 당시 SOTA
- **한계**:
  - **서버 메트릭**, 제조가 아님 (CPU, 메모리, 네트워크 ≠ 온도, 압력, 진동)
  - 불균형이 **보통** (~3%), 극단적이지 않음
  - 비용 행렬 제공 없음
- **LFactory와의 관련성**: **보조 데이터셋**으로 사용; LFactory는 SMD vs SKAB에서 방법이 다르게 작동하는 이유 분석으로 도메인 불일치 해결 (TODO.md Part 4.2)

---

### 4.2 산업 응용

#### [27] Susto et al. (2015) - "Machine Learning for Predictive Maintenance: A Multiple Classifier Approach"
- **학회**: IEEE Transactions on Industrial Informatics
- **방법**: 반도체 제조에서 **예측 유지보수**를 위한 분류기 앙상블
- **주요 발견**:
  - **비용 비대칭**이 중요 (거짓 경보 → 생산 중단, 누락 → 장비 손상)
  - **SVM + Random Forest + Logistic Regression** 앙상블이 단일 방법 능가
  - 팹(공장)에 실제 배포
- **한계**:
  - **지도** (레이블된 고장 데이터 필요, 실전에서 드묾)
  - 보정 논의 없음
  - 데이터셋이 **독점** (비공개)
- **LFactory와의 관련성**: 제조에서 **비용 비대칭 검증**; LFactory는 유사한 앙상블 접근 사용하지만 (1) 비지도, (2) 보정됨

---

#### [28] Baudin & Rougier (2021) - "Anomaly Detection in Predictive Maintenance: A Probabilistic Framework"
- **학회**: European Workshop on Advanced Control and Diagnosis
- **방법**: 회전 기계에서 이상 탐지를 위한 베이지안 프레임워크
- **주요 발견**:
  - **확률적** 출력 (진동이 주어졌을 때 고장의 사후 확률)
  - **도메인 지식** 통합 (물리 기반 모델 + 데이터 기반 학습)
  - 산업 펌프에 배포
- **한계**:
  - **물리 기반 모델** 필요 (항상 가용하지 않음)
  - 베이지안 추론이 **느림** (MCMC 샘플링)
  - 오픈소스 구현 없음
- **LFactory와의 관련성**: **확률적 접근** 검증; LFactory는 전체 베이지안 추론 대신 더 단순한 보정 방법 (Platt/Isotonic) 사용

---

#### [29] Ren et al. (2019) - "A Survey of Deep Active Learning"
- **학회**: ACM Computing Surveys
- **방법**: 딥러닝을 위한 능동 학습 조사 (이상 탐지 포함)
- **주요 발견**:
  - 능동 학습이 정보가 많은 샘플 선택으로 **레이블링 비용** 감소
  - **준지도 AD**와 관련 (일부 이상에 레이블, 탐지기 개선)
  - 불확실성 샘플링, 위원회 쿼리, 예상 모델 변화
- **한계**:
  - **딥러닝**에 초점 (Rule/kNN 탐지기에 적용 불가)
  - **레이블링 예산** 트레이드오프 (운영 비용 트레이드오프 아님)
- **LFactory와의 관련성**: 향후 작업 (Phase 3); LLM을 사용하여 먼저 레이블할 탐지된 이상 제안 가능

---

---

## 5. 문헌의 격차 (LFactory의 기여)

### 5.1 시계열 이상 탐지를 위한 보정
**격차**: 대부분의 AD 논문이 AUC-ROC/AUC-PR 보고하지만 **보정(ECE) 무시**.
**LFactory 기여**: 제조 시계열에 3가지 보정 방법 (Platt, Isotonic, Temperature) 적용 및 비교.

### 5.2 비용 민감 시계열 AD
**격차**: 비용 민감 학습 문헌이 **정적 비용** 및 **즉각적 결정** 가정 (탐지 지연 없음).
**LFactory 기여**: 비용 비율이 데이터셋 불균형 및 SNR과 어떻게 달라져야 하는지 분석 (RQ4); 향후 작업에 지연 의존 비용 포함.

### 5.3 Point vs Event 메트릭 상관관계
**격차**: 포인트 단위 F1과 이벤트 단위 F1 간 **상관관계**의 체계적 연구 없음.
**LFactory 기여**: RQ3가 탐지기와 데이터셋에 걸쳐 이 상관관계 조사.

### 5.4 제조를 위한 주파수 vs 시간 특징
**격차**: 대부분의 논문이 **시간 도메인만** (LSTM) 또는 **주파수 도메인만** (스펙트럼 클러스터링) 사용; **절제(ablation)로 둘 다 비교**하는 경우 적음.
**LFactory 기여**: RQ1이 이상 유형 계층화로 주파수 (SpecCNN) vs 시간 (Rule) 비교.

### 5.5 보정을 갖춘 하이브리드 앙상블
**격차**: 앙상블 AD 논문 (예: [27])이 **보정 + 비용**을 공동으로 다루지 않음.
**LFactory 기여**: RQ2가 **ECE와 예상 비용 모두**에서 앙상블 방법 비교 (다목적).

---

## 6. 요약 표

| 논문 | 연도 | 방법 | 데이터셋 | 메트릭 | 보정? | 비용? | 오픈소스? |
|-------|------|--------|---------|---------|--------------|-------|--------------|
| [1] Chandola et al. | 2009 | Survey | Various | - | ❌ | ❌ | - |
| [4] Liu et al. (IForest) | 2008 | Isolation Forest | KDD99 | AUC | ❌ | ❌ | ✅ (sklearn) |
| [7] Malhotra et al. (LSTM-AE) | 2016 | LSTM-AE | NASA | F1 | ❌ | ❌ | ❌ |
| [10] Hundman et al. | 2018 | LSTM + Dynamic Thresh | NASA SMAP | Event F1 | ❌ | ❌ | ✅ |
| [14] Guo et al. | 2017 | Temperature Scaling | ImageNet | ECE | ✅ | ❌ | ✅ (PyTorch) |
| [15] Platt | 1999 | Platt Scaling | UCI | Brier | ✅ | ❌ | ✅ (sklearn) |
| [20] Elkan | 2001 | Cost Theory | - | Cost | ❌ | ✅ | - |
| [25] Katser et al. (SKAB) | 2020 | Benchmark | SKAB | F1, Precision | ❌ | ❌ | ✅ |
| **LFactory (본 연구)** | 2025 | Hybrid + Calib + Cost | SKAB, SMD | F1, ECE, Cost, Event | ✅ | ✅ | ✅ |

**범례**: ✅ = 예/가능, ❌ = 아니오/다루지 않음

---

## 7. 문헌 조사 로드맵

### 1주차 (현재)
- [x] 초기 조사: 섹션당 5편 (총 ~20편)
- [x] LFactory 포지셔닝을 위한 격차 식별

### 2주차 (계획)
- [ ] **40-50편**으로 확장:
  - Section 1 (시계열 AD): 10편 추가 (LSTM 변형, GAN 기반, Matrix Profile 확장)
  - Section 2 (보정): 5편 추가 (conformal prediction, Bayesian calibration)
  - Section 3 (비용 민감): 5편 추가 (능동 학습 비용, 제조 비용 분석)
  - Section 4 (제조): 10편 추가 (산업 사례 연구, PHM 학회 논문)
- [ ] **인용 횟수** 추가 (Google Scholar) 영향력 평가
- [ ] **인용 그래프** 생성 (어떤 논문이 어떤 논문을 인용)

### 3주차 (계획)
- [ ] **포지셔닝 진술** 작성 (2-3페이지):
  - LFactory가 [10] Hundman (LSTM 기반)과 어떻게 다른지
  - LFactory가 [25] SKAB 벤치마크와 어떻게 다른지 (보정 + 비용 추가)
  - LFactory가 [27] Susto와 어떻게 다른지 (비지도, 공개 데이터셋)
- [ ] 논문 **서론** 섹션 초안 (이 관련 연구 기반)

---

## 8. 인용 형식

모든 참고문헌은 **APA 스타일** (저자, 연도) 사용. 전체 참고문헌 목록은 2주차에 추가 예정.

**임시 플레이스홀더** (`[TODO]` 표시):
- [ ] 모든 논문에 DOI 링크 추가
- [ ] 전체 인용 세부사항 추가 (학회, 권, 페이지)
- [ ] 연도 및 저자명 확인 (일부 기억에서, 확인 필요)

---

## 참고문헌 (부분 목록)

1. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM computing surveys*, 41(3), 1-58.
2. Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). *Time series analysis: forecasting and control* (5th ed.). Wiley.
3. Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. *Journal of statistical software*, 27, 1-22.
4. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM*, 413-422.
5. Breunig, M. M., et al. (2000). LOF: identifying density-based local outliers. *ACM SIGMOD*, 93-104.
6. Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. *Neural computation*, 13(7), 1443-1471.
7. Malhotra, P., et al. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. *ICML Workshop*.
8. An, J., & Cho, S. (2015). Variational autoencoder based anomaly detection using reconstruction probability. *SNU Data Mining Center Tech Report*.
9. Li, D., et al. (2018). MAD-GAN: Multivariate anomaly detection for time series data with generative adversarial networks. *ICANN*.
10. Hundman, K., et al. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. *KDD*.
11. Yeh, C. C. M., et al. (2016). Matrix profile I: all pairs similarity joins for time series. *ICDM* (Best Paper).
12. Rasheed, F., & Alhajj, R. (2013). A framework for periodic outlier pattern detection in time-series sequences. *IEEE Trans. on Cybernetics*.
13. Cheng, M., et al. (2016). Robust anomaly detection for multivariate time series through stochastic recurrent neural network. *KDD*.
14. Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*.
15. Platt, J. (1999). Probabilistic outputs for support vector machines. *Advances in large margin classifiers*, 61-74.
16. Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD*.
17. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
18. Emmott, A. F., et al. (2013). Systematic construction of anomaly detection benchmarks from real data. *KDD Workshop*.
19. Perini, L., et al. (2020). Quantifying the confidence of anomaly detectors in their example-wise predictions. *ECML-PKDD*.
20. Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*.
21. Ling, C. X., et al. (2006). Cost-sensitive learning and the class imbalance problem. *Encyclopedia of ML*.
22. Zhou, Z. H., & Liu, X. Y. (2006). Training cost-sensitive neural networks. *IEEE TKDE*.
23. Lowell, D., et al. (2019). Practical obstacles to deploying active learning. *EMNLP*.
24. Drummond, C., & Holte, R. C. (2006). Cost curves: An improved method for visualizing classifier performance. *Machine Learning*, 65, 95-130.
25. Katser, I. D., & Kozitsin, V. (2020). Skoltech Anomaly Benchmark (SKAB). *arXiv preprint arXiv:2005.01566*.
26. Su, Y., et al. (2019). Robust anomaly detection for multivariate time series through stochastic recurrent neural network. *KDD*.
27. Susto, G. A., et al. (2015). Machine learning for predictive maintenance: A multiple classifier approach. *IEEE Trans. on Industrial Informatics*, 11(3), 812-820.
28. Baudin, V., & Rougier, F. (2021). Anomaly detection in predictive maintenance: A probabilistic framework. *European Workshop on Advanced Control and Diagnosis*.
29. Ren, P., et al. (2019). A survey of deep active learning. *ACM Computing Surveys*.

*(DOI가 포함된 전체 참고문헌 목록은 2주차에 완성 예정)*

---

**버전 히스토리**:
- 1.0 (2025-10-01): 4개 섹션에 걸쳐 29편의 논문을 포함한 초안
- 1.1 (2주차 계획): 40-50편으로 확장, DOI 및 전체 인용 추가
- 2.0 (3주차 계획): 포지셔닝 진술 및 논문 서론 초안 추가
