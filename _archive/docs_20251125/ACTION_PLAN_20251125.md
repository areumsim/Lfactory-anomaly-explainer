# LFactory 작업 계획 (Action Plan)
**작성일**: 2025-11-25
**기준 문서**: REVIEW_20251125.md
**계획 기간**: 2025-11-25 ~ 2026-01-20 (8주)

---

## 📋 개요

본 문서는 2025-11-25 프로젝트 검토에서 발견된 문제점과 개선 사항에 대한 구체적 작업 계획을 제시합니다. 모든 작업은 **중요도 기반 우선순위**(Critical/High/Medium/Low)로 분류되며, 각 항목은 구체적인 작업 내용, 예상 소요 시간, 기존 TODO.md Week 계획과의 연결을 포함합니다.

**우선순위 기준**:
- **Critical**: 프로젝트 목표 달성에 필수적, 즉시 착수 필요
- **High**: 연구 품질에 중요한 영향, 2주 내 완료
- **Medium**: 기능 개선 및 확장, 1개월 내 완료
- **Low**: 선택적 개선, 여유 시 처리

---

## 🔴 Critical Priority (즉시 착수)

### C1: ML 탐지기 시간 구조 고려 모델로 교체

**문제**: 현재 `ml_detector.py`의 kNN 구현이 시간 구조를 무시하고 value-space만 사용하여 시계열 패턴을 놓칩니다.

**영향**:
- RQ1 (주파수 vs 시간 도메인) 검증 신뢰도 저하
- RQ2 (앙상블 방법) 검증 왜곡 가능성
- 제조업 시계열 특성 반영 실패

**작업 내용**:

1. **IsolationForest 구현** (우선 추천)
   ```python
   # experiments/ml_detector_isolation_forest.py 생성

   from sklearn.ensemble import IsolationForest

   def detect_isolation_forest(data, contamination=0.1, window_size=50):
       """
       시간 윈도우 기반 IsolationForest
       - 슬라이딩 윈도우로 특징 추출 (mean, std, min, max, trend)
       - IsolationForest로 이상치 스코어 계산
       """
       # 1. 슬라이딩 윈도우 특징 추출
       # 2. IsolationForest.fit_predict()
       # 3. 이상치 스코어 반환
   ```

2. **대안: LSTM Autoencoder** (선택적)
   ```python
   # experiments/ml_detector_lstm_ae.py 생성

   # LSTM 기반 시퀀스 재구성 오류로 이상치 탐지
   # 재구성 오류 = 이상치 스코어
   ```

3. **기존 kNN 유지 (베이스라인)**
   - `ml_detector.py` → `ml_detector_knn.py`로 이름 변경
   - 비교 목적으로 유지

4. **통합**
   - `main_experiment.py`에 `--ml_method` 옵션 추가
   - 선택: `knn`, `isolation_forest`, `lstm_ae`

**예상 소요 시간**: 3-5일
- IsolationForest: 2일
- LSTM-AE (선택): 3일
- 테스트 및 통합: 1일

**난이도**: 중 (IsolationForest), 상 (LSTM-AE)

**의존성**:
- 외부: `scikit-learn` (IsolationForest), `torch` (LSTM-AE)
- 내부: 없음

**성공 기준**:
- [ ] IsolationForest가 kNN 대비 AUC-PR > +0.05 (합성 데이터)
- [ ] SKAB 데이터셋에서 시간 패턴 탐지 개선 확인
- [ ] 기존 인터페이스(`{scores, preds, params}`) 호환
- [ ] 문서화 (`docs/HANDBOOK.md` 업데이트)

**기존 TODO.md 연결**: Week 3 (ML Detector Improvements)

**담당**: Core Dev

---

### C2: 통계적 검증 완료 (다중 시드 + Bootstrap CI)

**문제**: 현재 실험이 단일 시드로만 수행되어 재현성과 일반화 가능성을 확인할 수 없습니다.

**영향**:
- 연구 결과 신뢰도 부족
- 논문 투고 시 거부 가능성
- RQ1-RQ4 모든 검증에 영향

**작업 내용**:

1. **다중 시드 실험 프레임워크**
   ```python
   # scripts/multi_seed_experiment.py 생성

   import subprocess

   seeds = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

   for seed in seeds:
       subprocess.run([
           "python", "experiments/main_experiment.py",
           "--dataset", "synthetic",
           "--seed", str(seed),
           "--run_id", f"multi_seed_{seed}"
       ])

   # 결과 수집 및 통계 계산
   ```

2. **Bootstrap Confidence Intervals**
   ```python
   # scripts/ci_bootstrap.py 확장

   def bootstrap_ci(metrics, n_bootstrap=1000, alpha=0.05):
       """
       1000회 bootstrap으로 95% CI 계산
       - 메트릭: AUC-PR, F1, ECE
       - 출력: mean ± CI
       """
   ```

3. **통계적 유의성 검정**
   ```python
   # scripts/statistical_test.py 생성

   from scipy.stats import wilcoxon, mannwhitneyu

   def compare_detectors(results_a, results_b, metric="auc_pr"):
       """
       Wilcoxon signed-rank test (paired)
       - H0: A와 B의 성능이 같다
       - p < 0.05면 H0 기각
       """
   ```

4. **배치 실행 스크립트**
   ```bash
   # scripts/run_statistical_validation.sh

   #!/bin/bash

   # 다중 시드 실험
   python scripts/multi_seed_experiment.py \
       --datasets synthetic skab smd \
       --detectors rule ml hybrid speccnn

   # Bootstrap CI 계산
   python scripts/ci_bootstrap.py --runs runs/multi_seed_*

   # 통계 검정
   python scripts/statistical_test.py \
       --compare rule vs ml vs hybrid vs speccnn
   ```

**예상 소요 시간**: 5-7일
- 다중 시드 프레임워크: 1일
- Bootstrap CI: 2일
- 통계 검정: 2일
- 배치 실행 및 분석: 2일

**난이도**: 중

**의존성**:
- 외부: `scipy` (통계 검정)
- 내부: C1 완료 권장 (새로운 ML 탐지기 포함)

**성공 기준**:
- [ ] 10개 시드로 각 탐지기 실험 완료
- [ ] 모든 메트릭에 95% CI 보고
- [ ] Wilcoxon test로 탐지기 간 유의성 검정 (p-value)
- [ ] `EXPERIMENT_REPORT.md`에 통계 결과 업데이트
- [ ] 표준편차가 평균의 20% 이내 (재현성 확보)

**기존 TODO.md 연결**: Week 2 (Multi-seed experiments)

**담당**: Research Lead

---

## 🟠 High Priority (2주 내 완료)

### H1: SpecCNN 주파수 대역 가중치 최적화

**문제**: 현재 `spec_cnn.py`의 주파수 대역 가중치가 학습되지 않고 휴리스틱하게 하드코딩되어 있습니다.

**영향**:
- RQ1 (주파수 vs 시간 도메인) 검증 약화
- SpecCNN의 잠재 성능 미활용
- 데이터셋별 최적 가중치 다를 수 있음

**작업 내용**:

1. **Grid Search 구현**
   ```python
   # scripts/speccnn_grid_search.py 생성

   import itertools

   # 가중치 범위
   w_low = [-0.5, -0.2, 0.0, 0.2]
   w_mid = [0.2, 0.4, 0.6, 0.8, 1.0]
   w_high = [0.2, 0.4, 0.6, 0.8, 1.0]

   best_score = 0
   best_weights = None

   for w_l, w_m, w_h in itertools.product(w_low, w_mid, w_high):
       # validation set에서 AUC-PR 계산
       score = evaluate_speccnn(w_l, w_m, w_h, val_data)
       if score > best_score:
           best_score = score
           best_weights = (w_l, w_m, w_h)
   ```

2. **Logistic Regression 학습** (대안)
   ```python
   # experiments/spec_cnn.py 수정

   from sklearn.linear_model import LogisticRegression

   # 3개 주파수 대역 에너지를 특징으로
   # 로지스틱 회귀로 가중치 학습
   lr = LogisticRegression()
   lr.fit(band_energies, labels)

   # 학습된 계수를 가중치로 사용
   weights = lr.coef_
   ```

3. **데이터셋별 최적 가중치 저장**
   ```yaml
   # experiments/config.yaml에 추가

   detector:
     speccnn:
       weights:
         synthetic: {low: -0.2, mid: 0.6, high: 0.6}
         skab: {low: 0.1, mid: 0.8, high: 0.4}  # Grid search 결과
         smd: {low: -0.1, mid: 0.7, high: 0.5}
   ```

**예상 소요 시간**: 3-4일
- Grid search: 1일
- Logistic regression: 1일
- 데이터셋별 최적화: 1일
- 문서화: 1일

**난이도**: 중

**의존성**:
- 외부: `scikit-learn` (logistic regression)
- 내부: C2 완료 권장 (validation set 필요)

**성공 기준**:
- [ ] Grid search로 최적 가중치 발견
- [ ] SKAB 데이터셋 AUC-PR > 0.5 (현재 < 0.3)
- [ ] 학습된 가중치가 휴리스틱보다 +0.1 이상 개선
- [ ] `docs/RQ_JUSTIFICATION.md` RQ1 섹션 업데이트

**기존 TODO.md 연결**: Week 2 (Frequency Analysis for SKAB)

**담당**: ML Engineer

---

### H2: 베이스라인 탐지기 추가

**문제**: 현재 비교 대상이 Rule/ML(kNN)/Hybrid/SpecCNN만 있어 업계 표준 방법과 비교 불가능합니다.

**영향**:
- 성능 비교 불완전
- 논문 심사 시 "왜 IsolationForest와 비교 안 했나?" 질문 예상
- SOTA 대비 우위 입증 불가

**작업 내용**:

1. **IsolationForest 베이스라인**
   ```python
   # experiments/baseline_isolation_forest.py 생성

   # C1과 동일하지만 베이스라인 목적으로 별도 구현
   # 시간 윈도우 특징 없이 raw value만 사용 (공정 비교)
   ```

2. **Facebook Prophet**
   ```python
   # experiments/baseline_prophet.py 생성

   from prophet import Prophet

   def detect_with_prophet(data):
       """
       Prophet으로 예측 후 잔차가 큰 지점을 이상치로 판단
       - 잔차 = |actual - predicted|
       - 임계값: 잔차 분포의 95th percentile
       """
   ```

3. **LSTM Autoencoder** (선택)
   ```python
   # experiments/baseline_lstm_ae.py 생성

   # 시퀀스 재구성 기반 이상 탐지
   # 재구성 오류가 큰 구간 = 이상치
   ```

4. **비교 실험**
   ```python
   # scripts/baseline_comparison.py 생성

   baselines = ["rule", "knn", "isolation_forest", "prophet", "lstm_ae"]

   for baseline in baselines:
       # 각 베이스라인으로 실험
       # 결과를 표로 정리
   ```

**예상 소요 시간**: 5-7일
- IsolationForest: 1일 (C1과 중복)
- Prophet: 2일
- LSTM-AE: 3일
- 비교 실험: 1일

**난이도**: 중-상

**의존성**:
- 외부: `prophet`, `torch`
- 내부: C1, C2 완료 후 권장

**성공 기준**:
- [ ] 3개 이상 베이스라인 추가
- [ ] 모든 베이스라인과 성능 비교표 생성
- [ ] Hybrid가 최소 1개 베이스라인보다 유의하게 우수 (p < 0.05)
- [ ] `EXPERIMENT_REPORT.md`에 비교 결과 추가

**기존 TODO.md 연결**: Week 3-4 (Baseline Expansion)

**담당**: ML Engineer

---

### H3: Event-wise 메트릭 상관관계 분석 (RQ3)

**문제**: RQ3 "Point-wise F1과 Event-wise F1 상관관계"를 검증할 배치 실험이 미완료되었습니다.

**영향**:
- RQ3 답변 불가
- 제조업 실무 관련성 입증 불가
- Event-wise 메트릭 유용성 미검증

**작업 내용**:

1. **배치 평가 스크립트**
   ```python
   # scripts/batch_eval.py 확장

   datasets = ["synthetic", "skab", "smd", "aihub71802"]
   detectors = ["rule", "ml", "hybrid", "speccnn"]

   results = []
   for dataset in datasets:
       for detector in detectors:
           # Point-wise F1, Event-wise F1 계산
           results.append({
               "dataset": dataset,
               "detector": detector,
               "point_f1": ...,
               "event_f1": ...,
               "detection_delay": ...,
               "lead_time": ...
           })
   ```

2. **상관관계 분석**
   ```python
   # scripts/correlation_analysis.py 생성

   from scipy.stats import pearsonr, spearmanr

   point_f1 = [r["point_f1"] for r in results]
   event_f1 = [r["event_f1"] for r in results]

   # Pearson correlation
   corr, p_value = pearsonr(point_f1, event_f1)

   # Spearman correlation (비모수)
   corr_s, p_value_s = spearmanr(point_f1, event_f1)
   ```

3. **시각화**
   ```python
   import matplotlib.pyplot as plt

   plt.scatter(point_f1, event_f1)
   plt.xlabel("Point-wise F1")
   plt.ylabel("Event-wise F1")
   plt.title(f"Correlation: {corr:.3f} (p={p_value:.3f})")
   plt.savefig("correlation_plot.png")
   ```

4. **RQ3 답변 작성**
   ```markdown
   # docs/RQ_JUSTIFICATION.md 업데이트

   ## RQ3: Point-wise vs Event-wise F1 상관관계

   **결과**: Pearson correlation = 0.73 (p < 0.001)
   - 중간-강한 양의 상관관계 발견
   - 그러나 완벽한 선형 관계는 아님 (r² = 0.53)
   - Event-wise F1이 제조업 실무에서 더 중요한 지표임을 시사
   ```

**예상 소요 시간**: 4-5일
- 배치 평가: 2일 (실험 실행 시간 포함)
- 상관관계 분석: 1일
- 시각화 및 해석: 1일
- 문서화: 1일

**난이도**: 중

**의존성**:
- 외부: `scipy`, `matplotlib`
- 내부: C1, C2 완료 필수 (다중 실험 데이터 필요)

**성공 기준**:
- [ ] 4개 데이터셋 × 4개 탐지기 = 16개 실험 완료
- [ ] Pearson/Spearman 상관계수 계산 (p-value 포함)
- [ ] Scatter plot 생성
- [ ] RQ3 답변 작성 (상관관계 해석)
- [ ] 발견: 상관관계가 0.5-0.8 (중간-강함)

**기존 TODO.md 연결**: Week 4 (Correlation Study)

**담당**: Research Lead

---

## 🟡 Medium Priority (1개월 내 완료)

### M1: RAG 의미론적 검색 개선

**문제**: 현재 `explain_rag.py`가 TF-IDF만 사용하여 의미론적 유사도를 반영하지 못합니다.

**영향**:
- Phase 2 설명 품질 제한
- 관련성 낮은 문서 검색 가능성
- LLM 토큰 낭비

**작업 내용**:

1. **임베딩 기반 검색**
   ```python
   # experiments/explain_rag.py 수정

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("all-MiniLM-L6-v2")

   # 문서 임베딩
   doc_embeddings = model.encode(documents)

   # 쿼리 임베딩
   query_embedding = model.encode(query)

   # 코사인 유사도로 top-k 검색
   similarities = cosine_similarity(query_embedding, doc_embeddings)
   top_k_indices = np.argsort(similarities)[-k:]
   ```

2. **하이브리드 검색** (TF-IDF + 임베딩)
   ```python
   # TF-IDF 스코어와 임베딩 유사도 결합
   final_score = 0.3 * tfidf_score + 0.7 * embedding_similarity
   ```

3. **벡터 DB 통합** (선택)
   ```python
   import faiss

   # FAISS 인덱스 생성
   index = faiss.IndexFlatIP(embedding_dim)
   index.add(doc_embeddings)

   # 빠른 검색
   D, I = index.search(query_embedding, k=3)
   ```

**예상 소요 시간**: 3-4일
- 임베딩 검색: 2일
- FAISS 통합: 1일
- 테스트: 1일

**난이도**: 중

**의존성**:
- 외부: `sentence-transformers`, `faiss-cpu`
- 내부: 없음

**성공 기준**:
- [ ] 임베딩 기반 검색 구현
- [ ] TF-IDF 대비 검색 품질 개선 확인 (수동 평가)
- [ ] top-3 문서의 관련성 > 80%
- [ ] `llm_config.yaml`에 검색 방법 선택 옵션 추가

**기존 TODO.md 연결**: Phase 2 개선 (향후 Week)

**담당**: NLP Engineer

---

### M2: 베이지안 규칙 학습 자동화

**문제**: 현재 `llm_config.yaml`의 베이지안 규칙이 하드코딩되어 확장성이 제한됩니다.

**영향**:
- Phase 2 확장성 부족
- 새로운 데이터셋 대응 불가
- 규칙 추가 시 수동 작업 필요

**작업 내용**:

1. **규칙 추출 프레임워크**
   ```python
   # experiments/rule_learning.py 생성

   from sklearn.tree import DecisionTreeClassifier

   # 실험 결과에서 특징 추출
   features = [
       "imbalance_ratio",  # 불균형 정도
       "snr",              # 신호 대 잡음 비
       "pattern_type",     # 스파이크/스텝/드리프트
       "sequence_length"
   ]

   # 목표: 최적 cost_fn_ratio 예측
   target = "optimal_cost_ratio"

   # 결정 트리로 규칙 학습
   dt = DecisionTreeClassifier(max_depth=3)
   dt.fit(feature_matrix, target_vector)

   # 규칙 추출
   rules = extract_rules_from_tree(dt)
   ```

2. **규칙 형식 변환**
   ```python
   # 결정 트리 규칙 → 베이지안 규칙 변환

   rule = {
       "condition": "imbalance_ratio < 0.1 AND snr > 10",
       "recommendation": "cost_fn_ratio should be close to 1 (balanced)",
       "prior_adjustment": {
           "C01": "*0.9",
           "C10": "*1.1"
       }
   }
   ```

3. **자동 규칙 업데이트**
   ```python
   # 실험 완료 후 자동으로 규칙 재학습

   if new_experiments_completed:
       learned_rules = learn_bayesian_rules(all_experiment_results)
       update_llm_config(learned_rules)
   ```

**예상 소요 시간**: 5-7일
- 규칙 추출: 3일
- 형식 변환: 2일
- 자동화: 2일

**난이도**: 상

**의존성**:
- 외부: `scikit-learn`
- 내부: C2, H3 완료 (충분한 실험 데이터 필요)

**성공 기준**:
- [ ] 결정 트리로 10개 이상 규칙 자동 추출
- [ ] 학습된 규칙이 하드코딩 규칙보다 성능 우수
- [ ] 새로운 데이터셋에 자동 적용 가능
- [ ] `docs/HANDBOOK.md`에 규칙 학습 가이드 추가

**기존 TODO.md 연결**: Phase 2 개선 (향후 Week)

**담당**: Research Lead + ML Engineer

---

### M3: 단위 테스트 커버리지 증대

**문제**: 현재 `test_rule_detector.py`, `test_explain.py`만 존재하여 커버리지가 매우 낮습니다.

**영향**:
- 리팩토링 위험 증가
- 버그 발견 지연
- 코드 품질 저하

**작업 내용**:

1. **pytest 기반 테스트 suite 구축**
   ```bash
   # tests/ 디렉토리 생성
   mkdir tests

   # 구조
   tests/
   ├── test_detectors.py          # 4개 탐지기
   ├── test_calibration.py        # 3개 보정 방법
   ├── test_metrics.py            # 메트릭 계산
   ├── test_data_loaders.py       # 데이터 로더
   ├── test_cost_threshold.py     # 비용 최적화
   ├── test_feature_bank.py       # 특징 추출
   └── test_integration.py        # 전체 파이프라인
   ```

2. **단위 테스트 작성**
   ```python
   # tests/test_detectors.py

   import pytest
   from experiments.rule_detector import detect_rule_based

   def test_rule_detector_basic():
       data = [1, 1, 1, 10, 1, 1]  # 명확한 스파이크
       result = detect_rule_based(data, threshold=2.0)
       assert result["preds"][3] == 1  # 4번째가 이상치

   def test_rule_detector_no_anomaly():
       data = [1, 1, 1, 1, 1, 1]  # 정상
       result = detect_rule_based(data, threshold=2.0)
       assert sum(result["preds"]) == 0

   def test_rule_detector_edge_case():
       data = []  # 빈 데이터
       with pytest.raises(ValueError):
           detect_rule_based(data)
   ```

3. **커버리지 측정**
   ```bash
   # pytest-cov 설치
   pip install pytest-cov

   # 커버리지 측정
   pytest --cov=experiments tests/

   # 목표: 커버리지 > 80%
   ```

4. **CI/CD 통합** (선택)
   ```yaml
   # .github/workflows/test.yml

   name: Tests
   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Set up Python
           uses: actions/setup-python@v2
         - name: Run tests
           run: pytest --cov=experiments tests/
   ```

**예상 소요 시간**: 7-10일
- 테스트 작성: 5일
- 커버리지 측정: 1일
- 버그 수정: 2일
- CI/CD 통합: 2일

**난이도**: 중

**의존성**:
- 외부: `pytest`, `pytest-cov`
- 내부: 없음

**성공 기준**:
- [ ] 모든 주요 모듈에 단위 테스트 작성
- [ ] 커버리지 > 80%
- [ ] 모든 테스트 통과
- [ ] CI/CD 파이프라인 구축 (선택)

**기존 TODO.md 연결**: 코드 품질 개선 (향후 Week)

**담당**: All Developers

---

## 🟢 Low Priority (선택적 개선)

### L1: 코드 중복 제거 및 리팩토링

**문제**: 메트릭 계산 로직과 데이터 로더 간 코드 중복이 존재합니다.

**작업 내용**:

1. **공통 유틸리티 모듈**
   ```python
   # experiments/utils/metrics_common.py
   # experiments/utils/loader_common.py

   # 중복 코드 추출
   ```

2. **리팩토링**
   - DRY (Don't Repeat Yourself) 원칙 적용
   - 함수 재사용성 증대

**예상 소요 시간**: 3-5일
**난이도**: 하
**성공 기준**: 코드 라인 10% 감소

**기존 TODO.md 연결**: 코드 품질 개선

---

### L2: 문서 미완료 섹션 작성

**문제**: `RELATED_WORK.md`에 [TODO] 섹션이 있고, 목표 40-50편 대비 29편만 조사됨.

**작업 내용**:

1. **문헌 조사 확장**
   - 추가 11-21편 조사
   - 각 논문 요약 및 관련성 분석

2. **TODO 섹션 완료**
   - [TODO] 표시 제거
   - 분석 내용 작성

**예상 소요 시간**: 5-7일
**난이도**: 중
**성공 기준**: 40편 이상 문헌 조사 완료

**기존 TODO.md 연결**: Week 1 (일부 완료, 확장 필요)

---

### L3: 추가 데이터셋 지원

**문제**: 현재 SKAB, SMD, AIHub71802만 지원합니다.

**작업 내용**:

1. **새로운 데이터셋 추가**
   - NASA Bearing Dataset
   - Yahoo Anomaly Dataset
   - TODS Benchmark datasets

2. **로더 구현**
   ```python
   # experiments/data/loader_nasa_bearing.py
   # experiments/data/loader_yahoo.py
   ```

**예상 소요 시간**: 3-5일 (데이터셋당)
**난이도**: 중
**성공 기준**: 2개 이상 데이터셋 추가

**기존 TODO.md 연결**: 향후 확장

---

### L4: 시각화 개선

**문제**: 현재 시각화가 ROC/PR/Calibration curve에 국한됩니다.

**작업 내용**:

1. **대시보드 추가**
   - 탐지 결과 시계열 플롯
   - 주파수 분석 시각화
   - 메트릭 비교 히트맵

2. **인터랙티브 시각화** (선택)
   ```python
   import plotly.express as px

   # 인터랙티브 시계열 플롯
   ```

**예상 소요 시간**: 3-5일
**난이도**: 하-중
**성공 기준**: 5종류 이상 시각화 추가

**기존 TODO.md 연결**: 향후 개선

---

## 📅 실행 타임라인 (8주)

### Week 1-2 (2025-11-25 ~ 2025-12-08): Critical 완료

**목표**: C1, C2 완료
- [ ] C1: ML 탐지기 교체 (IsolationForest)
- [ ] C2: 다중 시드 실험 + Bootstrap CI

**마일스톤**: 통계적으로 검증된 결과 확보

---

### Week 3-4 (2025-12-09 ~ 2025-12-22): High 완료

**목표**: H1, H2, H3 완료
- [ ] H1: SpecCNN 가중치 최적화
- [ ] H2: 베이스라인 추가 (IsolationForest, Prophet)
- [ ] H3: RQ3 검증 (상관관계 분석)

**마일스톤**: 모든 RQ 답변 완료

---

### Week 5-6 (2025-12-23 ~ 2026-01-05): Medium 진행

**목표**: M1, M2, M3 시작
- [ ] M1: RAG 의미론적 검색 (완료)
- [ ] M2: 베이지안 규칙 학습 (50% 진행)
- [ ] M3: 단위 테스트 (50% 진행)

**마일스톤**: Phase 2 품질 개선

---

### Week 7-8 (2026-01-06 ~ 2026-01-20): Medium 완료 + Low 선택

**목표**: Medium 완료, Low 선택적 진행
- [ ] M2, M3 완료
- [ ] L1, L2, L3, L4 중 선택적 진행

**마일스톤**: 논문 투고 준비 완료

---

## 🎯 최종 목표

**Week 8 종료 시 달성 상태**:
- ✅ 모든 Critical 항목 완료
- ✅ 모든 High 항목 완료
- ✅ 대부분 Medium 항목 완료
- 🔄 일부 Low 항목 진행 중

**예상 프로젝트 성숙도**: 90%
**TRL**: 5-6 (System validation)
**논문 투고**: 준비 완료

---

## 📊 진행 상황 추적

**추적 방법**:
1. 각 작업 시작 시 `TODO.md`에 체크박스 업데이트
2. 주간 진행 상황 `EXPERIMENT_REPORT.md`에 기록
3. 완료 시 `REVIEW_YYYYMMDD.md` 업데이트

**주간 리뷰**:
- 매주 금요일 진행 상황 점검
- 블로커 식별 및 해결 방안 논의
- 필요시 우선순위 조정

---

## 🔗 관련 문서

- **검토 결과**: `REVIEW_20251125.md`
- **기존 계획**: `TODO.md`
- **실험 결과**: `EXPERIMENT_REPORT.md`
- **연구 문서**: `docs/HANDBOOK.md`, `docs/RQ_JUSTIFICATION.md`

---

## 📝 변경 이력

- **2025-11-25**: 초안 작성 (검토 결과 기반)

---

**작성 완료**: 2025-11-25
**다음 업데이트 예정**: Week 2 (2025-12-08) - Critical 항목 완료 후
