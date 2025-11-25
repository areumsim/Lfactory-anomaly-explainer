# LLM-Guided Local Anomaly Detection – 온보딩 & 연구 핸드북

이 문서는 신규 참여자를 위한 단일 “첫 문서”입니다. 온보딩, 연구 방법, 데이터셋, 실험, 모델, 테스트, 트러블슈팅, 재현성, 컨트리뷰팅을 하나로 통합합니다.

- 대상: 본 프로젝트에 합류하는 AI 연구자/소프트웨어 엔지니어
- 읽고 나면 가능한 일: 핵심 실험 실행, 지표/아티팩트 이해, 탐지기/피처 확장, 문제 디버깅, 재현 가능한 기여

## 1. 목표와 저장소 지도
- 연구 동기: Rule 베이스라인 + ML 탐지기 결합으로 AUC‑PR 및 기대비용 개선, 보정(ECE)으로 신뢰도 정렬. Detect를 우선 성숙 → Explain(근거) / Act(정책) 단계 확장.
- 엔지니어링 목표: 최소 의존성, 단일 CLI, 표준 runs/ 산출물 구조, per‑run REPORT.md 및 글로벌 인덱스.
- 구조 요약: experiments/(main_experiment, data 로더/정규화, detectors, metrics/calibration/cost/result_manager/feature_bank), scripts/(리포트/스윕/CI/정리), 최상위 문서들.

## 2. 10분 퀵스타트
- 환경: Python 3.9+, (선택) matplotlib pandas pyarrow
- 합성 스모크(ML+보정+비용)
  PYTHONPATH=. python -m experiments.main_experiment \
    --dataset synthetic --mode detect --detector ml \
    --length 300 --seed 123 \
    --out-json runs/quick/run.json --out-csv runs/quick/preds.csv \
    --plots-dir runs/quick/plots --calibrate isotonic --ece-bins 10 \
    --cost-optimize --costs 0,1,5,0 --apply-cost-threshold
- 실데이터 스모크(SKAB robust rule)
  PYTHONPATH=. python -m experiments.main_experiment \
    --dataset SKAB --mode detect --detector rule --z-robust \
    --data-root $DATA_ROOT --split test --seed 42 \
    --run-id skab_smoke --plots-dir runs/skab_smoke/plots
- 유틸 스크립트: calibration_eval, cost_ab_report, ablation_sweep, batch_eval, organize_runs, enforce_policy

## 3. 용어
- 포인트 지표 vs 이벤트 지표(세ग먼트, Delay, Lead), ROC/PR AUC, 보정/ECE, 비용행렬과 기대비용.

## 4. 연구 질문과 가설

본 프로젝트는 통합 이상탐지 프레임워크를 구성하는 4가지 비자명한(non-trivial) 연구 질문(RQ)을 다룹니다.

### 4.1 RQ1: 주파수 도메인 vs 시간 도메인 특징

**질문**: 주파수 도메인 특징(SpecCNN을 통한 DFT 대역)이 시간 도메인 특징(Rule 탐지기 롤링 통계)보다 제조 스파이크/단계/드리프트 이상에 더 효과적인가?

**비자명한 이유**:
- 스파이크는 일시적(duration << window) → DFT가 놓칠 수 있음 (leakage 효과)
- 단계 변화는 불연속 → DFT는 주기성 가정 (Gibbs 현상)
- 답은 이상 유형, SNR, 샘플링 레이트에 의존

**검증 가능한 가설**:
- H1.1: SpecCNN이 주기적 데이터셋(예: SKAB valve)에서 Rule보다 높은 AUC-PR 달성
- H1.2: Rule이 일시적 스파이크(< 10 샘플)에서 더 높은 Event Recall 달성
- H1.3: Hybrid(Rule + SpecCNN)가 최고 단일 탐지기 대비 ≥10% 개선

**수용 기준**:
- SKAB, AIHub에 대한 주파수 분석 완료 (PSD, 식별 대역)
- 제거 연구: Rule-only, SpecCNN-only, Hybrid with statistical significance (p < 0.05, n ≥ 5 seeds)
- `experiments/ABLATION_FREQUENCY_VS_TIME.md`에 문서화

**의존성**: → RQ2 (앙상블 방법은 어떤 특징이 더 나은지에 의존)

**참조**: `docs/RQ_JUSTIFICATION.md` Section 1

---

### 4.2 RQ2: 앙상블 방법

**질문**: 어떤 앙상블 방법(선형, 곱, 최댓값, 학습)이 보정-비용 트레이드오프에서 최고 성능을 달성하는가?

**비자명한 이유**:
- 선형은 가장 단순하지만 최적이 아닐 수 있음
- 곱은 합의 필요 (보수적, 이벤트 놓칠 수 있음)
- 최댓값은 공격적 (recall 편향, false alarm 증가 가능)
- 학습은 작은 검증 세트에서 과적합 가능

**검증 가능한 가설**:
- H2.1: 선형이 가장 낮은 ECE (< 0.05) 달성
- H2.2: 최댓값이 높은 불균형(> 1:50)에서 FN >> FP일 때 가장 낮은 기대비용 달성
- H2.3: 학습이 검증 크기 ≥ 500일 때 ≥5% AUC-PR 개선

**수용 기준**:
- 4가지 방법 모두 ≥3 데이터셋에서 구현 및 테스트
- 다중 지표 평가: AUC-PR, ECE, 기대비용, Point F1, Event F1
- 파레토 프론티어 플롯: ECE (x) vs 기대비용 (y)
- 통계 검정 (paired t-test, n ≥ 5 seeds)
- `experiments/ABLATION_ENSEMBLE_METHODS.md`에 문서화

**의존성**: RQ1 → RQ2 (특징 선택이 앙상블에 영향), RQ2 → RQ3 (앙상블이 지표에 영향)

**참조**: `docs/RQ_DEPENDENCIES.md` Section 2.1

---

### 4.3 RQ3: 포인트 지표 vs 이벤트 지표 상관관계

**질문**: 탐지기 전반에 걸쳐 포인트 단위 F1과 이벤트 단위 F1 간 상관관계는?

**비자명한 이유**:
- Point F1은 모든 FP 타임스탬프 패널티 → precision 선호
- Event F1은 이벤트 탐지 여부만 고려 → recall 선호
- 예: 보수적 탐지기는 낮은 Point F1이지만 높은 Event F1 가능

**검증 가능한 가설**:
- H3.1: 상관관계는 중간 수준 (0.5 < ρ < 0.8)
- H3.2: 긴 이벤트(> 50 샘플)에서 짧은 이벤트(< 10 샘플)보다 상관관계 높음
- H3.3: Rule 탐지기가 ML 탐지기보다 Event/Point F1 비율 높음

**수용 기준**:
- 전체 배치 평가 (상관관계 분석용 ≥ 50 데이터 포인트)
- 산점도: Point F1 (x) vs Event F1 (y), 탐지기 유형별 색상 구분
- Pearson + Spearman 상관관계 with 95% CI (bootstrap n=1000)
- 이상 길이별 계층화, `docs/METRICS_TRADEOFF.md`에 문서화

**의존성**: RQ2 → RQ3 (앙상블 선택이 지표 트레이드오프에 영향), RQ3 → RQ4 (지표 선택이 비용 설계에 영향)

**참조**: `docs/RQ_JUSTIFICATION.md` Section 3

---

### 4.4 RQ4: 데이터셋 특성에 대한 비용 민감도

**질문**: FN/FP 비용 비율은 데이터셋 불균형과 SNR에 따라 어떻게 변해야 하는가?

**비자명한 이유**:
- 높은 불균형 + 낮은 SNR: FN 비용 증가 시 임계값 붕괴 → 더 많은 FP
- 최적 비율은 (불균형, SNR, 탐지기 품질)의 함수

**검증 가능한 가설**:
- H4.1: 비용 비율은 중간 범위(1:10 ~ 1:50)에서 불균형과 선형 증가
  - 공식: `cost_ratio ≈ 0.5 × imbalance`
- H4.2: 비율은 극단적 불균형(> 1:100)에서 ~20에서 포화
- H4.3: SNR 감소 시 최적 비율 감소 (낮은 SNR에서 더 보수적)
- H4.4: 잘 보정된 탐지기(ECE < 0.05)가 더 강한 비용-임계값 상관관계 표시

**수용 기준**:
- 그리드 탐색: 불균형 × 비용 비율 → 기대비용 히트맵
- 각 데이터셋에 대한 SNR 계산 (공식 문서화)
- 적합 모델: `cost_ratio = f(imbalance, SNR)` with R² > 0.7
- 다양한 시나리오에 대한 실무자 가이드 테이블
- `experiments/COST_SENSITIVITY_ANALYSIS.md`에 문서화

**의존성**: RQ3 → RQ4 (지표 상관관계가 비용 행렬 설계에 영향), RQ4 → RQ1 (비용이 이상 유형 탐지에 의존)

**참조**: `docs/RQ_DEPENDENCIES.md` Section 3

---

### 4.5 수용 기준 (전체)

**출판 품질 결과를 위한 기준**:
- ≥2개 비자명한 RQ를 엄격하게 답변 (통계적 유의성, p < 0.05)
- 모든 비교에 95% 신뢰구간 포함 (n ≥ 5 seeds)
- 정직한 한계 문서화 (과장 금지)
- 베이스라인 비교는 방법의 강점과 약점 모두 표시

**표준 실험 배터리**:
- 단일 실행 → ROC/PR/ECE + REPORT.md
- 보정 비교 → ECE 감소 검증
- 비용 A/B → 기대비용 감소 정량화
- 제거 → 특징/앙상블/파라미터 민감도
- 배치 평가 → 교차 데이터셋 일반화

**보고 형식**:
- 실행별 REPORT.md: 메타데이터, 탐지기 설정, 지표(포인트 + 이벤트), 보정, 비용 A/B, 아티팩트 링크
- 배치 요약: 집계 CSV + 통계 검정
- 교차 참조: 상세 분석 문서 링크 (RQ_JUSTIFICATION.md 등)

## 5. 데이터셋 카드/검증
- 공통: CODE_ROOT≠DATA_ROOT, --data-root/--datasets-cfg 사용, 정규화 보존율 ≥95%
- SKAB: 시나리오 폴더, 라벨 컬럼 없으면 보수적 폴백, 구분자 혼재 주의
- SMD: test↔test_label 정합(Trim/Pad), 구분자/길이 주의
- AIHub71802: binary/risk4 스킴, 메트릭은 >0→1로 이진 변환, 포맷 혼재 주의
- 빠른 검증: scripts/inspect_dataset.py로 head/라벨분포/meta 확인

## 6. 실험 플레이북
- 단건: --out-json/--out-csv/--plots-dir(+ --features-csv)
- 보정 비교: calibration_eval(ECE/AUC)
- 비용 A/B: cost_ab_report(+ --apply-cost-threshold)
- 스윕: ablation_sweep(alpha/window/k/quantile)
- 배치/정리: batch_eval, organize_runs, enforce_policy

## 7. 모델/확장
- 내장 탐지기: rule/robust/ml/hybrid/speccnn-lite
- 새 탐지기 추가: {scores,preds,params} 반환 계약, CLI 플래그/문서화, 보정/비용/리포트 호환
- FeatureBank 확장: feature_bank.py 함수 추가(단일 행 CSV로 저장)

## 8. Phase 2 – LLM 기반 설명 (프로토타입)

Phase 2는 이상탐지 결과 설명을 위한 RAG(Retrieval-Augmented Generation)와 Bayes 사전 조정을 추가합니다. 이는 LLM 통합을 시연하는 프로토타입이며, 핵심 탐지(Phase 1)에는 불필요합니다.

### 8.1 빠른 시작

- LLM 없이 테스트 (TF-IDF 검색만):
python scripts/test_explain.py

- OpenAI GPT-3.5로 테스트:
python scripts/test_explain.py --llm-provider openai_gpt35

- 로컬 EXAONE 모델로 테스트:
python scripts/test_explain.py --llm-provider local_exaone_35_78b

- 설명과 함께 실험 실행:
PYTHONPATH=. python -m experiments.main_experiment \
  --dataset SKAB --mode detect --detector hybrid \
  --data-root $DATA_ROOT --split test --seed 42 \
  --run-id skab_explain --explain --llm-provider local_exaone_35_78b

### 8.2 LLM 제공자 선택

| 제공자 | 타입 | 사용 사례 | 의존성 | 속도 |
|--------|------|-----------|--------|------|
| None | TF-IDF만 | 최소 설정, LLM 없음 | 표준 라이브러리 | 즉시 |
| openai_gpt35 | OpenAI API | 빠른 프로토타이핑 | openai | ~5-10초 |
| openai_gpt4o | OpenAI API | 고품질 | openai | ~10-20초 |
| local_exaone_35_78b | 로컬 LLM | 온프레미스, API 비용 없음 | torch, transformers | ~20-40초 |
| local_exaone_35_32b | 로컬 LLM | 더 빠른 로컬 | torch, transformers | ~10-20초 |
| local_exaone_30_78b | 로컬 LLM | 안정적 베이스라인 | torch, transformers | ~20-40초 |

`experiments/llm_config.yaml` 설정:
- OpenAI 제공자용 API 키
- 로컬 제공자용 모델 경로 (cuda:3의 EXAONE 모델)
- 문서 소스: README.md, HANDBOOK.md, EVALUATION_PROTOCOL.md
- 검색 설정: top_k=3, chunk_size=500

### 8.3 생성된 설명

--explain이 활성화되면 main_experiment.py가 EXPLANATIONS.md를 생성하며 다음을 포함:

1. **쿼리 기반 설명**: 다음과 같은 질문에 대한 답변:
   - "이상탐지에서 보정을 사용해야 하는 이유는?"
   - "포인트 단위 F1과 이벤트 단위 F1의 차이는?"
   - "이 데이터셋에 대한 비용 행렬을 어떻게 설정해야 하는가?"

2. **검색된 증거**: 인용이 포함된 상위 3개 가장 관련성 높은 문서 청크

3. **Bayes 사전 권장사항**: 데이터셋 특성 기반 하드코딩 규칙:
   - 높은 불균형(>0.1) → FP 대비 FN 비용 증가
   - 낮은 SNR(<3.0) → 보수적 임계값 사용, Rule 또는 Hybrid 선호
   - 높은 불균형 + 낮은 SNR → 강력한 보정 권장

4. **LLM 합성** (제공자 지정 시): 검색된 문서와 Bayes 규칙을 결합한 자연어 설명

출력 구조 예시:
```markdown
# LLM 기반 설명

## 1. 보정
**쿼리**: 이상탐지에서 보정을 사용해야 하는 이유는?
**검색된 증거**:
- [EVALUATION_PROTOCOL.md] "보정은 예측 확률이 실제 빈도와 일치하도록 보장..."
**Bayes 권장사항**:
- 데이터셋의 불균형=0.12 → 비용 민감 임계값에 보정 필수
**LLM 응답**: [사용 가능한 경우] "보정은 다음 이유로 필수적입니다..."
```

### 8.4 Phase 2 테스트

테스트 스위트 `scripts/test_explain.py`는 5가지 테스트 케이스 포함:
1. **기본 검색**: LLM 없이 TF-IDF
2. **LLM 생성**: 지정된 제공자로 전체 RAG
3. **Bayes 규칙**: 다양한 컨텍스트에 대한 사전 조정 로직
4. **문서 소스**: 로드된 문서 검증
5. **오류 처리**: 엣지 케이스 (잘못된 제공자, 빈 쿼리, 큰 컨텍스트)

모든 테스트 실행:
python scripts/test_explain.py --test all

특정 테스트 실행:
python scripts/test_explain.py --test bayes

### 8.5 알려진 한계 (Phase 2)

- **Bayes 규칙은 하드코딩**: 데이터에서 학습되지 않음; 불균형(0.1)과 SNR(3.0)에 대한 고정 임계값
- **가설 검증 없음**: 규칙이 아직 RQ4 실험 결과와 대조 검증되지 않음
- **단일 검색 전략**: TF-IDF만; 의미론적 임베딩 또는 재순위 없음
- **로컬 LLM 추론 시간**: cuda:3에서 쿼리당 20-40초; 프로덕션에는 배치 처리 고려
- **피드백 루프 없음**: 설명이 탐지기 파라미터를 업데이트하지 않음 (Act 단계 미구현)

계획된 개선사항:
- Week 3: RQ4 비용 민감도 결과와 대조하여 Bayes 규칙 검증
- Week 4: 의미론적 검색 옵션 추가 (sentence-transformers)
- Week 5-6: Act 단계 구현 (정책 권장사항)

### 8.6 Phase 2 의존성

선택사항 (Phase 2 사용 시만 설치):
- OpenAI API: `pip install openai` (API 키 필요)
- 로컬 LLM: `pip install torch transformers` (GPU 필요, 7.8B 모델용 ~15GB VRAM)

핵심 탐지(Phase 1)는 Python 3.9+ 표준 라이브러리 외 외부 의존성 없음.

---

## 9. 알려진 이슈 및 한계 (Phase 1)

이 섹션은 현재 구현의 한계와 해결방법을 문서화합니다. 모든 이슈는 TODO.md에서 수정 우선순위와 함께 추적됩니다.

### 9.1 ML 탐지기: 시간 구조 무시 (치명적)

**이슈**: `experiments/ml_detector.py` 50-51행이 값으로만 훈련 데이터 정렬:
```python
X_sorted = X[np.argsort(X)]  # 값으로 정렬, 시간 순서 손실
```

**영향**:
- 시간 패턴 탐지 불가 (단계 변화, 시간 경과에 따른 드리프트)
- 정상적인 시간 진화의 일부인 경우에도 급격한 변화를 이상으로 잘못 분류 가능
- 비정상 프로세스가 있는 데이터셋에서 효과 감소

**정량화된 영향** (향후 실험에서 측정 예정):
- 예상: 시간 인식 베이스라인 대비 단계/드리프트 이상에 대해 Event F1 10-20% 감소
- [TODO] 수정 구현 후 SKAB valve 데이터셋에서 측정

**해결방법**: 단계/드리프트 이상에 대해 Rule 또는 Hybrid 탐지기 사용 (Rule 탐지기는 롤링 윈도우 사용, 시간 컨텍스트 보존)

**수정 우선순위**: 치명적 (Week 2)
- 계획된 수정: 슬라이딩 윈도우 kNN 또는 Matrix Profile 기반 거리 구현
- 참조: TODO.md Part 2.1

### 9.2 SpecCNN: 점수 식별력 실패 (치명적)

**이슈**: SKAB 및 SMD 데이터셋에서 **모든 이상 점수가 0.0** (분산 없음).

**근본 원인** (`experiments/spec_cnn.py:43-66`):
- 음수 저주파 가중치 (`w_low = -0.2`)가 저주파 신호에서 점수를 음수로 만듦
- `max(0, score)` 클리핑이 모든 음수 점수를 0으로 강제
- 결과: 모든 점수 = 0.0, AUC-PR = 0.0

**측정된 영향** (Phase 1 실험, 2025-10-01):
- **SKAB**: Point F1=0.518 ✅, Event F1=1.0 ✅, AUC-PR=0.0 ❌
- **SMD**: Point F1=0.173 ✅, Event F1=1.0 ✅, AUC-PR=0.0 ❌
- 이진 예측은 작동하지만, 순위화나 비용 민감 학습을 위한 **점수 분포 없음**

**연구 영향**:
- ❌ **RQ1** (주파수 특징): 검증 불가 - 점수가 식별하지 못함
- ❌ **RQ4** (비용 민감): SpecCNN에서 검증 불가 - 점수 분포 필요
- ⚠️ **RQ2** (앙상블): 부분적 - 이진 예측은 작동, 점수는 안됨

**해결방법**:
- Event F1 평가를 위해 이진 예측 사용 (여전히 유효)
- RQ1 검증을 위해 대안 주파수 도메인 탐지기 사용 (Wavelet, Spectral Residual)

**수정 옵션**:
- **Option A** (2-3주): 적응형 주파수 대역 선택 구현, 클리핑 제거
- **Option B** (1-2일, 권장): 한계 인정, 문서화하고 비용 최적화(62.81% 감소)를 주요 기여로 전환

**상세 분석**: `runs/status_report_2025-10-01/SPECCNN_DISCRIMINABILITY_ANALYSIS.md` 참조

**수정 우선순위**: 치명적 - 결정 필요 (Week 1)
- 참조: TODO.md Section 1.5.1 (SpecCNN 결정)
- 대안 탐지기: TODO.md Section 1.5.2 (Wavelet, Spectral Residual, IForest)

### 9.3 단일 시드: 통계적 신뢰도 없음 (필수)

**이슈**: 대부분의 실험이 단일 시드(42 또는 123) 사용; 신뢰구간 보고 없음

**영향**:
- 비교에 대한 통계적 유의성 주장 불가 (예: "Hybrid가 Rule보다 F1 5% 우수")
- 결과가 방법 품질이 아닌 무작위 변동 때문일 수 있음
- 출판 품질 기준 위반 (p < 0.05, n ≥ 5 필요)

**정량화된 영향**:
- Bootstrap CI 추정은 현재 방법에 대해 시드 간 F1 변동 ±3-5% 제안
- [TODO] 다중 시드 실험으로 검증

**해결방법**: 현재 결과는 방향성; CI 없이 "X가 Y보다 우수"와 같은 강한 주장 피함

**수정 우선순위**: 필수 (Week 2)
- 계획된 수정: 모든 RQ 실험을 n=5 시드(42, 123, 456, 789, 2024)로 실행
- 모든 지표에 대해 평균 ± 95% CI 계산
- 탐지기 비교용 paired t-test 추가
- 참조: TODO.md Part 3.3

### 9.4 이벤트 피크 정의 (경미)

**이슈**: `experiments/metrics.py` 95-96행이 세그먼트 끝으로 이벤트 피크 근사:
```python
# TODO: 도메인별 피크 정의 허용
peak_t = event_end  # 근사: 세그먼트 끝 사용
```

**영향**:
- 진짜 임계점이 세그먼트 중간에 발생하면 리드 타임 계산이 부정확할 수 있음
- 제조에서 진짜 피크는 종종 최대 편차 또는 임계값 위반에 해당

**정량화된 영향**: 낮음 (Lead Time 지표에만 영향; Detection Delay 및 F1은 영향 없음)

**해결방법**: 현재 근사는 초기 실험에 허용 가능; Lead Time은 여전히 방향적으로 정확

**수정 우선순위**: 경미 (Week 4)
- 계획된 수정: --peak-definition 옵션 추가 (선택: end, max, median, domain-specific)
- 선택적 컬럼을 통해 사용자가 피크 타임스탬프 제공 허용
- 참조: TODO.md Part 4.3

### 9.5 데이터셋별 이슈

**SKAB**:
- 일부 파일에 명시적 라벨 컬럼 없음 → 보수적 휴리스틱으로 폴백 (이상 과소계산 가능)
- 구분자 불일치 (일부 CSV는 ','대신 ';' 사용) → 로더가 폴백으로 처리

**SMD**:
- Test/label 길이 불일치 → 로더가 트리밍 또는 제로 패딩 (EVALUATION_PROTOCOL.md Section 7에 문서화)
- 높은 차원성 (38 채널) → 현재 탐지기는 단변량만

**AIHub71802**:
- 다중 클래스 라벨 (위험 레벨 0-3) → 다른 데이터셋과의 일관성을 위해 >0로 이진화
- 큰 파일 크기 → 정규화 캐시 필요 (Parquet 선호)

**해결방법**: 모든 데이터셋 로더에 폴백 로직 및 경고 포함; `scripts/inspect_dataset.py`로 검증

### 9.6 의존성 관리

**현재 상태**: Phase 1은 외부 의존성 없음 (matplotlib/pandas는 플롯/CSV용 선택사항)

**한계**: scipy, sklearn 등이 필요한 고급 방법 제한

**트레이드오프**: 재현성 및 배포 용이성 vs 방법 정교함

**수정 우선순위**: 낮음 (Week 5-6)
- 고급 방법용 선택적 의존성 추가 고려 (예: Matrix Profile, Isolation Forest 베이스라인)
- 핵심 탐지기는 의존성 없음 유지
- 참조: TODO.md Part 5.2

---

## 10. 테스트/트러블슈팅
- 권장 테스트: 보정 ECE 감소, 비용 임계 산술, 세그먼트/Delay/Lead 엣지, 로더 정합/보존율 경고
- 스모크: PYTHONPATH=. python scripts/test_rule_detector.py
- 골든파일: ROC/PR CSV, REPORT.md 스니펫
- 흔한 이슈: 경로/라벨 길이/의존성 부재/정규화 경고

## 11. 재현성
- 환경/의존성 최소, 개발 중 PYTHONPATH=.
- run.json: run_id/seed/git_sha/start_ts/지표/파라미터, config 스냅샷 저장
- 검증 체크: optimal≤fixed, 보정 후 ECE≤무보정, 재시드 AUC‑PR 안정, 부트스트랩 CI 스크립트

## 12. 기여 가이드
- 스타일: 작은 순수 함수, 명확한 파라미터, 출력에 파라미터 기록
- PR: 작게, 테스트/문서 동반, 가설/수용기준 연결
- 브랜치/커밋: 의미있는 이름, 간결한 명령형 메시지
