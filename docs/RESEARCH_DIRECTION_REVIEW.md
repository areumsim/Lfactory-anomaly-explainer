# LFactory 연구 방향 검토

**작성일**: 2026-03-30
**검토 관점**: 시니어 AI 엔지니어/리서처

---

## 1. 연구 목적 재확인

**핵심 연구 질문**: 제조 환경의 이상 탐지에서 ML 기반 탐지와 LLM 기반 설명을 어떻게 효과적으로 결합할 수 있는가?

**4개 RQ**:
- RQ1: 주파수 도메인 vs 시간 도메인 특징의 효과 비교
- RQ2: 앙상블 방법(linear/product/max/learned)의 최적 조합
- RQ3: Point-wise F1과 Event-wise F1의 상관관계
- RQ4: FN/FP 비용 비율의 데이터 특성별 최적화

---

## 2. 달성도 평가

### 2.1 Phase 1: ML 이상 탐지

| 항목 | 목표 | 달성도 | 비고 |
|------|------|--------|------|
| 데이터셋 커버리지 | 4개 데이터셋 | **100%** | SKAB(완), AIHub(완), **SMD 28머신(완)**, Synthetic(완) |
| 디텍터 수 | 6개 | **100%** | Rule, kNN, IF, LSTM-AE, Hybrid, SpecCNN |
| 통계적 검증 | Friedman + post-hoc | **100%** | 프레임워크 구현 완료, AIHub 검증 완료 |
| 재현성 | multi-seed 실험 | **80%** | 5 seeds, 대부분 검증 완료 |

### 2.2 Phase 2: LLM 설명

| 항목 | 목표 | 달성도 | 비고 |
|------|------|--------|------|
| 설명 생성 | 이상 포인트 설명 | **60%** | 10 샘플 × 4 조건 = 40 호출 완료 |
| 2×2 Ablation | 도메인지식 × feature importance | **100%** | 완료, 명확한 결과 |
| 평가 프로토콜 | faithfulness, consistency | **80%** | 자동화 완료, 인간 평가 미진행 |
| RAG 통합 | TF-IDF + 도메인 지식 | **50%** | explain_rag.py 구현, 실험 미진행 |

### 2.3 Phase 3: LLM 파라미터 최적화

| 항목 | 목표 | 달성도 | 비고 |
|------|------|--------|------|
| 파라미터 제안 | LLM이 디텍터 파라미터 제안 | **0%** | 코드 있으나 실험 없음 |
| Before/After 비교 | 10% 이상 개선 | **0%** | 미착수 |

---

## 3. 주요 발견과 연구적 의의

### 3.1 핵심 발견 (논문 기여 가능)

**Finding 1: Spectral Flux가 제조 센서에서 일관적으로 우수**
- SKAB(valve): AUC-PR 0.446 (1위)
- AIHub(제조운송): AUC-PR 0.661 (1위, medium effect)
- 기존 z-score 기반 SpecCNN의 스코어 붕괴 문제를 spectral flux로 해결
- **의의**: 주파수 도메인 접근이 주기적 산업 시계열에서 유효함을 3개 데이터셋에서 실증

**Finding 2: 데이터셋 특성에 따라 최적 디텍터가 다름**
- SKAB: 차이 없음 (Friedman p=0.887)
- SMD 28머신: **IF >> SpecCNN >> kNN >> Rule** (Friedman p<0.001, 모든 쌍 유의미)
- AIHub: **SpecCNN >> Rule ≈ kNN >> IF** (Friedman p<0.001)
- **의의**: "만능 디텍터"는 없으며, 데이터 특성 기반 디텍터 선택이 필수. 특히 IF는 SMD에서 1위이나 AIHub에서 4위로 완전히 역전됨.

**Finding 3: Feature Importance가 LLM 설명의 핵심 입력**
- Faithfulness: 0.425 → 0.675 (+59%)
- 도메인 지식 단독은 효과 없음 (0.425 유지)
- 다중 센서 데이터에서 효과 극대화 (AIHub 8채널: 0.25 → 0.75)
- **의의**: XAI에서 model attribution이 domain knowledge보다 중요하다는 실증적 증거

**Finding 4: FN:FP = 5:1이 실용적 최적점**
- 1:1 → 5:1 전환에서 F1 급상승 (0.05 → 0.52)
- 5:1 이후 포화 (10:1, 50:1에서 추가 개선 없음)
- **의의**: 제조 환경의 비용 비대칭성을 정량화

### 3.2 연구 방향 적합성 평가

| 측면 | 평가 | 근거 |
|------|------|------|
| 학술적 기여 | **양호** | 3개 데이터셋에서 frequency-domain AD의 실증, LLM 설명의 feature importance 중요성 |
| 실용적 기여 | **양호** | 비용 최적화 가이드라인(5:1), 디텍터 선택 가이드라인 |
| 독창성 | **보통** | Spectral flux는 기존 기법, 결합 방식이 기여. LLM 설명 평가는 비교적 새로운 시도 |
| 재현성 | **양호** | multi-seed, 3 데이터셋, 코드 공개 가능 |

---

## 4. 연구 목적 대비 Gap 분석

### 4.1 충분히 답변된 RQ

**RQ1 (주파수 vs 시간)**: SpecCNN spectral flux가 SKAB과 AIHub에서 1위 → **답변 가능**
- 단, SMD에서의 검증이 필요 (현재 진행 중)
- 통계적 유의성: AIHub에서 Friedman p < 0.000001 확인

**RQ4 (비용 민감도)**: FN:FP = 5:1 최적점 확인, 포화 현상 문서화 → **답변 가능**
- AIHub 검증은 향후 추가 필요

### 4.2 부분적 답변

**RQ2 (앙상블)**: 차이가 매우 작음 (0.324~0.344) → **"유의미한 차이 없음"이 결론일 수 있음**
- 이것 자체가 연구적 기여: "단순 linear ensemble이면 충분하다"
- SMD/AIHub에서의 검증 필요

**RQ3 (Point vs Event 상관)**: 스크립트 구현 완료, 데이터 대기 중 → **실행만 하면 답변 가능**

### 4.3 미답변

**Phase 3 (LLM 파라미터 최적화)**: 완전 미착수
- 코드는 있으나 (llm_parameter_advisor.py) 실험 없음
- 우선순위를 낮추거나 "향후 과제"로 전환 권장

---

## 5. 권장 사항

### 5.1 즉시 실행 (이번 주)

1. **SMD 전체 28머신 실험 완료** → RQ1 SMD 검증
2. **통합 결과 통계 검증** → 전체 Friedman + Nemenyi
3. **RQ3 상관분석 실행** → Point vs Event F1 상관 확인

### 5.2 단기 (1~2주)

4. **LLM 설명 샘플 확대** → SKAB 15 + AIHub 15 + SMD 10 = 40 샘플
5. **RQ2 AIHub 앙상블 실험** → 현재 SKAB만
6. **논문 테이블/Figure 확정** → 디텍터 비교 테이블, 비용 민감도 그래프, LLM ablation 바차트

### 5.3 중기 (3~4주)

7. **Phase 3 최소 실험** — LLM이 제안한 IF/kNN 파라미터로 SKAB 재실험 → baseline 대비 개선 측정
8. **논문 초고 작성**

### 5.4 연구 방향 조정 권장

- **Phase 3 (LLM 최적화)를 축소** → "향후 과제"로 언급하되, 1~2개 시나리오 시범 결과만 포함
- **Phase 2 (LLM 설명)에 집중** → Finding 3이 가장 독창적이고 기여도 높음
- **"데이터셋 특성 기반 디텍터 선택"을 메인 테마로** → Finding 2가 실용적 가치 높음

---

## 6. 논문 구성 제안

### 제목 (안)
"Frequency-Domain Anomaly Detection with LLM-Guided Explanation in Manufacturing Time Series"

### 구성
1. Introduction: 제조 이상 탐지 + LLM 설명의 필요성
2. Related Work: 시계열 AD, 주파수 도메인, LLM for XAI
3. Method: Spectral Flux detector, Ensemble, Cost optimization, LLM explanation pipeline
4. Experiments:
   - RQ1: Frequency vs Time domain (3 datasets, statistical tests)
   - RQ2: Ensemble comparison
   - RQ3: Point vs Event metrics
   - RQ4: Cost sensitivity
   - Phase 2: LLM explanation (2×2 ablation)
5. Results & Discussion
6. Conclusion + Future Work (Phase 3)

### 핵심 Table/Figure
- **Table 1**: 디텍터 × 데이터셋 AUC-PR (3×5 매트릭스)
- **Table 2**: AIHub Friedman + Nemenyi 결과
- **Table 3**: LLM 설명 2×2 Ablation
- **Figure 1**: 비용 비율 vs F1 곡선 (FN:FP 1~50)
- **Figure 2**: Spectral Flux 작동 원리 (정상 vs 이상 구간 비교)

---

## 7. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| SMD 전체 실험 실패 | RQ1 SMD 검증 불가 | 기존 file0 결과 + "단일 머신" 한계 명시 |
| LLM 설명 샘플 부족 | 통계적 유의성 약함 | effect size 보고, 정성적 분석 보강 |
| Phase 3 미착수 | 논문 범위 축소 | "향후 과제"로 전환, Phase 2에 집중 |
| GPU 환경 불안정 | LSTM-AE 실험 제한 | CPU 디텍터(Rule/kNN/IF/SpecCNN) 위주 분석 |
