# 논문 프레임 v2: 연구 목적 재정렬

**작성일**: 2026-04-06 (연구 목적 재정렬 후)
**이전 버전**: PAPER_NARRATIVE.md (C1=LLM 설명 중심 — 초기 목적과 괴리)
**변경 이유**: 초기 Detect→Explain→Act 3단계 목적 대비 현재 결과 재평가

---

## 제목 (안)

**"What Matters Most in Manufacturing Time Series Anomaly Detection:
An Empirical Study on Detector Selection, Calibration, and LLM Explanation"**

한국어:
**"제조 시계열 이상탐지에서 진짜 중요한 것: 탐지기 선택, 캘리브레이션, LLM 설명의 실증적 연구"**

---

## 연구 동기

제조 시계열 이상탐지 파이프라인을 구축할 때 실무자가 직면하는 질문들:
1. 어떤 탐지기를 선택해야 하는가? (IF, kNN, SpecCNN, Rule...)
2. 임계값은 어떻게 설정하는가?
3. 캘리브레이션은 어떻게 해야 하는가?
4. 탐지된 이상을 LLM으로 설명할 때 무엇을 제공해야 하는가?

이 연구는 **4개 데이터셋, 4개 탐지기, 5,600+ 실험**을 통해 이 질문들에 실증적으로 답한다.

---

## 3대 기여

### C1 (주력): 탐지기 선택은 데이터 의존적이며, 임계값이 탐지기 선택보다 3-29× 더 중요

**증거**:

1. **Rank Reversal (3개 데이터셋)**

| Rank | SKAB (1센서, 밸브) | SMD (38센서, 서버) | AIHub (8센서, AGV) |
|:---:|:---:|:---:|:---:|
| 1 | SpecCNN (0.446) | kNN (0.390) | SpecCNN (0.739) |
| 2 | IF (0.433) | IF (0.348) | Rule (0.659) |
| 3 | kNN (0.420) | SpecCNN (0.204) | kNN (0.649) |
| 4 | **AT (0.396)** | **AT (0.147)** | **AT (0.641)** |
| 5 | Rule (0.367) | Rule (0.057) | IF (0.583) |

- **Anomaly Transformer(ICLR 2022)가 모든 데이터셋에서 4위/5** — 최신 DL이 단순 방법보다 일관되게 열등
- Friedman test: SKAB p=0.036* (5탐지기), **SMD p<0.000001***, **AIHub p<0.0001***

2. **시나리오별 층화 (SKAB 내 rank reversal)**

| Rank | valve1 (밸브 폐쇄) | other (기타 이상) |
|:---:|:---:|:---:|
| 1 | **SpecCNN** (0.385) | **IF** (0.570) |
| 2 | Rule (0.339) | kNN (0.538) |
| 3 | kNN (0.334) | SpecCNN (0.536) |
| 4 | IF (0.323) | Rule (0.401) |

- Friedman: valve1 p=0.003*, other p=0.003* → **동일 데이터셋 내에서도 rank reversal**
- **해석**: SpecCNN(주파수)은 밸브 이상(주기적)에서 강하고, IF는 다양한 이상에서 강함

3. **Threshold > Detector (3.8-29.3×)**

| Detector | Fixed-F1 | Optimal-F1 | Ratio |
|----------|:--------:|:----------:|:-----:|
| Rule | 0.018 | 0.519 | **29.3×** |
| SpecCNN | 0.027 | 0.570 | **21.2×** |
| kNN | 0.025 | 0.561 | **22.2×** |
| IF | 0.160 | 0.615 | **3.8×** |

→ 탐지기를 바꾸는 것보다 **임계값 최적화가 3.8-29.3배 더 효과적**

**통계적 강도**: 5,600+ runs, 3 datasets, Friedman+Nemenyi+Cliff's delta+Bootstrap CI
**Power 분석**: SMD/AIHub adequate (power≥0.96), SKAB underpowered (power=0.09-0.67)

---

### C2 (보조): LLM 설명에서 Feature Attribution 효과 — 고차원(38+)에서만 유의

**증거** (N=20 per condition, Wilcoxon signed-rank test):

| Dataset (센서 수) | baseline strict | feature_only strict | p-value | 유의성 |
|:-:|:-:|:-:|:-:|:-:|
| SMD (38센서) TP | 0.690 | **0.830** | **0.034*** | 유의 |
| SMD (38센서) FN | 0.605 | **0.765** | **0.023*** | 유의 |
| AIHub (8센서) TP | 0.635 | 0.515 | 0.206 | **비유의 (하락)** |
| AIHub (8센서) FN | 0.625 | 0.695 | 0.258 | 비유의 |
| SKAB (1센서) | 0.520 | 0.520 | - | 효과 없음 |

**핵심 발견**:
1. **38센서(SMD)에서만 통계적으로 유의** — 8센서(AIHub)에서는 strict metric 기준 비유의
2. Direction accuracy: SMD baseline 60-69% → feature_only **100%** (극적 향상)
3. AIHub TP에서 feature attribution이 오히려 strict 점수 하락 (0.635→0.515)
4. 기존 발견 유지: names_only 91% vs zscore_only 0% (센서 이름이 핵심)

**Cross-Model Judge (GPT 생성 → Claude 평가)**:
| 조건 | GPT self-judge | Claude cross-judge | 차이 |
|------|:-:|:-:|:-:|
| baseline | 0.915 | 0.855 | -0.060 |
| feature_only | 0.885 | 0.730 | **-0.155** |
| full | 0.835 | 0.650 | **-0.185** |

→ **Self-evaluation bias 확인**: GPT가 자기 설명을 0.15-0.19 높게 평가
→ 기존 LLM-judge p=0.922 (차이 없음)는 self-eval bias 때문일 가능성 높음

**제한 사항**:
- 효과 범위 축소: 기존 "8+센서" → **"38+센서에서만 유의"**
- AIHub: keyword metric에서만 향상 (+150%), strict에서 비유의/하락
- Human evaluation 미실시 — future work
- Self-evaluation bias가 기존 결과를 오염시켰을 가능성

---

### C3 (방법론): Calibration Temporal Split 필수 — ECE 16.5× 왜곡 실증

**발견**: Platt scaling을 전체 test labels로 fitting하면 ECE가 인위적으로 낮아짐

| Detector | ECE (leakage) | ECE (temporal split) | 왜곡 배율 |
|----------|:---:|:---:|:---:|
| Rule | 0.011 | 0.187 | **16.5×** |
| kNN | 0.026 | 0.186 | **7.2×** |
| IF | 0.090 | 0.203 | **2.3×** |
| SpecCNN | 0.106 | 0.194 | **1.8×** |

**실무 가이드라인**: 캘리브레이션 파라미터는 반드시 temporal split(60/40)으로 fitting해야 함.

---

## Negative Results (정식 보고)

### RQ2: 앙상블 방법 간 차이 없음
- Linear, Product, Max, Learned 4개 방법: 기대 비용 0.324-0.344 (차이 무의미)
- **결론**: 가장 단순한 Linear 앙상블로 충분

### RQ1 (부분): 주파수 특징의 보편적 우위 없음
- SpecCNN은 valve1(밸브)에서만 1위, other/SMD에서는 중하위
- **결론**: 주파수 도메인의 우위는 이상 유형에 의존

---

## 초기 목적 대비 현재 상태 (솔직한 기술)

| 원래 계획 | 현재 상태 | 비고 |
|----------|----------|------|
| Phase 1: Detect (RQ1-4) | ✅ 완료 | RQ1 부분, RQ2 negative, RQ3 confound, RQ4 SKAB만 |
| Phase 2: Explain (LLM) | ⚠️ 60% | 2×2 ablation + granularity + FN 실험 완료. Human eval 미실시 |
| Phase 3: Act (LLM 파라미터) | ❌ 미실행 | future work |
| SpecCNN (주파수 탐지) | ⚠️ 시나리오별만 | 전체적으로는 유의하지 않음 (Friedman p=0.194) |
| Calibration ECE<0.05 | ❌ 달성 못함 | Leakage 수정 후 ECE 0.18-0.20 |

---

## Limitations

1. **절대 성능 낮음**: SKAB 최고 AUC-PR 0.446 (random 대비 +10%), 최신 DL baseline 제한적
2. **Power 부족**: SKAB pairwise 비교 6개 중 1개만 adequate (SMD/AIHub는 adequate)
3. **Human evaluation 없음**: LLM 설명 품질의 최종 검증 미실시
4. **Self-evaluation bias**: GPT-4o-mini가 자기 설명을 0.15-0.19 높게 평가 — Claude cross-judge로 확인됨
5. **C2 효과 범위 제한**: Feature attribution 효과는 38센서(SMD)에서만 통계적 유의. 8센서(AIHub)에서 비유의
6. **Phase 3 미실행**: LLM 파라미터 최적화 실험 0건
7. **AIHub 비공개**: 재현 불가 (label_rate 54% = 분류 문제에 가까움)
8. **단변량 분석**: 38차원 SMD를 단일 센서 기준으로 탐지 (다변량 상관 미활용)
9. **Cost sensitivity ratio 단일**: FN:FP = 5:1만 테스트. 다양한 ratio sweep 미실행

## Future Work

1. Phase 3 (LLM 파라미터 최적화) 실행
2. Human evaluation (N≥30, 도메인 전문가 2명)
3. 최신 DL baseline (Anomaly Transformer, TranAD) 비교
4. 다변량 탐지 방법 적용
5. Cost sensitivity를 SMD/AIHub로 확대

---

## 타겟 학회

| 타겟 | 판정 | 근거 |
|------|:---:|------|
| KSC/KIISE (국내) | ✅ | 5,600+ runs 실증, 3개 기여 |
| IEEE Access/Sensors | ⚠️ | Human eval 추가 시 가능 |
| CIKM/PAKDD | ⚠️ | 프레임 A(메커니즘 중심)로 전환 + human eval 필수 |
| NeurIPS/ICML/KDD | ❌ | Novelty 근본적 부족 |
