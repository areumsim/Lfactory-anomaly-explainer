# 논문 가치 냉철한 검토 보고서

## Context
사용자가 현재 실험 결과의 논문 가치, novelty, 부족한 점에 대한 정직한 평가를 요청. 이 문서는 검토 결과이며 실행 계획이 아님.

---

## 1. 기여별 Novelty 평가

### C1: Feature Attribution → LLM 설명 품질 (주력 기여)

**주장**: "다중 센서 환경에서 feature attribution이 LLM 설명의 센서 식별 능력을 향상시킨다"

**Novelty 판정: ⚠️ 조건부 — 있으나 제한적**

강점:
- LLM에 feature importance를 제공하면 설명이 좋아진다는 것을 **실증적으로 보인 연구가 거의 없음**
- 2×2 ablation 설계 + 이중 평가(keyword + LLM-judge)는 방법론적으로 견실
- AIHub 8채널에서 sensor_id 1.76→3.56 (+102%)는 강한 효과

약점 — 솔직한 평가:
- **"feature importance를 주면 LLM이 더 잘 안다"는 너무 직관적인 결론**. 리뷰어가 "당연한 거 아닌가?"라고 물을 수 있음
- 핵심 효과가 **keyword 기반에서만 통계적 유의** (p<0.001). LLM-judge에서는 **전체 점수 차이 없음** (p=0.922)
  → 리뷰어: "keyword matching은 trivial metric이고, 더 정교한 LLM-judge에서는 차이가 없다"
- sensor_id 차원에서만 개선 (+1.79/5)이지만 이것도 **Wilcoxon test 미실행** (차원별 통계 부재)
- SKAB/SMD에서 효과 **완전히 0** → 108개 중 34개(AIHub)에서만 작동 = 전체의 31%
- N=108이지만 실질적 효과 발생 N=34 (AIHub만)

**근본적 질문**: "LLM에 정보를 더 많이 주면 답이 좋아진다"는 것이 논문 기여인가?

→ **논문 기여가 되려면**: 단순히 "더 좋아진다"가 아니라 **어떤 조건에서, 왜, 얼마나** 좋아지는지를 밝혀야 함. 현재 "다중 센서일 때만"이라는 조건은 발견했으나, **메커니즘 분석이 부족**함.

---

### C2: Rank Reversal (데이터 의존적 성능)

**주장**: "만능 디텍터는 없다 — 데이터 특성에 따라 최적 디텍터가 다르다"

**Novelty 판정: ❌ 낮음**

- **No Free Lunch Theorem (Wolpert & Macready, 1997)**: 이론적으로 이미 알려진 사실
- 시계열 이상탐지에서도 여러 벤치마크 논문(Schmidl et al. 2022 "Anomaly Detection in Time Series: A Comprehensive Evaluation", Wu & Keogh 2021)이 동일한 결론을 도출
- 본 연구가 추가하는 것: SKAB/SMD/AIHub71802라는 **특정 3개 데이터셋**에서의 실증
  → 그러나 이 데이터셋 조합 자체가 novel하지 않음 (SKAB/SMD는 공개, AIHub는 비공개)

**절대 성능 문제 (심각)**:
- SKAB: 최고 AUC-PR = 0.446 (random baseline 0.349 대비 +0.098). **매우 낮음**
- AIHub: IF(0.583)가 **random(0.620)보다 낮음** → 아예 작동 안 함
- SMD: kNN best AUC-PR = 0.390 → SOTA (OmniAnomaly 등) 대비 **현저히 낮음**

→ 리뷰어: "성능이 이렇게 낮은 디텍터들의 rank를 비교해서 무슨 의미가 있는가?"

---

### C3: Threshold > Detector Choice

**주장**: "threshold 전략이 디텍터 선택보다 3.8-29.3배 더 중요하다"

**Novelty 판정: ⚠️ 낮으나 실무적 의미 있음**

- Optimal-F1은 **oracle threshold** (사후에 최적 임계값을 알 때) → 실무 불가
- "threshold가 중요하다"는 이미 알려진 사실
- 단, **수치적 정량화** (3.8x~29.3x)는 새로운 데이터 포인트

---

## 2. 방법론적 약점

### 2.1 디텍터가 너무 단순
- Rule (rolling z-score), kNN, IF, SpecCNN → 모두 **2010년대 이전 방법**
- 최신 방법 부재: Transformer-based (Anomaly Transformer, 2022), GNN-based, reconstruction-based deep models
- 리뷰어: "왜 최신 deep learning 방법과 비교하지 않았는가?"

### 2.2 단변량 분석
- SMD는 38차원, AIHub는 8차원이지만 **분석은 단일 센서 기반**
- 다변량 이상탐지의 핵심인 센서 간 상관관계를 활용하지 않음

### 2.3 AIHub 데이터셋 문제
- label_rate = 62% → **이상탐지가 아니라 분류 문제**
- 비공개 데이터셋 → **재현 불가**
- 시계열 길이 300-361점 → **매우 짧음**

### 2.4 LLM 평가의 본질적 한계
- Keyword matching: 4단계 (0.25/0.50/0.75/1.00) → 너무 조잡
- LLM-as-judge: **동일 모델로 생성 + 평가** → self-evaluation bias
  → 리뷰어: "Claude가 생성한 설명을 Claude가 평가하면 편향 아닌가?"
- Human evaluation 부재 → XAI 논문에서 human study 없는 것은 큰 약점

---

## 3. 현재 상태에서 논문이 되는가?

### 타겟 학회별 판단

| 타겟 | 판정 | 이유 |
|------|------|------|
| **Top-tier (NeurIPS, ICML, KDD)** | ❌ | Novelty 부족, 성능 낮음, 최신 baseline 없음 |
| **Mid-tier (CIKM, PAKDD, ECML)** | ⚠️ | C1이 강화되면 가능하나 현재는 부족 |
| **국내 학회 (KSC, KIISE)** | ✅ | 실증적 분석 + LLM 활용으로 충분 |
| **산업/응용 저널 (IEEE Access, Sensors)** | ⚠️ | C1 강화 + human eval 추가 시 가능 |

### 핵심 문제 3가지

1. **"So what?" 문제**: Rank reversal은 알려진 사실, feature attribution 효과는 직관적 → **새로운 통찰(insight)이 무엇인가?**

2. **성능 절대 수준**: 최고 AUC-PR이 0.45 (SKAB) / 0.39 (SMD) / 0.74 (AIHub, but random=0.62) → **SOTA와 비교 불가능한 수준**

3. **평가 신뢰성**: Human evaluation 없는 XAI 논문은 설득력 제한적

---

## 4. 논문을 살리려면 (우선순위 순)

### 필수 보완 (이것 없이는 submit 어려움)

1. **C1의 메시지를 "당연한 것" → "비자명한 통찰"로 전환**
   - 현재: "feature importance 주면 좋아진다" (trivial)
   - 필요: "feature importance의 **어떤 요소**가 핵심인가?"
     - 예: top-1 센서만 알려줘도 되는가? 전체 ranking이 필요한가?
     - 예: z-score 값을 주는 것 vs 센서 이름만 주는 것 → 어디서 효과가 발생?
     - 이런 **세분화된 ablation**이 novelty를 만듦

2. **Human evaluation 추가** (최소 2명의 도메인 전문가)
   - LLM 설명의 유용성을 5점 척도로 평가
   - 20-30개 설명에 대해 baseline vs feature_only 비교
   - XAI 논문에서 필수적

3. **최신 디텍터 1-2개 추가**
   - Anomaly Transformer (Xu et al., 2022) 또는 USAD (Audibert et al., 2020)
   - "최신 방법도 포함했으나 rank reversal은 여전히 존재"를 보여야 함

### 권장 보완

4. **AIHub 대신/추가로 공개 다중센서 데이터셋**
   - WADI, SWaT (water treatment, 50+ sensors) → 재현 가능 + 다중 센서
   - HAI (Hardware-in-the-loop, 공개) → 산업 데이터

5. **Feature attribution 세분화 실험**
   - Top-1만 제공 vs Top-3 vs 전체 ranking vs z-score 수치 제공
   - 이것이 C1의 novelty를 결정적으로 높임

---

## 5. 최종 냉철한 판단

**현재 상태**: 실험 인프라와 데이터 무결성은 우수하나, **논문으로서의 기여(contribution)가 명확하지 않음**.

"2,105 runs를 돌렸다"는 양적으로 인상적이지만, **"그래서 무엇을 새로 알게 되었는가?"**에 대한 답이:
- "만능 디텍터는 없다" → 이미 알려짐
- "threshold가 중요하다" → 이미 알려짐
- "feature importance 주면 LLM 설명이 좋아진다" → 직관적

**논문이 되려면 C1을 "왜, 어떤 조건에서, 어떤 granularity에서"로 심화해야 함.**
이것이 현재 가장 큰 갭이며, feature attribution 세분화 실험 + human eval이 해결책임.
