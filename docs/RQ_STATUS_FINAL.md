# RQ 최종 상태 보고서

**작성일**: 2026-04-06
**목적**: 원래 4개 RQ + Phase 2-3의 현실적 상태를 솔직하게 기록

---

## RQ별 가설 검증 결과

### RQ1: 주파수 vs 시간 도메인

| 가설 | 결과 | 판정 |
|------|------|:---:|
| H1.1 SpecCNN > Rule (주기 데이터) | SKAB 전체: p=0.194 (NS). **valve1만: p=0.003*** | **부분 지지** |
| H1.2 Rule > SpecCNN (스파이크) | 이상 유형별 층화 분석 미실행 | **미검증** |
| H1.3 Hybrid ≥10% 향상 | Hybrid 0.375 < 단독 최고 0.446. 오히려 하락 | **기각** |
| H1.4 데이터 특성으로 예측 | 미실행 | **미검증** |

**결론**: 주파수 도메인의 보편적 우위는 없음. 밸브 폐쇄(valve1) 시나리오에서만 SpecCNN 1위.
시나리오별 rank reversal(valve1 vs other)이 **동일 데이터셋 내에서도** 발생 — 이상 유형 의존성 확인.

---

### RQ2: 앙상블 방법

| 가설 | 결과 | 판정 |
|------|------|:---:|
| H2.1 Linear ECE < 0.05 | Leakage 수정 후 ECE = 0.19 | **기각** |
| H2.2 Max > Product (고불균형) | 0.324-0.344, 차이 없음 | **기각** |
| H2.3 Learned ≥5% 향상 | N=5, 차이 없음 | **기각** |

**결론**: **Negative result**. 4개 앙상블 방법 간 유의미한 차이 없음. Linear으로 충분.
이 자체가 실무적 가이드라인: 복잡한 앙상블 불필요.

---

### RQ3: Point vs Event 메트릭 상관

| 가설 | 결과 | 판정 |
|------|------|:---:|
| H3.1 ρ = 0.5~0.8 (중간) | SKAB ρ=0.82, SMD ρ=0.09 — 양극단 | **기각** (예상과 다름) |
| H3.2 긴 이상 → 높은 상관 | **반대**: 짧은 시계열(SKAB)에서 높은 ρ | **기각** |

**결론**: 상관은 이상 길이가 아니라 **시계열 길이**에 의존. 짧은 시계열에서 point=event, 긴 시계열에서 분리.
Confound 주의: 시계열 길이와 이상 분포가 혼재.

---

### RQ4: 비용 민감도

| 가설 | 결과 | 판정 |
|------|------|:---:|
| H4.1 ratio ≈ 0.5 × imbalance | 5:1 최적, 비선형 포화 | **부분 지지** |
| H4.2 20:1에서 포화 | 10:1에서 이미 포화 | **근접** |
| H4.3 낮은 SNR → 보수적 | SKAB에서만 테스트 | **미검증** |
| H4.4 ECE<0.05 → 강한 상관 | ECE<0.05 달성 못함 | **전제 붕괴** |

**결론**: FN:FP = 5:1이 실무적 최적. SKAB에서 62.8% 비용 절감. 단, 1개 데이터셋 한정.

---

### Phase 2: LLM 설명

| 가설 | 결과 | 판정 |
|------|------|:---:|
| Feature attribution → 설명 향상 | SMD: strict +20% (p=0.023-0.034) | **지지** (38센서 한정) |
| AIHub(8센서)에서도 효과 | strict 비유의 (p=0.21), TP에서 하락 | **기각** (8센서 불충분) |
| 센서 이름이 핵심 | names_only 91% vs zscore_only 0% | **강하게 지지** |
| Top-3이 최적 | sensor_id 4.65 > full 3.32 | **지지** |
| FN에서도 효과적 | SMD FN direction 100%, baseline 60% | **지지** |
| Domain confound | full_nosensor ≥ full | **confound 없음 확인** |
| Self-eval bias 없음 | GPT judge 0.15-0.19 높게 평가 | **bias 확인** (Claude cross-judge) |

**결론 (수정됨)**: Feature attribution은 **38센서 이상(SMD)**에서만 통계적으로 유의.
8센서(AIHub)에서는 keyword metric에서만 향상, strict metric에서 비유의 또는 하락.
메커니즘: 센서 이름(의미론적 레이블)이 핵심.
Self-evaluation bias가 기존 LLM-judge 결과를 오염시켰을 가능성 있음.

---

### Phase 3: LLM 파라미터 최적화

| 가설 | 결과 | 판정 |
|------|------|:---:|
| LLM → 10% 성능 향상 | 실험 0건 | **미검증** (포기) |

**결론**: 원래 가장 novel한 기여였으나 실행되지 않음. Future work.

---

## 종합: 논문에 사용 가능한 발견

### 확실한 기여 (통계적 뒷받침 있음)
1. **Rank reversal**: 3개 데이터셋 + 시나리오 내 (Friedman p<0.004)
2. **Threshold 3-29× > Detector**: SKAB 정량화
3. **Calibration leakage**: ECE 1.8-16.5× 왜곡 실증
4. **센서 이름 > z-score**: names_only 91% vs zscore_only 0%

### 조건부 기여 (추가 검증 필요)
5. Feature attribution → FN에서도 효과적 (N=10, 확대 필요)
6. Cost ratio 5:1 최적 (SKAB만, 타 데이터셋 확대 필요)
7. 앙상블 negative result (N=5로 power 부족)

### 주장 불가 (증거 부족)
- "주파수 도메인이 보편적으로 우수" — Friedman p=0.194
- "ECE < 0.05 달성" — leakage 수정 후 0.19
- "LLM이 파라미터를 제안하여 성능 향상" — 미실행
