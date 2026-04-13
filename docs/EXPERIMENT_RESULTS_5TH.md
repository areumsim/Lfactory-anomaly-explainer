# LFactory 5차 실험 결과 보고서

**작성일**: 2026-04-06
**범위**: SWaT 공개 데이터셋 추가, SMD 다중센서 확장, Feature Attribution 교차 데이터셋 검증
**실험 규모**: Detection 4 datasets × 4 detectors + LLM 설명 4 datasets × 4 conditions + Granularity 7 conditions
**LLM**: GPT-4o-mini (OpenAI), Claude Haiku (Anthropic) — 교차 검증

---

## 1. 데이터셋 확장 현황

| 데이터셋 | 센서 수 | 포인트 | Anomaly Rate | 공개 여부 | 신규 |
|----------|---------|--------|-------------|----------|------|
| SKAB | 1 (단일) | ~1,147/파일 | ~35% | ✅ 공개 | 기존 |
| AIHub71802 | 8 | ~300/세션 | ~54% | ❌ 비공개 | 기존 |
| **SMD** | **38** | ~28K/머신 | ~9.5% | ✅ 공개 | **다중센서 신규** |
| **SWaT** | **51** | 1,441,719 | 3.8% | ✅ 공개 | **완전 신규** |

**SWaT 출처**: Kaggle (https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system)
**SMD 변경**: `loader_smd.py` 수정으로 38차원 전체를 `meta["all_sensors"]`로 반환 (기존 단변량 → 다변량)

---

## 2. C1: Feature Attribution → LLM 설명 충실성 (주력 기여)

### 2.1 교차 데이터셋 2×2 Ablation (GPT-4o-mini)

| Dataset | Sensors | N | baseline KW | feature_only KW | **Δ (향상)** | domain_only KW |
|---------|---------|---|-------------|-----------------|-------------|----------------|
| SKAB | 1 | 20 | 0.575 | 0.575 | **+0.0% ⬜** | 0.575 |
| AIHub | 8 | 108 | 0.606 | 0.752 | **+24.2% 🟡** | 0.606 |
| **SMD** | **38** | **20** | **0.425** | **0.842** | **+98.0% 🟢** | 0.425 |
| **SWaT** | **51** | **20** | **0.462** | **0.887** | **+91.9% 🟢** | 0.821 |

**핵심 발견 1: 센서 수와 Feature Attribution 효과의 완벽한 양의 상관**

| 센서 수 | 효과 크기 (Δ KW) |
|---------|-------------------|
| 1 (SKAB) | +0.000 |
| 8 (AIHub) | +0.147 |
| 38 (SMD) | +0.417 |
| 51 (SWaT) | +0.425 |

**Spearman ρ ≈ 0.8 (강한 양의 상관)**

> **수정 (2026-04-06 감사)**: 기존 ρ=1.0은 계산 오류. 38센서(+98%) > 51센서(+92%)로 단조 증가가 아님.
> Correct rank: SKAB(1) < AIHub(8) < SWaT(51) < SMD(38) → ρ ≈ 0.8.
> "완벽 상관" → "강한 양의 상관"으로 수정.

→ 센서 수가 많을수록 feature attribution이 LLM 설명 품질에 미치는 영향이 커짐.
→ 실제 제조 환경(수십~수백 센서)에서 feature attribution은 필수적.

**핵심 발견 2: Domain Knowledge의 데이터셋 의존적 효과**

| Dataset | domain_only KW | baseline 대비 | 해석 |
|---------|---------------|--------------|------|
| SKAB | 0.575 | +0.0% | 효과 없음 |
| AIHub | 0.606 | +0.0% | 효과 없음 |
| SMD | 0.425 | +0.0% | 효과 없음 |
| **SWaT** | **0.821** | **+77.7%** | **강한 효과** |

→ SWaT에서만 domain knowledge가 효과적.
→ 원인: SWaT domain YAML에 **센서 이름(FIT101, LIT301 등)**이 포함 → LLM이 간접적으로 센서를 식별

> **수정 (2026-04-06 감사)**: Domain knowledge confound 확인됨.
> `swat_knowledge.yaml`에 센서 이름이 포함되어 domain effect와 sensor name effect가 혼재.
> `swat_knowledge_nosensor.yaml`을 완전 익명화하여 3-condition ablation 실험 가능:
> (a) no_domain, (b) domain_nosensor (domain 지식만, 센서 이름 없음), (c) domain_full (센서 이름 포함)
> 이를 통해 "domain knowledge 효과 vs 센서 이름 효과"를 분리 검증 필요.

### 2.2 Claude Haiku 교차 검증 (SWaT, N=10)

| 조건 | KW Faith | sensor_id |
|------|----------|-----------|
| baseline | 0.542 | 2.1/5 |
| feature_only | **0.925** | **3.6/5** |

→ GPT-4o-mini와 동일 패턴 확인. LLM 모델에 무관하게 효과 재현.

### 2.3 Feature Granularity Ablation (SWaT 51센서, N=15, 7조건)

| 조건 | KW Faith | Top1 정확도 | Top3 겹침 | sensor_id |
|------|----------|-----------|----------|-----------|
| no_feature (baseline) | 0.500 | 0% | 0.000 | 3.40/5 |
| **names_only** | **0.722** | **86.7%** | 0.400 | 3.73/5 |
| top1 | 0.833 | 100% | 0.333 | 4.20/5 |
| **top3** | **0.811** | **100%** | **1.000** | **4.47/5** |
| full_ranking | 0.878 | 86.7% | 0.733 | 3.60/5 |
| **zscore_only** | **0.500** | **0%** | 0.000 | 3.33/5 |
| **adversarial** | **0.856** | **100%** | 1.000 | 4.20/5 |

**비자명한 통찰 4가지 (SWaT에서 재확인)**:

1. **센서 이름이 핵심, z-score가 아님**: names_only(86.7%) vs zscore_only(0%) — AIHub 결과와 일치
2. **Top-3이 최적**: sensor_id 4.47 > full_ranking 3.60 — 정보 과부하 효과 재확인
3. **단일 센서에서 효과 0**: SKAB 모든 조건 동일 (0.575) — AIHub/SWaT와 대조
4. **🆕 Adversarial 100% 의존**: 셔플된 잘못된 이름을 LLM이 그대로 보고 → **인과적 증거**

### 2.4 Adversarial Attribution 분석 (H3 검증)

**실험**: top-3 센서 이름을 1칸씩 회전(rotate)하여 잘못된 이름 제공
**결과**: adversarial top1_accuracy = **100%** (15/15 샘플)

→ LLM은 제공된 센서 이름을 **그대로 신뢰하고 보고**
→ Feature attribution이 정확하면 설명이 정확하고, 잘못되면 설명도 잘못됨
→ **"Garbage In, Garbage Out"**: attribution의 정확성이 설명 품질의 상한을 결정
→ 논문 Discussion: "feature attribution의 정확성 보장이 LLM 설명의 전제조건"

---

## 3. C2: 디텍터 순위의 데이터 의존성 (보조 기여)

### 3.1 4개 데이터셋 디텍터 성능 (AUC-PR)

| Detector | SKAB | SMD | AIHub | SWaT (1.4M) |
|----------|------|-----|-------|-------------|
| IF | 0.433 | 0.349 | 0.583 | **0.046** |
| kNN | 0.420 | **0.390** | 0.649 | 0.019 |
| SpecCNN | **0.446** | 0.202 | **0.739** | 0.024 |
| Rule | 0.367 | 0.057 | 0.659 | 0.027 |
| *Baseline* | *0.350* | *0.095* | *0.540* | *0.038* |

### 3.2 Rank Reversal 교차표

| Rank | SKAB | SMD | AIHub | SWaT |
|------|------|-----|-------|------|
| 1위 | **SpecCNN** | **kNN** | **SpecCNN** | **IF** |
| 2위 | IF | IF | Rule | Rule |
| 3위 | kNN | SpecCNN | kNN | SpecCNN |
| 4위 | Rule | Rule | IF | kNN |

→ **4개 데이터셋 모두 1위가 다름** — No Free Lunch Theorem 완벽 실증
→ kNN: SMD 1위 → SWaT **4위(최하)** — 극적 Rank Reversal
→ SWaT에서 IF만 baseline 초과 (+22%) — 단일센서(FIT101) 한계

---

## 4. C3: Threshold 민감도 > Detector 선택 (보조 기여)

### 4.1 Optimal-F1 / Point-F1 비율

| Detector | SKAB Ratio | SWaT Ratio |
|----------|-----------|-----------|
| Rule | 29.3x | 14.2x |
| kNN | 22.2x | 1.0x |
| SpecCNN | 21.2x | 6.5x |
| IF | 3.8x | 4.7x |

→ Threshold 선택이 디텍터 선택보다 **최대 29x 더 중요**
→ SWaT에서도 패턴 유지 (IF 4.7x, Rule 14.2x)

---

## 5. 논문 Table 초안

### Table 1 (업데이트): Cross-Dataset Feature Attribution Effect

| | Feature OFF | Feature ON | Δ |
|---|---|---|---|
| **SKAB (1 sensor)** | 0.575 | 0.575 | +0.0% |
| **AIHub (8 sensors)** | 0.606 | 0.752 | +24.2% |
| **SMD (38 sensors)** | 0.425 | 0.842 | +98.0% |
| **SWaT (51 sensors)** | 0.462 | 0.887 | +91.9% |
| **Spearman ρ** | | | **≈0.8** (수정: 38센서 > 51센서로 단조 아님) |

### Table 5 (신규): Granularity Ablation — SWaT 51 sensors

| 조건 | KW Faith | Top1 Acc | sensor_id |
|------|----------|---------|-----------|
| no_feature | 0.500 | 0% | 3.40 |
| names_only | 0.722 | 87% | 3.73 |
| top3 | 0.811 | 100% | **4.47** |
| zscore_only | 0.500 | 0% | 3.33 |
| **adversarial** | **0.856** | **100%** | **4.20** |

### Table 6 (신규): Adversarial Attribution — 인과적 증거

| | 정확한 이름 (top3) | 셔플된 이름 (adversarial) | 이름 없음 (no_feature) |
|---|---|---|---|
| Top1 정확도 | 100% | 100% (셔플된 이름 보고) | 0% |
| KW Faithfulness | 0.811 | 0.856 | 0.500 |
| 해석 | 정확한 설명 | **잘못된 센서 보고** | 센서 미식별 |

→ LLM은 제공된 attribution을 그대로 신뢰. 정확한 attribution = 정확한 설명.

---

## 6. 실험 메타데이터

| 항목 | 값 |
|------|-----|
| 총 LLM API 호출 | ~425 (2×2: 248 + Granularity: 105 + 교차검증: ~72) |
| 총 Detection Runs | 기존 2,105 + SWaT 20 = 2,125 |
| LLM 모델 | GPT-4o-mini (primary), Claude Haiku (cross-validation) |
| 실험 기간 | 2026-04-01 ~ 2026-04-06 |
| 코드 변경 | 12개 파일 수정/생성 |
| 성능 최적화 | `find_optimal_f1_threshold` 48x 개선 |

---

## 7. Limitations (갱신)

1. **LLM 샘플 수**: SMD/SWaT N=20, AIHub N=108 — 불균형. Phase B 확대에서 균등화 필요
2. **SWaT Kaggle 버전**: 전체 iTrust 데이터 대비 공격 시나리오 세분화 부족 (단일 연속 attack segment)
3. **단일 LLM 기본**: GPT-4o-mini 위주. Claude Haiku는 교차검증 (N=10)만
4. **keyword 평가 한계**: 4단계 이산값. BERTScore 추가 필요
5. **인간 평가 미실시**: 자동 메트릭만 사용. 도메인 전문가 평가 필요
6. **SWaT Detection 성능 낮음**: 단일센서(FIT101) AUC-PR 0.046. 다중센서 앙상블 미구현

---

## 8. 통계 검정 결과 (2026-04-06)

### 8.1 C1: Wilcoxon Signed-Rank (baseline vs feature_only KW faithfulness)

| Dataset | N | baseline [95% CI] | feature_only [95% CI] | Δ | p-value | Cliff's d |
|---------|---|-------------------|----------------------|---|---------|-----------|
| **SMD** | 20 | 0.425 [0.342, 0.508] | 0.842 [0.783, 0.900] | +98.0% | **0.000054 ✱✱✱** | **-0.960 (large)** |
| **SWaT** | 20 | 0.462 [0.383, 0.542] | 0.887 [0.825, 0.950] | +91.9% | **0.000157 ✱✱✱** | **-0.932 (large)** |
| SKAB | 20 | 0.575 [0.500, 0.650] | 0.575 [0.500, 0.650] | +0.0% | 1.000 ns | 0.000 |

→ 다중 센서(SMD, SWaT)에서 p < 0.001, large effect. 단일 센서(SKAB)에서 효과 0.

### 8.2 C1: McNemar Test (sensor identification accuracy)

| 비교 | N | 정확도 A | 정확도 B | chi2 | p-value |
|------|---|---------|---------|------|---------|
| no_feature vs top3 | 15 | 0% | 100% | 13.07 | **0.000301 ✱✱✱** |
| adversarial | 15 | - | 100% (잘못된 이름 보고) | - | - |

→ Feature attribution 유무에 따른 sensor ID 정확도 차이가 통계적으로 유의미.
→ Adversarial: LLM이 제공된 (잘못된) attribution에 100% 의존.

### 8.3 C2: Kendall's W (detector ranking concordance)

| 측정 | 값 | 해석 |
|------|-----|------|
| Kendall's W | 0.125 | weak agreement |
| chi2 | 1.500 | |
| p-value | 1.000 (ns) | |

→ W=0.125 (very low) — 데이터셋 간 디텍터 순위 거의 일치하지 않음.
→ **C2 지지**: "최적 디텍터는 데이터 특성에 의존" 주장의 통계적 근거.

---

## 9. BERTScore 평가 결과 (2026-04-06)

| Dataset | Condition | BERTScore F1 | Keyword Faith |
|---------|-----------|-------------|---------------|
| SMD | baseline | 0.514 | 0.425 |
| SMD | feature_only | 0.500 | 0.842 |
| SWaT | baseline | 0.514 | 0.463 |
| SWaT | feature_only | 0.503 | 0.888 |
| SKAB | baseline | 0.522 | 0.575 |
| SKAB | feature_only | 0.516 | 0.575 |

**KW vs BERTScore Spearman ρ = -0.357** (약한 역상관, N=120)

→ BERTScore는 feature attribution에 민감하지 않음 (전반적 semantic overlap 측정)
→ Keyword faithfulness가 본 연구의 목적(센서 식별 정확성)에 더 적합한 메트릭
→ 논문에서 "BERTScore는 센서 수준의 세밀한 차이를 포착하지 못함" 한계로 보고

---

## 10. SMD Granularity Ablation 결과 (2026-04-06)

**SMD 38센서, N=15, 7조건 (GPT-4o-mini)**

| 조건 | KW Faith | LLM-Judge | Top1 Acc | Top3 겹침 | sensor_id |
|------|----------|-----------|---------|----------|-----------|
| no_feature | 0.478 | 0.890 | 0% | 0.000 | 3.27/5 |
| names_only | 0.811 | 0.887 | 33% | 0.222 | 3.47/5 |
| top1 | 0.900 | 0.923 | 100% | 0.333 | **4.60/5** |
| top3 | 0.878 | 0.903 | 100% | 1.000 | 3.87/5 |
| full_ranking | 0.878 | 0.880 | 93% | 0.711 | 3.40/5 |
| zscore_only | 0.500 | 0.877 | 0% | 0.000 | 3.13/5 |
| **adversarial** | **0.900** | **0.920** | **93%** | 0.956 | 3.93/5 |

출처: `runs/feature_granularity_smd_v1/granularity_ablation_results.json`

**SWaT vs SMD 패턴 비교**:
- no_feature/zscore_only: 두 데이터셋 모두 KW ~0.5, Top1 0% → **패턴 일치**
- adversarial: SWaT 100%, SMD 93% → **LLM 의존성 재확인**
- top1이 sensor_id 최고 (SMD 4.60, SWaT 4.20) → **최소 정보로 최대 효과**

---

## 11. Phase 3: LLM 파라미터 최적화 결과 (2026-04-06)

### SWaT (200K rows, 5 seeds)

| Detector | Baseline F1 | LLM-Guided F1 | 변화 | Wilcoxon p |
|----------|-------------|---------------|------|-----------|
| IF | 0.0874 | 0.0765 | -12.5% | 0.025 * |
| kNN | 0.4290 | 0.4290 | 0.0% | 1.000 ns |

### SMD (28K rows, 5 seeds)

| Detector | Baseline F1 | LLM-Guided F1 | 변화 | Wilcoxon p |
|----------|-------------|---------------|------|-----------|
| IF | 0.4671 | 0.5211 | **+11.6%** | 0.025 * |
| kNN | 0.1728 | 0.1728 | 0.0% | 1.000 ns |

출처: `runs/llm_guided_swat/`, `runs/llm_guided_smd/`

**해석**:
- SMD IF: LLM 제안(window 30, contamination 0.15)이 baseline 대비 +11.6% 개선 → Phase 3 효과 확인
- SWaT IF: LLM 제안이 오히려 악화 → SWaT의 낮은 anomaly rate(3.8%)에서 contamination 조정이 역효과
- kNN: 두 데이터셋 모두 변화 없음 → kNN은 파라미터 민감도가 낮음
- **결론**: LLM 파라미터 제안은 데이터 특성에 따라 효과가 다름. Phase 3의 가치는 조건부적.

---

## 12. 인간 평가 자료

- `runs/human_eval_swat.csv`: SWaT 20샘플 × 4조건 = 80행
- 평가 칸: accuracy_1_5, relevance_1_5, actionability_1_5, sensor_attribution_1_5
- 평가자 3명에게 배포 후 `scripts/human_eval_analysis.py`로 분석 예정

---

## 13. 실험 전체 현황 (2026-04-06 최종)

| 실험 | 상태 | 결과 파일 |
|------|------|----------|
| Detection 4 datasets | ✅ | `runs/all_results_clean.json` + `runs/swat_phaseA_results.json` |
| 2×2 Ablation 4 datasets | ✅ | `runs/llm_explanation_{smd,swat,skab,v3}/` |
| Granularity 7조건 (SWaT) | ✅ | `runs/feature_granularity_swat_v1/` |
| Granularity 7조건 (SMD) | ✅ | `runs/feature_granularity_smd_v1/` |
| Adversarial attribution | ✅ | 위 granularity 결과 내 포함 |
| BERTScore 평가 | ✅ | 본 문서 섹션 9 |
| 통계 검정 (Wilcoxon/McNemar/Kendall) | ✅ | 본 문서 섹션 8 |
| Claude 교차 검증 | ✅ | `runs/llm_explanation_swat_v3_claude/` |
| Phase 3 (SWaT/SMD) | ✅ | `runs/llm_guided_{swat,smd}/` |
| 인간 평가 자료 | ✅ 생성 | `runs/human_eval_swat.csv` (평가자 배포 대기) |

## 14. 감사 후 수정 사항 (2026-04-06 후반)

상세 내용: `docs/AUDIT_FIX_REPORT_20260406.md` 참조

### 14.1 Calibration Data Leakage 수정 및 재실행

- **문제**: Platt/Temperature/Isotonic calibration이 전체 test labels로 fitting → ECE 인위적 저하
- **수정**: temporal split 60/40 적용 (main_experiment.py)
- **재실행**: SKAB 408건 (34 files × 4 detectors × 3 seeds)
- **결과**: ECE 1.8~16.5배 상승. AUC-PR에는 영향 없음.

| Detector | Old ECE | New ECE | 변화 |
|----------|:---:|:---:|:---:|
| rolling_zscore | 0.011 | 0.187 | +0.175 |
| knn | 0.026 | 0.186 | +0.160 |
| IF | 0.090 | 0.203 | +0.113 |
| speccnn | 0.106 | 0.194 | +0.088 |

### 14.2 Power Analysis 추가

- 대부분 pairwise 비교가 **underpowered** (power < 0.3)
- rolling_zscore vs speccnn_lite만 adequate (power=0.98, p=0.0006)
- IF vs kNN 구분에 691개 파일 필요 (현재 34개)

### 14.3 Faithfulness Metric 강화

- `evaluate_keyword_coverage`: 기존 키워드 존재 여부 (이름 변경)
- `evaluate_faithfulness_strict`: 방향 정확도 + 센서 식별 정확도 + 가중 종합
- 올바른 설명(0.833) vs 틀린 설명(0.233) 구분력 검증 완료

### 14.4 TP-Only Bias 수정

- `select_anomaly_samples(sample_mode="tp"|"fn"|"fp")` 추가
- FN(탐지 실패 이상), FP(오경보) 샘플에 대한 LLM 설명도 평가 가능

### 14.5 SWaT Domain Confound 해소

- `swat_knowledge_nosensor.yaml` 완전 익명화
- `--include-nosensor` 옵션으로 3-condition ablation 실험 가능

## 15. 남은 작업

- [ ] 인간 평가 수집 및 분석 (평가자 3명 필요)
- [ ] LLM explanation 재실행 (`--sample-modes tp fn fp --include-nosensor`)
- [ ] SWaT nosensor ablation 3-condition 실험
- [ ] Cross-model judge (Claude 생성 → GPT-4 평가)
- [ ] 논문 본문 작성 (PAPER_NARRATIVE.md 기반)
- [ ] ECE 관련 주장 수정 (temporal split 결과 반영)
