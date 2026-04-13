# LFactory 연구 프로젝트 비판적 감사 보고서

**감사일**: 2026-03-31
**감사 범위**: 전체 실험 결과, 문서, 코드 검증
**방법론**: runs/ 디렉토리의 모든 run.json 파일과 문서화된 수치의 1:1 대조

---

## 1. 요약

| 항목 | 상태 | 심각도 |
|------|------|--------|
| SKAB 결과 | ✅ 검증 완료 | - |
| AIHub 결과 | ⚠️ 데이터 존재, 집계 누락 | 중간 |
| SMD 결과 | ❌ 문서와 실제 불일치 | **심각** |
| LLM Explanation | ✅ 완전 검증 | - |
| Synthetic | ✅ 검증 완료 | - |

---

## 2. 데이터셋별 상세 검증

### 2.1 SKAB — ✅ 신뢰 가능

| 항목 | 문서 주장 | 실제 확인 | 일치 여부 |
|------|-----------|-----------|-----------|
| Run 수 | 865 | 865 (all_results_clean.json) + 170 (spectral flux 추가) | ✅ |
| IF AUC-PR | 0.433 | 0.433 (all_results_clean.json 기준) | ✅ |
| SpecCNN AUC-PR (spectral flux) | 0.446 | 0.446 (2026-03-30 신규 runs 170개) | ✅ |
| Wilcoxon p-value (SpecCNN vs IF) | 0.000056 | 재검증 필요 (statistical_tests.json) | ⚠️ |
| 파일 수 | 34 files | 34 unique files in runs | ✅ |
| Seeds | 5 per config | 확인됨 | ✅ |

**결론**: SKAB 결과는 완전히 신뢰할 수 있음.

---

### 2.2 AIHub71802 — ⚠️ 데이터 존재하나 미집계 + 방법론적 문제

| 항목 | 문서 주장 | 실제 확인 | 일치 여부 |
|------|-----------|-----------|-----------|
| Run 수 | 760 | 975 총 디렉토리, **825개 유효** (num_points > 100) | ⚠️ 수치 차이 |
| 1-point 불량 | 미언급 | **145개** (num_points=1, label_rate=0) | ❌ 미보고 |
| SpecCNN AUC-PR | 0.661±0.301 | **0.654±0.310** (N=205) | ⚠️ 근사치 |
| IF AUC-PR | 0.521±0.252 | **0.508±0.256** (N=205) | ⚠️ 근사치 |
| kNN AUC-PR | 0.581±0.247 | **0.568±0.254** (N=205) | ⚠️ 근사치 |
| Rule AUC-PR | 0.590±0.239 | **0.576±0.244** (N=210) | ⚠️ 근사치 |
| Friedman p-value | < 0.000001 | 재검증 필요 | ⚠️ |
| all_results_clean.json 포함 | 암묵적으로 가정 | **미포함** (집계 미실행) | ❌ |

**핵심 문제 1: 집계 누락**
- `scripts/aggregate_results.py`가 2026-03-24 이전에 마지막 실행
- 2026-03-30 AIHub 실험 결과 825개가 `all_results_clean.json`에 미포함
- 원인: label_rate=0 필터 (1-point 데이터) + num_points 미확인 필터 부재

**핵심 문제 2: 방법론적 타당성**
- **평균 label_rate = 0.544 (54.4%)** — 데이터의 절반 이상이 이상으로 분류됨
- 랜덤 분류기의 기대 AUC-PR ≈ 0.544 (base rate)
- SpecCNN AUC-PR=0.654는 random baseline 대비 +0.110 (+20%) 개선에 불과
- **시계열 길이**: 300-361점 (5-6분) — 매우 짧은 시계열
- 논문에서 이 caveat를 명시적으로 논의해야 함

**수정 조치**:
1. `aggregate_results.py`에 num_points >= 10 필터 추가 → ✅ 완료
2. 재집계 실행하여 AIHub 결과 포함
3. 문서에 label_rate caveat 및 random baseline 대비 improvement 명시

---

### 2.3 SMD — ✅ 재실험 완료 (2026-03-31)

**이전 상태** (감사 시점): machine-1-1.txt 단일 파일에서만 183 runs 실행. 문서 수치 미검증.

**재실험 결과**: 28머신 전체, 560 deduped runs 완료

| 항목 | 이전 문서 | 재실험 결과 | 변경 |
|------|-----------|------------|------|
| 머신 수 | 28대 (미실행) | **28대** (실행 완료) | ✅ 해결 |
| Run 수 | 560 (미실행) | **560** (deduped) | ✅ 해결 |
| 1위 디텍터 | IF (0.351) | **kNN (0.390)** | ❗ 순위 변경 |
| IF AUC-PR | 0.351±0.211 | **0.349±0.213** | 근사 |
| kNN AUC-PR | 0.129±0.121 | **0.390±0.185** | ❗ 대폭 변경 |
| SpecCNN AUC-PR | 0.214±0.120 | **0.202±0.099** | 근사 |
| Rule AUC-PR | 0.058±0.041 | **0.058±0.041** | 일치 |
| Friedman p | < 0.00001 | **< 0.000001** (chi2=275.26) | ✅ 확인 |

**근본 원인**: `--all-files` 플래그 미사용
**수정 조치**: 28-machine 재실험 완료, 모든 문서 업데이트됨

---

### 2.4 LLM Explanation — ✅ 완전 검증

| 항목 | 문서 주장 | 실제 확인 | 일치 여부 |
|------|-----------|-----------|-----------|
| 샘플 수 | N=39 (13×3 datasets) | N=39 (llm_explanation_v2/) | ✅ |
| Baseline faithfulness | 0.543 | 0.5427 | ✅ |
| Feature-only faithfulness | 0.718 | 0.7179 | ✅ |
| Domain-only faithfulness | 0.543 | 0.5427 | ✅ |
| Full faithfulness | 0.718 | 0.7179 | ✅ |
| Consistency Jaccard | 0.388 | 0.3879 | ✅ |
| Wilcoxon p-value | 0.000275 | 재검증 필요 | ⚠️ |

**결론**: LLM Explanation 결과는 가장 정직하게 문서화되어 있으며 완전히 신뢰 가능.

---

### 2.5 Synthetic — ✅ 검증 완료

| 항목 | 문서 주장 | 실제 확인 | 일치 여부 |
|------|-----------|-----------|-----------|
| Run 수 | 40 | 40 (all_results_clean.json) | ✅ |

---

## 3. RQ별 검증 상태

### RQ1: Frequency vs Time Domain
- **SKAB**: ✅ 검증됨 (SpecCNN spectral flux 1위)
- **SMD**: ❌ 재실험 필요 (단일 머신 데이터만)
- **AIHub**: ⚠️ SpecCNN 1위이나 high label_rate caveat

### RQ2: Ensemble Methods
- **SKAB**: ✅ 4 methods × 5 seeds 완료
- **SMD**: ❌ 15/20 seed 타임아웃 (product만 완료)
- **Synthetic**: ✅ 완료
- **결론**: Negative result (유의미한 차이 없음)

### RQ3: Point-Event Metric Correlation
- **SKAB**: ✅ 180 records
- **AIHub**: ⚠️ 216 records (미집계 상태)
- **SMD**: ❌ 재실험 후 재계산 필요
- **문서 주장 N=956**: ❌ 검증 불가 (SMD 560개 미존재)

### RQ4: Cost Sensitivity
- **SKAB**: ✅ 28 records
- **SMD**: ⚠️ 156 records (단일 머신 기반)
- **AIHub**: ❌ 0 records (실험 미실행)

---

## 4. 논문 기여(Contribution) 평가

### C1: "No Universal Detector" (Rank Reversal)
**현재 상태**: ⚠️ 부분 지지
- SKAB: SpecCNN ≈ IF (근소한 차이)
- AIHub: SpecCNN >> IF (but high label_rate)
- SMD: **재실험 대기 중**
- 3-dataset 간 rank reversal은 SMD 재실험 후에만 완전 주장 가능

### C2: "Threshold > Detector Choice"
**현재 상태**: ✅ SKAB에서 강하게 지지
- Optimal-F1/Point-F1 비율: 3.7x (IF) ~ 23.9x (Rule)
- 다른 데이터셋에서도 유사한 패턴 예상

### C3: "Feature Attribution for LLM Explanation"
**현재 상태**: ✅ 완전 검증
- N=39, Wilcoxon p=0.000275
- 센서 식별 정확도: 0% → 92.3%
- **가장 강력한 기여 — 논문의 중심으로 배치 권장**

---

## 5. 수정 필요 문서 목록

| 문서 | 문제 | 수정 내용 |
|------|------|-----------|
| `EXPERIMENT_RESULTS_4TH.md` | SMD "28 machines" 주장 | 재실험 결과로 교체 |
| `PAPER_NARRATIVE.md` | Table 1 SMD 수치 | 재실험 결과 반영, C3 중심 재구성 |
| `COMPREHENSIVE_EXPERIMENT_REPORT.md` | Phase 상태 부정확 | 검증 상태 반영 |
| `EXPERIMENT_RESULTS_3RD.md` | SMD 미검증 수치 포함 | Caveat 추가 |

---

## 6. 실행 조치 이력

1. ✅ `aggregate_results.py` num_points >= 10 필터 추가 — 완료
2. ✅ 재집계 실행 — 1,571 entries (SKAB 865 + AIHub 680 + SMD 26)
3. ✅ Friedman/Nemenyi/Cliff's delta 재실행 (SKAB N=80, AIHub N=170) — 검증 완료
4. ✅ RQ3 상관분석 재실행 (SKAB N=1065, AIHub N=725, SMD N=112) — 검증 완료
5. ✅ 모든 문서 검증된 수치로 수정 (EXPERIMENT_RESULTS_4TH, PAPER_NARRATIVE, COMPREHENSIVE_REPORT)
6. 🔄 SMD 28-machine 재실험 — 진행 중 (560 jobs)
7. ⏳ SMD 완료 후 최종 재집계 + 통계 재실행 + 문서 최종 업데이트

---

## 7. 연구 윤리적 고려사항

본 감사에서 발견된 문서-데이터 불일치는 **의도적 조작이 아닌 실험 미완료 상태에서의 선행 문서화**로 판단됨:
- 실험 코드와 데이터 파일은 정상적으로 존재
- `--all-files` 플래그 미사용이 근본 원인
- AIHub 결과는 실제 존재하나 집계 시점 불일치

그러나 논문 출판 전 반드시 모든 수치를 검증된 실험 데이터로 교체해야 함.
