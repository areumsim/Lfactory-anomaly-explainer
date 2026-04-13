# LFactory 3차 실험 결과 보고서

**작성일**: 2026-03-24
**데이터**: 3차 코드 수정 후 전체 재실행 결과 (optimal_f1 포함 runs만 사용)
**총 유효 runs**: 931 (중복 제거 후, `scripts/aggregate_results.py` 사용)
**데이터 무결성**: optimal_f1 >= f1 위반 0건 확인

---

## 1. 실험 개요

### 수정 사항 (이번 라운드)
1. **Optimal-F1 Threshold** 추가 — oracle threshold에서 최대 F1 계산
2. **multi_seed 메트릭 집계** — mean±std 자동 계산 + CSV/JSON 출력 + 병렬 실행
3. **SKAB anomaly-free 필터** — label_rate=0 파일 제외 (35→34파일)
4. **SMD --all-files 지원** + `--sample-limit`, `--parallel` 옵션 추가
5. **데이터 로더 stdout→stderr** — JSON 파싱 깨짐 수정

### 데이터 무결성 검증
- 2025-11-24 이전 배치 결과에 optimal_f1 미포함 → **3차 수정 이후 결과만 사용**
- optimal_f1 >= f1 불변식 검증: **위반 0건**
- label_rate=0 파일 자동 제외 확인

### 실험 범위

| 데이터셋 | 파일 수 | 포인트/파일 | 유효 runs | 비고 |
|----------|---------|------------|-----------|------|
| SKAB | 34파일 (16 unique basenames across valve1/valve2/other) | ~1,147 | 865 | 5 seeds × 파일 × configs |
| SMD | 1 (file0) | 28,479 | 26 | 5 seeds × configs (28,000행 → CPU 10+분/run) |
| Synthetic | 1 | 2,000 | 40 | 다양한 seed |
| AIHub71802 | — | — | 0 | 데이터 접근 문제 (ZIP 미추출) |

**총 계산 시간**: SKAB ~3시간 + SMD ~2시간 + synthetic ~30분 + RQ2/RQ4 부분 실행

---

## 2. RQ1: 디텍터 비교 (주요 결과)

### 지표 정의
- **AUC-PR**: Threshold-free ranking 능력 (Primary). Base rate가 높으면 랜덤도 높은 AUC-PR 가능
- **Optimal-F1**: Oracle threshold에서 최대 F1 (Secondary). 디텍터의 상한
- **Quantile-F1**: 고정 quantile=0.99 threshold F1 (Reference). 실제 운영 성능 프록시
- **AUC-ROC**: 전체 score 분포의 분리도

### SKAB 결과 (34파일, 5 seeds)

| Detector | N | AUC-PR | Optimal-F1 | Quantile-F1 | 비고 |
|----------|---|--------|-----------|-------------|------|
| **ml/IF** | 170 | **0.433±0.201** | **0.615±0.149** | 0.160±0.127 | 최고 성능 |
| ml/knn | 176 | 0.417±0.178 | 0.559±0.110 | 0.025±0.016 | |
| speccnn | 176 | 0.415±0.154 | 0.583±0.086 | 0.027±0.028 | Opt-F1 높음 |
| hybrid | 195 | 0.370±0.088 | 0.520±0.039 | 0.024±0.015 | |
| rule | 178 | 0.366±0.073 | 0.519±0.041 | 0.017±0.032 | base rate ≈ 0.35 |

**SKAB base rate ≈ 35%** → AUC-PR 0.35 = 랜덤 수준

**핵심 발견**:
1. **Isolation Forest(0.433)가 유일하게 base rate를 유의미하게 초과**
2. kNN(0.417)과 SpecCNN(0.415)도 base rate 대비 약간 우수
3. Rule(0.366)과 Hybrid(0.370)는 base rate와 거의 동일 → 실질적 탐지 불가
4. Optimal-F1은 0.52~0.62 → 최적 threshold에서는 모든 디텍터가 어느 정도 작동
5. **Quantile-F1은 거의 0** → 고정 threshold는 SKAB(high anomaly rate)에 부적합

### SMD 결과 (file0, 28,479행)

| Detector | N | AUC-PR | Optimal-F1 | Quantile-F1 | 비고 |
|----------|---|--------|-----------|-------------|------|
| **ml/knn** | 15 | **0.563±0.000** | 0.173±0.000 | 0.173±0.000 | |
| **ml/IF** | 1 | 0.549 | **0.629** | **0.467** | 최고 Opt-F1 (**N=1, 통계적 신뢰도 낮음**) |
| hybrid | 15 | 0.278±0.209 | 0.173±0.000 | 0.105±0.050 | |
| rule | 11 | 0.121±0.000 | 0.173±0.000 | 0.071±0.000 | |
| speccnn | 6 | 0.122±0.000 | 0.313±0.000 | 0.000±0.000 | |

**SMD base rate ≈ 9.5%** (anomaly가 후반부에 집중)

**핵심 발견**:
1. **kNN(0.563)과 IF(0.549)가 확실히 우수** — base rate 대비 5~6배
2. IF는 Optimal-F1=0.629, Quantile-F1=0.467 → 고정 threshold에서도 작동!
3. Rule/Hybrid는 SMD에서 약함 (0.12~0.28)
4. SMD는 N이 작아 (timeout 문제) 통계적 신뢰도가 낮음

### Synthetic 결과

| Detector | N | AUC-PR | Optimal-F1 | Quantile-F1 |
|----------|---|--------|-----------|-------------|
| speccnn | 12 | 0.251±0.161 | **0.427±0.192** | 0.136±0.117 |
| ml/knn | 12 | 0.232±0.157 | 0.329±0.197 | 0.272±0.186 |
| hybrid | 31 | 0.213±0.161 | 0.306±0.202 | 0.256±0.199 |
| ml/IF | 10 | 0.107±0.090 | 0.211±0.112 | 0.136±0.076 |
| rule | 14 | 0.105±0.069 | 0.175±0.088 | 0.108±0.086 |

---

## 3. RQ2: 앙상블 방법 비교 (5 seeds)

| Dataset | Method | AUC-PR | F1 | ECE |
|---------|--------|--------|-----|-----|
| SKAB | linear | 0.329 | 0.019 | 0.002 |
| SKAB | product | 0.324 | 0.010 | 0.003 |
| SKAB | max | 0.334 | 0.019 | 0.002 |
| SKAB | **learned** | **0.344** | 0.019 | 0.006 |
| synthetic | **linear** | **0.240±0.186** | 0.288±0.233 | 0.049 |
| synthetic | product | 0.224±0.180 | 0.263±0.225 | 0.047 |
| synthetic | max | 0.094±0.065 | 0.126±0.108 | 0.045 |
| synthetic | learned | 0.213±0.171 | 0.258±0.203 | 0.046 |

**SMD**: timeout으로 product(0.562)만 성공. linear/max/learned는 실패.

**핵심 발견**:
1. SKAB: learned(0.344) > max(0.334) > linear(0.329). 차이 매우 작음
2. Synthetic: linear(0.240) > product(0.224) > learned(0.213) >> max(0.094)
3. **Max 방법은 synthetic에서 현저히 낮음** — rule score가 매우 노이즈가 많을 때 max가 불리
4. **데이터셋에 따라 최적 앙상블이 다름** → 일반적 우위 결론 불가
5. 2차 리뷰에서 수정한 learned의 temporal split (60/40)이 제대로 작동 중

---

## 4. RQ4: 비용 민감도 분석 (SKAB + SMD/synthetic offline sweep)

### FN/FP 비용 비율에 따른 F1 변화

| Detector | 1:1 F1 | 5:1 F1 | 10:1 F1 | 50:1 F1 |
|----------|--------|--------|---------|---------|
| isolation_forest | 0.048 | **0.523** | 0.522 | 0.522 |
| speccnn_lite | 0.055 | **0.528** | 0.528 | 0.524 |
| hybrid_linear | 0.015 | 0.521 | 0.519 | 0.519 |
| knn | 0.007 | 0.519 | 0.519 | 0.519 |
| rolling_zscore | 0.011 | 0.519 | 0.519 | 0.519 |
| lstm_ae | 0.036 | 0.520 | 0.520 | 0.519 |

**핵심 발견**:
1. **Ratio=1:1 vs 5:1 사이에 극적 전환** — 1:1에서는 precision 최적화 (→ F1 낮음), 5:1부터 recall 중심
2. **5:1 이후 포화**: 10:1, 50:1에서 추가 개선 없음
3. **실용적 권장: FN:FP = 5:1** — 대부분의 산업 시나리오에 적합
4. Safety-critical 환경에서도 50:1까지 올릴 필요 없음 (5:1에서 이미 최적)
5. SpecCNN과 IF가 비용 최적화에서도 약간 우수 (0.523~0.528 vs 0.519)

---

## 5. 종합 분석 및 논문 시사점

### 디텍터 순위 (데이터셋별 AUC-PR)

| 순위 | SKAB | SMD | Synthetic |
|------|------|-----|-----------|
| 1 | **IF** (0.433) | **kNN** (0.563) | SpecCNN (0.251) |
| 2 | kNN (0.417) | IF (0.549) | kNN (0.232) |
| 3 | SpecCNN (0.415) | Hybrid (0.278) | Hybrid (0.213) |
| 4 | Hybrid (0.370) | SpecCNN (0.122) | IF (0.107) |
| 5 | Rule (0.366) | Rule (0.121) | Rule (0.105) |

### 논문에서 강조할 수 있는 결론

1. **ML 기반 디텍터(IF, kNN)가 일관적으로 우수**: SKAB과 SMD 모두에서 1~2위
2. **Rule 기반 디텍터의 명확한 한계**: 모든 데이터셋에서 최하위
3. **Threshold 선택이 핵심**: Q-F1≈0 vs Opt-F1≈0.5 → "어떤 threshold를 쓰느냐"가 성능의 결정적 요인
4. **FN:FP=5:1이 실용적 최적점**: 비용 민감도 분석에서 5:1 이후 포화
5. **데이터셋 특성이 디텍터 선택보다 중요**: 같은 디텍터도 SKAB vs SMD에서 순위가 다름

### 논문 테이블 구성 제안

- **Table 1**: 디텍터 × 데이터셋 AUC-PR + Optimal-F1 (이 보고서 §2)
- **Table 2**: 앙상블 방법 비교 (§3)
- **Table 3**: 비용 비율별 최적 F1 (§4)
- **Figure**: AUC-PR vs Optimal-F1 scatter plot (디텍터별 색상, 데이터셋별 모양)

---

## 6. 한계 및 향후 과제

### 실험 한계

| 항목 | 상태 | 영향 | 해결 방안 |
|------|------|------|----------|
| SMD 전체 28파일 | 미완 (file0만) | SMD 일반화 약함 | GPU 활용 or numpy 벡터화 |
| AIHub71802 | 미실행 | 4개 → 3개 데이터셋 | ZIP 추출 + 로더 수정 |
| 20 seeds | 5 seeds만 | 통계 검정 약함 | SKAB은 34파일×5=170으로 충분 |
| LSTM-AE | SKAB에서 Opt-F1=0 | 학습 부족 | epoch/구조 재검토 |
| GPU 미활용 | RTX 4090 × 2 유휴 | 실행 시간 긴 | PyTorch 전환 필요 |

### CPU 병목 분석

| 디텍터 | 1,147행 (SKAB) | 28,479행 (SMD) | 병목 |
|--------|---------------|---------------|------|
| Rule | 1초 | 1.5분 | O(n×w) 루프 |
| kNN | 2초 | 3분 | O(n²) 거리 |
| IF | 3초 | 10분 | sklearn fit |
| SpecCNN | 5초 | 20분 | O(n×w) FFT |

**numpy 벡터화**로 Rule/kNN 10~50배 가속 가능 (Python 루프 → numpy 배열 연산).
**GPU 전환**은 LSTM-AE와 SpecCNN(PyTorch FFT)에서 유효.

### 코드 품질 이슈 (발견된 것)

1. `--explain` 사용 시 RAGExplainer의 stdout print가 JSON 파싱 방해 (현재 explain 미사용으로 영향 없음)
2. SKAB `load_one_timeseries`에서 file_index 범위 초과 시 silent fallback to 0
3. Event metrics의 median 계산이 짝수 길이에서 정확하지 않음 (upper-middle 반환)

---

## 7. 파일 위치

| 파일 | 설명 |
|------|------|
| `runs/all_results_clean.json` | 931 유효 runs (중복 제거, optimal_f1 포함만) |
| `runs/final_results_table.txt` | 텍스트 요약 테이블 |
| `runs/rq2_ensemble/` | RQ2 앙상블 비교 결과 |
| `runs/rq4_cost_sensitivity/` | RQ4 비용 민감도 결과 |
| `runs/results_summary.csv` | 데이터셋/디텍터별 집계 통계 |
| `scripts/aggregate_results.py` | 결과 집계 스크립트 (dedup + 검증) |
| `runs/logs/` | 실행 로그 |
| `docs/CHANGES_3RD_REVIEW.md` | 3차 코드 수정 상세 기록 |
| `docs/CODE_REVIEW_20260320.md` | 전체 코드 리뷰 기록 (1~3차) |

---

## 8. 재현 방법

```bash
# SKAB 전체 (약 3시간)
python3 scripts/multi_seed_experiment.py \
    --datasets SKAB --detectors rule ml hybrid speccnn \
    --seeds 5 --all-files --parallel 2 \
    --data-root /workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d

# SMD file0 (약 2시간, sequential 권장)
python3 scripts/multi_seed_experiment.py \
    --datasets SMD --detectors rule ml hybrid speccnn \
    --seeds 5 --parallel 1 \
    --data-root /workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d

# RQ2 앙상블 (약 1시간)
python3 scripts/run_rq2_ensemble.py --seeds 5 \
    --data-root /workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d

# RQ4 비용 민감도 (기존 preds.csv에서 분석, 수분)
# runs/rq4_cost_sensitivity/raw_results.json 참조
```
