# LFactory 3차 코드 리뷰 수정 기록

**날짜**: 2026-03-20
**목적**: 2차 리뷰 후 발견된 미구현 항목 3개 + 데이터 로더 stdout 오염 수정

---

## 수정 항목 요약

| # | 항목 | 파일 | 심각도 | 상태 |
|---|------|------|--------|------|
| A | Optimal-F1 Threshold 함수 추가 | `metrics.py`, `main_experiment.py` | HIGH | 완료 |
| B | multi_seed 메트릭 추출+집계 | `multi_seed_experiment.py` | HIGH | 완료 |
| C | SKAB anomaly-free 파일 필터 | `loader_skab.py` | MED | 완료 |
| D | SMD --all-files 지원 | `multi_seed_experiment.py` | MED | 완료 |
| E | 데이터 로더 stdout→stderr 리다이렉트 | `loader_skab.py`, `loader_smd.py`, `loader_aihub_71802.py`, `data_router.py` | HIGH | 완료 |
| F | RQ2 스크립트 optimal_f1 추출 추가 | `run_rq2_ensemble.py` | LOW | 완료 |

---

## A. Optimal-F1 Threshold (`experiments/metrics.py`)

### 문제

- Quantile=0.99 기반 F1이 SKAB에서 0.000 (anomaly rate=33%에 상위 1%만 flag)
- 논문에 보고할 "실제 탐지 성능" 지표가 없음
- AUC-PR은 ranking 능력이지 "얼마나 잘 잡는가"는 아님

### 해결

`find_optimal_f1_threshold(y_true, scores)` 함수 추가:
- Unique score 값 기반 threshold 후보 생성
- 성능 최적화: unique > 2000이면 균등 서브샘플링
- 각 threshold에서 F1 직접 계산, 최대값 반환
- 반환: `(best_threshold, best_f1, best_metrics_dict)`

### 통합

- `main_experiment.py`: AUC-PR 계산 직후 optimal-F1 계산
- 결과 JSON에 `optimal_f1`, `optimal_f1_threshold`, `optimal_f1_precision`, `optimal_f1_recall` 추가
- REPORT.md에 Optimal-F1 라인 추가

### 지표 전략

| 지표 | 역할 | 의미 |
|------|------|------|
| AUC-PR | Primary | Threshold-free ranking 능력 |
| Optimal-F1 | Secondary | Oracle threshold에서 최대 F1 (upper bound) |
| Quantile-F1 | Reference | 기존 고정 threshold F1 |

---

## B. multi_seed_experiment.py 메트릭 집계

### 문제

- subprocess 실행 후 성공/실패만 기록
- AUC-PR, F1, optimal-F1 mean±std 계산 없음 → 논문 테이블 생성 불가

### 해결

1. `_extract_metrics(result)`: subprocess stdout JSON 파싱 → f1/auc_pr/auc_roc/optimal_f1/precision/recall 추출
2. `_aggregate_metrics(results)`: (dataset, detector, ml_method) 그룹별 mean±std 계산
3. 출력: `runs/multi_seed_metric_summary.csv` + `.json` 자동 생성
4. 콘솔에 정렬된 요약 테이블 출력

---

## C. SKAB anomaly-free 파일 필터링

### 문제

- SKAB `anomaly-free/` 디렉토리의 파일은 label_rate=0
- AUC-PR이 정의 불가 (positive 없음) → 집계 시 NaN/오류

### 해결

- `_find_skab_files(root, include_anomaly_free=True)`: 파라미터화
- `list_all_files()`: default `include_anomaly_free=False`
- `load_one_timeseries()`: `include_anomaly_free=False`
- 결과: 35 → 34 파일 (anomaly-free 1개 제외)

---

## D. SMD --all-files 지원

### 문제

- `--all-files`가 SKAB만 지원, SMD 28대 머신 순회 불가

### 해결

- `_get_smd_file_count(data_root)`: SMD/test 디렉토리의 .txt/.csv 파일 수 반환
- 메인 루프: `all_files and dataset == "SMD"` 분기 추가

---

## E. 데이터 로더 stdout→stderr 리다이렉트

### 문제

- `loader_skab.py`, `loader_smd.py`, `loader_aihub_71802.py`, `data_router.py`의 진단 print가 stdout으로 출력
- `main_experiment.py`의 JSON 결과와 섞여 `multi_seed_experiment.py`의 JSON 파싱 실패
- 예: `[SKAB] Loaded 1147 points from 0.csv` + `{...JSON...}` → JSONDecodeError

### 해결

- 모든 데이터 로더의 진단 print를 `print(..., file=sys.stderr)`로 변경
- stdout은 순수 JSON만 출력되도록 보장

---

## F. RQ2 스크립트 optimal_f1 추출

### 변경

- `run_rq2_ensemble.py`: results에 `optimal_f1` 필드 추가

---

## 검증 결과

### Unit Test: find_optimal_f1_threshold

```
labels = [0]*90 + [1]*10
scores = [0.1]*85 + [0.5]*5 + [0.9]*8 + [0.3]*2
→ Optimal F1=0.889 at thr=0.7000 (P=1.000, R=0.800)
Edge case (no positives): F1=0.000 ✓
Edge case (empty): F1=0.000 ✓
```

### SKAB anomaly-free 필터

```
With anomaly-free: 35 files
Without anomaly-free: 34 files
anomaly-free excluded from default: ✓
```

### Smoke Test (synthetic, 4 detectors)

| Detector | F1 | Optimal-F1 | AUC-PR | AUC-ROC |
|----------|-----|-----------|--------|---------|
| rule | 0.177 | 0.270 | 0.178 | 0.563 |
| ml | 0.374 | 0.475 | 0.352 | 0.698 |
| hybrid | 0.374 | 0.485 | 0.359 | 0.652 |
| speccnn | 0.088 | 0.360 | 0.194 | 0.908 |

모든 디텍터에서 `optimal_f1 >= f1` 성립 ✓

### SKAB file_index=0 검증

| Detector | F1 | Optimal-F1 | AUC-PR | P@opt | R@opt |
|----------|-----|-----------|--------|-------|-------|
| rule | 0.000 | 0.519 | 0.335 | 0.350 | 1.000 |
| ml | 0.010 | 0.518 | 0.316 | 0.350 | 1.000 |
| hybrid | 0.019 | 0.519 | 0.329 | 0.350 | 1.000 |
| speccnn | 0.000 | 0.543 | 0.287 | 0.373 | 0.995 |

- AUC-PR ≈ 0.33 = base rate → 디텍터가 랜덤 수준 (이 파일에서)
- Optimal-F1 ≈ 0.52: 최적 threshold 시 의미 있는 탐지 가능
- Rule/ML/Hybrid의 P@opt=0.350, R@opt=1.000: "모든 것을 anomaly로 예측"이 최적 → score가 anomaly를 구분하지 못함
- SpecCNN은 P@opt=0.373으로 약간 나음

---

## 변경된 파일 목록

| 파일 | 변경 유형 | 줄 수 (approx) |
|------|----------|---------------|
| `experiments/metrics.py` | 함수 추가 | +27 |
| `experiments/main_experiment.py` | 통합 코드 | +10 |
| `scripts/multi_seed_experiment.py` | 전면 리팩토링 | +60 (net) |
| `experiments/data/loader_skab.py` | 파라미터 추가 | +8 |
| `experiments/data/loader_smd.py` | stderr 리다이렉트 | +2 |
| `experiments/data/loader_aihub_71802.py` | stderr 리다이렉트 | +2 |
| `experiments/data/data_router.py` | stderr 리다이렉트 | +4 |
| `scripts/run_rq2_ensemble.py` | 필드 추가 | +1 |

---

## 4차 검토 수정 (2026-03-24)

### 발견된 문제 및 수정

#### G. AIHub71802 stdout 오염 잔존 (CRITICAL)

- **파일**: `experiments/data/loader_aihub_71802.py` lines 153, 170, 186, 188
- **문제**: 3차에서 line 199의 stderr 리다이렉트만 수정, WARN 메시지 4곳은 여전히 stdout
- **영향**: AIHub 실험 시 JSON 파싱 깨짐
- **수정**: 4곳 모두 `print(..., file=sys.stderr)` 추가, `import sys` 상단 이동

#### H. all_results_clean.json 중복 문제

- **문제**: 같은 (dataset, detector, seed, file) 실험이 여러 디렉토리에 중복 존재
  - 원인 1: Step1 + multi_seed + RQ4에서 같은 실험 중복 실행
  - 원인 2: SKAB valve1/0.csv와 valve2/0.csv가 basename=0.csv로 동일
- **해결**: `scripts/aggregate_results.py` 작성 — `meta.path` (full path) 기반 dedup, 최신 run 유지
- **결과**: 1,023 raw → 931 unique entries

#### I. RQ4 timeout 부족 + SMD/synthetic 미포함

- **파일**: `scripts/run_rq4_cost_sensitivity.py`
- **문제**: 600초 timeout → SMD 일부 디텍터 실패
- **수정**: 1800초로 증가
- **보완**: 기존 preds.csv에서 offline cost_sensitivity_sweep 실행하여 SMD/synthetic 결과 추가

#### J. 보고서 SKAB "16파일" 표기 오류

- **파일**: `docs/EXPERIMENT_RESULTS_3RD.md`
- **수정**: "16파일" → "34파일 (16 unique basenames across valve1/valve2/other)"

#### K. SMD IF N=1 주석 추가

- **파일**: `docs/EXPERIMENT_RESULTS_3RD.md`
- **수정**: IF 결과에 "N=1, 통계적 신뢰도 낮음" 주석 추가

### 변경된 파일

| 파일 | 변경 유형 |
|------|----------|
| `experiments/data/loader_aihub_71802.py` | 4 prints → stderr, import sys 상단 이동 |
| `scripts/run_rq4_cost_sensitivity.py` | timeout 600→1800 |
| `scripts/aggregate_results.py` | 신규: dedup 집계 스크립트 |
| `docs/EXPERIMENT_RESULTS_3RD.md` | 파일 수 정정, N=1 주석, RQ4 범위, 수치 갱신 |
| `docs/CHANGES_3RD_REVIEW.md` | 4차 검토 섹션 추가 |
| `docs/CODE_REVIEW_20260320.md` | 섹션 10 추가 |
| `docs/CODE_REVIEW_20260320.md` | 문서 갱신 | +60 |
