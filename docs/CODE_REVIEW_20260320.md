# 2차 코드 리뷰 & 수정 기록 (2026-03-20)

## 1. 기존 실험 결과 분석 (2025-11-24, 503 runs)

| Dataset | Detector | F1 | AUC-PR | 데이터포인트 | 비고 |
|---------|----------|-----|--------|-------------|------|
| SKAB | Rule | 0.000 | 0.335 | 1,147 | anomaly rate ~33%, q=0.99 → recall≈0 |
| SKAB | kNN | 0.010 | 0.316 | 1,147 | Rule과 동급 |
| SKAB | IF | 0.039 | 0.247 | 1,147 | 랜덤 수준 이하 |
| SKAB | LSTM-AE | 0.087 | 0.338 | 1,147 | 미미한 개선 |
| SKAB | Hybrid | 0.019 | 0.329 | 1,147 | 앙상블 효과 없음 |
| SKAB | SpecCNN | 0.518 | **0.000** | 1,147 | score 전부 0 (1차에서 수정) |
| AIHub | ALL | 0.000 | 0.000 | **1** | 경로 버그 → 1포인트만 로드 (1차에서 수정) |

**핵심 진단**: SKAB AUC-PR ≈ 0.33은 base rate(anomaly_rate=33.5%)와 동일 → 모든 디텍터가 랜덤과 구별 불가. 원인: `quantile=0.99` 임계값이 anomaly rate=33%인 SKAB에 부적합.

---

## 2. 발견된 5개 버그

### 2-1. Hybrid "learned" 앙상블 — train/test 데이터 누수 (P0)

- **파일**: `experiments/hybrid_detector.py:156-164`
- **원인**: `_fit_logistic_2d(ms[:n], rs[:n], labels[:n])` 후 동일 데이터로 scoring → 과적합
- **영향**: RQ2 (앙상블 비교) 결과 무효화. "learned가 linear보다 우수" 결론 신뢰 불가
- **수정**: Temporal split — 앞 60%로 학습, 전체에 적용. `params["learned_train_frac"] = 0.6` 기록

### 2-2. `_fit_logistic_2d` degenerate case 미처리 (P0)

- **파일**: `experiments/hybrid_detector.py:42-44`
- **원인**: 전체 label이 0 또는 1이면 gradient=0 → 의미없는 weight
- **수정**: `sum(y)==0 or sum(y)==n` 체크 추가, neutral fallback `(1.0, 1.0, 0.0)` 반환

### 2-3. RQ4 cost sweep 미구현 (P0)

- **파일**: `scripts/run_rq4_cost_sensitivity.py:71-92`
- **원인**: 모든 ratio에 동일한 5:1 결과 기록. `cost_sensitivity_sweep()` 미호출
- **영향**: RQ4 (비용 민감도 분석) 전체 무효
- **수정**: preds.csv에서 scores/labels 추출 → `cost_sensitivity_sweep()` 호출하여 ratio별 실제 최적 임계 계산

### 2-4. explain_rag 변수명 대소문자 불일치 (P0)

- **파일**: `experiments/explain_rag.py:250`
- **원인**: config에 `"SNR < 3.0"` (대문자), 코드에서 `snr=snr` (소문자) → 조건 무시
- **수정**: `_safe_eval_condition` 내 변수 조회를 case-insensitive로 변경

### 2-5. main_experiment.py explain 컨텍스트 하드코딩 (P1)

- **파일**: `experiments/main_experiment.py:557-566`
- **원인**: `"snr": 0.0` 하드코딩, `"anomaly_types"` 누락 → Bayes rules 전부 미작동
- **수정**: `cost_threshold.estimate_snr()` 호출, 검출 결과에서 anomaly type 분류 (segment 길이 기반)

---

## 3. 설계 문제 수정

### 3-1. Quantile 임계값 vs Anomaly Rate 불일치

- **문제**: SKAB anomaly_rate=33.5%에서 `quantile=0.99` → 상위 1%만 flag → recall ≈ 3%
- **해결 방향**: AUC-PR (threshold-free) 기반 primary 보고 + optimal-threshold F1 secondary
- **이론적 근거**: AUC-PR은 imbalanced data에서 threshold 선택 무관하게 ranking 능력 평가 (Davis & Goadrich, 2006)

### 3-2. SpecCNN weight config 누락

- **문제**: `sc_w_low`, `sc_w_mid`, `sc_w_high`가 config.yaml에 없어 override 불가
- **수정**: `config.yaml`에 추가, `main_experiment.py`에 `maybe_set` 연결

---

## 4. 데이터 경로 변경

| 항목 | 이전 | 이후 |
|------|------|------|
| datasets.yaml root | `/workspace/data1/arsim/LFactory_d` | `/workspace/data/nas_kbn02_01/dataset/ar_all/LFactory_d` |
| multi_seed_experiment.py | 동일 | 동일 |
| run_rq2_ensemble.py | 동일 | 동일 |
| run_rq4_cost_sensitivity.py | 동일 | 동일 |
| config.yaml 주석 | 동일 | 동일 |

---

## 5. 수정의 연구 영향도

| RQ | 영향받는 버그 | 영향 수준 | 설명 |
|----|-------------|----------|------|
| RQ1 (디텍터 비교) | 없음 (설계) | 높음 | AUC-PR 기반 재평가 필요 |
| RQ2 (앙상블) | 2-1 (데이터 누수) | **치명적** | learned 방법 결과 전면 재실행 |
| RQ3 (설명 생성) | 2-4, 2-5 | 중간 | Bayes rules 미작동 → 재실행 |
| RQ4 (비용 민감도) | 2-3 (sweep 미구현) | **치명적** | 전체 결과 무효 → 재실행 |

---

## 6. 향후 실험 계획

### Phase A: 검증 (데이터 불필요)
- [x] synthetic에서 모든 디텍터 동작 확인
- [x] 버그 수정 유닛 테스트 통과

### Phase B: 전체 재실행
- [ ] 데이터 경로 확인 후 SKAB 전체 파일(30+) 실행
- [ ] SMD 전체 머신(28) 실행
- [ ] AIHub71802 재실행

### Phase C: 분석
- [ ] AUC-PR 기반 디텍터 순위 비교
- [ ] Optimal-threshold F1 보고
- [ ] 통계 검정 (파일 간 변동을 random effect로)

---

## 7. 3차 수정 사항 (2026-03-20)

### 7-A. Optimal-F1 Threshold 함수 추가

- **파일**: `experiments/metrics.py` — `find_optimal_f1_threshold()` 추가
- **파일**: `experiments/main_experiment.py` — optimal-F1 계산 통합 + REPORT.md 출력
- **이유**: quantile F1이 SKAB에서 0.00이므로 oracle upper bound F1이 필수
- **방법**: unique score 기반 threshold sweep, F1 직접 최대화 (cost 선형결합 아님)

### 7-B. multi_seed_experiment.py 메트릭 집계

- **파일**: `scripts/multi_seed_experiment.py`
- **변경**: subprocess stdout에서 JSON 파싱 → f1/auc_pr/auc_roc/optimal_f1 추출
- **추가**: `_aggregate_metrics()` — (dataset, detector, ml_method) 그룹별 mean±std 계산
- **출력**: `runs/multi_seed_metric_summary.csv` + `.json` 자동 생성

### 7-C. SKAB anomaly-free 파일 필터링

- **파일**: `experiments/data/loader_skab.py`
- **변경**: `_find_skab_files(include_anomaly_free)` 파라미터 추가
- `list_all_files()`: default `include_anomaly_free=False`
- `load_one_timeseries()`: anomaly-free 제외
- **이유**: label_rate=0인 파일은 AUC-PR 정의 불가 → 집계 시 NaN/오류

### 7-D. SMD --all-files 지원

- **파일**: `scripts/multi_seed_experiment.py`
- **변경**: `_get_smd_file_count()` 추가, 메인 루프에서 SMD all-files 분기

---

## 8. 데이터 현황 상세

| 데이터셋 | 위치 | 파일 수 | 형식 | 실험 가능 파일 |
|----------|------|---------|------|--------------|
| SKAB | `LFactory_d/SKAB` | 35 CSV | valve1:16, valve2:4, other:14, anomaly-free:1 | 34 (anomaly-free 제외) |
| SMD | `LFactory_d/SMD` | 28 machines | test/train/test_label | 28 |
| AIHub71802 | `LFactory_d/manufacturing_transport_71802` | 18,533 CSV + 400 ZIP | Training/Validation | 확인 필요 (ZIP 미추출분) |
| Synthetic | 내장 생성 | 무한 | on-the-fly | 설정 자유 |

---

## 9. 추가 데이터셋 고려

서버 내 `/workspace/data/nas_kbn02_01/dataset/ar_all/`에 다른 데이터셋 존재:
- `LibraryCompany/`, `construction_vlm/`, `danger_state_origin/`, `emoknob_data/`, `mobilint_new_demo/`
- 이들의 시계열 이상치 탐지 적합성은 미확인

외부 공개 벤치마크:
- **NAB (Numenta Anomaly Benchmark)**: 58개 labeled 시계열, 자유 이용, 단순 CSV 구조 → 로더 추가 용이
- **Yahoo S5 (Webscope)**: 367개 시계열, 데이터 이용 동의서 필요
- **UCR Time Series Anomaly Archive**: 250+ 시계열, 최신 벤치마크

현재 4개 데이터셋(SKAB+SMD+AIHub+Synthetic)은 논문 제출에 충분. NAB 추가 시 5개로 경쟁력 강화 가능하나, 기존 데이터 재실행 결과를 먼저 확인 후 판단.

---

## 10. 4차 검증 (2026-03-24)

### 10-1. AIHub71802 stdout 버그 잔존 발견

- **문제**: 3차 수정(항목 E)에서 line 199의 stderr 리다이렉트만 수정. WARN 메시지 4곳(lines 153, 170, 186, 188)은 여전히 `print()` → stdout 오염
- **원인**: `import sys as _sys`가 line 199에 있어 그 이전 print에서 사용 불가했고, 3차에서 이를 간과
- **수정**: `import sys`를 파일 상단으로 이동, 4곳 모두 `file=sys.stderr` 추가, line 199의 `import sys as _sys` 제거

### 10-2. all_results_clean.json 중복 원인 및 해결

- **문제**: 1,023개 raw 결과에 92개 중복 포함
  - 원인 1: 동일 실험이 Step1 + multi_seed + RQ4 등 여러 배치에서 중복 실행되어 별도 디렉토리 생성
  - 원인 2: SKAB의 `valve1/0.csv`와 `valve2/0.csv`가 basename만으로는 구분 불가 (둘 다 `file=0.csv`)
- **해결**: `scripts/aggregate_results.py` 작성
  - Dedup key: `(dataset, detector_method, seed, meta.path)` — basename이 아닌 full path 사용
  - 같은 key에 여러 run이 있으면 가장 최신 디렉토리만 유지
  - 결과: 931 unique entries, 0 duplicates, 0 opt_f1 violations

### 10-3. RQ4 SMD/synthetic 보완

- **문제**: `run_rq4_cost_sensitivity.py`의 timeout=600초가 SMD(28K행)에 부족 → SKAB만 결과 존재
- **수정**: timeout 1800초로 증가
- **보완**: 기존 SMD/synthetic preds.csv에서 offline `cost_sensitivity_sweep()` 실행하여 결과 추가 (신규 실험 불필요)

### 10-4. 보고서 정정

- SKAB "16파일" → "34파일 (16 unique basenames across valve1/valve2/other)"
- SMD IF N=1 결과에 "통계적 신뢰도 낮음" 주석 추가
- 총 유효 runs: 1,022 → 931 (중복 제거 후)
