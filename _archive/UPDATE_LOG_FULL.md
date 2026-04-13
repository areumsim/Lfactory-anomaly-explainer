## 2025-09-11 – Loop 1 (Detect-only Minimal)

Changes
- Added initial `experiments/` package with:
  - `data_loader.py` (synthetic sine baseline with spike/step/drift anomalies)
  - `rule_detector.py` (rolling z-score detector)
  - `metrics.py` (precision/recall/f1/accuracy)
  - `main_experiment.py` (CLI: `--mode detect` only)
- Created initial `README.md` (quickstart and layout)

Notes
- Results are printed to stdout or saved via `--out-json`.

## 2025-09-11 – Loop 2 (Result Manager & ROC)

Changes
- Added `experiments/result_manager.py` to centralize persistence and plots:
  - Save predictions CSV, compute ROC curve and AUC, export ROC CSV
  - Save ROC PNG and heuristic calibration PNG if matplotlib is available
- Wired CLI options: `--out-csv`, `--plots-dir` and added `auc_roc` to metrics
- Updated `README.md` (status, quickstart) and marked `result_manager.py` done in `TODO.md`

Notes
- Calibration plot currently used min-max normalized scores (before calibration module existed).

## 2025-09-11 – Loop 3 (데이터 라우팅 & 실제 데이터 스모크)

Changes
- `experiments/data/` 데이터 레이어 확장:
  - `data_router.py`에 맞춰 `loader_smd.py`, `loader_aihub_71802.py` 추가
  - `datasets.yaml` 샘플 구성 파일 추가
- `main_experiment.py` 갱신:
  - `--dataset {synthetic,SKAB,SMD,AIHub71802}`
  - `--data-root`, `--split`, `--label-scheme`, `--sample-limit` 추가
  - 합성/실데이터 라우팅 자동 전환
- `README.md`를 한글로 전면 갱신하고, SKAB/SMD 실행 예시 추가

Notes
- AIHub71802는 초기 버전으로 센서 CSV/TXT 우선, 라벨 파일 미발견 시 0으로 대체합니다.
- SMD 라벨 길이 불일치 시 데이터 길이에 맞춰 자르거나 0 padding 처리합니다.

## 2025-09-12 – Loop 4 (ML Detector + FeatureBank 초기)

Changes
- Added `experiments/ml_detector.py`: kNN 값-공간 기반 점수 + 분위수 임계 임계값(`--ml-k`, `--ml-quantile`).
- Added `experiments/feature_bank.py`: 기본 통계/요약치(feature vector) 산출.
- Updated `experiments/result_manager.py`: `save_features_csv` 추가.
- Updated `experiments/main_experiment.py`: `--detector {rule,ml}`, `--features-csv` 옵션 및 ML 파라미터 추가.
- Updated `README.md`: ML Detector/FeatureBank 사용법 및 예시 추가.
- Updated `TODO.md`: B2/B3/B4 체크박스 갱신.

Notes
- ML Detector는 초기 버전으로 시계열의 시간구조를 고려하지 않는 값-공간 kNN 근사입니다. 후속 루프에서 IsolationForest/LSTM-AE로 대체 예정.
- FeatureBank는 초기로 시간영역 요약치만 포함(FFT/웨이블릿은 추후 추가).

## 2025-09-12 – Loop 5 (Calibration Isotonic + Cost Threshold)

Changes
- `experiments/calibration.py`: Isotonic(PAV) 보정 추가(`fit_isotonic`, `apply_isotonic`).
- `experiments/main_experiment.py`: `--calibrate isotonic` 지원, `decision` 블록에 비용 민감 임계 결과 기록.
- `experiments/cost_threshold.py`: 기대 비용 최소 임계 탐색 유틸 추가.
- `experiments/rule_detector.py`: 로버스트 Z-score(rolling median/MAD) 추가, CLI 토글(`--z-robust`).
- `README.md`: Isotonic/비용 임계/로버스트 Z-score 사용법 갱신.

Notes
- 비용행렬 입력은 간단한 CSV 문자열(`--costs c00,c01,c10,c11`)로 처리. JSON/YAML 파일 입력은 후속 루프에서 확장 가능.

 

## 2025-09-12 – Loop 6 (정규화 + 실행 메타데이터)

Changes
- Added `experiments/data/normalize.py` with Parquet write and CSV fallback:
  - `python -m experiments.data.normalize --dataset {SKAB|SMD} --data-root <ROOT>`
  - Outputs to `parquet/{skab|smd}/*.parquet` (or `.csv` if pandas unavailable)
- Updated `experiments/main_experiment.py`:
  - Added `--run-id` and recorded `run_id`, `seed`, `git_sha`, `start_ts` in results
- Updated `README.md` (한글): run metadata와 정규화 사용법 추가
- Updated `TODO.md`: 스키마 표준화/정규화(초기) 완료 체크, R1 완료 표기

Notes
- AIHub71802 정규화는 다음 루프에서 지원 예정
- Parquet은 pandas/pyarrow 의존이 없는 환경에서는 CSV로 자동 폴백합니다

## 2025-09-15 – Loop 7 (문서 보강 + 최종 실행)

Changes
- Polished `README.md` (한글): 설치/요구사항, 생성 아티팩트, 문서 섹션 추가
- Added `EXPERIMENT_REPORT.md`: 최종 스모크 실행 결과 및 해석 정리
- Ran final synthetic experiments (rule, ml) with isotonic calibration and cost optimization:
  - Outputs in `runs/final_rule/*` and `runs/final_ml/*`

Notes
- 실제 데이터셋(SKAB/SMD/AIHub71802)은 동일 CLI로 실행 가능 (DATA_ROOT 필요)

## 2025-09-15 – Loop 8 (AIHub 정규화 + Hybrid + SpecCNN)

Changes
- `experiments/data/normalize.py`: AIHub71802 정규화 추가(`--dataset AIHub71802 --split {Training|Validation} --label-scheme {binary|risk4}`)
- `experiments/hybrid_detector.py`: Rule+ML 가중합 하이브리드 탐지기 추가
- `experiments/spec_cnn.py`: SpecCNN-lite(주파수 도메인 휴리스틱) 추가
- `experiments/main_experiment.py`: `--detector {hybrid,speccnn}` 및 관련 파라미터 추가
- `README.md` / `EXPERIMENT_REPORT.md`: 사용법/결과 갱신

Notes
- SpecCNN-lite는 초기 휴리스틱으로 정밀도가 낮아 튜닝 필요(밴드/가중치)
- Hybrid는 합성 데이터에서 비용 최적화 이득을 크게 보임

## 2025-09-15 – Loop 9 (Batch 평가 + 파일 선택 + 정리 스크립트)

Changes
- Added file selection options to CLI: `--file-index`, `--min-length` (라우터/로더/CLI 연동)
- Scripts:
  - `scripts/organize_runs.py`: runs 폴더 표준화 + REPORT.md 생성
  - `scripts/archive_runs.py`: 확실한 실행 아카이브, 불명확 항목 삭제
  - `scripts/batch_eval.py`: SKAB/SMD 배치 실행 요약(`runs/reports/<UTC>/`)
- README.md: 배치 실행/파일 선택/정리 스크립트 사용법 추가 (reports 경로를 runs 내부로 변경)

Notes
- 배치 스크립트는 시연 환경에서 시간이 오래 걸릴 수 있어 요약만 생성하도록 설계됨

## 2025-09-15 – Loop 10 (공통 설정 파일 추가 + CLI 연동)

Changes
- Added `experiments/config.yaml`: 공통 기본값(데이터셋/경로/탐지기/캘리브레이션/비용/로더) 정의
- Updated `experiments/main_experiment.py`: `--config` 지원, 간단 YAML 파서(의존성 없이 동작) 추가, 설정값을 CLI 기본값으로 주입(명시된 CLI 인자가 우선)
- Updated `README.md`(한글): `config.yaml` 사용법/예시 추가
- Updated `TODO.md`: D1(datasets.yaml·config.yaml 정의) 완료 체크

Notes
- PyYAML 미설치 환경에서도 동작하도록 최소 파서를 포함했습니다. 추후 정식 YAML 의존성을 선택적으로 도입 가능
- `datasets.yaml`은 샘플 문서용이며, 데이터 로딩은 `--data-root` 또는 config의 `general.data_root`를 사용합니다

## 2025-09-15 – Loop 11 (정규화 무결성 체크 + SKAB 라벨 정책 + Rule 테스트)

Changes
- `experiments/data/normalize.py`: 변환 후 보존율(샘플 수) 경고 출력(95% 미만) 추가. SKAB/SMD/AIHub 모두 적용
- `experiments/data/loader_skab.py`: 시나리오 추론(`_infer_scenario`) 및 라벨 폴백 정책 명시(meta에 `label_policy` 기록). `anomaly-free`는 전부 0으로 매핑
- `scripts/test_rule_detector.py`: rule detector 스모크 테스트(상수열 무이상, 스파이크 탐지, 로버스트 동작) 추가
- `README.md`: 무결성 체크/라벨 정책 문서화
- `TODO.md`: D3/D4/B1 체크박스 완료 처리

Notes
- SKAB 라벨 폴백은 보수적 정책으로, 라벨 컬럼 부재 시 `anomaly-free`를 제외하고는 0으로 매핑합니다. 세부 구간 라벨은 추후 정교한 규칙/주석 기반으로 확장 예정

## 2025-09-15 – Loop 12 (공통 로더 인터페이스 + 인스펙션)

Changes
- `experiments/data/data_router.py`: `load_dataset(...)` 추가 – DataFrame/리스트 반환 + head/label 분포 로깅
- `scripts/inspect_dataset.py`: 단일 파일 스모크 인스펙션 스크립트 추가
- `README.md`: 공통 로더 API/스크립트 사용법 추가
- `TODO.md`: D2 완료 처리

Notes
- pandas 미설치 환경에서도 동작(리스트 딕셔너리로 폴백)하며, value_counts 유사 통계를 출력합니다

## 2025-09-15 – Loop 13 (Calibration/Cost 보고 자동화)

Changes
- Added `scripts/calibration_eval.py`: calibrate=[none,platt,isotonic] 실행 → ECE/AUC 요약 CSV 생성 (`runs/cal_eval.csv` 예시)
- Added `scripts/cost_ab_report.py`: 비용 임계 A/B 요약 CSV 생성 (`runs/cost_ab.csv` 예시)
- Updated `experiments/main_experiment.py`: REPORT.md A/B 섹션을 표 형태로 정리
- Updated `EXPERIMENT_REPORT.md`: 생성된 CSV 수치로 Calibration/Cost 결과 표 추가

Notes
- 스크립트는 내부적으로 `experiments.main_experiment`를 호출하므로 의존성 추가 없이 동작합니다

## 2025-09-15 – Loop 14 (평가 프로토콜 문서화 + 배치 보고)

Changes
- Updated `EVALUATION_PROTOCOL.md`: Detection Delay/Lead Time/Point-Adjust 정식 정의 추가, SMD 공식 split 규약 명시
- Added `scripts/batch_report.py`: 여러 데이터셋에 대해 calibration/cost 보고를 한 번에 생성, 타임스탬프 디렉토리에 수집
- Updated `README.md`: 배치 보고 스크립트 사용법 추가
- Updated `TODO.md`: R3 문서화 완료 표기

Notes
- Delay/Lead Time 계산 로직은 차기 루프에서 도입 예정이나, 이벤트 단위 F1은 이미 제공


## 2025-09-15 – Loop 15 (datasets.yaml 런타임 연동 + Delay/Lead 보고)

Changes
- `experiments/main_experiment.py`: `--datasets-cfg` 추가, `experiments/data/data_router.resolve_data_root`로 `--data-root` 자동 해석
- `experiments/data/data_router.py`: 단순 YAML 로더 및 경로 해석기 추가
- `experiments/main_experiment.py`: REPORT.md에 이벤트 타이밍(Events/Detected/Mean Delay/Mean Lead) 섹션 추가
- `README.md`: `--datasets-cfg` 사용법/우선순위 문서화
- `TODO.md`: D5 완료 표시, R5 상태 업데이트(평균 Delay/Lead 보고)

Notes
- `--data-root`가 명시되면 `--datasets-cfg`보다 우선합니다. YAML 경고/해석 실패 시 친절 메시지 출력

## 2025-09-16 – Loop 16 (Temperature Scaling + Ablation + Config 스냅샷)

Changes
- `experiments/calibration.py`: Temperature scaling(1-파라미터) 추가 – 표준화 점수에 대해 T 최적화
- `experiments/main_experiment.py`:
  - `--calibrate temperature` 지원, 결과 JSON에 파라미터(mu,std,T) 기록
  - 실행 시 `--config` 사용하면 run 폴더에 `config_snapshot.yaml` 사본 저장
- risk4 라벨 지원: metrics/ROC/PR/ECE 계산 시 `risk4>0→1`로 이진 변환하여 사용(meta.label_policy 기록)
- `scripts/calibration_eval.py`: 비교 목록에 `temperature` 추가
- `scripts/ablation_sweep.py`: 하이퍼파라미터 스윕 스크립트 신규 추가(Detector/alpha/z-window/ml-k/quantile)
- `scripts/ci_bootstrap.py`: preds.csv 기반 AUC-PR/ECE 95% CI 산출 스크립트 추가
- `README.md`(한글): Temperature 예시/설명 추가, 아티팩트에 config 스냅샷 명시, Ablation 사용법 섹션 추가
- `TODO.md`: C1+/R6/R7 체크박스 완료 처리

Notes
- Temperature는 1-파라미터 방식으로 경량이며, Platt/Isotonic 대비 경향 비교에 유용합니다.
- Ablation 스크립트는 실행 시간이 길 수 있으므로 스윕 범위를 소규모로 시작하는 것을 권장합니다.

## 2025-09-29 – Loop 17 (문서-코드 싱크 + 이벤트 지표 + 스모크 검증)

Changes
- `experiments/metrics.py`: 이벤트 지표에 `median_lead_time` 추가, 리드타임 계산/주석 보강
- `experiments/main_experiment.py`:
  - 비용 민감 고정 임계 계산 시 탐지기별 분위수/임계와 매핑 일치(`ml/hybrid/speccnn` → 각 분위수, `rule` → z-threshold)
  - 캘리브레이션 플롯/계산에 `labels_metric`과 `--ece-bins` 반영
- `experiments/rule_detector.py`: 임계 비교식을 `>`→`>=`로 조정(스모크 기대 반영)
- `experiments/config.yaml`: 캘리브레이션 옵션 주석에 `temperature` 반영
- 문서 정합: `EVALUATION_PROTOCOL.md`(리드타임 끝점 근사 명시), `RESULTS_POLICY.md`(index 산출물 일치)
- 스모크 실행(4 탐지기) + runs 표준화/인덱스 생성

Notes
- 스모크 배치 결과는 `runs/synthetic_20250929_*_smoke_*_*` 폴더 및 `runs/index.csv`/`index.md`에 반영되었습니다.
