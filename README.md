# LLM-Guided Local Anomaly Detection – 실험

이 저장소는 제조 시계열 특화의 LLM-가이드드 하이브리드(ML+Rule) 이상탐지를 검증하기 위한 모듈형 연구 파이프라인입니다. 모든 변경·실험은 TODO.md(PRD)에 따라 작은 루프로 진행됩니다.

---

## 📌 프로젝트 범위 (Phase 1 vs Phase 2)

본 프로젝트는 **2단계 파이프라인**으로 구성됩니다:

### **Phase 1: Detect** (현재 구현 완료)
- **목표**: 보정(Calibration) 및 비용 민감(Cost-Sensitive) 이상탐지 파이프라인
- **구현 상태**: ✅ 완전 기능
  - 4가지 탐지기: Rule, ML (kNN), Hybrid, SpecCNN-lite
  - 3가지 보정 방법: Platt, Isotonic, Temperature Scaling
  - 비용 민감 임계값 최적화
  - 점수 기반 및 이벤트 기반 지표 (F1, AUC-PR, ECE, Detection Delay, Lead Time)
- **의존성**: Python 3.9+ 표준 라이브러리만 (matplotlib/pandas는 선택사항)

### **Phase 2: Explain + Act** (프로토타입)
- **목표**: LLM 기반 설명 생성 및 적응적 파라미터 조정
- **구현 상태**: 🔬 **프로토타입** (Week 1 - RAG-Bayes stub)
  - RAG (Retrieval-Augmented Generation): 문서 검색 기반 설명 생성
  - Bayes 사전 조정 규칙: 데이터셋 특성(불균형, SNR)에 따른 비용 행렬 권장
  - LLM 지원: OpenAI (GPT-3.5, GPT-4) + 로컬 EXAONE 모델 (온프레미스 배포)
- **사용법**:
  ```bash
  python -m experiments.main_experiment --dataset SKAB --mode detect \
    --explain --llm-provider local_exaone_35_78b \
    --calibrate temperature --cost-optimize
  ```
- **출력**: `EXPLANATIONS.md` (탐지 결과에 대한 LLM 기반 설명 및 권장사항)
- **의존성 (선택)**: `openai` (API 제공자), `transformers` + `torch` (로컬 모델)

**중요**: Phase 1 (Detect)은 LLM 없이 독립적으로 동작합니다. `--explain` 플래그 없이 실행하면 Phase 1만 사용됩니다.

---

## 📚 문서 맵 (Documentation Map)

### 시작하기
- **[docs/HANDBOOK_ko.md](docs/HANDBOOK_ko.md)** — 온보딩 & 연구 핸드북 **← 여기서 시작하세요!**

### 연구 문서
- **[TODO.md](TODO.md)** — 연구 개선 계획 (Research Improvement Plan) - 7주 로드맵
- **[TODO_REVISED.md](TODO_REVISED.md)** — Phase 2-3 LLM 통합 계획
- **[docs/LOCAL_DEFINITION_ko.md](docs/LOCAL_DEFINITION_ko.md)** — "Local" 이상탐지의 3가지 차원 정의
- **[docs/RQ_DEPENDENCIES_ko.md](docs/RQ_DEPENDENCIES_ko.md)** — 4개 연구 질문(RQ) 상호의존성 분석
- **[docs/RQ_JUSTIFICATION_ko.md](docs/RQ_JUSTIFICATION_ko.md)** — 연구 질문 정당화 및 실증 증거
- **[docs/RELATED_WORK_ko.md](docs/RELATED_WORK_ko.md)** — 관련 연구 문헌 조사 (~29편)

### 평가 및 실험
- **[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)** — 평가 프로토콜 (지표, 보정, 비용 민감 임계값, 시계열 특화 지표)
- **[COMPREHENSIVE_EXPERIMENT_REPORT.md](COMPREHENSIVE_EXPERIMENT_REPORT.md)** — 종합 실험 보고서 (480 runs)

### Phase 2 (LLM 설명)
- **[experiments/llm_config.yaml](experiments/llm_config.yaml)** — LLM 제공자 설정 (OpenAI + 로컬 EXAONE)
- **[experiments/explain_rag.py](experiments/explain_rag.py)** — RAG-Bayes 설명 모듈
- **[scripts/test_explain.py](scripts/test_explain.py)** — RAG 설명기 테스트 스크립트

---

처음 시작하시는 분은 핸드북을 먼저 읽어주세요:
- **[docs/HANDBOOK_ko.md](docs/HANDBOOK_ko.md)** — 온보딩 & 연구 핸드북

## 현재 상태 (Loop 1++)
- 합성 시계열 생성기: spike/step/drift 이상 라벨 포함
- 규칙 기반 롤링 Z-score 탐지기(투명한 베이스라인)
- 옵션: 로버스트 Z-score(rolling median/MAD) 선택 가능(`--z-robust`)
- 기본 지표: Precision, Recall, F1, Accuracy, ROC AUC, PR AUC, ECE
- 결과 매니저: 예측 CSV 저장, ROC/PR 곡선 CSV/PNG(옵션) 저장
- 캘리브레이션: min-max 정규화, Platt scaling, Temperature scaling, ECE 계산/플롯
- 캘리브레이션 확장: Isotonic(PAV) 추가 선택지
- Detect 모드 CLI: JSON 출력 및 CSV/플롯 저장 옵션
- 실제 데이터셋 라우팅(초기): SKAB, SMD, AIHub71802 1개 파일 로딩
- ML Detector(초기): 값-공간 kNN 기반 점수 + 분위수 임계값
 - Hybrid Detector: Rule+ML 스코어 가중합(분위수 임계)
 - SpecCNN-lite: 주파수 도메인 휴리스틱 기반 스코어링(초기)
- FeatureBank(초기): 기본 통계/시계열 요약치 CSV 저장 옵션
- 실행 메타데이터 기록: `run_id`, `seed`, `git_sha`, `start_ts` 결과 JSON 포함
- 데이터 정규화(초기): `experiments/data/normalize.py`로 SKAB/SMD 공통 스키마 Parquet(또는 CSV) 캐시 생성
 
## 설치 및 요구사항
- Python 3.9+ (표준 라이브러리만으로 동작)
- 선택: `matplotlib`(ROC/PR/Calibration PNG 저장), `pandas/pyarrow`(Parquet 저장)

설치 예(선택 패키지만):
```
pip install matplotlib pandas pyarrow
```

## 경로 규칙 (중요)
- CODE_ROOT: 이 저장소의 루트 경로(예: `/workspace/arsim/LFactory`). 코드와 문서가 위치합니다.
- DATA_ROOT: 실제 데이터셋의 루트 경로(예: `/workspace/data1_arsim/LFactory_d`). SKAB/SMD/AIHub 데이터가 위치합니다.
- CODE_ROOT와 DATA_ROOT는 반드시 분리되어야 합니다. 코드는 데이터가 코드 폴더와 동일 위치에 있다고 가정하지 않습니다.
- 실제 데이터셋을 사용할 때는 항상 `--data-root <DATA_ROOT>`를 명시하세요.

## 빠른 시작
Python 3.9+ 필요. 초기 루프에서는 외부 패키지 없이 동작합니다.

- 합성 데이터 탐지 실행 예:
```
python -m experiments.main_experiment --dataset synthetic --mode detect \
  --detector ml --ml-k 10 --ml-quantile 0.99 \
  --length 2000 --anomaly-rate 0.02 --noise-std 0.2 --seed 42 \
  --z-window 50 --z-threshold 3.0 --min-std 1e-3 \
  --run-id demo_synth \
  --out-json runs/run_detect.json \
  --out-csv runs/predictions.csv \
  --features-csv runs/features.csv \
  --plots-dir runs/plots \
  --calibrate temperature --ece-bins 10 \
  --cost-optimize --costs 0,1,5,0

- 하이브리드 탐지 실행 예:
```
python -m experiments.main_experiment --dataset synthetic --mode detect \
  --detector hybrid --hybrid-alpha 0.5 --hybrid-quantile 0.99 \
  --length 2000 --anomaly-rate 0.02 --noise-std 0.2 --seed 42 \
  --out-json runs/hybrid/run.json --out-csv runs/hybrid/preds.csv \
  --plots-dir runs/hybrid/plots --calibrate temperature --ece-bins 10 \
  --cost-optimize --costs 0,1,5,0
```

- SpecCNN-lite 실행 예:
```
python -m experiments.main_experiment --dataset synthetic --mode detect \
  --detector speccnn --sc-window 128 --sc-hop 16 --sc-quantile 0.99 \
  --length 2000 --anomaly-rate 0.02 --noise-std 0.2 --seed 42 \
  --out-json runs/speccnn/run.json --out-csv runs/speccnn/preds.csv \
  --plots-dir runs/speccnn/plots --calibrate temperature --ece-bins 10 \
  --cost-optimize --costs 0,1,5,0
```
```

- 경로 규칙(중요): 코드 루트와 데이터 루트는 서로 다릅니다.
  - CODE_ROOT: 이 저장소 루트(예: `/workspace/arsim/LFactory`) – 코드와 문서가 위치
  - DATA_ROOT: 실제 데이터셋 루트(예: `/workspace/data1_arsim/LFactory_d`) – SKAB/SMD/AIHub 데이터가 위치
  - 실제 데이터셋을 사용할 때는 반드시 `--data-root`로 DATA_ROOT를 명시하세요. 코드는 데이터가 코드 폴더에 함께 있다고 가정하지 않습니다.

- SKAB 스모크 실행 예(최초 1개 파일):
```
python -m experiments.main_experiment --dataset SKAB --mode detect \
  --detector rule --z-robust \
  --data-root /workspace/data1_arsim/LFactory_d --split test --seed 42 \
  --run-id skab_smoke \
  --out-json runs/skab_smoke/run.json \
  --out-csv runs/skab_smoke/preds.csv \
  --features-csv runs/skab_smoke/features.csv \
  --plots-dir runs/skab_smoke/plots
```

- SMD 스모크 실행 예(최초 1개 파일):
```
python -m experiments.main_experiment --dataset SMD --mode detect \
  --detector ml --ml-k 10 --ml-quantile 0.99 \
  --data-root /workspace/data1_arsim/LFactory_d --split test --seed 42 \
  --run-id smd_smoke \
  --out-json runs/smd_smoke/run.json \
  --out-csv runs/smd_smoke/preds.csv \
   --features-csv runs/smd_smoke/features.csv \
  --plots-dir runs/smd_smoke/plots
```

AIHub71802도 동일하게 `--dataset AIHub71802 --split Validation --label-scheme {binary|risk4}`로 동작합니다(초기 버전은 센서 시계열만, 라벨 파일 미존재 시 0으로 대체).

## 데이터 정규화(Parquet/CSV 캐시)
- 공통 스키마: `timestamp,value,label,dataset,file,machine_id`
- Parquet 우선(환경에 pandas/pyarrow 없으면 동일 경로 CSV로 폴백)
- 사용 예시:
```
python -m experiments.data.normalize --dataset SKAB --data-root /workspace/data1_arsim/LFactory_d --out-root parquet
python -m experiments.data.normalize --dataset SMD  --data-root /workspace/data1_arsim/LFactory_d --split test --out-root parquet
python -m experiments.data.normalize --dataset AIHub71802 --data-root /workspace/data1_arsim/LFactory_d --split Validation --label-scheme binary --out-root parquet
```
생성 경로 예: `parquet/skab/*.parquet` 또는 `parquet/skab/*.csv`

무결성 체크(정규화 단계): 변환 시 원본 행 대비 보존율을 계산하여 95% 미만이면 경고를 출력합니다. 숫자 파싱 실패나 결측이 많은 파일은 경고가 발생할 수 있습니다.

라벨 정책:
- SKAB: 라벨 컬럼이 없으면 시나리오 폴더명을 기준으로 매핑합니다. `anomaly-free`는 전부 0으로 매핑, 그 외 폴더는 보수적으로 0으로 폴백합니다(메타 `label_policy`에 기록).
- SMD: `test_label`을 조인하여 0/1을 사용(길이 불일치 시 자르거나 0 padding).
- AIHub71802: `--label-scheme {binary|risk4}` 모두 지원.

## 비용 민감 임계 최적화
- 옵션: `--cost-optimize`를 지정하면 스코어 기반으로 기대 비용이 최소가 되는 임계값을 탐색하여 보고합니다.
- 비용행렬 입력: `--costs c00,c01,c10,c11` (기본 `0,1,5,0`)
- 결과는 `run.json`의 `decision` 블록에 포함됩니다.
 
## 생성 아티팩트와 구조
- `run.json`: 실행 메타데이터(R1), 탐지 파라미터, 지표(Precision/Recall/F1/Accuracy/AUC-ROC/AUC-PR/ECE), 비용 임계 결과
- `preds.csv`: 시계열 포인트별 `index,value,label,score,pred[,prob]`
- `plots/roc_curve.csv|png`, `plots/pr_curve.csv|png`, `plots/calibration.png`
- 선택: `features.csv`(FeatureBank 요약치)
- `args.json`: 실행 시 사용한 CLI 인자 스냅샷(재현성)
- `config_snapshot.yaml`: `--config`로 지정한 설정 파일의 사본(선택)
- REPORT.md: 핵심 지표/비용 임계 A/B/이벤트 타이밍(Events/Detected/Delay μ/med, Lead μ/med)

## 저장소 구조
- `experiments/`
  - `data_loader.py` – 합성 데이터 로더
  - `data/` – 공개/AI Hub 데이터셋 라우터+로더
    - `data_router.py`, `loader_skab.py`, `loader_smd.py`, `loader_aihub_71802.py`, `datasets.yaml`
  - `rule_detector.py` – 롤링 Z-score 탐지기
  - `metrics.py` – 분류 지표(ROC/PR AUC 포함)
  - `calibration.py` – Platt scaling + ECE
  - `result_manager.py` – JSON/CSV/플롯 저장 유틸
  - `main_experiment.py` – Detect 모드 CLI
  - `data/normalize.py` – SKAB/SMD 공통 스키마 캐시(Parquet/CSV)
- `TODO.md` – PRD 및 작업 계획
- `UPDATE_LOG.md` – 루프별 변경 기록

## 다음 작업(요약)
- AIHub71802 정규화/라벨 스킴(`binary|risk4`) 정식 지원
- Hybrid(ML+Rule) 융합 및 비용 민감 임계 통합 평가
- 설정 파일 연동(`datasets.yaml`/config) 및 실행 기본값 정리
- 제조 특화 모델(SpecCNN 등) 시제품 및 AUC-PR 비교

## 연구 주제 추가 검토 및 권고
- Calibration 심화
  - Reliability diagram CSV/PNG와 함께 bin 별 샘플 수/오차 표 기록
  - Temperature scaling 추가 비교(선택), 데이터 분포/스코어 스케일 변화에 대한 강건성 평가
- 비용 민감/운영 KPI 연계
  - 비용행렬 민감도(0/1/5/0 등) 스윕 및 A/B 표 자동화, Alarm Burden/MTTR와의 맵핑 정식화
  - 최적 임계의 드리프트 추적(시간 구간별 비용/성능 변화)
- 이벤트/지연 지표
  - Detection Delay/Lead Time 공식에 따른 보고(평균/중앙값), Point-Adjust와 포인트 기준의 차이 비교
  - 알람 정책(선택적 알람/지연 알람)과 Delay/Lead의 트레이드오프 분석
- 모델/피처 아블레이션
  - Rule, ML, Hybrid, SpecCNN, FeatureBank 조합 비교 표; 핵심 하이퍼파라미터(alpha, window, k, quantile) 그리드 평가
- AUC-PR/Expected Cost/Alarm Burden 동시 보고
 
## Ablation/민감도 스윕 실행
핵심 하이퍼파라미터(alpha, z-window, ml-k, quantile)를 스윕하여 지표를 CSV로 요약합니다.

예시:
```
python scripts/ablation_sweep.py --dataset synthetic --out runs/ablate.csv \
  --detectors rule ml hybrid \
  --alphas 0.3 0.5 0.7 --z-windows 30 50 80 \
  --ml-ks 5 10 20 --quantiles 0.95 0.99 \
  --calibrate temperature
```

실데이터 사용 시 `--data-root`를 추가로 지정하세요.
- 데이터 품질/편향
  - 데이터셋별 라벨 비율 불균형, 비수치/결측 비율, 파일 간 길이 분포; 정규화 보존율<95% 파일 목록 관리
  - risk4 라벨과 binary 변환에 따른 정보 손실/편향 분석
- 재현성/통계적 유의성
  - 부트스트랩 95% CI(AUC-PR, ECE 등) 산출 및 표/에러바 추가
  - config 스냅샷을 run 폴더에 저장, 실험 환경(패키지 버전) 기록(선택)
- RAG-Bayes 초기 설계
  - 로컬 문서 인덱스(TF-IDF/키워드), 비용 prior 조정 규칙 설계, 근거 인용 표준화(문서 경로/구절)
  - 오프라인 모드로 재현 가능하게 시작하고, LLM API는 후속(옵션)

## 데이터셋 경로와 포맷
- DATA_ROOT 예시: `/workspace/data1_arsim/LFactory_d`

주의: 위 경로는 예시 DATA_ROOT입니다. 실제 환경 경로에 맞게 `--data-root`를 지정해야 하며, 코드(CODE_ROOT) 위치와 데이터(DATA_ROOT) 위치는 분리되어 있어야 합니다.

### 1) SKAB
- `${ROOT}/SKAB/{valve1,valve2,other,anomaly-free}/*.csv`
- 라벨: 컬럼 존재 시 사용, 없으면 0

### 2) SMD
- `${ROOT}/SMD/{train,test,test_label,interpretation_label}/*.txt`
- 라벨: `test_label` 0/1 사용(길이 불일치 시 잘라서 맞춤)

### 3) AI-Hub 71802
- `${ROOT}/manufacturing_transport_71802/{Training,Validation}/{data,label}/...`
- 포맷: JSON/CSV 혼재(초기: 센서 CSV/TXT만), 라벨은 JSON/CSV 우선
- 라벨 스킴: `binary`(>0→1) 또는 `risk4`
- 메트릭 계산 정책: `risk4` 로딩 시 지표/곡선 계산에는 `>0 → 1`로 이진 변환하여 사용(meta.label_policy=`risk4>0->1` 기록)

## 데이터셋 설정 파일 (datasets.yaml)
경로: `experiments/data/datasets.yaml` – 실제 경로에 맞게 수정해 사용하세요.

```yaml
root: "/workspace/data1_arsim/LFactory_d"

datasets:
  SKAB:
    type: skab
    path: "${root}/SKAB"
    pattern: "{subset}/{file}.csv"
    time_column: "auto"
    label: "auto"
  SMD:
    type: smd
    path: "${root}/SMD"
    train_dir: "train"
    test_dir: "test"
    label_dir: "test_label"
  AIHub71802:
    type: aihub71802
    path: "${root}/manufacturing_transport_71802"
    splits: ["Training", "Validation"]
    modalities: ["sensor"]
    label_scheme: "binary"
```

참고: `--datasets-cfg`를 지정하면 `--data-root`가 비어있을 때 YAML에서 루트를 자동 해석합니다(우선순위: `--data-root` > `--datasets-cfg`).
 
## 공통 설정 파일(config.yaml) 사용
- 경로: `experiments/config.yaml`
- 목적: 반복 실험 시 공통 기본값(데이터셋/경로/탐지기/캘리브레이션/비용/로더 옵션)을 한 곳에서 관리
- 규칙: CLI 인자가 지정되면 CLI가 설정 파일보다 우선합니다.

예시 설정:
```yaml
general:
  dataset: "synthetic"
  data_root: ""
  split: "test"
  label_scheme: "binary"
  seed: 42
  run_id: ""
output:
  out_dir: ""
  out_json: ""
  out_csv: ""
  features_csv: ""
  plots_dir: ""
detector:
  type: "rule"
  z_window: 50
  z_threshold: 3.0
  min_std: 0.001
  z_robust: false
  mad_eps: 0.000001
calibration:
  method: "none"
  ece_bins: 10
decision:
  cost_optimize: false
  apply_cost_threshold: false
  costs: "0,1,5,0"
loader:
  sample_limit: 0
  file_index: -1
  min_length: 0
```

실행 시 사용:
```
python -m experiments.main_experiment --config experiments/config.yaml \
  --dataset SKAB --data-root /workspace/data1_arsim/LFactory_d --mode detect \
  --run-id skab_conf_smoke --plots-dir runs/skab_conf/plots
```

## datasets.yaml 런타임 연동
```
python -m experiments.main_experiment --dataset SKAB --mode detect \
  --datasets-cfg experiments/data/datasets.yaml --run-id skab_auto_cfg
```
유효하지 않은 경로나 키는 경고 후 무시되며, 이 경우 `--data-root`를 명시해야 합니다.

## 데이터 로더 공통 인터페이스
- 함수: `experiments.data.data_router.load_dataset(name, data_root, split, label_scheme, ...)`
- 반환: `{data: DataFrame(또는 dict of lists), label: list/ndarray, meta: dict}`
- 로깅: 상위 5행(head)과 라벨 분포(value_counts 유사) 출력

인스펙션 스크립트 사용 예:
```
python scripts/inspect_dataset.py --dataset SKAB --data-root /workspace/data1_arsim/LFactory_d --split test
```
출력 예:
- head(5) 테이블 또는 튜플 리스트
- `[loader] label counts: {0: N0, 1: N1}`
 - `[meta] {dataset, file, label_rate, ...}`

## 실험 스크립트 (도움 도구)
- Calibration 비교: 방법별 ECE/AUC 요약 CSV 생성
```
python scripts/calibration_eval.py --dataset synthetic --out runs/cal_eval.csv
python scripts/calibration_eval.py --dataset SKAB --data-root /workspace/data1_arsim/LFactory_d --out runs/skab_cal_eval.csv
```

- 비용 임계 A/B 요약: 기대 비용/정확도/F1 비교 CSV 생성
```
python scripts/cost_ab_report.py --dataset synthetic --detector ml --out runs/cost_ab.csv
python scripts/cost_ab_report.py --dataset SKAB --data-root /workspace/data1_arsim/LFactory_d --out runs/skab_cost_ab.csv
```

- 배치 보고: 여러 데이터셋에 대해 위 두 보고서를 한 번에 생성
```
python scripts/batch_report.py --datasets synthetic --out-dir runs/reports
python scripts/batch_report.py --datasets SKAB SMD --data-root /workspace/data1_arsim/LFactory_d --out-dir runs/reports
```

- Bootstrap CI (AUC-PR, ECE): 예측 CSV 기반 95% CI 산출
```
python scripts/ci_bootstrap.py --preds runs/tmp_preds.csv --n 1000 --out runs/ci.csv
```
설명: `preds.csv`에 `prob` 컬럼이 있으면 이를 사용, 없으면 score의 min-max 정규화를 사용하여 ECE를 계산합니다.

## 문서
- 설계/로드맵: `TODO.md` (PRD) – 항상 최신화
- 변경 이력: `UPDATE_LOG.md` – 루프별 변경 사항 기록
- 최종/중간 실험 보고: `EXPERIMENT_REPORT.md` – 본문 및 결과 스냅샷

--------------------------------------------------------------------------------
# 상세 문서: 주제 / 목표 / 데이터셋 / 구조 / 실행 / 지표 / 산출물 / 특징

## 1) 주제(Problem)
- 제조 시계열 이상탐지에서 Rule과 ML을 결합한 Hybrid 접근으로 AUC-PR 개선과 운영 비용(오탐/미탐 비용) 최소화를 동시에 추구합니다.
- Detect–Explain–Act 파이프라인 중 Detect를 중심으로 Calibration(ECE)과 비용 민감 임계 최적화를 포함합니다.

## 2) 실험 목표(Objectives)
- Hybrid(ML+Rule) 성능 검증: AUC-PR/ROC 및 비용 최적화(Expected Cost) 비교
- Calibration(Platt/Isotonic) 적용 및 temporal split ECE 평가 (수정 후 ECE ≈ 0.19)
- 재현성: seed 고정, run_id/git_sha/start_ts 자동 기록

## 3) 데이터셋(Datasets)
- DATA_ROOT 예: `/workspace/data1_arsim/LFactory_d` (코드 루트와 분리)
- 경로 구조
  - SKAB: `${ROOT}/SKAB/{valve1,valve2,other,anomaly-free}/*.csv`
  - SMD: `${ROOT}/SMD/{train,test,test_label,interpretation_label}/*.txt`
  - AIHub71802: `${ROOT}/manufacturing_transport_71802/{Training,Validation}/{data,label}/...`
- 로더/라우터: `experiments/data/*.py` (의존성 최소, 표준 라이브러리로 동작)
- 정규화(Parquet/CSV 캐시): 공통 스키마 `timestamp,value,label,dataset,file,machine_id`
  - `python -m experiments.data.normalize --dataset {SKAB|SMD|AIHub71802} --data-root <ROOT> [--split ..] [--label-scheme binary|risk4] --out-root parquet`

## 4) 코드 구조(Codebase)
- 탐지기
  - `rule_detector.py`: 롤링 Z-score, 옵션 Robust(Median/MAD)
  - `ml_detector.py`: 값-공간 kNN 점수 + 분위수 임계
  - `hybrid_detector.py`: Rule/ML 정규화 점수의 가중합 + 분위수 임계
  - `spec_cnn.py`: SpecCNN-lite(주파수 밴드 에너지 기반 휴리스틱)
- 파이프라인/지표
  - `main_experiment.py`: Detect 모드 실행/저장/보고서 자동화
  - `metrics.py`: Precision/Recall/F1/Accuracy, AUC-ROC/PR
  - `calibration.py`: Platt/Isotonic(PAV), ECE 계산
  - `cost_threshold.py`: 비용행렬 기반 최적 임계 탐색
  - `result_manager.py`: CSV/JSON/플롯 저장(의존성 없을 시 CSV만)
- 데이터 계층
  - `data_loader.py`(합성), `data/loader_*.py`(SKAB/SMD/AIHub), `data/normalize.py`

## 5) 실행(Commands)
- 공통
  - `--dataset {synthetic|SKAB|SMD|AIHub71802}` (합성 외에는 `--data-root` 필수)
  - `--run-id` (폴더명 및 메타 기록), `--out-dir`(옵션)
  - `--calibrate {none|platt|isotonic}`, `--cost-optimize --costs c00,c01,c10,c11`
  - 파일 선택: `--file-index N`(Nth 파일 선택), `--min-length K`(길이 K 이상 후보 우선)
- 탐지기 파라미터
  - Rule: `--z-window --z-threshold --min-std --z-robust --mad-eps`
  - ML: `--ml-k --ml-quantile`
  - Hybrid: `--hybrid-alpha --hybrid-quantile`
  - SpecCNN: `--sc-window --sc-hop --sc-quantile`
- 예시(합성 + Hybrid, 자동 폴더/REPORT.md 생성)
```
python -m experiments.main_experiment --dataset synthetic --mode detect \
  --detector hybrid --length 3000 --anomaly-rate 0.02 --noise-std 0.2 --seed 42 \
  --hybrid-alpha 0.5 --hybrid-quantile 0.99 --run-id demo \
  --calibrate isotonic --ece-bins 10 --cost-optimize --costs 0,1,5,0
```

## 6) 지표(Metrics)
- Precision, Recall, F1, Accuracy (point-wise)
- AUC-ROC/PR: 점수 정렬 기반 내부 구현(의존성 없음)
- ECE: 확률 구간(bin)별 예측-관측 편차의 가중 평균
- Expected Cost: `TN*C00 + FP*C01 + FN*C10 + TP*C11` / N — 임계 후보 중 최소 선택
  - 적용 모드: `--cost-optimize --apply-cost-threshold`를 함께 사용하면 최적 임계에서 예측/지표를 재계산하고 `preds_cost_opt.csv` 저장, REPORT.md에 A/B 비교 표기
  - 상세 평가지침: `EVALUATION_PROTOCOL.md` 참고
 
## 참고 문서
- 평가 프로토콜: `EVALUATION_PROTOCOL.md`
- 결과 폴더/정책: `RESULTS_POLICY.md`

## 7) 산출물(Artifacts)
- 자동 폴더: `runs/<dataset>_<UTCYYYYMMDD_HHMMSS>_<run_id>/`
- 파일
  - `run.json`: 메타(run_id, seed, git_sha, start_ts), 탐지 파라미터, 지표, 비용 임계 결과
  - `preds.csv`: `index,value,label,score,pred[,prob]`
  - `plots/roc_curve.csv|png`, `plots/pr_curve.csv|png`, `plots/calibration.png`
  - `REPORT.md`: 사람이 읽기 쉬운 실행/분석 요약(자동 생성)

### 배치 실행 자동화(요약 리포트)
- 스크립트: `scripts/batch_eval.py`
  - SKAB(robust rule), SMD(ML kNN)를 다중 파일에 대해 실행해 `runs/reports/<UTC>/`에 `summary.csv`와 `REPORT.md`를 생성
  - 예: `DATA_ROOT=/workspace/data1_arsim/LFactory_d python scripts/batch_eval.py`

### runs 폴더 정리/아카이브
- 정돈: `python scripts/organize_runs.py` → `runs/<dataset>_<UTC>_<run_id>_<detector>/`로 표준화 + REPORT.md 생성
- 아카이브/삭제 정책: `python scripts/archive_runs.py` → 확실한 실행은 `runs_archive/<UTC>/`로 이동, 불명확 항목만 삭제

## 8) 특징(Highlights)
- 의존성 최소(순수 파이썬) + 선택 패키지(`matplotlib`, `pandas/pyarrow`)
- Hybrid/Calibration/Cost-optimization 내장
- 데이터 정규화 CLI 제공(공통 스키마)
- 재현 메타데이터 자동 기록(run_id/seed/git_sha/start_ts)

--------------------------------------------------------------------------------
## 보고서 확인 방법
- 각 실행 폴더의 `REPORT.md`에서 한 번에 핵심 결과를 확인할 수 있습니다.
- 예: `runs/synthetic_20250915_015726_final_hybrid/REPORT.md`

## 정리 옵션
- 과거 산출물을 정리하고 위 규칙으로만 남기길 원하시면 요청해 주세요.
- 자동 폴더 규칙으로 4개 실행(rule/ml/hybrid/speccnn)을 재실행하여 동일 구조로 리프레시하고,
  불필요한 이전 산출물은 삭제 또는 별도 폴더로 보관 처리할 수 있습니다.
