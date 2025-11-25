# 실험 보고서 (최신 스냅샷)

본 문서는 현재 코드 기준 최종 스모크/데모 실행 결과를 요약합니다. 데이터 접근 제약으로 공개/AIHub 실제 데이터 대신 합성(synthetic) 데이터로 최종 실행을 재현했습니다. 동일 설정으로 외부 데이터셋(SM D/SKAB/AIHub71802)을 실행할 수 있습니다.

## 1) 실행 환경 및 설정
- Python 3.9+
- 선택 패키지: matplotlib(설치됨), pandas/pyarrow(선택, 본 실행은 CSV만 사용)
- 코드 루트: `CODE_ROOT = /workspace/arsim/LFactory`
- 데이터 루트: `DATA_ROOT = /workspace/data1_arsim/LFactory_d` (본 실행에선 미사용)

공통 옵션
- 캘리브레이션: `--calibrate isotonic` (PAV)
- 비용 민감 임계: `--cost-optimize --costs 0,1,5,0`
- 메타데이터 기록: `run_id, seed, git_sha, start_ts`

## 2) 최종 실행 커맨드

규칙 기반(Rule, Robust Z-score 비활성; 표준 Z-score)
```
python -m experiments.main_experiment --dataset synthetic --mode detect \
  --detector rule --length 300 --seed 123 --run-id final_rule \
  --out-json runs/final_rule/run.json \
  --out-csv runs/final_rule/preds.csv \
  --plots-dir runs/final_rule/plots \
  --calibrate isotonic --ece-bins 10 \
  --cost-optimize --costs 0,1,5,0
```

ML 기반(kNN 값-공간 점수)
```
python -m experiments.main_experiment --dataset synthetic --mode detect \
  --detector ml --ml-k 10 --ml-quantile 0.99 \
  --length 300 --seed 123 --run-id final_ml \
  --out-json runs/final_ml/run.json \
  --out-csv runs/final_ml/preds.csv \
  --plots-dir runs/final_ml/plots \
  --calibrate isotonic --ece-bins 10 \
  --cost-optimize --costs 0,1,5,0
```

주: 합성 길이 300은 스모크/데모용이며, 연구 목적의 본 실행에서는 길이 2000~3000, 이상률 2~3% 수준을 권장합니다.

Hybrid (Rule+ML 가중합)
```
python -m experiments.main_experiment --dataset synthetic --mode detect \
  --detector hybrid --hybrid-alpha 0.5 --hybrid-quantile 0.99 \
  --length 300 --seed 123 --run-id final_hybrid \
  --out-json runs/final_hybrid/run.json \
  --out-csv runs/final_hybrid/preds.csv \
  --plots-dir runs/final_hybrid/plots \
  --calibrate isotonic --ece-bins 10 \
  --cost-optimize --costs 0,1,5,0
```

SpecCNN-lite (주파수 도메인 휴리스틱)
```
python -m experiments.main_experiment --dataset synthetic --mode detect \
  --detector speccnn --sc-window 128 --sc-hop 16 --sc-quantile 0.99 \
  --length 300 --seed 123 --run-id final_speccnn \
  --out-json runs/final_speccnn/run.json \
  --out-csv runs/final_speccnn/preds.csv \
  --plots-dir runs/final_speccnn/plots \
  --calibrate isotonic --ece-bins 10 \
  --cost-optimize --costs 0,1,5,0
```

## 3) 결과 요약(스모크)

규칙 기반(rule, length=300, seed=123) – 실제 산출물에서 발췌
- 지표: precision=1.00, recall=0.20, f1=0.333, accuracy=0.960
- AUC-ROC=0.886, AUC-PR=0.399, ECE≈0.0
- 비용 임계 최적화: fixed_cost=0.20 → optimal_cost=0.16 (gain=0.04)
- 아티팩트: `runs/final_rule/{run.json,preds.csv,plots/*}`

ML 기반(ml, length=300, seed=123) – 실제 산출물에서 발췌
- 지표: precision=0.667, recall=0.133, f1=0.222, accuracy=0.953
- AUC-ROC=0.997, AUC-PR=0.780, ECE≈0.0
- 비용 임계 최적화: fixed_cost≈0.203 → optimal_cost≈0.0033 (gain≈0.20)
- 아티팩트: `runs/final_ml/{run.json,preds.csv,plots/*}`

Hybrid (length=300, seed=123) – 실제 산출물에서 발췌
- 지표: precision 1.000, recall 0.200, f1 0.333, accuracy 0.960
- AUC-ROC 0.998, AUC-PR 0.849, ECE≈0.0
- 비용 임계 최적화: fixed ≈0.183 → optimal ≈0.0033 (gain ≈0.18)
- 아티팩트: `runs/final_hybrid/{run.json,preds.csv,plots/*}`

SpecCNN-lite (length=300, seed=123) – 실제 산출물에서 발췌
- 지표: precision 0.050, recall 1.000, f1 0.095, accuracy 0.050
- AUC-ROC 0.5, AUC-PR 0.0, ECE≈0.05 (초기 휴리스틱: 재학습/튜닝 필요)
- 비용 임계 최적화: fixed 0.95 → optimal 0.25 (gain 0.70)
- 아티팩트: `runs/final_speccnn/{run.json,preds.csv,plots/*}`

샘플 파일 구조(규칙 기반)
- `runs/final_rule/run.json`
- `runs/final_rule/preds.csv`
- `runs/final_rule/plots/roc_curve.csv|png`, `pr_curve.csv|png`, `calibration.png`

## 4) 해석 및 권고
- 합성 데이터에서 규칙 기반은 높은 정밀도(Precision 1.0)와 낮은 재현율(Recall 0.2)을 보임 → 알람 부담은 낮지만 미검출이 존재.
- 비용 민감 임계 최적화는 False Negative 비용이 높은 가정에서 고정 임계 대비 기대 비용을 추가로 낮춤.
- ML(kNN) 탐지기는 시나리오에 따라 AUC-PR 개선 여지가 있으며, 실제 제조 데이터에서는 IsolationForest/LSTM-AE 대체로 개선 기대.
- Calibration(Isotonic) 적용으로 신뢰도-정확도 정렬이 강화되며, ECE가 낮게 유지됨.
- Hybrid는 ML의 AUC-PR 이점을 유지하면서 Rule의 보수적 특성을 결합해 비용 최적화 이득을 제공.
- SpecCNN-lite는 휴리스틱 초기형으로 재현율은 높으나 과다 탐지 경향(정밀도 낮음) — 밴드/가중치 튜닝 및 주파수 특징 확대 필요.

## 5) 재현 절차 체크리스트
- 합성: 위 커맨드 그대로 실행 → 동일 seed에서 유사 지표 예상(AUC 차이 ±1e-6 수준)
- SKAB/SMD/AIHub71802: 데이터 루트를 지정하고 `--dataset`만 변경하여 동일 파이프라인 실행
- 정규화 캐시 필요 시: `python -m experiments.data.normalize --dataset {SKAB|SMD} --data-root <ROOT>`

## 6) 다음 단계 제안
- AIHub71802 정규화/라벨 스킴(risk4/binary) 지원 추가
- Rule+ML 하이브리드 스코어 융합 및 Cost-aware 최적화 통합
- 제조 특화 모델(주파수-도메인 SpecCNN) 프로토타입 추가 및 AUC-PR 비교
 
## 7) 추가 결과 요약 (자동 스크립트 기반)

### 7.1 Calibration 비교 (합성 데이터)
- 생성 CSV: `runs/cal_eval.csv`

| method   |   F1   | Accuracy | AUC-ROC | AUC-PR |   ECE   |
|---|---:|---:|---:|---:|---:|
| none     | 0.177 | 0.9675 | 0.5629 | 0.1781 | 0.3029 |
| platt    | 0.177 | 0.9675 | 0.5629 | 0.1781 | 0.0451 |
| isotonic | 0.177 | 0.9675 | 0.5629 | 0.1781 | 0.0000 |

메모: 본 스모크에서는 Platt/Isotonic 적용 시 ECE가 크게 감소(0.303 → 0.045/0.000). 점수 분포가 동일해 AUC는 동일합니다.

### 7.2 비용 임계 A/B (합성, ML 탐지기)
- 생성 CSV: `runs/cost_ab.csv`

| Metric        | Original | Cost-Opt |
|---|---:|---:|
| Expected Cost | 0.1340   | 0.1205   |
| F1            | 0.3736   | 0.4752   |
| Accuracy      | 0.9715   | 0.9735   |

메모: 비용 최적 임계 적용으로 기대 비용이 추가로 감소하고, F1/Accuracy가 함께 개선되었습니다.

### 7.3 Synthetic 스모크 배치 (4 탐지기)
- 실행일시: UTC 2025-09-29
- 공통 설정: `length=500`, `seed=42`, `calibrate=isotonic`, `ece_bins=10`, `cost_optimize+apply`
- 산출 폴더 표준: `runs/synthetic_YYYYMMDD_HHMM_<run_id>_<detector>/`

| Run ID | Detector | F1 | AUC-PR | ECE | Expected Cost (fixed→opt) | Folder |
|---|---|---:|---:|---:|---|---|
| smoke_rule_robust | rolling_robust_zscore | 0.000 | 0.026 | 0.000 | 0.208 → 0.160 | runs/synthetic_20250929_0234_smoke_rule_robust_rolling_robust_zscore |
| smoke_ml | knn_value_density | 0.476 | 0.576 | 0.000 | 0.100 → 0.060 | runs/synthetic_20250929_0234_smoke_ml_knn_value_density |
| smoke_hybrid | hybrid_rule_ml | 0.286 | 0.352 | 0.000 | 0.136 → 0.086 | runs/synthetic_20250929_0235_smoke_hybrid_hybrid_rule_ml |
| smoke_speccnn | speccnn_lite | 0.062 | 0.000 | 0.032 | 0.968 → 0.160 | runs/synthetic_20250929_0236_smoke_speccnn_speccnn_lite |

메모:
- ML은 AUC-PR과 비용 측면에서 우위. Hybrid는 조합 효과로 비용 이득을 확보.
- Robust Rule은 보수적 베이스라인으로 비용은 개선되나 F1은 낮음.
- SpecCNN-lite는 초기 휴리스틱으로 과다 탐지 경향(정밀도 낮음) — 밴드/가중치 튜닝 필요.

재현 명령(각 1줄):
- Rule(robust): `PYTHONPATH=. python -m experiments.main_experiment --dataset synthetic --mode detect --detector rule --z-robust --length 500 --seed 42 --run-id smoke_rule_robust --calibrate isotonic --ece-bins 10 --cost-optimize --costs 0,1,5,0 --apply-cost-threshold`
- ML: `PYTHONPATH=. python -m experiments.main_experiment --dataset synthetic --mode detect --detector ml --length 500 --seed 42 --run-id smoke_ml --calibrate isotonic --ece-bins 10 --cost-optimize --costs 0,1,5,0 --apply-cost-threshold`
- Hybrid: `PYTHONPATH=. python -m experiments.main_experiment --dataset synthetic --mode detect --detector hybrid --hybrid-alpha 0.5 --length 500 --seed 42 --run-id smoke_hybrid --calibrate isotonic --ece-bins 10 --cost-optimize --costs 0,1,5,0 --apply-cost-threshold`
- SpecCNN-lite: `PYTHONPATH=. python -m experiments.main_experiment --dataset synthetic --mode detect --detector speccnn --sc-window 128 --sc-hop 16 --length 500 --seed 42 --run-id smoke_speccnn --calibrate isotonic --ece-bins 10 --cost-optimize --costs 0,1,5,0 --apply-cost-threshold`
