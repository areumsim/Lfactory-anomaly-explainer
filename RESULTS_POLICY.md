# 실험 결과 폴더/정책 관리 (Results Policy)

본 문서는 `runs/` 산출물의 명명 규칙, 보존/아카이브 정책, 인덱스/리포트 생성 절차를 정의합니다.

## 1) 명명 규칙(Naming)
- 기본 단건 실행 폴더:
  - `runs/<dataset>_<UTCYYYYMMDD_HHMMSS>_<run_id>_<detector>/`
  - 예: `runs/SKAB_20250915_022134_real_skab_rolling_robust_zscore/`
- 포함 파일:
  - `run.json`, `preds.csv`, `plots/`(ROC/PR/Calibration), `REPORT.md`
- 배치 요약:
  - `runs/reports/<UTC>/summary.csv`, `runs/reports/<UTC>/REPORT.md`

## 2) 보존/아카이브(Retention/Archive)
- 보존 단위: (dataset, detector) 조합별 최근 N개 유지, 초과분은 아카이브
- 아카이브 경로: `runs_archive/<UTC>/...` (원 폴더명 유지)
- 불명확 항목(unknown): run.json 누락/파싱 실패/메타 없음 → 삭제
- 기본값(N): 5 (scripts/enforce_policy.py에서 설정 가능)

## 2-1) 중복 제거(Deduplication)
- 시그니처 구성: dataset, file/path, split/label_scheme, detector(method+params), seed, git_sha, num_points
- 동일 시그니처의 중복 실행은 최근 실행만 유지, 이전 실행은 아카이브로 이동
- 구현: `python scripts/enforce_policy.py` 실행 시 자동 적용

## 3) 인덱스/검색(Index)
- `runs/index.csv`, `runs/index.md`를 주기적으로 생성
- 컬럼: dataset, start_ts, run_id, detector, precision, recall, f1, accuracy, auc_roc, auc_pr, ece, fixed_cost, optimal_cost, gain, path

## 4) 실행/도구
- 폴더 표준화 + 리포트 생성: `python scripts/organize_runs.py`
- 아카이브/정리(unknown 삭제): `python scripts/archive_runs.py`
- 정책 일괄 적용(표준화+중복제거+보존+아카이브+인덱스): `python scripts/enforce_policy.py`
- 배치 요약 리포트: `DATA_ROOT=... python scripts/batch_eval.py` → `runs/reports/<UTC>/`

## 5) 권장 운영 절차
1. 실험 실행 (auto run dir + REPORT.md 생성됨)
2. 주기적으로 정책 적용
   - `python scripts/enforce_policy.py --retain 5`
3. 필요 시 배치 평가 실행 → `runs/reports/<UTC>/` 확인
4. 산출물 공유 시 `runs/`(최근) + `runs_archive/`(보관) + `runs/index.*` 제공

---

참고: 정책은 최소침습적(unknown만 삭제)이며, 파괴적 정리는 아카이브 이동을 거친 후에만 수행합니다.
