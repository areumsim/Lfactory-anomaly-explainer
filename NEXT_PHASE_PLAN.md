# LFactory 다음 단계 종합 계획서
## Next Phase Comprehensive Plan

**작성일**: 2025-11-24
**현재 상태**: Phase 1 (Detect) 100% 완료
**목표**: Phase 1 고도화 + Phase 2 (Explain) 완전 구현

---

# 📊 현재 완료 상태

## ✅ Phase 1: Detect (100% 완료)

### 완료된 작업
- [x] 6가지 detector 구현 (Rule, kNN, IF, LSTM-AE, Hybrid, SpecCNN)
- [x] 480 runs 다중 시드 실험 (96% 성공률)
- [x] 통계적 검증 (Wilcoxon, Bootstrap CI)
- [x] 상관관계 분석 (Point-wise vs Event-wise)
- [x] 종합 실험 보고서 (35페이지)
- [x] Calibration (Platt, Isotonic, Temperature)
- [x] Cost-sensitive threshold optimization

### 발견된 개선 영역
- [ ] SpecCNN weight 최적화 (현재 AUC-PR=0)
- [ ] 하이퍼파라미터 전체 grid search
- [ ] 다중 파일 평가 (SKAB 30개 파일)
- [ ] AIHub71802 데이터 문제 해결
- [ ] 실시간 배포 시스템 구축

---

# 🎯 다음 단계 전체 로드맵

## Phase 1.5: Detect 고도화 (2-3주)

### Week 1: 비교 분석 및 시각화
- [ ] 알고리즘 비교 보고서 작성
- [ ] 시각화 대시보드 생성
- [ ] 논문 제출용 figure 생성

### Week 2: 최적화 및 검증
- [ ] SpecCNN weight grid search
- [ ] 하이퍼파라미터 전체 최적화
- [ ] 다중 파일 SKAB 평가

### Week 3: 실무 적용
- [ ] 실시간 모니터링 시스템 프로토타입
- [ ] 배포 가이드 작성
- [ ] 사용자 매뉴얼 작성

## Phase 2: Explain (3-4주)

### Week 1: LLM 통합
- [ ] OpenAI API 또는 Local EXAONE 설정
- [ ] RAG (Retrieval-Augmented Generation) 구현
- [ ] Prompt engineering

### Week 2: 설명 생성
- [ ] Anomaly 설명 템플릿 작성
- [ ] 도메인 지식 베이스 구축
- [ ] 설명 품질 평가

### Week 3: Bayesian Prior
- [ ] Cost matrix 동적 조정
- [ ] 사용자 피드백 학습
- [ ] A/B 테스트

### Week 4: 통합 및 검증
- [ ] End-to-end 파이프라인 통합
- [ ] 사용자 테스트
- [ ] 최종 보고서

---

# 📋 Stage 7-12 상세 계획

---

## Stage 7: 알고리즘 비교 보고서 작성 (2-3일)

### 목표
실험 결과를 바탕으로 **학술 논문급 비교 분석 보고서** 작성

### 작업 내역

#### 7.1 성능 비교 시각화 생성
**출력물**: `figures/` 디렉토리에 publication-quality figures

1. **알고리즘 성능 비교 차트**
   - Grouped bar chart: F1, AUC-PR by detector
   - Box plot: Performance distribution across seeds
   - Radar chart: Multi-metric comparison (F1, AUC-PR, Precision, Recall, ECE)

2. **통계적 유의성 히트맵**
   - Wilcoxon p-value heatmap (6×6 detector pairs)
   - 색상 코드: p<0.001 (dark green), p<0.05 (light green), p≥0.05 (red)

3. **신뢰구간 비교**
   - Error bar plot: Mean ± 95% CI for each detector
   - Dataset별 subplot (SKAB, SMD, Synthetic)

4. **ROC/PR 곡선 비교**
   - Overlay all 6 detectors on same plot
   - Dataset별 subplot

5. **Calibration 비교**
   - Reliability diagram (predicted prob vs actual freq)
   - 6 detectors × 3 datasets = 18 subplots

6. **Event-wise 성능 분석**
   - Detection delay distribution (box plot)
   - Lead time vs F1 scatter plot
   - Event recall vs Point recall correlation

7. **Cost-sensitive 분석**
   - Cost reduction by detector (bar chart)
   - Optimal threshold distribution (histogram)

**구현 방법**:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("runs/all_results.csv")

# Figure 1: Performance comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, dataset in enumerate(["SKAB", "SMD", "synthetic"]):
    data = df[df["dataset"] == dataset]
    data.groupby("detector")["auc_pr"].mean().plot(kind="bar", ax=axes[i])
    axes[i].set_title(f"{dataset} - AUC-PR by Detector")
plt.savefig("figures/performance_comparison.png", dpi=300)
```

#### 7.2 데이터셋별 상세 분석
**출력물**: `ALGORITHM_COMPARISON_REPORT.md`

각 데이터셋별로:
1. **Winner 선정 및 근거**
   - Best F1, Best AUC-PR, Best stability (낮은 std)
   - 통계적 유의성 확인

2. **알고리즘 특성 분석**
   - 왜 특정 알고리즘이 잘 작동했는가?
   - 데이터셋 특성과의 관계

3. **실무 권장사항**
   - 이 데이터셋과 유사한 경우 어떤 알고리즘?
   - 하이퍼파라미터 설정 가이드

#### 7.3 크로스 데이터셋 일반화 분석

**질문**:
- 모든 데이터셋에서 우수한 알고리즘이 있는가?
- 데이터셋 특성에 따라 알고리즘 선택이 달라지는가?

**분석**:
1. **Ranking consistency**
   - Kendall's Tau correlation between dataset rankings
   - Example: SKAB에서 1등인 LSTM-AE가 SMD에서도 top-3에 드는가?

2. **데이터셋 특성 vs 알고리즘 성능**
   - Anomaly rate vs 최적 알고리즘
   - Series length vs 최적 알고리즘
   - Number of features vs 최적 알고리즘

3. **Meta-learning 가능성**
   - 데이터셋 특성만으로 최적 알고리즘 예측 가능?

#### 7.4 Ablation Study 확장

**현재**: SKAB에서만 기본 파라미터 테스트
**목표**: 전체 grid search 수행

**Grid Search 설계**:
```python
# Full factorial design
hyperparameters = {
    "Rule": {
        "z_window": [30, 50, 80, 100],
        "threshold": [2.5, 3.0, 3.5],
        "robust": [True, False]
    },
    "kNN": {
        "k": [5, 10, 15, 20],
        "quantile": [0.95, 0.97, 0.99]
    },
    "IsolationForest": {
        "window": [30, 50, 80],
        "contamination": [0.05, 0.1, 0.15],
        "n_estimators": [50, 100, 200]
    },
    "LSTM-AE": {
        "seq_len": [30, 50, 80],
        "latent_dim": [16, 32, 64],
        "epochs": [30, 50],
        "lr": [0.0005, 0.001, 0.002]
    },
    "Hybrid": {
        "alpha": [0.3, 0.5, 0.7, 0.9],
        "rule_window": [30, 50, 80],
        "ml_k": [5, 10, 20]
    }
}
```

**Total combinations**:
- Rule: 4×3×2 = 24
- kNN: 4×3 = 12
- IsolationForest: 3×3×3 = 27
- LSTM-AE: 3×3×2×3 = 54 (expensive!)
- Hybrid: 4×3×3 = 36
- **Total**: 153 runs per dataset

**Optimization**:
- Random search: 20% of full grid (30 runs per detector)
- Bayesian optimization: TPE (Tree-structured Parzen Estimator)

#### 7.5 논문 작성

**목표**: 국제 학술지 투고 수준 논문

**구조**:
1. **Title**: "Comprehensive Evaluation of Time Series Anomaly Detection: A Multi-Algorithm, Multi-Dataset Study with Statistical Validation"

2. **Abstract** (250 words)
   - Context, gap, method, results, conclusion

3. **Introduction**
   - Problem statement
   - Research questions (RQ1-4)
   - Contributions

4. **Related Work**
   - Rule-based methods
   - ML methods
   - Deep learning methods
   - Comparison studies

5. **Methodology**
   - Datasets (4)
   - Algorithms (6)
   - Evaluation protocol (multi-seed, metrics)
   - Statistical validation

6. **Results**
   - Performance comparison (Table + Figure)
   - Statistical tests (Wilcoxon, Bootstrap CI)
   - Correlation analysis

7. **Discussion**
   - RQ answers
   - Algorithm selection guide
   - Limitations

8. **Conclusion**
   - Summary
   - Future work

**Target Journals**:
- IEEE Transactions on Knowledge and Data Engineering (TKDE)
- ACM Transactions on Knowledge Discovery from Data (TKDD)
- Data Mining and Knowledge Discovery (DMKD)

**예상 작업 시간**: 3-5일

---

## Stage 8: SpecCNN Weight 최적화 (1-2일)

### 목표
SpecCNN의 AUC-PR=0 문제 해결 → 실용적인 frequency-domain detector로 개선

### 8.1 Grid Search 구현

**현재 문제**:
- Heuristic weights: w_low=-0.2, w_mid=0.6, w_high=0.6
- 결과: F1은 높지만 AUC-PR=0 (ranking 완전 실패)

**해결 방법**: Grid search로 최적 weights 찾기

**구현**:
```python
# scripts/speccnn_grid_search.py 개선

import itertools
import json
from pathlib import Path

def grid_search_speccnn_weights(dataset="SKAB", data_root=""):
    """Grid search for optimal SpecCNN frequency band weights."""

    # Define search space
    w_low_range = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5]
    w_mid_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    w_high_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    best_auc_pr = 0.0
    best_weights = None
    results = []

    total_combinations = len(w_low_range) * len(w_mid_range) * len(w_high_range)
    print(f"Total combinations: {total_combinations}")

    for i, (w_l, w_m, w_h) in enumerate(itertools.product(w_low_range, w_mid_range, w_high_range)):
        print(f"[{i+1}/{total_combinations}] Testing: w_low={w_l}, w_mid={w_m}, w_high={w_h}")

        # Run SpecCNN with these weights
        cmd = [
            "python3", "-m", "experiments.main_experiment",
            "--dataset", dataset,
            "--detector", "speccnn",
            "--seed", "42",
            "--sc-weights", str(w_l), str(w_m), str(w_h),
            "--run-id", f"speccnn_grid_{i}",
            "--out-json", f"runs/speccnn_grid_{i}.json"
        ]

        if dataset != "synthetic":
            cmd.extend(["--data-root", data_root])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Load result
            with open(f"runs/speccnn_grid_{i}.json") as f:
                data = json.load(f)
                auc_pr = data["metrics"]["auc_pr"]
                f1 = data["metrics"]["f1"]

                results.append({
                    "w_low": w_l,
                    "w_mid": w_m,
                    "w_high": w_h,
                    "auc_pr": auc_pr,
                    "f1": f1,
                })

                if auc_pr > best_auc_pr:
                    best_auc_pr = auc_pr
                    best_weights = (w_l, w_m, w_h)
                    print(f"  ✨ New best! AUC-PR={auc_pr:.4f}, F1={f1:.4f}")
        else:
            print(f"  ❌ Failed")

    # Save results
    output = {
        "dataset": dataset,
        "best_weights": {"low": best_weights[0], "mid": best_weights[1], "high": best_weights[2]},
        "best_auc_pr": best_auc_pr,
        "all_results": results
    }

    with open(f"runs/speccnn_grid_search_{dataset}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Grid search complete!")
    print(f"📊 Best weights: low={best_weights[0]}, mid={best_weights[1]}, high={best_weights[2]}")
    print(f"📈 Best AUC-PR: {best_auc_pr:.4f}")

    return best_weights, best_auc_pr
```

**실행**:
```bash
python3 scripts/speccnn_grid_search.py --dataset SKAB --data-root /workspace/data1/arsim/LFactory_d
python3 scripts/speccnn_grid_search.py --dataset SMD --data-root /workspace/data1/arsim/LFactory_d
python3 scripts/speccnn_grid_search.py --dataset synthetic
```

**예상 시간**: 6×6×6 = 216 combinations × 5초 = 18분 per dataset

### 8.2 최적 Weights로 재실험

최적 weights 발견 후:
1. 20-seed 재실험
2. 성능 비교: Before (AUC-PR=0) vs After
3. COMPREHENSIVE_REPORT 업데이트

### 8.3 Frequency-Domain 분석

**추가 분석**:
1. **Anomaly의 주파수 특성**
   - Normal vs Anomaly의 STFT 차이 시각화
   - 어떤 주파수 대역이 anomaly를 가장 잘 구분하는가?

2. **Adaptive Band Selection**
   - 데이터셋마다 다른 band 사용
   - 학습 기반 band weight 선택

3. **SpecCNN vs LSTM-AE 비교**
   - Frequency domain vs Time domain
   - 언제 SpecCNN이 유리한가?

---

## Stage 9: 하이퍼파라미터 전체 최적화 (3-5일)

### 목표
각 detector의 최적 hyperparameter 발견 → 성능 상한선 측정

### 9.1 Bayesian Optimization 구현

**Random Search보다 효율적**:
- 이전 결과를 활용하여 다음 시도 결정
- 전체 grid의 10-20%만 탐색으로 최적값 근사

**구현**:
```python
# scripts/bayesian_optimization.py

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

def bayesian_optimize(detector, dataset, n_iterations=30):
    """Bayesian optimization for hyperparameter tuning."""

    # Define search space
    if detector == "isolation_forest":
        space = {
            "window": (20, 100, "int"),
            "contamination": (0.01, 0.2, "float"),
            "n_estimators": (50, 200, "int")
        }
    elif detector == "lstm_ae":
        space = {
            "seq_len": (20, 100, "int"),
            "latent_dim": (8, 64, "int"),
            "lr": (0.0001, 0.01, "float")
        }
    # ... other detectors

    # Gaussian Process model
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        n_restarts_optimizer=10
    )

    X_observed = []
    y_observed = []

    for i in range(n_iterations):
        # Acquisition function: Expected Improvement
        if i < 5:
            # Random exploration for first 5 iterations
            params = sample_random(space)
        else:
            # Bayesian optimization
            params = acquisition_function(gp, space, X_observed, y_observed)

        # Run experiment
        auc_pr = run_experiment(detector, dataset, params)

        # Update observations
        X_observed.append(params)
        y_observed.append(auc_pr)

        # Fit GP
        gp.fit(X_observed, y_observed)

        print(f"[{i+1}/{n_iterations}] Params={params}, AUC-PR={auc_pr:.4f}")

    # Return best
    best_idx = np.argmax(y_observed)
    return X_observed[best_idx], y_observed[best_idx]
```

### 9.2 데이터셋별 최적 설정 도출

**출력물**: `HYPERPARAMETER_GUIDE.md`

각 detector × dataset 조합의 최적 설정:
```markdown
## IsolationForest on SKAB
- window: 65
- contamination: 0.08
- n_estimators: 120
- **Performance**: AUC-PR=0.28 (vs 0.24 baseline)
- **Improvement**: +16.7%

## LSTM-AE on SMD
- seq_len: 80
- latent_dim: 48
- lr: 0.0008
- epochs: 45
- **Performance**: F1=0.52 (vs 0.46 baseline)
- **Improvement**: +13%
```

### 9.3 최적 설정으로 재실험

1. 최적 hyperparameter로 20-seed 재실험
2. Baseline (default params) vs Optimized 비교
3. 성능 개선률 측정

**예상 결과**:
- IsolationForest: +10-20% 개선
- LSTM-AE: +5-15% 개선
- kNN: +5-10% 개선

---

## Stage 10: 다중 파일 SKAB 평가 (2-3일)

### 목표
SKAB 전체 30개 파일로 일반화 성능 검증

### 10.1 SKAB 전체 파일 실험

**현재**: valve1/0.csv 1개 파일만 사용
**목표**: 전체 파일로 평가

**SKAB 구조**:
```
SKAB/
├── valve1/
│   ├── 0.csv, 1.csv, ..., 9.csv  (10 files)
├── valve2/
│   ├── 0.csv, 1.csv, ..., 9.csv  (10 files)
├── other/
│   ├── 0.csv, 1.csv, ..., 4.csv  (5 files)
└── anomaly-free/
    ├── 0.csv, 1.csv, ..., 4.csv  (5 files)
```

**실행**:
```python
# scripts/skab_full_evaluation.py

import subprocess
import json

skab_files = []
for subset in ["valve1", "valve2", "other"]:
    for i in range(10 if subset != "other" else 5):
        skab_files.append(f"{subset}/{i}.csv")

results = []
for file in skab_files:
    for detector in ["rule", "ml", "hybrid", "speccnn"]:
        for seed in [42, 142, 242]:  # 3 seeds per file
            cmd = [
                "python3", "-m", "experiments.main_experiment",
                "--dataset", "SKAB",
                "--data-root", "/workspace/data1/arsim/LFactory_d",
                "--detector", detector,
                "--file", file,
                "--seed", str(seed),
                "--run-id", f"skab_{file.replace('/', '_')}_{detector}_seed{seed}"
            ]

            if detector == "ml":
                for ml_method in ["knn", "isolation_forest", "lstm_ae"]:
                    cmd_ml = cmd + ["--ml-method", ml_method]
                    result = subprocess.run(cmd_ml, capture_output=True)
                    results.append({...})
            else:
                result = subprocess.run(cmd, capture_output=True)
                results.append({...})

# Total runs: 30 files × 6 detectors × 3 seeds = 540 runs
```

### 10.2 파일별 난이도 분석

**분석**:
1. **파일별 anomaly rate**
   - anomaly-free: 0%
   - valve1: 평균 35%
   - valve2: 평균 ??%
   - other: 평균 ??%

2. **파일별 최적 알고리즘**
   - 쉬운 파일 (높은 anomaly rate): 모든 detector 성공
   - 어려운 파일 (낮은 anomaly rate): LSTM-AE만 성공

3. **Cross-file 일반화**
   - valve1에서 학습 → valve2에서 테스트
   - Transfer learning 가능성

### 10.3 Ensemble 전략

**아이디어**: 파일 특성에 따라 detector 자동 선택

**방법**:
1. **Meta-learning**:
   - Input: 파일 통계 (길이, anomaly rate, 분산, ...)
   - Output: 최적 detector 예측

2. **Voting ensemble**:
   - 3개 detector의 majority vote
   - Weighted voting (성능 기반 가중치)

3. **Stacking**:
   - Level 1: 6개 detector 예측
   - Level 2: Meta-model이 최종 결정

---

## Stage 11: AIHub71802 데이터 문제 해결 (1일)

### 목표
AIHub71802 zero performance 원인 규명 및 해결

### 11.1 데이터 검사

**체크리스트**:
```python
# scripts/debug_aihub.py

import pandas as pd
import numpy as np

# 1. Load data
data_path = "/workspace/data1/arsim/LFactory_d/manufacturing_transport_71802/Validation/sensor/..."
label_path = "/workspace/data1/arsim/LFactory_d/manufacturing_transport_71802/Validation/label/..."

data = pd.read_csv(data_path)
labels = pd.read_csv(label_path)

# 2. Check data structure
print(f"Data shape: {data.shape}")
print(f"Label shape: {labels.shape}")
print(f"Data columns: {data.columns.tolist()}")
print(f"Label columns: {labels.columns.tolist()}")

# 3. Check label distribution
print(f"Label distribution:\n{labels.value_counts()}")

# 4. Check for NaN
print(f"Data NaN: {data.isna().sum().sum()}")
print(f"Label NaN: {labels.isna().sum().sum()}")

# 5. Check alignment
print(f"Data length: {len(data)}")
print(f"Label length: {len(labels)}")

# 6. Sample visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.plot(data.iloc[:, 0].values[:1000], label="Sensor Value")
plt.plot(labels.iloc[:, 0].values[:1000] * data.iloc[:, 0].max(), label="Label (scaled)", alpha=0.5)
plt.legend()
plt.savefig("aihub_sample.png")
```

### 11.2 문제별 해결 방안

**Problem 1**: 라벨이 모두 0
- **Solution**: Training split 사용 (Validation이 anomaly 없을 수 있음)

**Problem 2**: 센서 데이터 형식 불일치
- **Solution**: Loader 수정, 데이터 정규화 재검토

**Problem 3**: Multi-modal fusion 필요
- **Solution**: Image modality 추가 (advanced, Phase 3)

**Problem 4**: Label scheme 불일치
- **Solution**: binary vs risk4 확인, 올바른 scheme 선택

### 11.3 수정 후 재실험

해결 후:
1. 6 detectors × 20 seeds = 120 runs
2. 성능 측정
3. COMPREHENSIVE_REPORT 업데이트

---

## Stage 12: Phase 2 - Explain 구현 (3-4주)

### 목표
Anomaly detection 결과에 대한 **자연어 설명 자동 생성**

### 12.1 LLM 통합 (Week 1: 5-7일)

#### Option A: OpenAI API
```python
# experiments/llm_explainer.py

import openai

class LLMExplainer:
    def __init__(self, api_key=None, model="gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.model = model

    def explain_anomaly(self, context):
        """Generate natural language explanation for anomaly."""
        prompt = self._build_prompt(context)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in industrial anomaly detection."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        return response.choices[0].message.content

    def _build_prompt(self, context):
        """Build prompt from detection context."""
        template = f"""
        An anomaly was detected in {context['dataset']} data using {context['detector']}.

        **Detection Details**:
        - Time: {context['timestamp']}
        - Anomaly score: {context['score']:.3f}
        - Threshold: {context['threshold']:.3f}
        - Sensor values: {context['values']}
        - Historical average: {context['historical_avg']}
        - Deviation: {context['deviation']:.1f}%

        **Context**:
        - Previous 10 timesteps: {context['history']}
        - Detector type: {context['detector_type']}
        - Feature importance: {context['feature_importance']}

        Please provide:
        1. **What happened**: Describe the anomaly in simple terms
        2. **Why it's anomalous**: Explain why the detector flagged this
        3. **Possible causes**: List 2-3 potential root causes
        4. **Recommended actions**: Suggest next steps for operators

        Format the response in a clear, actionable manner for industrial operators.
        """
        return template
```

#### Option B: Local EXAONE Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLMExplainer:
    def __init__(self, model_name="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def explain_anomaly(self, context):
        prompt = self._build_prompt(context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )

        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation
```

**선택 기준**:
- **OpenAI**: 품질 최고, 비용 발생, 외부 의존성
- **Local EXAONE**: 무료, 프라이버시, GPU 필요 (7.8B model)

**구현 작업**:
1. [x] LLM wrapper 클래스 작성
2. [ ] Prompt template 설계 (5-10개 예시)
3. [ ] API key 관리 (환경변수)
4. [ ] Rate limiting (OpenAI API 제한 고려)
5. [ ] Error handling (API 실패, timeout)

### 12.2 RAG (Retrieval-Augmented Generation) (Week 2: 5-7일)

#### 목표
도메인 지식을 LLM에 주입하여 설명 품질 향상

#### 지식 베이스 구축
```python
# experiments/knowledge_base.py

knowledge_base = {
    "SKAB": {
        "domain": "Industrial valve monitoring",
        "sensors": {
            "TE1": "Temperature sensor 1 (inlet)",
            "TE2": "Temperature sensor 2 (outlet)",
            "P1": "Pressure sensor (inlet)",
            "P2": "Pressure sensor (outlet)",
            "F1": "Flow rate sensor"
        },
        "common_anomalies": [
            {
                "type": "Valve stuck",
                "symptoms": "P1 increases, F1 decreases suddenly",
                "causes": ["Mechanical failure", "Foreign object", "Corrosion"],
                "actions": ["Inspect valve", "Check for blockage", "Replace if needed"]
            },
            {
                "type": "Temperature spike",
                "symptoms": "TE1 or TE2 > 80°C",
                "causes": ["Cooling system failure", "Excessive friction", "External heat"],
                "actions": ["Check cooling system", "Reduce load", "Emergency shutdown if >90°C"]
            }
        ]
    },
    "SMD": {
        "domain": "Server monitoring",
        "metrics": {
            "cpu_usage": "CPU utilization (%)",
            "memory": "Memory usage (MB)",
            "disk_io": "Disk I/O operations/sec",
            "network_in": "Network incoming traffic (MB/s)"
        },
        "common_anomalies": [
            {
                "type": "CPU spike",
                "symptoms": "CPU > 90% for extended period",
                "causes": ["Runaway process", "DDoS attack", "Memory leak"],
                "actions": ["Identify process (top/htop)", "Kill if malicious", "Restart service"]
            }
        ]
    }
}
```

#### Retrieval 구현
```python
# TF-IDF 기반 간단한 retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeRetriever:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.vectorizer = TfidfVectorizer()

        # Index all knowledge
        self.documents = []
        self.metadata = []
        for domain, data in knowledge_base.items():
            for anomaly in data.get("common_anomalies", []):
                doc = f"{anomaly['type']} {anomaly['symptoms']} {' '.join(anomaly['causes'])}"
                self.documents.append(doc)
                self.metadata.append({"domain": domain, "anomaly": anomaly})

        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant knowledge entries."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [self.metadata[i] for i in top_indices]
        return results
```

#### RAG 통합
```python
class RAGExplainer:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def explain_anomaly(self, context):
        # 1. Retrieve relevant knowledge
        query = f"{context['dataset']} {context['sensor']} anomaly score {context['score']}"
        knowledge = self.retriever.retrieve(query, top_k=3)

        # 2. Augment prompt with retrieved knowledge
        prompt = self._build_rag_prompt(context, knowledge)

        # 3. Generate explanation
        explanation = self.llm.explain_anomaly({"prompt": prompt})
        return explanation

    def _build_rag_prompt(self, context, knowledge):
        kb_text = "\n\n".join([
            f"**{k['anomaly']['type']}**:\n"
            f"- Symptoms: {k['anomaly']['symptoms']}\n"
            f"- Causes: {', '.join(k['anomaly']['causes'])}\n"
            f"- Actions: {', '.join(k['anomaly']['actions'])}"
            for k in knowledge
        ])

        template = f"""
        You are an expert in {context['dataset']} anomaly detection.

        **Relevant Domain Knowledge**:
        {kb_text}

        **Current Anomaly**:
        {context}

        Based on the domain knowledge above, explain this anomaly.
        """
        return template
```

**작업 체크리스트**:
1. [ ] 도메인 지식 베이스 작성 (SKAB, SMD, Synthetic)
2. [ ] TF-IDF retriever 구현
3. [ ] RAG prompt template 설계
4. [ ] End-to-end 테스트 (5-10개 샘플)
5. [ ] 설명 품질 평가 (사람 평가 or GPT-4 as judge)

### 12.3 Bayesian Prior for Cost Matrix (Week 3: 5-7일)

#### 목표
사용자 피드백을 학습하여 cost matrix 동적 조정

#### 시나리오
```
User marks FP as "acceptable" → Reduce C_FP
User marks FN as "critical" → Increase C_FN
```

#### Bayesian Update
```python
# experiments/bayesian_cost_learner.py

import numpy as np
from scipy import stats

class BayesianCostLearner:
    def __init__(self, prior_c_fp=1.0, prior_c_fn=5.0):
        """Initialize with prior cost matrix."""
        self.c_fp_prior = stats.gamma(a=2, scale=prior_c_fp/2)  # Gamma distribution
        self.c_fn_prior = stats.gamma(a=2, scale=prior_c_fn/2)

        # Posterior (updated with feedback)
        self.c_fp_posterior = self.c_fp_prior
        self.c_fn_posterior = self.c_fn_prior

    def update(self, feedback):
        """Update cost matrix based on user feedback.

        Args:
            feedback: dict with keys:
                - type: "FP" or "FN"
                - severity: float in [0, 10]
        """
        if feedback["type"] == "FP":
            # Update FP cost
            # Higher severity → higher cost
            observed_cost = feedback["severity"]
            self.c_fp_posterior = self._bayesian_update(
                self.c_fp_posterior, observed_cost
            )
        elif feedback["type"] == "FN":
            observed_cost = feedback["severity"]
            self.c_fn_posterior = self._bayesian_update(
                self.c_fn_posterior, observed_cost
            )

    def _bayesian_update(self, prior, observation):
        """Bayesian update using conjugate prior."""
        # Simplified: use sample mean as posterior
        # In practice, use proper Bayesian inference
        prior_mean = prior.mean()
        posterior_mean = 0.8 * prior_mean + 0.2 * observation  # Weighted average
        return stats.gamma(a=2, scale=posterior_mean/2)

    def get_cost_matrix(self):
        """Return current cost matrix estimate."""
        return {
            "c00": 0.0,
            "c01": self.c_fp_posterior.mean(),
            "c10": self.c_fn_posterior.mean(),
            "c11": 0.0
        }
```

#### Feedback Collection UI (간단한 CLI)
```python
def collect_feedback():
    """Collect user feedback on detection result."""
    print("Detection result: Anomaly detected at t=1234")
    print("Ground truth: Normal")
    print("This is a False Positive (FP).")

    severity = input("How severe is this false alarm? (0=harmless, 10=critical): ")
    severity = float(severity)

    return {"type": "FP", "severity": severity}
```

**작업 체크리스트**:
1. [ ] Bayesian update 수식 검증
2. [ ] Feedback collection mechanism
3. [ ] A/B test: Fixed cost vs Adaptive cost
4. [ ] Simulation: 100 feedback cycles
5. [ ] 수렴 속도 분석 (몇 번 feedback 후 안정화?)

### 12.4 End-to-End 파이프라인 통합 (Week 4: 3-5일)

#### 목표
Detect + Explain + Learn 전체 파이프라인 구축

#### 통합 시스템
```python
# experiments/end_to_end_pipeline.py

class AnomalyDetectionPipeline:
    def __init__(self, detector, explainer, cost_learner):
        self.detector = detector
        self.explainer = explainer
        self.cost_learner = cost_learner

    def process_stream(self, data_stream):
        """Process time series stream with detection, explanation, and learning."""
        for t, value in enumerate(data_stream):
            # 1. Detect
            score = self.detector.get_score(value)
            threshold = self.detector.get_threshold(
                cost_matrix=self.cost_learner.get_cost_matrix()
            )
            is_anomaly = score > threshold

            if is_anomaly:
                # 2. Explain
                context = {
                    "timestamp": t,
                    "value": value,
                    "score": score,
                    "threshold": threshold,
                    "history": data_stream[max(0, t-10):t],
                    "detector": self.detector.name
                }
                explanation = self.explainer.explain_anomaly(context)

                # 3. Present to user
                print(f"⚠️ Anomaly at t={t}")
                print(f"Score: {score:.3f} (threshold: {threshold:.3f})")
                print(f"\n{explanation}\n")

                # 4. Collect feedback
                feedback = self.collect_feedback(t, value, is_anomaly)

                # 5. Learn
                if feedback:
                    self.cost_learner.update(feedback)
                    print(f"✅ Cost matrix updated: {self.cost_learner.get_cost_matrix()}")

    def collect_feedback(self, t, value, predicted_anomaly):
        """Collect user feedback (simulation or real user)."""
        # In real system, prompt user
        # For simulation, use ground truth
        ground_truth = self.get_ground_truth(t)

        if predicted_anomaly and not ground_truth:
            # False Positive
            severity = random.uniform(1, 5)  # Simulate user rating
            return {"type": "FP", "severity": severity}
        elif not predicted_anomaly and ground_truth:
            # False Negative
            severity = random.uniform(5, 10)  # FN more severe
            return {"type": "FN", "severity": severity}
        else:
            # Correct prediction, no feedback needed
            return None
```

#### 실행 예시
```python
# Demo script
detector = IsolationForestDetector(...)
explainer = RAGExplainer(llm=OpenAILLM(), retriever=KnowledgeRetriever(...))
learner = BayesianCostLearner(prior_c_fp=1.0, prior_c_fn=5.0)

pipeline = AnomalyDetectionPipeline(detector, explainer, learner)

# Load SKAB data
data = load_skab("valve1/0.csv")
pipeline.process_stream(data["values"])
```

**작업 체크리스트**:
1. [ ] Pipeline 클래스 구현
2. [ ] Streaming mode 구현
3. [ ] Feedback loop 테스트
4. [ ] 사용자 인터페이스 (CLI or Web)
5. [ ] Demo 비디오 녹화

### 12.5 Phase 2 평가 (Week 4: 2일)

#### 설명 품질 평가

**Metrics**:
1. **Faithfulness**: 설명이 실제 detector 동작을 정확히 반영하는가?
   - Method: 설명에서 언급한 feature를 ablation → score 변화 측정
   - Good explanation: 언급한 feature 제거 시 score 크게 변화

2. **Plausibility**: 설명이 도메인 전문가에게 합리적인가?
   - Method: 사람 평가 (5-point Likert scale)
   - 5-10명 평가자, 10-20개 샘플

3. **Actionability**: 설명이 구체적인 조치를 제안하는가?
   - Method: Count actionable items (e.g., "Check valve", "Restart server")

#### Bayesian Learning 평가

**Metrics**:
1. **Convergence speed**: 몇 번의 feedback 후 cost matrix 안정화?
2. **Final accuracy**: 최종 learned cost vs ground-truth cost
3. **Regret**: Cumulative cost over time (early mistakes)

#### 최종 보고서

**출력물**: `PHASE2_EXPLANATION_REPORT.md`

**구조**:
1. **Introduction**: Phase 2 목표 및 접근법
2. **LLM Integration**: OpenAI vs Local EXAONE 비교
3. **RAG System**: 지식 베이스 구축 및 retrieval 성능
4. **Bayesian Learning**: Cost matrix adaptation 실험 결과
5. **Case Studies**: 5-10개 실제 anomaly 설명 예시
6. **User Study**: 사람 평가 결과 (if available)
7. **Limitations & Future Work**

---

# 📅 전체 일정 (Timeline)

## Week 1-2: Phase 1.5 - Detect 고도화

| 날짜 | Stage | 작업 | 예상 시간 |
|------|-------|------|-----------|
| Day 1-2 | Stage 7.1-7.2 | 성능 비교 시각화 + 데이터셋별 분석 | 2일 |
| Day 3 | Stage 7.3-7.4 | 크로스 데이터셋 분석 + Ablation 설계 | 1일 |
| Day 4-5 | Stage 7.5 | 논문 작성 (초안) | 2일 |
| Day 6 | Stage 8 | SpecCNN weight 최적화 | 1일 |
| Day 7-9 | Stage 9 | 하이퍼파라미터 Bayesian 최적화 | 3일 |
| Day 10-12 | Stage 10 | SKAB 전체 파일 평가 | 3일 |
| Day 13 | Stage 11 | AIHub71802 디버깅 | 1일 |
| Day 14 | - | 버퍼 / 리뷰 | 1일 |

**Milestone 1 (Week 2 완료)**:
- ✅ 알고리즘 비교 논문 초안 완성
- ✅ 모든 detector 최적화 완료
- ✅ 4개 데이터셋 완전 평가

## Week 3-6: Phase 2 - Explain

| 날짜 | Stage | 작업 | 예상 시간 |
|------|-------|------|-----------|
| Day 15-17 | Stage 12.1 | LLM 통합 (OpenAI + Local EXAONE) | 3일 |
| Day 18-19 | Stage 12.1 | Prompt engineering | 2일 |
| Day 20-22 | Stage 12.2 | RAG 구현 (지식 베이스 + Retrieval) | 3일 |
| Day 23-24 | Stage 12.2 | RAG 테스트 및 평가 | 2일 |
| Day 25-27 | Stage 12.3 | Bayesian cost learner 구현 | 3일 |
| Day 28-29 | Stage 12.3 | Feedback simulation 및 A/B test | 2일 |
| Day 30-32 | Stage 12.4 | End-to-end 파이프라인 통합 | 3일 |
| Day 33-34 | Stage 12.4 | 사용자 인터페이스 (CLI/Web) | 2일 |
| Day 35-36 | Stage 12.5 | Phase 2 평가 및 보고서 | 2일 |
| Day 37-40 | - | 버퍼 / 사용자 테스트 / 수정 | 4일 |

**Milestone 2 (Week 6 완료)**:
- ✅ LLM 기반 설명 생성 시스템 완성
- ✅ RAG 지식 베이스 구축
- ✅ Bayesian cost learning 작동
- ✅ Phase 2 보고서 완성

---

# 🎯 핵심 성과물 (Deliverables)

## Phase 1.5 성과물

### 1. 알고리즘 비교 논문
- **파일**: `ALGORITHM_COMPARISON_PAPER.pdf`
- **페이지**: 10-12 pages (IEEE format)
- **내용**: RQ1-4 답변, 통계 검정, 비교 분석
- **목표**: IEEE TKDE 투고

### 2. 시각화 패키지
- **디렉토리**: `figures/`
- **내용**:
  - Performance comparison charts (7개)
  - Statistical heatmaps (2개)
  - ROC/PR curves (3×6=18개)
  - Calibration diagrams (3×6=18개)
- **형식**: PNG (300 DPI), PDF (vector)

### 3. 하이퍼파라미터 가이드
- **파일**: `HYPERPARAMETER_GUIDE.md`
- **내용**:
  - 각 detector × dataset 최적 설정
  - Sensitivity analysis
  - Tuning workflow

### 4. SKAB 전체 평가 보고서
- **파일**: `SKAB_FULL_EVALUATION.md`
- **내용**:
  - 30개 파일별 성능
  - 파일 난이도 분석
  - Cross-file 일반화

### 5. 업데이트된 종합 보고서
- **파일**: `COMPREHENSIVE_EXPERIMENT_REPORT_v2.md`
- **변경사항**:
  - SpecCNN 최적화 결과 추가
  - 하이퍼파라미터 최적 설정 반영
  - AIHub71802 수정 결과

## Phase 2 성과물

### 1. LLM 설명 시스템
- **디렉토리**: `experiments/llm/`
- **파일**:
  - `llm_explainer.py` - OpenAI wrapper
  - `local_llm_explainer.py` - EXAONE wrapper
  - `prompt_templates.py` - 10개 템플릿
- **Demo**: 5-10개 anomaly 설명 샘플

### 2. RAG 지식 베이스
- **파일**: `knowledge_base.json`
- **내용**:
  - SKAB: 10개 common anomalies
  - SMD: 8개 common anomalies
  - Synthetic: 5개 pattern types
- **크기**: ~500 entries

### 3. Bayesian Cost Learner
- **파일**: `bayesian_cost_learner.py`
- **기능**:
  - Feedback collection
  - Bayesian update
  - A/B test framework
- **Validation**: 100-iteration simulation

### 4. End-to-End 파이프라인
- **파일**: `end_to_end_pipeline.py`
- **기능**:
  - Streaming detection
  - Real-time explanation
  - Interactive feedback
- **Interface**: CLI (기본) + Web (선택)

### 5. Phase 2 최종 보고서
- **파일**: `PHASE2_EXPLANATION_REPORT.md`
- **페이지**: 20-25 pages
- **내용**:
  - LLM integration 비교
  - RAG system 평가
  - Bayesian learning 실험
  - Case studies (5-10개)
  - User study 결과

### 6. Demo 비디오
- **파일**: `demo.mp4`
- **길이**: 5-10분
- **내용**:
  - Live anomaly detection
  - Explanation generation
  - Feedback & learning
- **형식**: Screen recording + narration

---

# 📊 성공 지표 (Success Metrics)

## Phase 1.5 목표

| Metric | Target | Measurement |
|--------|--------|-------------|
| **SpecCNN AUC-PR** | > 0.3 (현재 0) | Grid search 후 재측정 |
| **IsolationForest 개선** | +15% AUC-PR | Optimized vs Baseline |
| **LSTM-AE 개선** | +10% F1 | Optimized vs Baseline |
| **SKAB 전체 평가** | 30 files × 6 detectors | 180 runs 완료 |
| **논문 작성** | 10-12 pages | Peer review ready |

## Phase 2 목표

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Explanation Faithfulness** | > 0.7 | Feature ablation test |
| **Explanation Plausibility** | > 4.0/5.0 | Human evaluation (5 judges) |
| **Actionability** | > 2 actions/explanation | Automatic counting |
| **Cost Matrix Convergence** | < 20 feedbacks | Simulation |
| **User Satisfaction** | > 4.0/5.0 | User study (10 users) |

---

# 🔧 개발 환경 및 도구

## 필요한 추가 패키지

```bash
# LLM
pip install openai transformers accelerate

# Bayesian Optimization
pip install scikit-optimize bayesian-optimization

# Visualization
pip install seaborn plotly

# Web UI (optional)
pip install streamlit fastapi

# Evaluation
pip install rouge_score bert_score
```

## 하드웨어 요구사항

### Phase 1.5 (현재 환경으로 충분)
- CPU: 8+ cores
- RAM: 16GB
- GPU: Optional (LSTM-AE 가속용)

### Phase 2
- CPU: 8+ cores
- RAM: 32GB (Local LLM 사용 시)
- GPU: 16GB+ VRAM (EXAONE-7.8B 실행 시)
  - RTX 3090, A6000, A100 권장

**대안**: OpenAI API 사용 (GPU 불필요, 비용 발생)

---

# ⚠️ 위험 요소 및 대응 방안

## Phase 1.5 위험

| 위험 | 확률 | 영향 | 대응 방안 |
|------|------|------|-----------|
| SpecCNN grid search 시간 초과 | 중 | 중 | Random search 20% 사용 |
| LSTM-AE 최적화 너무 느림 | 고 | 중 | Early stopping, smaller grid |
| SKAB 다중 파일에서 성능 저하 | 중 | 중 | Ensemble 전략 적용 |
| AIHub71802 데이터 복구 실패 | 중 | 저 | 3개 데이터셋만으로 진행 |

## Phase 2 위험

| 위험 | 확률 | 영향 | 대응 방안 |
|------|------|------|-----------|
| OpenAI API 비용 초과 | 중 | 중 | Local EXAONE으로 전환 |
| EXAONE GPU 메모리 부족 | 중 | 고 | Quantization (4-bit), Smaller model |
| 설명 품질 낮음 | 중 | 고 | Prompt engineering 반복, RAG 강화 |
| Bayesian learning 수렴 안 됨 | 저 | 중 | 다른 prior 시도, Beta distribution |
| 사용자 테스트 모집 실패 | 중 | 저 | Simulated feedback 사용 |

---

# 📝 체크리스트 요약

## Phase 1.5 (Week 1-2)
- [ ] Stage 7: 알고리즘 비교 보고서
  - [ ] 7.1 시각화 생성 (7개 차트)
  - [ ] 7.2 데이터셋별 분석
  - [ ] 7.3 크로스 데이터셋 일반화
  - [ ] 7.4 Ablation study 확장
  - [ ] 7.5 논문 작성 (10-12 pages)
- [ ] Stage 8: SpecCNN 최적화
  - [ ] 8.1 Grid search (216 combinations)
  - [ ] 8.2 최적 weights로 재실험
  - [ ] 8.3 Frequency-domain 분석
- [ ] Stage 9: 하이퍼파라미터 최적화
  - [ ] 9.1 Bayesian optimization 구현
  - [ ] 9.2 데이터셋별 최적 설정
  - [ ] 9.3 최적 설정으로 재실험
- [ ] Stage 10: SKAB 전체 평가
  - [ ] 10.1 30개 파일 실험
  - [ ] 10.2 파일별 난이도 분석
  - [ ] 10.3 Ensemble 전략
- [ ] Stage 11: AIHub71802 디버깅
  - [ ] 11.1 데이터 검사
  - [ ] 11.2 문제 해결
  - [ ] 11.3 재실험

## Phase 2 (Week 3-6)
- [ ] Stage 12.1: LLM 통합
  - [ ] OpenAI API wrapper
  - [ ] Local EXAONE wrapper
  - [ ] Prompt template 설계
  - [ ] Error handling
- [ ] Stage 12.2: RAG 구현
  - [ ] 지식 베이스 작성
  - [ ] TF-IDF retriever
  - [ ] RAG prompt template
  - [ ] End-to-end 테스트
- [ ] Stage 12.3: Bayesian Cost Learning
  - [ ] Bayesian update 구현
  - [ ] Feedback collection
  - [ ] A/B test
  - [ ] Simulation (100 iterations)
- [ ] Stage 12.4: 파이프라인 통합
  - [ ] End-to-end pipeline
  - [ ] Streaming mode
  - [ ] User interface (CLI/Web)
  - [ ] Demo 비디오
- [ ] Stage 12.5: Phase 2 평가
  - [ ] 설명 품질 평가
  - [ ] Bayesian learning 평가
  - [ ] 최종 보고서

---

# 🎓 학습 및 참고 자료

## Phase 1.5 참고 논문
1. Hyperparameter Optimization:
   - Bergstra & Bengio (2012) - Random Search
   - Snoek et al. (2012) - Bayesian Optimization

2. Time Series Anomaly Detection:
   - Su et al. (2019) - Robust Anomaly Detection for Multivariate Time Series
   - Lai et al. (2021) - Revisiting Time Series Outlier Detection

3. Frequency-Domain Methods:
   - Cleveland et al. (1990) - STL: Seasonal-Trend decomposition
   - Wen et al. (2020) - Spectral Residual for Anomaly Detection

## Phase 2 참고 자료
1. LLM for Explanation:
   - Lewis et al. (2020) - RAG: Retrieval-Augmented Generation
   - Wei et al. (2022) - Chain-of-Thought Prompting

2. Explainable AI:
   - Ribeiro et al. (2016) - LIME
   - Lundberg & Lee (2017) - SHAP

3. Bayesian Learning:
   - Murphy (2012) - Machine Learning: A Probabilistic Perspective
   - Ghahramani (2015) - Probabilistic Machine Learning and AI

---

**문서 작성**: 2025-11-24
**예상 완료**: 2025-12-31 (Phase 1.5) + 2026-01-31 (Phase 2)
**담당자**: LFactory Team
**상태**: 계획 단계 → 실행 대기

---

**다음 단계**: Stage 7부터 시작 (사용자 승인 후)
