# LFactory 연구 계획 (수정판)
## LLM-Guided Manufacturing Anomaly Detection & Interpretation

**작성일**: 2025-11-25  
**상태**: Phase 1 완료 → Phase 2-3 진행 중  
**핵심 목표**: 제조 공정 데이터에서 ML 이상 탐지 + LLM 해석 + LLM 기반 최적화  

---

## 🎯 연구의 진짜 목적 (Real Objective)

```
제조 공정 데이터
    ↓ (Phase 1: Detect)
ML 이상 탐지 (IsolationForest, LSTM-AE, ...)
    ↓ (Phase 2: Explain)
LLM이 3가지 해석:
  1. 이상치 자체 해석 (왜 이상인가?)
  2. ML 모델 해석 (모델이 왜 이렇게 판단했나?)
  3. 도메인 지식 연결 (제조 공정 관점에서 의미는?)
    ↓ (Phase 3: Optimize)
LLM이 도메인 지식으로 ML 파라미터 제안
    ↓
개선된 ML 탐지기
```

### 핵심 차별점
- ❌ **단순 이상 탐지**: 기존 연구 많음
- ❌ **ML만**: 해석 불가능
- ✅ **ML + LLM 통합**: 탐지 + 해석 + 최적화 ← **우리 연구!**

---

## 📊 현재 완료 상태 (2025-11-25)

### ✅ Phase 1: ML 이상 탐지 (100% 완료)
- [x] 6가지 detector 구현: Rule, kNN, IsolationForest, LSTM-AE, Hybrid, SpecCNN
- [x] 480 runs 실험 (4 datasets × 6 detectors × 20 seeds)
- [x] 통계 검증 (Wilcoxon test, Bootstrap CI, Correlation)
- [x] **핵심 발견**:
  - SKAB: LSTM-AE 최고 (F1=0.087, AUC-PR=0.338)
  - SMD: IsolationForest 최고 (F1=0.458, AUC-PR=0.543)
  - Point-wise Recall ↔ Event-wise Precision 강한 상관 (r=0.799)
- [x] 종합 보고서: `COMPREHENSIVE_EXPERIMENT_REPORT.md` (35페이지)

### ❌ Phase 2-3: LLM 통합 (0% 완료 - 최우선!)
- [ ] LLM 이상치 해석
- [ ] **LLM ML 모델 해석** ← **새로운 요구사항!**
- [ ] LLM 도메인 최적화

---

## 🔥 Phase 2: LLM 해석 시스템 (Week 1-2, 최우선!)

### 목표
제조 공정 이상을 LLM이 **3가지 관점에서 해석**

### 2.1 이상치 해석 (Anomaly Explanation)

**입력**:
```python
anomaly = {
    "timestamp": 1234,
    "dataset": "SKAB",
    "sensor_values": {"P1": 5.2, "TE1": 85.3, "F1": 0.3},
    "anomaly_score": 0.89,
    "detector": "IsolationForest",
    "threshold": 0.5
}
```

**LLM 출력**:
```
⚠️ 이상 탐지 (SKAB Valve System, t=1234)

**무엇이 발생했나**:
- 압력(P1): 5.2 bar ⬆️ (정상 범위: 3-4 bar)
- 온도(TE1): 85.3°C ⬆️ (정상 범위: 60-70°C)
- 유량(F1): 0.3 L/min ⬇️ (정상 범위: 1.5-2.0 L/min)

**왜 이상인가**:
IsolationForest 스코어 0.89는 이 (압력↑, 온도↑, 유량↓) 조합이
정상 동작 패턴과 매우 다름을 나타냅니다.

**제조 공정 관점**:
밸브 막힘의 전형적 신호 - 압력 증가, 유량 감소, 마찰열 발생

**가능한 원인**:
1. 밸브 내부 이물질 또는 스케일 축적 (가능성 70%)
2. 밸브 시트 손상으로 인한 불완전 개폐 (가능성 20%)
3. 센서 오류 (가능성 10%)

**권장 조치**:
1. 즉시: 밸브 점검 스케줄 (다음 정지 시)
2. 모니터링 강화: 압력/온도 1분 간격 체크
3. 예비 밸브 준비
4. 센서 캘리브레이션 기록 확인
```

**구현**:
```python
# experiments/llm_anomaly_explainer.py

import openai

OPENAI_API_KEY = "sk-proj-..." # .claude/claude.md에서 가져옴

def explain_anomaly(anomaly, domain_knowledge):
    """LLM이 이상치를 해석"""

    prompt = f"""
You are an expert in manufacturing anomaly analysis.

Dataset: {anomaly['dataset']}
Domain: {domain_knowledge['domain_description']}
Normal ranges: {domain_knowledge['sensor_ranges']}

Detected anomaly:
- Time: {anomaly['timestamp']}
- Sensors: {anomaly['sensor_values']}
- Anomaly score: {anomaly['anomaly_score']}
- Detector: {anomaly['detector']}

Explain this anomaly in 4 parts:
1. What happened (sensor deviations)
2. Why it's anomalous (statistical perspective)
3. Manufacturing perspective (domain meaning)
4. Possible causes with probability estimates
5. Recommended actions

Be specific, actionable, and consider manufacturing domain knowledge.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a manufacturing anomaly expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
```

### 2.2 ML 모델 해석 (Model Explanation) ⭐ **핵심!**

**목표**: LLM이 **ML 모델의 판단 과정을 해석**

**시나리오**:
```
"IsolationForest가 왜 이 점을 anomaly로 판단했는가?"
"어떤 feature가 가장 중요했는가?"
"Decision boundary는 어떻게 생겼는가?"
```

**구현 방법**:

#### 2.2.1 Feature Importance 해석
```python
# IsolationForest feature importance 추출
from sklearn.inspection import permutation_importance

# 모델 학습
model = IsolationForest(...)
model.fit(X_train)

# Feature importance
result = permutation_importance(model, X_test, scoring='roc_auc')
feature_importance = dict(zip(feature_names, result.importances_mean))

# LLM에게 해석 요청
llm_explanation = explain_feature_importance(
    model="IsolationForest",
    features=feature_importance,
    domain="SKAB valve system",
    anomaly_point=anomaly
)
```

**LLM 출력 예시**:
```
**IsolationForest 모델 해석**

이 모델이 anomaly로 판단한 이유:

**가장 중요한 Feature (Top 3)**:
1. **유량(F1) 차이**: 중요도 0.42
   - 정상: 1.8 L/min, 이상: 0.3 L/min
   - 83% 감소 → 모델이 가장 강하게 반응

2. **압력(P1) 증가**: 중요도 0.31
   - 정상: 3.5 bar, 이상: 5.2 bar
   - 49% 증가 → 두 번째로 중요한 신호

3. **온도(TE1) 증가**: 중요도 0.18
   - 정상: 65°C, 이상: 85.3°C
   - 31% 증가 → 보조 신호

**모델의 판단 논리**:
IsolationForest는 "유량 급감 + 압력 급증" 조합이
정상 데이터에서 거의 관찰되지 않는다고 학습했습니다.

이 조합은 전체 학습 데이터의 0.5%에서만 발생 → Isolation 쉬움

**신뢰도**: 높음 (3개 feature 모두 일관된 신호)
```

#### 2.2.2 SHAP/LIME 해석 + LLM
```python
import shap

# SHAP values 계산
explainer = shap.Explainer(model, X_train)
shap_values = explainer(anomaly_point)

# LLM이 SHAP values를 자연어로 설명
llm_shap_explanation = explain_shap_values(
    shap_values=shap_values,
    feature_names=feature_names,
    domain="manufacturing"
)
```

**LLM 출력 예시**:
```
**SHAP 분석 결과 (모델의 세부 판단)**

이 anomaly score 0.89는 다음과 같이 구성됩니다:

Base score (평균):        0.15
+ F1 contribution:       +0.35 (유량 급감의 영향)
+ P1 contribution:       +0.28 (압력 증가의 영향)
+ TE1 contribution:      +0.11 (온도 증가의 영향)
= Final score:            0.89

**해석**:
- 유량 감소가 가장 큰 영향 (전체 score의 47%)
- 압력 증가가 두 번째 (전체 score의 37%)
- 온도는 보조적 역할 (전체 score의 15%)

만약 유량이 정상이었다면 score는 0.54 정도였을 것 (threshold 0.5 초과하지만 덜 확실)
```

#### 2.2.3 Decision Boundary 시각화 + LLM 설명
```python
# 2D projection으로 decision boundary 시각화
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
anomaly_2d = pca.transform(anomaly_point)

# Plot decision boundary
plot_decision_boundary(model, X_2d, anomaly_2d)

# LLM이 시각화를 설명
llm_boundary_explanation = explain_decision_boundary(
    anomaly_position=anomaly_2d,
    normal_region_center=normal_center_2d,
    distance_to_boundary=distance
)
```

**LLM 출력 예시**:
```
**Decision Boundary 분석**

이상 점의 위치:
- PCA 2D 공간에서 좌표: (3.2, -1.8)
- 정상 영역 중심에서 거리: 4.5 (표준편차 단위)

**시각적 설명**:
정상 데이터는 원점 (0,0) 주변에 밀집된 타원형 분포.
이 anomaly는 정상 영역에서 오른쪽 아래 방향으로
크게 벗어나 있습니다 (4.5σ).

이는 "높은 압력 + 낮은 유량" 조합이
정상 동작 패턴과 근본적으로 다름을 의미합니다.

**유사한 과거 사례**:
학습 데이터에서 이 영역에 있던 점들은:
- 80%가 "밸브 막힘" 라벨
- 15%가 "센서 오류" 라벨
- 5%가 오탐지
```

### 2.3 도메인 지식 베이스 구축

**목표**: LLM이 참조할 제조 공정 지식 작성

**파일**: `experiments/knowledge_base/SKAB_valve_system.md`

```markdown
# SKAB Valve System Domain Knowledge

## System Description
Industrial valve monitoring system with 8 sensors tracking pressure, temperature, and flow.

## Sensors
1. **P1**: Inlet pressure (bar)
   - Normal range: 3-4 bar
   - Critical: >6 bar (safety valve trigger)

2. **P2**: Outlet pressure (bar)
   - Normal range: 2-3 bar
   - Should be lower than P1

3. **TE1**: Inlet temperature (°C)
   - Normal range: 60-70°C
   - Critical: >90°C (overheating risk)

4. **TE2**: Outlet temperature (°C)
   - Normal range: 55-65°C

5. **F1**: Flow rate (L/min)
   - Normal range: 1.5-2.0 L/min
   - Critical: <0.5 L/min (blockage)

## Common Anomalies

### 1. Valve Blockage
**Symptoms**:
- P1 increases (pressure builds up)
- F1 decreases (flow restricted)
- TE1 increases (friction heat)

**Root Causes**:
- Scale buildup (70% of cases)
- Foreign object (20%)
- Valve seat damage (10%)

**Actions**:
1. Schedule valve inspection
2. Check maintenance log (last cleaning?)
3. Prepare backup valve

### 2. Cooling System Failure
**Symptoms**:
- TE1, TE2 both increase
- P1, P2, F1 remain normal

**Root Causes**:
- Coolant pump failure
- Coolant leak
- Heat exchanger fouling

**Actions**:
1. Check coolant level
2. Inspect pump operation
3. Emergency shutdown if >90°C

### 3. Sensor Drift
**Symptoms**:
- Only 1 sensor shows abnormal value
- Other sensors normal
- Value changes slowly, not suddenly

**Actions**:
1. Cross-check with manual gauge
2. Review calibration records
3. Schedule sensor replacement if needed
```

**구현**:
```python
# experiments/knowledge_base_manager.py

def load_domain_knowledge(dataset):
    """Load domain knowledge for dataset"""
    knowledge_files = {
        "SKAB": "knowledge_base/SKAB_valve_system.md",
        "SMD": "knowledge_base/SMD_server_metrics.md"
    }

    with open(knowledge_files[dataset]) as f:
        return f.read()

def retrieve_relevant_knowledge(query, knowledge_base):
    """RAG: Retrieve relevant sections from knowledge base"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Split knowledge base into sections
    sections = knowledge_base.split("##")

    # TF-IDF similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + sections)

    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Return top 3 relevant sections
    top_indices = similarities.argsort()[-3:][::-1]
    return [sections[i] for i in top_indices]
```

### 2.4 통합 설명 시스템

**최종 구현**:
```python
# experiments/integrated_explainer.py

class IntegratedExplainer:
    def __init__(self, openai_api_key):
        self.api_key = openai_api_key
        openai.api_key = self.api_key

    def explain_full(self, anomaly, model, domain_knowledge):
        """3-in-1 explanation: Anomaly + Model + Domain"""

        # 1. Anomaly explanation
        anomaly_exp = self.explain_anomaly(anomaly, domain_knowledge)

        # 2. Model explanation (SHAP + feature importance)
        model_exp = self.explain_model_decision(
            model=model,
            anomaly_point=anomaly['features'],
            feature_names=anomaly['feature_names']
        )

        # 3. Domain-connected explanation (RAG)
        domain_exp = self.explain_domain_context(
            anomaly=anomaly,
            model_explanation=model_exp,
            knowledge_base=domain_knowledge
        )

        # LLM synthesizes all 3
        return self.synthesize_explanation(
            anomaly_exp, model_exp, domain_exp
        )

    def synthesize_explanation(self, anomaly_exp, model_exp, domain_exp):
        """LLM combines 3 explanations into coherent narrative"""

        prompt = f"""
Synthesize a comprehensive explanation from:

1. Anomaly Analysis:
{anomaly_exp}

2. Model Decision Analysis:
{model_exp}

3. Domain Context:
{domain_exp}

Create a unified explanation that:
- Tells a coherent story
- Links model decision to domain meaning
- Provides actionable insights
- Uses clear language for operators

Format:
## Executive Summary (2 sentences)
## What Happened (data-driven)
## Why The Model Flagged It (model-driven)
## Manufacturing Perspective (domain-driven)
## Root Cause Analysis (synthesized)
## Recommended Actions (prioritized)
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert integrating ML and manufacturing knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content
```

### 2.5 Phase 2 작업 목록 (Week 1-2)

**Week 1: 기본 구현**
- [ ] Day 1: OpenAI API 통합, 기본 prompt template
- [ ] Day 2: Anomaly explanation 구현 및 테스트 (5개 샘플)
- [ ] Day 3: SHAP/Feature importance 추출 코드
- [ ] Day 4: ML 모델 해석 LLM prompt
- [ ] Day 5: SKAB 도메인 지식 베이스 작성
- [ ] Day 6: SMD 도메인 지식 베이스 작성
- [ ] Day 7: RAG retrieval 구현

**Week 2: 통합 및 검증**
- [ ] Day 8: 3-in-1 통합 설명 시스템
- [ ] Day 9: 10개 anomaly 샘플 설명 생성
- [ ] Day 10: 설명 품질 평가 (사람 or GPT-4 judge)
- [ ] Day 11: 문제점 수정 및 개선
- [ ] Day 12-14: Phase 2 보고서 작성

**성과물**:
- `experiments/llm_explainer.py` - 통합 설명 시스템
- `experiments/knowledge_base/` - 도메인 지식 (SKAB, SMD)
- `PHASE2_LLM_EXPLANATION_REPORT.md` - 설명 예시 10개 포함
- Demo 스크립트: `scripts/demo_explanation.py`

---

## 🚀 Phase 3: LLM 기반 도메인 최적화 (Week 3-4)

### 목표
LLM이 제조 도메인 지식을 활용하여 ML 파라미터를 제안 → 성능 개선

### 3.1 LLM Parameter Advisor

**시나리오**:
```
User: "SKAB 데이터에서 IsolationForest F1이 0.033으로 낮습니다. 어떻게 개선하나요?"

LLM: "SKAB 밸브 시스템의 특성을 고려하면:
1. window_size: 50 → 150
   - 이유: 밸브 이상은 50-200 timesteps 지속
2. contamination: 0.1 → 0.35
   - 이유: Anomaly rate가 35%
3. n_estimators: 100 → 200
   - 이유: 8개 센서로 feature space 복잡

예상 개선: F1 0.033 → 0.15+"
```

**구현**:
```python
# experiments/llm_parameter_advisor.py

def suggest_parameters(dataset, detector, current_performance, domain_knowledge):
    """LLM suggests optimal parameters based on domain knowledge"""

    prompt = f"""
You are an ML expert specializing in manufacturing anomaly detection.

Dataset: {dataset}
Domain characteristics:
{domain_knowledge}

Current setup:
- Detector: {detector}
- Performance: {current_performance}
- Parameters: {current_parameters}

Based on domain knowledge, suggest optimal parameters.

For each parameter:
1. Current value
2. Suggested value
3. Reason (domain-driven)
4. Expected impact

Also estimate expected performance improvement.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an ML parameter tuning expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    # Parse LLM response to extract parameters
    suggestions = parse_parameter_suggestions(response.choices[0].message.content)
    return suggestions
```

### 3.2 실험 프로토콜

**Baseline vs LLM-guided 비교**:

```python
# experiments/llm_optimization_experiment.py

def run_optimization_experiment(dataset, detector):
    """Compare Baseline vs LLM-guided parameters"""

    # Baseline (default parameters)
    baseline_params = get_default_params(detector)
    baseline_results = run_experiments(
        dataset=dataset,
        detector=detector,
        params=baseline_params,
        seeds=range(42, 1042, 50)  # 20 seeds
    )

    # LLM-guided
    domain_knowledge = load_domain_knowledge(dataset)
    llm_params = suggest_parameters(
        dataset=dataset,
        detector=detector,
        current_performance=baseline_results['mean_f1'],
        domain_knowledge=domain_knowledge
    )

    llm_results = run_experiments(
        dataset=dataset,
        detector=detector,
        params=llm_params,
        seeds=range(42, 1042, 50)  # Same 20 seeds
    )

    # Statistical comparison
    improvement = compare_results(baseline_results, llm_results)

    return {
        "baseline": baseline_results,
        "llm_guided": llm_results,
        "improvement": improvement,
        "llm_suggestions": llm_params
    }
```

### 3.3 Phase 3 작업 목록 (Week 3-4)

**Week 3: LLM Parameter Advisor**
- [ ] Day 15: Parameter advisor prompt 설계
- [ ] Day 16: 3개 detector × 3개 dataset = 9가지 제안 생성
- [ ] Day 17: 제안 파라미터 검증 (합리성 체크)
- [ ] Day 18-20: LLM-guided 파라미터로 재실험 (20 seeds each)
- [ ] Day 21: Baseline vs LLM-guided 통계 비교

**Week 4: 평가 및 보고서**
- [ ] Day 22: 성능 개선 분석 (Wilcoxon test)
- [ ] Day 23: 실패 케이스 분석 (왜 개선 안 됐는가?)
- [ ] Day 24-26: Phase 3 보고서 작성
- [ ] Day 27-28: 전체 통합 및 Demo

**성공 기준**:
- LLM-guided params가 baseline보다 **평균 10% 이상 개선** (F1 또는 AUC-PR)
- 9개 케이스 중 **최소 6개에서 개선** (66% 성공률)
- 개선이 **통계적으로 유의미** (p<0.05)

---

## 📊 최종 성과물 (Deliverables)

### Phase 1 (완료 ✅)
- [x] `COMPREHENSIVE_EXPERIMENT_REPORT.md` (35 pages)
- [x] `runs/all_results.csv` (353 runs)
- [x] `runs/statistical_tests.json`, `bootstrap_ci.json`, `correlation_analysis.json`

### Phase 2 (Week 1-2)
- [ ] `experiments/llm_explainer.py` - 통합 설명 시스템
- [ ] `experiments/knowledge_base/` - SKAB, SMD 도메인 지식
- [ ] `PHASE2_LLM_EXPLANATION_REPORT.md` - 10개 설명 예시
- [ ] Demo: `scripts/demo_explanation.py`

### Phase 3 (Week 3-4)
- [ ] `experiments/llm_parameter_advisor.py` - Parameter suggestion system
- [ ] `PHASE3_LLM_OPTIMIZATION_REPORT.md` - Baseline vs LLM-guided 비교
- [ ] Demo: `scripts/demo_optimization.py`

### 최종 통합
- [ ] `FINAL_INTEGRATED_REPORT.md` - Phase 1-3 통합 보고서
  - ML 탐지 성능 (Phase 1)
  - LLM 해석 능력 (Phase 2)
  - LLM 최적화 효과 (Phase 3)
  - 결론: "LLM-guided approach가 제조 anomaly detection을 X% 개선"

---

## ⚠️ 기존 TODO.md와의 차이점

### 기존 TODO.md (학술 중심)
- RQ1: Frequency vs Time domain
- RQ2: Ensemble methods
- RQ3: Metric correlation
- RQ4: Cost sensitivity

### 새 TODO (실용 중심)
- **핵심**: ML + LLM 통합
- **차별점**:
  1. 이상치 해석 (LLM)
  2. **모델 해석 (LLM)** ← 새로운!
  3. 도메인 최적화 (LLM)

### 우선순위
1. **Phase 2-3 (LLM 통합)** - 연구의 핵심!
2. Phase 1 개선 (SpecCNN 수정 등) - 부차적

---

## 📅 타임라인 (4주)

| Week | Phase | 목표 | 성과물 |
|------|-------|------|--------|
| **1** | Phase 2.1 | 이상치 + 모델 해석 구현 | llm_explainer.py, 지식 베이스 |
| **2** | Phase 2.2 | 통합 설명 시스템, 검증 | Phase 2 보고서 + 10개 예시 |
| **3** | Phase 3.1 | LLM parameter advisor, 재실험 | llm_parameter_advisor.py |
| **4** | Phase 3.2 | 성능 비교, 최종 보고서 | Final integrated report |

---

## ✅ 다음 즉시 작업 (Next Immediate Actions)

**오늘 (Day 1)**:
1. [ ] OpenAI API 통합 (`experiments/llm_config.py`)
2. [ ] 첫 번째 anomaly 설명 생성 (1개 샘플)
3. [ ] SKAB 도메인 지식 작성 시작

**내일 (Day 2)**:
1. [ ] Feature importance 추출 코드
2. [ ] SHAP values 계산
3. [ ] ML 모델 해석 prompt 작성

**모레 (Day 3)**:
1. [ ] 3-in-1 통합 설명 시스템
2. [ ] 5개 anomaly 설명 생성
3. [ ] 설명 품질 평가

---

**연구 핵심 재확인**:
> 제조 공정 데이터에서 ML로 이상 탐지 + LLM이 (1) 이상치, (2) 모델, (3) 도메인 해석 + LLM이 도메인 지식으로 ML 최적화
