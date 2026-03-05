# Competitor Baseline 구현 지시서

## 배경

현재 우리의 베이스라인은 naive 통계(GlobalMean, CategoryAverage, SimilarItemAverage)뿐이다.
논문에서 기존 방법론 대비 비교를 하려면 실제 선행 연구의 방법론을 구현해야 한다.

우리의 세팅: **Cold-start demand forecasting**
- Cold items 100개: 판매 이력 없음. 상품 속성(카테고리, 부서, 가격대 등)만 있음
- Warm items 300개: 판매 이력 있음 (학습 데이터)
- 예측 기간: 17주 (주간 판매량)
- 평가 지표: MAE, RMSE, WRMSSE, DirectionalAccuracy

핵심 제약: cold items는 자기 자신의 판매 이력이 전혀 없으므로,
시계열 자체를 입력으로 쓰는 방법(ARIMA, Prophet 등)은 직접 적용 불가.
대신 "유사 상품의 이력을 활용"하거나 "상품 속성 기반으로 예측"하는 방법을 써야 한다.

## 구현할 Competitor 5개 (LightGBM·LLM Direct 각각 변형 포함)

### Competitor 1: k-NN Analogous Forecasting
**논문 근거:** 가장 전통적인 cold-start 대응 방법. DDPFF (2025), Amazon Forecast 등에서 baseline으로 사용.

**방법:**
1. Warm items 300개의 상품 속성(category, department, price_range 등)을 feature vector로 인코딩
2. Cold item 각각에 대해, warm items 중 속성이 가장 유사한 k개를 찾음 (Euclidean distance or cosine similarity)
3. k개 유사 상품의 실제 주간 판매량을 거리 가중 평균으로 합산 → cold item의 예측값

**구현 세부사항:**
- k = 3, 5, 10 각각 실행 (k=5를 기본으로 보고)
- Feature: one-hot(category_id, dept_id, store_id) + numeric(sell_price 평균, sell_price std)
- Distance: cosine similarity
- 가중치: 1/distance (distance=0이면 해당 아이템 그대로 사용)

**검증 방법:**
- Warm items에서 leave-one-out으로 검증: warm item 1개를 빼고 나머지 299개로 예측 → MAE 계산
- 이 warm-item MAE가 우리 naive baseline보다 나은지 확인 (나아야 정상)

**파일:** `src/models/competitors/knn_analog.py`, `scripts/run_knn_analog.py`

---

### Competitor 2: LightGBM Cross-Learning (2 variants)
**논문 근거:** M5 competition 상위 솔루션에서 가장 많이 사용된 방법. Gradient boosting이 수요예측에서 SOTA급.

**공통 방법:**
1. Warm items 300개의 (item, week) 단위 데이터를 학습 데이터로 구성
2. Target: weekly_sales
3. LightGBM regressor 학습 (warm items) → Cold items에 동일 feature 구성 → 예측

---

#### 2-A: LightGBM (Static)
현재 계획대로. Cold-start에 완전히 충실한 버전 — cold item 자체의 이력을 일절 사용하지 않음.

**Feature engineering:**
- 정적 features: category_id, dept_id, store_id, sell_price (one-hot + numeric)
- 시간 features: week_of_year, month, is_snap (해당 주 SNAP 이벤트 여부)
- 집계 features: 해당 카테고리 warm item 평균 판매량, 해당 부서 warm item 평균 판매량
- (주의: cold item 자체의 lag features 사용 불가 — cold-start이므로)

#### 2-B: LightGBM (Proxy Lags)
k-NN에서 찾은 유사 상품 top-3의 rolling mean을 추가 feature로 포함. "유사 상품의 lag"를 cold item의 proxy로 활용.

**추가 Feature (2-A 위에 추가):**
- knn_top3_rolling_mean_7d: 유사 warm item 3개의 해당 주 이전 7일 판매량 rolling mean 가중 평균
- knn_top3_rolling_mean_28d: 동일, 28일 rolling mean
- knn_top3_distance: k-NN 거리 (유사도의 신뢰 수준 표현)

**구현 순서:** 2-A 먼저 실행해서 k-NN 결과가 있어야 2-B의 proxy lag feature를 생성할 수 있음.

---

**공통 구현 세부사항:**
- LightGBM params: objective='tweedie', tweedie_variance_power=1.5 (zero-inflation 대응)
- 또는 objective='regression_l1' (MAE 직접 최적화) — 두 가지 실행 후 나은 쪽 보고
- Hyperparameter: num_leaves=31, learning_rate=0.05, n_estimators=500, early_stopping_rounds=50
- Validation: warm items 중 20%를 hold-out으로 사용

**검증 방법:**
- Warm item hold-out MAE (validation set)
- Feature importance 출력 → 어떤 feature가 가장 중요한지 확인
- Cold item 예측 분포가 합리적인지 확인 (음수 없는지, 범위가 warm items와 유사한지)
- 2-B가 2-A보다 나아야 proxy lag 전략의 유효성 확인

**파일:** `src/models/competitors/lightgbm_cross.py`, `scripts/run_lightgbm.py`
- 하나의 파일에서 `--variant static | proxy-lags` 인수로 선택

---

### Competitor 3: LLM Direct Prediction (LLMTime-style, 3 variants)
**논문 근거:** Gruver et al. (2023) "LLMs Are Zero-Shot Time Series Forecasters". LLM에 시계열을 텍스트로 주고 직접 예측.

세 가지 조건을 비교해서 "LLM에 무엇을 주는가"에 따른 성능 차이를 측정.
모두 GPT-4o-mini, 출력 형식은 17-week JSON array로 통일.

---

#### 3-1: Zero-shot Direct
**설명:** 상품 메타데이터(속성)만 주고 LLM에게 직접 주간 판매량 예측 요청. 유사 상품 이력 없음.

**Prompt 구조:**
```
System: You are a demand forecasting expert for a large retail store.

User:
A new product is launching with no prior sales history.
Product attributes: category=FOODS, dept=FOODS_3, store=CA_1, price=$2.60

Predict the weekly sales for this product over the next 17 weeks.
Return only a JSON array of 17 numbers (weekly unit sales). Example: [3, 5, 2, 4, ...]
```

**API 호출:** 100 items × 1 call = **100 calls** (~$0.01)
**의의:** 가장 순수한 LLM prior knowledge 테스트. Track A(페르소나)와 비교해서 "시뮬레이션 없이 직접 묻기"의 한계를 확인.

---

#### 3-2: Similar Item Direct (기존 계획)
**설명:** k-NN으로 찾은 유사 warm item 3개의 이력을 텍스트로 제공 후 예측 요청.

**Prompt 구조:**
```
System: You are a demand forecasting expert. Given sales histories of similar products,
predict weekly sales for a new product.

User:
Similar products' recent 17-week sales:
Product A (category: FOODS, dept: FOODS_3, price: $2.50): [3, 5, 2, 4, 6, 3, 5, ...]
Product B (category: FOODS, dept: FOODS_3, price: $2.80): [2, 4, 1, 3, 5, 2, 4, ...]
Product C (category: FOODS, dept: FOODS_3, price: $2.30): [4, 6, 3, 5, 7, 4, 6, ...]

New product attributes: category=FOODS, dept=FOODS_3, price=$2.60

Predict the next 17 weeks of sales as a JSON array of numbers.
```

**API 호출:** 100 items × 1 call = **100 calls** (~$0.02, 프롬프트가 길어 3-1보다 비쌈)
**의의:** 시계열 이력 정보가 LLM 예측에 실제로 도움이 되는가? k-NN analog(Competitor 1)보다 LLM 활용이 나은가?

---

#### 3-3: Aggregate Prompt
**설명:** 개별 페르소나 없이, "50명의 구매 행동을 종합적으로 시뮬레이션해서 매장 수준 총 판매량 예측"을 단일 호출로 요청.

**Prompt 구조:**
```
System: You are a demand forecasting expert with expertise in consumer behavior simulation.

User:
A new product is launching in a large retail store (CA_1, Walmart-scale).
Product attributes: category=FOODS, dept=FOODS_3, store=CA_1, price=$2.60

Simulate the aggregate purchasing behavior of approximately 50 typical shoppers
who visit this store regularly. Based on their collective purchasing decisions,
predict the total weekly store-level sales for this product over the next 17 weeks.

Consider: shopping frequency, price sensitivity, category preferences, seasonal patterns.
Return only a JSON array of 17 numbers (total weekly unit sales at store level).
```

**API 호출:** 100 items × 1 call = **100 calls** (~$0.01)
**의의:** Track A(50명 개별 시뮬레이션 후 합산)와 직접 비교. "분해 후 합산 vs 한 번에 aggregate" — Li et al. (2025)의 분해 효과 검증.

---

**공통 구현 세부사항:**
- API: GPT-4o-mini
- temperature: 0.0 (재현성)
- max_tokens: 300
- JSON 파싱 실패 시: 최대 3회 재시도, 그래도 실패하면 카테고리 평균으로 대체 (실패율 기록)
- 파싱 성공률도 결과에 포함

**검증 방법:**
- JSON 파싱 성공률 (각 변형별)
- 예측값 범위 (음수 없는지, 극단값 없는지)
- 3-1 < 3-2 < Track A 순서로 나오면 "이력 정보 + 페르소나 분해" 전략이 점진적으로 효과적임을 입증
- 3-3 vs Track A: 분해(decomposition)의 효과

**파일:** `src/models/competitors/llm_direct.py`, `scripts/run_llm_direct.py`
- `--variant zero-shot | similar-item | aggregate` 인수로 선택

---

### Competitor 4: Category Average + Seasonal Pattern (Enhanced Naive) — 구현 1순위
**논문 근거:** 실무에서 가장 많이 쓰이는 방법. Van Steenbergen & Mes (2020), industry practice.

**방법:**
1. Warm items 중 동일 카테고리의 주간 판매 패턴을 추출
2. 각 카테고리별로 "평균 주간 프로필"을 계산 (어떤 주에 많이/적게 팔리는지)
3. Cold item의 카테고리에 해당하는 프로필로 예측

**구현 세부사항:**
- 카테고리: category_id (FOODS, HOBBIES, HOUSEHOLD) × dept_id 조합 = 약 7개 그룹
- 각 그룹 내 warm items의 주간 판매량 평균 → 주간 프로필
- 주간 프로필의 총합을 category_avg baseline의 level로 스케일
- 이건 기존 category_avg baseline에 "주간 변동 패턴"을 추가한 것

**검증 방법:**
- 기존 GlobalCategoryAverage baseline보다 반드시 나아야 함 (주간 변동을 추가했으므로)
- 만약 안 나아지면 데이터에 주간 계절성이 없다는 뜻 → 이것 자체가 발견

**파일:** `src/models/competitors/seasonal_pattern.py`, `scripts/run_seasonal_pattern.py`

---

### Competitor 5: DeepAR (Deep Probabilistic Forecasting)
**논문 근거:** Salinas et al. (2020) "DeepAR: Probabilistic forecasting with autoregressive recurrent networks". Amazon 공식 채택 방법론. Cold-start를 static feature로 처리하는 접근.

**방법:**
1. Warm items 300개의 일별 판매 시계열로 DeepAR 모델 학습
2. Static features: category_id, dept_id, store_id, sell_price (범주형 embedding + numeric)
3. Time-varying covariates: day_of_week, week_of_year, is_snap, sell_price (주별)
4. Cold items에 대해 static feature만 주고 예측 (이력 없이 static covariate만으로 초기화)

**구현:**
- `gluonts` 또는 `pytorch-forecasting` 라이브러리 사용
  - GluonTS: `gluonts.model.deepar.DeepAREstimator`
  - PyTorch Forecasting: `TemporalFusionTransformer` (DeepAR 대안으로도 가능)
- 권장: GluonTS + MXNet 또는 PyTorch backend
- 학습: warm items 300개, context_length=28 (4주), prediction_length=7 (1주)
- Cold item 예측: 각 cold item의 static feature만 세팅, context는 0-패딩 또는 카테고리 평균으로 초기화

**구현 세부사항:**
```python
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

estimator = DeepAREstimator(
    freq="W",
    context_length=4,       # 4주 컨텍스트
    prediction_length=1,    # 1주 예측 (17번 반복)
    num_layers=2,
    hidden_size=40,
    dropout_rate=0.1,
    use_feat_static_cat=True,
    cardinality=[3, 7, 10],  # category, dept, store
    trainer=Trainer(epochs=50, learning_rate=1e-3),
)
```

**한계 및 기대:**
- Warm items 300개는 DeepAR 학습에 다소 적을 수 있음 → 성능이 LightGBM보다 낮을 가능성 있음
- 그러나 "확률적 예측" 능력 자체는 논문에서 언급 가치 있음
- Cold item에 이력이 없어 context=0으로 시작하는 것의 한계도 분석 가치 있음
- **환경 구성이 어려우면 마지막에 별도로 처리해도 됨** (pip install gluonts 충돌 가능성)

**파일:** `src/models/competitors/deepar_model.py`, `scripts/run_deepar.py`

---

## 구현 순서

1. **Competitor 4 (Seasonal Pattern)** — 가장 단순, API 불필요
2. **Competitor 1 (k-NN Analog)** — API 불필요; Competitor 2-B와 3-2에서 k-NN 결과 재사용
3. **Competitor 2-A (LightGBM Static)** → **2-B (Proxy Lags)** 순서로 (2-A 먼저 실행해야 2-B feature 생성 가능)
4. **Competitor 3-1 (Zero-shot)** → **3-2 (Similar Item)** → **3-3 (Aggregate)** 순서로 (API 필요)
5. **Competitor 5 (DeepAR)** — 환경 구성 별도. 라이브러리 충돌 시 가상환경 분리

> **총 API 호출 예상:**
> 3-1: 100 calls, 3-2: 100 calls, 3-3: 100 calls → 합계 300 calls (~$0.04~0.06)

## 결과 저장 구조

```
experiments/
  exp006_competitors/
    seasonal_pattern/
      predictions/        # (100 items × 17 weeks)
      results/            # MAE, RMSE, WRMSSE, DirAcc
      metadata.json
    knn_analog/
      predictions/
      results/
      metadata.json       # k값, distance metric 등
    lightgbm_static/
      predictions/
      results/
      feature_importance.csv
      metadata.json
    lightgbm_proxy_lags/
      predictions/
      results/
      feature_importance.csv
      metadata.json
    llm_zero_shot/
      predictions/
      results/
      metadata.json       # 파싱 성공률, API 호출 수 포함
    llm_similar_item/
      predictions/
      results/
      metadata.json
    llm_aggregate/
      predictions/
      results/
      metadata.json
    deepar/
      predictions/
      results/
      metadata.json       # 학습 epoch, loss curve 포함
    comparison_table.csv  # 모든 방법 + 기존 baseline + Track A 통합 비교
```

## 평가 스크립트

모든 competitor의 예측을 동일한 평가 파이프라인으로 평가해야 한다.
기존 `scripts/evaluate_baselines.py`를 확장하거나 `scripts/evaluate_all.py`를 새로 만들어서:
- 입력: predictions/ 폴더의 CSV
- 출력: MAE, RMSE, WRMSSE, DirectionalAccuracy
- 모든 모델을 하나의 비교 테이블로 출력

## 검증 체크리스트 (구현 정확성 확인)

각 competitor 구현 후 반드시 확인:

1. **데이터 누수 확인**: cold item의 실제 판매 데이터가 학습/feature에 절대 포함되지 않았는지
   - `assert cold_item_ids not in training_data.item_id`
2. **예측 shape 확인**: (100 items × 17 weeks) = 1,700개 예측값
3. **예측 범위 확인**: 음수가 없는지, 극단값이 없는지
   - `assert predictions.min() >= 0`
   - `print(predictions.describe())`
4. **Warm item 검증** (Competitor 1, 2): leave-one-out 또는 hold-out으로 warm item에서의 성능 확인
5. **기존 baseline 대비 확인**: 최소한 random baseline보다는 나아야 함
6. **재현성**: random seed 고정 (42)

위 체크리스트를 각 competitor 실행 후 출력해줘.

## 최종 비교 테이블 포맷

```
| Model                        | MAE    | RMSE   | WRMSSE | DirAcc | Note                     |
|------------------------------|--------|--------|--------|--------|--------------------------|
| GlobalMean                   | 1.xx   | 5.xx   | 2.xx   | 0.258  | Naive                    |
| GlobalCategoryAvg            | 1.64   | 5.13   | 2.98   | 0.258  | Naive                    |
| SimilarItemAvg               | 1.57   | 5.17   | 2.99   | 0.258  | Naive                    |
| SeasonalPattern              | ?.??   | ?.??   | ?.??   | ?.???  | Enhanced Naive           |
| k-NN Analog (k=5)            | ?.??   | ?.??   | ?.??   | ?.???  | Traditional              |
| LightGBM (Static)            | ?.??   | ?.??   | ?.??   | ?.???  | ML                       |
| LightGBM (Proxy Lags)        | ?.??   | ?.??   | ?.??   | ?.???  | ML + k-NN feature        |
| LLM Zero-shot Direct (3-1)   | ?.??   | ?.??   | ?.??   | ?.???  | LLM, no history          |
| LLM Similar Item (3-2)       | ?.??   | ?.??   | ?.??   | ?.???  | LLM + analog history     |
| LLM Aggregate (3-3)          | ?.??   | ?.??   | ?.??   | ?.???  | LLM, 1-shot aggregate    |
| DeepAR                       | ?.??   | ?.??   | ?.??   | ?.???  | Deep Probabilistic       |
| Track A Calibrated           | 1.60   | 5.17   | 2.99   | 0.409  | Ours (persona, Cond. A)  |
| Track B (Qwen 32B)           | TBD    | TBD    | TBD    | TBD    | Ours (2-stage)           |
```
