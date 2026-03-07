# Phase 1: 모든 Competitor의 논문 출처 및 검증 가능성 분석

작성일: 2026-03-07
작성자: Opus 4.6 (연구 보조)

---

## 종합 매핑 테이블

| # | 우리 모델명 | 논문 출처 | 논문의 M5 사용 | 논문의 실험 설정 | 논문의 보고 성능 | 재현 가능성 |
|---|-----------|---------|-------------|--------------|--------------|-----------|
| 1 | global_category_avg | M5 공식 벤치마크 (Makridakis et al. 2022) | ✅ 사용 | 30,490 시계열 × 28일, WRMSSE | Naive WRMSSE ~0.94 (비공식), ES_bu가 top benchmark | **재현 가능** — 공식 repository에 예측값 있음 |
| 2 | similar_item_avg | **특정 논문 없음** — 자체 설계 | N/A | N/A | N/A | 재현 불가 (원 논문 없음) |
| 3 | store_category_avg | **특정 논문 없음** — 자체 설계 | N/A | N/A | N/A | 재현 불가 (원 논문 없음) |
| 4 | seasonal_pattern | **특정 논문 없음** — 일반 개념 | N/A | N/A | N/A | 재현 불가 (원 논문 없음) |
| 5 | knn_analog | Van Steenbergen & Mes (2020) **개념 차용** | ❌ 미사용 | 5개 회사 데이터, MAE/RMSE | 논문 Table 참조 필요 | **부분 재현** — 논문 데이터 비공개 |
| 6 | lightgbm_static | M5 1위 솔루션 (YJ_STU) **변형** | ✅ (원본은 사용) | 30,490×28일, WRMSSE | 1위: 0.520, 공식 벤치마크: ~0.67 | **재현 가능** — 공식 repository |
| 7 | lightgbm_proxy_lags | **특정 논문 없음** — 자체 설계 | N/A | N/A | N/A | 재현 불가 (원 논문 없음) |
| 8 | deepar | Salinas et al. (2020) | ❌ M5 미사용 | ec, ec-sub, parts, electricity, traffic | ND, NRMSE, 0.5-risk, 0.9-risk | **재현 가능** — GluonTS 공식 구현 |
| 9 | llm_zero_shot | Gruver et al. (2023) **변형** | ❌ M5 미사용 | Darts(8개), Monash(30개), Informer | NLL, CRPS, MAE | **부분 재현** — 코드 공개, Darts 데이터 공개 |
| 10 | llm_similar_item | **자체 설계** | N/A | N/A | N/A | N/A |
| 11 | llm_aggregate | **자체 설계** | N/A | N/A | N/A | N/A |
| 12 | Track A (persona) | **자체 설계** | N/A | N/A | N/A | N/A |
| 13 | Track B (hidden states) | **자체 설계** | N/A | N/A | N/A | N/A |

---

## 모델별 상세 분석

### 1. global_category_avg — M5 공식 Naive Benchmark

**논문:** Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). "M5 accuracy competition: Results, findings, and conclusions." International Journal of Forecasting, 38(4), 1346-1364.

**논문의 실험 설정:**
- 데이터: M5 전체 (30,490 product-store 시계열, 10개 매장, 3개 주)
- 예측 horizon: 28일 (daily)
- 평가 지표: WRMSSE (12개 aggregation level의 가중 평균)
- 벤치마크 24개: Naive, sNaive, SES, CRO, ES_bu, ES_td, ESX 등

**논문의 보고 성능:**
- Naive: 비공식적으로 WRMSSE ~0.94 (한 참가자가 naive로 0.939를 보고)
- sNaive (seasonal naive): Naive 대비 11.5% 개선
- ES_bu (top benchmark): CRO 대비 29.9% 개선
- 1위 (YJ_STU): ES_bu 대비 22.4% 개선 → WRMSSE ~0.520

**우리 구현과의 차이:**
- 우리: "동일 카테고리 warm items의 평균"을 상수로 예측
- M5 Naive: "직전 28일의 값을 그대로 반복"
- **이 둘은 같은 방법이 아니다.** 우리의 global_category_avg는 M5의 어떤 공식 벤치마크와도 정확히 대응하지 않음. "카테고리 평균"은 우리가 자체 설계한 naive baseline임.

**재현 방법:** M5 공식 repository (github.com/Mcompetitions/M5-methods)에서 Naive, sNaive 벤치마크의 예측값을 다운로드 → 우리의 300 items × CA_1 subset에 해당하는 부분만 추출 → 동일 evaluation으로 WRMSSE 계산. 이 수치가 우리의 naive baseline과 비교 가능한 참조점이 됨.

---

### 2-3. similar_item_avg, store_category_avg — 자체 설계

**논문 출처: 없음.** 이 두 baseline은 특정 논문의 방법론이 아니라, 우리가 sanity check용으로 만든 단순 통계 baseline이다.

**문제:** 논문에서 이것들을 "competitor"로 제시하면 "너무 약한 baseline과 비교한 것 아니냐"는 비판을 받을 수 있음. 다만 "naive baseline"으로서 하한선 역할은 가능.

**대안:** M5 공식 Naive/sNaive 벤치마크를 대신 사용하면 공인된 baseline이 됨.

---

### 4. seasonal_pattern — 일반 개념 (특정 논문 없음)

**논문 출처: 특정 단일 논문 없음.** "카테고리별 계절 패턴을 적용"하는 것은 실무에서 널리 사용되는 일반 개념이지만, 특정 논문의 구체적 알고리즘을 구현한 것이 아님.

**문제:** 
- "seasonal pattern"이라는 이름만으로는 리뷰어가 어떤 방법인지 알 수 없음
- 구현의 정확성을 외부 참조로 검증할 방법이 없음

**대안:** M5 공식 벤치마크 중 sNaive (seasonal naive)가 이 개념에 가장 가까움. sNaive는 "1년 전 같은 기간의 값을 반복"하는 방법. 우리의 seasonal_pattern과 직접 비교 가능.

---

### 5. knn_analog — Van Steenbergen & Mes (2020) 개념 차용

**논문:** Van Steenbergen, R. M., & Mes, M. R. K. (2020). "Forecasting demand profiles of new products." Decision Support Systems, 139, 113401.

**논문의 실험 설정:**
- 데이터: 5개 회사의 실제 데이터 (비공개)
- 방법: DemandForest — 유사 상품의 수요 프로필을 특성 기반으로 결합
- Distance metric: 논문 내 확인 필요 (Euclidean일 가능성 높음)
- 평가 지표: MAE, RMSE, 여러 변형

**우리 구현과의 차이:**
- 우리: cosine similarity, one-hot(cat+dept) + sell_price, k=5
- 논문: DemandForest라는 별도의 프레임워크로, 단순 k-NN이 아님
- **우리의 knn_analog는 이 논문의 정확한 재현이 아니라, 일반적인 k-NN 개념을 구현한 것임**

**논문 데이터 공개 여부:** 비공개. 재현 불가능.

**더 적절한 참조 논문:**
- Hu et al. (2019) "Forecasting New Product Life Cycle Curves" — Dell 데이터, 비공개
- Singh et al. (2019) "Fashion Retail: Forecasting Demand for New Items" — 비공개 데이터
- cold-start에서 k-NN을 사용하는 표준 참조 논문으로 Amazon Forecast의 기술 문서가 있지만 학술 논문은 아님

**검증 방법:** 논문 데이터가 비공개이므로 직접 재현 불가. Leave-one-out 검증으로 대체해야 함.

---

### 6. lightgbm_static — M5 1위 솔루션 변형

**논문:** M5 1위 팀 (YJ_STU)의 방법론. 상세 설명은 Makridakis et al. (2022)에 있고, Kaggle discussion에도 공개됨.

**논문의 실험 설정:**
- 데이터: M5 전체 (30,490 시계열)
- 모델: LightGBM, 매장별 10개 모델 × recursive/non-recursive 앙상블
- Features: lag_7, lag_28, rolling_mean_7, rolling_mean_28, 가격, 캘린더, SNAP 등
- 평가: WRMSSE

**논문의 보고 성능:** WRMSSE 0.520 (전체 M5)

**우리 구현과의 근본적 차이:**
1. **우리는 lag features를 제거했음** — cold-start이므로 불가피하지만, 이것은 M5 1위의 핵심 feature를 뺀 것
2. **우리는 300 items × 1 store** — 원본은 30,490 × 10 stores
3. **우리는 주간 집계** — 원본은 일별

**이 차이 때문에 우리의 lightgbm_static은 M5 1위와 직접 비교 불가.** "LightGBM을 cold-start에 적용"하는 것은 우리의 자체 적응이며, 특정 논문의 방법론 재현이 아님.

**재현 방법:** exp007 (warm items에서 lag features 포함)으로 이미 부분 검증 완료 (WRMSSE 0.913). 하지만 M5 공식 repository의 공식 벤치마크와 300-item subset에서 직접 비교해야 확정적.

---

### 7. lightgbm_proxy_lags — 자체 설계

**논문 출처: 없음.** k-NN으로 찾은 유사 상품의 lag features를 proxy로 사용하는 것은 우리가 리뷰어 피드백을 반영해 설계한 변형. 특정 논문의 방법론이 아님.

---

### 8. deepar — Salinas et al. (2020)

**논문:** Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." International Journal of Forecasting, 36(3), 1181-1191.

**논문의 실험 설정:**
- 데이터셋 5개:
  - **ec** (e-commerce): Amazon 내부 데이터, ~500K 시계열, 비공개
  - **ec-sub**: ec의 subset, 비공개
  - **parts** (auto parts): 비공개
  - **electricity**: 370 고객 × 3년 시간당 소비량, **공개** (UCI ML Repository)
  - **traffic**: 963 차선 × 2년 시간당 점유율, **공개** (UCI ML Repository)
- 평가 지표: ND (Normalized Deviation), NRMSE, ρ0.5-risk, ρ0.9-risk
- 예측 horizon: electricity/traffic은 24시간 (hourly)

**논문의 보고 성능 (Table 2, electricity):**
- DeepAR: ND=0.070, NRMSE=0.082
- MatFact (baseline): ND=0.089, NRMSE=0.108

**논문의 보고 성능 (Table 2, traffic):**
- DeepAR: ND=0.129, NRMSE=0.172
- MatFact: ND=0.144, NRMSE=0.197

**우리 구현과의 차이:**
1. 우리: M5 데이터, weekly, cold-start (이력 없는 아이템에 category mean 초기화)
2. 원본: hourly electricity/traffic, 모든 시계열에 풍부한 이력 있음
3. 원본은 context_length가 수백~수천 time steps, 우리는 4주
4. **cold-start에서 DeepAR을 사용하는 것은 이 논문의 설계 의도와 다름**

**재현 방법:**
- **electricity 데이터셋으로 재현 가능.** UCI ML Repository에서 다운로드 → GluonTS의 DeepAR 공식 구현으로 학습 → ND/NRMSE 비교
- 이 재현이 성공하면 "우리의 GluonTS DeepAR 구현은 정상이고, cold-start에서 성능이 나쁜 것은 데이터/설정의 차이 때문"이라고 말할 수 있음

**데이터 접근:**
- electricity: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
- traffic: https://archive.ics.uci.edu/ml/datasets/PEMS-SF (또는 GluonTS에 내장 데이터셋으로 제공)

---

### 9. llm_zero_shot — Gruver et al. (2023) 변형

**논문:** Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). "Large Language Models Are Zero-Shot Time Series Forecasters." NeurIPS 2023.

**논문의 실험 설정:**
- 데이터: Darts (8개 univariate), Monash (30개), Informer 벤치마크
- LLM: GPT-3 (text-davinci-003), LLaMA-2 70B
- 방법: 시계열을 숫자 문자열로 인코딩, 특정 토큰화/스케일링 전략 (alpha=0.95, beta=0.3)
- 평가: NLL/D (Negative Log-Likelihood per Dimension), CRPS, MAE

**논문의 보고 성능 (Darts, MAE 기준):**
- LLMTime (GPT-3): 대부분 데이터셋에서 1위 또는 2위
- LLMTime > ARIMA, TCN, N-HiTS, N-BEATS (zero-shot에서)

**우리 구현과의 근본적 차이:**
1. **토큰화 전략:** 원 논문은 숫자를 space로 구분하고, 소수점 자릿수를 제어하는 specific한 방법 사용. 우리는 단순 프롬프트에 숫자를 넣음.
2. **스케일링:** 원 논문은 alpha/beta 파라미터로 데이터 스케일을 조정. 우리는 안 함.
3. **입력 형태:** 원 논문은 과거 시계열 자체를 입력. 우리는 유사 상품의 이력을 텍스트로 설명.
4. **LLM:** 원 논문은 GPT-3 (text-davinci-003), 우리는 GPT-4o-mini.
5. **우리의 llm_zero_shot은 LLMTime의 재현이 아니다.** 이름이 혼동을 줄 수 있음.

**재현 방법:**
- GitHub: https://github.com/ngruver/llmtime
- Darts 데이터셋은 공개 (GluonTS/darts 라이브러리에 내장)
- 원 논문의 코드를 그대로 실행하여 Darts 벤치마크 결과 재현 가능
- 다만 GPT-3 (text-davinci-003)는 현재 deprecated → GPT-4o-mini로 대체 시 결과가 달라질 수 있음

**데이터 접근:** Darts 라이브러리 (pip install darts) 내장

---

### 10-13. 자체 설계 모델들

llm_similar_item, llm_aggregate, Track A, Track B는 모두 우리 연구의 자체 설계 방법론이므로 외부 재현 대상이 아님.

---

## 핵심 문제 진단

### 문제 1: 대부분의 competitor가 특정 논문의 정확한 재현이 아님

13개 모델 중 **특정 논문의 방법론을 정확히 구현한 것은 하나도 없다.**

- lightgbm_static: M5 1위의 "변형" (lag features 제거)
- deepar: Salinas (2020)의 구현이지만 cold-start 적응은 원 논문에 없는 우리의 변형
- knn_analog: "k-NN 개념"을 구현했지만 특정 논문의 알고리즘이 아님
- llm_zero_shot: LLMTime의 "정신"만 차용, 토큰화/스케일링 등 핵심 방법론 미반영

### 문제 2: "개념적 아이디어"를 구현하면서 SOTA를 놓침

- k-NN analog: 단순 cosine similarity + weighted average인데, cold-start 수요예측의 SOTA는 더 정교한 방법(DemandForest, DDPFF, MetaEmb 등)
- DeepAR: 2017/2020년 모델. 2024-2025 기준으로 Chronos, TimesFM 등 foundation model이 SOTA
- LightGBM: 구현 자체는 맞지만 cold-start 적응에서 lag features를 빼는 것이 표준적인 cold-start 적응 방식인지 확인 안 됨

### 문제 3: 외부 검증이 가능한 모델이 극히 제한적

외부 데이터로 검증 가능한 것:
1. **LightGBM** — M5 공식 벤치마크와 비교 가능 (warm items subset)
2. **DeepAR** — electricity/traffic 데이터셋으로 재현 가능
3. **M5 Naive/sNaive** — 공식 예측값과 비교 가능

나머지는 전부 자체 설계이므로 Leave-one-out 같은 간접 검증만 가능.

---

## 구체적 재현 실험 계획

### 실험 V1: M5 공식 벤치마크 비교 (LightGBM + Naive)

**데이터:** github.com/Mcompetitions/M5-methods에서 다운로드
**절차:**
1. M5 공식 벤치마크 예측값 (Naive, sNaive, ES_bu) 다운로드
2. 우리의 CA_1 매장 300 warm items에 해당하는 예측값 추출
3. 공식 WRMSSE 코드로 이 subset의 WRMSSE 계산
4. 우리 exp007의 LightGBM WRMSSE (0.913)와 비교
5. 공식 Naive도 이 subset에서 0.9 근방이면 → subset이 원래 어려운 것 → 우리 구현 정상
6. 공식 Naive가 0.6인데 우리가 0.9이면 → 구현 문제 의심

**필요한 것:** M5 repository clone, Python/R evaluation code

### 실험 V2: DeepAR — Electricity 데이터셋 재현

**데이터:** UCI ML Repository electricity dataset (공개)
**절차:**
1. GluonTS의 DeepAR 공식 예제로 electricity 데이터셋 학습/평가
2. ND, NRMSE 계산
3. 원 논문의 결과 (ND=0.070, NRMSE=0.082)와 비교
4. 10% 이내 차이면 구현 정상

**필요한 것:** pip install gluonts, electricity 데이터 다운로드

### 실험 V3: LLMTime — Darts 벤치마크 재현 (선택적)

**데이터:** Darts 라이브러리 내장 데이터셋
**절차:**
1. github.com/ngruver/llmtime의 코드를 clone
2. GPT-4o-mini로 Darts 벤치마크 중 AirPassengers 등 2-3개 실행
3. MAE/CRPS 비교

**주의:** 원 논문은 GPT-3 (text-davinci-003)를 사용했고, 이 모델은 현재 deprecated. GPT-4o-mini로 대체하면 결과가 달라질 수 있으므로, "모델 차이로 인한 재현 한계"를 명시해야 함.

**필요한 것:** OpenAI API key, llmtime repository clone

---

## Competitor 적절성 진단

### 현재 competitor 중 논문 수준으로 적합한 것

| 모델 | 적합성 | 이유 |
|------|--------|------|
| LightGBM (with lag features, warm items) | ✅ 적합 | M5 1위 방법론 변형, 공식 벤치마크로 검증 가능 |
| DeepAR | △ 조건부 | 원 논문의 재현 검증 필요, cold-start 적용은 자체 변형임을 명시 |
| M5 공식 Naive/sNaive | ✅ 적합 | 공인된 벤치마크 |

### 현재 competitor 중 논문 수준으로 부적합한 것

| 모델 | 문제 | 개선 방안 |
|------|------|---------|
| knn_analog | 특정 논문의 알고리즘이 아님, SOTA 아님 | Van Steenbergen (2020)의 DemandForest를 정확히 구현하거나, 논문에서 "simple k-NN baseline"으로 명시 |
| seasonal_pattern | 특정 논문 없음 | M5 공식 sNaive로 대체 |
| lightgbm_proxy_lags | 자체 설계, 검증 불가 | 논문에서 "우리의 변형"으로 명시, ablation으로 제시 |
| llm_zero_shot | LLMTime과 다름, 토큰화/스케일링 미반영 | 이름을 "LLM Prompt-based Direct"로 변경, LLMTime과의 차이를 명시 |
| llm_aggregate | 실패 (MAE 408) | 결과에서 제외하거나, "프롬프트 설계 실패 사례"로 논문에 기록 |

---

## Claude Code에게 확인 요청 사항 (코드 레벨)

Claude Code가 확인해야 하지만 내가 확인할 수 없는 사항들:

### 1. 데이터 전처리 완전성 확인
```
cold_start.csv를 만들 때 원본 M5 데이터에서 어떤 컬럼이 제거되었는지 확인해줘.
원본 M5에는 item_id, dept_id, cat_id, store_id, state_id 외에
sell_price, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI 등이 있다.
이 중 cold_start.csv에 포함된 것과 제외된 것을 목록으로 보여줘.
특히 sell_price가 모든 cold items에 대해 존재하는지 확인해줘.
```

### 2. LLM 프롬프트에 전달된 정보 확인
```
Track A의 persona_predictor.py에서 LLM에게 전달하는 프롬프트의 전체 텍스트를 보여줘.
특히:
- 아이템 정보로 무엇을 전달하는지 (item_id? item_name? category? price?)
- 페르소나 정보로 무엇을 전달하는지 (전체 JSON? 일부 필드만?)
- 시간 정보로 무엇을 전달하는지 (주차? 날짜? SNAP 이벤트?)
```

### 3. DeepAR 구현 상세 확인
```
deepar_model.py에서:
- GluonTS의 어떤 클래스를 사용했는지 (DeepAREstimator?)
- context_length, prediction_length, freq 설정값
- cold items의 context를 어떻게 초기화했는지 (zero? category mean? 어떤 값?)
- static features로 무엇을 전달했는지
- 학습 epochs, learning rate 등 hyperparameters
```

### 4. 평가 지표 DirAcc 구현 확인
```
metrics.py에서 DirAcc 계산 로직을 보여줘.
특히:
- sign(pred[t] - pred[t-1]) == sign(actual[t] - actual[t-1])로 계산하는 건지?
- t-1이 없는 첫 주는 어떻게 처리하는지?
- pred나 actual이 0인 경우 sign은 어떻게 처리하는지?
- flat (변화 없음)을 match로 치는지?
```

---

## 요약: 현재 상황의 솔직한 평가

**좋은 점:**
- M5 데이터셋 자체는 표준적이고 잘 알려진 벤치마크
- LightGBM 구현은 부분 검증됨 (exp007)
- 평가 지표(MAE, DirAcc)는 표준적

**나쁜 점:**
- 13개 모델 중 특정 논문의 정확한 재현은 0개
- 대부분이 "개념 차용 + 자체 변형"으로, 외부 검증이 불가능
- SOTA 방법론과의 비교가 없음 (2024-2025 기준)
- cold-start 세팅에서의 벤치마크가 학계에 확립되어 있지 않아서, "이 비교가 공정한가?"에 대한 합의가 없음

**즉시 해야 할 것:**
1. M5 공식 벤치마크(Naive, sNaive, ES_bu) 예측값을 가져와서 우리 subset에서의 성능 확인 (실험 V1)
2. DeepAR의 electricity 데이터셋 재현 (실험 V2)
3. 각 competitor의 이름과 출처를 논문에서 정확히 명시할 수 있도록 정리
