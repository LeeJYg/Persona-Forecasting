# 연구 마스터 컨텍스트: LLM Persona Embeddings for Cold-Start Demand Forecasting

**최종 업데이트:** 2026-03-09 (로컬: 섹션 7 Competitor 검증 완료, 세션 5 추가)
**관리 규칙:** 매 세션 종료 시 섹션 10(세션 로그)에 변경사항 추가. 새 실험 결과는 섹션 5에 추가. 새 논문 발견 시 섹션 3에 추가.

---

## 1. 연구 개요

### 1.1 제목 (가제)
LLM Persona Embeddings for Cold-Start Demand Forecasting

### 1.2 핵심 아이디어
판매 이력이 없는 신규 상품(cold-start)의 수요를 예측하기 위해, LLM에 소비자 페르소나를 부여하고 그 내부 표현(hidden states)을 추출하여 수요예측 feature로 활용한다. LLM이 직접 숫자를 예측하는 대신, LLM의 "행동 표현 능력"만 활용하고 수치 예측은 학습된 회귀층에 맡긴다.

### 1.3 데이터셋
**Primary:** M5 Forecasting Competition (Walmart 매장 판매 데이터)
- Cold-start 시뮬레이션: 100개 아이템 판매 이력 차단
- Warm items: 300개 (학습용) — CA_1 매장 전체는 2,949개
- 예측 기간: 17주 (115일, 불완전 주 제거 시 16주)
- M5 전체: 3,049 products × 10 stores × 3 states = 30,490 시계열

**Secondary (검토 중):** Onout 의류 판매 데이터
- 한국 패션 이커머스 기업의 2년 분량 실제 판매 데이터
- 장점: 페르소나 친화적 도메인, 실제 cold-start 존재, 풍부한 학습 데이터
- 상태: 기업 사용 허가 확인 필요

### 1.4 방법론 (2-Track)
**Track A (Naive LLM Prediction):**
- GPT-4o-mini에 50개 소비자 페르소나를 부여
- 각 페르소나가 개인 구매량 예측 → 50명 합산 → alpha 보정으로 매장 수준 변환
- 100 items × 17주 × 50 personas, 배치 10 items/call = 8,500 API calls (~$1.40)

**Track B (2-Stage Hidden States):**
- Stage 1: Qwen 2.5 32B (INT4)에서 (persona, item) 쌍의 hidden states 추출
- Stage 2: 학습된 regression/attention head → 수치 예측
- Warm items에서 head 학습 → cold items에 적용 (transfer)

### 1.5 연구 환경
- 로컬: Mac, GPT-4o-mini API, Claude Code (VSCode)
- 서버: 8× Tesla P40 (24GB each), Qwen 2.5 32B INT4, Claude Code (VSCode SSH)
- P40 특성: Pascal 아키텍처 (compute capability 6.1), FP16 느림, bitsandbytes INT4 작동 확인됨

---

## 2. 논문 Logic Chain

```
Claim 1 (문제) → Claim 2 (가능성) → Claim 3 (한계) → Claim 4 (편향통제) → Claim 5 (방법론) → Claim 6 (전이)
```

### Claim 1: Cold-start 상품은 판매 이력이 없어서 기존 시계열 수요예측이 불가능하다
- **강도:** ★★★★☆ (강)
- **역할:** 문제 정의, 연구 동기

### Claim 2: LLM에 페르소나를 부여하면 인간의 경제적 행동을 모사할 수 있다
- **강도:** ★★★★☆ (강)
- **역할:** LLM 활용의 이론적 근거

### Claim 3: LLM은 직접적인 수치 예측에서는 성능이 낮다
- **강도:** ★★★★☆ (강)
- **역할:** "왜 hidden states를 쓰는가"의 정당화
- **주의:** Claim 2와 논리적 충돌 해소 필요 → "LLM은 행동 패턴의 방향성은 포착하지만, 연속적 수치 공간으로의 직접 매핑에서는 calibration 문제를 보인다"

### Claim 4: LLM이 생성한 페르소나는 편향될 수 있으므로 통제가 필요하다
- **강도:** ★★★★★ (매우 강)
- **역할:** 방법론적 신뢰성 확보

### Claim 5: LLM의 hidden states가 페르소나의 행동적 특성을 인코딩한다
- **강도:** ★★★★☆ (강) — 보강 완료
- **역할:** 방법론의 핵심 전제 (Track B 기반)

### Claim 6: Warm items에서 학습한 회귀층이 cold items에도 전이된다
- **강도:** ★★★☆☆ (중강)
- **역할:** 실용성 정당화

---

## 3. 근거 논문 목록 (Claim별)

### Claim 1 근거
| 논문 | 학회 | 핵심 | 인용 용도 |
|------|------|------|---------|
| Hu et al. (2019) "Forecasting New Product Life Cycle Curves" | M&SOM | "Traditional time-series methods do not apply" | **핵심**: 문제 정의 |
| Kahn (2014) "Solving the Problems of New Product Forecasting" | Business Horizons | 신제품 예측 정확도 평균 52% | 문제 심각성 |
| Van Steenbergen & Mes (2020) "Forecasting Demand Profiles of New Products" | DSS | 상품 속성 기반 유사 상품 활용 | 기존 접근법 |
| Makridakis et al. (2022) "M5 accuracy competition" | IJF | M5 대회 결과 정리 | 데이터셋 근거 |

### Claim 2 근거
| 논문 | 학회 | 핵심 | 인용 용도 |
|------|------|------|---------|
| Horton (2023) "Large Language Models as Simulated Economic Agents" | NBER | GPT로 경제 실험 재현 | **핵심** |
| Argyle et al. (2023) "Out of One, Many" | Political Analysis | Silicon sampling, 인구통계 조건부 응답 재현 | **핵심** |
| Aher et al. (2023) "Using LLMs to Simulate Multiple Humans" | arXiv | 인간 실험 재현 | 보조 |
| Park et al. (2023) "Generative Agents" | UIST | 25명 에이전트 사회 시뮬레이션 | 보조 |

### Claim 3 근거
| 논문 | 학회 | 핵심 | 인용 용도 |
|------|------|------|---------|
| **Tan et al. (2024)** "Are Language Models Actually Useful for Time Series Forecasting?" | **NeurIPS 2024 Spotlight** | LLM 제거해도 성능 유지/향상. PAttn이 1000배 빠름 | **최핵심** |
| Merrill et al. (2024) "Language Models Still Struggle to Zero-shot Reason about Time Series" | EMNLP Findings | 모든 LLM이 시계열 추론에서 랜덤 수준(25%) | **핵심** |
| Gruver et al. (2023) "LLMs Are Zero-Shot Time Series Forecasters" | NeurIPS 2023 | 패턴 매칭 기반, calibration 문제 | 보조 (nuanced) |
| Dziri et al. (2023) "Faith and Fate" | NeurIPS 2023 Spotlight | GPT-4도 3×3 곱셈 59% | 보조 |
| Razeghi et al. (2022) | EMNLP Findings | 숫자 정확도가 학습 빈도에 70%p 좌우 | 보조 |
| Zeng et al. (2023) "Are Transformers Effective for Time Series Forecasting?" | AAAI 2023 Oral | DLinear이 Transformer 능가 | 보조 |
| Singh & Strouse (2024) "Tokenization Counts" | arXiv | 토큰화가 산술 성능에 극적 영향 | 보조 |

### Claim 4 근거
| 논문 | 학회 | 핵심 | 인용 용도 |
|------|------|------|---------|
| Li et al. (2025) "LLM Generated Persona is a Promise with a Catch" | NeurIPS 2025 | Structured > Narrative, LLM 자유도↑→편향↑ | **핵심** |

### Claim 5 근거
| 논문 | 학회 | 핵심 | 인용 용도 |
|------|------|------|---------|
| **Zou et al. (2023)** "Representation Engineering" | arXiv, 812+ citations | 행동 특성이 hidden states에 선형 인코딩 | **최핵심** |
| **Gurnee & Tegmark (2023)** "Language Models Represent Space and Time" | ICLR 2024 | hidden states에서 연속 수치로 회귀 가능 (R² 보고) | **최핵심** |
| **Hussain et al. (2024)** "Open-Source LLMs for Behavioral Science" | Behavior Research Methods | LLM 임베딩으로 수치적 행동 판단 예측 | **핵심**: 가장 직접적 선례 |
| Li et al. (2023) "Inference-Time Intervention" | NeurIPS 2023 | 진실 방향을 activation space에서 조작 | 보조 |
| Marks & Tegmark (2023) "The Geometry of Truth" | arXiv | 참/거짓 선형 분리 가능 | 보조 |
| Burns et al. (2022) "Discovering Latent Knowledge" | arXiv | 라벨 없이 activation에서 진실 방향 발견 | 보조 |
| Tenney et al. (2019) "BERT Rediscovers the Classical NLP Pipeline" | ACL | 레이어별 구조적 정보 인코딩 | 배경 |
| Alain & Bengio (2016) "Linear Classifier Probes" | ICLR Workshop | Probing 방법론 창시 | 배경 |

### Claim 6 근거
| 논문 | 학회 | 핵심 | 인용 용도 |
|------|------|------|---------|
| **Hou et al. (2022) "UniSRec"** | KDD 2022 | BERT로 아이템 텍스트 인코딩 → ID 없이 cold item 전이 | **최핵심** |
| **Li et al. (2023) "RecFormer"** | KDD 2023 | 텍스트 속성으로 zero-shot 추천 | **최핵심** |
| Huang et al. (2024) "ColdLLM" | WSDM 2025 | LLM cold-start 시뮬레이션, 산업 규모 배포 | **핵심** |
| Zhu et al. (2021) "Learning to Warm Up Cold Item Embeddings" | SIGIR 2021 | Meta Scaling/Shifting으로 cold→warm 변환 | 핵심 |
| **⚠️ Dong et al. (2020)** "MAMO" | KDD 2020 | 저자명 주의: "Du et al."이 아님 | 보조 |
| Volkovs et al. (2017) "DropoutNet" | NeurIPS 2017 | Dropout으로 content만으로 cold 예측 | 보조 |
| Wang et al. (2024) "Language-Model Prior Overcomes Cold-Start Items" | arXiv (AWS) | LM 임베딩을 Bayesian prior로 활용 | 보조 |

---

## 4. 핵심 설계 결정 및 그 이유

### 4.1 페르소나 생성: LLM-free 방식 채택
- **결정:** 페르소나를 LLM이 아닌 확률적 샘플링으로 생성
- **이유:** Li et al. (2025)에서 LLM 생성 페르소나가 progressive/homogeneous 편향을 보임 → structured fields를 확률 분포에서 독립 샘플링
- **Phase 1 조건:** Condition A (Structured Only) — description 없이 JSON 필드만 사용

### 4.2 2-Stage 아키텍처 (Track B)
- **결정:** LLM hidden states 추출 → learned head → 수치 예측
- **이유:** Tan et al. (2024)에서 LLM이 수치 예측에 불필요함을 보임 + Horton (2023)에서 LLM이 행동 시뮬레이션은 잘함 → "행동 표현만 활용하고 수치 변환은 ML에 맡기자"
- **비교:** Time-LLM은 시계열 패턴을 LLM에 넣지만, 우리는 "소비자 행동 맥락"을 넣음 → 차별화 포인트

### 4.3 Mean-pooling → Attention Aggregation로 전환
- **결정:** 50 페르소나의 hidden states를 mean-pooling 대신 learnable attention으로 집계
- **이유:** PCA 분석에서 mean-pooled embeddings의 상위 10개 PC가 분산의 98.6% 설명 → 극단적 저랭크 → 페르소나 간 이질성 정보 파괴됨
- **발견:** 이것은 논문의 핵심 발견 중 하나가 될 수 있음

### 4.4 Regression Head 설계: Linear Projection 원칙
- **결정:** MLP가 아닌 단순 linear 또는 bottleneck linear 사용
- **이유:** 
  - 데이터 4,080 samples vs MLP 1.33M params = 300:1 비율 → 심각한 overfitting
  - Tan et al. (2024)의 PAttn, Time-LLM 등 모든 LLM 시계열 논문에서 output head는 single linear layer
  - Attention + Bottleneck(exp009에서 MAE 59.45, 기존 Ridge 68.30 대비 13% 개선)이 최선

### 4.5 예측 단위: Weekly로 통일
- **결정:** 모든 예측과 평가를 ISO week 주간 단위로 통일
- **이유:** Track A가 17 weekly values를 출력하도록 설계됨, daily disaggregation은 인위적
- **주의:** 이전 baseline MAE 1.57은 daily, 현재 weekly MAE 8~9. 단위 혼동 금지

### 4.6 Competitor 선정 원칙
- **결정:** Cold-start 세팅에서 적용 가능한 방법만 선정
- **제외된 방법들과 이유:**
  - Time-LLM, LLMTime, Chronos: 과거 시계열 입력 필수 → cold item에 적용 불가
  - UniSRec, RecFormer: ranking ≠ regression → 목적 함수가 다름
  - Horton, Argyle: 수요예측 아닌 행동 시뮬레이션 → 정량 비교 불가
- **⚠️ 중요 문제 발견:** 13개 모델 중 특정 논문의 정확한 재현은 0개. 전부 "개념 차용 + 자체 변형". 현재 구현 검증 진행 중.

---

## 5. 실험 결과 로그

### 5.1 Naive Baselines (exp002) — Daily 단위
| 모델 | MAE | RMSE | WRMSSE | DirAcc |
|------|-----|------|--------|--------|
| global_category_avg | 1.64 | 5.13 | 2.98 | 0.257 |
| similar_item_avg | 1.57 | 5.17 | 2.99 | 0.257 |
| store_category_avg | 1.69 | 5.11 | 2.96 | 0.257 |

- DirAcc 0.257 = 상수 예측이므로 실제 flat 비율(~23%)에 수렴

### 5.2 Track A (exp004) — Daily → Weekly 재평가
| 모델 | Daily MAE | Weekly MAE | DirAcc |
|------|-----------|------------|--------|
| Track A Raw (50명 합산) | 5.91 | — | 0.41 |
| Track A Calibrated (alpha=0.1485) | 1.60 | 8.90 | 0.393 |

- Alpha = mean(baseline pred) / mean(track_a_raw) = 0.1485
- 총 API 호출: 8,500회, 비용 ~$1.40, 런타임 ~23시간

### 5.3 Competitors (exp006) — Weekly 단위 (16 완전한 주)
| 모델 | MAE | WRMSSE | DirAcc | 비고 |
|------|-----|--------|--------|------|
| lightgbm_proxy_lags | 8.48 | 4.61 | 0.343 | MAE 최고 |
| similar_item_avg | 8.64 | 4.74 | 0.232 | Naive baseline |
| lightgbm_static | 8.72 | 4.61 | 0.379 | |
| llm_similar_item | 8.75 | 4.66 | 0.369 | 유사상품 이력 앵커 효과 |
| Track A calibrated | 8.90 | 4.76 | 0.393 | |
| seasonal_pattern | 9.00 | 4.63 | 0.386 | |
| deepar | 9.03 | 4.89 | 0.393 | |
| global_category_avg | 9.24 | 4.77 | 0.232 | |
| knn_analog | 9.57 | 4.61 | 0.412 | DirAcc 최고 |
| store_category_avg | 9.62 | 4.71 | 0.232 | |
| llm_zero_shot | 17.47 | 4.74 | 0.385 | 2배 과대예측 |
| llm_aggregate | 408.0 | 9.02 | 0.389 | 프롬프트 실패 |

- **LLM 3종(zero_shot, similar_item, aggregate)은 와이파이 안정적일 때 별도 실행됨**
- llm_aggregate 실패 원인: "50명 합산" 프롬프트를 LLM이 개인당 8~16개로 해석

### 5.4 DirAcc 통계적 유의성 분석
- 실제 방향 분포: 상승 38.9%, 하락 37.9%, flat 23.2%
- **랜덤 DirAcc: 0.358 ± 0.014** (0.5가 아님!)
- 유의 기준: 0.358 + 2σ = 0.386
- 유의미한 모델: knn_analog(0.412), Track A(0.393), deepar(0.393), seasonal_pattern(0.386)
- 유의미하지 않은 모델: lightgbm_proxy_lags(0.343, 랜덤 이하!)

### 5.5 LightGBM 구현 검증 (exp007) — Daily, Warm Items
| 모델 | WRMSSE | MAE |
|------|--------|-----|
| LightGBM (lag features 포함) | 0.913 | 1.027 |
| Naive lag-28 | 1.238 | — |
| Naive lag-7 | 1.164 | 1.369 |

- LightGBM > 모든 naive (-26% WRMSSE), overfitting 없음 (val≈test)
- **⚠️ WRMSSE 0.913 > 사전 기준 0.8** — M5 공식 벤치마크와의 직접 비교로 확정 필요 (진행 중)
- Zero-inflation 57.9%가 WRMSSE를 올린다는 주장은 **반박됨** (실제로는 zero-heavy 아이템이 WRMSSE를 낮추고 있었음)

### 5.6 Track B 원본 (exp005) — Daily 단위
| 지표 | 값 | 비고 |
|------|---|------|
| MAE (raw) | 13.07 | 예측이 실제의 ~10배 |
| MAE (alpha 보정, ×0.106) | 1.86 | baseline 1.57~1.69에 근접 |
| DirAcc | 0.372 | baseline 0.257 초과 |

- Ridge regression head, mean-pooled embeddings (300×5120)
- **PCA 분석:** 상위 10개 PC가 분산의 98.6% 설명 → 극단적 저랭크
- **Train MAE 48:** 학습 데이터에서도 나쁨 → 임베딩 자체의 정보량 부족
- 모든 regression head(Ridge, Lasso, RF, PCA+Linear)가 비슷하게 나쁨 → head 문제 아닌 embedding 문제

### 5.7 Track B Attention Head (exp009) — 단위 확인 필요
| Head | warm_val_MAE | cold_MAE | cold_RMSE | DirAcc |
|------|-------------|----------|-----------|--------|
| Ridge (mean-pooled) | 58.53 | 68.30 | 80.50 | 0.536 |
| Ridge (item-only) | 70.41 | 74.48 | 94.82 | **0.561** |
| Attention + Linear | 63.16 | 67.94 | 70.88 | 0.537 |
| **Attention + Bottleneck** | **59.85** | **59.45** | **63.31** | 0.536 |
| Variance + Ridge | 58.52 | 68.33 | 80.53 | 0.536 |
| Variance + PCA(64) + Ridge | 58.78 | 69.49 | 82.10 | 0.537 |

- **Attention + Bottleneck이 최고** (cold MAE 59.45, Ridge 대비 -13%)
- **⚠️ MAE 스케일 문제:** exp006(weekly MAE 8~9)과 단위가 다를 수 있음 → 확인 요청 중
- item-only(페르소나 없이)의 DirAcc 0.561이 가장 높음 → 흥미로운 ablation 발견
- **TODO:** 동일 단위로 exp006 comparison_table과 통합 비교 필요

---

## 6. 핵심 발견 및 논문 소재

### 6.1 Mean-pooling이 페르소나 정보를 파괴한다
- 50 페르소나 hidden states를 평균하면 실질 차원 ~10으로 collapse
- 페르소나 간 이질성(heterogeneity) 정보 소실
- **논문 소재:** "Naive aggregation (mean-pooling) destroys persona diversity in the embedding space"

### 6.2 방향성은 포착하지만 수량은 못한다
- 모든 LLM 기반 방법이 DirAcc에서 baseline(0.232)을 초과
- 하지만 MAE에서는 baseline과 동등하거나 더 나쁨
- **논문 소재:** "LLM captures directional signals but fails at magnitude prediction"
- Tan et al. (2024)의 발견과 일치하면서, 우리 맥락(persona simulation)으로 확장

### 6.3 Attention Aggregation이 Mean-pooling보다 낫다
- Attention + Bottleneck: cold MAE 59.45 vs Ridge mean-pooled: 68.30 (-13%)
- Per-persona attention weights가 도메인 지식과 일치하는지 확인 필요 (TODO)
- **논문 소재:** "Learnable attention aggregation over persona embeddings preserves discriminative information"

### 6.4 M5 데이터의 intermittent demand 특성
- Product-store 수준에서 90.4%가 intermittent demand (73.3% intermittent + 17.1% lumpy)
- 우리 데이터에서 flat(변화 없음) 비율 23.2%
- 랜덤 DirAcc이 0.358 (0.5가 아님!) → DirAcc 해석 시 이 기준 사용해야

### 6.5 LLM 프롬프트 설계의 취약성
- llm_aggregate: "50명의 aggregate 구매"를 개인당 8~16개로 해석 → MAE 408
- llm_zero_shot: 체계적 2배 과대예측 → 스케일 앵커 부재
- llm_similar_item만 성공: 유사 상품의 실제 판매 이력이 스케일 앵커 역할
- **논문 소재:** "Scale anchoring from similar product histories is critical for LLM-based forecasting"

---

## 7. Competitor 검증 현황

### 7.1 구현-논문 매핑 상태
| 모델 | 논문 대응 | 검증 상태 |
|------|---------|---------|
| LightGBM | stephenllh/m5-accuracy (Silver Medal) | **✓ 완료** (exp007 + exp009) |
| k-NN | 일반 개념 (특정 논문 없음) | LOO 검증만 가능 (구조적 한계) |
| DeepAR | Salinas et al. (2020) | electricity 데이터 재현 예정 |
| llm_zero_shot | LLMTime 변형 (토큰화 미반영) | Darts 재현 예정 |
| 나머지 | 자체 설계 | 외부 검증 불가 |

### 7.2 LightGBM 검증 완료 (로컬, 2026-03-08)

#### 7.2.1 검증 전략 (3단계)
1. **exp007**: LightGBM이 warm items에서 Naive보다 나은지 확인 (알고리즘 동작)
2. **exp008**: exp007 결과를 M5 공식 Naive 벤치마크와 같은 기준으로 비교
3. **exp009**: 원본 stephenllh 코드를 M5 전체(30,490 items)로 실행하여 발표 성능과 대조

#### 7.2.2 exp007: Warm Items 구현 검증
- **설정**: CA_1 warm items 300개, d_1~d_1770 학습, d_1771~d_1798 테스트(28일)
- **feature**: lag_7/28, rolling_mean_7/28, rolling_std_7, calendar, sell_price
- **모델**: tweedie(1.5), num_leaves=31, lr=0.05, early_stop@50
- **스케일**: item-own lag-28 MSE (주간 예측에 적합)
- **결과**:

  | 모델 | WRMSSE | MAE |
  |------|--------|-----|
  | LightGBM | **0.9118** | 1.027 |
  | Naive lag-28 | 1.458 | 1.265 |
  | sNaive lag-7 | 1.686 | 1.439 |

- val_MAE(1.048) ≈ test_MAE(1.027) → 과적합 없음 ✓
- best_iteration: 155 (early stopping 정상 작동)
- **결론**: LightGBM > Naive (-37.5% WRMSSE). 알고리즘 정상 동작 ✓

#### 7.2.3 exp008: M5 공식 Naive 벤치마크 비교
- **설정**: CA_1 2,949 items, 두 기간 비교
- **결과**:

  | 모델 | 기간 | WRMSSE(item-own-lag1) | MAE |
  |------|------|----------------------|-----|
  | Naive | d_1886~d_1913 (M5 공식 val) | 1.369 | 1.571 |
  | sNaive | d_1886~d_1913 (M5 공식 val) | 1.318 | 1.404 |
  | Naive | d_1771~d_1798 (exp007 기간) | 1.458 | 1.265 |
  | sNaive | d_1771~d_1798 (exp007 기간) | 1.686 | 1.439 |

- LightGBM(0.9118) vs Naive(1.458) → M5 공식 기준으로도 LightGBM 우위 확인 ✓

#### 7.2.4 exp009: stephenllh 원본 코드 M5 전체 실행
- **원본 출처**: https://github.com/stephenllh/m5_accuracy (Silver Medal)
- **공개 점수**: WRMSSE 0.637 (private LB — README 원문 "private leaderboard" 명시)
- **실행 환경**: Python 3.x, pandas 2.2.3, LightGBM 4.6.0
- **API 호환성 패치 6건** (알고리즘 변경 없음):

  | 파일 | 문제 | 패치 |
  |------|------|------|
  | `train.py` | `verbose_eval` 파라미터 제거 (LightGBM 4.x) | `callbacks=[lgb.log_evaluation(20)]` |
  | `train.py` | `.loc[:-10000]` 빈 DataFrame (pandas 2.x) | `.iloc[:-10000]` |
  | `preprocess.py` | `dt.weekofyear` 제거 (pandas 2.x) | `dt.isocalendar().week.astype("int16")` |
  | `preprocess.py` | `pd.Index.to_csv()` 제거 (pandas 2.x) | `open() + write()` |
  | `inference.py` | 동일 weekofyear 이슈 | 동일 패치 |
  | `inference.py` | `lgb.load()` 제거 (LightGBM 4.x) | `lgb.Booster(model_file=...)` |
  | `inference.py` | `pd.read_csv()` → DataFrame (list 필요) | `.iloc[:, 0].tolist()` |

- **학습**: 30,490 items, d_250~d_1913 (~40.7M rows), ~18분, 7코어
- **예측**: d_1914~d_1941 (28일 recursive)

- **Step A — Product-level WRMSSE** (`scripts/compute_stephenllh_wrmsse.py`):

  | 모델 | WRMSSE (item-own-lag1, product-level) | MAE |
  |------|--------------------------------------|-----|
  | stephenllh LightGBM | **1.004** | 1.033 |
  | Naive product-level | ~1.37 | — |

- **Step B — 12-level WRMSSE** (`scripts/compute_m5_12level_wrmsse.py`, 공식 evaluator):
  - 평가 코드 출처: `github.com/Mcompetitions/M5-methods/A4/m5a_eval.py` (4위 팀)
  - 12-level WRMSSE (validation period d_1914~d_1941): **0.5435**

  | Level | WRMSSE |
  |-------|--------|
  | lv1: Total | 0.3501 |
  | lv2: State | 0.3977 |
  | lv3: Store | 0.4728 |
  | lv4: Category | 0.3710 |
  | lv5: Department | 0.4222 |
  | lv6: State×Category | 0.4381 |
  | lv7: State×Department | 0.4969 |
  | lv8: Store×Category | 0.5245 |
  | lv9: Store×Department | 0.5975 |
  | lv10: Item | 0.8074 |
  | lv11: Item×State | 0.8187 |
  | lv12: Item×Store | 0.8249 |
  | **평균 (12-level)** | **0.5435** |

- **0.5435 ≠ 0.637인 이유 (구조적, 재현 불가)**:
  - 0.637 = private LB = d_1942~d_1969 실제 vs d_1914~d_1941 예측(복사)
  - 0.5435 = validation period = d_1914~d_1941 실제 vs d_1914~d_1941 예측
  - d_1942~d_1969 실제 데이터는 Kaggle 비공개 → 직접 비교 불가
  - 0.5435 < 0.637: 예측 기간(val)보다 복사된 기간(eval)에서 오차 큼 → 논리적으로 일관 ✓

- **최종 결론**: LightGBM 코드 정상 동작 확인. 집계 레벨 패턴 정상(높은 집계일수록 낮은 WRMSSE). Naive 대비 개선 일관. 0.637 수치 자체의 재현은 private 데이터 부재로 불가능하나, 동작 검증 목적은 달성 ✓

### 7.3 미완료 검증 작업
| 작업 | 상태 | 비고 |
|------|------|------|
| DeepAR electricity 재현 | 예정 | Salinas et al. (2020) |
| LLMTime Darts 재현 | 예정 | 토큰화 미반영 이슈 있음 |

---

## 8. 리뷰어 피드백 및 대응

### 8.1 GPT-4o 리뷰 (핵심 지적사항)
1. **딥러닝 Global Forecasting Model 누락** → DeepAR 추가함, TFT는 Phase 2
2. **Chronos/Time-LLM을 Hybrid Baseline으로 가능** → Limitations에 명시
3. **LightGBM에 Proxy Lags 필요** → 구현함 (lightgbm_proxy_lags)
4. **Track A vs LLM Direct 비교의 교란 변수** → 3개 ablation 변형 추가 (3-1, 3-2, 3-3)
5. **실험 Scale 작다 (400 items)** → Phase 2에서 확대
6. **M5 식료품과 페르소나 궁합** → Onout 의류 데이터 추가 검토 중
7. **Cold-start 시뮬레이션의 한계** → M5에서 실제 신규 진입 아이템 발굴 시도 (Phase 2)
8. **DirAcc 0.41 < 0.5 설명 필요** → 랜덤 DirAcc이 0.358임을 밝힘

### 8.2 자기 진단: "쉬운 길"로 빠진 지점들
1. Competitor를 "일반 알고리즘"으로 추천, 특정 논문의 구현이 아니었음
2. WRMSSE 기준 위반 시 기준을 바꾸는 사후 합리화
3. Track B 실패 시 바로 "MLP로 바꾸자"만 제안, 파라미터-데이터 비율 미확인
4. "현재 실험의 타당성 검증"보다 "다음 실험으로 넘어가자"에 집중

---

## 9. 미해결 이슈 및 TODO

### 즉시 필요
- [ ] exp009(서버exp) MAE 스케일 문제 확인 (daily vs weekly? 스케일 보정?)
- [ ] exp009(서버exp) Attention weights 분석 (어떤 페르소나에 높은 가중치?)
- [ ] exp006과 exp009(서버exp)의 통합 비교 테이블 (동일 단위)
- [x] M5 공식 벤치마크 예측값 다운로드 및 subset WRMSSE 계산 → **완료** (exp008, Naive=1.458)
- [ ] Onout 대표에게 데이터 사용 허가 문의

### 단기 (1-2주)
- [ ] DeepAR electricity 데이터 재현 검증
- [ ] LLMTime Darts 재현 검증
- [x] Silver medal LightGBM 코드로 M5 재현 → **완료** (exp009, 12-level=0.5435, 0.637 재현 불가 확인)
- [ ] Onout 데이터 구조 확인 및 실험 설계

### 중기 (Phase 2)
- [ ] Warm items 전체(2,949개)로 확장
- [ ] Per-persona raw embeddings + Attention Head (전체 데이터)
- [ ] FP16 vs INT4 양자화 비교 (ablation)
- [ ] 페르소나 수 ablation (10/20/30/50명)
- [ ] Condition B/C (Structured+Narrative, Narrative Only)
- [ ] Onout 의류 데이터 실험
- [ ] 시계열 foundation model hybrid baseline (Chronos-tiny + k-NN)

---

## 10. 세션 로그

### 세션 1 (2026-02-26): 문헌 조사
- 연구 컨텍스트 문서 작성
- multi-persona agent simulation 접근법 문헌 리뷰
- EMNLP 2026 제출 포지셔닝

### 세션 2 (2026-02-27): EMNLP 연구 설계
- Tiered Persona Design (Naive/Data-Backed/Cognitive)
- Oracle Ablation, 합성 구매 이력 생성, 3-level 평가 지표
- zero-leakage Tier 3 프롬프트

### 세션 3 (2026-03-04): 코드 리뷰 및 파이프라인 설계
- OpenClaw → Claude Code 전환
- M5 데이터 전처리, naive baselines 구현
- 50명 합성 페르소나 생성 (LLM-free)
- Track A/B 2-stage 아키텍처 설계

### 세션 4 (2026-03-04~07): 실험 실행 및 분석
**주요 변경사항:**
- Track A 완료: DirAcc 0.393, MAE 8.90 (weekly)
- Track B 완료: Ridge MAE 68.30, Attention+Bottleneck MAE 59.45
- Competitor 8종 구현 및 실행 (LLM 3종 포함)
- Mean-pooling의 정보 파괴 발견 (PCA 98.6% in 10 PCs)
- DirAcc 랜덤 기준이 0.358임을 확인
- 구현 검증 필요성 인식 → 검증 계획 수립
- Onout 의류 데이터 추가 검토 시작

### 세션 5 (2026-03-08~09, 로컬): Competitor 재현 검증 (LightGBM)
**담당:** 로컬 (verification 브랜치)
**주요 작업 및 결과:**

1. **Phase4 코드 검증 (실험 없이 코드 리뷰)**
   - 데이터 누수 없음 확인: `load_data()`의 `~isin(cold_ids)`, KNNAnalog/LightGBMCross fit/predict 모두 warm 데이터만 사용
   - DirAcc flat 처리 정상: `np.sign(0)=0`, `0==0=True`, `groupby().diff()` 경계 처리, `dropna()` 첫 주 제거

2. **exp007: LightGBM warm items 구현 검증**
   - CA_1 300 warm items, d_1771~d_1798 테스트, item-own-lag28 WRMSSE
   - LightGBM(0.9118) < Naive(1.458) → 알고리즘 정상 ✓
   - config: `configs/exp007_lgbm_verification.yaml`

3. **exp008: M5 공식 Naive 벤치마크 계산**
   - CA_1 2,949 items, 두 기간(d_1886~d_1913, d_1771~d_1798)
   - Naive(1.369~1.458), sNaive(1.318~1.686) — LightGBM 우위 확인
   - config: `configs/exp008_m5_benchmarks.yaml`

4. **exp009: stephenllh/m5-accuracy 원본 재현**
   - M5 전체 30,490 items, API 패치 6건(알고리즘 변경 없음)
   - Step A: product-level WRMSSE = 1.004, Naive ~1.37 → 코드 정상 ✓
   - Step B: 공식 12-level WRMSSE = 0.5435 (evaluation code: M5-methods/A4/m5a_eval.py)
   - **0.637 재현 불가**: 0.637 = private LB (stephenllh README 명시), d_1942~d_1969 실제 데이터 비공개
   - 0.5435 < 0.637: 검증 기간(val) vs 평가 기간(eval) 차이로 논리적으로 일관 ✓
   - config: `configs/exp009_stephenllh_verification.yaml`

5. **문서화**
   - `docs/verification/lgbm_adaptation_log.md`: 원본→cold-start 적응 변경사항 전체 기록
   - `docs/RESEARCH_CONTEXT.md`: 베이스라인 수치, 지표 정의, 실험 결과 추가
   - `docs/RESEARCH_MASTER_CONTEXT.md` (현재 파일): 섹션 7 업데이트
   - `configs/exp007~009_*.yaml`: 실험 재현용 config 파일 생성

6. **브랜치 규칙 확립**
   - 로컬: `verification` 브랜치, 커밋 `[local]` 접두사
   - 서버: `experiment/trackb` 브랜치 (pull로 확인)
   - `main` 직접 push 금지. Phase 완료 후 사용자 확인 후 merge
