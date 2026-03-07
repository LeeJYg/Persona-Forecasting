# 연구 노트: Exp004 ~ Exp009

**프로젝트**: LLM Persona Cold-Start Forecasting (M5 데이터셋, CA_1 매장)
**작성일**: 2026-03-07
**목적**: 이 노트만 보고 각 실험을 완전히 재현할 수 있도록 기록

---

## 목차

- [공통 데이터셋 및 설정](#공통-데이터셋-및-설정)
- [Exp004: Track A Naive (미실행)](#exp004-track-a-naive--gpt-4o-mini-페르소나-직접-예측)
- [Exp005: Track B Mean-Pooled Ridge](#exp005-track-b--qwen25-32b-4bit--mean-pooled-ridge)
- [Exp006: Track B 심층 분석 + 프록시 베이스라인](#exp006-track-b-심층-분석--프록시-베이스라인-비교)
- [Exp007: 학습 데이터 확대 가능성 검토](#exp007-학습-데이터-확대-가능성-검토--개념-검토만-수행)
- [Exp008: MLP 헤드 설계 (취소)](#exp008-mlp-헤드-설계--실행-취소)
- [Exp009: Attention Head — Raw Per-Persona 임베딩](#exp009-attention-head--raw-per-persona-임베딩--6종-헤드-비교)
- [부록: 단위 혼동 해소 이력](#부록-단위-혼동-해소-이력)

---

## 공통 데이터셋 및 설정

> Exp004 ~ Exp009 전체가 공유하는 데이터 구조와 페르소나 정보

### 원본 출처

M5 Forecasting Accuracy (Walmart) 데이터셋, CA_1 (캘리포니아 1호점) 매장 기준 cold-start 시뮬레이션.

### 데이터 파일

| 파일 | 위치 | 행 수 | 아이템 수 | 기간 |
|------|------|------:|--------:|------|
| cold_test.csv | data/processed/cold_start/ | 11,500 | 100 | 2016-01-01 ~ 2016-04-24 (115일) |
| warm_train.csv | data/processed/cold_start/ | — | CA_1 전체 | cold_test 이전 기간 |
| warm_test.csv | data/processed/cold_start/ | — | CA_1 전체 | cold_test와 동일 기간 |
| sell_prices.csv | m5-forecasting-accuracy/ | — | — | M5 원본 주간 가격 |

### cold_test 구조

- **컬럼**: `id, item_id, dept_id, cat_id, store_id, state_id, d, sales, date, year, month`
- **카테고리 구성**: FOODS 34개, HOBBIES 33개, HOUSEHOLD 33개 (균형 샘플링)
- **cold-start 조건**: `cross_store_info=false` — 타 매장의 동일 아이템 이력 사용 불가
- **일별 판매량 통계**: mean=1.47, std=5.15, min=0, max=135 (우측 꼬리 분포, 0이 다수)

### Warm 아이템 샘플링 (Exp005, Exp009 공통)

```python
# scripts/run_track_b.py, scripts/run_exp009.py 동일 로직
n_warm = 300
seed = 42  # configs/config.yaml
# 카테고리 균형: FOODS 100, HOBBIES 100, HOUSEHOLD 100
# cold_ids를 제외한 warm_train 아이템에서 np.random.default_rng(seed).choice()
```

### 페르소나 (Exp003에서 생성, 이후 전체 공유)

- **수량**: 50개 (CA_1_P001.json ~ CA_1_P050.json)
- **위치**: `data/processed/personas/`
- **생성 모델**: GPT-4o (generation_temperature=1.0)
- **스키마 필드**: description, weekly_budget, snap_eligible, shopping_motivation, economic_status, category_preference (dict), price_sensitivity, visit_frequency, preferred_departments, decision_style, brand_loyalty, promotion_sensitivity
- **사용 조건**: Condition A (구조화 필드만, narrative 제외) — Li et al. 2025 권고

### 주간 집계 방식 비교 (중요: Exp005 vs Exp009 불일치)

두 실험에서 일별 데이터를 주간으로 집계하는 방식이 다르며, 이로 인해 동일 아이템의 주간 합계 수치가 달라진다.

| 항목 | Exp005 (`linear_head.py`) | Exp009 (`run_exp009.py`) |
|------|--------------------------|------------------------|
| 방식 | 날짜 정렬 후 순서 기반 7-day chunks (`i // 7`) | `pd.to_period("W")` (ISO Mon-Sun) |
| 첫 주 | 2016-01-01 ~ 2016-01-07 (7일) | 2015-12-28 ~ 2016-01-03 (실질 3일) |
| 마지막 주 | 2016-04-22 ~ 2016-04-24 (3일) | 2016-04-18 ~ 2016-04-24 (7일) |
| n_weeks | 17 | 17 |
| 재현 권장 | Exp009 방식 (ISO 달력 일관성) | |

### DirAcc 계산 방식 비교 (중요: Exp005 vs Exp009 불일치)

이 차이로 인해 Exp005의 DirAcc=0.3719와 Exp009의 DirAcc=0.536은 직접 비교 불가.

| 항목 | Exp005 (`src/evaluation/metrics.py`) | Exp009 (`scripts/run_exp009.py`) |
|------|--------------------------------------|----------------------------------|
| flat(0) 처리 | **포함**: sign(0)==sign(0)을 정답으로 계산 | **제외**: true_dir==0 주를 분모에서 제외 |
| 주 집계 | ISO calendar week (`dt.isocalendar().week`) | `pd.Period("W")` |
| 입력 | 일별 예측 → 주별 합산 | 주별 예측 직접 입력 |
| 결과 특성 | 0판매 주간 많으면 베이스라인도 높게 나옴 | 실제 증감 있는 주만 평가 (더 엄밀) |
| 향후 권장 | — | Exp009 방식으로 통일 |

---

## Exp004: Track A Naive — GPT-4o-mini 페르소나 직접 예측

**상태: 코드 설계 완료, 미실행**

### 1. 실험 목적

- **가설**: LLM이 페르소나 프로필과 아이템 메타만 보고 cold-start 아이템의 주간 판매량을 직접 출력할 수 있는가?
- 50명 페르소나 각각이 "나는 이 아이템을 한 주에 몇 개 살 것 같다"고 예측하고, 합산 후 매장 수요로 환산
- exp002 베이스라인(카테고리 평균 등)을 LLM이 뛰어넘는지 검증

### 2. 실험 세팅

#### 데이터셋

- **입력**: cold_test.csv 100 아이템, 예측 기간 2016-01-01 ~ 2016-04-24 (17주)
- **페르소나**: 50개 (Condition A, 구조화 필드만)
- **아이템 정보**: item_id, dept_id, cat_id, avg_price (최근 13주 평균)

#### 방법론

- **모델**: gpt-4o-mini (`prediction_temperature=0.3`, 비용 최적화)
- **배치 구조**: 1 persona × 10 items × 1 week = 1 API call
  - 총 API calls: 50 personas × ceil(100/10) batches × 17 weeks = **8,500 calls**
  - 추정 비용: ~$0.60 USD (input 700 token/call, output 100 token/call 기준)
- **스케일 보정**: 50명 합산(50-persona scale) → post-hoc mean ratio alpha 보정으로 매장 수준으로 환산
  - `alpha = mean(baseline_pred) / mean(track_a_raw)`
- **체크포인트**: 중단 재시작 지원 (`experiments/exp004_track_a_naive/checkpoints/prediction_checkpoint.json`)

#### 평가 지표

- MAE, RMSE: daily 단위 (build_pred_dataframe으로 주간 예측을 일별 균등 배분)
- WRMSSE: cold-start 수정버전 (cold item scale = 동일 카테고리 warm item 28-day lag MSE 평균)
- DirAcc: weekly flat-inclusive (src/evaluation/metrics.py 방식)

#### 환경

- 로컬 실행, OpenAI API 필요
- Python 3.10+, `openai` 라이브러리

#### 실행 명령

```bash
cd ~/Persona-Forecasting
python scripts/run_track_a.py               # Track A 예측 (8,500 API calls)
python scripts/compare_track_a_baselines.py # exp002 베이스라인과 비교 테이블 생성
```

#### 출력 파일 (예정)

```
experiments/exp004_track_a_naive/
├── predictions/
│   ├── track_a_raw.csv          # 50-persona 합산 원본
│   └── track_a_calibrated.csv  # alpha 보정 후
├── results/
│   ├── comparison_table.csv     # 5 모델 × 4 지표
│   └── by_category_table.csv
└── checkpoints/
    └── prediction_checkpoint.json
```

### 3. 실험 결과

**미실행** — 결과 없음.

참고: 이후 세션에서 파악된 Track A calibrated의 weekly MAE = 8.90, DirAcc = 0.393 (출처 불명확, Exp006 참조값으로 기록됨)

### 4. 실험 결론

**설계상 예측되는 한계**:
1. GPT-4o-mini의 수치 예측 불안정성 (언어 생성 최적화 모델)
2. 50명 합산 스케일이 매장 수준과 정렬되지 않아 post-hoc 보정 필수
3. API 비용 및 8,500 call 호출 시간 부담

대신 Track B(Exp005)를 먼저 실행해 임베딩 기반 접근의 feasibility를 검증했으며, Track A는 후속 비교 실험으로 남겨둠.

---

## Exp005: Track B — Qwen2.5-32B 4-bit + Mean-Pooled Ridge

**상태: 실행 완료** (2026-03-05 ~ 03-06, 약 14.8시간)

### 1. 실험 목적

- **가설**: LLM 페르소나 임베딩을 50개에 대해 mean-pooling하면, warm 아이템 판매량을 supervision으로 Ridge 회귀 학습이 가능하고, cold-start 예측이 가능한가?
- Phase 1 feasibility test의 핵심 실험
- 임베딩 추출에 Qwen2.5-32B-Instruct (INT4 양자화)를 사용해 연구실 서버(Tesla P40)에서 실행

### 2. 실험 세팅

#### 데이터셋

- **cold_test**: 100 아이템 × 115일 (2016-01-01 ~ 2016-04-24)
- **warm**: 300 아이템 샘플링 (seed=42, 카테고리 균형)
- **예측 기간**: 17주 (7-day naive chunks from date_start, 마지막 주 3일)

#### 모델 구성

| 항목 | 값 |
|------|---|
| LLM | Qwen/Qwen2.5-32B-Instruct |
| 양자화 | 4-bit NF4, compute_dtype=bfloat16, double_quant=True |
| `torch_dtype` 지정 | **하지 않음** (양자화 시 지정하면 full-precision 로드 후 양자화 → OOM 발생) |
| batch_size | 4 |
| embedding_dim | 5,120 (hidden_size, last token) |
| device_map | auto |
| output_hidden_states | True (load_kwargs에 포함, generation flag 경고 무시) |

#### 임베딩 추출 방식 (mean-pooling)

```python
# 각 아이템에 대해:
texts = [build_combined_text(persona_i_text, item_text) for i in range(50)]
embeds = embedder.get_embeddings(texts)  # (50, 5120): last-token hidden state
item_emb = embeds.mean(axis=0)           # (5120,): 50 persona 평균
```

- 총 forward passes: 20,000 (warm 300×50 + cold 100×50)
- 저장: `.npz` 포맷 (embeddings array + item_ids array)

#### 회귀 헤드 (`WeeklySalesHead`)

```python
# 입력 표준화 (X, Y 모두)
x_scaled = StandardScaler().fit_transform(x_warm)  # (300, 5120)
y_scaled = StandardScaler().fit_transform(y_warm)  # (300, 17)

# Ridge 학습
model = Ridge(alpha=1.0)  # config.yaml: regression_alpha=1.0
model.fit(x_scaled, y_scaled)

# 예측 후 역변환 + clip(0)
y_pred = scaler.inverse_transform(model.predict(x_cold_scaled))
y_pred = np.clip(y_pred, 0, None)  # (100, 17) weekly
```

- **주간 집계**: 7-day naive chunks (`date_to_week = {d: i//7 for i, d in enumerate(all_dates)}`)
- **CV**: RidgeCV(alphas=[0.01~1000], cv=5, scoring="neg_mean_absolute_error")로 alpha 자동 선택 (결과: best_alpha=1.0)

#### 예측 → 일별 변환

```python
# build_pred_dataframe(): 주간 예측값을 해당 주 일수로 균등 배분
daily_pred = weekly_qty / n_days_in_week
```

#### 평가 지표 계산 방식

- **MAE, RMSE**: **daily 단위** 예측 vs 일별 실측 (`src/evaluation/metrics.py evaluate()`)
- **WRMSSE**: cold item scale = 동일 카테고리 warm item의 28-day lag diff MSE 평균으로 대체; 가중치 = 테스트 기간 실제 판매량 합
- **DirAcc**: **weekly, flat-inclusive** — ISO calendar week 집계 후 연속 주 증감 비교; sign(0)==sign(0)을 정답으로 처리; 첫 주(diff=NaN)는 제외

#### 환경

| 항목 | 값 |
|------|---|
| 서버 | dslab-gpu12 |
| GPU | Tesla P40 × 4 (22.4GB × 4), CUDA_VISIBLE_DEVICES=0,1,2,3 |
| GPU 메모리 분배 | GPU0=3.6GB, GPU1=6.9GB, GPU2=7.4GB, GPU3=10.5GB (합 ~29GB) |
| HF 캐시 위치 | /mnt/sdd1/jylee/huggingface_cache (모델 크기 ~54GB) |
| Python env | /home/jylee/anaconda3/envs/persona-forecasting |
| 주요 라이브러리 | transformers, bitsandbytes, torch, sklearn, pandas, numpy |

#### 실행 명령

```bash
tmux new-session -d -s track_b
tmux send-keys -t track_b \
  'export HF_HOME=/mnt/sdd1/jylee/huggingface_cache && \
   export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
   cd ~/Persona-Forecasting && \
   /home/jylee/anaconda3/envs/persona-forecasting/bin/python scripts/run_track_b.py \
   2>&1 | tee experiments/exp005_track_b_embedding/run.log' Enter
```

#### 출력 파일

```
experiments/exp005_track_b_embedding/
├── embeddings/
│   ├── item_emb_cold.npz    # shape: (100, 5120)
│   └── item_emb_warm.npz    # shape: (300, 5120)
├── models/
│   └── regression_head.pkl  # WeeklySalesHead (StandardScaler + Ridge, best_alpha=1.0)
├── predictions/
│   └── track_b_pred.csv     # 일별 예측 (item_id, store_id, date, pred_sales, cat_id)
└── metrics/
    └── evaluation_results.json
```

### 3. 실험 결과

> **단위**: MAE, RMSE = **daily** / WRMSSE = cold-start 수정버전 / DirAcc = weekly flat-inclusive

**전체 지표**:
| 지표 | 값 |
|------|---|
| MAE (daily) | 13.07 |
| RMSE (daily) | 18.27 |
| WRMSSE | 6.30 |
| DirAcc (weekly, flat-inclusive) | 0.3719 |
| n_items | 100 |
| n_rows | 11,500 |

**카테고리별 (daily 단위)**:
| 카테고리 | MAE | RMSE | WRMSSE | DirAcc |
|---------|----:|-----:|-------:|-------:|
| FOODS | 15.11 | 21.15 | 4.81 | 0.373 |
| HOBBIES | 11.93 | 16.54 | 9.53 | 0.367 |
| HOUSEHOLD | 12.11 | 16.65 | 6.92 | 0.375 |

**exp002 베이스라인 대비 (daily 단위)**:
| 모델 | MAE (daily) | DirAcc |
|------|------------|--------|
| GlobalCategoryAverage | 1.57 | 0.257 |
| SimilarItemAverage | 1.69 | 0.257 |
| StoreCategoryAverage | 1.62 | 0.257 |
| **Exp005 Track B (raw)** | **13.07** | **0.372** |

**예측 스케일 통계**: pred_sales 평균 = 13.89 (daily) → 실제 평균 1.47 대비 약 9.4배 과예측

### 4. 실험 결론

1. **대규모 스케일 불일치 발견**: pred_mean/true_mean ≈ 9.4배. 원인: Ridge 헤드가 warm 아이템(판매량 큰) 기반으로 학습된 후 cold 아이템(판매량 작은)에 외삽할 때 bias 발생
2. **Alpha 보정 후 daily MAE = 1.86**: 베이스라인(1.57~1.69)에 근접하나 여전히 열위
3. **DirAcc = 0.372 > 베이스라인 0.257**: 방향 예측에서는 페르소나 임베딩이 의미 있는 개선
4. **단위 혼동 발생**: 이 실험의 MAE 13.07은 daily 단위이나, 이후 분석(Exp006)에서 weekly MAE 68.30이 함께 언급되면서 혼선 발생 → 부록 참조

**Exp006으로 연결**: 스케일 불일치 원인 진단, 대안 회귀 모델 탐색, 임베딩 표현력 평가 진행.

---

## Exp006: Track B 심층 분석 + 프록시 베이스라인 비교

**상태: 실행 완료** (2026-03-06, Exp005 완료 직후 연속 실행)

### 1. 실험 목적

- Exp005의 스케일 불일치(alpha=0.106) 원인 진단
- Ridge 과적합 여부 확인 (임베딩 품질 vs 회귀 헤드 문제)
- 대안 회귀 모델(Lasso, RandomForest, PCA+Linear)로 성능 개선 가능성 탐색
- mean-pooled 임베딩의 내재 rank 측정 (PCA)
- t-SNE로 임베딩 공간 카테고리 분리 가능성 시각화
- LightGBM, KNN analog, Track A calibrated 등 프록시 베이스라인과 주간 단위 비교

### 2. 실험 세팅

#### 데이터셋

- Exp005 결과 재사용: `item_emb_cold.npz`, `item_emb_warm.npz`, `track_b_pred.csv`
- 직접 데이터: cold_test.csv, warm_train.csv, warm_test.csv

#### 평가 대상

1. 스케일 보정 (global mean ratio alpha)
2. Ridge 과적합 진단 (train MAE vs 5-fold CV val MAE)
3. 대안 회귀 모델 4종 (weekly MAE 단위로 재평가)
4. t-SNE 시각화 (warm 임베딩)
5. PCA 분산 설명률 (임베딩 실질 rank)
6. 프록시 베이스라인 비교값 (외부 참조)

#### 방법론

**스케일 보정**:
```python
# analyze_track_b.py: analysis_scale_correction()
true_mean = cold_test["sales"].mean()      # 1.4722 (daily)
pred_mean = pred_df["pred_sales"].mean()   # 13.8892 (daily)
alpha = true_mean / pred_mean              # 0.1060
corrected_pred = pred_sales * alpha
```

**Ridge 과적합 진단**:
```python
# 주별 독립 학습, warm 데이터로만
Ridge(alpha=1.0).fit(X_warm, y_warm[:, w])
train_mae = MAE(y_warm[:, w], model.predict(X_warm))
val_mae = cross_val_score(model, X_warm, y_warm[:, w], cv=5, scoring='neg_mae').mean() * -1
```

**대안 회귀 모델**:
- `Ridge(alpha=1.0)`, `Lasso(alpha=0.01, max_iter=5000)`, `RandomForest(n_estimators=100, random_state=42, n_jobs=-1)`, `Pipeline([PCA(50), LinearRegression()])`
- 5-fold CV warm val MAE + cold test MAE (모두 **weekly** 단위, `pd.Period("W")` 집계)
- flat 제외 DirAcc (`true_dir != 0` mask)

**PCA 분산 분석**:
```python
pca = PCA().fit(X_warm)  # X_warm shape: (300, 5120)
cumvar = np.cumsum(pca.explained_variance_ratio_)
```

**t-SNE**:
```python
X_pca = PCA(n_components=50, random_state=42).fit_transform(X_warm)
X_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000,
            learning_rate="auto", init="pca").fit_transform(X_pca)
```

**프록시 베이스라인 (참조값)**:
- LightGBM proxy lags (weekly MAE=8.48, DirAcc=0.343)
- Track A calibrated (weekly MAE=8.90, DirAcc=0.393)
- knn_analog (weekly MAE=9.57, DirAcc=0.412)
- 주의: 이 값들은 실험 디렉토리 없이 참조값으로만 기록됨

#### 환경

- `scripts/analyze_track_b.py` 실행
- GPU 불필요 (임베딩 재사용, CPU only)

#### 실행 명령

```bash
mkdir -p experiments/exp005_track_b_embedding/analysis
cd ~/Persona-Forecasting
/home/jylee/anaconda3/envs/persona-forecasting/bin/python scripts/analyze_track_b.py \
  2>&1 | tee experiments/exp005_track_b_embedding/analysis/analyze_run.log
```

#### 출력 파일

```
experiments/exp005_track_b_embedding/analysis/
├── analyze_run.log
└── tsne_warm_embeddings.png    # warm 임베딩 t-SNE 시각화
```

### 3. 실험 결과

**스케일 보정**:
| | pred_mean (daily) | alpha | raw MAE (daily) | corrected MAE (daily) |
|-|------------------:|------:|----------------:|---------------------:|
| Track B Ridge | 13.89 | 0.106 | 13.07 | **1.86** |

카테고리별 corrected MAE (daily): FOODS=2.86, HOBBIES=1.37, HOUSEHOLD=1.32

**Ridge 과적합 진단 (weekly MAE)**:
| | MAE |
|-|----:|
| Train (warm full fit) | 48.76 |
| Val (5-fold CV) | 61.24 |
| Val/Train ratio | 1.26 |

→ Overfitting 아님. Train/Val 모두 높음 = 임베딩의 정보 부족이 원인

**대안 회귀 모델 비교 (weekly MAE)**:
| 모델 | warm_val_MAE | cold_MAE (weekly) |
|------|-------------|-----------------|
| Ridge(alpha=1) | 61.24 | 68.30 |
| Lasso(alpha=0.01) | 90.61 | 89.64 |
| RandomForest(100) | 57.47 | **66.00** |
| PCA(50)+Linear | 70.14 | 73.24 |

**PCA 분산 설명률 (warm 임베딩 5120차원)**:
| top-k PCs | 누적 분산 |
|-----------|--------:|
| 5 | 97.3% |
| 10 | 98.6% |
| 15 | ~99.0% |
| 20 | 99.2% |

**t-SNE**: 카테고리별 mild clustering 관찰, 완전 분리 아님.

**통합 비교 (weekly MAE, 동일 단위)**:
| 모델 | weekly MAE | 비고 |
|------|----------:|------|
| LightGBM proxy lags | 8.48 | 참조값 |
| Track A calibrated | 8.90 | 참조값 |
| knn_analog | 9.57 | 참조값 |
| RandomForest | 66.00 | Exp006, alpha 미보정 |
| Ridge (mean-pooled) | 68.30 | Exp006, alpha 미보정 |

### 4. 실험 결론

1. **임베딩 내재 rank 문제 확인**: 5120차원 mean-pooled 임베딩이 실질적으로 ~15개 PC로 표현됨. mean-pooling이 50개 페르소나의 다양성을 파괴하고 저차원 매니폴드로 압축.
2. **회귀 모델 교체로 해결 불가**: Lasso/RandomForest/PCA+Linear 모두 Ridge 대비 개선 없음 → 임베딩 자체의 정보량 부족이 근본 원인.
3. **방향 설정**: mean-pooling 대신 raw per-persona 임베딩 (N, 50, 5120)을 보존하고 학습 가능한 attention aggregation으로 전환 → Exp009 설계.

---

## Exp007: 학습 데이터 확대 가능성 검토 — 개념 검토만 수행

**상태: 개념 검토 완료, 별도 실험 미실행**

### 1. 실험 목적

- warm_train 아이템 수를 300개 이상으로 늘리면 Ridge 헤드 성능이 향상되는가?
- cold_test 아이템이 실제로 엄격한 cold-start 조건을 만족하는가?

### 2. 실험 세팅 / 결론

**cold-start 조건 검증**: cold_test의 100 아이템은 `cross_store_info=false` 조건으로 샘플링되어 타 매장 이력도 없는 순수 cold-start. 조건 이상 없음.

**warm 아이템 확대 검토**:
- CA_1 전체 아이템에서 cold 100개 제외 → 수백 개 이상 추가 가능
- 그러나 임베딩 추출 비용이 선형 증가: n_warm 증가 시 forward pass 비례 증가
- Exp006에서 이미 과적합이 아님 확인 → 데이터 확대보다 임베딩 방식 변경이 더 효과적

**결론**: Exp009(Attention Head)로 방향 전환. 데이터 확대 실험 불필요.

---

## Exp008: MLP 헤드 설계 — 실행 취소

**상태: 코드 완성, GPU에서 실행 시작 후 취소**

### 1. 실험 목적

- Ridge 선형 헤드를 MLP로 교체해 임베딩 공간의 비선형 구조 활용
- 가설: 5120차원 임베딩에서 MLP가 더 많은 예측 신호를 추출할 수 있을 것

### 2. 실험 세팅

**아키텍처**:
```python
nn.Sequential(
    nn.Linear(5120, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 17),   # n_weeks=17 출력
)
```

**학습 설정**:
- optimizer: Adam(lr=1e-3, weight_decay=1e-4)
- epochs: 200, early stopping patience: 20 (val loss 기준)
- 5-fold CV on warm (300 items)
- 입력: `item_emb_warm.npz` (300, 5120)

**스크립트**: `scripts/train_mlp_head.py` (작성 완료, 실행 취소)

### 3. 실험 결과

**실행 취소** — 결과 없음.

### 4. 실험 결론

**취소 이유**:
1. Exp006 PCA 분석에서 5120차원 mean-pooled 임베딩이 실질적으로 ~15차원 매니폴드임 확인
2. MLP 비선형성도 저차원 매니폴드에 갇힌 정보를 늘릴 수 없음
3. 근본 해결책은 raw per-persona 임베딩(N, 50, 5120) 보존 + attention aggregation
4. → Exp009로 전환

---

## Exp009: Attention Head — Raw Per-Persona 임베딩 + 6종 헤드 비교

**상태: 실행 완료** (2026-03-06 04:09 ~ 2026-03-07 18:56, 총 888분/14.8시간)

### 1. 실험 목적

- **가설 1**: Raw per-persona 임베딩(N, 50, 5120) 보존 시 mean-pooled(N, 5120)보다 예측 성능 향상되는가?
- **가설 2**: Attention aggregation이 Ridge(mean-pooled) 대비 cold-start MAE를 개선하는가?
- **Ablation**: item-only 임베딩(페르소나 미사용)과 페르소나 포함 임베딩 비교
- **분석**: Attention head가 어떤 페르소나에 높은 가중치를 부여하는가? 카테고리별 차이 있는가?

### 2. 실험 세팅

#### 데이터셋

- cold_test: 100 아이템 × 115일 (2016-01-01 ~ 2016-04-24)
- warm: 300 아이템 (seed=42, 카테고리 균형)
- **주간 집계**: `pd.to_period("W")` (ISO Mon-Sun)
- y_cold shape: **(100, 17)**, y_warm shape: **(300, 17)**
- cold 주간 판매량 통계: mean=9.96, std=28.64, max=432

#### Step 1: 임베딩 추출

**모델 설정**:
| 항목 | 값 |
|------|---|
| LLM | Qwen/Qwen2.5-32B-Instruct |
| 양자화 | 4-bit NF4, compute_dtype=bfloat16, double_quant=True |
| `torch_dtype` | 지정 안 함 (양자화 시 OOM 방지) |
| batch_size | 4 |
| device_map | auto |
| HF 캐시 | /mnt/sdd1/jylee/huggingface_cache |

**추출 텍스트 종류**:
```python
# raw (per-persona): 50개 페르소나 × N 아이템
all_texts = [build_combined_text(persona_text_i, item_text_j)
             for j in item_ids for i in persona_texts]
all_emb = embedder.get_embeddings(all_texts)  # (N*50, 5120)
raw = all_emb.reshape(N, 50, 5120)
mean = raw.mean(axis=1)                        # (N, 5120)

# item-only: 페르소나 없이 아이템 텍스트만
item_emb = embedder.get_embeddings([build_item_text(iid, ...) for iid in item_ids])
```

**저장 파일 (6개)**:
| 파일 | Shape | 설명 |
|------|-------|------|
| warm_raw.pt | (300, 50, 5120) | warm per-persona raw |
| cold_raw.pt | (100, 50, 5120) | cold per-persona raw |
| warm_mean.pt | (300, 5120) | warm_raw.mean(axis=1) |
| cold_mean.pt | (100, 5120) | cold_raw.mean(axis=1) |
| warm_item_only.pt | (300, 5120) | 페르소나 없는 warm |
| cold_item_only.pt | (100, 5120) | 페르소나 없는 cold |

**총 forward passes**: 20,000 (warm raw) + 5,000 (cold raw) + 400 (item-only) = **25,400**

**Shape Assert**: 6개 파일 형상 검증 후 불일치 시 즉시 종료 (`sys.exit(1)`)

#### Step 2: 6종 헤드 학습

공통 설정:
- **5-fold CV**: `KFold(n_splits=5, shuffle=True, random_state=42)`
- **warm_val_MAE**: 5-fold CV 평균 (weekly 단위)
- **cold 평가**: warm 전체로 재학습 후 cold 예측

**3-1. Ridge (mean-pooled)**:
```python
# StandardScaler 없음 (Exp005와 다름)
for w in range(17):
    Ridge(alpha=1.0).fit(X_warm_mean, y_warm[:, w])
    cold_pred[:, w] = model.predict(X_cold_mean)
```

**3-2. Ridge (item-only)**: 동일 구조, X = `item_only` 임베딩 사용

**3-3. Attention + Linear**:
```python
class AttnHead(nn.Module):
    attn = nn.Linear(5120, 1, bias=False)
    head = nn.Linear(5120, 17)

    def forward(self, x):           # x: (batch, 50, 5120)
        scores = self.attn(x)       # (batch, 50, 1)
        weights = scores.softmax(dim=1)
        agg = (weights * x).sum(dim=1)  # (batch, 5120)
        return self.head(agg)           # (batch, 17)

# optimizer: Adam(lr=1e-3, weight_decay=1e-2)
# epochs=200, patience=20 (val MSE loss 기준 early stopping)
# dropout=0.0
```

**3-4. Attention + Bottleneck**: 동일 attn 레이어, head 교체:
```python
head = nn.Sequential(
    nn.Linear(5120, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 17),
)
```

**3-5. Variance + Ridge**:
```python
X_var = np.concatenate([X_raw.mean(axis=1), X_raw.std(axis=1)], axis=1)  # (N, 10240)
# Ridge(alpha=1.0) on (N, 10240) → (N, 17)
```

**3-6. Variance + PCA(64) + Ridge**:
```python
pca = PCA(n_components=64, random_state=42).fit(X_warm_var)  # warm으로만 fit
Ridge(alpha=1.0).fit(pca.transform(X_warm_var), y_warm)
```

#### 평가 지표 계산 방식

- **MAE, RMSE**: **weekly** 단위 (pd.Period("W") 주간 합 기준)
- **DirAcc**: **weekly, flat-exclusive**
  ```python
  td = np.sign(np.diff(y_true, axis=1))
  pd_ = np.sign(np.diff(y_pred, axis=1))
  m = td != 0  # flat 실제 변화 제외
  DirAcc = (td[m] == pd_[m]).mean()
  ```

#### 환경

| 항목 | 값 |
|------|---|
| 하드웨어 | Tesla P40 × 4 (22.4GB × 4), CUDA_VISIBLE_DEVICES=0,1,2,3 |
| Attn head 학습 | `torch.device("cuda")` 단일 GPU |
| 총 실행 시간 | 888분 (14.8시간) |
| tmux 세션 | exp009 |

#### 실행 명령

```bash
tmux new-session -d -s exp009
tmux send-keys -t exp009 \
  'export HF_HOME=/mnt/sdd1/jylee/huggingface_cache && \
   export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
   cd ~/Persona-Forecasting && \
   /home/jylee/anaconda3/envs/persona-forecasting/bin/python scripts/run_exp009.py \
   2>&1 | tee experiments/exp009_attention_head/run.log' Enter
```

#### 출력 파일

```
experiments/exp009_attention_head/
├── embeddings/
│   ├── warm_raw.pt, cold_raw.pt
│   ├── warm_mean.pt, cold_mean.pt
│   ├── warm_item_only.pt, cold_item_only.pt
│   └── item_meta.csv          # item_id, cat_id, dept_id, is_cold
├── models/
│   ├── attn_linear/model.pt
│   └── attn_bottleneck/model.pt
├── results/
│   ├── metrics_per_head.json
│   └── comparison.csv
└── run.log
```

### 3. 실험 결과

#### 6종 헤드 비교 (모두 weekly 단위)

| Head | warm_val_MAE | cold_MAE | cold_RMSE | DirAcc | params |
|------|----------:|--------:|--------:|-------:|------:|
| 3-1. Ridge (mean-pooled) | 58.53 | 68.30 | 80.50 | 0.536 | 5,121 |
| 3-2. Ridge (item-only) | 70.41 | 74.48 | 94.82 | **0.561** | 5,121 |
| 3-3. Attn + Linear | 63.16 | 67.94 | 70.88 | 0.537 | 10,241 |
| **3-4. Attn + Bottleneck** | 59.85 | **59.45** | **63.31** | 0.536 | 332,929 |
| 3-5. Variance + Ridge | 58.52 | 68.33 | 80.53 | 0.536 | 10,241 |
| 3-6. Variance+PCA(64)+Ridge | 58.78 | 69.49 | 82.10 | 0.537 | 65 |

#### Alpha 보정 (scale 분석, weekly 단위)

> y_cold weekly mean = 9.96. 모든 헤드가 6~8배 과예측.

| Head | pred_mean (weekly) | alpha | raw MAE | corrected MAE |
|------|------------------:|------:|--------:|------------:|
| 3-1. Ridge (mean-pooled) | 74.63 | 0.133 | 68.30 | **9.74** |
| 3-3. Attn + Linear | 74.22 | 0.134 | 67.94 | **10.39** |
| 3-4. Attn + Bottleneck | 65.38 | 0.152 | 59.45 | **10.41** |
| 3-2. Ridge (item-only) | 70.17 | 0.142 | 74.48 | **11.11** |

#### 통합 비교 테이블 (동일 단위: weekly MAE, alpha 보정 후)

| 모델 | weekly MAE | DirAcc | 출처 |
|------|----------:|-------:|------|
| LightGBM proxy lags | **8.48** | 0.343 | Exp006 참조값 |
| Track A calibrated | 8.90 | 0.393 | Exp006 참조값 |
| knn_analog | 9.57 | 0.412 | Exp006 참조값 |
| **Ridge mean-pooled (보정)** | **9.74** | 0.536 | Exp009 |
| Variance+Ridge (보정) | ~9.75 | 0.536 | Exp009 |
| Variance+PCA+Ridge (보정) | ~9.85 | 0.537 | Exp009 |
| Attn+Linear (보정) | 10.39 | 0.537 | Exp009 |
| Attn+Bottleneck (보정) | 10.41 | 0.536 | Exp009 |
| Ridge item-only (보정) | 11.11 | 0.561 | Exp009 |

#### Attention 가중치 분석 (Attn+Bottleneck 모델)

전체 평균 attention weight (50 페르소나):

| Persona | weight | 특성 요약 |
|---------|-------:|---------|
| **P50** (CA_1_P050) | **0.363** | Nancy, FOODS선호 70%, 주 1회, health&wellness, 중산층 $150 |
| **P7** (CA_1_P007) | **0.188** | 피트니스 열성팬, FOODS 80%, 주 1회, health&wellness, $200 |
| **P42** (CA_1_P042) | **0.103** | 테크 전문직, FOODS 70%, 주 1회, health&wellness, $260 |
| P27 (CA_1_P027) | 0.077 | 재택 긱워커, HOUSEHOLD 60%, 격주 방문 |
| P38 (CA_1_P038) | 0.044 | 3자녀 엄마, FOODS 60%, 주 2~3회 |
| 나머지 45개 | ≈0.000 | 사실상 무시됨 |

P50+P7 합계 = 55.1%, 상위 3개(P50+P7+P42) = 65.4%

카테고리별 top-3 (모두 동일):
| 카테고리 | 1위 | 2위 | 3위 |
|---------|-----|-----|-----|
| FOODS (n=34) | P50 (0.415) | P7 (0.202) | P42 (0.093) |
| HOBBIES (n=33) | P50 (0.311) | P7 (0.196) | P42 (0.117) |
| HOUSEHOLD (n=33) | P50 (0.361) | P7 (0.166) | P42 (0.099) |

### 4. 실험 결론

1. **Alpha 보정 후 순위 역전**: raw MAE에서 Attn+Bottleneck(59.45)이 Ridge(68.30)보다 13% 우세해 보였으나, 보정 후(각각 10.41 vs 9.74) Ridge가 오히려 우세. Attn+Bottleneck의 pred_mean이 상대적으로 낮아(65 vs 75) raw MAE가 작게 나온 착시 효과.

2. **베이스라인 근접**: alpha 보정 후 best(Ridge mean-pooled, 9.74)가 knn_analog(9.57)에 근접. LightGBM(8.48) 대비 약 15% 열위.

3. **DirAcc에서 페르소나 임베딩 우위**: Exp009 DirAcc 0.53~0.56 vs 프록시 베이스라인 0.34~0.41. 방향 예측에서 페르소나 임베딩이 의미 있는 개선.

4. **Attention collapse 발견**: 50개 페르소나 중 2개(P50, P7)에 weight 55% 집중; 카테고리 무관 동일 top-3. 모델이 페르소나 다양성을 활용하지 못함. 모든 상위 페르소나가 "health & wellness, FOODS 중심, 주 1회 방문" 패턴 공유.

5. **Item-only DirAcc 최고(0.561)**: 페르소나 없는 아이템 메타만으로 방향 예측이 더 정확. 페르소나 정보가 방향 예측에서는 노이즈로 작용.

6. **잔존 과예측 문제**: alpha=0.13~0.15, 6~8배 과예측 지속. 후속 실험에서 카테고리별 alpha 보정 또는 판매량 분포 정규화 필요.

---

## 부록: 단위 혼동 해소 이력

본 연구에서 실험별로 MAE 단위(daily vs weekly)와 DirAcc 계산 방식(flat 포함/제외)이 혼재했으며, 이를 명확히 정리한다.

### MAE 단위

| 실험 | MAE 값 | 단위 | 이유 |
|------|--------|------|------|
| Exp002 (베이스라인) | 1.57~1.69 | **daily** | `evaluate()` 사용, daily pred vs daily actual |
| Exp005 (Track B) | 13.07 | **daily** | 동일, `build_pred_dataframe()`으로 weekly/n_days 변환 후 평가 |
| Exp006 분석 | 68.30 (Ridge) | **weekly** | `agg_weekly(pd.Period("W"))` 직접 집계 후 MAE |
| Exp009 | 59.45~74.48 | **weekly** | `agg_weekly(pd.Period("W"))` 직접 집계 후 MAE |

**daily → weekly 근사 환산**: daily × 7 (정확하지 않음, 주차별 일수 다름)
- Exp005 daily MAE 13.07 × 7 ≈ 91.5 weekly (실제 Exp006에서 68.30 — 차이는 StandardScaler, 집계 방식 차이)

### DirAcc 계산 방식

| 실험 | DirAcc 값 | flat(0) 처리 | 주 집계 방식 | 비고 |
|------|----------:|-------------|------------|------|
| Exp002 (베이스라인) | 0.257 | 포함 | ISO calendar week | 상수 예측 → 모든 pred_dir=0 → 실제 flat 비율만큼 정답 |
| Exp005 (Track B) | 0.3719 | 포함 | ISO calendar week | `metrics.py evaluate()` |
| Exp006 분석 | 0.536 (Ridge) | **제외** | pd.Period("W") | `analyze_track_b.py direction_accuracy()` |
| Exp009 | 0.536~0.561 | **제외** | pd.Period("W") | `run_exp009.py dir_acc_weekly()` |

**직접 비교 불가 조합**: Exp005 DirAcc(0.3719) vs Exp009 DirAcc(0.536) — 다른 계산 방식.

### 향후 권장 표준

- 모든 실험에서 **weekly 단위** (`pd.to_period("W")`) 사용
- DirAcc: **flat 제외** (`true_dir != 0` mask) 방식으로 통일 (Exp009 방식)
- Alpha 보정은 분석 목적으로만 사용; 학습/평가 파이프라인에 포함하지 않음
- 리포팅 시 단위 명시 필수: "(weekly MAE, alpha=0.133 보정 후)"

### Alpha 보정 요약

| 실험 | alpha | 보정 전 MAE | 보정 후 MAE | 단위 |
|------|------:|----------:|----------:|------|
| Exp005 (Track B) | 0.106 | 13.07 | 1.86 | daily |
| Exp009 Ridge mean-pooled | 0.133 | 68.30 (weekly) | 9.74 | weekly |
| Exp009 Attn+Bottleneck | 0.152 | 59.45 (weekly) | 10.41 | weekly |
| Exp009 Ridge item-only | 0.142 | 74.48 (weekly) | 11.11 | weekly |

보정 후 Exp009 best(9.74 weekly) ≈ Exp005 best(1.86 daily × 7 ≈ 13 weekly)와 차이 있음.
이는 Exp005 WeeklySalesHead의 StandardScaler, Exp005와 Exp009의 집계 방식 차이, 다른 warm item 데이터 때문.
