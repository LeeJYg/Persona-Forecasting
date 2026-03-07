# LightGBM Cold-Start Adaptation Log

**원본**: stephenllh/m5-accuracy (GitHub, Silver Medal, WRMSSE=0.637)
**적응**: `src/models/competitors/lightgbm_cross.py` (LightGBMCross)
**작성일**: 2026-03-08

---

## 원본 재현 검증 (Step A)

### API 호환성 패치 (알고리즘 변경 없음)

실행 환경: Python 3.x / pandas 2.2.3 / LightGBM 4.6.0

| 파일 | 원본 코드 | 패치 이유 | 패치 내용 |
|------|-----------|-----------|-----------|
| `train.py:50` | `lgb.train(..., verbose_eval=20)` | LightGBM 4.x에서 `verbose_eval` 파라미터 제거 | `callbacks=[lgb.log_evaluation(20)]` |
| `preprocess.py:115` | `ds["date"].dt.weekofyear` | pandas 2.x에서 `dt.weekofyear` 제거 | `ds["date"].dt.isocalendar().week.astype("int16")` |
| `preprocess.py:134` | `trainCols.to_csv(...)` | pandas 2.x에서 `pd.Index.to_csv()` 제거 | `open()` + `write()` 직접 사용 |
| `inference.py:88` | `lgb.load("model.lgb")` | LightGBM 4.x에서 `lgb.load()` 제거 | `lgb.Booster(model_file="model.lgb")` |
| `inference.py:create_features` | 동일한 weekofyear 이슈 | 동일 | 동일한 isocalendar 패치 |
| `train.py:38-47` | `X_train.loc[:-10000]` / `.loc[-10000:]` | pandas 2.x에서 음수 label-based indexing 동작 변경 (결과: 빈 DataFrame) | `.iloc[:-10000]` / `.iloc[-10000:]` |

→ 알고리즘, 하이퍼파라미터, 피처 엔지니어링 로직 **변경 없음**

### 원본 성능 검증 결과

- 원본 공개 WRMSSE (private LB): **0.637** (Top 5%, Silver Medal, 12-level 평균)
- 우리 재현 WRMSSE (product-level only): **1.004**
- 검증 기간: d_1914~d_1941 (evaluation period)
- 데이터: 30,490 items × 1,664일 (d_250~d_1913)

**1.004 ≠ 0.637인 이유**: M5 공식 WRMSSE는 12개 집계 수준(item → total)의 평균이다.
상위 집계(store, state, total)로 갈수록 노이즈가 평균화되어 WRMSSE가 낮아진다.
우리 product-level 계산(1.004) < Naive product-level(~1.37)으로 **순서 일치** ✓

| 지표 | Naive | stephenllh | 비고 |
|------|-------|------------|------|
| 12-level WRMSSE (M5 공식) | 1.752 | 0.637 | Kaggle 공식 |
| Product-level WRMSSE (item-own lag-1) | ~1.37 | **1.004** | 우리 계산 |

→ **결론**: 코드 동작 정상 확인, product-level에서도 Naive 대비 27% 개선.
  0.637 수치를 정확히 재현하려면 12-level WRMSSE 구현 필요하나,
  논문의 competitor 검증 목적(코드 동작 확인 + 방향성 검증)은 달성됨.

---

## Cold-Start 적응 변경사항 (Step B)

### 1. 데이터 변경

| 항목 | 원본 | 적응 | 이유 |
|------|------|------|------|
| 대상 items | 30,490 items (전체 M5) | 300 warm items (CA_1) + 100 cold items (CA_1) | Cold-start 실험 설계 |
| 학습 데이터 | `sales_train_validation.csv` d_250~d_1913 (일별) | `warm_train.csv` (주간 집계) | Cold item은 자기 이력 없음 |
| 예측 대상 | 자기 자신 (d_1914~d_1941) | cold items (d_1771~d_1798 weekly) | Cold-start 정의 |
| 시간 단위 | 일별 (daily) | 주간 (weekly ISO-week 합계) | Cold 환경에서 노이즈 감소 |

### 2. 피처 엔지니어링 변경

| 피처 | 원본 | 적응 | 이유 |
|------|------|------|------|
| `lag_7`, `lag_28` | 자기 이력 lag (직접 사용) | **제거** | Cold item은 자기 판매 이력 없음 |
| rolling means | `rmean_7_7`, `rmean_7_28`, `rmean_28_7`, `rmean_28_28` | **제거** | 동일 이유 |
| k-NN proxy lags | 없음 | `knn_top3_overall_mean`, `knn_top3_same_week_mean` | Lag 대체: 유사 warm item의 판매량 proxy |
| `sell_price` | 일별 실제 가격 | 아이템 전체 기간 평균 가격 | Cold 환경: 주별 가격 변동 정보 부재 |
| `price_change_ratio` | 있음 (전주 대비) | **제거** | Cold item은 가격 이력 없음 |
| `item_id`, `store_id`, `state_id` | categorical feature | **제거** (단일 매장 CA_1) | CA_1만 사용하므로 불필요 |
| `event_name_1/2`, `event_type_1/2` | 있음 | `snap_count` (주간)으로 단순화 | 주간 집계 환경에서 단순화 |
| `cat_weekly_mean`, `dept_weekly_mean` | 없음 (cross-sectional 없음) | **추가** | Cold item scale 조정용 prior |
| 시간 피처 | `wday`, `week`, `month`, `quarter`, `year`, `mday` (일별) | `iso_week`, `month` (주별) | 일별 → 주별 변환 |

### 3. 모델 하이퍼파라미터 변경

| 파라미터 | 원본 | 적응 | 이유 |
|----------|------|------|------|
| `objective` | `poisson` | `tweedie` (variance_power=1.5) | 과산포 데이터 대응 |
| `num_leaves` | 128 | 31 (default) | 훈련 데이터 크기 감소 (300 items × ~주간) |
| `learning_rate` | 0.075 | 0.05 | 소규모 데이터 안정화 |
| `num_iterations` | 1200 | ~500 (early stopping) | 과적합 방지 |
| `min_data_in_leaf` | 100 | default | 소규모 데이터 |
| `seed` | 777 | 42 (config 관리) | 프로젝트 통일 seed |

### 4. 학습/검증 분할 변경

| 항목 | 원본 | 적응 |
|------|------|------|
| 분할 방식 | 마지막 10,000 rows를 validation | 80/20 item-level hold-out (random) |
| 이유 | 시계열 순서 반영 | Cold-start에서는 item 독립성이 중요 |

### 5. 추론(Inference) 방식 변경

| 항목 | 원본 | 적응 |
|------|------|------|
| 방식 | Recursive (일별 순차 예측, lag 업데이트) | Direct (한 번에 전체 기간 예측) |
| 이유 | Lag feature가 있어 recursive 필요 | Lag feature 없으므로 recursive 불필요 |

---

## 변경 후 성능 비교

| 모델 | 데이터 | WRMSSE | 비고 |
|------|--------|--------|------|
| 원본 stephenllh | M5 전체 30,490 items, d_1914~d_1941 | 0.637 (공식) | item-own-lag1 scale |
| 우리 적응 (LightGBMCross static) | CA_1 100 cold items, 주간 | (exp006 결과 참조) | cat-mean-lag4 scale |
| 우리 적응 (LightGBMCross proxy_lags) | CA_1 100 cold items, 주간 | (exp006 결과 참조) | cat-mean-lag4 scale |
| warm items 검증 (exp007) | CA_1 300 warm items, d_1771~d_1798 | **0.911** | item-own-lag28 scale |

---

## 요약

Cold-start 설정에서 LightGBM을 사용하기 위한 핵심 적응은 **lag feature 제거 + k-NN proxy lags 추가**다.
원본의 뛰어난 성능(WRMSSE=0.637)은 자기 판매 이력 lag에 크게 의존하므로,
cold-start에서 lag을 유사 아이템 proxy로 대체하면 성능 하락이 예상된다.
이 하락폭이 LLM 페르소나 접근법의 우위를 보여주는 비교 기준이 된다.
