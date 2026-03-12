# M5 데이터셋 구조 및 Cold-Start 아이템 선정 방법

**작성일:** 2026-03-11
**관련 스크립트:** `scripts/preprocess_cold_start.py`, `src/data/cold_start.py`, `src/data/loader.py`
**설정 파일:** `configs/config.yaml`

---

## 1. M5 데이터셋 개요

M5 Forecasting — Accuracy 대회(Kaggle 2020)에서 제공된 Walmart 실제 판매 데이터셋.

### 1.1 원본 파일 구성

| 파일 | 설명 | 규모 |
|------|------|------|
| `sales_train_evaluation.csv` | 아이템별 일별 판매량 (wide format) | 30,490행 × (5 메타 + 1,941 날짜 컬럼) |
| `calendar.csv` | 날짜 메타 정보 (이벤트, SNAP 등) | 1,969행 |
| `sell_prices.csv` | 매장-아이템-주차별 판매가격 | 6,841,121행 |

### 1.2 계층 구조

```
Walmart
├── 주(state): CA, TX, WI  (3개)
│   └── 매장(store): CA_1~CA_4, TX_1~TX_3, WI_1~WI_3  (10개)
│       └── 아이템(item): 3,049개 SKU
│           ├── 카테고리(cat_id): FOODS, HOBBIES, HOUSEHOLD  (3개)
│           └── 부서(dept_id): FOODS_1/2/3, HOBBIES_1/2, HOUSEHOLD_1/2  (7개)
```

**전체 레코드**: 30,490 아이템 × 10 매장 = 30,490개 시계열
※ item_id는 모든 매장에 공통 — store-item 쌍이 실질적인 예측 단위

### 1.3 시간 범위

| 구간 | 날짜 | M5 컬럼 | 기간 |
|------|------|---------|------|
| **전체 기간** | 2011-01-29 ~ 2016-06-19 | d_1 ~ d_1,969 | 약 5.4년 |
| **학습 기간** | 2011-01-29 ~ 2016-04-24 | d_1 ~ d_1,913 | 약 5.2년 |
| **평가 기간** (공식 test) | 2016-04-25 ~ 2016-06-19 | d_1,914 ~ d_1,941 | 28일 |

### 1.4 CA_1 매장 구성

본 연구의 대상 매장(target_store = CA_1):

| 카테고리 | 아이템 수 | 비율 |
|---------|---------|------|
| FOODS | 1,437 | 47.1% |
| HOBBIES | 565 | 18.5% |
| HOUSEHOLD | 1,047 | 34.3% |
| **합계** | **3,049** | 100% |

### 1.5 판매량 특성

CA_1 기준 일별 판매량:
- 평균: **1.323개/일** (아이템-일 단위)
- 중앙값: **0** (희소 시계열)
- 최대: **648개/일**
- **0 비율: 63.8%** — 대부분의 날에 판매가 없음

---

## 2. Cold-Start 시뮬레이션 설계

### 2.1 연구 질문과 시뮬레이션 전제

> "신규 매장 오픈 시점에서 판매 이력이 전혀 없는 아이템의 수요를 LLM 페르소나로 예측할 수 있는가?"

이를 M5 데이터에서 시뮬레이션하기 위해:
- **Cold 아이템**: CA_1 매장에서 실제 판매 이력이 있지만, 학습 데이터에서 해당 이력을 **의도적으로 제거**하여 신규 아이템으로 간주
- **예측 대상**: 2016-01-01 ~ 2016-04-24 (115일 ≈ 16주)의 판매량
- **학습 가능 정보**: 나머지 warm 아이템들의 2011-01-29 ~ 2015-12-31 이력만 사용

### 2.2 Cold-Start 시뮬레이션 타임라인

```
2011-01-29                    2015-12-31  2016-01-01         2016-04-24
     │                              │          │                   │
     ├──────── warm 아이템 학습 기간 ────────────┤                   │
     │                                         ├─── cold 테스트 기간 ─┤
     │                                              (예측 대상: 115일)
     │
     └── cold 아이템: 이 기간 이력 완전 제거 (cross_store_info=False)
```

**핵심**: cold 아이템은 CA_1뿐 아니라 **다른 모든 매장의 이력도 제거** (cross_store_info=False). 즉, 어느 매장에도 이 아이템이 팔린 기록이 없는 것처럼 시뮬레이션.

---

## 3. Cold 아이템 선정 방법

### 3.1 샘플링 전략

전략 목표: **대표성 있는 100개 store-item 쌍** 선정
- 카테고리별 균형(category_balance=True): FOODS/HOBBIES/HOUSEHOLD 각 33~34개
- 판매량 티어별 층화 샘플링(tier_weights): High 25%, Medium 50%, Low 25%

### 3.2 단계별 선정 과정

```
[1단계] item_stats 계산 (src/data/cold_start.py: compute_item_stats)
        CA_1 매장의 모든 아이템(3,049개)에 대해 학습 기간 판매량 통계 계산:
        - total_sales, mean_sales, std_sales
        - sales_tier: pd.qcut(total_sales, q=3) → Low / Medium / High

[2단계] sample_cold_ids (src/data/cold_start.py: sample_cold_ids)
        카테고리 × 판매량 티어 격자(3×3=9 셀) 기준 층화 샘플링:
        - 카테고리별 기본 33개 + 나머지 1개(FOODS에 배분)
        - 각 카테고리 내에서 tier_weights에 따라 High/Medium/Low 할당
        - np.random.seed(42) 고정

[3단계] split (src/data/cold_start.py: split)
        cold 아이템 분리:
        - warm_train: cold 아이템 제거 (cross_store_info=False → 전 매장 제거)
        - cold_test: CA_1의 cold 아이템 테스트 기간 (2016-01-01 ~ 2016-04-24)
        - cold_train: 빈 DataFrame (학습 이력 없음)
```

### 3.3 샘플링 파라미터

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `n_cold_items` | 100 | cold 아이템 수 |
| `target_store` | CA_1 | 대상 매장 |
| `sampling_level` | store_item | item×store 쌍 단위 |
| `category_balance` | True | 카테고리 균형 |
| `tier_weights` | High:0.25 / Medium:0.50 / Low:0.25 | 판매량 티어별 비율 |
| `cross_store_info` | False | 다른 매장 이력도 차단 |
| `seed` | 42 | 재현성 |

### 3.4 선정 결과

**카테고리 분포:**
| 카테고리 | Cold 아이템 수 | CA_1 전체 | 샘플링 비율 |
|---------|-------------|---------|-----------|
| FOODS | 34 | 1,437 | 2.4% |
| HOBBIES | 33 | 565 | 5.8% |
| HOUSEHOLD | 33 | 1,047 | 3.2% |
| **합계** | **100** | **3,049** | **3.3%** |

**판매량 티어 분포:**
| 티어 | Cold 아이템 수 | 목표 비율 |
|-----|-------------|---------|
| High | 24 | 25% |
| Medium | 52 | 50% |
| Low | 24 | 25% |

**Cold 아이템 판매량 (테스트 기간, 115일):**
- 아이템당 평균 총 판매량: **169.3개**
- 아이템당 평균 일판매량: **1.472개/일**
- 판매량 = 0인 아이템: **2.0%** (전체 cold 아이템 중 2개)
- 최소 / 최대 총 판매량: 0 / 4,061개

---

## 4. 데이터셋 분리 결과

| 분리 집합 | 아이템 수 | 행 수 | 기간 |
|---------|---------|------|------|
| warm_train | 2,949개 item_id | 53,023,020 | 2011-01-29 ~ 2015-12-31 |
| warm_test  | 2,949개 item_id | 3,494,850  | 2016-01-01 ~ 2016-04-24 |
| cold_train | 0 (빈 DataFrame) | 0 | — |
| cold_test  | 100개 item_id    | 11,500     | 2016-01-01 ~ 2016-04-24 |

> **warm_test**: 베이스라인의 MASE 분모, WRMSSE 스케일 계산, Track B 헤드 학습 타겟으로 사용
> **cold_test**: 모든 모델의 최종 평가 데이터 (115일 × 100아이템 = 11,500행)

---

## 5. 관련 스크립트 및 파일 경로

### 실행 스크립트

| 역할 | 파일 |
|------|------|
| 전처리 실행 진입점 | `scripts/preprocess_cold_start.py` |
| Cold-start 샘플링 로직 | `src/data/cold_start.py` |
| M5 데이터 로더 | `src/data/loader.py` |
| 실험 설정 | `configs/config.yaml` |

### 출력 파일

| 파일 | 내용 |
|------|------|
| `data/processed/cold_start/cold_test.csv` | cold 아이템 테스트 데이터 (11,500행) |
| `data/processed/cold_start/warm_train.csv` | warm 아이템 학습 데이터 |
| `data/processed/cold_start/warm_test.csv` | warm 아이템 테스트 데이터 (헤드 학습 타겟) |
| `data/processed/cold_start/cold_ids.csv` | cold store-item id 목록 |
| `data/processed/cold_start/cold_item_stats.csv` | cold 아이템 판매량 통계 (tier 포함) |
| `data/processed/cold_start/metadata.json` | 샘플링 파라미터 및 분포 요약 |

### 실행 방법

```bash
# 기본 설정으로 실행
conda run --no-capture-output -n persona-forecasting \
  python scripts/preprocess_cold_start.py

# config 명시
conda run --no-capture-output -n persona-forecasting \
  python scripts/preprocess_cold_start.py --config configs/config.yaml
```

---

## 6. 설계 선택의 근거

### cross_store_info = False 선택 이유
실제 신규 아이템 출시 시나리오를 충실히 재현. 다른 매장에서도 동일 아이템 이력이 없는 **완전한 cold-start** 조건을 가정. cross_store_info=True는 미래 실험 조건(다른 매장 이력을 참조하는 warm transfer)을 위해 코드 수준에서 분기 구현됨.

### 판매량 티어 층화 샘플링 이유
- **High(25%)**: 임계값 이상의 고판매 아이템 포함 → 모델의 스케일 예측 능력 평가
- **Medium(50%)**: 일반적인 아이템 다수 → 평균 성능의 대표성 확보
- **Low(25%)**: 저판매·희소 아이템 포함 → 롱테일 예측 난이도 평가

단순 랜덤 샘플링 시 FOODS 고판매 아이템이 과대 대표될 수 있어 계층 구조를 사용.

### 평가 기간(115일) 선택 이유
M5 공식 test 기간(28일)보다 길게 설정하여 주간(weekly) 방향성 정확도(DirAcc) 계산에 충분한 주 수(16~17주)를 확보. 단, 2016-04-24(d_1913)까지만 사용하여 공식 evaluation 기간과 겹치지 않음.
