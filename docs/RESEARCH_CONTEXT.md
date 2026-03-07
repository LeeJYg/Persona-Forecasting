# Research Context: LLM Persona Cold-Start Forecasting

## 1. 연구 배경 및 동기

### 데이터셋: M5 (Walmart)
- 3,049개 상품, 10개 매장, 2011-2016 일별 판매량
- 카테고리: Hobbies / Foods / Household
- 외부 변수: 가격(sell_prices.csv), 프로모션, 특별 이벤트(calendar.csv)
- **한계**: 구매자 ID가 없음 → 개인 수준 구매 이력 불가 → 합성 데이터 필요

### Cold-Start 정의
- **신규 매장 오픈 시점**: 해당 매장에서 특정 상품의 판매 이력이 전혀 없는 상황
- M5 데이터에서 시뮬레이션: 특정 매장의 판매 시작 N주 이내 데이터만 사용
- Foods 카테고리는 반복 구매 상품이 많아 cold-start 효과가 약할 수 있음 (Limitations에 명시)

### 핵심 선행 연구
- **Tan et al. (2024)**: LLM 페르소나 에이전트가 정량적 예측 태스크에서 어려움을 겪음을 입증
- **Park et al. (2023)**: Generative Agents - LLM 기반 페르소나 시뮬레이션의 가능성 제시
- **Chronos (Amazon, 2024)**: LLM 시계열 예측에서 스케일링 방식(절대 평균 기반) 제안
- **Li et al. (2025)**: "LLM Generated Persona is a Promise with a Catch" (NeurIPS 2025)
  - 페르소나 생성 4-tier 분류: Meta → Objective Tabular → Subjective Tabular → Descriptive
  - 핵심 발견: LLM 생성 내용이 많아질수록 시뮬레이션 결과가 현실에서 체계적으로 벗어남
  - 원인: LLM이 긍정적, 진보적, 동질적 페르소나를 편향적으로 생성
  - 우리 연구 적용: 예측 시 description(서사)보다 구조화 필드 우선 제공이 안전
  - 차별점: 이 논문은 의견(opinion) 시뮬레이션, 우리는 수량(quantity) 예측 → 물리적 제약(예산, 가격)이 편향을 제한할 수 있는지 검증
  - Task 4 설계 근거: Structured Only / Structured+Narrative / Narrative Only 3조건 비교

## 2. 연구 설계: Phase 1 → 2 → 3 순차적 접근

### Phase 1: Feasibility Test (현재 단계)
- **목표**: LLM 페르소나가 cold-start 예측에서 단순 베이스라인보다 의미있게 나은가?
- **방법**:
  1. M5에서 cold-start 아이템 샘플링 (카테고리 균형, 판매량 tier 균형)
  2. 합성 페르소나 생성 (M5 가격/카테고리 데이터 기반)
  3. LLM에게 페르소나 역할로 구매 의사결정 시뮬레이션 요청
  4. 베이스라인과 비교 (카테고리 평균, 유사 아이템 평균)
- **판단 기준**:
  - 유의미하게 우수 → Phase 1에서 논문 가능 (왜 작동하는지 분석)
  - 유사 → 어떤 정보를 포착하고 놓치는지 분석
  - 유의미하게 열등 → Phase 2로 진행 (실패 원인 진단)

### Phase 2: Diagnosis (Phase 1에서 열등한 경우만)
- 실패 원인이 텍스트 표현(narrativization)인지, LLM 추론 한계인지 분리
- 2×2 factorial design: {Textual vs Latent} × {LLM vs Statistical}

### Phase 3: Solution (Phase 1, 2 결과에 따라 방향 결정)

## 3. 비판적 리뷰에서 지적된 핵심 문제들

### 문제 1: 합성 데이터의 순환 논리
- M5에는 구매자 ID가 없으므로 합성 마이크로데이터를 생성해야 함
- 그러나 합성 데이터는 연구자가 만든 것이므로 "통제 변수"라고 주장하면 안 됨
- **대응**: 합성 데이터는 통제 변수가 아니라 "실험 조건의 일부"로 명시.
  합성 과정의 가정을 투명하게 공개하고, sensitivity analysis 수행

### 문제 2: 비교 공정성
- Textual Persona vs Latent Vector 비교 시, 텍스트 쪽이 strawman이 되면 안 됨
- **대응**: 동일한 정보량을 양쪽에 공급. 텍스트에만 추가 정보가 있거나 빠지면 안 됨

### 문제 3: Tan et al. (2024) 대비 차별성
- "Representation vs Calculation" 구분이 단순한 말장난이라는 비판
- **대응**: Phase 1에서 먼저 실증적 결과를 확보한 후, 이론적 프레이밍은 결과에 맞춰 조정

### 문제 4: Reverse Inference의 사전 지식 교란
- LLM이 페르소나 텍스트에서 정보를 추출할 때, LLM 자체의 사전 지식이 개입
- **대응**: null baseline 필요 (빈 페르소나를 주었을 때의 복원 정확도 측정)

## 4. 합성 페르소나 생성 요구사항

### 현재 문제 (OpenClaw가 만든 코드의 한계)
- 페르소나 2개만 생성 (P1, P2) → 통계적 검정력 부족
- 영수증만 생성, 멀티모달 아님
- 카테고리 선호도, 방문 패턴 등 페르소나 특성이 구매에 미반영
- 날짜 생성이 단순 등간격 (비현실적)

### 필요한 합성 페르소나 구조
```json
{
  "persona_id": "CA_1_P001",
  "store_id": "CA_1",
  "profile": {
    "description": "주중 저녁에 주로 방문하는 가족형 소비자...",
    "category_preference": {"FOODS": 0.6, "HOUSEHOLD": 0.3, "HOBBIES": 0.1},
    "price_sensitivity": "medium",
    "visit_frequency": "2-3 times/week",
    "preferred_departments": ["FOODS_1", "FOODS_2", "HOUSEHOLD_1"]
  },
  "purchase_history": [
    {
      "date": "2016-01-15",
      "items": [...],
      "total_spent": 23.45
    }
  ]
}
```

### 생성 원칙
- 최소 50개 페르소나 (Phase 1 feasibility test용)
- 카테고리 선호도가 실제 구매에 반영되어야 함
- M5의 실제 가격 데이터 사용
- 구매 패턴에 요일/계절 효과 반영
- seed 고정으로 재현 가능

## 5. 베이스라인 정의 및 결과

### Baseline 1: Global Category Average
- 동일 카테고리 상품들의 전체 평균 판매량

### Baseline 2: Similar Item Average
- 동일 카테고리 + 유사 가격대 상품들의 평균 판매량

### Baseline 3: Store-Category Average
- 동일 매장 + 동일 카테고리의 평균 판매량

### 베이스라인 실험 결과 (exp002, CA_1 100 cold items)

| 모델 | MAE | WRMSSE(cat-mean-lag4) | DirAcc |
|------|-----|----------------------|--------|
| Global Category Average | 1.57 | 2.96 | 0.257 |
| Similar Item Average | 1.69 | 2.99 | 0.257 |
| Store-Category Average | ~1.6 | ~2.97 | 0.257 |

- DirAcc 0.257 = LLM 페르소나가 뛰어넘어야 할 하한선
- FOODS 오차(MAE ~2.7)가 HOBBIES·HOUSEHOLD(1.0~1.2)보다 2~3배 큼
- 세 모델의 DirAcc가 모두 동일한 이유: 방향성 예측을 전혀 못하는 flat 예측과 동일

## 6. 평가 지표 정의

### WRMSSE 변형 3종 (혼동 주의)

| 변형 | scale_i | 집계 | 가중치 | 사용 맥락 |
|------|---------|------|--------|-----------|
| **M5 공식 12-level** | item-own lag-1 일별 MSE | 12개 집계 수준 평균 | 달러 판매액 | Kaggle 공식 평가 |
| **우리 cold-start** | cat-mean lag-4 주간 MSE | product-level만 | 판매량(단위) | exp002~exp006 결과 비교 |
| **벤치마크 비교용** | item-own lag-1 일별 MSE | product-level만 | 판매량(단위) | exp007~exp008 warm item 검증 |

- M5 공식 12-level WRMSSE: 아이템→부서→카테고리→매장→주→전국 집계 12단계 평균.
  상위 집계일수록 개별 노이즈가 평균화되어 WRMSSE가 낮아짐.
  stephenllh silver medal 0.637 (12-level) ↔ product-level 1.004 (우리 계산) — 동일 코드, 다른 집계.

### DirAcc (Direction Accuracy) 정의
- 주간 판매량 변화 방향(증가/감소/flat)의 예측 정확도
- 구현: `src/evaluation/metrics.py::direction_accuracy_weekly()`
  ```python
  diff = groupby("item_id").diff()   # 아이템 내 주간 차분 (경계 방지)
  sign = np.sign(diff)               # -1, 0(flat), +1
  dropna()                           # 첫 주 NaN 제거
  match = (sign_pred == sign_actual) # flat==flat → 정답 처리
  ```
- **flat 처리**: `np.sign(0)=0`, `0==0=True` → flat 예측이 flat 실제와 일치 시 정답으로 처리
- **첫 주 NaN 처리**: `groupby().diff()` 후 첫 주는 NaN → `dropna()`로 제거

### 데이터 누수 검증 결과 (Phase 4-A, 2026-03-08)

모든 competitor 모델에서 cold item 실제 판매 데이터가 학습/feature에 포함되지 않음 확인:
- `run_competitors.py::load_data()`: `~item_id.isin(cold_ids)` 명시적 필터
- `KNNAnalog.fit()`: warm_train + sell_prices (static) 만 사용
- `KNNAnalog.predict()`: cold item의 item_id/cat_id/dept_id/sell_price (static) 만 사용 (sales 불사용)
- `LightGBMCross.fit/predict()`: warm 기반 cat/dept mean 사용, cold sales 불사용
- `print_checklist()`: 런타임 시 `cold_ids & warm_ids` 교집합 = 0 확인

## 7. Cold-Start 실험 설계 상세

### 데이터 설정
- **대상 매장**: CA_1 (California 1)
- **Cold items**: 100개 (CA_1 내 카테고리 균형 샘플링)
- **Warm items (전체)**: 2,949개 (CA_1 내 cold 제외 전체)
- **Warm items (학습용 서브샘플)**: 300개 (exp007에서 속도 목적)
- **학습 기간**: d_1 ~ d_1770 (2011-01-29 ~ 2015-10-15)
- **테스트 기간**: d_1771 ~ d_1798 (2015-10-16 ~ 2015-11-12, 약 16~17주)
- **시간 단위**: 주간 (ISO-week 집계), 일별 데이터를 주별 합산
- **cross_store_info**: false (타 매장 정보 사용 안 함)

### 실험 조건 정의
- Cold item = CA_1에서 해당 상품의 판매 이력이 d_1770 이후에만 존재하는 아이템으로 시뮬레이션
- Warm item = d_1770 이전에 충분한 판매 이력이 있는 아이템

## 8. LLM 예측 파이프라인 (Task 4)

### 예측 방식: Track A (집계 예측)
- LLM에 페르소나 정보 + 상품 정보를 주고 주간 판매량을 직접 예측
- 50개 페르소나 × 100개 cold items × 16주 = 80,000 예측 포인트 (이론적)
- 실제는 페르소나당 일부 아이템만 예측 후 집계

### 페르소나 정보 제공 3조건 (Li et al. 2025 반영)
| 조건 | 제공 정보 | 근거 |
|------|-----------|------|
| **Condition A (Structured Only)** | JSON 구조화 필드만 (weekly_budget, snap_eligible, category_preference 등) | Phase 1 기본 실험 |
| **Condition B (Structured + Narrative)** | 구조화 필드 + description 서사 | 추가 실험 |
| **Condition C (Narrative Only)** | description 서사만 | 추가 실험 |

- Phase 1에서는 **Condition A**로 먼저 feasibility 검증
- Li et al. (2025): 서사(narrative)가 많을수록 시뮬레이션 편향 증가 → 구조화 필드 우선

### 페르소나 스키마 (현재 구현: `src/models/persona/schema.py`)
- `description`: Show/Don't Tell 방식 서사 (100-200자)
- `weekly_budget`: 주간 쇼핑 예산 (달러)
- `snap_eligible`: SNAP 수급 여부 (bool)
- `shopping_motivation`: 구매 동기 (예: "routine", "stockpile", "treat")
- `economic_status`: 경제 상태
- `category_preference`: 카테고리별 선호도 dict
- `price_sensitivity`: 가격 민감도
- `visit_frequency`: 방문 빈도
- `preferred_departments`: 선호 부서 목록
- `decision_style`: 결정 스타일
- `brand_loyalty`: 브랜드 충성도
- `promotion_sensitivity`: 프로모션 민감도

## 9. Competitor 모델 목록 및 검증 현황

Phase 1에서 LLM 페르소나와 비교할 competitor 모델들:

| 모델 | 유형 | 원본 | 검증 상태 | 비고 |
|------|------|------|-----------|------|
| **KNNAnalog** | k-NN | 자체 구현 | ✓ exp006 실행 | cosine similarity, warm item ISO-week 평균 |
| **LightGBMCross (static)** | GBDT | stephenllh/m5-accuracy (Silver Medal) | ✓ exp006+exp007 | lag 없이 cat/dept mean feature |
| **LightGBMCross (proxy_lags)** | GBDT | 동일 | ✓ exp006 | k-NN top-3 proxy lag 추가 |
| **DeepAR** | RNN | Amazon GluonTS | 미완 | Phase 1 실험 예정 |
| **LLMTime** | LLM | Darts/원논문 | 미완 | Phase 1 실험 예정 |

### stephenllh/m5-accuracy 원본 재현 (exp009, 2026-03-08)
- **원본**: GitHub Silver Medal, M5 Kaggle WRMSSE=0.637 (12-level 공식)
- **재현 환경**: Python 3.x, pandas 2.2.3, LightGBM 4.6.0
- **데이터**: 30,490 items × 1,664일 (d_250~d_1913), 일별
- **재현 WRMSSE(product-level, item-own-lag1)**: **1.004** (product-level만 계산했으므로 0.637과 다름)
- **결론**: 코드 정상 동작 확인. Naive product-level(~1.37) > stephenllh product-level(1.004) — 방향 일치 ✓
- **API 호환성 패치 6건** (알고리즘 변경 없음):
  - `verbose_eval=20` → `callbacks=[lgb.log_evaluation(20)]` (LightGBM 4.x)
  - `dt.weekofyear` → `dt.isocalendar().week.astype("int16")` (pandas 2.x)
  - `pd.Index.to_csv()` → `open()` + `write()` (pandas 2.x)
  - `lgb.load()` → `lgb.Booster(model_file=...)` (LightGBM 4.x)
  - `X_train.loc[:-10000]` → `X_train.iloc[:-10000]` (pandas 2.x 정수 인덱스 음수 label 동작 변경)
  - `pd.read_csv(...).tolist()` → `.iloc[:, 0].tolist()` (DataFrame → list 변환)
- 상세 문서: `docs/verification/lgbm_adaptation_log.md`

### LightGBM Cold-Start 적응 (warm item 검증, exp007)
- 300 warm items, d_1771~d_1798 테스트
- item-own lag-28 WRMSSE: **0.911** (Naive 1.458, sNaive 1.686 대비 우수)
- 주간 집계, direct inference, tweedie objective, k-NN proxy lags

## 10. 실험 결과 요약

| 실험 | 대상 | 기간 | 모델 | WRMSSE | 스케일 | MAE | DirAcc |
|------|------|------|------|--------|--------|-----|--------|
| exp002 | CA_1 100 cold items | d_1771~d_1798 | Global Cat Avg | 2.96 | cat-mean-lag4 | 1.57 | 0.257 |
| exp002 | CA_1 100 cold items | d_1771~d_1798 | Similar Item Avg | 2.99 | cat-mean-lag4 | 1.69 | 0.257 |
| exp007 | CA_1 300 warm items | d_1771~d_1798 | LightGBM (proxy_lags) | 0.911 | item-own-lag28 | - | - |
| exp008 comp1 | CA_1 2949 items | d_1886~d_1913 | naive_official | 1.369 | item-own-lag1 | 1.571 | - |
| exp008 comp1 | CA_1 2949 items | d_1886~d_1913 | snaive_official | 1.318 | item-own-lag1 | 1.404 | - |
| exp008 comp2 | CA_1 warm items | d_1771~d_1798 | naive_exp007 | 1.458 | item-own-lag1 | 1.265 | - |
| exp009 | M5 전체 30,490 items | d_1914~d_1941 | stephenllh LightGBM | 1.004 | item-own-lag1 | 1.033 | - |

## 11. 타임라인

| 기간 | 작업 | 상태 |
|------|------|------|
| ~3월 중순 | Phase 1 데이터 전처리 + 베이스라인 구현 | ✓ 완료 |
| ~3월 중순 | Competitor 모델 구현 (KNN, LightGBM) | ✓ 완료 (exp006) |
| ~3월 중순 | stephenllh 원본 재현 검증 | ✓ 완료 (exp009, WRMSSE=1.004) |
| ~4월 중순 | 합성 페르소나 생성 + LLM 예측 파이프라인 | 진행 중 |
| ~4월 중순 | DeepAR, LLMTime competitor 추가 | 예정 |
| ~5월 중순 | Phase 1 실험 완료 + 결과 분석 | 예정 |
| ~6월 | 논문 초안 작성 | 예정 |
| ~7월 | 논문 수정 + 심사 제출 | 예정 |
