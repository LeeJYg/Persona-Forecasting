# 구현 검증 마스터 플랜: 모든 Competitor의 논문 기반 검증

## 배경과 목적

우리가 구현한 모든 competitor 방법론에 대해, **해당 방법론의 원 논문에서 보고한 성능을 우리 구현이 재현할 수 있는지** 체계적으로 검증한다.

이 검증이 필요한 이유:
- "우리가 구현한 LightGBM이 진짜 LightGBM의 성능을 내는가?"를 확인하지 않으면, competitor가 약해서 우리가 이긴 건지, competitor를 잘못 구현해서 이긴 건지 구분할 수 없다.
- 리뷰어는 반드시 "competitor를 공정하게 구현했는가?"를 공격한다.
- 우리 스스로도 결과를 신뢰할 수 없다.

---

## Phase 1: 각 Competitor의 논문 출처 정리

먼저 우리가 구현한 모든 방법론에 대해 다음을 정리해줘:
- 어떤 논문/방법론에서 가져왔는지
- 그 논문에서 M5 또는 유사 데이터셋으로 실험한 결과가 있는지
- 있다면 실험 설정(데이터 규모, 예측 horizon, 평가 지표, 보고된 수치)이 무엇인지
- 없다면 해당 방법론의 가장 대표적인 벤치마크가 무엇인지

### 검증 대상 목록

아래 표를 채워줘. 모르는 부분은 빈칸으로 두고 "확인 필요"라고 표시할 것.

| # | 우리 구현 모델명 | 원 논문/방법론 | 논문에서 M5 사용 여부 | 논문의 실험 설정 | 논문의 보고 성능 | 우리가 재현 가능한지 |
|---|-----------------|--------------|---------------------|----------------|----------------|-------------------|
| 1 | global_category_avg | Naive baseline (M5 대회 공식 벤치마크) | ? | ? | ? | ? |
| 2 | similar_item_avg | ? | ? | ? | ? | ? |
| 3 | store_category_avg | ? | ? | ? | ? | ? |
| 4 | seasonal_pattern | ? | ? | ? | ? | ? |
| 5 | knn_analog | ? | ? | ? | ? | ? |
| 6 | lightgbm_static | ? | ? | ? | ? | ? |
| 7 | lightgbm_proxy_lags | ? | ? | ? | ? | ? |
| 8 | deepar | Salinas et al. (2020) | ? | ? | ? | ? |
| 9 | llm_zero_shot | Gruver et al. (2023) LLMTime 변형 | ? | ? | ? | ? |
| 10 | llm_similar_item | 자체 설계 | N/A | N/A | N/A | N/A |
| 11 | llm_aggregate | 자체 설계 | N/A | N/A | N/A | N/A |
| 12 | Track A (persona) | 자체 설계 | N/A | N/A | N/A | N/A |
| 13 | Track B (hidden states + Ridge) | 자체 설계 | N/A | N/A | N/A | N/A |

---

## 핵심 원칙 (모든 Phase에 적용)

논문의 competitor로 사용하기 위한 필수 절차:

1. **원본 구현**: 원 논문의 핵심 방법론을 정확히 구현한다.
2. **원본 검증**: 논문의 벤치마크 데이터에서 논문이 보고한 성능과 동일한지 확인한다.
3. **적응 및 명시**: 검증된 구현을 우리 실험 설정(cold-start M5)에 맞게 수정하고, 모든 수정 사항(이유 포함)을 명시한다.

재현 불가능한 부분이 있으면 "검증 불가"로 기록하고 이유를 논문 Limitations에 명시한다.
"유사한 방법론"이나 "정신만 차용"은 competitor로 사용할 수 없다.

---

## Phase 2: 논문 기반 Competitor 원본 재현 및 적응

### 2-1. LightGBM

**원 논문/출처:** Makridakis et al. (2022), M5 공식 제출 예측값, stephenllh/m5-accuracy (Silver medal, WRMSSE 0.637)
**참고:** 1위 YJ_STU (WRMSSE 0.520)의 완전 재현은 매장별 10모델 × recursive/non-recursive 앙상블로 복잡도가 매우 높아 배제.

**단계 A-1: M5 공식 제출 예측값으로 CA_1 subset WRMSSE 계산 (모델 학습 불필요)**

목표: M5 공식 repository의 기제출 예측값(24개 벤치마크 + 상위 50팀)에서 우리 CA_1 300 items subset의 WRMSSE를 계산해 비교 기준을 확보한다.

1. GitHub: https://github.com/Mcompetitions/M5-methods 에서 예측값 CSV 다운로드
2. 우리 CA_1 300 warm items subset에 해당하는 행만 추출
3. 우리 평가 코드(metrics.py)로 subset WRMSSE 계산
4. 결과를 표로 정리: 벤치마크별 WRMSSE → 우리 exp007 결과(0.913)와 비교
5. 해석 기준:
   - 공식 벤치마크도 이 subset에서 0.9 근방 → 우리 데이터가 원래 어려운 것, 구현 정상
   - 공식 벤치마크가 0.5~0.6인데 우리만 0.9 → 우리 구현 문제 의심

**단계 A-2: stephenllh/m5-accuracy LightGBM 코드 실행 및 재현**

목표: 코드가 잘 정리된 Silver medal 솔루션(WRMSSE 0.637)을 그대로 실행해 우리 파이프라인의 기반으로 삼는다.

1. GitHub: https://github.com/stephenllh/m5-accuracy clone
2. 코드 그대로 실행 (M5 전체 데이터)
3. 보고 성능 WRMSSE 0.637과 비교 → ±10% (≤ 0.701) 이내면 재현 성공
4. 재현 성공 시: 이 코드를 단계 B의 기반으로 사용

**단계 B: Cold-Start 적응**

단계 A-2 재현 성공 후, 우리 실험 설정에 맞게 수정:
- 제거: lag_7, lag_28, rolling_mean_7, rolling_mean_28 (cold item은 이력 없음)
- 추가: k-NN proxy lag features (lightgbm_proxy_lags), 또는 제거만 (lightgbm_static)
- 데이터: 300 warm items × CA_1 학습 → 100 cold items 예측
- 예측 horizon: 17주 (주간 집계)

논문에 명시할 사항: 기반 코드 출처(stephenllh), 제거된 features, 추가된 features, 데이터 규모 차이

---

### 2-2. DeepAR (Salinas et al., 2020)

**원 논문:** Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). International Journal of Forecasting 36(3):1181-1191.
**논문 보고 성능:** electricity 데이터셋 ND=0.070, NRMSE=0.082

**단계 A: 원본 재현**

목표: electricity 데이터셋에서 논문 Table 2의 수치를 GluonTS 공식 구현으로 재현한다.

1. electricity 데이터 다운로드: UCI ML Repository (370 고객 × hourly 소비량, 공개)
2. GluonTS `DeepAREstimator`로 논문 설정 그대로 학습
   - freq="H", context_length=논문 설정, prediction_length=24
   - 논문의 hyperparameter 사용 (확인 후 적용)
3. ND, NRMSE 계산 → 논문 Table 2 결과와 비교
4. 허용 오차: ±10% (ND ≤ 0.077, NRMSE ≤ 0.090) 이내면 재현 성공

**단계 B: Cold-Start 적응**

원본 재현 성공 후, 우리 실험 설정에 맞게 수정:
- freq: hourly → weekly
- context_length: 수백 hours → 4 weeks (cold item은 이력 없음)
- Cold item context 초기화: category weekly mean (실제 이력 대신)
- Static features: cat_id, dept_id

논문에 명시할 사항: freq 변경, context 초기화 방법, cold-start 설정이 원 논문 의도와 다름

---

### 2-3. LLMTime (Gruver et al., 2023)

**원 논문:** Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). "Large Language Models Are Zero-Shot Time Series Forecasters." NeurIPS 2023.
**논문 보고 성능:** Darts 8개 데이터셋 MAE 기준 대부분 1-2위 (GPT-3 text-davinci-003 사용)

**단계 A: 핵심 방법론 구현 및 검증**

목표: LLMTime의 핵심 방법론을 정확히 구현하고 Darts 벤치마크에서 검증한다.

LLMTime 핵심 방법론 (현재 우리 코드에 없는 것):
- **숫자 토큰화**: 각 숫자를 space로 구분, 소수점 자릿수 제어 (e.g., "1 2 3 4 5")
- **스케일링**: alpha=0.95, beta=0.3 파라미터로 시계열 스케일 조정
- **입력 형식**: 과거 시계열 자체를 숫자 문자열로 직접 입력 (설명 텍스트 없음)

구현 절차:
1. GitHub: https://github.com/ngruver/llmtime 코드 분석
2. 핵심 방법론(토큰화, 스케일링) 우리 코드에 구현
3. Darts 라이브러리 내장 데이터셋 2-3개 선택 (AirPassengers 포함)
4. GPT-4o-mini로 실행 (text-davinci-003 deprecated → 한계로 명시)
5. MAE 비교 → 논문 결과와 정성적으로 유사한지 확인

**단계 B: M5 Cold-Start 적응**

원본 검증 후, M5 cold-start에 적용:
- 입력: 유사 warm 아이템의 이력을 LLMTime 방식(숫자 문자열)으로 제공
- 수정 사항 명시: 입력 형식, LLM 모델 변경, cold-start 적응 방법

논문에 명시할 사항: LLM 모델 변경(GPT-3→GPT-4o-mini), 이로 인한 재현 한계, M5 적응 방법

---

### 2-4. k-NN Analog (Van Steenbergen & Mes, 2020 — 검증 불가)

**원 논문:** Van Steenbergen & Mes (2020), Decision Support Systems 139:113401.
**데이터 공개 여부:** 비공개 (5개 회사 실제 데이터)

**결론:** 원 논문의 벤치마크 데이터가 비공개이므로 외부 검증 불가.

처리 방안:
- 논문에서 "k-NN 기반 cold-start baseline"으로 명시 (Van Steenbergen & Mes 인용, 정확한 재현이 아님)
- 우리 구현의 설계 선택(cosine similarity, one-hot + price, k=5)을 논문에 명시
- Phase 3 Leave-one-out 검증으로 구현의 논리적 일관성만 확인

---

## Phase 3: 재현 불가능한 방법론에 대한 대안 검증

논문에서 동일 데이터셋 결과가 없는 방법론은, **warm items에서의 leave-one-out 검증**으로 대체한다.

### 방법:
1. warm items 300개 중 1개를 "pseudo-cold"로 설정 (이력 제거)
2. 나머지 299개로 학습/참조
3. pseudo-cold 아이템의 판매량을 예측
4. 이를 300번 반복 (또는 50개 샘플링)
5. warm items의 실제 판매량과 비교하여 MAE/DirAcc 계산

이 검증의 의미:
- 모든 competitor가 동일한 검증을 거치므로 **상대적 비교가 공정**해진다
- warm items는 이력이 있으므로, "이력을 사용하는 모델"과 "이력을 사용하지 않는 모델"의 성능 차이를 직접 볼 수 있다
- 이 결과가 합리적이면(예: k-NN이 random보다 낫고, LightGBM이 k-NN보다 나은 등), 구현이 최소한 논리적으로 작동한다고 볼 수 있다

---

## Phase 4: 코드 레벨 검증 (구현 버그 탐지)

각 competitor 구현 파일에 대해 다음을 확인한다:

### 4-1. 데이터 누수 검증
```python
# 각 competitor에서 cold item의 실제 판매 데이터가 학습/feature에 포함되지 않았는지
for model_name in all_competitors:
    # training data에 cold item_id가 없는지
    assert set(training_data.item_id) & set(cold_items.item_id) == set()
    # feature 계산에 cold item의 sales가 사용되지 않았는지
    # (예: category_weekly_mean에 cold item의 sales가 포함되면 누수)
```

### 4-2. 평가 코드 검증
```python
# MAE, RMSE, WRMSSE, DirAcc 각각에 대해:
# 1. 수동 계산 예시 (5개 아이템, 3주)를 만들어서
# 2. 우리 evaluate 함수의 결과와 수동 계산 결과가 일치하는지 확인
# 3. 특히 DirAcc에서 flat(변화 없음)을 어떻게 처리하는지 명시
```

### 4-3. 데이터 전처리 검증
```python
# cold_test_weekly의 각 아이템에 대해:
# 1. daily sales의 합 == weekly sales인지 확인
# 2. 주 경계가 ISO 8601 기준인지 확인
# 3. 불완전한 주가 제거됐는지 확인 (16주 맞는지)
# 4. warm_train_weekly도 동일한 방식으로 생성됐는지 확인
```

---

## Phase 5: 최종 보고서

모든 검증 완료 후 아래 표를 작성한다:

### 5-1. 논문 재현 검증 결과

| 모델 | 원 논문 | 벤치마크 데이터 | 논문 보고 성능 | 우리 재현 성능 (단계 A) | 판정 | Cold-Start 적응 수정 사항 |
|------|--------|-------------|--------------|----------------------|------|--------------------------|
| LightGBM | YJ_STU / Makridakis 2022 | M5 전체 | WRMSSE 0.520 | ? | ? | lag features 제거, proxy features 추가 |
| DeepAR | Salinas 2020 | electricity (UCI) | ND=0.070, NRMSE=0.082 | ? | ? | freq hourly→weekly, context category mean 초기화 |
| LLMTime | Gruver 2023 | Darts 2-3개 | 대부분 1-2위 (MAE) | ? (GPT-4o-mini 사용) | ? | LLM 모델 변경, cold-start 입력 형식 변경 |
| k-NN | Van Steenbergen 2020 | 비공개 | N/A | 검증 불가 | 검증 불가 | cosine similarity, one-hot+price, k=5 명시 |

### 5-2. Leave-One-Out 검증 결과 (warm items)

| 모델 | LOO MAE | LOO DirAcc | 순위 | 논리적 일관성 |
|------|---------|------------|------|-------------|
| similar_item_avg | ? | ? | ? | ? |
| knn_analog | ? | ? | ? | 기대: similar_item보다 나음 |
| lightgbm_static | ? | ? | ? | 기대: k-NN보다 나음 |
| lightgbm_proxy_lags | ? | ? | ? | 기대: static보다 나음 |
| seasonal_pattern | ? | ? | ? | ? |
| deepar | ? | ? | ? | ? |

### 5-3. 코드 검증 결과

| 검증 항목 | 결과 | 발견된 문제 |
|-----------|------|-----------|
| 데이터 누수 | ? | ? |
| 평가 코드 정확성 | ? | ? |
| 데이터 전처리 일관성 | ? | ? |

---

## 실행 순서

1. **Phase 1 완료** ✅: 표 채움 (phase1_competitor_verification.md 참조)
2. **Phase 2**: 각 competitor에 대해 반드시 단계 A → 단계 B 순서로 진행.
   - 2-1: LightGBM 원본 재현 (M5 전체 데이터, WRMSSE ~0.52) → cold-start 적응
   - 2-2: DeepAR 원본 재현 (electricity, ND ~0.070) → cold-start 적응
   - 2-3: LLMTime 핵심 방법론 구현 및 Darts 검증 → M5 cold-start 적응
   - 2-4: k-NN 검증 불가 명시 → 논문 표기 방식만 결정
3. **Phase 3**: Leave-one-out 검증. API 불필요, 로컬에서 실행.
4. **Phase 4**: 코드 레벨 검증 (데이터 누수, 평가 코드, 전처리).
5. **Phase 5**: 최종 보고서 작성.

**중요: Phase 2의 각 단계에서 단계 A 검증 결과를 먼저 보고하고, 사용자 확인 후 단계 B(cold-start 적응)로 넘어갈 것. 단계 A 실패(±10% 초과) 시 단계 B로 넘어가지 말고 원인 분석 먼저.**

---

## 주의사항

- **"아마 괜찮을 것이다"는 판단 금지.** 확인하지 않은 것은 "미확인"으로 표시할 것.
- **사후 합리화 금지.** 기준을 정하고, 기준에 맞지 않으면 기준이 아니라 구현을 의심할 것.
- **상대적 개선과 절대적 정확성을 혼동하지 말 것.** "naive보다 낫다"가 "구현이 맞다"를 증명하지 않음.
- **검증 불가능한 항목은 솔직히 "검증 불가"로 기록할 것.** 이것이 논문의 Limitations에 들어갈 수 있음.
