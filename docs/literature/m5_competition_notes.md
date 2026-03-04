# M5 Competition 요약

- **대회 개요:** 월마트 미국 10개 주(3개 주, 10개 매장)에서 3,049개 제품의 5년간(2011~2016) 일별 판매량(time-series)을 예측하는 대규모 시계열 forecasting competition
- **평가 지표:** Weighted Root Mean Squared Scaled Error (WRMSSE)
  - 판매액 큰 상품/카테고리에 더 높은 가중치
  - 전통적 RMSE 대신 변화율(scale)에 따른 비교 가능
- **데이터셋:**
  - sales_train_validation.csv: 훈련 데이터(상품별 일별 판매)
  - calendar.csv: 날짜와 이벤트/공휴일 매핑
  - sell_prices.csv: 상품/매장별 주차 단위 가격 정보
- **예측 목표**: 마지막 28일(test set)에 대한 각각의 상품별 일별 매출 예측
- **주요 특성:**
  - 제품별, 매장별, 주별 판매 패턴이 각기 다름 (Seasonality, Promotion effects, holidays 등)
  - Feature engineering(이벤트, 가격변동, lags, calendar embedding 등)이 성능에 큰 영향
- **대표적 우승 솔루션 특징:**
  - LightGBM/meta-ensemble + 시계열 모델 혼합
  - Feature 자동화 + rolling validation, exogenous variables 활용
  - Kalman smoothing, SARIMA, XGBoost 등 다양한 베이스라인 믹스
---