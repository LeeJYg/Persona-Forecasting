# ARIMA 이론 요약

- **정의:** ARIMA(p,d,q)는 시계열 데이터를 설명하는 통계적 예측모델
- **구성요소:**
  - AR(p): 자기회귀, 과거 관측값 활용
  - I(d): 차분, 비정상성 제거 목적
  - MA(q): 이동평균, 과거 예측오차 활용
- **파라미터:**
  - p: AR 계수(시차의 수)
  - d: 차분 횟수(정상성([3mstationarity[0m) 확보용)
  - q: MA 계수
- **모델 적합 순서:**
  1. 정상성 검정 (ADF test 등)
  2. (비정상→)차분해서 정상성 확보
  3. ACF/PACF plot로 p/q 후보군 탐색
  4. Grid search 및 information criterion(AIC/BIC)로 최적화
- **실제 적용:**
  - 데이터가 트렌드/시즌/이벤트효과 가지면 SARIMA(X) 등 확장 고려
  - 수천~수만 개 시계열에는 efficient tuning/parallelization 필요
  - 정상성 위반/seasonality/이상치에 취약할 수 있음

---