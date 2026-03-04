# 연구 프로포절 요약

- **연구 목표:** 콜드스타트 및 Persona 기반 M5 수요예측 문제에서 인간적 다양성과 aggregation을 활용해 예측 성능 제고
- **핵심 방법:**
  - (1) 다양한 Persona(인구통계/취향/소비습관 등 가상 agent) 생성
  - (2) 엔트로피 기반 rich persona set 설계 및 feature embedding
  - (3) Meta-learned aggregation (meta-ensemble)로 각 agent 예측치 통합
- **데이터셋/실험:**
  - M5와 추가 synthetic dataset 실험 병행
  - Baseline은 ARIMA, LightGBM, 대형 LLM 활용 예시
- **기여:**
  - LLM 기반 제품/new-user cold-start 시나리오 평가
  - aggregation strategy의 대규모 계량적 검증
  - 실제 고난이도 cold-start forecasting 벤치마크 제시
---