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

## 5. 베이스라인 정의

### Baseline 1: Global Category Average
- 동일 카테고리 상품들의 전체 평균 판매량

### Baseline 2: Similar Item Average
- 동일 카테고리 + 유사 가격대 상품들의 평균 판매량

### Baseline 3: Store-Category Average
- 동일 매장 + 동일 카테고리의 평균 판매량

### 평가 지표
- MAE (Mean Absolute Error)
- WRMSSE (M5 공식 지표)
- 방향성 정확도 (Direction Accuracy)

## 6. 타임라인

| 기간 | 작업 |
|------|------|
| ~3월 중순 | Phase 1 데이터 전처리 + 베이스라인 구현 |
| ~4월 중순 | 합성 페르소나 생성 + LLM 예측 파이프라인 |
| ~5월 중순 | Phase 1 실험 완료 + 결과 분석 |
| ~6월 | 논문 초안 작성 |
| ~7월 | 논문 수정 + 심사 제출 |
