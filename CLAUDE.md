# LLM Persona Cold-Start Forecasting Research

## What This Project Is
M5 (Walmart) 데이터셋 기반, LLM 페르소나 에이전트를 활용한 Cold-Start 수요 예측 연구.
석사 논문 (2026년 8월 졸업 목표). 현재 Phase 1: Feasibility Test 단계.

## Research Context (반드시 숙지)
- Cold-Start 정의: 신규 매장 오픈 시점에서 해당 매장의 판매 이력이 없는 상황
- 핵심 질문: LLM 페르소나가 단순 베이스라인(카테고리 평균 등)보다 cold-start 예측을 잘 하는가?
- 상세 연구 맥락은 `docs/RESEARCH_CONTEXT.md`를 참조할 것

## Project Structure
```
RESEARCH/
├── data/                  # 전처리된 데이터
├── docs/                  # 연구 문서 (RESEARCH_CONTEXT.md 포함)
├── experiments/           # 실험 결과 저장
├── m5-forecasting-accuracy/  # M5 원본 데이터셋
├── notebooks/             # EDA, 분석용 노트북
├── scripts/               # 실행 스크립트
├── src/                   # 핵심 소스 코드
├── tests/                 # 테스트 코드
└── configs/               # 설정 파일 (없으면 생성할 것)
```

## Coding Conventions (엄격히 준수)
- 모든 경로는 `configs/` 디렉토리의 yaml/json 파일에서 관리. 코드 내 하드코딩 절대 금지
- Python 3.10+, type hints 필수
- 함수/클래스에 docstring 작성
- seed 값은 config에서 관리 (재현성 보장)
- 로깅은 Python logging 모듈 사용 (print 금지)
- 새 파일 생성 시 반드시 `__init__.py` 업데이트

## Task Rules
- 코드 수정 전 반드시 기존 코드를 읽고 구조를 파악할 것
- 실험 실행 결과는 `experiments/` 하위에 날짜별로 저장
- config 변경 사항은 반드시 git commit 메시지에 명시
- 불확실한 연구 판단이 필요한 경우 사용자에게 먼저 확인

## Current Phase: Phase 1 - Feasibility Test
목표: LLM 페르소나 → cold-start 예측이 베이스라인보다 나은지 검증
- [x] M5 데이터 전처리 (cold-start 아이템 샘플링)
      → CA_1 매장 기준 100 store-item 쌍, cross_store_info=false
      → 출력: data/processed/cold_start/ (cold_test 11,500행)
- [x] 베이스라인 구현 (카테고리 평균, 유사 아이템 평균, 매장-카테고리 평균)
      → 결과: experiments/exp002_cold_start_baselines/
      → MAE 1.57~1.69 / WRMSSE 2.96~2.99 / DirAcc 0.257 (3모델 동일)
      → FOODS 오차(MAE ~2.7)가 HOBBIES·HOUSEHOLD(1.0~1.2)보다 2~3배 큼
      → DirAcc 0.257 = LLM 페르소나가 뛰어넘어야 할 하한선
- [x] 합성 페르소나 생성 파이프라인 구축
      → 50개 페르소나, CA_1 매장, LLM 자유 생성 (매장 데이터 제약 없음)
      → 출력: data/processed/personas/ (개별 JSON + all_personas.json)
      → 스키마: description(Show/Don't Tell), weekly_budget, snap_eligible,
               shopping_motivation, economic_status, category_preference,
               price_sensitivity, visit_frequency, preferred_departments,
               decision_style, brand_loyalty, promotion_sensitivity
      → 구현: src/llm/client.py, src/models/persona/{schema,generator}.py
      → 실행: python scripts/generate_personas.py
- [ ] LLM 페르소나 기반 예측 파이프라인 구축
- [ ] 비교 실험 및 결과 분석

## Task 4 설계 메모 (Li et al. 2025 반영)
예측 시 페르소나 정보 제공 방식을 3조건으로 실험:
- Condition A (Structured Only): JSON 구조화 필드만 제공
- Condition B (Structured + Narrative): 구조화 필드 + description
- Condition C (Narrative Only): description만 제공
Phase 1에서는 Condition A로 먼저 feasibility 검증, B/C는 후속 실험
근거: Li et al. (2025) - LLM 생성 서사가 많을수록 시뮬레이션 편향 증가
