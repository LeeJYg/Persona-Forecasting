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

### Phase 1: Feasibility Test (진행 중)
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

### Phase 1 진행 현황 (2026-03-07 기준)

| Task | 상태 | 결과 위치 |
|------|------|----------|
| Task 1: M5 데이터 전처리 (cold-start 샘플링) | **완료** | data/processed/cold_start/ |
| Task 2: 베이스라인 구현 및 평가 | **완료** | experiments/exp002_cold_start_baselines/ |
| Task 3: 합성 페르소나 생성 파이프라인 | **완료** | data/processed/personas/ |
| Task 4a: Track B (임베딩 기반 예측) | **완료** | experiments/exp005_track_b_embedding/ |
| Task 4b: Track B 심층 분석 | **완료** | experiments/exp005_track_b_embedding/analysis/ |
| Task 4c: Attention Head 실험 | **완료** | experiments/exp009_attention_head/ |
| Task 4d: Track A (직접 LLM 예측) | 미실행 | — |

**현재 판단**: alpha 보정 후 best 모델(Ridge mean-pooled, weekly MAE=9.74)이 프록시 베이스라인(LightGBM 8.48)에 15% 열위. DirAcc는 반대로 페르소나 임베딩이 우위(0.536 vs 0.343). Phase 2 진입 여부는 Track A 결과 확보 후 종합 판단 예정.

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

### 생성 완료 (2026-03, exp003)

- **50개 페르소나** (CA_1_P001 ~ CA_1_P050), GPT-4o로 생성 (temperature=1.0, 다양성 확보)
- 저장 위치: `data/processed/personas/CA_1_P*.json` (개별 파일) + `all_personas.json`
- 생성 방식: Show Don't Tell 원칙으로 description 작성, 매장 데이터 직접 제약 없이 LLM 자유 생성
- Phase 1에서 **Condition A** (구조화 필드만 제공, narrative 제외) 사용 — Li et al. 2025 권고 반영
- 상위 주목 페르소나 (Exp009 attention 분석): P50(weight 0.363), P7(0.188), P42(0.103) — 모두 "health & wellness, FOODS 중심, 주 1회 방문" 공통 특성

### 실제 생성된 페르소나 스키마 (구현 완료)

```json
{
  "persona_id": "CA_1_P050",
  "store_id": "CA_1",
  "profile": {
    "description": "Nancy recently moved to a rural area and loves experimenting with new recipes using local produce. Her trips are methodically planned, but she occasionally treats herself to premium ingredients.",
    "weekly_budget": 150.0,
    "snap_eligible": false,
    "shopping_motivation": "health & wellness focus",
    "economic_status": "middle-income",
    "category_preference": {"FOODS": 0.7, "HOBBIES": 0.1, "HOUSEHOLD": 0.2},
    "price_sensitivity": "medium",
    "visit_frequency": "weekly",
    "preferred_departments": ["FOODS_3", "HOUSEHOLD_1"],
    "decision_style": "planned",
    "brand_loyalty": "high",
    "promotion_sensitivity": "medium"
  }
}
```

**당초 설계 대비 변경**: `purchase_history` 필드는 구현에서 제외됨 (LLM 생성 시 불필요). 예측은 페르소나 프로필 + 아이템 메타만으로 수행.

### 생성 원칙 (달성)
- 50개 페르소나 생성 완료 (Phase 1 feasibility test용)
- description은 Show/Don't Tell 원칙 (직업/나이 직접 언급 금지, 행동으로 묘사)
- 구조화 필드(category_preference, weekly_budget 등)는 수치 명시 → Condition A 예측 시 직접 사용
- seed 불필요 (GPT-4o temperature=1.0, 다양성 우선)

## 5. 베이스라인 정의 및 결과 (exp002, 완료)

### Baseline 1: Global Category Average
- 동일 카테고리 상품들의 전체 평균 판매량

### Baseline 2: Similar Item Average
- 동일 카테고리 + 유사 가격대(3분위 기준) 상품들의 평균 판매량

### Baseline 3: Store-Category Average
- 동일 매장 + 동일 카테고리의 평균 판매량

### exp002 결과 (CA_1 매장, cold 100 아이템, daily 단위)

| 모델 | MAE (daily) | WRMSSE | DirAcc (weekly) |
|------|----------:|-------:|---------------:|
| GlobalCategoryAverage | 1.57 | 2.96 | 0.257 |
| SimilarItemAverage | 1.69 | 2.99 | 0.257 |
| StoreCategoryAverage | 1.62 | 2.97 | 0.257 |

- FOODS MAE(~2.7)가 HOBBIES/HOUSEHOLD(1.0~1.2)보다 2~3배 높음
- **DirAcc 0.257 = LLM 페르소나가 뛰어넘어야 할 하한선** (daily 기준)
- 상수 예측 모델이므로 DirAcc = flat 실제 비율과 동일

### 평가 지표 표준 정의 (Phase 1 전체 통일 기준)

#### MAE
```
MAE = mean(|actual_sales - pred_sales|)
```
- **daily 단위** 사용 시: 일별 예측 vs 일별 실측
- **weekly 단위** 사용 시: pd.to_period("W") 집계 후 주간 합 비교

#### WRMSSE (Cold-Start 수정버전)
- M5 공식 WRMSSE의 cold item scale(s_i)을 동일 카테고리 warm item들의 평균 scale로 대체
- 가중치(w_i) = 테스트 기간 실제 판매량 합 (학습 이력 없으므로)
- lag = 28일 (M5 공식 기준)
- 구현: `src/evaluation/metrics.py:wrmsse()`

#### DirAcc (Direction Accuracy)
**최종 표준 (Exp009 방식, flat 제외):**
```python
# 주간 집계: pd.to_period("W") (ISO Mon-Sun)
td = np.sign(np.diff(y_weekly_true,  axis=1))   # 연속 주 차분 부호
pd_ = np.sign(np.diff(y_weekly_pred, axis=1))
mask = td != 0                                   # flat(0) 제외
DirAcc = (td[mask] == pd_[mask]).mean()
```
**주의**: exp002/exp005는 flat 포함, ISO calendar week 방식으로 계산됨 → 직접 비교 불가
(상세 비교: `docs/research_notes_exp004-009.md` 부록 참조)

## 6. LLM 예측 접근법: Track A / Track B

Phase 1에서는 LLM 페르소나를 예측에 활용하는 두 가지 방식을 비교한다.

### Track A: 직접 예측 (Naive LLM Prediction)

- **아이디어**: 50개 페르소나 각각이 GPT-4o-mini에게 "이 아이템을 이번 주에 몇 개 살 것 같은가?"를 직접 물어봄
- **스케일 처리**: 50명 개인 합산(50-persona scale) → post-hoc mean ratio alpha 보정으로 매장 수준 환산
- **API 비용**: 50 personas × 10 batches × 17 weeks = 8,500 calls/실험
- **조건**: Condition A (구조화 필드만, description 제외)
- **스크립트**: `scripts/run_track_a.py` → `scripts/compare_track_a_baselines.py`
- **출력**: `experiments/exp004_track_a_naive/`
- **상태**: 미실행 (Track B 우선 검증)
- **참조 결과**: Track A calibrated weekly MAE=8.90, DirAcc=0.393 (출처 불명확, 비공식 참조값)

### Track B: 임베딩 기반 2-Stage 예측

- **아이디어**: LLM의 last-token hidden state를 임베딩으로 활용 → warm 아이템 판매량으로 Ridge 회귀 학습 → cold 아이템에 적용
- **LLM**: Qwen/Qwen2.5-32B-Instruct (4-bit NF4 양자화, Tesla P40 × 4)
- **Stage 1**: 아이템-페르소나 텍스트 → LLM hidden state 추출
- **Stage 2**: 임베딩 → Ridge 회귀 → 주간 판매량 예측
- **스크립트**: `scripts/run_track_b.py` (Exp005), `scripts/run_exp009.py` (Exp009)

**Track B 구현 변천**:

| 실험 | 임베딩 방식 | 헤드 | cold_MAE (weekly, 보정) |
|------|------------|------|----------------------:|
| Exp005 | mean-pool(N, 50, 5120) → (N, 5120) | WeeklySalesHead (StandardScaler+Ridge) | ~13 (daily→weekly 환산 근사) |
| Exp009 3-1 | mean-pool → (N, 5120) | Ridge(alpha=1.0), no scaler | 9.74 |
| Exp009 3-4 | raw(N, 50, 5120) + Attention+Bottleneck | AttnHead(Lin→64→17) | 10.41 |

**핵심 설계 결정**:
- `torch_dtype` + `load_in_4bit=True` 동시 지정 금지 → full-precision 로드 후 양자화로 OOM 발생
- HF 모델 캐시: `/mnt/sdd1/jylee/huggingface_cache` (54GB, 루트 디스크 절약)
- `output_hidden_states=True`는 generation flag로 경고 발생하나 동작 이상 없음

## 7. Phase 1 실험 결과 요약

### 통합 비교 테이블 (weekly MAE, 동일 단위)

| 모델 | weekly MAE | DirAcc | 단위/방식 | 출처 |
|------|----------:|-------:|---------|------|
| GlobalCategoryAverage | ~11 | 0.257 | daily→weekly×7 환산 | exp002 |
| LightGBM proxy lags | **8.48** | 0.343 | weekly, flat 제외 | 비공식 참조 |
| Track A calibrated | 8.90 | 0.393 | weekly | 비공식 참조 |
| knn_analog | 9.57 | 0.412 | weekly | 비공식 참조 |
| **Ridge mean-pooled (α보정)** | **9.74** | 0.536 | weekly, flat 제외 | exp009 |
| Variance+Ridge (α보정) | ~9.75 | 0.536 | weekly, flat 제외 | exp009 |
| Attn+Bottleneck (α보정) | 10.41 | 0.536 | weekly, flat 제외 | exp009 |
| Ridge item-only (α보정) | 11.11 | **0.561** | weekly, flat 제외 | exp009 |

> α 보정: `alpha = y_cold_weekly_mean / pred_mean` (각 헤드별 0.13~0.15)

### 카테고리별 패턴 (exp005 기준, daily 단위)

| 카테고리 | 아이템 수 | MAE (daily, raw) | 특성 |
|---------|---------|----------------:|------|
| FOODS | 34 | 15.11 | 수요 변동 크고 오차 가장 높음 |
| HOBBIES | 33 | 11.93 | 중간 |
| HOUSEHOLD | 33 | 12.11 | 중간 |

### Alpha 보정 분석

모든 Track B 헤드에서 6~9배 과예측 발생. 원인: Ridge/Attention 헤드가 warm 아이템(판매량 큰) 기반으로 학습 후 cold 아이템(판매량 작은)에 외삽할 때 체계적 bias.

| 헤드 | pred_mean (weekly) | alpha | 보정 후 MAE (weekly) |
|------|------------------:|------:|-------------------:|
| exp005 Ridge | 97 (일별×7) | 0.106 | 1.86 (daily) |
| exp009 Ridge mean | 74.6 | 0.133 | 9.74 |
| exp009 Attn+Bottleneck | 65.4 | 0.152 | 10.41 |

## 8. 주요 발견 및 연구 함의

### 발견 1: 임베딩의 내재 low-rank 문제

Mean-pooled 임베딩(5120차원)의 PCA 분석:
- Top-5 PC: 97.3% 분산 설명
- Top-10 PC: 98.6% 분산 설명
- **5120차원이 실질적으로 ~15차원 매니폴드** → mean-pooling이 50개 페르소나 다양성을 파괴

함의: 임베딩을 raw per-persona로 보존(N, 50, 5120)하고 학습 가능한 aggregation이 필요.

### 발견 2: Attention Collapse

Exp009 Attn+Bottleneck 모델의 attention 가중치:
- P50(0.363), P7(0.188), P42(0.103) — 상위 3개가 65% 집중
- 나머지 47개 페르소나 가중치 ≈ 0
- **카테고리(FOODS/HOBBIES/HOUSEHOLD) 무관하게 동일 top-3**
- 모든 상위 페르소나: "health & wellness, FOODS 중심, 주 1회 방문" 공통 특성

함의: 학습 데이터(n=300)가 부족해 attention이 소수 페르소나로 collapse. 페르소나 다양성이 예측에 활용되지 못하고 있음.

### 발견 3: DirAcc에서의 페르소나 임베딩 우위

| 방법 | DirAcc (weekly, flat 제외) |
|------|------------------------:|
| 베이스라인 (상수 예측) | ~0.257 (flat-inclusive, 단위 다름) |
| LightGBM proxy | 0.343 |
| Track A calibrated | 0.393 |
| Track B (Ridge mean-pooled) | **0.536** |
| Track B (Ridge item-only) | **0.561** |

방향 예측에서 페르소나 임베딩이 의미 있는 개선. 단, item-only가 페르소나 포함보다 DirAcc가 높아 페르소나가 방향 예측에서는 노이즈로 작용할 수 있음.

### 발견 4: 지속되는 스케일 불일치

alpha 보정 후에도 완전한 해결이 아님. 원인은 warm/cold 도메인 shift (판매량 분포 차이). 후속 연구 방향:
- 카테고리별 독립 alpha 보정
- 학습 시 cold item의 판매량 스케일 정보 주입 (if available)
- 판매량 분포 정규화 후 학습

### 발견 5: Phase 1 잠정 결론

- **MAE 기준**: 보정 후 best(9.74) ≈ knn_analog(9.57), LightGBM(8.48) 대비 15% 열위 → 아직 베이스라인 수준 미달
- **DirAcc 기준**: 페르소나 임베딩이 유의미하게 우위 (0.536 vs 0.343~0.412)
- **종합**: "단순 베이스라인보다 유의미하게 나은가?" → MAE에서는 No, DirAcc에서는 Yes
- Track A 결과 확보 후 종합 판단 필요. 현재는 Phase 2 진입 경계선.

## 9. 타임라인

| 기간 | 작업 | 상태 |
|------|------|------|
| ~3월 중순 | Phase 1 데이터 전처리 + 베이스라인 구현 | **완료** (exp002) |
| ~3월 말 | 합성 페르소나 생성 (50개) | **완료** (exp003) |
| ~3월 말 | Track B 임베딩 파이프라인 구축 및 실행 | **완료** (exp005, exp009) |
| ~4월 초 | Track A (직접 LLM 예측) 실행 및 비교 | **미완료** |
| ~4월 중순 | Phase 1 종합 분석 (Track A vs Track B vs 베이스라인) | 대기 중 |
| ~5월 중순 | Phase 1 결과 확정 + Phase 2 진입 여부 판단 | 대기 중 |
| ~6월 | 논문 초안 작성 | 대기 중 |
| ~7월 | 논문 수정 + 심사 제출 (석사논문, 2026년 8월 졸업 목표) | 대기 중 |

### 추가 후보 실험 (우선순위 순)

1. **Track A 실행** (exp004): GPT-4o-mini 직접 예측 → Track B와 공정 비교
2. **카테고리별 alpha 보정**: 현재 global alpha(0.133) → FOODS/HOBBIES/HOUSEHOLD 별도 보정
3. **n_warm 확대 실험**: 300 → 500~1000으로 늘렸을 때 attention collapse 해소 여부 확인
4. **FP16 실험** (config_fp16.yaml 준비 완료): INT4 vs FP16 임베딩 품질 비교
5. **Condition B/C**: description 추가 시 예측 성능 변화 (Li et al. 2025 검증)
