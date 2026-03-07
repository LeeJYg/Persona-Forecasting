"""M5 공식 12-level WRMSSE 계산 (stephenllh/m5-accuracy 예측 결과 검증).

출처: github.com/Mcompetitions/M5-methods
      Code of Winning Methods/A4/kaggle-m5a-4th/m5a_eval.py (4th place evaluation code)

공식 WRMSSE 정의:
  - 12개 집계 수준 × 각 수준의 시리즈 수 = 42,840 시리즈
  - scale_i: 각 집계 시리즈의 학습기간 lag-1 MSE (첫 비영 값 이후부터)
  - weight_i: 학습 마지막 28일(d_1886~d_1913) 달러 판매액 비중
  - WRMSSE = mean(WRMSSE_lv1, ..., WRMSSE_lv12) where
    WRMSSE_lv = Σ(w_i * RMSSE_i) for each aggregation level
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

M5_DIR = ROOT / "m5-forecasting-accuracy"
SUBMISSION_PATH = Path("/tmp/m5-accuracy/src/submission_.csv")
EXP_DIR = ROOT / "experiments" / "exp009_stephenllh_verification"
EXP_DIR.mkdir(parents=True, exist_ok=True)

EVAL_D_COLS = [f"d_{i}" for i in range(1914, 1942)]   # 28일 (evaluation period)
TRAIN_D_COLS = [f"d_{i}" for i in range(1, 1914)]      # d_1~d_1913 (training)
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# ============================================================
# 공식 WRMSSEEvaluator (출처: M5-methods/A4/m5a_eval.py, 원본 유지)
# ============================================================

class WRMSSEEvaluator:
    """12-level WRMSSE evaluator.

    출처: github.com/Mcompetitions/M5-methods (4th place team, A4)
    원본 코드를 그대로 사용하고 logging만 추가.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        calendar: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> None:
        train_y = train_df.loc[:, train_df.columns.str.startswith("d_")]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()  # d_1886~d_1913

        train_df["all_id"] = "all"  # lv1 집계용

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith("d_")].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith("d_")].columns.tolist()

        if not all(c in valid_df.columns for c in id_columns):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices
        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            "all_id",
            "state_id",
            "store_id",
            "cat_id",
            "dept_id",
            ["state_id", "cat_id"],
            ["state_id", "dept_id"],
            ["store_id", "cat_id"],
            ["store_id", "dept_id"],
            "item_id",
            ["item_id", "state_id"],
            ["item_id", "store_id"],
        )

        for i, group_id in enumerate(tqdm(self.group_ids, desc="Building evaluator")):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]  # 첫 비영 값 이후
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f"lv{i+1}_scale", np.array(scale))
            setattr(self, f"lv{i+1}_train_df", train_y)
            setattr(self, f"lv{i+1}_valid_df", valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f"lv{i+1}_weight", lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = (
            self.train_df[["item_id", "store_id"] + self.weight_columns]
            .set_index(["item_id", "store_id"])
        )
        weight_df = weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"])
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = (
            weight_df.set_index(["item_id", "store_id", "d"])
            .unstack(level=2)["value"]
            .loc[zip(self.train_df.item_id, self.train_df.store_id), :]
            .reset_index(drop=True)
        )
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f"lv{lv}_valid_df")
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f"lv{lv}_scale")
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            valid_preds_grp = valid_preds.groupby(group_id)[self.valid_target_columns].sum()
            setattr(self, f"lv{i+1}_valid_preds", valid_preds_grp)

            lv_scores = self.rmsse(valid_preds_grp, i + 1)
            setattr(self, f"lv{i+1}_scores", lv_scores)

            weight = getattr(self, f"lv{i+1}_weight")
            lv_scores_w = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores_w[~lv_scores_w.isin([np.inf])].sum())

        self.all_scores = all_scores
        return float(np.mean(all_scores))


# ============================================================
# 메인
# ============================================================

def main() -> None:
    # 1. 데이터 로드
    logger.info("sales_train_evaluation.csv 로드 중 (d_1~d_1941)...")
    train_eval = pd.read_csv(
        M5_DIR / "sales_train_evaluation.csv",
        usecols=ID_COLS + TRAIN_D_COLS + EVAL_D_COLS,
    )
    logger.info("shape: %s", train_eval.shape)

    train_df = train_eval[ID_COLS + TRAIN_D_COLS].copy()
    valid_df_actual = train_eval[ID_COLS + EVAL_D_COLS].copy()  # 실제 d_1914~d_1941

    logger.info("calendar / sell_prices 로드 중...")
    calendar = pd.read_csv(M5_DIR / "calendar.csv")
    prices = pd.read_csv(M5_DIR / "sell_prices.csv")

    # 2. stephenllh 예측 로드
    logger.info("submission_.csv 로드 중: %s", SUBMISSION_PATH)
    sub = pd.read_csv(SUBMISSION_PATH)
    sub_val = sub[sub["id"].str.endswith("_validation")].copy()
    logger.info("validation rows: %d", len(sub_val))

    # F1..F28 → d_1914..d_1941
    f_to_d = {f"F{i}": f"d_{1913 + i}" for i in range(1, 29)}
    sub_val = sub_val.rename(columns=f_to_d)

    # submission id → item_id / store_id 파싱
    # 형식: CATEGORY_NUM_ITEM_STORE_validation  예) FOODS_1_001_CA_1_validation
    # sales_train_evaluation.csv의 id는 _evaluation 접미사 → item_id+store_id 기준 join
    sub_val["item_id"] = sub_val["id"].str.replace(
        r"_(CA_\d|TX_\d|WI_\d)_validation$", "", regex=True
    )
    sub_val["store_id"] = sub_val["id"].str.extract(r"_(CA_\d|TX_\d|WI_\d)_validation$")[0]

    # train_eval 행 순서에 맞춰 예측값 정렬 (item_id + store_id 기준 join)
    pred_aligned = train_eval[["item_id", "store_id"]].merge(
        sub_val[["item_id", "store_id"] + EVAL_D_COLS],
        on=["item_id", "store_id"],
        how="left",
    )
    missing = pred_aligned[EVAL_D_COLS].isna().any(axis=1).sum()
    if missing:
        logger.warning("예측값 없는 rows: %d → 0으로 채움", missing)
    else:
        logger.info("모든 rows 예측값 매칭 완료")
    pred_values = pred_aligned[EVAL_D_COLS].fillna(0.0).clip(lower=0.0).values
    logger.info("pred_values shape: %s", pred_values.shape)

    # 3. WRMSSEEvaluator 초기화 (scale + weight 계산, 수분 소요)
    # valid_df는 d_ 컬럼만 전달 (id 컬럼은 evaluator가 train_df에서 붙임)
    logger.info("WRMSSEEvaluator 초기화 중 (12 aggregation levels)...")
    valid_df_donly = valid_df_actual[EVAL_D_COLS].reset_index(drop=True)
    evaluator = WRMSSEEvaluator(train_df, valid_df_donly, calendar, prices)

    # 4. 12-level WRMSSE 계산
    logger.info("score() 호출 중...")
    wrmsse_12level = evaluator.score(pred_values)

    # 5. 레벨별 상세 출력
    level_names = [
        "lv1: Total",
        "lv2: State",
        "lv3: Store",
        "lv4: Category",
        "lv5: Department",
        "lv6: State×Category",
        "lv7: State×Department",
        "lv8: Store×Category",
        "lv9: Store×Department",
        "lv10: Item",
        "lv11: Item×State",
        "lv12: Item×Store",
    ]

    level_results = []
    logger.info("\n=== 12-level WRMSSE 레벨별 결과 ===")
    for i, name in enumerate(level_names):
        lv_score = evaluator.all_scores[i]
        level_results.append({"level": name, "wrmsse": round(lv_score, 4)})
        logger.info("  %s: %.4f", name, lv_score)

    logger.info("\n=== 최종 결과 ===")
    logger.info("  12-level WRMSSE (공식): %.4f", wrmsse_12level)
    logger.info("  원본 stephenllh 공개값: 0.637")
    logger.info("  허용 범위 (±10%%): 0.573 ~ 0.701")

    if 0.573 <= wrmsse_12level <= 0.701:
        verdict = "PASS"
        logger.info("  판정: ✓ PASS — 원본 재현 성공 (범위 내)")
    else:
        verdict = "FAIL"
        logger.warning("  판정: ✗ FAIL — 범위 벗어남 (차이 분석 필요)")

        # 레벨별 차이가 큰 순서로 정렬
        sorted_levels = sorted(level_results, key=lambda x: abs(x["wrmsse"] - 0.637 / 12), reverse=True)
        logger.warning("\n  [레벨별 기여도 — 높은 순]")
        for r in sorted_levels:
            logger.warning("    %s: %.4f", r["level"], r["wrmsse"])

    # 6. 결과 저장
    result = {
        "model": "stephenllh_m5_accuracy",
        "period": "d_1914~d_1941 (evaluation)",
        "evaluator_source": "github.com/Mcompetitions/M5-methods/A4/m5a_eval.py",
        "wrmsse_12level_official": wrmsse_12level,
        "original_wrmsse": 0.637,
        "within_10pct": 0.573 <= wrmsse_12level <= 0.701,
        "verdict": verdict,
        "level_scores": level_results,
    }
    out_path = EXP_DIR / "metrics_12level.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
