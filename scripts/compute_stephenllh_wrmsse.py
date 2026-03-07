"""stephenllh/m5-accuracy 예측 결과로 WRMSSE 계산.

submission_.csv (validation rows only) 또는 submission.csv에서
d_1914~d_1941 evaluation 기간의 WRMSSE를 계산한다.

M5 공식 WRMSSE:
  scale_i = mean((y_t - y_{t-1})^2)  over full train (d_1~d_1913)
  w_i     = dollar_sales in test (M5 공식은 달러 가중치 사용)
           ※ 판매량 단위 가중치로 근사 (sell_prices 없이도 계산 가능)
  WRMSSE  = Σ(w_i * RMSSE_i) / Σ(w_i)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

M5_DIR = ROOT / "m5-forecasting-accuracy"
SUBMISSION_DIR = Path("/tmp/m5-accuracy/src")
EXP_DIR = ROOT / "experiments" / "exp009_stephenllh_verification"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# evaluation 기간: d_1914 ~ d_1941
EVAL_D_COLS = [f"d_{i}" for i in range(1914, 1942)]  # 28일
TRAIN_D_COLS = [f"d_{i}" for i in range(1, 1914)]     # full train (d_1~d_1913)


def main() -> None:
    # 1. submission 파일 로드
    sub_path = SUBMISSION_DIR / "submission_.csv"
    if not sub_path.exists():
        logger.error("submission_.csv 없음: %s", sub_path)
        sys.exit(1)

    logger.info("submission 로드: %s", sub_path)
    sub = pd.read_csv(sub_path)
    logger.info("submission shape: %s", sub.shape)
    logger.info("id sample: %s", sub["id"].head(3).tolist())

    # validation 행만 (id ends with '_validation')
    sub_val = sub[sub["id"].str.endswith("_validation")].copy()
    logger.info("validation rows: %d", len(sub_val))

    # id → item_id, store_id 분리
    # format: CATEGORY_NUM_ITEM_STORE_validation
    sub_val["store_id"] = sub_val["id"].str.extract(r"_(CA_\d|TX_\d|WI_\d)_validation")[0]
    sub_val["item_id"] = sub_val["id"].str.replace(r"_CA_\d_validation$|_TX_\d_validation$|_WI_\d_validation$",
                                                     "", regex=True)
    logger.info("store_ids: %s", sub_val["store_id"].value_counts().to_dict())

    # 2. 실제값 로드 (sales_train_evaluation.csv, d_1914~d_1941)
    logger.info("actual values 로드 (d_1914~d_1941)...")
    id_cols = ["id", "item_id", "store_id", "cat_id", "dept_id", "state_id"]
    actual_wide = pd.read_csv(
        M5_DIR / "sales_train_evaluation.csv",
        usecols=id_cols + EVAL_D_COLS,
    )
    logger.info("actual_wide: %d rows", len(actual_wide))

    # 3. scale 계산용 train 데이터 (d_1~d_1913, item-own lag-1 MSE)
    logger.info("scale 계산용 train 로드 (d_1~d_1913, lag-1 MSE)...")
    train_wide = pd.read_csv(
        M5_DIR / "sales_train_evaluation.csv",
        usecols=["item_id", "store_id"] + TRAIN_D_COLS,
    )

    # item별 scale_i = mean((y_t - y_{t-1})^2) over d_2~d_1913
    logger.info("item-own scale 계산 중...")
    item_scales: dict[str, float] = {}
    for _, row in train_wide.iterrows():
        vals = row[TRAIN_D_COLS].values.astype(float)
        diffs = vals[1:] - vals[:-1]
        scale = float(np.mean(diffs ** 2))
        key = f"{row['item_id']}_{row['store_id']}"
        item_scales[key] = max(scale, 1e-8)
    logger.info("scale 계산 완료: %d items", len(item_scales))

    # 4. wide → long 변환
    # submission: F1..F28 → d_1914..d_1941
    f_to_d = {f"F{i}": f"d_{1913 + i}" for i in range(1, 29)}
    sub_val = sub_val.rename(columns=f_to_d)

    sub_long = sub_val.melt(
        id_vars=["id", "item_id", "store_id"],
        value_vars=EVAL_D_COLS,
        var_name="d",
        value_name="pred_sales",
    )

    actual_long = actual_wide.melt(
        id_vars=id_cols,
        value_vars=EVAL_D_COLS,
        var_name="d",
        value_name="actual_sales",
    )

    # 5. merge
    merged = actual_long.merge(
        sub_long[["item_id", "store_id", "d", "pred_sales"]],
        on=["item_id", "store_id", "d"],
        how="left",
    )
    merged["pred_sales"] = merged["pred_sales"].fillna(0.0).clip(lower=0.0)
    logger.info("merged: %d rows, %d items",
                len(merged), merged["item_id"].nunique())

    # 6. WRMSSE 계산
    logger.info("WRMSSE 계산 중...")
    rmsse_list, weight_list = [], []
    n_missing_scale = 0

    for (item_id, store_id), grp in merged.groupby(["item_id", "store_id"]):
        key = f"{item_id}_{store_id}"
        actual = grp["actual_sales"].values
        pred = grp["pred_sales"].values
        scale = item_scales.get(key, None)
        if scale is None:
            n_missing_scale += 1
            scale = 1.0
        mse_i = float(np.mean((actual - pred) ** 2))
        rmsse_i = float(np.sqrt(mse_i / scale))
        w_i = float(actual.sum())
        rmsse_list.append(rmsse_i)
        weight_list.append(w_i)

    if n_missing_scale:
        logger.warning("scale 없는 items: %d (scale=1.0으로 대체)", n_missing_scale)

    weights = np.array(weight_list)
    total_w = weights.sum()
    wrmsse = float(np.average(rmsse_list, weights=weights)) if total_w > 0 else float("nan")

    all_actual = merged["actual_sales"].values
    all_pred = merged["pred_sales"].values
    mae_val = float(np.abs(all_actual - all_pred).mean())
    rmse_val = float(np.sqrt(np.mean((all_actual - all_pred) ** 2)))

    logger.info(
        "\n=== stephenllh 원본 재현 결과 ===\n"
        "  기간: d_1914~d_1941 (evaluation period)\n"
        "  items: %d\n"
        "  WRMSSE(item-own-lag1): %.4f\n"
        "  MAE:   %.4f\n"
        "  RMSE:  %.4f\n"
        "  참고: 원본 공개 WRMSSE = 0.637 (±10% 허용: 0.573~0.701)",
        len(rmsse_list), wrmsse, mae_val, rmse_val,
    )

    if 0.573 <= wrmsse <= 0.701:
        logger.info("✓ WRMSSE 0.637±10% 범위 내 → 원본 재현 성공")
    else:
        logger.warning("⚠ WRMSSE %.4f가 0.573~0.701 범위 밖 → 구현 차이 확인 필요", wrmsse)

    result = {
        "model": "stephenllh_m5_accuracy",
        "period": "d_1914~d_1941 (evaluation)",
        "wrmsse_item_own_lag1": wrmsse,
        "mae": mae_val,
        "rmse": rmse_val,
        "n_items": len(rmsse_list),
        "original_wrmsse": 0.637,
        "within_10pct": 0.573 <= wrmsse <= 0.701,
    }
    out_path = EXP_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
