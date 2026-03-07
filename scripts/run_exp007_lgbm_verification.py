"""exp007: LightGBM 구현 검증 실험 (M5 표준 세팅, warm items).

300개 warm items (CA_1) × daily prediction.
학습: 마지막 28일 제외 전체. 테스트: 마지막 28일.
M5 표준 feature engineering + WRMSSE 계산.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# --- 프로젝트 루트를 sys.path에 추가 ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 설정
# ============================================================
N_ITEMS = 300
TEST_DAYS = 28
SEED = 42
TARGET_STORE = "CA_1"
EXP_DIR = ROOT / "experiments" / "exp007_lgbm_verification"
EXP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 데이터 로드
# ============================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """warm_train(CA_1 300 items), calendar, sell_prices 로드."""
    data_dir = ROOT / "data" / "processed" / "cold_start"
    m5_dir = ROOT / "m5-forecasting-accuracy"

    # cold_ids.csv: 컬럼 "id" = "ITEM_ID_STORE_ID" 형식
    cold_ids_raw = pd.read_csv(data_dir / "cold_ids.csv")["id"].tolist()
    cold_ids = set("_".join(x.split("_")[:-1]) for x in cold_ids_raw)  # "FOODS_3_120_CA_1" → "FOODS_3_120"

    logger.info("warm_train 로드 중 (CA_1 필터링)...")
    chunks = []
    for chunk in pd.read_csv(data_dir / "warm_train.csv", parse_dates=["date"],
                              chunksize=500_000):
        filtered = chunk[
            (chunk["store_id"] == TARGET_STORE) &
            (~chunk["item_id"].isin(cold_ids))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)
    warm = pd.concat(chunks, ignore_index=True)
    logger.info("warm_train (CA_1): %d rows, %d items", len(warm), warm["item_id"].nunique())

    # 300개 random 샘플
    all_items = sorted(warm["item_id"].unique())
    rng = np.random.default_rng(SEED)
    sampled_items = rng.choice(all_items, size=N_ITEMS, replace=False).tolist()
    warm = warm[warm["item_id"].isin(sampled_items)].copy()
    logger.info("샘플링 후: %d rows, %d items", len(warm), warm["item_id"].nunique())

    calendar = pd.read_csv(m5_dir / "calendar.csv", parse_dates=["date"])
    sell_prices = pd.read_csv(m5_dir / "sell_prices.csv")
    return warm, calendar, sell_prices


# ============================================================
# 2. Feature Engineering
# ============================================================
def build_features(df: pd.DataFrame, calendar: pd.DataFrame, sell_prices: pd.DataFrame) -> pd.DataFrame:
    """M5 표준 feature matrix 생성."""
    df = df.sort_values(["item_id", "date"]).copy()

    # --- calendar merge (warm_train에 이미 month/year 있으므로 제외) ---
    cal_cols = ["date", "wday", "snap_CA", "snap_TX", "snap_WI",
                "event_type_1", "event_type_2", "wm_yr_wk"]
    cal = calendar[cal_cols].copy()
    df = df.merge(cal, on="date", how="left")
    # date에서 year/month 재계산 (warm_train 기존 컬럼 대체)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # event_type 인코딩 (event_type_1 기준)
    event_types = sorted(df["event_type_1"].dropna().unique())
    event_map = {e: i + 1 for i, e in enumerate(event_types)}  # NaN → 0
    df["event_type_enc"] = df["event_type_1"].map(event_map).fillna(0).astype(int)

    # --- sell_price merge (wm_yr_wk 기준) ---
    sp = sell_prices[sell_prices["store_id"] == TARGET_STORE][["item_id", "wm_yr_wk", "sell_price"]].copy()
    df = df.merge(sp, on=["item_id", "wm_yr_wk"], how="left")

    # price_change_ratio (전주 대비)
    df = df.sort_values(["item_id", "wm_yr_wk"])
    df["sell_price_lag1"] = df.groupby("item_id")["sell_price"].shift(7)
    df["price_change_ratio"] = (df["sell_price"] - df["sell_price_lag1"]) / (df["sell_price_lag1"] + 1e-8)
    df["price_change_ratio"] = df["price_change_ratio"].fillna(0.0)
    df = df.drop(columns=["sell_price_lag1"])

    # --- cross-sectional encoding ---
    for col in ["item_id", "dept_id", "cat_id", "store_id"]:
        df[f"{col}_enc"] = pd.Categorical(df[col]).codes

    # --- lag features (item별 정렬 필요) ---
    df = df.sort_values(["item_id", "date"])
    for lag in [7, 28]:
        df[f"lag_{lag}"] = df.groupby("item_id")["sales"].shift(lag)

    # rolling features
    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("item_id")["sales"]
            .shift(1)
            .groupby(df["item_id"])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    df["rolling_std_7"] = (
        df.groupby("item_id")["sales"]
        .shift(1)
        .groupby(df["item_id"])
        .transform(lambda x: x.rolling(7, min_periods=2).std().fillna(0.0))
    )

    return df


def get_feature_cols() -> list[str]:
    return [
        "lag_7", "lag_28",
        "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
        "wday", "month", "year",
        "snap_CA", "snap_TX", "snap_WI", "event_type_enc",
        "sell_price", "price_change_ratio",
        "item_id_enc", "dept_id_enc", "cat_id_enc", "store_id_enc",
    ]


# ============================================================
# 3. Train / Test Split
# ============================================================
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """마지막 TEST_DAYS 일을 test, 나머지를 train."""
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["date"] < test_start].copy()
    test  = df[df["date"] >= test_start].copy()
    logger.info("train: %s ~ %s (%d rows)", train["date"].min().date(),
                train["date"].max().date(), len(train))
    logger.info("test:  %s ~ %s (%d rows, %d days)",
                test["date"].min().date(), test["date"].max().date(),
                len(test), test["date"].nunique())
    return train, test


# ============================================================
# 4. WRMSSE (M5 공식)
# ============================================================
def compute_wrmsse(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pred: pd.DataFrame,
    lag: int = 28,
) -> dict[str, float]:
    """M5 공식 WRMSSE.

    scale_i = mean((y_t - y_{t-lag})^2) over train
    w_i = sum of actual sales in test (달러 가중치 대신 판매량 사용)
    WRMSSE = sum(w_i * RMSSE_i) / sum(w_i)
    """
    test_cols = ["item_id", "cat_id", "date", "sales"]
    merged = test[test_cols].merge(pred[["item_id", "date", "pred_sales"]], on=["item_id", "date"])

    item_scales: dict[str, float] = {}
    for item_id, grp in train.groupby("item_id"):
        series = grp.sort_values("date")["sales"].values
        if len(series) <= lag:
            item_scales[str(item_id)] = 1.0
            continue
        diffs = series[lag:] - series[:-lag]
        scale = float(np.mean(diffs ** 2))
        item_scales[str(item_id)] = scale if scale > 0 else 1.0

    rmsse_list, weight_list = [], []
    for item_id, grp in merged.groupby("item_id"):
        actual = grp["sales"].values
        predicted = grp["pred_sales"].values
        scale = item_scales.get(str(item_id), 1.0)
        mse_i = float(np.mean((actual - predicted) ** 2))
        rmsse_i = float(np.sqrt(mse_i / scale))
        w_i = float(actual.sum())
        rmsse_list.append(rmsse_i)
        weight_list.append(w_i)

    weights = np.array(weight_list)
    total_w = weights.sum()
    if total_w == 0:
        wrmsse = float(np.mean(rmsse_list))
    else:
        wrmsse = float(np.average(rmsse_list, weights=weights))

    all_actual = merged["sales"].values
    all_pred   = merged["pred_sales"].values
    mae  = float(np.abs(all_actual - all_pred).mean())
    rmse = float(np.sqrt(np.mean((all_actual - all_pred) ** 2)))

    return {"wrmsse": wrmsse, "mae": mae, "rmse": rmse,
            "n_items": len(rmsse_list), "n_rows": len(merged)}


# ============================================================
# 5. Main
# ============================================================
def main() -> None:
    import json
    import lightgbm as lgb

    warm, calendar, sell_prices = load_data()

    logger.info("Feature engineering 시작...")
    df = build_features(warm, calendar, sell_prices)
    feature_cols = get_feature_cols()

    # lag feature NaN 제거 (처음 28일 → 데이터 충분하므로 단순 drop)
    df = df.dropna(subset=["lag_7", "lag_28"])
    logger.info("NaN 제거 후: %d rows", len(df))

    train_full, test = split_data(df)

    # 20% validation split (시간순)
    train_dates = sorted(train_full["date"].unique())
    val_start_idx = int(len(train_dates) * 0.8)
    val_start = train_dates[val_start_idx]
    train = train_full[train_full["date"] < val_start].copy()
    val   = train_full[train_full["date"] >= val_start].copy()
    logger.info("train: %d rows, val: %d rows", len(train), len(val))

    X_train = train[feature_cols]
    y_train = train["sales"].values
    X_val   = val[feature_cols]
    y_val   = val["sales"].values
    X_test  = test[feature_cols]

    # LightGBM
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    logger.info("학습 완료 | best_iter=%d", model.best_iteration_)

    # val MAE
    val_pred = np.maximum(model.predict(X_val), 0.0)
    val_mae = float(np.abs(y_val - val_pred).mean())
    logger.info("Val MAE=%.4f", val_mae)

    # test 예측
    test = test.copy()
    test["pred_sales"] = np.maximum(model.predict(X_test), 0.0)
    pred_df = test[["item_id", "date", "pred_sales"]].copy()

    # 평가
    metrics = compute_wrmsse(train_full, test, pred_df)
    logger.info(
        "=== exp007 결과 ===\n  WRMSSE=%.4f\n  MAE=%.4f\n  RMSE=%.4f",
        metrics["wrmsse"], metrics["mae"], metrics["rmse"],
    )

    # feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # 저장
    test[["item_id", "store_id", "date", "sales", "pred_sales"]].to_csv(
        EXP_DIR / "predictions.csv", index=False
    )
    fi.to_csv(EXP_DIR / "feature_importance.csv", index=False)
    results = {**metrics, "val_mae": val_mae, "best_iteration": int(model.best_iteration_)}
    (EXP_DIR / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 출력
    print("\n" + "=" * 60)
    print("exp007: LightGBM Verification (M5 표준 세팅)")
    print("=" * 60)
    print(f"  아이템: {N_ITEMS}개 warm items (CA_1)")
    print(f"  학습 기간: {train_full['date'].min().date()} ~ {train_full['date'].max().date()}")
    print(f"  테스트 기간: {test['date'].min().date()} ~ {test['date'].max().date()} (28일)")
    print(f"  best_iteration: {model.best_iteration_}")
    print(f"  val_MAE: {val_mae:.4f}")
    print()
    print(f"  WRMSSE: {metrics['wrmsse']:.4f}   (기대 범위: 0.5~0.8)")
    print(f"  MAE:    {metrics['mae']:.4f}")
    print(f"  RMSE:   {metrics['rmse']:.4f}")
    print()
    if metrics["wrmsse"] < 0.4:
        print("  ⚠️  WRMSSE < 0.4: 데이터 누수 가능성")
    elif metrics["wrmsse"] <= 0.8:
        print("  ✓  WRMSSE 0.4~0.8: 구현 정상 (silver medal 수준)")
    else:
        print("  ⚠️  WRMSSE > 0.8: 구현 문제 가능성")
    print()
    print("  [Feature Importance Top-10]")
    print(fi.head(10).to_string(index=False))
    print("=" * 60)
    print(f"  저장 위치: {EXP_DIR}")


if __name__ == "__main__":
    main()
