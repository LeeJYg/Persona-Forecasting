"""평가 지표: MAE, RMSE, WRMSSE (cold-start 수정버전), Direction Accuracy (weekly).

Cold-Start WRMSSE 처리 방침
----------------------------
M5 공식 WRMSSE는 아이템별 학습 기간의 판매 시계열을 사용해 스케일 인수(scale_i)를
계산한다. Cold-start 아이템은 학습 기간 이력이 없으므로 scale_i가 정의되지 않는다.
본 구현에서는 scale_i를 같은 카테고리 warm 아이템들의 평균 스케일로 대체한다.
(논문 Appendix에 명시 예정)

Direction Accuracy (Weekly) 정의
---------------------------------
1. 각 (item_id, ISO_week) 단위로 actual_sales, pred_sales를 합산.
2. 연속된 두 주 사이의 방향: sign(week_t - week_{t-1}).
   sign > 0 → "up", sign < 0 → "down", sign = 0 → "flat".
3. DA = 예측 방향이 실제 방향과 일치하는 비율.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# M5 WRMSSE 28일 래그 (공식 기준)
_WRMSSE_LAG = 28


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def evaluate(
    cold_test: pd.DataFrame,
    pred: pd.DataFrame,
    warm_train: pd.DataFrame,
    model_name: str = "unknown",
) -> dict[str, Any]:
    """
    cold_test(실측)와 pred(예측)를 비교해 모든 지표를 계산한다.

    Args:
        cold_test: cold-start 아이템의 실제 판매량 데이터프레임.
                   컬럼: item_id, store_id, date, sales, cat_id 등.
        pred:      predict()가 반환한 데이터프레임.
                   컬럼: item_id, store_id, date, pred_sales (필수).
        warm_train: WRMSSE 스케일 계산용 warm 아이템 학습 데이터.
        model_name: 로깅/결과 키에 사용할 모델 이름.

    Returns:
        dict: {
            "model"              : str,
            "mae"                : float,
            "rmse"               : float,
            "wrmsse"             : float,
            "direction_accuracy" : float,
            "by_category"        : { cat_id: {mae, rmse, wrmsse, direction_accuracy} },
            "n_items"            : int,
            "n_rows"             : int,
        }
    """
    merged = _merge(cold_test, pred)

    result: dict[str, Any] = {
        "model": model_name,
        "n_items": merged["item_id"].nunique(),
        "n_rows": len(merged),
    }

    result["mae"] = mae(merged["sales"], merged["pred_sales"])
    result["rmse"] = rmse(merged["sales"], merged["pred_sales"])
    result["wrmsse"] = wrmsse(merged, warm_train)
    result["direction_accuracy"] = direction_accuracy_weekly(merged)

    # 카테고리별 세분화
    by_cat: dict[str, dict[str, float]] = {}
    for cat, grp in merged.groupby("cat_id"):
        warm_cat = warm_train[warm_train["cat_id"] == cat]
        by_cat[str(cat)] = {
            "mae": mae(grp["sales"], grp["pred_sales"]),
            "rmse": rmse(grp["sales"], grp["pred_sales"]),
            "wrmsse": wrmsse(grp, warm_cat),
            "direction_accuracy": direction_accuracy_weekly(grp),
        }
    result["by_category"] = by_cat

    logger.info(
        "[%s] MAE=%.4f  RMSE=%.4f  WRMSSE=%.4f  DirAcc=%.4f",
        model_name,
        result["mae"],
        result["rmse"],
        result["wrmsse"],
        result["direction_accuracy"],
    )
    return result


# ---------------------------------------------------------------------------
# 개별 지표 함수
# ---------------------------------------------------------------------------


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Error."""
    return float(np.abs(actual.values - predicted.values).mean())


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual.values - predicted.values) ** 2)))


def wrmsse(
    merged: pd.DataFrame,
    warm_train: pd.DataFrame,
    lag: int = _WRMSSE_LAG,
) -> float:
    """
    Weighted Root Mean Squared Scaled Error (M5 공식 지표, cold-start 수정버전).

    Cold 아이템의 scale은 같은 cat_id warm 아이템들의 평균 scale로 대체.
    가중치(w_i)는 테스트 기간 실제 판매량 합으로 계산 (학습 기간 이력 없으므로).

    Args:
        merged: _merge()로 결합된 DataFrame (item_id, sales, pred_sales, cat_id 필수).
        warm_train: warm 아이템 학습 데이터 (scale 계산용).
        lag: WRMSSE 래그 (기본 28).

    Returns:
        float: WRMSSE 값. 값이 작을수록 좋음.
    """
    # 1. warm 아이템별 scale 계산 (28-day lag MSE)
    cat_scales = _compute_category_scales(warm_train, lag=lag)

    item_results: list[float] = []
    item_weights: list[float] = []

    for item_id, grp in merged.groupby("item_id"):
        cat = grp["cat_id"].iloc[0]
        scale = cat_scales.get(str(cat), 1.0)
        if scale == 0.0:
            scale = 1.0  # 분모 0 방지

        mse_i = float(np.mean((grp["sales"].values - grp["pred_sales"].values) ** 2))
        rmsse_i = np.sqrt(mse_i / scale)
        w_i = float(grp["sales"].sum())  # 테스트 기간 판매량으로 가중치

        item_results.append(rmsse_i)
        item_weights.append(w_i)

    if not item_results:
        return float("nan")

    weights = np.array(item_weights)
    total_weight = weights.sum()
    if total_weight == 0:
        # 모든 실제 판매가 0인 경우: 단순 평균
        return float(np.mean(item_results))

    return float(np.average(item_results, weights=weights))


def direction_accuracy_weekly(merged: pd.DataFrame) -> float:
    """
    주간(weekly) 집계 기준 방향성 정확도.

    연속된 두 주의 판매량 합을 비교해 방향(up/down/flat)이 일치하는 비율을 반환한다.
    상수 예측 모델은 주간 합이 모두 동일하므로 예측 방향이 항상 "flat"이 되며,
    이는 실제 방향이 flat인 주 비율에 해당한다.

    Args:
        merged: item_id, date, sales, pred_sales 컬럼 포함 DataFrame.

    Returns:
        float: [0, 1] 범위의 방향 정확도. 높을수록 좋음.
    """
    df = merged.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)

    # (item_id, year, week) 단위 주간 합산
    weekly = (
        df.groupby(["item_id", "iso_year", "iso_week"])
        .agg(actual_sum=("sales", "sum"), pred_sum=("pred_sales", "sum"))
        .reset_index()
        .sort_values(["item_id", "iso_year", "iso_week"])
    )

    # 전주 대비 방향 계산
    weekly["actual_dir"] = (
        weekly.groupby("item_id")["actual_sum"].diff().apply(np.sign)
    )
    weekly["pred_dir"] = (
        weekly.groupby("item_id")["pred_sum"].diff().apply(np.sign)
    )

    # 첫 주는 diff=NaN → 제외
    weekly = weekly.dropna(subset=["actual_dir", "pred_dir"])
    if weekly.empty:
        logger.warning("Not enough weeks for Direction Accuracy (need ≥ 2 weeks per item).")
        return float("nan")

    correct = (weekly["actual_dir"] == weekly["pred_dir"]).sum()
    total = len(weekly)
    return float(correct / total)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _merge(cold_test: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """cold_test와 pred를 (item_id, store_id, date) 기준으로 결합."""
    key = ["item_id", "store_id", "date"]
    merged = cold_test.merge(pred[key + ["pred_sales"]], on=key, how="left")
    missing = merged["pred_sales"].isna().sum()
    if missing:
        logger.warning("evaluate(): %d rows have no prediction → filling with 0.", missing)
        merged["pred_sales"] = merged["pred_sales"].fillna(0.0)
    # cat_id 컬럼이 cold_test에서 오는지 확인
    if "cat_id" not in merged.columns and "cat_id" in pred.columns:
        merged = merged.merge(pred[["item_id", "cat_id"]].drop_duplicates(), on="item_id")
    return merged


def _compute_category_scales(
    warm_train: pd.DataFrame, lag: int = _WRMSSE_LAG
) -> dict[str, float]:
    """
    카테고리별 WRMSSE 스케일을 계산한다.

    각 warm 아이템에 대해 lag-day 차분 MSE를 계산하고,
    카테고리 내 평균을 반환한다.

    Returns:
        dict: {cat_id: scale}
    """
    if warm_train.empty:
        return {}

    cat_scales: dict[str, float] = {}
    for cat, cat_grp in warm_train.groupby("cat_id"):
        item_scales: list[float] = []
        for _, item_grp in cat_grp.groupby("item_id"):
            series = item_grp.sort_values("date")["sales"].values
            if len(series) <= lag:
                continue
            diffs = series[lag:] - series[:-lag]
            scale = float(np.mean(diffs ** 2))
            if scale > 0:
                item_scales.append(scale)
        cat_scales[str(cat)] = float(np.mean(item_scales)) if item_scales else 1.0

    logger.debug("Category WRMSSE scales: %s",
                 {k: round(v, 4) for k, v in cat_scales.items()})
    return cat_scales
