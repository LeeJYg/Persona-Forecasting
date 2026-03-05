"""전체 모델 통합 비교 테이블 생성 스크립트.

완료된 competitor 모델만 포함 (부분 결과 지원).
기존 naive baseline(exp002)과 Track A(exp004)도 weekly 재평가해서 포함.

사용법:
    python scripts/compare_all_models.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation.metrics import evaluate_weekly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _to_weekly(df: pd.DataFrame, sales_col: str = "sales") -> pd.DataFrame:
    """일별 데이터를 ISO-week 주간 합계로 집계."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    group_cols = [c for c in ["item_id", "store_id", "cat_id", "dept_id",
                               "iso_year", "iso_week"] if c in df.columns]
    return (
        df.groupby(group_cols)
        .agg(**{sales_col: (sales_col, "sum"), "date": ("week_start", "first")})
        .reset_index()
    )


def _build_row(result: dict) -> dict:
    return {
        "model": result["model"],
        "MAE": round(result["mae"], 4),
        "RMSE": round(result["rmse"], 4),
        "WRMSSE": round(result["wrmsse"], 4),
        "DirAcc": round(result["direction_accuracy"], 4),
        "n_items": result["n_items"],
        "n_rows": result["n_rows"],
    }


def main() -> None:
    config = load_config()
    cs_dir = ROOT / config.paths.cold_start_dir
    exp002_dir = ROOT / config.experiment.baselines.output_dir
    exp004_dir = ROOT / config.experiment.track_a.output_dir
    comp_dir = ROOT / config.experiment.competitors.output_dir

    # 데이터 로드 (CA_1 warm items만 사용, cold items 제외)
    logger.info("데이터 로드 중...")
    cold_test = pd.read_csv(cs_dir / "cold_test.csv", parse_dates=["date"])
    warm_train_raw = pd.read_csv(cs_dir / "warm_train.csv", parse_dates=["date"])
    cold_ids_set = set(cold_test["item_id"].unique())
    target_store = config.experiment.cold_start.target_store
    warm_train = warm_train_raw[
        (warm_train_raw["store_id"] == target_store) &
        (~warm_train_raw["item_id"].isin(cold_ids_set))
    ].copy()
    logger.info("warm_train 필터링: %s store → %d rows (%d warm items)",
                target_store, len(warm_train), warm_train["item_id"].nunique())
    cold_test_weekly = _to_weekly(cold_test, "sales")
    warm_train_weekly = _to_weekly(warm_train, "sales")

    # 불완전한 주 제거 (7일 미만 ISO week)
    days_per_week = (
        cold_test.assign(
            iso_year=cold_test["date"].dt.isocalendar().year.astype(int),
            iso_week=cold_test["date"].dt.isocalendar().week.astype(int),
        )
        .groupby(["iso_year", "iso_week"])["date"]
        .nunique()
        .reset_index(name="n_days")
    )
    complete_weeks = days_per_week[days_per_week["n_days"] == 7][["iso_year", "iso_week"]]
    incomplete = days_per_week[days_per_week["n_days"] < 7]
    if not incomplete.empty:
        logger.warning("불완전한 주 제거: %s", incomplete.to_dict("records"))
    cold_test_weekly = cold_test_weekly.merge(complete_weeks, on=["iso_year", "iso_week"])
    logger.info("cold_test_weekly (완전한 주만): %d rows (%d items × %d weeks)",
                len(cold_test_weekly), cold_test_weekly["item_id"].nunique(),
                cold_test_weekly["iso_week"].nunique())
    cold_ids = set(cold_test_weekly["item_id"].unique())

    all_rows: list[dict] = []

    # ── 1. Naive baselines (exp002) ─────────────────────────────────
    baseline_files = {
        "global_category_average": exp002_dir / "predictions" / "global_category_average.csv",
        "similar_item_average": exp002_dir / "predictions" / "similar_item_average.csv",
        "store_category_average": exp002_dir / "predictions" / "store_category_average.csv",
    }
    for bname, bpath in baseline_files.items():
        if not bpath.exists():
            logger.warning("건너뜀 (파일 없음): %s", bpath)
            continue
        pred_daily = pd.read_csv(bpath, parse_dates=["date"])
        pred_daily = pred_daily[pred_daily["item_id"].isin(cold_ids)]
        pred_weekly = _to_weekly(pred_daily, "pred_sales")
        result = evaluate_weekly(cold_test_weekly, pred_weekly, warm_train_weekly, bname)
        all_rows.append(_build_row(result))
        logger.info("[%s] MAE=%.4f RMSE=%.4f WRMSSE=%.4f DirAcc=%.4f",
                    bname, result["mae"], result["rmse"], result["wrmsse"], result["direction_accuracy"])

    # ── 2. Track A calibrated (exp004) ──────────────────────────────
    track_a_path = exp004_dir / "predictions" / "track_a_calibrated.csv"
    if track_a_path.exists():
        pred_daily = pd.read_csv(track_a_path, parse_dates=["date"])
        pred_daily = pred_daily[pred_daily["item_id"].isin(cold_ids)]
        pred_weekly = _to_weekly(pred_daily, "pred_sales")
        result = evaluate_weekly(cold_test_weekly, pred_weekly, warm_train_weekly, "track_a_calibrated")
        all_rows.append(_build_row(result))
        logger.info("[track_a_calibrated] MAE=%.4f DirAcc=%.4f", result["mae"], result["direction_accuracy"])

    # ── 3. Competitors (exp006) ──────────────────────────────────────
    competitor_models = [
        "seasonal_pattern",
        "knn_analog",
        "lightgbm_static",
        "lightgbm_proxy_lags",
        "deepar",
        "llm_zero_shot",
        "llm_similar_item",
        "llm_aggregate",
    ]
    for mname in competitor_models:
        metrics_path = comp_dir / mname / "results" / "metrics.json"
        if metrics_path.exists():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            all_rows.append(_build_row(result))
            logger.info("[%s] MAE=%.4f DirAcc=%.4f", mname, result["mae"], result["direction_accuracy"])
        else:
            logger.info("건너뜀 (미완료): %s", mname)

    # ── 4. 테이블 구성 및 저장 ────────────────────────────────────────
    model_order = [
        "global_category_average",
        "similar_item_average",
        "store_category_average",
        "seasonal_pattern",
        "knn_analog",
        "lightgbm_static",
        "lightgbm_proxy_lags",
        "deepar",
        "llm_zero_shot",
        "llm_similar_item",
        "llm_aggregate",
        "track_a_calibrated",
    ]
    df = pd.DataFrame(all_rows)
    df["_order"] = df["model"].map({m: i for i, m in enumerate(model_order)}).fillna(99)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    out_path = comp_dir / "comparison_table.csv"
    df.to_csv(out_path, index=False)
    logger.info("비교 테이블 저장: %s (%d models)", out_path, len(df))

    # ── 5. 콘솔 출력 ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("=== 전체 모델 비교 (Weekly 단위, cold_test 17주) ===")
    print(df.to_string(index=False))
    print("=" * 80)
    print("(↓ 낮을수록 좋음: MAE/RMSE/WRMSSE  |  ↑ 높을수록 좋음: DirAcc)")

    if not df.empty:
        best_mae_row = df.loc[df["MAE"].idxmin()]
        best_dir_row = df.loc[df["DirAcc"].idxmax()]
        print(f"\n최고 MAE:    {best_mae_row['model']} ({best_mae_row['MAE']:.4f})")
        print(f"최고 DirAcc: {best_dir_row['model']} ({best_dir_row['DirAcc']:.4f})")

    # JSON 저장
    json_path = comp_dir / "comparison_table.json"
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("JSON 저장: %s", json_path)


if __name__ == "__main__":
    main()
