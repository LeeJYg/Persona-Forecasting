"""Track A 예측 결과 심층 분석 스크립트.

분석 항목:
1. 주차별(17주) DirAcc 추이
2. 아이템별 MAE 분포 (상위/하위 10개)
3. 카테고리별 DirAcc 비교
4. pred_raw 분포 통계

출력: experiments/exp004_track_a_naive/analysis/
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def direction_sign(series: pd.Series) -> pd.Series:
    return series.diff().apply(np.sign)


def weekly_agg(df: pd.DataFrame, sales_col: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    return (
        df.groupby(["item_id", "iso_year", "iso_week"])[sales_col]
        .sum()
        .reset_index()
        .sort_values(["item_id", "iso_year", "iso_week"])
    )


def main() -> None:
    config = load_config()
    out_dir = ROOT / config.experiment.track_a.output_dir
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    cold_start_dir = ROOT / config.paths.cold_start_dir
    pred_dir = out_dir / "predictions"

    # ── 데이터 로드 ──────────────────────────────────────────────────
    logger.info("데이터 로드 중...")
    cold_test = pd.read_csv(cold_start_dir / "cold_test.csv", parse_dates=["date"])
    pred_cal = pd.read_csv(pred_dir / "track_a_calibrated.csv", parse_dates=["date"])
    pred_raw = pd.read_csv(pred_dir / "track_a_raw.csv", parse_dates=["date"])

    key = ["item_id", "store_id", "date"]
    merged = cold_test.merge(pred_cal[key + ["pred_sales"]], on=key, how="left")
    merged["pred_sales"] = merged["pred_sales"].fillna(0.0)
    merged["date"] = pd.to_datetime(merged["date"])
    merged["iso_year"] = merged["date"].dt.isocalendar().year.astype(int)
    merged["iso_week"] = merged["date"].dt.isocalendar().week.astype(int)

    results: dict = {}

    # ── 1. 주차별 DirAcc ────────────────────────────────────────────
    logger.info("1) 주차별 DirAcc 계산 중...")
    actual_weekly = weekly_agg(merged, "sales").rename(columns={"sales": "actual_sum"})
    pred_weekly = weekly_agg(merged, "pred_sales").rename(columns={"pred_sales": "pred_sum"})
    weekly = actual_weekly.merge(pred_weekly, on=["item_id", "iso_year", "iso_week"])
    weekly = weekly.sort_values(["item_id", "iso_year", "iso_week"])
    weekly["actual_dir"] = weekly.groupby("item_id")["actual_sum"].transform(direction_sign)
    weekly["pred_dir"] = weekly.groupby("item_id")["pred_sum"].transform(direction_sign)
    weekly = weekly.dropna(subset=["actual_dir", "pred_dir"])

    # 주차 순서 번호 부여 (1~16: diff 기준 연속 2주 비교 쌍)
    weekly["week_rank"] = weekly.groupby("item_id").cumcount() + 1  # 1-indexed after diff
    weekly_dir_acc = (
        weekly.groupby("week_rank")
        .apply(lambda g: (g["actual_dir"] == g["pred_dir"]).mean())
        .reset_index(name="direction_accuracy")
    )
    weekly_dir_acc.to_csv(analysis_dir / "weekly_dir_acc.csv", index=False)
    logger.info("  주차별 DirAcc 저장: weekly_dir_acc.csv")
    results["weekly_dir_acc"] = weekly_dir_acc.to_dict(orient="records")

    # ── 2. 아이템별 MAE 분포 ────────────────────────────────────────
    logger.info("2) 아이템별 MAE 계산 중...")
    item_mae = (
        merged.groupby(["item_id", "cat_id"])
        .apply(lambda g: pd.Series({
            "mae": float(np.abs(g["sales"] - g["pred_sales"]).mean()),
            "rmse": float(np.sqrt(np.mean((g["sales"] - g["pred_sales"]) ** 2))),
            "mean_actual": float(g["sales"].mean()),
            "mean_pred": float(g["pred_sales"].mean()),
            "n_days": len(g),
        }))
        .reset_index()
        .sort_values("mae")
    )
    item_mae.to_csv(analysis_dir / "item_mae.csv", index=False)
    logger.info("  아이템별 MAE 저장: item_mae.csv")

    top10 = item_mae.tail(10).sort_values("mae", ascending=False)
    bottom10 = item_mae.head(10)
    results["item_mae_top10_worst"] = top10[["item_id", "cat_id", "mae", "mean_actual", "mean_pred"]].to_dict(orient="records")
    results["item_mae_top10_best"] = bottom10[["item_id", "cat_id", "mae", "mean_actual", "mean_pred"]].to_dict(orient="records")
    results["item_mae_stats"] = {
        "mean": round(float(item_mae["mae"].mean()), 4),
        "median": round(float(item_mae["mae"].median()), 4),
        "std": round(float(item_mae["mae"].std()), 4),
        "min": round(float(item_mae["mae"].min()), 4),
        "max": round(float(item_mae["mae"].max()), 4),
        "p25": round(float(item_mae["mae"].quantile(0.25)), 4),
        "p75": round(float(item_mae["mae"].quantile(0.75)), 4),
    }

    # ── 3. 카테고리별 DirAcc 상세 ───────────────────────────────────
    logger.info("3) 카테고리별 DirAcc 상세 분석...")
    cat_dir_acc = {}
    for cat, grp in weekly.groupby(weekly["item_id"].map(
        merged.drop_duplicates("item_id").set_index("item_id")["cat_id"]
    )):
        correct = (grp["actual_dir"] == grp["pred_dir"]).sum()
        total = len(grp)
        # 방향 분포
        actual_dist = grp["actual_dir"].value_counts(normalize=True).to_dict()
        pred_dist = grp["pred_dir"].value_counts(normalize=True).to_dict()
        cat_dir_acc[str(cat)] = {
            "direction_accuracy": round(float(correct / total), 4),
            "n_pairs": int(total),
            "actual_direction_dist": {str(k): round(float(v), 4) for k, v in actual_dist.items()},
            "pred_direction_dist": {str(k): round(float(v), 4) for k, v in pred_dist.items()},
        }
    results["category_dir_acc"] = cat_dir_acc

    # ── 4. pred_raw 분포 ────────────────────────────────────────────
    logger.info("4) pred_raw 분포 분석...")
    raw_merged = cold_test.merge(pred_raw[key + ["pred_sales"]], on=key, how="left")
    raw_merged["pred_sales"] = raw_merged["pred_sales"].fillna(0.0)
    results["pred_raw_distribution"] = {
        "mean": round(float(raw_merged["pred_sales"].mean()), 4),
        "median": round(float(raw_merged["pred_sales"].median()), 4),
        "std": round(float(raw_merged["pred_sales"].std()), 4),
        "min": round(float(raw_merged["pred_sales"].min()), 4),
        "max": round(float(raw_merged["pred_sales"].max()), 4),
        "p25": round(float(raw_merged["pred_sales"].quantile(0.25)), 4),
        "p75": round(float(raw_merged["pred_sales"].quantile(0.75)), 4),
        "p95": round(float(raw_merged["pred_sales"].quantile(0.95)), 4),
        "pct_zero": round(float((raw_merged["pred_sales"] == 0).mean()), 4),
    }
    results["pred_calibrated_distribution"] = {
        "mean": round(float(merged["pred_sales"].mean()), 4),
        "std": round(float(merged["pred_sales"].std()), 4),
        "min": round(float(merged["pred_sales"].min()), 4),
        "max": round(float(merged["pred_sales"].max()), 4),
        "p95": round(float(merged["pred_sales"].quantile(0.95)), 4),
    }
    results["actual_distribution"] = {
        "mean": round(float(cold_test["sales"].mean()), 4),
        "std": round(float(cold_test["sales"].std()), 4),
        "min": round(float(cold_test["sales"].min()), 4),
        "max": round(float(cold_test["sales"].max()), 4),
        "p95": round(float(cold_test["sales"].quantile(0.95)), 4),
        "pct_zero": round(float((cold_test["sales"] == 0).mean()), 4),
    }

    # ── 5. 저장 ─────────────────────────────────────────────────────
    out_path = analysis_dir / "deep_analysis.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("분석 결과 저장: %s", out_path)

    # ── 6. 요약 출력 ─────────────────────────────────────────────────
    logger.info("\n=== 분석 요약 ===")
    logger.info("[주차별 DirAcc] min=%.3f  max=%.3f  mean=%.3f",
                weekly_dir_acc["direction_accuracy"].min(),
                weekly_dir_acc["direction_accuracy"].max(),
                weekly_dir_acc["direction_accuracy"].mean())
    logger.info("[아이템별 MAE] mean=%.3f  median=%.3f  std=%.3f  max=%.3f",
                results["item_mae_stats"]["mean"],
                results["item_mae_stats"]["median"],
                results["item_mae_stats"]["std"],
                results["item_mae_stats"]["max"])
    logger.info("[카테고리 DirAcc] %s",
                {k: v["direction_accuracy"] for k, v in cat_dir_acc.items()})
    logger.info("[pred_raw] mean=%.4f  std=%.4f  pct_zero=%.1f%%",
                results["pred_raw_distribution"]["mean"],
                results["pred_raw_distribution"]["std"],
                results["pred_raw_distribution"]["pct_zero"] * 100)


if __name__ == "__main__":
    main()
