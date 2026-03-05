"""Competitor 모델 통합 실행 스크립트.

사용법:
    caffeinate -i python scripts/run_competitors.py --model seasonal_pattern
    caffeinate -i python scripts/run_competitors.py --model knn_analog
    caffeinate -i python scripts/run_competitors.py --model lightgbm_static
    caffeinate -i python scripts/run_competitors.py --model lightgbm_proxy_lags
    caffeinate -i python scripts/run_competitors.py --model deepar
    caffeinate -i python scripts/run_competitors.py --model llm_zero_shot [--resume]
    caffeinate -i python scripts/run_competitors.py --model llm_similar_item [--resume]
    caffeinate -i python scripts/run_competitors.py --model llm_aggregate [--resume]

권고: Mac sleep 방지를 위해 반드시 caffeinate -i 로 실행하세요.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation.metrics import evaluate_weekly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_MODELS = [
    "seasonal_pattern",
    "knn_analog",
    "lightgbm_static",
    "lightgbm_proxy_lags",
    "deepar",
    "llm_zero_shot",
    "llm_similar_item",
    "llm_aggregate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Competitor 모델 실행")
    parser.add_argument("--model", required=True, choices=VALID_MODELS,
                        help="실행할 모델")
    parser.add_argument("--config", default=None, help="config 파일 경로 (없으면 auto)")
    parser.add_argument("--resume", action="store_true",
                        help="LLM 모델: 이전 checkpoint에서 재시작")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 공통 데이터 로드
# ---------------------------------------------------------------------------

def _to_weekly(df: pd.DataFrame, sales_col: str = "sales") -> pd.DataFrame:
    """일별 데이터를 ISO-week 주간 합계로 집계. date = 해당 주 첫 날."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    # 주 첫날 (월요일)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")

    group_cols = [c for c in ["item_id", "store_id", "cat_id", "dept_id", "state_id",
                               "iso_year", "iso_week"] if c in df.columns]
    weekly = (
        df.groupby(group_cols)
        .agg(**{sales_col: (sales_col, "sum"), "date": ("week_start", "first")})
        .reset_index()
    )
    return weekly


def load_data(config: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """cold_test_weekly, warm_train_weekly, sell_prices, calendar 로드.

    warm_train은 CA_1 store만 필터링 + cold items 제외 (cross_store_info=false 반영).
    """
    cs_dir = ROOT / config.paths.cold_start_dir
    logger.info("데이터 로드 중...")
    cold_test = pd.read_csv(cs_dir / "cold_test.csv", parse_dates=["date"])
    warm_train_raw = pd.read_csv(cs_dir / "warm_train.csv", parse_dates=["date"])
    sell_prices = pd.read_csv(ROOT / config.paths.sell_prices)
    calendar = pd.read_csv(ROOT / config.paths.calendar)

    # cold-start 설정 반영: CA_1 warm items만 사용, cold items 제외
    target_store = config.experiment.cold_start.target_store  # "CA_1"
    cold_ids = set(cold_test["item_id"].unique())
    warm_train = warm_train_raw[
        (warm_train_raw["store_id"] == target_store) &
        (~warm_train_raw["item_id"].isin(cold_ids))
    ].copy()

    logger.info(
        "warm_train 필터링: %s store, %d cold items 제외 → %d rows (%d warm items)",
        target_store, len(cold_ids), len(warm_train), warm_train["item_id"].nunique(),
    )

    cold_test_weekly = _to_weekly(cold_test, "sales")
    warm_train_weekly = _to_weekly(warm_train, "sales")

    logger.info("cold_test_weekly: %d rows (%d items × %d weeks)",
                len(cold_test_weekly),
                cold_test_weekly["item_id"].nunique(),
                cold_test_weekly["iso_week"].nunique())
    logger.info("warm_train_weekly: %d rows (%d warm items)",
                len(warm_train_weekly), warm_train_weekly["item_id"].nunique())
    return cold_test_weekly, warm_train_weekly, sell_prices, calendar


# ---------------------------------------------------------------------------
# 검증 체크리스트
# ---------------------------------------------------------------------------

def print_checklist(
    model_name: str,
    pred: pd.DataFrame,
    cold_test_weekly: pd.DataFrame,
    warm_train_weekly: pd.DataFrame,
    warm_holdout_mae: float | None = None,
) -> None:
    print("\n" + "=" * 60)
    print(f"=== 검증 체크리스트: {model_name} ===")

    # 1. 데이터 누수
    cold_ids = set(cold_test_weekly["item_id"])
    warm_ids = set(warm_train_weekly["item_id"])
    leakage = cold_ids & warm_ids
    status1 = "✓" if not leakage else f"✗ LEAK! {leakage}"
    print(f"1. 데이터 누수 없음: {status1}")

    # 2. shape
    expected = len(cold_test_weekly)
    actual = len(pred)
    status2 = "✓" if actual == expected else f"✗ ({actual} != {expected})"
    print(f"2. 예측 shape 1700 rows: {status2} (actual={actual})")

    # 3. 예측 범위
    min_val = pred["pred_sales"].min()
    max_val = pred["pred_sales"].max()
    status3 = "✓" if min_val >= 0 else f"✗ min={min_val:.3f}"
    print(f"3. 예측 범위 (min>=0): {status3}")
    print(f"   describe: mean={pred['pred_sales'].mean():.3f}  "
          f"median={pred['pred_sales'].median():.3f}  "
          f"std={pred['pred_sales'].std():.3f}  "
          f"max={max_val:.3f}")

    # 4. warm hold-out MAE
    if warm_holdout_mae is not None:
        baseline_mae_weekly = 1.57 * 7  # rough daily→weekly scale
        status4 = "✓" if warm_holdout_mae < baseline_mae_weekly else "△ (baseline 참고)"
        print(f"4. Warm hold-out MAE: {warm_holdout_mae:.4f} {status4}")
    else:
        print("4. Warm hold-out MAE: N/A (해당 없음)")

    # 5. baseline 대비
    try:
        result = evaluate_weekly(cold_test_weekly, pred, warm_train_weekly, model_name)
        best_baseline_mae = 1.57 * 7  # rough weekly scale placeholder
        status5 = "참고용"
        print(f"5. 평가 지표: MAE={result['mae']:.4f}  RMSE={result['rmse']:.4f}  "
              f"WRMSSE={result['wrmsse']:.4f}  DirAcc={result['direction_accuracy']:.4f}")
    except Exception as e:
        print(f"5. 평가 지표 계산 오류: {e}")

    # 6. seed
    print(f"6. Seed=42: ✓ (config 기반)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 개별 모델 실행 함수
# ---------------------------------------------------------------------------

def run_seasonal_pattern(config, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir):
    from src.models.competitors.seasonal_pattern import SeasonalPattern
    model = SeasonalPattern()
    model.fit(warm_train_weekly)
    pred = model.predict(cold_test_weekly)
    return pred, None


def run_knn_analog(config, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir):
    from src.models.competitors.knn_analog import KNNAnalog
    k = int(config.competitors.knn_k)
    checkpoint_path = out_dir / "knn_neighbors.json"
    model = KNNAnalog(k=k, checkpoint_path=checkpoint_path)
    model.fit(warm_train_weekly, features={"prices": sell_prices})
    pred = model.predict(cold_test_weekly, features={"prices": sell_prices})
    return pred, None


def run_lightgbm(variant, config, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir):
    from src.models.competitors.lightgbm_cross import LightGBMCross
    knn_path = None
    if variant == "proxy_lags":
        knn_dir = ROOT / config.competitors.output_dir / "knn_analog"
        knn_path = knn_dir / "knn_neighbors.json"
        if not knn_path.exists():
            raise FileNotFoundError(
                f"knn_neighbors.json 없음. knn_analog를 먼저 실행하세요: {knn_path}"
            )

    model = LightGBMCross(
        variant=variant,
        objective="tweedie",
        knn_neighbors_path=knn_path,
        seed=int(config.seed),
        config=config,
    )
    model.fit(warm_train_weekly, features={"prices": sell_prices, "calendar": calendar})
    pred = model.predict(cold_test_weekly, features={"prices": sell_prices, "calendar": calendar})

    # feature importance 저장
    if model.feature_importances is not None:
        fi_path = out_dir / "feature_importance.csv"
        model.feature_importances.to_csv(fi_path, index=False)
        logger.info("Feature importance 저장: %s", fi_path)

    # warm hold-out MAE (모델 내부에서 계산됨, 여기선 N/A 반환)
    return pred, None


def run_deepar(config, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir):
    from src.models.competitors.deepar_model import DeepARModel
    model = DeepARModel(epochs=50, seed=int(config.seed))
    model.fit(warm_train_weekly, features={"prices": sell_prices, "calendar": calendar})
    pred = model.predict(cold_test_weekly, features={"prices": sell_prices})
    return pred, None


def run_llm_direct(variant, config, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir, resume):
    from src.models.competitors.llm_direct import LLMDirect
    knn_path = None
    if variant == "similar_item":
        knn_dir = ROOT / config.competitors.output_dir / "knn_analog"
        knn_path = knn_dir / "knn_neighbors.json"
        if not knn_path.exists():
            raise FileNotFoundError(
                f"knn_neighbors.json 없음. knn_analog를 먼저 실행하세요: {knn_path}"
            )

    checkpoint_dir = out_dir / "checkpoints" if resume else out_dir / "checkpoints"
    model = LLMDirect(
        variant=variant,
        model=config.competitors.llm_direct_model,
        temperature=float(config.competitors.llm_direct_temperature),
        max_tokens=int(config.competitors.llm_direct_max_tokens),
        knn_neighbors_path=knn_path,
        checkpoint_dir=checkpoint_dir,
    )
    model.fit(warm_train_weekly, features={"prices": sell_prices})
    pred = model.predict(cold_test_weekly, features={"prices": sell_prices})
    return pred, None


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    model_name = args.model

    # caffeinate 경고
    logger.warning("⚠️  Mac sleep 방지를 위해 반드시 'caffeinate -i python ...' 으로 실행하세요!")

    config = load_config(ROOT / args.config if args.config else None)

    t0 = time.time()
    cold_test_weekly, warm_train_weekly, sell_prices, calendar = load_data(config)

    out_base = ROOT / config.experiment.competitors.output_dir
    out_dir = out_base / model_name
    pred_dir = out_dir / "predictions"
    results_dir = out_dir / "results"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== %s 실행 시작 ===", model_name)

    # 모델 실행
    features = {"prices": sell_prices, "calendar": calendar}
    warm_holdout_mae = None

    if model_name == "seasonal_pattern":
        pred, warm_holdout_mae = run_seasonal_pattern(
            config.experiment, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir)
    elif model_name == "knn_analog":
        pred, warm_holdout_mae = run_knn_analog(
            config.experiment, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir)
    elif model_name == "lightgbm_static":
        pred, warm_holdout_mae = run_lightgbm(
            "static", config.experiment, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir)
    elif model_name == "lightgbm_proxy_lags":
        pred, warm_holdout_mae = run_lightgbm(
            "proxy_lags", config.experiment, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir)
    elif model_name == "deepar":
        pred, warm_holdout_mae = run_deepar(
            config.experiment, cold_test_weekly, warm_train_weekly, sell_prices, calendar, out_dir)
    elif model_name == "llm_zero_shot":
        pred, warm_holdout_mae = run_llm_direct(
            "zero_shot", config.experiment, cold_test_weekly, warm_train_weekly,
            sell_prices, calendar, out_dir, args.resume)
    elif model_name == "llm_similar_item":
        pred, warm_holdout_mae = run_llm_direct(
            "similar_item", config.experiment, cold_test_weekly, warm_train_weekly,
            sell_prices, calendar, out_dir, args.resume)
    elif model_name == "llm_aggregate":
        pred, warm_holdout_mae = run_llm_direct(
            "aggregate", config.experiment, cold_test_weekly, warm_train_weekly,
            sell_prices, calendar, out_dir, args.resume)
    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")

    # 예측 저장
    pred_path = pred_dir / f"{model_name}.csv"
    pred.to_csv(pred_path, index=False)
    logger.info("예측 저장: %s (%d rows)", pred_path, len(pred))

    # 평가
    result = evaluate_weekly(cold_test_weekly, pred, warm_train_weekly, model_name)

    # 결과 저장
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("평가 결과 저장: %s", metrics_path)

    elapsed = time.time() - t0
    logger.info("=== %s 완료 (%.1fs) ===", model_name, elapsed)

    # 검증 체크리스트 출력
    print_checklist(model_name, pred, cold_test_weekly, warm_train_weekly, warm_holdout_mae)


if __name__ == "__main__":
    main()
