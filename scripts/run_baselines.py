"""
Cold-start 베이스라인 실험 실행 스크립트 (thin runner).

사용법:
    python -m scripts.run_baselines
    python -m scripts.run_baselines --config configs/config.yaml

실행 결과:
    experiments/exp002_cold_start_baselines/
        predictions/
            global_category_average.csv
            similar_item_average.csv
            store_category_average.csv
        metrics/
            evaluation_results.json
        summary.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation.metrics import evaluate
from src.models.baselines import (
    GlobalCategoryAverage,
    SimilarItemAverage,
    StoreCategoryAverage,
)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_baselines_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-start 베이스라인 실험 실행")
    parser.add_argument(
        "--config", type=str, default=None,
        help="config.yaml 경로 (기본값: configs/config.yaml)",
    )
    return parser.parse_args()


def load_data(config, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """warm_train, cold_test, sell_prices, cold_item_stats 로드."""
    cs_dir = root / config.paths.cold_start_dir
    logger = logging.getLogger(__name__)

    logger.info("Loading warm_train ...")
    warm_train = pd.read_csv(cs_dir / "warm_train.csv", parse_dates=["date"])
    logger.info("warm_train: %s rows", f"{len(warm_train):,}")

    logger.info("Loading cold_test ...")
    cold_test = pd.read_csv(cs_dir / "cold_test.csv", parse_dates=["date"])
    logger.info("cold_test: %s rows, %d items",
                f"{len(cold_test):,}", cold_test["item_id"].nunique())

    logger.info("Loading sell_prices ...")
    sell_prices = pd.read_csv(root / config.paths.sell_prices)

    logger.info("Loading cold_item_stats ...")
    cold_item_stats = pd.read_csv(cs_dir / "cold_item_stats.csv")

    return warm_train, cold_test, sell_prices, cold_item_stats


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    out_dir = ROOT / config.experiment.baselines.output_dir
    setup_logging(out_dir / "logs")
    logger = logging.getLogger(__name__)

    pred_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 데이터 로드
    warm_train, cold_test, sell_prices, _ = load_data(config, ROOT)
    features = {"prices": sell_prices}

    target_store: str = config.experiment.cold_start.target_store
    n_quantiles: int = config.experiment.baselines.price_tier_n_quantiles
    price_lookback: int = config.experiment.baselines.price_lookback_weeks

    # ------------------------------------------------------------------ 모델 정의
    models = [
        GlobalCategoryAverage(),
        SimilarItemAverage(
            n_quantiles=n_quantiles,
            price_lookback_weeks=price_lookback,
        ),
        StoreCategoryAverage(target_store=target_store),
    ]

    # ------------------------------------------------------------------ 실험 루프
    all_results: list[dict] = []

    for model in models:
        logger.info("=" * 60)
        logger.info("Running model: %s", model.name)

        # fit: B2만 prices feature 필요, 나머지는 무시됨
        model.fit(warm_train, features=features)

        # predict
        pred = model.predict(cold_test, features=features)

        # 예측값 저장
        pred_path = pred_dir / f"{model.name}.csv"
        pred.to_csv(pred_path, index=False)
        logger.info("Predictions saved: %s", pred_path)

        # 평가
        result = evaluate(
            cold_test=cold_test,
            pred=pred,
            warm_train=warm_train,
            model_name=model.name,
        )
        all_results.append(result)

    # ------------------------------------------------------------------ 결과 저장
    results_path = metrics_dir / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation results saved: %s", results_path)

    # 요약 테이블 (콘솔 출력 + CSV 저장)
    summary_rows = [
        {
            "model": r["model"],
            "mae": round(r["mae"], 4),
            "rmse": round(r["rmse"], 4),
            "wrmsse": round(r["wrmsse"], 4),
            "direction_accuracy": round(r["direction_accuracy"], 4),
            "n_items": r["n_items"],
        }
        for r in all_results
    ]
    summary_df = pd.DataFrame(summary_rows).set_index("model")
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path)

    logger.info("\n%s\n%s", "=" * 60, summary_df.to_string())
    logger.info("Done. Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
