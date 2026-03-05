"""Track B 실행 스크립트: Qwen 2.5 32B 임베딩 + Ridge 회귀 헤드.

연구실 서버에서 git pull 후 바로 실행 가능하도록 설계.

사용법:
    # 전체 실행 (GPU 자동 감지)
    python scripts/run_track_b.py

    # GPU 지정 (단일 GPU 또는 멀티)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_track_b.py

    # 임베딩 재사용 (이미 추출된 경우 skip)
    python scripts/run_track_b.py --skip-embedding

    # 아이템 수 제한 (빠른 테스트)
    python scripts/run_track_b.py --n-warm 10 --n-cold 5

    # CPU 모드 (디버깅용)
    python scripts/run_track_b.py --device-map cpu --dtype float32

파이프라인:
    1. cold_test에서 100개 cold 아이템 메타데이터 수집
    2. warm_train에서 300개 warm 아이템 샘플링 + 메타데이터
    3. Qwen 임베딩 추출:
       - cold 100 items × 50 personas = 5,000 forward passes → item_emb_cold (100, 5120)
       - warm 300 items × 50 personas = 15,000 forward passes → item_emb_warm (300, 5120)
       → 총 20,000 forward passes
    4. warm item 실제 주간 판매량 집계 (y_warm_weekly)
    5. Ridge 회귀 헤드 학습: (item_emb_warm, y_warm_weekly)
    6. cold item 예측: item_emb_cold → y_cold_weekly
    7. 평가 및 결과 저장

출력:
    experiments/exp005_track_b_embedding/
        embeddings/
            item_emb_cold.npz      ← cold 아이템 임베딩
            item_emb_warm.npz      ← warm 아이템 임베딩
        predictions/
            track_b_pred.csv
        models/
            regression_head.pkl
        metrics/
            evaluation_results.json
        run_metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation.metrics import evaluate
from src.models.forecasting.linear_head import (
    WeeklySalesHead,
    aggregate_weekly_sales,
    build_pred_dataframe,
)
from src.models.forecasting.qwen_embedder import QwenEmbedder
from src.models.persona.schema import Persona


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_track_b_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track B: Qwen 임베딩 + Ridge 회귀 예측 실행",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="config.yaml 경로")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="임베딩 추출 단계를 건너뜀 (기존 .npz 파일 재사용)",
    )
    parser.add_argument(
        "--n-warm",
        type=int,
        default=None,
        help="회귀 헤드 학습에 사용할 warm 아이템 수 (기본: config 값)",
    )
    parser.add_argument(
        "--n-cold",
        type=int,
        default=None,
        help="예측할 cold 아이템 수 제한 (기본: 전체 100개)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="HuggingFace device_map ('auto', 'cuda', 'cpu', ...)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="모델 dtype ('bfloat16', 'float16', 'float32'). 기본: config 값",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="임베딩 배치 크기 (기본: config 값)",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="HuggingFace 모델 캐시 디렉토리",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_personas(personas_dir: Path) -> list[Persona]:
    files = sorted(personas_dir.glob("CA_1_P*.json"))
    personas: list[Persona] = []
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            personas.append(Persona.from_dict(data))
        except Exception as e:
            logging.getLogger(__name__).warning("페르소나 로드 실패 %s: %s", fp.name, e)
    return personas


def sample_warm_items(
    warm_train: pd.DataFrame,
    cold_item_ids: set[str],
    n_warm: int,
    seed: int,
) -> list[str]:
    """warm 아이템에서 n_warm개를 카테고리 균형 샘플링한다."""
    warm_items = (
        warm_train[~warm_train["item_id"].isin(cold_item_ids)]
        [["item_id", "cat_id"]]
        .drop_duplicates("item_id")
    )
    categories = sorted(warm_items["cat_id"].unique())
    n_cats = len(categories)
    per_cat = n_warm // n_cats
    remainder = n_warm % n_cats

    sampled: list[str] = []
    rng = np.random.default_rng(seed)
    for i, cat in enumerate(categories):
        n = per_cat + (1 if i < remainder else 0)
        cat_items = warm_items[warm_items["cat_id"] == cat]["item_id"].tolist()
        chosen = rng.choice(cat_items, size=min(n, len(cat_items)), replace=False)
        sampled.extend(chosen.tolist())

    logging.getLogger(__name__).info(
        "warm 아이템 샘플링: %d / %d", len(sampled), len(warm_items)
    )
    return sampled


def build_item_meta(
    item_ids: list[str],
    sales_df: pd.DataFrame,
    sell_prices: pd.DataFrame,
    store_id: str,
    lookback_weeks: int = 13,
) -> dict[str, dict]:
    """아이템 메타데이터 (dept_id, cat_id, avg_price)를 수집한다."""
    meta_df = (
        sales_df[sales_df["item_id"].isin(item_ids)]
        [["item_id", "dept_id", "cat_id"]]
        .drop_duplicates("item_id")
        .set_index("item_id")
    )

    # 가격: sell_prices에서 최근 lookback_weeks 평균
    price_df = sell_prices[
        (sell_prices["store_id"] == store_id)
        & (sell_prices["item_id"].isin(item_ids))
    ].groupby("item_id").apply(
        lambda g: g.nlargest(lookback_weeks, "wm_yr_wk")["sell_price"].mean()
    ).rename("avg_price")

    result: dict[str, dict] = {}
    for iid in item_ids:
        row = meta_df.loc[iid] if iid in meta_df.index else None
        result[iid] = {
            "dept_id": str(row["dept_id"]) if row is not None else "UNKNOWN",
            "cat_id": str(row["cat_id"]) if row is not None else "UNKNOWN",
            "avg_price": float(price_df.loc[iid]) if iid in price_df.index else None,
        }
    return result


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config if args.config else None)
    tb_cfg = config.experiment.track_b
    out_dir = ROOT / tb_cfg.output_dir
    log_dir = ROOT / config.paths.log_dir

    setup_logging(log_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("=== Track B 시작 ===")
    logger.info("출력 디렉토리: %s", out_dir)

    # 설정값 결합 (CLI override 우선)
    n_warm = args.n_warm or int(tb_cfg.n_warm_items_for_training)
    embedding_batch_size = args.batch_size or int(tb_cfg.embedding_batch_size)
    dtype = args.dtype or str(tb_cfg.embedding_dtype)
    model_name = str(tb_cfg.model_name)
    # quantization 설정: DotDict → 일반 dict로 변환
    quant_cfg = dict(tb_cfg.quantization) if hasattr(tb_cfg, "quantization") else {"mode": "none"}
    seed = int(config.experiment.seed)

    # ---------------------------------------------------------------- #
    # 1. 데이터 로드                                                    #
    # ---------------------------------------------------------------- #
    cold_start_dir = ROOT / config.paths.cold_start_dir
    logger.info("데이터 로드 중...")
    cold_test = pd.read_csv(cold_start_dir / "cold_test.csv", parse_dates=["date"])
    warm_train = pd.read_csv(cold_start_dir / "warm_train.csv", parse_dates=["date"])
    warm_test = pd.read_csv(cold_start_dir / "warm_test.csv", parse_dates=["date"])
    sell_prices = pd.read_csv(ROOT / config.paths.sell_prices)
    store_id = str(config.experiment.cold_start.target_store)

    cold_item_ids = sorted(cold_test["item_id"].unique().tolist())
    if args.n_cold is not None:
        cold_item_ids = cold_item_ids[: args.n_cold]
        cold_test = cold_test[cold_test["item_id"].isin(cold_item_ids)]

    # warm 아이템 샘플링
    warm_item_ids = sample_warm_items(
        warm_train, set(cold_item_ids), n_warm, seed
    )

    logger.info(
        "cold items: %d, warm items (for training): %d",
        len(cold_item_ids), len(warm_item_ids),
    )

    # ---------------------------------------------------------------- #
    # 2. 페르소나 로드                                                  #
    # ---------------------------------------------------------------- #
    personas = load_personas(ROOT / config.paths.personas_dir)
    n_personas = int(tb_cfg.n_personas_for_embedding)
    personas = personas[:n_personas]
    if not personas:
        logger.error("페르소나 없음: %s", ROOT / config.paths.personas_dir)
        sys.exit(1)
    logger.info("페르소나: %d개", len(personas))

    # ---------------------------------------------------------------- #
    # 3. 아이템 메타데이터                                               #
    # ---------------------------------------------------------------- #
    all_item_ids = cold_item_ids + warm_item_ids
    all_sales_df = pd.concat([cold_test, warm_train], ignore_index=True)
    item_meta = build_item_meta(all_item_ids, all_sales_df, sell_prices, store_id)

    # ---------------------------------------------------------------- #
    # 4. Qwen 임베딩 추출                                               #
    # ---------------------------------------------------------------- #
    emb_dir = out_dir / "embeddings"
    emb_cold_path = emb_dir / "item_emb_cold.npz"
    emb_warm_path = emb_dir / "item_emb_warm.npz"

    if args.skip_embedding and emb_cold_path.exists() and emb_warm_path.exists():
        logger.info("임베딩 로드 (skip-embedding 모드): %s", emb_dir)
        item_emb_cold, _ = QwenEmbedder.load_embeddings(emb_cold_path)
        item_emb_warm, _ = QwenEmbedder.load_embeddings(emb_warm_path)
    else:
        embedder = QwenEmbedder(
            model_name=model_name,
            dtype=dtype,
            quantization=quant_cfg,
            device_map=args.device_map,
            batch_size=embedding_batch_size,
            cache_dir=args.model_cache_dir,
        )
        embedder.load()

        logger.info("cold 아이템 임베딩 추출 중 (%d × %d)...", len(cold_item_ids), len(personas))
        item_emb_cold = embedder.build_item_embeddings(
            item_ids=cold_item_ids,
            item_meta=item_meta,
            personas=personas,
            condition="A",
        )
        embedder.save_embeddings(item_emb_cold, cold_item_ids, emb_cold_path)

        logger.info("warm 아이템 임베딩 추출 중 (%d × %d)...", len(warm_item_ids), len(personas))
        item_emb_warm = embedder.build_item_embeddings(
            item_ids=warm_item_ids,
            item_meta=item_meta,
            personas=personas,
            condition="A",
        )
        embedder.save_embeddings(item_emb_warm, warm_item_ids, emb_warm_path)

    logger.info(
        "임베딩 shape: cold=%s, warm=%s",
        item_emb_cold.shape, item_emb_warm.shape,
    )

    # ---------------------------------------------------------------- #
    # 5. warm 아이템 주간 판매량 집계 (regression target)              #
    # ---------------------------------------------------------------- #
    date_start = str(cold_test["date"].min().date())
    date_end = str(cold_test["date"].max().date())
    logger.info("주간 판매량 집계: %s ~ %s", date_start, date_end)

    # warm_test 기간 중 cold_test와 동일한 날짜 범위 사용
    y_warm_weekly = aggregate_weekly_sales(
        sales_df=warm_test,
        item_ids=warm_item_ids,
        date_start=date_start,
        date_end=date_end,
    )
    logger.info("y_warm_weekly shape: %s", y_warm_weekly.shape)

    # ---------------------------------------------------------------- #
    # 6. Ridge 회귀 헤드 학습                                          #
    # ---------------------------------------------------------------- #
    alpha = float(tb_cfg.regression_alpha)
    cv_folds = int(tb_cfg.regression_cv_folds)

    head = WeeklySalesHead(
        alpha=alpha if alpha > 0 else None,
        cv_folds=cv_folds,
    )
    head.fit(item_emb_warm, y_warm_weekly)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    head.save(model_dir / "regression_head.pkl")

    # ---------------------------------------------------------------- #
    # 7. cold 아이템 예측                                              #
    # ---------------------------------------------------------------- #
    y_cold_weekly = head.predict(item_emb_cold)
    # shape: (n_cold, n_weeks)

    # 주간 예측 → 일별 DataFrame
    pred_df = build_pred_dataframe(
        y_weekly=y_cold_weekly,
        item_ids=cold_item_ids,
        store_id=store_id,
        cold_test=cold_test,
    )

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_dir / "track_b_pred.csv", index=False)
    logger.info("예측 저장: %s", pred_dir / "track_b_pred.csv")

    # ---------------------------------------------------------------- #
    # 8. 평가                                                           #
    # ---------------------------------------------------------------- #
    metrics = evaluate(
        cold_test=cold_test,
        pred=pred_df,
        warm_train=warm_train,
        model_name="track_b_embedding",
    )

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "run_at": datetime.now().isoformat(),
        "model": model_name,
        "n_cold_items": len(cold_item_ids),
        "n_warm_items": len(warm_item_ids),
        "n_personas": len(personas),
        "embedding_dim": int(item_emb_cold.shape[1]),
        "best_alpha": head._best_alpha,
        "date_start": date_start,
        "date_end": date_end,
        "metrics": metrics,
    }
    (metrics_dir / "evaluation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------- #
    # 9. 요약 출력                                                      #
    # ---------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("=== Track B 결과 요약 ===")
    logger.info(
        "  MAE=%.4f  RMSE=%.4f  WRMSSE=%.4f  DirAcc=%.4f",
        metrics["mae"], metrics["rmse"],
        metrics["wrmsse"], metrics["direction_accuracy"],
    )
    logger.info("베이스라인 참고 (exp002): MAE≈1.57~1.69 / WRMSSE≈2.96~2.99 / DirAcc≈0.257")
    logger.info("카테고리별:")
    for cat, cat_m in metrics.get("by_category", {}).items():
        logger.info(
            "  %s: MAE=%.4f  DirAcc=%.4f",
            cat, cat_m["mae"], cat_m["direction_accuracy"],
        )
    logger.info("전체 결과 저장: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
