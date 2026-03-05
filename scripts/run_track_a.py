"""Track A 실행 스크립트: GPT-4o-mini 페르소나 예측 (Naive, Condition A).

사용법:
    # 전체 실행 (50 personas × 100 cold items × 17 weeks)
    python scripts/run_track_a.py

    # 드라이런 (API 호출 없이 파이프라인 검증)
    python scripts/run_track_a.py --dry-run

    # 중단 후 재시작 (체크포인트 자동 감지)
    python scripts/run_track_a.py --resume

    # 아이템 수 제한 (빠른 테스트)
    python scripts/run_track_a.py --n-items 10 --dry-run

예측 스케일 보정:
    Track A 원본 예측 = 50명의 개인 구매량 합산 (50-persona scale)
    post-hoc 보정: alpha = mean(baseline_pred) / mean(track_a_raw)
    track_a_calibrated = track_a_raw × alpha  (매장 수준)
    베이스라인 예측 비교에도 동일 기준 적용

출력:
    experiments/exp004_track_a_naive/
        predictions/
            track_a_raw.csv          ← 50-persona 합산 원본
            track_a_calibrated.csv   ← 매장 수준 보정 후
        metrics/
            evaluation_results.json
        checkpoints/
            prediction_checkpoint.json  ← 재시작용
        run_metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation.metrics import evaluate
from src.models.baselines import GlobalCategoryAverage
from src.models.forecasting.persona_predictor import PersonaPredictor
from src.models.forecasting.prompt_builder import PromptBuilder
from src.models.persona.schema import Persona


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_track_a_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track A: GPT-4o-mini 페르소나 예측 실행"
    )
    parser.add_argument("--config", default=None, help="config.yaml 경로")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="체크포인트에서 재시작 (자동으로 checkpoint 파일 탐색)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM API 호출 없이 파이프라인 검증 (모든 예측 = 0)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="처리할 cold 아이템 수 제한 (테스트용)",
    )
    parser.add_argument(
        "--n-personas",
        type=int,
        default=None,
        help="사용할 페르소나 수 제한 (테스트용)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_personas(personas_dir: Path) -> list[Persona]:
    """personas_dir의 CA_1_P*.json 파일을 모두 로드한다."""
    files = sorted(personas_dir.glob("CA_1_P*.json"))
    personas: list[Persona] = []
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            personas.append(Persona.from_dict(data))
        except Exception as e:
            logging.getLogger(__name__).warning("페르소나 로드 실패 %s: %s", fp.name, e)
    return personas


def _safe_round(value: float | None, ndigits: int = 6) -> float | None:
    """NaN / None을 None(JSON null)으로 변환하고, 정상 float은 반올림한다."""
    import math
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(value, ndigits)


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    # Config 로드
    config = load_config(ROOT / args.config if args.config else None)
    ta_cfg = config.experiment.track_a
    out_dir = ROOT / ta_cfg.output_dir
    log_dir = ROOT / config.paths.log_dir

    setup_logging(log_dir, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("=== Track A 시작 ===")
    logger.info("출력 디렉토리: %s", out_dir)

    # 데이터 로드
    cold_start_dir = ROOT / config.paths.cold_start_dir
    logger.info("cold-start 데이터 로드 중...")
    cold_test = pd.read_csv(cold_start_dir / "cold_test.csv", parse_dates=["date"])
    warm_train = pd.read_csv(cold_start_dir / "warm_train.csv", parse_dates=["date"])

    # cold 아이템 수 제한 (테스트용)
    if args.n_items is not None:
        cold_ids_limited = cold_test["item_id"].unique()[: args.n_items]
        cold_test = cold_test[cold_test["item_id"].isin(cold_ids_limited)]
        logger.info("아이템 제한: %d개", args.n_items)

    # 캘린더 & 가격 로드
    logger.info("캘린더 / 판매가 로드 중...")
    calendar = pd.read_csv(ROOT / config.paths.calendar)
    sell_prices = pd.read_csv(ROOT / config.paths.sell_prices)

    # 페르소나 로드
    personas_dir = ROOT / config.paths.personas_dir
    personas = load_personas(personas_dir)
    if args.n_personas is not None:
        personas = personas[: args.n_personas]
    if not personas:
        logger.error("페르소나를 찾을 수 없습니다: %s", personas_dir)
        sys.exit(1)
    logger.info("페르소나 로드: %d개", len(personas))

    # 체크포인트 경로
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "prediction_checkpoint.json"
    if not args.resume and checkpoint_path.exists():
        logger.warning(
            "체크포인트가 존재하지만 --resume 플래그가 없습니다. "
            "기존 체크포인트를 사용합니다. 처음부터 재실행하려면 체크포인트 파일을 삭제하세요: %s",
            checkpoint_path,
        )

    # PromptBuilder 초기화
    pb = PromptBuilder(
        calendar_df=calendar,
        sell_prices_df=sell_prices,
        store_id=str(config.experiment.cold_start.target_store),
        condition=str(ta_cfg.condition),
    )

    # PersonaPredictor 초기화
    predictor = PersonaPredictor(
        config=config,
        personas=personas,
        prompt_builder=pb,
        checkpoint_path=checkpoint_path,
        dry_run=args.dry_run,
    )
    predictor.fit(warm_train)

    # 예측 실행
    logger.info("예측 실행 중 (이 작업에 오랜 시간이 소요될 수 있습니다)...")
    pred_raw = predictor.predict(cold_test)

    # ---------- 출력 저장 ----------
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_raw.to_csv(pred_dir / "track_a_raw.csv", index=False)
    logger.info("원본 예측 저장: %s", pred_dir / "track_a_raw.csv")

    # ---------- 스케일 보정 (post-hoc) ----------
    scale_method = str(ta_cfg.scale_align_method)
    alpha = 1.0
    if scale_method == "post_hoc_mean":
        logger.info("스케일 보정: 베이스라인(GlobalCategoryAverage) 예측 계산 중...")
        baseline_model = GlobalCategoryAverage()
        baseline_model.fit(warm_train)
        baseline_pred = baseline_model.predict(cold_test)
        # cold_test와 동일한 item_id 필터
        baseline_pred = baseline_pred[baseline_pred["item_id"].isin(cold_test["item_id"].unique())]
        alpha = predictor.compute_scale_factor(pred_raw, baseline_pred)
    else:
        logger.info("스케일 보정 없음 (scale_align_method=%s)", scale_method)

    pred_calibrated = predictor.apply_scale_factor(pred_raw, alpha)
    pred_calibrated.to_csv(pred_dir / "track_a_calibrated.csv", index=False)
    logger.info(
        "보정 예측 저장 (alpha=%.2f): %s",
        alpha,
        pred_dir / "track_a_calibrated.csv",
    )

    # ---------- 평가 ----------
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # (1) raw 평가 (50-persona scale)
    metrics_raw = evaluate(
        cold_test=cold_test,
        pred=pred_raw,
        warm_train=warm_train,
        model_name="track_a_raw",
    )

    # (2) 보정 후 평가 (매장 수준)
    metrics_cal = evaluate(
        cold_test=cold_test,
        pred=pred_calibrated,
        warm_train=warm_train,
        model_name="track_a_calibrated",
    )

    results = {
        "run_at": datetime.now().isoformat(),
        "n_personas": len(personas),
        "n_cold_items": cold_test["item_id"].nunique(),
        "scale_factor_alpha": alpha,
        "scale_align_method": scale_method,
        "dry_run": args.dry_run,
        "metrics_raw": metrics_raw,
        "metrics_calibrated": metrics_cal,
    }
    (metrics_dir / "evaluation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---------- 논문용 투명성 메타데이터 ----------
    # alpha 보정의 의미, 전후 MAE를 단일 파일에 기록해 논문 Appendix에 직접 인용 가능하게 함
    transparency_meta = {
        "run_at": datetime.now().isoformat(),
        "experiment": "Track A — Naive LLM Persona Prediction (Condition A: Structured Only)",
        "n_personas": len(personas),
        "n_cold_items": cold_test["item_id"].nunique(),
        "prediction_model": str(config.experiment.llm.prediction_model),
        "scale_calibration": {
            "method": scale_method,
            "description": (
                "post-hoc mean ratio: alpha = mean(GlobalCategoryAverage_pred) / mean(track_a_raw_pred). "
                "track_a_calibrated = track_a_raw * alpha  →  store-level scale."
            ),
            "alpha": round(alpha, 6),
            "alpha_interpretation": (
                f"50 personas represent ~{alpha:.1f}x fewer purchase events than "
                "the store-level baseline; calibration scales predictions up."
            ),
        },
        "mae_comparison": {
            "raw_50persona_scale": round(metrics_raw["mae"], 6),
            "calibrated_store_scale": round(metrics_cal["mae"], 6),
            "baseline_reference_exp002": {
                "global_category_average": 1.5700,
                "similar_item_average": 1.6900,
                "store_category_average": 1.5900,
            },
        },
        "direction_accuracy_comparison": {
            "track_a_raw": round(metrics_raw["direction_accuracy"], 6),
            "track_a_calibrated": round(metrics_cal["direction_accuracy"], 6),
            "note": "DirAcc is scale-invariant; raw == calibrated values should match.",
            "baseline_reference_exp002": 0.257,
        },
        "wrmsse_comparison": {
            "track_a_calibrated": round(metrics_cal["wrmsse"], 6),
            "baseline_reference_exp002": {
                "global_category_average": 2.9600,
                "similar_item_average": 2.9900,
                "store_category_average": 2.9600,
            },
        },
        "by_category": {
            cat: {
                "mae_raw": _safe_round(metrics_raw["by_category"].get(cat, {}).get("mae")),
                "mae_calibrated": _safe_round(metrics_cal["by_category"].get(cat, {}).get("mae")),
                "direction_accuracy": _safe_round(metrics_cal["by_category"].get(cat, {}).get("direction_accuracy")),
            }
            for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]
        },
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(transparency_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("논문용 메타데이터 저장: %s", out_dir / "run_metadata.json")

    # ---------- 요약 출력 ----------
    logger.info("=" * 60)
    logger.info("=== Track A 결과 요약 ===")
    logger.info("보정 전 (50-persona scale):")
    logger.info(
        "  MAE=%.4f  RMSE=%.4f  WRMSSE=%.4f  DirAcc=%.4f",
        metrics_raw["mae"], metrics_raw["rmse"],
        metrics_raw["wrmsse"], metrics_raw["direction_accuracy"],
    )
    logger.info("보정 후 (매장 수준, alpha=%.2f):", alpha)
    logger.info(
        "  MAE=%.4f  RMSE=%.4f  WRMSSE=%.4f  DirAcc=%.4f",
        metrics_cal["mae"], metrics_cal["rmse"],
        metrics_cal["wrmsse"], metrics_cal["direction_accuracy"],
    )
    logger.info("베이스라인 참고 (exp002): MAE≈1.57~1.69 / WRMSSE≈2.96~2.99 / DirAcc≈0.257")
    logger.info("전체 결과 저장: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
