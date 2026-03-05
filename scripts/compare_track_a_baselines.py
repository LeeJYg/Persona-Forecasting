"""Track A vs 베이스라인 비교 평가 스크립트.

Track A (raw / calibrated) 예측과 exp002 베이스라인 3종을 동일 지표로 비교한다.
run_track_a.py 완료 후 실행한다.

사용법:
    python scripts/compare_track_a_baselines.py

    # 로그 파일에서 실행 시간 자동 파싱 (없으면 생략)
    python scripts/compare_track_a_baselines.py --log-file logs/track_a_full_run.log

출력 (experiments/exp004_track_a_naive/results/):
    comparison_table.csv      ← 5 모델 × 4 지표 요약
    comparison_table.json     ← 동일 내용 JSON
    by_category_table.csv     ← 카테고리별 세분화
    run_metadata.json 업데이트 ← alpha, API 호출 수, 비용 추정, 실행 시간 추가
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation.metrics import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# gpt-4o-mini 가격 (2024년 기준, USD/1M tokens)
_PRICE_INPUT_PER_1M = 0.150
_PRICE_OUTPUT_PER_1M = 0.600
# 콜당 평균 추정 토큰 (10 items × 1 persona × 1 week prompt)
_EST_INPUT_TOKENS_PER_CALL = 700
_EST_OUTPUT_TOKENS_PER_CALL = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track A vs 베이스라인 비교 평가")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-file", default=None, help="Track A 실행 로그 경로 (런타임 파싱용)")
    return parser.parse_args()


def _parse_runtime_from_log(log_path: Path) -> tuple[str | None, str | None, float | None]:
    """로그 파일에서 시작/종료 시간과 총 실행 시간(초)을 파싱한다."""
    if not log_path.exists():
        return None, None, None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    # 로그 형식: "HH:MM:SS [INFO] ..."
    timestamps: list[str] = []
    for line in lines:
        m = re.match(r"^(\d{2}:\d{2}:\d{2})", line)
        if m:
            timestamps.append(m.group(1))
    if not timestamps:
        return None, None, None
    start_ts, end_ts = timestamps[0], timestamps[-1]
    try:
        fmt = "%H:%M:%S"
        from datetime import datetime as dt
        t0 = dt.strptime(start_ts, fmt)
        t1 = dt.strptime(end_ts, fmt)
        elapsed = (t1 - t0).total_seconds()
        # 자정을 넘긴 경우 (end < start): 하루를 더함
        if elapsed < 0:
            elapsed += 24 * 3600
    except Exception:
        elapsed = None
    return start_ts, end_ts, elapsed


def _load_checkpoint_completed(checkpoint_path: Path) -> dict:
    """체크포인트에서 completed 페르소나 dict를 반환한다. 구버전/신버전 모두 지원."""
    if not checkpoint_path.exists():
        return {}
    try:
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if "completed" in raw:
            return raw.get("completed", {})
        # 구버전 포맷: {pid: {item_id: [...]}}
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    except Exception as e:
        logger.warning("체크포인트 로드 실패: %s", e)
        return {}


def _count_api_calls_from_checkpoint(checkpoint_path: Path) -> int:
    """체크포인트에서 완료된 페르소나 수 × item_batches × weeks를 추산한다."""
    completed = _load_checkpoint_completed(checkpoint_path)
    if not completed:
        return 0
    total = 0
    for _, item_preds in completed.items():
        n_items = len(item_preds)
        n_weeks = max((len(v) for v in item_preds.values()), default=0)
        # batch_size=10 가정 (config와 일치)
        n_batches = (n_items + 9) // 10
        total += n_batches * n_weeks
    return total


def _estimate_cost(n_calls: int) -> dict[str, float]:
    """gpt-4o-mini 호출 비용 추정 (USD)."""
    input_cost = n_calls * _EST_INPUT_TOKENS_PER_CALL / 1_000_000 * _PRICE_INPUT_PER_1M
    output_cost = n_calls * _EST_OUTPUT_TOKENS_PER_CALL / 1_000_000 * _PRICE_OUTPUT_PER_1M
    return {
        "estimated_input_tokens": n_calls * _EST_INPUT_TOKENS_PER_CALL,
        "estimated_output_tokens": n_calls * _EST_OUTPUT_TOKENS_PER_CALL,
        "estimated_input_cost_usd": round(input_cost, 4),
        "estimated_output_cost_usd": round(output_cost, 4),
        "estimated_total_cost_usd": round(input_cost + output_cost, 4),
        "note": (
            f"추정값: 콜당 평균 input {_EST_INPUT_TOKENS_PER_CALL} / "
            f"output {_EST_OUTPUT_TOKENS_PER_CALL} tokens 가정. "
            "실제 비용은 OpenAI 대시보드에서 확인 요망."
        ),
        "pricing_reference": (
            f"gpt-4o-mini: input ${_PRICE_INPUT_PER_1M}/1M, "
            f"output ${_PRICE_OUTPUT_PER_1M}/1M tokens"
        ),
    }


def _build_row(result: dict) -> dict:
    """evaluate() 결과를 테이블 행 dict로 변환."""
    return {
        "model": result["model"],
        "MAE": round(result["mae"], 4),
        "RMSE": round(result["rmse"], 4),
        "WRMSSE": round(result["wrmsse"], 4),
        "DirAcc": round(result["direction_accuracy"], 4),
        "n_items": result["n_items"],
    }


def _build_by_cat_rows(result: dict) -> list[dict]:
    """카테고리별 세분화 행 생성."""
    rows = []
    for cat, metrics in result.get("by_category", {}).items():
        rows.append({
            "model": result["model"],
            "category": cat,
            "MAE": round(metrics.get("mae", float("nan")), 4),
            "RMSE": round(metrics.get("rmse", float("nan")), 4),
            "WRMSSE": round(metrics.get("wrmsse", float("nan")), 4),
            "DirAcc": round(metrics.get("direction_accuracy", float("nan")), 4),
        })
    return rows


def _print_comparison_table(df: pd.DataFrame) -> None:
    """터미널에 비교 테이블 출력."""
    logger.info("\n%s\n=== Track A vs 베이스라인 비교 ===\n%s\n%s\n%s",
                "=" * 70,
                df.to_string(index=False),
                "=" * 70,
                "(낮을수록 좋음: MAE / RMSE / WRMSSE | 높을수록 좋음: DirAcc)")


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config if args.config else None)

    out_dir = ROOT / config.experiment.track_a.output_dir
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cold_start_dir = ROOT / config.paths.cold_start_dir
    exp002_dir = ROOT / config.experiment.baselines.output_dir

    # ── 데이터 로드 ─────────────────────────────────────────────────
    logger.info("데이터 로드 중...")
    cold_test = pd.read_csv(cold_start_dir / "cold_test.csv", parse_dates=["date"])
    warm_train = pd.read_csv(cold_start_dir / "warm_train.csv", parse_dates=["date"])

    # ── Track A 예측 로드 ────────────────────────────────────────────
    pred_dir = out_dir / "predictions"
    raw_path = pred_dir / "track_a_raw.csv"
    cal_path = pred_dir / "track_a_calibrated.csv"

    if not raw_path.exists():
        logger.error("Track A raw 예측 파일 없음: %s", raw_path)
        logger.error("run_track_a.py 완료 후 이 스크립트를 실행하세요.")
        sys.exit(1)

    pred_raw = pd.read_csv(raw_path, parse_dates=["date"])
    pred_cal = pd.read_csv(cal_path, parse_dates=["date"])
    logger.info("Track A 예측 로드: raw=%d rows, calibrated=%d rows", len(pred_raw), len(pred_cal))

    # ── 베이스라인 예측 로드 ──────────────────────────────────────────
    baseline_files = {
        "global_category_average": exp002_dir / "predictions" / "global_category_average.csv",
        "similar_item_average": exp002_dir / "predictions" / "similar_item_average.csv",
        "store_category_average": exp002_dir / "predictions" / "store_category_average.csv",
    }
    baseline_preds: dict[str, pd.DataFrame] = {}
    for model_name, path in baseline_files.items():
        if not path.exists():
            logger.error("베이스라인 예측 파일 없음: %s", path)
            sys.exit(1)
        baseline_preds[model_name] = pd.read_csv(path, parse_dates=["date"])
    logger.info("베이스라인 예측 로드: %d 모델", len(baseline_preds))

    # ── cold_test의 item 목록에만 베이스라인 필터 ────────────────────
    cold_item_ids = cold_test["item_id"].unique()
    for k in baseline_preds:
        baseline_preds[k] = baseline_preds[k][
            baseline_preds[k]["item_id"].isin(cold_item_ids)
        ]

    # ── 평가 실행 ─────────────────────────────────────────────────────
    logger.info("평가 지표 계산 중...")
    all_results: list[dict] = []

    # Track A raw (50-persona scale)
    res_raw = evaluate(cold_test, pred_raw, warm_train, model_name="track_a_raw")
    all_results.append(res_raw)

    # Track A calibrated (store scale)
    res_cal = evaluate(cold_test, pred_cal, warm_train, model_name="track_a_calibrated")
    all_results.append(res_cal)

    # 베이스라인 3종
    for model_name, pred in baseline_preds.items():
        res = evaluate(cold_test, pred, warm_train, model_name=model_name)
        all_results.append(res)

    # ── 비교 테이블 구성 ──────────────────────────────────────────────
    rows = [_build_row(r) for r in all_results]
    df_main = pd.DataFrame(rows)

    # 모델 순서 정렬 (Track A 먼저, 베이스라인 다음)
    model_order = [
        "track_a_raw",
        "track_a_calibrated",
        "global_category_average",
        "similar_item_average",
        "store_category_average",
    ]
    df_main["_order"] = df_main["model"].map({m: i for i, m in enumerate(model_order)})
    df_main = df_main.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    _print_comparison_table(df_main)

    # ── 카테고리별 비교 테이블 ─────────────────────────────────────────
    cat_rows: list[dict] = []
    for r in all_results:
        cat_rows.extend(_build_by_cat_rows(r))
    df_cat = pd.DataFrame(cat_rows)
    if not df_cat.empty:
        df_cat["_order"] = df_cat["model"].map({m: i for i, m in enumerate(model_order)})
        df_cat = df_cat.sort_values(["_order", "category"]).drop(columns=["_order"]).reset_index(drop=True)

    # ── 결과 저장 ─────────────────────────────────────────────────────
    df_main.to_csv(results_dir / "comparison_table.csv", index=False)
    df_cat.to_csv(results_dir / "by_category_table.csv", index=False)
    logger.info("비교 테이블 저장: %s", results_dir)

    # JSON 저장 (논문용)
    comparison_json = {
        "generated_at": datetime.now().isoformat(),
        "models": rows,
        "by_category": [
            {k: (v if not (isinstance(v, float) and __import__("math").isnan(v)) else None)
             for k, v in r.items()}
            for r in cat_rows
        ],
    }
    (results_dir / "comparison_table.json").write_text(
        json.dumps(comparison_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── run_metadata.json 업데이트 ─────────────────────────────────────
    metadata_path = out_dir / "run_metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # API 호출 수 & 비용 추정
    checkpoint_path = out_dir / "checkpoints" / "prediction_checkpoint.json"
    n_api_calls = _count_api_calls_from_checkpoint(checkpoint_path)
    if n_api_calls == 0:
        # 체크포인트 없으면 이론값 계산 (50 personas × 10 batches × 17 weeks)
        n_items_cold = cold_test["item_id"].nunique()
        n_personas = len(json.loads(checkpoint_path.read_text())
                         if checkpoint_path.exists() else {})
        n_api_calls = 50 * ((n_items_cold + 9) // 10) * 17
    cost_info = _estimate_cost(n_api_calls)

    # 런타임 파싱
    log_path = ROOT / (args.log_file or "logs/track_a_full_run.log")
    start_ts, end_ts, elapsed_sec = _parse_runtime_from_log(log_path)

    # 메타데이터에 추가 필드
    metadata["comparison_summary"] = {
        "generated_at": datetime.now().isoformat(),
        "track_a_raw": {
            "mae": round(res_raw["mae"], 6),
            "rmse": round(res_raw["rmse"], 6),
            "wrmsse": round(res_raw["wrmsse"], 6),
            "direction_accuracy": round(res_raw["direction_accuracy"], 6),
        },
        "track_a_calibrated": {
            "mae": round(res_cal["mae"], 6),
            "rmse": round(res_cal["rmse"], 6),
            "wrmsse": round(res_cal["wrmsse"], 6),
            "direction_accuracy": round(res_cal["direction_accuracy"], 6),
        },
        "best_baseline_mae": round(
            min(r["mae"] for r in all_results if "baseline" not in r["model"] and "track" not in r["model"]),
            6,
        ),
        "beats_best_baseline": bool(res_cal["mae"] < min(
            r["mae"] for r in all_results if "track" not in r["model"]
        )),
    }
    metadata["api_usage"] = {
        "total_api_calls": n_api_calls,
        "n_completed_personas": len(_load_checkpoint_completed(checkpoint_path)),
        **cost_info,
    }
    if elapsed_sec is not None:
        metadata["runtime"] = {
            "start_time": start_ts,
            "end_time": end_ts,
            "elapsed_seconds": round(elapsed_sec, 1),
            "elapsed_human": f"{int(elapsed_sec // 3600)}h {int((elapsed_sec % 3600) // 60)}m {int(elapsed_sec % 60)}s",
        }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("run_metadata.json 업데이트 완료: %s", metadata_path)

    # ── 최종 요약 출력 ────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("=== 비교 결과 요약 ===")
    logger.info("Track A Raw (50-persona):  MAE=%.4f / WRMSSE=%.4f / DirAcc=%.4f",
                res_raw["mae"], res_raw["wrmsse"], res_raw["direction_accuracy"])
    logger.info("Track A Calibrated:        MAE=%.4f / WRMSSE=%.4f / DirAcc=%.4f",
                res_cal["mae"], res_cal["wrmsse"], res_cal["direction_accuracy"])
    logger.info("Best Baseline (MAE):       MAE=%.4f",
                min(r["mae"] for r in all_results if "track" not in r["model"]))
    logger.info("Track A beats baseline:    %s",
                "YES ✓" if metadata["comparison_summary"]["beats_best_baseline"] else "NO ✗")
    if n_api_calls:
        logger.info("총 API 호출: %d 회 / 추정 비용: $%.2f USD",
                    n_api_calls, cost_info["estimated_total_cost_usd"])
    logger.info("결과 저장 위치: %s", results_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
