"""exp011: DeepAR Electricity 재현 검증 (GluonTS 0.16.2).

Salinas et al. (2020) 논문의 electricity 벤치마크를 GluonTS 공식 구현으로 재현.
논문 기준값 (Table 3): ND=0.038, NRMSE=0.095 (±10% 허용).

환경: aias-x86, GluonTS 0.16.2 (CPU 전용)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXP_DIR = ROOT / "experiments" / "exp011_deepar_verification"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# 논문 기준값 (Salinas et al. 2020, Table 3)
TARGET_ND = 0.038
TARGET_NRMSE = 0.095
TOLERANCE = 0.10  # ±10%


def main() -> None:
    # GluonTS import (aias-x86에서만 사용 가능)
    try:
        from gluonts.dataset.repository.datasets import get_dataset
        from gluonts.evaluation import Evaluator, backtest_metrics
        from gluonts.mx import DeepAREstimator
        from gluonts.mx.trainer import Trainer
    except ImportError as e:
        logger.error("GluonTS / MXNet import 실패: %s", e)
        logger.error("환경 확인: conda activate deepar-verification")
        sys.exit(1)

    # 1. 데이터셋 로드
    logger.info("electricity 데이터셋 로드 중...")
    dataset = get_dataset("electricity", regenerate=False)
    logger.info(
        "train size: %d, test size: %d",
        len(list(dataset.train)),
        len(list(dataset.test)),
    )

    # 2. DeepAR 모델 정의 (Salinas et al. 2020 Table 4)
    logger.info("DeepAR 모델 초기화 중...")
    estimator = DeepAREstimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        num_layers=2,
        num_cells=40,
        dropout_rate=0.1,
        trainer=Trainer(epochs=100, batch_size=32),
    )

    # 3. 학습
    logger.info("학습 시작 (epochs=100)...")
    predictor = estimator.train(dataset.train)

    # 4. 평가
    logger.info("평가 중 (num_samples=100)...")
    evaluator = Evaluator(num_workers=0)
    agg_metrics, item_metrics = backtest_metrics(
        test_dataset=dataset.test,
        predictor=predictor,
        evaluator=evaluator,
    )

    nd = agg_metrics["ND"]
    nrmse = agg_metrics["NRMSE"]
    crps = agg_metrics.get("mean_wQuantileLoss", float("nan"))

    logger.info("\n=== exp011: DeepAR Electricity 재현 결과 ===")
    logger.info("  ND:    %.4f  (논문: %.3f, 허용 범위: %.3f~%.3f)",
                nd, TARGET_ND,
                TARGET_ND * (1 - TOLERANCE), TARGET_ND * (1 + TOLERANCE))
    logger.info("  NRMSE: %.4f  (논문: %.3f, 허용 범위: %.3f~%.3f)",
                nrmse, TARGET_NRMSE,
                TARGET_NRMSE * (1 - TOLERANCE), TARGET_NRMSE * (1 + TOLERANCE))
    logger.info("  CRPS:  %.4f", crps)

    nd_pass = TARGET_ND * (1 - TOLERANCE) <= nd <= TARGET_ND * (1 + TOLERANCE)
    nrmse_pass = TARGET_NRMSE * (1 - TOLERANCE) <= nrmse <= TARGET_NRMSE * (1 + TOLERANCE)
    verdict = "PASS" if (nd_pass and nrmse_pass) else "FAIL"
    logger.info("  판정: %s (ND=%s, NRMSE=%s)",
                verdict,
                "✓" if nd_pass else "✗",
                "✓" if nrmse_pass else "✗")

    # 5. 저장
    result = {
        "model": "DeepAR",
        "dataset": "electricity",
        "nd": nd,
        "nrmse": nrmse,
        "crps": crps,
        "nd_target": TARGET_ND,
        "nrmse_target": TARGET_NRMSE,
        "nd_pass": nd_pass,
        "nrmse_pass": nrmse_pass,
        "verdict": verdict,
    }
    out_path = EXP_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
