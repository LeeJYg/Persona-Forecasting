"""
Cold-start 전처리 실행 스크립트 (thin runner).

사용법:
    python -m scripts.preprocess_cold_start
    python -m scripts.preprocess_cold_start --config configs/config.yaml

이 스크립트는 src.data 모듈을 호출하는 얇은 진입점입니다.
핵심 로직은 src/data/loader.py, src/data/cold_start.py에 위치합니다.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (스크립트를 직접 실행하는 경우 대비)
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.cold_start import ColdStartSampler
from src.data.loader import M5DataLoader


def setup_logging() -> None:
    """루트 로거를 INFO 레벨로 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M5 cold-start 아이템 샘플링 및 warm/cold 데이터 분리"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="config.yaml 경로 (기본값: configs/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)

    # 1. 데이터 로드
    loader = M5DataLoader(config=config, root=ROOT)
    train = loader.load_train()
    test = loader.load_test()

    # 2. 아이템 통계 계산 (target_store 기준)
    sampler = ColdStartSampler(config=config, root=ROOT)
    item_stats = sampler.compute_item_stats(train)

    # 3. cold-start id 샘플링 (store-item 쌍)
    cold_ids = sampler.sample_cold_ids(item_stats)

    # 4. warm/cold 분리
    warm_train, warm_test, cold_train, cold_test = sampler.split(
        train, test, cold_ids
    )

    # 5. 저장
    out_dir = sampler.save(
        warm_train, warm_test, cold_train, cold_test, cold_ids, item_stats
    )

    # 6. 요약 출력
    sampler.summary(cold_ids, item_stats)

    logging.getLogger(__name__).info("Done. Output: %s", out_dir)


if __name__ == "__main__":
    main()
