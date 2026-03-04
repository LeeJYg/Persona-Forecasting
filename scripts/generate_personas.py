"""합성 페르소나 생성 스크립트.

사용법:
    # 전체 생성 (config 기준 50개)
    python scripts/generate_personas.py

    # 테스트 모드 (n개만 생성)
    python scripts/generate_personas.py --n 3

    # 사용자 지정 config
    python scripts/generate_personas.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (scripts/ 에서 직접 실행 시)
_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_ROOT))

from src.config import load_config
from src.llm.client import LLMClient
from src.models.persona.generator import LLMPersonaGenerator


def _load_dotenv(env_path: Path) -> None:
    """프로젝트 루트의 .env 파일을 os.environ에 로드한다.

    python-dotenv 없이 동작하도록 직접 구현.
    이미 설정된 환경 변수는 덮어쓰지 않는다.
    """
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


def _setup_logging(level: str = "INFO") -> None:
    """로깅 설정."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="합성 페르소나 생성")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="config 파일 경로 (기본: configs/config.yaml)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="생성할 페르소나 수 (기본: config의 n_personas). 테스트 시 사용.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="출력 디렉토리 경로 덮어쓰기 (기본: config의 personas_dir)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # .env 로드 (API 키)
    _load_dotenv(_ROOT / ".env")

    # config 로드
    config_path = _ROOT / args.config
    config = load_config(config_path)
    logger.info("Config 로드 완료: %s", config_path)

    # 출력 디렉토리 결정
    output_dir = Path(args.output_dir) if args.output_dir else _ROOT / config.paths.personas_dir
    logger.info("출력 디렉토리: %s", output_dir)

    # LLM 클라이언트 초기화
    llm_client = LLMClient(config)

    # 생성기 초기화 및 실행
    generator = LLMPersonaGenerator(config, llm_client)
    personas = generator.generate(n_personas=args.n)

    if not personas:
        logger.error("생성된 페르소나 없음. 종료.")
        sys.exit(1)

    # 저장
    generator.save(personas, output_dir)

    # 결과 요약 출력
    logger.info("=" * 50)
    logger.info("생성 완료: %d개 페르소나", len(personas))
    logger.info("저장 위치: %s", output_dir.resolve())

    # 테스트 모드(--n 지정)면 첫 번째 페르소나 JSON을 콘솔에 출력
    if args.n is not None and personas:
        print("\n=== 첫 번째 페르소나 (미리보기) ===")
        print(json.dumps(personas[0].to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
