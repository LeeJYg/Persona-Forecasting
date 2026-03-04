"""Config loader: configs/config.yaml을 읽어 DotDict로 반환."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """dot 표기법으로 접근 가능한 dict."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(key)
        return DotDict(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(config_path: str | Path | None = None) -> DotDict:
    """
    YAML config 파일을 로드한다.

    Args:
        config_path: config.yaml 경로.
                     None이면 프로젝트 루트의 configs/config.yaml을 자동 탐색.

    Returns:
        DotDict: dot 표기법으로 접근 가능한 설정 객체.
    """
    if config_path is None:
        # __file__ = src/config/__init__.py → 두 단계 위가 프로젝트 루트
        root = Path(__file__).parent.parent.parent.resolve()
        config_path = root / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return DotDict(raw)
