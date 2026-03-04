"""M5 전처리 데이터 로더 (config-driven)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DotDict, load_config

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"id", "item_id", "store_id", "cat_id", "dept_id",
                  "state_id", "date", "sales"}


class M5DataLoader:
    """
    전처리된 M5 CSV 파일(m5_train.csv, m5_test.csv)을 로드한다.

    모든 경로는 config를 통해 관리하며 코드 내 하드코딩 금지.

    Args:
        config: load_config()로 반환된 DotDict.
                None이면 default config.yaml을 자동 탐색.
        root: 프로젝트 루트 경로.
              None이면 이 파일 기준으로 자동 계산.
    """

    def __init__(
        self,
        config: DotDict | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.config = config or load_config()
        self.root = Path(root) if root else Path(__file__).parent.parent.parent.resolve()
        self._train: pd.DataFrame | None = None
        self._test: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_train(self) -> pd.DataFrame:
        """
        m5_train.csv를 로드해 반환한다.

        Returns:
            pd.DataFrame: 필수 컬럼을 포함한 학습 데이터프레임.
        """
        if self._train is None:
            path = self.root / self.config.paths.processed_dir / "m5_train.csv"
            logger.info("Loading train data from %s", path)
            self._train = self._read_and_validate(path)
            logger.info("Train loaded: %s rows", f"{len(self._train):,}")
        return self._train

    def load_test(self) -> pd.DataFrame:
        """
        m5_test.csv를 로드해 반환한다.

        Returns:
            pd.DataFrame: 필수 컬럼을 포함한 테스트 데이터프레임.
        """
        if self._test is None:
            path = self.root / self.config.paths.processed_dir / "m5_test.csv"
            logger.info("Loading test data from %s", path)
            self._test = self._read_and_validate(path)
            logger.info("Test loaded: %s rows", f"{len(self._test):,}")
        return self._test

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_and_validate(self, path: Path) -> pd.DataFrame:
        """CSV를 읽고 필수 컬럼 존재 여부를 검증한다."""
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "Run the M5 base preprocessing first."
            )
        df = pd.read_csv(path, parse_dates=["date"])
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df
