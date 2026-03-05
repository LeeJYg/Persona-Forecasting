"""Competitor 4: Category Average + Seasonal Pattern (Enhanced Naive).

Warm items의 (cat × dept, ISO week) 평균 주간 판매 프로필을 cold item에 적용.
GlobalCategoryAverage baseline에 주간 계절성 패턴을 추가한 것.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines.base import ForecastModel

logger = logging.getLogger(__name__)


class SeasonalPattern(ForecastModel):
    """Category × Dept 수준의 ISO-week 계절 프로필 기반 예측.

    fit(): warm_train_weekly에서 (cat_id, dept_id, iso_week)별 평균 주간 판매량 계산.
    predict(): cold_test_weekly의 각 행에 해당 프로필 값 배정.
    """

    def __init__(self) -> None:
        self._profile: pd.Series | None = None  # (cat_id, dept_id, iso_week) → mean weekly sales

    @property
    def name(self) -> str:
        return "seasonal_pattern"

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "SeasonalPattern":
        """warm_train_weekly에서 계절 프로필 학습.

        Args:
            warm_train: 주간 집계된 warm items 데이터.
                        컬럼: item_id, cat_id, dept_id, iso_week, sales.
        """
        required = {"cat_id", "dept_id", "iso_week", "sales"}
        missing = required - set(warm_train.columns)
        if missing:
            raise ValueError(f"[{self.name}] warm_train missing columns: {missing}")

        self._profile = (
            warm_train.groupby(["cat_id", "dept_id", "iso_week"])["sales"]
            .mean()
        )
        logger.info(
            "[%s] 계절 프로필 학습 완료: %d (cat×dept×week) 그룹",
            self.name, len(self._profile),
        )
        return self

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """cold_test_weekly의 각 행에 계절 프로필 값 배정."""
        if self._profile is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출해야 합니다.")

        pred = cold_test[["item_id", "store_id", "cat_id", "dept_id", "iso_year", "iso_week", "date"]].copy()
        pred["pred_sales"] = pred.apply(
            lambda r: self._profile.get((r["cat_id"], r["dept_id"], r["iso_week"]), np.nan),
            axis=1,
        )

        # fallback: cat 수준 평균 (dept 없는 주)
        cat_week_mean = (
            cold_test.merge(
                self._profile.reset_index().rename(columns={"sales": "_prof"}),
                on=["cat_id", "dept_id", "iso_week"],
                how="left",
            )["_prof"]
        )
        cat_fallback = (
            self._profile
            .groupby(["cat_id", "iso_week"])
            .mean()
        )
        mask = pred["pred_sales"].isna()
        if mask.any():
            pred.loc[mask, "pred_sales"] = pred[mask].apply(
                lambda r: cat_fallback.get((r["cat_id"], r["iso_week"]), 0.0),
                axis=1,
            )
            logger.warning("[%s] %d 행에 cat_week fallback 적용", self.name, mask.sum())

        pred["pred_sales"] = pred["pred_sales"].fillna(0.0).clip(lower=0.0)
        self._validate_predict_output(pred)
        return pred
