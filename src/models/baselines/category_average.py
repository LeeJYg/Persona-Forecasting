"""Baseline 1: Global Category Average.

예측 값 = warm_train 전체에서 동일 cat_id 아이템들의 평균 일별 판매량.
매장 구분 없이 전체 매장의 판매 이력을 사용한다 (가장 단순한 베이스라인).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.models.baselines.base import ForecastModel

logger = logging.getLogger(__name__)


class GlobalCategoryAverage(ForecastModel):
    """
    카테고리 전체 평균으로 cold-start 아이템 판매량을 예측한다.

    학습: warm_train에서 cat_id별 평균 일별 판매량을 계산.
    예측: cold 아이템의 cat_id에 해당하는 평균값을 모든 테스트 날짜에 적용.

    Notes
    -----
    features 인수는 이 모델에서 사용하지 않는다.
    LightGBM 등 상위 모델과 동일 인터페이스 유지를 위해 시그니처에 포함.
    """

    def __init__(self) -> None:
        self._cat_means: pd.Series | None = None  # {cat_id: mean_sales}

    @property
    def name(self) -> str:
        return "global_category_average"

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "GlobalCategoryAverage":
        """
        cat_id별 평균 일별 판매량을 계산한다.

        Args:
            warm_train: cold 아이템이 제외된 학습 데이터.
            features: 미사용 (인터페이스 호환용).
        """
        logger.info("[%s] Fitting on %d rows ...", self.name, len(warm_train))
        self._cat_means = warm_train.groupby("cat_id")["sales"].mean()
        logger.info("[%s] Category means: %s", self.name,
                    self._cat_means.round(4).to_dict())
        return self

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        각 cold 아이템의 cat_id로 예측값을 조회해 반환한다.

        Args:
            cold_test: cold 아이템 테스트 데이터 (sales 컬럼은 평가용, 미사용).
            features: 미사용.

        Returns:
            pd.DataFrame: [item_id, store_id, date, pred_sales, cat_id]
        """
        if self._cat_means is None:
            raise RuntimeError("Call fit() before predict().")

        pred = cold_test[["item_id", "store_id", "date", "cat_id"]].copy()
        pred["pred_sales"] = pred["cat_id"].map(self._cat_means)

        unmapped = pred["pred_sales"].isna().sum()
        if unmapped:
            logger.warning("[%s] %d rows have unmapped cat_id → filling with 0.",
                           self.name, unmapped)
            pred["pred_sales"] = pred["pred_sales"].fillna(0.0)

        self._validate_predict_output(pred)
        logger.info("[%s] Prediction complete: %d rows.", self.name, len(pred))
        return pred
