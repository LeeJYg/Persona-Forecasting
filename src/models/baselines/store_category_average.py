"""Baseline 3: Store-Category Average.

예측 값 = warm_train에서 동일 매장(target_store) + 동일 cat_id 아이템들의
평균 일별 판매량. 동일 매장의 유사 카테고리 상품 이력을 활용하므로
Global Category Average보다 더 맥락적인 베이스라인이다.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.models.baselines.base import ForecastModel

logger = logging.getLogger(__name__)


class StoreCategoryAverage(ForecastModel):
    """
    대상 매장(target_store)의 카테고리 평균으로 cold-start 아이템 판매량을 예측한다.

    학습: warm_train에서 target_store 행만 필터링한 후 cat_id별 평균 계산.
    예측: cold 아이템의 (store_id, cat_id)에 해당하는 평균값을 모든 날짜에 적용.

    Notes
    -----
    cold_test의 store_id가 항상 target_store와 동일해야 한다.
    cross_store_info=False 조건에서 warm_train의 target_store 데이터는
    cold 아이템이 제거된 상태임에 유의.
    """

    def __init__(self, target_store: str) -> None:
        """
        Args:
            target_store: cold-start 시뮬레이션 대상 매장 ID (예: "CA_1").
        """
        self.target_store = target_store
        self._store_cat_means: pd.Series | None = None  # {(store_id, cat_id): mean}

    @property
    def name(self) -> str:
        return "store_category_average"

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "StoreCategoryAverage":
        """
        target_store의 cat_id별 평균 일별 판매량을 계산한다.

        Args:
            warm_train: cold 아이템이 제외된 학습 데이터 (모든 매장 포함).
            features: 미사용.
        """
        logger.info(
            "[%s] Fitting on store=%s ...", self.name, self.target_store
        )
        store_data = warm_train[warm_train["store_id"] == self.target_store]
        if store_data.empty:
            raise ValueError(
                f"[{self.name}] No warm_train data found for store={self.target_store}"
            )

        self._store_cat_means = (
            store_data.groupby(["store_id", "cat_id"])["sales"].mean()
        )
        logger.info(
            "[%s] Store-category means (%d groups): %s",
            self.name,
            len(self._store_cat_means),
            self._store_cat_means.round(4).to_dict(),
        )
        return self

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        cold 아이템의 (store_id, cat_id)로 예측값을 조회해 반환한다.

        Args:
            cold_test: cold 아이템 테스트 데이터.
            features: 미사용.

        Returns:
            pd.DataFrame: [item_id, store_id, date, pred_sales, cat_id]
        """
        if self._store_cat_means is None:
            raise RuntimeError("Call fit() before predict().")

        pred = cold_test[["item_id", "store_id", "date", "cat_id"]].copy()
        pred["pred_sales"] = pred.set_index(["store_id", "cat_id"]).index.map(
            self._store_cat_means
        )

        unmapped = pred["pred_sales"].isna().sum()
        if unmapped:
            logger.warning(
                "[%s] %d rows have unmapped (store_id, cat_id) → filling with 0.",
                self.name, unmapped,
            )
            pred["pred_sales"] = pred["pred_sales"].fillna(0.0)

        self._validate_predict_output(pred)
        logger.info("[%s] Prediction complete: %d rows.", self.name, len(pred))
        return pred
