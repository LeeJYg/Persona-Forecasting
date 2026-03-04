"""Baseline 2: Similar Item Average.

예측 값 = warm_train에서 동일 cat_id + 유사 가격 tier 아이템들의 평균 판매량.
sell_prices.csv를 활용해 cold 아이템의 가격 tier를 결정하고,
동일 tier warm 아이템들의 평균을 예측값으로 사용한다.

features 인수 컨벤션:
    fit()   → features["prices"] : sell_prices DataFrame
    predict() → features["prices"] : (동일 DataFrame, cold 아이템 가격 조회용)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines.base import ForecastModel

logger = logging.getLogger(__name__)


class SimilarItemAverage(ForecastModel):
    """
    가격 tier가 유사한 warm 아이템들의 카테고리 평균으로 cold-start를 예측한다.

    학습 단계:
        1. sell_prices에서 warm 아이템들의 평균 가격 계산
        2. 카테고리 내 n_quantiles 분위로 price_tier 부여 (Low / Medium / High)
        3. (cat_id, price_tier) 그룹의 평균 판매량 계산

    예측 단계:
        1. sell_prices에서 cold 아이템의 가격 조회
        2. fit에서 학습한 분위 경계로 price_tier 결정
        3. 해당 (cat_id, price_tier)의 평균 판매량을 예측값으로 사용

    Args:
        n_quantiles: 가격 tier 분위 수 (기본 3 = Low/Medium/High).
        price_lookback_weeks: cold 아이템 가격 조회 시 직전 N주 평균.
    """

    _TIER_LABELS = {3: ["Low", "Medium", "High"]}

    def __init__(
        self, n_quantiles: int = 3, price_lookback_weeks: int = 13
    ) -> None:
        self.n_quantiles = n_quantiles
        self.price_lookback_weeks = price_lookback_weeks
        self._group_means: pd.Series | None = None          # {(cat_id, tier): mean}
        self._price_bins: dict[str, pd.Series] | None = None  # {cat_id: bin_edges}
        self._tier_labels: list[str] = self._TIER_LABELS.get(
            n_quantiles, [str(i) for i in range(n_quantiles)]
        )

    @property
    def name(self) -> str:
        return "similar_item_average"

    # ------------------------------------------------------------------

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "SimilarItemAverage":
        """
        warm 아이템의 (cat_id, price_tier) 그룹 평균 판매량을 계산한다.

        Args:
            warm_train: cold 아이템이 제외된 학습 데이터.
            features: {"prices": sell_prices DataFrame} 필수.
                      sell_prices 컬럼: store_id, item_id, wm_yr_wk, sell_price
        """
        if features is None or "prices" not in features:
            raise ValueError(
                f"[{self.name}] fit() requires features['prices'] (sell_prices DataFrame)."
            )
        prices = features["prices"]
        logger.info("[%s] Fitting ...", self.name)

        # 1. warm 아이템 평균 가격 계산 (학습 기간 전체 평균)
        warm_item_ids = warm_train["item_id"].unique()
        warm_prices = (
            prices[prices["item_id"].isin(warm_item_ids)]
            .groupby("item_id")["sell_price"]
            .mean()
            .rename("mean_price")
        )

        # 2. warm 아이템 메타 정보 결합
        item_meta = (
            warm_train[["item_id", "cat_id"]]
            .drop_duplicates("item_id")
            .set_index("item_id")
        )
        warm_info = item_meta.join(warm_prices, how="left")
        warm_info["mean_price"] = warm_info["mean_price"].fillna(
            warm_info.groupby("cat_id")["mean_price"].transform("median")
        )

        # 3. 카테고리 내 분위 경계 계산 및 price_tier 부여
        self._price_bins = {}
        tier_col_list = []
        for cat, grp in warm_info.groupby("cat_id"):
            prices_cat = grp["mean_price"].dropna()
            quantiles = np.quantile(
                prices_cat, np.linspace(0, 1, self.n_quantiles + 1)
            )
            # 경계 중복 제거 (동일 가격 상품이 많은 경우 대비)
            quantiles = np.unique(quantiles)
            self._price_bins[cat] = quantiles

            tier = pd.cut(
                grp["mean_price"],
                bins=quantiles,
                labels=self._tier_labels[: len(quantiles) - 1],
                include_lowest=True,
            ).astype(str)
            tier_col_list.append(grp.assign(price_tier=tier))
            logger.debug("[%s] %s bins: %s", self.name, cat, quantiles.round(2))

        warm_info_tiered = pd.concat(tier_col_list)

        # 4. (cat_id, price_tier) 그룹별 평균 판매량 계산
        # warm_train에 tier 정보 결합 후 집계
        warm_with_tier = warm_train.merge(
            warm_info_tiered[["cat_id", "price_tier"]].reset_index(),
            on=["item_id", "cat_id"],
            how="left",
        )
        warm_with_tier["price_tier"] = warm_with_tier["price_tier"].fillna("Medium")

        self._group_means = (
            warm_with_tier.groupby(["cat_id", "price_tier"])["sales"].mean()
        )
        logger.info(
            "[%s] Group means (%d groups):\n%s",
            self.name,
            len(self._group_means),
            self._group_means.round(4).to_string(),
        )
        return self

    # ------------------------------------------------------------------

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        cold 아이템의 가격 tier를 결정하고 그룹 평균을 예측값으로 반환한다.

        Args:
            cold_test: cold 아이템 테스트 데이터.
            features: {"prices": sell_prices DataFrame} 필수.

        Returns:
            pd.DataFrame: [item_id, store_id, date, pred_sales, cat_id, price_tier]
        """
        if self._group_means is None or self._price_bins is None:
            raise RuntimeError("Call fit() before predict().")
        if features is None or "prices" not in features:
            raise ValueError(
                f"[{self.name}] predict() requires features['prices']."
            )
        prices = features["prices"]

        # 1. cold 아이템의 가격 조회 (직전 N주 평균)
        cold_item_ids = cold_test["item_id"].unique()
        cold_prices = (
            prices[prices["item_id"].isin(cold_item_ids)]
            .nlargest(
                len(prices[prices["item_id"].isin(cold_item_ids)]),
                "wm_yr_wk"
            )  # 최신 주가 큰 순서
        )
        # 직전 price_lookback_weeks 주 평균
        recent_cold_prices = (
            cold_prices
            .groupby("item_id")
            .apply(lambda g: g.nlargest(self.price_lookback_weeks, "wm_yr_wk")["sell_price"].mean())
            .rename("mean_price")
        )

        # cold 아이템 메타 결합
        cold_meta = (
            cold_test[["item_id", "cat_id"]]
            .drop_duplicates("item_id")
            .set_index("item_id")
        )
        cold_info = cold_meta.join(recent_cold_prices, how="left")

        # 2. 카테고리 내 분위 경계로 price_tier 결정
        tier_map: dict[str, str] = {}
        for cat, grp in cold_info.groupby("cat_id"):
            bins = self._price_bins.get(str(cat))
            if bins is None:
                logger.warning(
                    "[%s] No price bins for cat=%s → defaulting to 'Medium'", self.name, cat
                )
                for iid in grp.index:
                    tier_map[iid] = "Medium"
                continue
            tier = pd.cut(
                grp["mean_price"].fillna(np.median(bins)),
                bins=bins,
                labels=self._tier_labels[: len(bins) - 1],
                include_lowest=True,
            ).astype(str)
            tier_map.update(tier.to_dict())

        # 3. 예측값 매핑
        pred = cold_test[["item_id", "store_id", "date", "cat_id"]].copy()
        pred["price_tier"] = pred["item_id"].map(tier_map).fillna("Medium")
        pred["pred_sales"] = pred.set_index(["cat_id", "price_tier"]).index.map(
            self._group_means
        )

        unmapped = pred["pred_sales"].isna().sum()
        if unmapped:
            # fallback: 카테고리 전체 평균
            cat_fallback = self._group_means.groupby(level="cat_id").mean()
            pred.loc[pred["pred_sales"].isna(), "pred_sales"] = (
                pred.loc[pred["pred_sales"].isna(), "cat_id"].map(cat_fallback)
            )
            logger.warning(
                "[%s] %d rows fell back to category mean (no matching price tier).",
                self.name, unmapped,
            )

        pred["pred_sales"] = pred["pred_sales"].fillna(0.0)
        self._validate_predict_output(pred)
        logger.info("[%s] Prediction complete: %d rows.", self.name, len(pred))
        return pred
