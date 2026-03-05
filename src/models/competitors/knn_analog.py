"""Competitor 1: k-NN Analogous Forecasting.

상품 속성(cat, dept, price) 기반 cosine similarity로 k개의 유사 warm item을 찾고,
그들의 ISO-week 평균 주간 판매량으로 cold item을 예측한다.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.models.baselines.base import ForecastModel

logger = logging.getLogger(__name__)


class KNNAnalog(ForecastModel):
    """k-NN Analogous Forecasting.

    Args:
        k: 유사 이웃 수 (기본 5).
        checkpoint_path: k-NN 결과(이웃 정보)를 저장할 JSON 파일 경로.
                         LightGBM Proxy Lags 등에서 재사용.
    """

    def __init__(self, k: int = 5, checkpoint_path: Path | None = None) -> None:
        self._k = k
        self._checkpoint_path = checkpoint_path
        self._warm_items: list[str] = []
        self._warm_feat: np.ndarray | None = None
        self._warm_iso_week_mean: pd.DataFrame | None = None  # (item_id, iso_week) → mean weekly sales
        self._cold_neighbors: dict[str, list[dict]] = {}  # {cold_item_id: [{item_id, similarity}, ...]}
        self._cat_enc = LabelEncoder()
        self._dept_enc = LabelEncoder()
        self._price_scaler = MinMaxScaler()

    @property
    def name(self) -> str:
        return f"knn_analog_k{self._k}"

    def _build_feature_matrix(
        self,
        items: pd.DataFrame,  # (item_id, cat_id, dept_id, sell_price)
        fit: bool = False,
    ) -> np.ndarray:
        """item 속성 → feature vector (one-hot cat + one-hot dept + normalized price)."""
        df = items.copy()
        if fit:
            self._cat_enc.fit(df["cat_id"])
            self._dept_enc.fit(df["dept_id"])
            self._price_scaler.fit(df[["sell_price"]])

        cat_onehot = pd.get_dummies(
            self._cat_enc.transform(df["cat_id"]), prefix="cat"
        )
        dept_onehot = pd.get_dummies(
            self._dept_enc.transform(df["dept_id"]), prefix="dept"
        )
        price_norm = self._price_scaler.transform(df[["sell_price"]])
        return np.hstack([
            pd.get_dummies(df["cat_id"]).values,
            pd.get_dummies(df["dept_id"]).values,
            price_norm,
        ])

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "KNNAnalog":
        """warm_train_weekly에서 ISO-week 평균 판매량과 feature vector 계산.

        Args:
            warm_train: 주간 집계된 warm items 데이터.
                        컬럼: item_id, cat_id, dept_id, iso_week, sales.
            features: {"prices": sell_prices_df, "cold_stats": cold_item_stats_df} 필요.
        """
        if features is None or "prices" not in features:
            raise ValueError(f"[{self.name}] fit() requires features['prices']")

        prices = features["prices"]

        # warm items의 sell_price 대표값 (전체 기간 평균)
        warm_price = (
            prices[prices["item_id"].isin(warm_train["item_id"].unique())]
            .groupby("item_id")["sell_price"]
            .mean()
            .reset_index()
        )

        # warm items 메타 정보
        warm_meta = (
            warm_train[["item_id", "cat_id", "dept_id"]]
            .drop_duplicates("item_id")
            .merge(warm_price, on="item_id", how="left")
        )
        warm_meta["sell_price"] = warm_meta["sell_price"].fillna(warm_meta["sell_price"].median())

        self._warm_items = warm_meta["item_id"].tolist()
        self._warm_feat = self._build_feature_matrix(warm_meta, fit=True)

        # ISO-week 평균: 여러 해의 같은 ISO week를 평균
        self._warm_iso_week_mean = (
            warm_train.groupby(["item_id", "iso_week"])["sales"]
            .mean()
            .reset_index()
            .rename(columns={"sales": "mean_weekly_sales"})
        )

        logger.info("[%s] fit 완료: %d warm items, %d iso_week 레코드",
                    self.name, len(self._warm_items), len(self._warm_iso_week_mean))
        return self

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """각 cold item에 대해 k-NN 가중 평균 weekly sales 예측.

        Args:
            cold_test: 주간 집계된 cold items 테스트 데이터.
                       컬럼: item_id, store_id, cat_id, dept_id, iso_year, iso_week, date, sales.
            features: {"prices": sell_prices_df} or {"cold_stats": cold_item_stats_df}.
        """
        if self._warm_feat is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출해야 합니다.")
        if features is None or "prices" not in features:
            raise ValueError(f"[{self.name}] predict() requires features['prices']")

        prices = features["prices"]
        cold_items = cold_test[["item_id", "cat_id", "dept_id"]].drop_duplicates("item_id")

        # cold item sell_price
        cold_price = (
            prices[prices["item_id"].isin(cold_items["item_id"])]
            .groupby("item_id")["sell_price"]
            .mean()
            .reset_index()
        )
        cold_meta = cold_items.merge(cold_price, on="item_id", how="left")
        cold_meta["sell_price"] = cold_meta["sell_price"].fillna(
            prices["sell_price"].median()
        )

        cold_feat = self._build_feature_matrix(cold_meta, fit=False)

        # cosine similarity: (n_cold, n_warm)
        sims = cosine_similarity(cold_feat, self._warm_feat)

        # warm_iso_week_mean을 (item_id × iso_week) pivot으로 미리 계산
        iso_week_pivot = self._warm_iso_week_mean.pivot(
            index="item_id", columns="iso_week", values="mean_weekly_sales"
        ).reindex(self._warm_items)  # shape: (n_warm, n_iso_weeks)

        preds = []
        self._cold_neighbors = {}

        for i, cold_item_id in enumerate(cold_meta["item_id"]):
            sim_vec = sims[i]
            top_k_idx = np.argsort(sim_vec)[::-1][: self._k]
            top_k_sims = sim_vec[top_k_idx]
            top_k_items = [self._warm_items[j] for j in top_k_idx]

            # 이웃 정보 저장 (checkpoint용)
            self._cold_neighbors[cold_item_id] = [
                {"item_id": iid, "similarity": float(s)}
                for iid, s in zip(top_k_items, top_k_sims)
            ]

            # 가중치 정규화 (cosine < 0 처리)
            weights = np.maximum(top_k_sims, 0.0)
            weight_sum = weights.sum()
            if weight_sum == 0:
                weights = np.ones(self._k) / self._k
            else:
                weights = weights / weight_sum

            # 이 cold item의 test rows
            item_rows = cold_test[cold_test["item_id"] == cold_item_id].copy()
            for iso_week in item_rows["iso_week"].unique():
                neighbor_sales = np.array([
                    iso_week_pivot.loc[iid, iso_week]
                    if iso_week in iso_week_pivot.columns and not np.isnan(iso_week_pivot.loc[iid, iso_week])
                    else iso_week_pivot.loc[iid].mean()  # fallback: 연간 평균
                    for iid in top_k_items
                ])
                pred_val = float(np.dot(weights, neighbor_sales))
                preds.append({
                    "item_id": cold_item_id,
                    "iso_week": iso_week,
                    "pred_sales": max(pred_val, 0.0),
                })

        pred_df = pd.DataFrame(preds)
        result = cold_test[["item_id", "store_id", "cat_id", "iso_year", "iso_week", "date"]].merge(
            pred_df[["item_id", "iso_week", "pred_sales"]], on=["item_id", "iso_week"], how="left"
        )
        result["pred_sales"] = result["pred_sales"].fillna(0.0)

        # checkpoint 저장
        if self._checkpoint_path is not None:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_path.write_text(
                json.dumps(self._cold_neighbors, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("[%s] k-NN 이웃 정보 저장: %s", self.name, self._checkpoint_path)

        self._validate_predict_output(result)
        return result
