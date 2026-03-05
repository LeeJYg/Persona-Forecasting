"""Competitor 2: LightGBM Cross-Learning (Static / Proxy Lags).

2-A (Static): 정적 속성 + 시간 feature로 학습.
2-B (Proxy Lags): 2-A features + k-NN top-3 이웃의 rolling 판매량 proxy.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines.base import ForecastModel

logger = logging.getLogger(__name__)


class LightGBMCross(ForecastModel):
    """LightGBM Cross-Learning for cold-start demand forecasting.

    Args:
        variant: "static" | "proxy_lags"
        objective: "tweedie" | "l1"
        knn_neighbors_path: proxy_lags 사용 시 k-NN 이웃 JSON 파일 경로.
        seed: 재현성 seed.
        config: DotDict 설정 객체 (num_leaves, learning_rate 등).
    """

    def __init__(
        self,
        variant: str = "static",
        objective: str = "tweedie",
        knn_neighbors_path: Path | None = None,
        seed: int = 42,
        config: Any = None,
    ) -> None:
        if variant not in ("static", "proxy_lags"):
            raise ValueError(f"variant must be 'static' or 'proxy_lags', got '{variant}'")
        self._variant = variant
        self._objective = objective
        self._knn_neighbors_path = knn_neighbors_path
        self._seed = seed
        self._config = config
        self._model: Any = None
        self._cat_means: dict[str, float] = {}
        self._dept_means: dict[str, float] = {}
        self._knn_neighbors: dict[str, list[dict]] = {}
        self._warm_iso_week_mean: pd.DataFrame | None = None
        self._feature_names: list[str] = []
        self._feature_importances: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return f"lightgbm_{self._variant}"

    def _build_snap_feature(self, calendar: pd.DataFrame, iso_week_col: pd.Series, iso_year_col: pd.Series) -> pd.Series:
        """ISO year+week에 해당하는 주간 SNAP 일수 계산."""
        if calendar is None:
            return pd.Series(0, index=iso_week_col.index)
        cal = calendar.copy()
        cal["date"] = pd.to_datetime(cal["date"])
        cal["iso_year"] = cal["date"].dt.isocalendar().year.astype(int)
        cal["iso_week"] = cal["date"].dt.isocalendar().week.astype(int)
        snap_weekly = (
            cal.groupby(["iso_year", "iso_week"])["snap_CA"]
            .sum()
            .reset_index()
            .rename(columns={"snap_CA": "snap_count"})
        )
        tmp = pd.DataFrame({"iso_year": iso_year_col, "iso_week": iso_week_col})
        return tmp.merge(snap_weekly, on=["iso_year", "iso_week"], how="left")["snap_count"].fillna(0)

    def _build_features(
        self,
        df: pd.DataFrame,
        warm_train: pd.DataFrame,
        prices: pd.DataFrame,
        calendar: pd.DataFrame | None,
        fit: bool = False,
    ) -> pd.DataFrame:
        """feature matrix 생성 (static + time + cross-sectional)."""
        feat = df[["item_id", "cat_id", "dept_id", "iso_year", "iso_week"]].copy()

        # 카테고리 인코딩
        feat["cat_enc"] = pd.Categorical(feat["cat_id"]).codes
        feat["dept_enc"] = pd.Categorical(feat["dept_id"]).codes

        # sell_price: 해당 item의 전체 평균
        item_price = prices.groupby("item_id")["sell_price"].mean()
        feat["sell_price"] = feat["item_id"].map(item_price).fillna(item_price.median())

        # 시간 features
        feat["month"] = ((feat["iso_week"] - 1) // 4 + 1).clip(1, 12)
        feat["snap_count"] = self._build_snap_feature(calendar, feat["iso_week"], feat["iso_year"])

        # 집계 features (학습 시 warm 기준, 예측 시 저장된 값 사용)
        if fit:
            self._cat_means = warm_train.groupby("cat_id")["sales"].mean().to_dict()
            self._dept_means = warm_train.groupby("dept_id")["sales"].mean().to_dict()
        feat["cat_weekly_mean"] = feat["cat_id"].map(self._cat_means).fillna(0.0)
        feat["dept_weekly_mean"] = feat["dept_id"].map(self._dept_means).fillna(0.0)

        base_features = [
            "cat_enc", "dept_enc", "sell_price", "iso_week", "month",
            "snap_count", "cat_weekly_mean", "dept_weekly_mean",
        ]

        if self._variant == "proxy_lags":
            if fit:
                self._warm_iso_week_mean = (
                    warm_train.groupby(["item_id", "iso_week"])["sales"]
                    .mean()
                    .reset_index()
                    .rename(columns={"sales": "mean_sales"})
                )
            feat = self._add_proxy_lag_features(feat)
            base_features += ["knn_top3_overall_mean", "knn_top3_same_week_mean"]

        self._feature_names = base_features
        return feat[base_features]

    def _add_proxy_lag_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        """k-NN top-3 이웃의 판매량 proxy 추가."""
        if not self._knn_neighbors:
            feat["knn_top3_overall_mean"] = 0.0
            feat["knn_top3_same_week_mean"] = 0.0
            return feat

        iso_week_mean_map = (
            self._warm_iso_week_mean.set_index(["item_id", "iso_week"])["mean_sales"]
            if self._warm_iso_week_mean is not None
            else None
        )
        overall_mean_map = (
            self._warm_iso_week_mean.groupby("item_id")["mean_sales"].mean().to_dict()
            if self._warm_iso_week_mean is not None
            else {}
        )

        def proxy_for_row(row: pd.Series) -> tuple[float, float]:
            neighbors = self._knn_neighbors.get(row["item_id"], [])
            if not neighbors:
                return 0.0, 0.0
            n_ids = [n["item_id"] for n in neighbors[:3]]
            sims = np.array([n["similarity"] for n in neighbors[:3]])
            weights = np.maximum(sims, 0.0)
            wsum = weights.sum()
            if wsum == 0:
                weights = np.ones(len(n_ids)) / len(n_ids)
            else:
                weights /= wsum

            overall = np.array([overall_mean_map.get(iid, 0.0) for iid in n_ids])
            same_wk = np.array([
                iso_week_mean_map.get((iid, row["iso_week"]), overall_mean_map.get(iid, 0.0))
                if iso_week_mean_map is not None else 0.0
                for iid in n_ids
            ])
            return float(np.dot(weights, overall)), float(np.dot(weights, same_wk))

        proxy = feat.apply(proxy_for_row, axis=1, result_type="expand")
        feat["knn_top3_overall_mean"] = proxy[0]
        feat["knn_top3_same_week_mean"] = proxy[1]
        return feat

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "LightGBMCross":
        """warm_train_weekly로 LightGBM 학습.

        Args:
            warm_train: 주간 집계 warm items 데이터.
                        컬럼: item_id, cat_id, dept_id, iso_year, iso_week, sales.
            features: {"prices": sell_prices_df, "calendar": calendar_df}.
        """
        import lightgbm as lgb

        if features is None or "prices" not in features:
            raise ValueError(f"[{self.name}] fit() requires features['prices']")

        prices = features["prices"]
        calendar = features.get("calendar")

        if self._variant == "proxy_lags":
            if self._knn_neighbors_path is None or not self._knn_neighbors_path.exists():
                raise ValueError(
                    f"[{self.name}] proxy_lags requires knn_neighbors_path. "
                    f"Run knn_analog first. Path: {self._knn_neighbors_path}"
                )
            self._knn_neighbors = json.loads(
                self._knn_neighbors_path.read_text(encoding="utf-8")
            )

        # hold-out split (80/20)
        all_items = warm_train["item_id"].unique()
        rng = np.random.default_rng(self._seed)
        val_items = rng.choice(all_items, size=int(len(all_items) * 0.2), replace=False)
        train_mask = ~warm_train["item_id"].isin(val_items)
        val_mask = warm_train["item_id"].isin(val_items)

        X_train = self._build_features(warm_train[train_mask], warm_train, prices, calendar, fit=True)
        y_train = warm_train[train_mask]["sales"].values
        X_val = self._build_features(warm_train[val_mask], warm_train, prices, calendar, fit=False)
        y_val = warm_train[val_mask]["sales"].values

        cfg = self._config
        if cfg is not None:
            num_leaves = int(cfg.competitors.lgbm_num_leaves)
            lr = float(cfg.competitors.lgbm_learning_rate)
            n_est = int(cfg.competitors.lgbm_n_estimators)
            es = int(cfg.competitors.lgbm_early_stopping_rounds)
            tweedie_vp = float(cfg.competitors.lgbm_tweedie_variance_power)
        else:
            num_leaves, lr, n_est, es, tweedie_vp = 31, 0.05, 500, 50, 1.5

        params: dict[str, Any] = {
            "objective": self._objective,
            "num_leaves": num_leaves,
            "learning_rate": lr,
            "n_estimators": n_est,
            "random_state": self._seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        if self._objective == "tweedie":
            params["tweedie_variance_power"] = tweedie_vp

        self._model = lgb.LGBMRegressor(**params)
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(100)],
        )

        val_pred = np.maximum(self._model.predict(X_val), 0.0)
        val_mae = float(np.abs(y_val - val_pred).mean())
        logger.info(
            "[%s] 학습 완료 | best_iter=%d | val_MAE=%.4f (hold-out %d items)",
            self.name, self._model.best_iteration_, val_mae, len(val_items),
        )

        self._feature_importances = pd.DataFrame({
            "feature": self._feature_names,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return self

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """cold_test_weekly 각 행의 주간 판매량 예측."""
        if self._model is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출해야 합니다.")
        if features is None or "prices" not in features:
            raise ValueError(f"[{self.name}] predict() requires features['prices']")

        prices = features["prices"]
        calendar = features.get("calendar")

        X_cold = self._build_features(cold_test, None, prices, calendar, fit=False)
        pred_vals = np.maximum(self._model.predict(X_cold), 0.0)

        result = cold_test[["item_id", "store_id", "cat_id", "iso_year", "iso_week", "date"]].copy()
        result["pred_sales"] = pred_vals
        self._validate_predict_output(result)
        return result

    @property
    def feature_importances(self) -> pd.DataFrame | None:
        """Feature importance DataFrame (학습 후 사용 가능)."""
        return self._feature_importances
