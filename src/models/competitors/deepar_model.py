"""Competitor 5: DeepAR (GluonTS).

Warm items 300개로 DeepAR 모델 학습, cold items에 대해 static features로 예측.
Cold item의 context는 category mean으로 초기화.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines.base import ForecastModel

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH = 4   # 4 weeks context
_PREDICTION_LENGTH = 17  # 17 weeks horizon


class DeepARModel(ForecastModel):
    """GluonTS DeepAR 기반 cold-start demand forecasting.

    Args:
        epochs: 학습 epoch 수 (기본 50).
        seed: 재현성 seed.
    """

    def __init__(self, epochs: int = 50, seed: int = 42) -> None:
        self._epochs = epochs
        self._seed = seed
        self._predictor: Any = None
        self._cat_weekly_mean: dict[str, list[float]] = {}  # {cat_id: [52 weekly means]}
        self._warm_item_meta: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "deepar"

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "DeepARModel":
        """warm_train_weekly로 DeepAR 학습."""
        try:
            from gluonts.dataset.common import ListDataset
            from gluonts.torch.model.deepar import DeepAREstimator
            import torch
        except ImportError as e:
            raise ImportError(
                "gluonts[torch] 패키지가 필요합니다: pip install gluonts[torch]"
            ) from e

        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        # cold item context용 category weekly mean 계산
        for cat, grp in warm_train.groupby("cat_id"):
            wk_mean = grp.groupby("iso_week")["sales"].mean()
            self._cat_weekly_mean[cat] = [float(wk_mean.get(w, 0.0)) for w in range(1, 53)]

        # warm item 메타 정보
        self._warm_item_meta = warm_train[["item_id", "cat_id", "dept_id"]].drop_duplicates("item_id")

        # cat_id, dept_id 인코딩
        cats = sorted(warm_train["cat_id"].unique())
        depts = sorted(warm_train["dept_id"].unique())
        cat2idx = {c: i for i, c in enumerate(cats)}
        dept2idx = {d: i for i, d in enumerate(depts)}
        self._cat2idx = cat2idx
        self._dept2idx = dept2idx

        # GluonTS ListDataset 구성: date 컬럼(주 월요일)으로 Period 생성
        start_date = warm_train["date"].min()
        start_ts = pd.Period(start_date, freq="W")

        train_data = []
        for item_id, grp in warm_train.groupby("item_id"):
            grp_sorted = grp.sort_values(["iso_year", "iso_week"])
            target = grp_sorted["sales"].values.astype(float)
            meta = self._warm_item_meta[self._warm_item_meta["item_id"] == item_id].iloc[0]
            train_data.append({
                "start": start_ts,
                "target": target,
                "feat_static_cat": [cat2idx[meta["cat_id"]], dept2idx[meta["dept_id"]]],
            })

        dataset = ListDataset(train_data, freq="W")

        estimator = DeepAREstimator(
            freq="W",
            context_length=_CONTEXT_LENGTH,
            prediction_length=_PREDICTION_LENGTH,
            num_layers=2,
            hidden_size=40,
            dropout_rate=0.1,
            num_feat_static_cat=2,
            cardinality=[len(cats), len(depts)],
            trainer_kwargs={
                "max_epochs": self._epochs,
                "accelerator": "cpu",
                "enable_progress_bar": False,
            },
        )

        self._predictor = estimator.train(dataset)
        logger.info("[%s] 학습 완료: %d warm items, epochs=%d", self.name, len(train_data), self._epochs)
        return self

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """cold items에 대해 context=category mean으로 초기화 후 예측."""
        if self._predictor is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출해야 합니다.")

        from gluonts.dataset.common import ListDataset

        cold_items = cold_test[["item_id", "cat_id", "dept_id"]].drop_duplicates("item_id")
        # cold_test의 첫 주 date(월요일)로 context start 계산
        first_date = cold_test["date"].min()
        context_start = pd.Period(first_date, freq="W") - _CONTEXT_LENGTH

        test_data = []
        for _, row in cold_items.iterrows():
            cat_means = self._cat_weekly_mean.get(row["cat_id"], [0.5] * 52)
            # context = category mean for _CONTEXT_LENGTH recent weeks
            context = [cat_means[w % 52] for w in range(_CONTEXT_LENGTH)]
            test_data.append({
                "start": context_start,
                "target": np.array(context, dtype=float),
                "feat_static_cat": [
                    self._cat2idx.get(row["cat_id"], 0),
                    self._dept2idx.get(row["dept_id"], 0),
                ],
            })

        dataset = ListDataset(test_data, freq="W")
        forecast_it = self._predictor.predict(dataset)

        preds: list[dict] = []
        iso_weeks_per_item = (
            cold_test.sort_values(["item_id", "iso_year", "iso_week"])
            .groupby("item_id")[["iso_year", "iso_week"]]
            .apply(lambda g: g.to_dict(orient="records"))
            .to_dict()
        )

        for item_row, fc in zip(cold_items.itertuples(), forecast_it):
            mean_pred = fc.mean.tolist()
            weeks = iso_weeks_per_item.get(item_row.item_id, [])
            for w_idx, week_info in enumerate(weeks):
                pred_val = float(mean_pred[w_idx]) if w_idx < len(mean_pred) else 0.0
                preds.append({
                    "item_id": item_row.item_id,
                    "iso_year": week_info["iso_year"],
                    "iso_week": week_info["iso_week"],
                    "pred_sales": max(pred_val, 0.0),
                })

        result = cold_test[["item_id", "store_id", "cat_id", "iso_year", "iso_week", "date"]].merge(
            pd.DataFrame(preds)[["item_id", "iso_year", "iso_week", "pred_sales"]],
            on=["item_id", "iso_year", "iso_week"],
            how="left",
        )
        result["pred_sales"] = result["pred_sales"].fillna(0.0)
        self._validate_predict_output(result)
        return result
