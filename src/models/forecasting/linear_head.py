"""Track B: 선형 회귀 헤드 (임베딩 → 주별 판매량 예측).

설계 원칙
----------
1. 입력: x_item (n_items, hidden_size) — QwenEmbedder.build_item_embeddings() 결과
2. 출력: y_pred (n_items, n_weeks) — 아이템별 주간 판매량
3. 모델: Ridge Regression (sklearn) — cross-validation으로 alpha 최적화
4. 학습: warm 아이템 300개의 실제 주간 판매량을 target으로 사용
5. 예측: cold 아이템 100개에 적용

주간 판매량 집계:
    warm_test.csv의 일별 판매량을 ISO 주 단위로 집계.
    cold_test와 동일한 예측 기간(d_1799 ~ d_1913)에 해당하는 주를 사용.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WeeklySalesHead:
    """임베딩 → 주간 판매량 Ridge 회귀 헤드.

    Args:
        alpha: Ridge 정규화 강도. None이면 cross-validation으로 자동 선택.
        cv_folds: cross-validation fold 수.
        alphas_to_try: CV에서 탐색할 alpha 값 목록.
    """

    def __init__(
        self,
        alpha: float | None = None,
        cv_folds: int = 5,
        alphas_to_try: list[float] | None = None,
    ) -> None:
        self._alpha = alpha
        self._cv_folds = cv_folds
        self._alphas_to_try = alphas_to_try or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self._model = None
        self._best_alpha: float | None = None
        self._n_weeks: int | None = None
        self._item_scaler = None   # StandardScaler for X
        self._sales_scaler = None  # StandardScaler for y

    def fit(
        self,
        x_warm: np.ndarray,
        y_warm_weekly: np.ndarray,
    ) -> "WeeklySalesHead":
        """warm 아이템 임베딩과 주간 판매량으로 Ridge 회귀를 학습한다.

        Args:
            x_warm: shape (n_warm, hidden_size) — warm 아이템 임베딩.
            y_warm_weekly: shape (n_warm, n_weeks) — warm 아이템 주간 실제 판매량.

        Returns:
            self
        """
        from sklearn.linear_model import Ridge, RidgeCV
        from sklearn.preprocessing import StandardScaler

        n_warm, hidden_size = x_warm.shape
        self._n_weeks = y_warm_weekly.shape[1]

        logger.info(
            "회귀 헤드 학습: x_warm=%s, y_warm=%s",
            x_warm.shape, y_warm_weekly.shape,
        )

        # 입력 표준화
        self._item_scaler = StandardScaler()
        x_scaled = self._item_scaler.fit_transform(x_warm)

        # 출력 표준화 (주별 판매량 scale이 다를 수 있으므로)
        self._sales_scaler = StandardScaler()
        y_scaled = self._sales_scaler.fit_transform(y_warm_weekly)

        if self._alpha is None:
            # Cross-validation으로 최적 alpha 선택
            model_cv = RidgeCV(
                alphas=np.array(self._alphas_to_try),
                cv=min(self._cv_folds, n_warm),
                scoring="neg_mean_absolute_error",
            )
            model_cv.fit(x_scaled, y_scaled)
            self._best_alpha = float(model_cv.alpha_)
            logger.info("최적 alpha (CV): %.4f", self._best_alpha)
        else:
            self._best_alpha = self._alpha

        # 최적 alpha로 최종 학습
        self._model = Ridge(alpha=self._best_alpha)
        self._model.fit(x_scaled, y_scaled)

        # 학습 성능 로깅
        y_pred_train = self._sales_scaler.inverse_transform(
            self._model.predict(x_scaled)
        )
        train_mae = float(np.abs(y_warm_weekly - y_pred_train).mean())
        logger.info(
            "학습 완료: n_warm=%d, hidden=%d, n_weeks=%d, alpha=%.4f, train_MAE=%.4f",
            n_warm, hidden_size, self._n_weeks, self._best_alpha, train_mae,
        )
        return self

    def predict(self, x_cold: np.ndarray) -> np.ndarray:
        """cold 아이템 임베딩으로 주간 판매량을 예측한다.

        Args:
            x_cold: shape (n_cold, hidden_size).

        Returns:
            np.ndarray: shape (n_cold, n_weeks), non-negative values.
        """
        if self._model is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        x_scaled = self._item_scaler.transform(x_cold)
        y_scaled = self._model.predict(x_scaled)
        y_pred = self._sales_scaler.inverse_transform(y_scaled)

        # 음수 예측 clip (판매량은 non-negative)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

        logger.info(
            "예측 완료: shape=%s, mean=%.4f, min=%.4f, max=%.4f",
            y_pred.shape, y_pred.mean(), y_pred.min(), y_pred.max(),
        )
        return y_pred

    def save(self, save_path: Path) -> None:
        """학습된 모델 파라미터를 저장한다.

        Args:
            save_path: 저장 경로 (디렉토리 자동 생성).
        """
        import pickle
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self._model,
            "item_scaler": self._item_scaler,
            "sales_scaler": self._sales_scaler,
            "best_alpha": self._best_alpha,
            "n_weeks": self._n_weeks,
        }
        with save_path.open("wb") as f:
            pickle.dump(state, f)
        logger.info("모델 저장: %s", save_path)

    def load(self, load_path: Path) -> "WeeklySalesHead":
        """저장된 모델 파라미터를 로드한다.

        Args:
            load_path: .pkl 파일 경로.

        Returns:
            self
        """
        import pickle
        with load_path.open("rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._item_scaler = state["item_scaler"]
        self._sales_scaler = state["sales_scaler"]
        self._best_alpha = state["best_alpha"]
        self._n_weeks = state["n_weeks"]
        logger.info("모델 로드: %s (alpha=%.4f)", load_path, self._best_alpha)
        return self


def aggregate_weekly_sales(
    sales_df: pd.DataFrame,
    item_ids: list[str],
    date_start: str,
    date_end: str,
) -> np.ndarray:
    """판매 DataFrame에서 지정 기간의 아이템별 주간 판매량 행렬을 만든다.

    Args:
        sales_df: 컬럼 [item_id, date, sales] 포함.
        item_ids: 아이템 ID 목록 (출력 행 순서 결정).
        date_start: 예측 시작 날짜 ("YYYY-MM-DD").
        date_end: 예측 종료 날짜 ("YYYY-MM-DD").

    Returns:
        np.ndarray: shape (len(item_ids), n_weeks).
                    n_weeks = 7일 단위 집계 주 수.
    """
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.Timestamp(date_start)) & (df["date"] <= pd.Timestamp(date_end))
    df = df[mask & df["item_id"].isin(item_ids)].copy()

    # 날짜 정렬 기반 주 번호 부여 (실제 달력 주 아닌 순서 기준)
    all_dates = sorted(df["date"].unique())
    date_to_week = {d: i // 7 for i, d in enumerate(all_dates)}
    df["week_idx"] = df["date"].map(date_to_week)
    n_weeks = max(date_to_week.values()) + 1 if date_to_week else 0

    weekly = (
        df.groupby(["item_id", "week_idx"])["sales"]
        .sum()
        .unstack(fill_value=0)
    )
    # 아이템 순서 정렬 및 누락 아이템 0으로 채우기
    missing_items = [iid for iid in item_ids if iid not in weekly.index]
    if missing_items:
        logger.warning("판매 데이터 없는 아이템: %d개 → 0으로 채움", len(missing_items))
    weekly = weekly.reindex(item_ids, fill_value=0)

    # 주 컬럼 정렬
    all_week_cols = list(range(n_weeks))
    weekly = weekly.reindex(columns=all_week_cols, fill_value=0)

    return weekly.values.astype(float)


def build_pred_dataframe(
    y_weekly: np.ndarray,
    item_ids: list[str],
    store_id: str,
    cold_test: pd.DataFrame,
) -> pd.DataFrame:
    """주별 예측 행렬을 [item_id, store_id, date, pred_sales] DataFrame으로 변환한다.

    주별 예측값을 해당 주의 일수로 균등 배분한다.

    Args:
        y_weekly: shape (n_items, n_weeks).
        item_ids: 아이템 ID 목록.
        store_id: 매장 ID.
        cold_test: 일별 날짜 정보 조회용.

    Returns:
        pd.DataFrame with columns: item_id, store_id, date, pred_sales, cat_id.
    """
    cold_test = cold_test.copy()
    cold_test["date"] = pd.to_datetime(cold_test["date"])

    # 날짜 → 주 인덱스 매핑
    all_dates = sorted(cold_test["date"].unique())
    date_to_week_idx = {d: i // 7 for i, d in enumerate(all_dates)}

    rows: list[dict] = []
    for item_idx, item_id in enumerate(item_ids):
        item_sub = cold_test[cold_test["item_id"] == item_id].copy()
        cat_id = str(item_sub["cat_id"].iloc[0]) if not item_sub.empty else "UNKNOWN"

        for _, row in item_sub.iterrows():
            d = row["date"]
            wk_idx = date_to_week_idx.get(d, 0)
            if wk_idx < y_weekly.shape[1]:
                weekly_qty = float(y_weekly[item_idx, wk_idx])
            else:
                weekly_qty = 0.0

            # 주 내 일수 계산 (마지막 주 부분 처리)
            same_week_dates = [dd for dd, wi in date_to_week_idx.items() if wi == wk_idx]
            n_days_in_week = len(same_week_dates)
            daily_pred = weekly_qty / max(n_days_in_week, 1)

            rows.append({
                "item_id": item_id,
                "store_id": store_id,
                "date": d,
                "pred_sales": daily_pred,
                "cat_id": cat_id,
            })

    return pd.DataFrame(rows)
