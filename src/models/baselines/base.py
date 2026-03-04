"""추상 기본 클래스: 모든 forecasting 모델이 구현해야 할 인터페이스.

설계 원칙
---------
현재 3개 단순 베이스라인(category/similar/store-category average)뿐만 아니라
향후 추가될 복잡한 모델(LightGBM, k-NN, LLMTime 등)도 동일 인터페이스로 수용한다.

확장성을 위한 핵심 설계 결정:
- fit() / predict() 모두 `features: dict[str, pd.DataFrame] | None` 인수를 받는다.
  예) {"prices": sell_prices_df, "calendar": calendar_df, "embeddings": embed_df}
- **kwargs 로 모델별 실행 시점 파라미터(n_estimators, n_neighbors 등)를 전달한다.
- predict()는 항상 표준 컬럼 [item_id, store_id, date, pred_sales]를 포함하는
  DataFrame을 반환하므로, 평가 파이프라인이 모델 종류와 무관하게 동작한다.
- fit()은 self를 반환(메서드 체이닝 지원).

features dict 컨벤션 (향후 모델 작성 시 준수):
    "prices"       : sell_prices DataFrame   (B2, LightGBM, k-NN)
    "calendar"     : calendar DataFrame      (LightGBM, LLMTime)
    "embeddings"   : item embedding matrix   (k-NN)
    "cold_stats"   : cold item 메타데이터    (LLMTime, LightGBM)
    "llm_scores"   : LLM 구매 의향 점수      (Persona 모델)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# predict()가 반드시 포함해야 하는 컬럼
PRED_REQUIRED_COLS = {"item_id", "store_id", "date", "pred_sales"}


class ForecastModel(ABC):
    """
    Cold-start 수요 예측 모델의 추상 기본 클래스.

    모든 서브클래스는 :meth:`name`, :meth:`fit`, :meth:`predict` 를 구현해야 한다.
    단순 평균 베이스라인부터 LightGBM·k-NN·LLMTime 까지 동일 인터페이스를 공유한다.
    """

    # ------------------------------------------------------------------
    # 필수 구현 항목
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """모델의 고유 이름 (로깅·결과 파일명에 사용)."""

    @abstractmethod
    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "ForecastModel":
        """
        warm_train 데이터와 추가 feature로 모델을 학습한다.

        Args:
            warm_train: cold-start 아이템이 제외된 학습 데이터프레임.
                        컬럼: item_id, store_id, cat_id, dept_id, date, sales 등.
            features: 모델별 추가 데이터.
                      key 컨벤션:
                        "prices"      → sell_prices DataFrame
                        "calendar"    → calendar DataFrame
                        "embeddings"  → 아이템 임베딩 (k-NN용)
                        "cold_stats"  → cold 아이템 메타정보
                        "llm_scores"  → LLM 구매 의향 점수 (persona용)
            **kwargs: 모델별 추가 파라미터.
                      예) LightGBM: n_estimators=500, early_stopping_rounds=50

        Returns:
            self (메서드 체이닝 지원).
        """

    @abstractmethod
    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        cold_test 각 행에 대한 판매량 예측값을 반환한다.

        Args:
            cold_test: cold-start 아이템의 테스트 데이터프레임.
                       컬럼: item_id, store_id, cat_id, dept_id, date 등.
                       `sales` 컬럼은 평가용이며 모델에 사용하면 안 됨.
            features: fit()과 동일한 key 컨벤션.
                      예측 시점에 필요한 추가 데이터 (calendar, 가격 등).
            **kwargs: 예측 시점 파라미터.

        Returns:
            pd.DataFrame: 반드시 아래 컬럼을 포함해야 함.
                - item_id  (str)
                - store_id (str)
                - date     (datetime)
                - pred_sales (float)
                추가 컬럼(pred_lower, pred_upper, confidence 등)은 선택.
        """

    # ------------------------------------------------------------------
    # 공통 유틸리티 (서브클래스에서 super() 호출 가능)
    # ------------------------------------------------------------------

    def _validate_predict_output(self, pred: pd.DataFrame) -> None:
        """predict() 반환 DataFrame의 필수 컬럼을 검증한다."""
        missing = PRED_REQUIRED_COLS - set(pred.columns)
        if missing:
            raise ValueError(
                f"[{self.name}] predict() output is missing columns: {missing}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
