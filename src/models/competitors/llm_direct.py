"""Competitor 3: LLM Direct Prediction (LLMTime-style, 3 variants).

3-1 (zero_shot):   상품 메타데이터만 제공.
3-2 (similar_item): k-NN top-3 유사 warm item 이력 제공.
3-3 (aggregate):   50명 구매자 행동 종합 시뮬레이션 요청.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines.base import ForecastModel

logger = logging.getLogger(__name__)

_N_WEEKS = 17
_MAX_RETRIES = 3


class LLMDirect(ForecastModel):
    """LLM Direct Prediction.

    Args:
        variant: "zero_shot" | "similar_item" | "aggregate"
        model: OpenAI 모델명 (기본 "gpt-4o-mini")
        temperature: 생성 temperature (기본 0.0)
        max_tokens: 최대 출력 토큰 (기본 400)
        knn_neighbors_path: similar_item 사용 시 k-NN 이웃 JSON 파일 경로.
        checkpoint_dir: item별 결과 저장 디렉토리 (resume 지원).
    """

    def __init__(
        self,
        variant: str = "zero_shot",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 400,
        knn_neighbors_path: Path | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        if variant not in ("zero_shot", "similar_item", "aggregate"):
            raise ValueError(f"variant must be 'zero_shot', 'similar_item', or 'aggregate', got '{variant}'")
        self._variant = variant
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._knn_neighbors_path = knn_neighbors_path
        self._checkpoint_dir = checkpoint_dir
        self._knn_neighbors: dict[str, list[dict]] = {}
        self._warm_iso_week_mean: dict[str, dict[int, float]] = {}  # {item_id: {iso_week: mean_sales}}
        self._client: Any = None
        self._checkpoint: dict[str, list[float]] = {}  # {item_id: [17 weekly sales]}

    @property
    def name(self) -> str:
        return f"llm_{self._variant}"

    def _load_checkpoint(self) -> None:
        if self._checkpoint_dir is None:
            return
        cp_path = self._checkpoint_dir / "checkpoint.json"
        if cp_path.exists():
            self._checkpoint = json.loads(cp_path.read_text(encoding="utf-8"))
            logger.info("[%s] checkpoint 로드: %d items 완료", self.name, len(self._checkpoint))

    def _save_checkpoint(self) -> None:
        if self._checkpoint_dir is None:
            return
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self._checkpoint_dir / "checkpoint.json").write_text(
            json.dumps(self._checkpoint, ensure_ascii=False), encoding="utf-8"
        )

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "LLMDirect":
        """warm 데이터로 참조 이력 준비 (similar_item용) + OpenAI 클라이언트 초기화."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        import httpx
        import openai
        self._client = openai.OpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
        )

        if self._variant == "similar_item":
            if self._knn_neighbors_path is None or not self._knn_neighbors_path.exists():
                raise ValueError(
                    f"[{self.name}] similar_item requires knn_neighbors_path. "
                    f"Run knn_analog first. Path: {self._knn_neighbors_path}"
                )
            self._knn_neighbors = json.loads(
                self._knn_neighbors_path.read_text(encoding="utf-8")
            )
            # warm items의 ISO-week 판매 이력 (상위 3 이웃에게 보여줄 용도)
            for item_id, grp in warm_train.groupby("item_id"):
                self._warm_iso_week_mean[item_id] = (
                    grp.groupby("iso_week")["sales"].mean().to_dict()
                )

        self._load_checkpoint()
        logger.info("[%s] fit 완료 (variant=%s)", self.name, self._variant)
        return self

    def _build_prompt(
        self,
        item_id: str,
        cat_id: str,
        dept_id: str,
        sell_price: float,
        n_weeks: int = _N_WEEKS,
    ) -> str:
        """variant별 user prompt 생성."""
        attr_line = f"category={cat_id}, dept={dept_id}, store=CA_1, price=${sell_price:.2f}"

        if self._variant == "zero_shot":
            return (
                f"A new product is launching with no prior sales history.\n"
                f"Product attributes: {attr_line}\n\n"
                f"Predict the weekly sales for this product over the next {n_weeks} weeks.\n"
                f"Return ONLY a JSON array of {n_weeks} numbers (weekly unit sales). "
                f"Example: [3, 5, 2, 4, 7, 3, 5, 2, 4, 6, 3, 5, 2, 4, 7, 3, 5]"
            )

        elif self._variant == "similar_item":
            neighbors = self._knn_neighbors.get(item_id, [])[:3]
            lines = []
            for nb in neighbors:
                nb_id = nb["item_id"]
                weeks_str = ", ".join(
                    str(round(self._warm_iso_week_mean.get(nb_id, {}).get(w, 0), 1))
                    for w in range(1, n_weeks + 1)
                )
                lines.append(f"  - {nb_id}: [{weeks_str}]")
            history_block = "\n".join(lines) if lines else "  (no similar items found)"
            return (
                f"Similar products' recent {n_weeks}-week sales history:\n"
                f"{history_block}\n\n"
                f"New product attributes: {attr_line}\n\n"
                f"Based on the similar products above, predict the weekly sales for the new product "
                f"over the next {n_weeks} weeks.\n"
                f"Return ONLY a JSON array of {n_weeks} numbers (weekly unit sales)."
            )

        else:  # aggregate
            return (
                f"A new product is launching in a large retail store (CA_1, Walmart-scale).\n"
                f"Product attributes: {attr_line}\n\n"
                f"Simulate the aggregate purchasing behavior of approximately 50 typical shoppers "
                f"who visit this store regularly. Based on their collective purchasing decisions, "
                f"predict the total weekly store-level sales for this product over the next {n_weeks} weeks.\n\n"
                f"Consider: shopping frequency, price sensitivity, category preferences, seasonal patterns.\n"
                f"Return ONLY a JSON array of {n_weeks} numbers (total weekly unit sales at store level)."
            )

    def _call_api(self, system_prompt: str, user_prompt: str) -> list[float] | None:
        """API 호출 + JSON 파싱. 실패 시 None 반환."""
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content.strip()
                # JSON array 추출
                match = re.search(r"\[[\d\s.,\-]+\]", content)
                if match:
                    arr = json.loads(match.group())
                    if isinstance(arr, list) and len(arr) == _N_WEEKS:
                        return [max(float(x), 0.0) for x in arr]
                logger.warning("[%s] attempt %d: JSON 파싱 실패: %s", self.name, attempt + 1, content[:200])
            except Exception as e:
                import openai
                if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
                    wait = min(30 * (attempt + 1), 120)
                    logger.warning("[%s] 네트워크 오류: %s — %ds 후 재시도", self.name, type(e).__name__, wait)
                    time.sleep(wait)
                else:
                    logger.warning("[%s] attempt %d API 오류: %s", self.name, attempt + 1, e)
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(2)
        return None

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """각 cold item에 대해 LLM 호출로 17-week 판매량 예측."""
        if self._client is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출해야 합니다.")
        if features is None or "prices" not in features:
            raise ValueError(f"[{self.name}] predict() requires features['prices']")

        prices = features["prices"]
        item_price = prices.groupby("item_id")["sell_price"].mean()

        cold_items = cold_test[["item_id", "cat_id", "dept_id"]].drop_duplicates("item_id")
        # cold_test의 (item, week) 순서 매핑
        iso_weeks_per_item = (
            cold_test.sort_values(["item_id", "iso_year", "iso_week"])
            .groupby("item_id")[["iso_year", "iso_week"]]
            .apply(lambda g: g.to_dict(orient="records"))
            .to_dict()
        )

        system_prompt = (
            "You are a demand forecasting expert for a large retail store. "
            "Your task is to predict weekly sales for new products with no prior sales history."
        )

        parse_failures = 0
        total_items = len(cold_items)

        preds: list[dict] = []
        for idx, row in cold_items.iterrows():
            item_id = row["item_id"]

            # resume: 이미 완료된 item skip
            if item_id in self._checkpoint:
                weekly_preds = self._checkpoint[item_id]
                logger.debug("[%s] SKIP (resume): %s", self.name, item_id)
            else:
                price = float(item_price.get(item_id, prices["sell_price"].median()))
                user_prompt = self._build_prompt(item_id, row["cat_id"], row["dept_id"], price)
                result = self._call_api(system_prompt, user_prompt)

                done = len(self._checkpoint)
                logger.info("[%s] %d / %d — %s", self.name, done + 1, total_items, item_id)

                if result is None:
                    # fallback: cat 평균 (이미 checkpoint에 저장된 유사 아이템 평균)
                    parse_failures += 1
                    cat_avg = prices[prices["item_id"].isin(
                        cold_test[cold_test["cat_id"] == row["cat_id"]]["item_id"]
                    )]["sell_price"].mean()
                    result = [max(cat_avg * 0.3, 0.5)] * _N_WEEKS
                    logger.warning("[%s] %s: fallback 적용 (cat mean)", self.name, item_id)

                weekly_preds = result
                self._checkpoint[item_id] = weekly_preds
                self._save_checkpoint()

            # week 매핑
            weeks = iso_weeks_per_item.get(item_id, [])
            for w_idx, week_info in enumerate(weeks):
                pred_val = weekly_preds[w_idx] if w_idx < len(weekly_preds) else 0.0
                preds.append({
                    "item_id": item_id,
                    "iso_year": week_info["iso_year"],
                    "iso_week": week_info["iso_week"],
                    "pred_sales": max(pred_val, 0.0),
                })

        result_df = cold_test[["item_id", "store_id", "cat_id", "iso_year", "iso_week", "date"]].merge(
            pd.DataFrame(preds)[["item_id", "iso_year", "iso_week", "pred_sales"]],
            on=["item_id", "iso_year", "iso_week"],
            how="left",
        )
        result_df["pred_sales"] = result_df["pred_sales"].fillna(0.0)

        logger.info(
            "[%s] 예측 완료: total=%d, parse_failures=%d (%.1f%%)",
            self.name, total_items, parse_failures, parse_failures / total_items * 100,
        )
        self._validate_predict_output(result_df)
        return result_df
