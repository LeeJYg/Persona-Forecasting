"""Track A: GPT-4o-mini를 이용한 Naive 페르소나 예측 모델.

설계 원칙
----------
1. 50 personas × 10 item batches × 17 weeks = 8,500 API calls (cold 100 items 기준)
2. 각 call: 1 persona × 10 items × 1 week → 개인 구매량 예측
3. 최종 예측 = 50명 합산 (50-persona scale)
4. 스케일 보정(post-hoc): mean(baseline_pred) / mean(track_a_raw)로 매장 수준으로 환산
5. 체크포인트: persona × item_batch × week 단위로 저장 → 재시작(resume) 지원
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import openai
import pandas as pd

from src.models.baselines.base import ForecastModel
from src.models.forecasting.prompt_builder import ItemInfo, PromptBuilder, WeekContext
from src.models.persona.schema import Persona

logger = logging.getLogger(__name__)

# predict() 출력에 반드시 포함되어야 하는 컬럼
_PRED_COLS = {"item_id", "store_id", "date", "pred_sales"}


class PersonaPredictor(ForecastModel):
    """페르소나 기반 cold-start 수요 예측 모델 (Track A Naive).

    Args:
        config: load_config()로 로드한 DotDict 설정 객체.
        personas: 예측에 사용할 Persona 리스트.
        prompt_builder: PromptBuilder 인스턴스.
        checkpoint_path: 중간 결과를 저장할 JSON 파일 경로.
                         None이면 체크포인트 미사용.
        dry_run: True이면 LLM을 실제 호출하지 않고 더미 예측(0)을 반환.
    """

    def __init__(
        self,
        config: Any,
        personas: list[Persona],
        prompt_builder: PromptBuilder,
        checkpoint_path: Path | None = None,
        dry_run: bool = False,
    ) -> None:
        import os
        self._config = config
        self._personas = personas
        self._pb = prompt_builder
        self._checkpoint_path = checkpoint_path
        self._dry_run = dry_run

        # LLM 설정
        llm_cfg = config.experiment.llm
        model_name = getattr(llm_cfg, "prediction_model", "gpt-4o-mini")
        self._model: str = model_name
        self._temperature: float = float(llm_cfg.prediction_temperature)
        self._max_retries: int = int(llm_cfg.max_retries)

        # Track A 설정
        ta_cfg = config.experiment.track_a
        self._item_batch_size: int = int(ta_cfg.item_batch_size)

        # OpenAI 클라이언트
        if not dry_run:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."
                )
            import httpx
            self._client = openai.OpenAI(
                api_key=api_key,
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
            )
        else:
            self._client = None  # type: ignore[assignment]

        # 체크포인트 로드
        # 신규 포맷: {"completed": {pid: {iid: [w...]}}, "partial": {"persona_id": ..., "last_batch": int, "data": {iid: [w...]}}}
        # 구버전 포맷(하위 호환): {pid: {iid: [w...]}}  → completed 로 자동 변환
        self._checkpoint: dict[str, dict[str, list[float]]] = {}   # completed personas
        self._partial: dict = {}  # 현재 진행 중인 페르소나 부분 결과
        if checkpoint_path and checkpoint_path.exists():
            raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if "completed" in raw:
                # 신규 포맷
                self._checkpoint = raw.get("completed", {})
                self._partial = raw.get("partial", {})
            else:
                # 구버전 포맷: 값이 dict[str, list]인 항목만 completed로 처리
                self._checkpoint = {
                    k: v for k, v in raw.items()
                    if isinstance(v, dict)
                }
            logger.info(
                "체크포인트 로드: 완료 %d 페르소나, 부분 완료: %s (%s)",
                len(self._checkpoint),
                self._partial.get("persona_id", "없음"),
                checkpoint_path,
            )

        # fit()에서 채워질 warm 통계 (스케일 보정용)
        self._warm_stats: dict[str, float] = {}

        logger.info(
            "PersonaPredictor 초기화: model=%s, personas=%d, batch=%d, dry_run=%s",
            self._model, len(personas), self._item_batch_size, dry_run,
        )

    # ------------------------------------------------------------------ #
    # ForecastModel 인터페이스                                             #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "track_a_persona_naive"

    def fit(
        self,
        warm_train: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> "PersonaPredictor":
        """warm 통계를 기록해 스케일 보정에 활용한다 (실제 학습 없음).

        Args:
            warm_train: warm 아이템 학습 데이터 (스케일 참고용).
        """
        # warm 아이템 카테고리별 일평균 판매량 (스케일 보정 참고)
        cat_stats = (
            warm_train.groupby("cat_id")["sales"]
            .mean()
            .to_dict()
        )
        self._warm_stats = {k: float(v) for k, v in cat_stats.items()}
        logger.info("fit() warm 카테고리 평균 일판매량: %s", self._warm_stats)
        return self

    def predict(
        self,
        cold_test: pd.DataFrame,
        features: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """cold_test 각 행에 대한 예측값 DataFrame을 반환한다.

        내부적으로:
        1. cold_test 날짜를 주 단위로 그룹화
        2. 각 주 × 페르소나 × 아이템 배치 → LLM 호출
        3. 50명 합산 → 주별 예측량
        4. 주별 예측량을 일별로 분배 (/ n_days_in_week)
        5. [item_id, store_id, date, pred_sales] DataFrame 반환

        Args:
            cold_test: cold-start 테스트 데이터프레임.
                       컬럼: item_id, store_id, cat_id, date, (d, sales, ...).
            features: 사용하지 않음 (인터페이스 호환용).

        Returns:
            pd.DataFrame with columns: item_id, store_id, date, pred_sales, cat_id
        """
        cold_test = cold_test.copy()
        cold_test["date"] = pd.to_datetime(cold_test["date"])

        item_ids = sorted(cold_test["item_id"].unique().tolist())
        store_id = str(cold_test["store_id"].iloc[0])
        n_items = len(item_ids)
        logger.info("예측 시작: %d 아이템, store=%s", n_items, store_id)

        # 주별 날짜 그룹 (week_num → [date 리스트])
        week_contexts = self._pb.build_week_contexts(cold_test)
        n_weeks = len(week_contexts)

        # 아이템 가격 정보 (첫 번째 예측 주의 wm_yr_wk 기준)
        first_wm_yr_wk = self._get_wm_yr_wk_for_week(cold_test, week_contexts[0])
        item_infos_map: dict[str, ItemInfo] = {
            info.item_id: info
            for info in self._pb.get_item_info(
                item_ids, cold_test, forecast_wm_yr_wk=first_wm_yr_wk
            )
        }

        # LLM 예측 실행: {item_id: [weekly_qty × n_weeks]} (50명 합산)
        weekly_sums = self._run_prediction_loop(
            item_ids, item_infos_map, week_contexts, store_id
        )

        # 주별 예측 → 일별 DataFrame 변환
        pred_rows: list[dict[str, Any]] = []
        for item_id in item_ids:
            item_weekly = weekly_sums.get(item_id, [0.0] * n_weeks)
            cat_id = str(
                cold_test.loc[cold_test["item_id"] == item_id, "cat_id"].iloc[0]
            )
            # cold_test에서 해당 아이템의 날짜 목록 추출
            item_dates = (
                cold_test[cold_test["item_id"] == item_id]
                .sort_values("date")["date"]
                .tolist()
            )
            for date in item_dates:
                # 날짜가 속하는 주 번호 계산
                wk_idx = self._date_to_week_idx(date, week_contexts)
                weekly_qty = item_weekly[wk_idx] if wk_idx < len(item_weekly) else 0.0
                # 주 내 일수로 나눠 일별 분배
                n_days = self._n_days_in_week(wk_idx, week_contexts)
                daily_pred = weekly_qty / max(n_days, 1)
                pred_rows.append({
                    "item_id": item_id,
                    "store_id": store_id,
                    "date": date,
                    "pred_sales": daily_pred,
                    "cat_id": cat_id,
                })

        pred_df = pd.DataFrame(pred_rows)
        self._validate_predict_output(pred_df)
        logger.info(
            "예측 완료: %d rows, mean_daily_pred=%.4f",
            len(pred_df),
            pred_df["pred_sales"].mean(),
        )
        return pred_df

    # ------------------------------------------------------------------ #
    # 예측 루프                                                            #
    # ------------------------------------------------------------------ #

    def _run_prediction_loop(
        self,
        item_ids: list[str],
        item_infos_map: dict[str, ItemInfo],
        week_contexts: list[WeekContext],
        store_id: str,
    ) -> dict[str, list[float]]:
        """50 personas × item batches × weeks를 순회하며 예측을 수집한다.

        체크포인트에 이미 완료된 (persona_id, item_batch, week) 조합을 건너뜀.

        Returns:
            {item_id: [weekly_sum_week1, ..., weekly_sum_weekN]}
            (50명 합산 주별 예측량)
        """
        n_weeks = len(week_contexts)

        # 누적 합산 배열 초기화: {item_id: [0.0 × n_weeks]}
        weekly_sums: dict[str, list[float]] = {
            iid: [0.0] * n_weeks for iid in item_ids
        }

        # 체크포인트에서 이미 완료된 페르소나 기여분 반영
        for pid, item_preds in self._checkpoint.items():
            for iid, weekly_list in item_preds.items():
                if iid in weekly_sums:
                    for w, qty in enumerate(weekly_list):
                        if w < n_weeks:
                            weekly_sums[iid][w] += qty

        completed_personas = set(self._checkpoint.keys())
        n_batches = len(item_ids) // self._item_batch_size + (1 if len(item_ids) % self._item_batch_size else 0)
        total_calls = len(self._personas) * n_batches * n_weeks
        call_count = 0

        for persona in self._personas:
            pid = persona.persona_id
            if pid in completed_personas:
                logger.info("스킵 (체크포인트): %s", pid)
                continue

            persona_preds: dict[str, list[float]] = {iid: [0.0] * n_weeks for iid in item_ids}

            # 부분 완료 복원: 이전 실행에서 같은 페르소나를 중간까지 진행했으면 이어받기
            resume_from_batch = 0
            if self._partial.get("persona_id") == pid:
                saved_data: dict[str, list[float]] = self._partial.get("data", {})
                for iid, wlist in saved_data.items():
                    if iid in persona_preds:
                        persona_preds[iid] = list(wlist)
                        # weekly_sums에도 이미 저장된 값 반영
                        for w, qty in enumerate(wlist):
                            if w < n_weeks:
                                weekly_sums[iid][w] += qty
                resume_from_batch = self._partial.get("last_batch", 0) + self._item_batch_size
                logger.info(
                    "부분 복원: %s — batch %d까지 완료, batch %d부터 재개",
                    pid, resume_from_batch - self._item_batch_size, resume_from_batch,
                )

            # 아이템 배치 단위 순회
            for batch_start in range(0, len(item_ids), self._item_batch_size):
                if batch_start < resume_from_batch:
                    continue  # 이미 완료된 배치 건너뜀

                batch_ids = item_ids[batch_start: batch_start + self._item_batch_size]
                batch_infos = [item_infos_map[iid] for iid in batch_ids]

                for week_ctx in week_contexts:
                    wk_idx = week_ctx.week_num - 1
                    call_count += 1

                    try:
                        weekly_preds = self._call_llm_single(
                            persona, batch_infos, week_ctx
                        )
                    except Exception as e:
                        logger.error(
                            "LLM 호출 실패 (persona=%s, batch=%d~%d, week=%d): %s. "
                            "해당 주 예측 = 0으로 처리.",
                            pid,
                            batch_start,
                            batch_start + len(batch_ids),
                            week_ctx.week_num,
                            e,
                        )
                        weekly_preds = {iid: 0 for iid in batch_ids}

                    for iid, qty in weekly_preds.items():
                        persona_preds[iid][wk_idx] = float(qty)
                        weekly_sums[iid][wk_idx] += float(qty)

                    if call_count % 50 == 0:
                        logger.info(
                            "진행: %d / %d calls (%.1f%%)",
                            call_count, total_calls, 100 * call_count / total_calls,
                        )

                # 배치 완료 → 부분 체크포인트 저장 (페르소나 내 중간 저장)
                self._partial = {
                    "persona_id": pid,
                    "last_batch": batch_start,
                    "data": persona_preds,
                }
                self._save_checkpoint()

            # 페르소나 완료 → completed로 이동, partial 초기화
            self._checkpoint[pid] = persona_preds
            self._partial = {}
            self._save_checkpoint()
            logger.info("페르소나 완료: %s (%d/%d)", pid, len(self._checkpoint), len(self._personas))

        return weekly_sums

    # ------------------------------------------------------------------ #
    # LLM 호출 (단일 call)                                                #
    # ------------------------------------------------------------------ #

    def _call_llm_single(
        self,
        persona: Persona,
        item_infos: list[ItemInfo],
        week_ctx: WeekContext,
    ) -> dict[str, int]:
        """LLM API를 한 번 호출해 아이템별 주간 구매량을 반환한다.

        Args:
            persona: 예측 주체 페르소나.
            item_infos: 이번 배치의 아이템 목록.
            week_ctx: 이번 주 달력 컨텍스트.

        Returns:
            {item_id: quantity} 딕셔너리.
        """
        if self._dry_run:
            # 드라이런: 0 예측 반환
            return {info.item_id: 0 for info in item_infos}

        system_prompt, user_prompt = self._pb.build_prediction_prompt(
            persona, item_infos, week_ctx
        )

        # api_attempt: 실제 API 오류 횟수 (max_retries 적용 대상)
        # net_retry:   네트워크/timeout 오류 횟수 (무제한 재시도, 0 대체 금지)
        api_attempt = 0
        net_retry = 0
        while api_attempt < self._max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content or "{}"
                raw = json.loads(content)
                predictions = raw.get("predictions", {})
                result: dict[str, int] = {}
                for info in item_infos:
                    val = predictions.get(info.item_id, 0)
                    try:
                        qty = max(0, int(round(float(val))))
                    except (TypeError, ValueError):
                        qty = 0
                    result[info.item_id] = qty
                return result

            except json.JSONDecodeError as e:
                api_attempt += 1
                logger.warning("JSON 파싱 실패 (%d/%d): %s", api_attempt, self._max_retries, e)

            except openai.RateLimitError:
                # Rate limit은 max_retries 차감 없이 대기 후 재시도
                wait = min(2 ** (net_retry + 1), 60)
                logger.warning("Rate limit. %d초 후 재시도...", wait)
                time.sleep(wait)

            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                # 네트워크 단절 / timeout: 복구될 때까지 무제한 대기 (데이터 0 오염 방지)
                net_retry += 1
                wait = min(30 * net_retry, 300)  # 30 → 60 → 90 → ... → 최대 5분
                logger.warning(
                    "네트워크 오류 #%d: %s — %d초 후 재시도 (연결 복구 대기중)...",
                    net_retry, type(e).__name__, wait,
                )
                time.sleep(wait)

            except openai.OpenAIError as e:
                api_attempt += 1
                logger.error("OpenAI 오류 (%d/%d): %s", api_attempt, self._max_retries, e)
                if api_attempt < self._max_retries:
                    time.sleep(2)

        logger.error("최대 재시도 초과 → 해당 배치 예측 = 0 처리")
        return {info.item_id: 0 for info in item_infos}

    # ------------------------------------------------------------------ #
    # 스케일 보정                                                          #
    # ------------------------------------------------------------------ #

    def compute_scale_factor(
        self,
        track_a_pred: pd.DataFrame,
        baseline_pred: pd.DataFrame,
    ) -> float:
        """post-hoc 스케일 보정 계수 alpha를 계산한다.

        alpha = mean(baseline_pred_sales) / mean(track_a_pred_sales)
        track_a_calibrated = track_a_pred_sales × alpha  → 매장 수준으로 환산

        Args:
            track_a_pred: Track A 예측 DataFrame (pred_sales 컬럼).
            baseline_pred: 베이스라인 예측 DataFrame (pred_sales 컬럼).

        Returns:
            float: 스케일 보정 계수 (> 1 예상, 50 personas < 전체 고객).
        """
        mean_ta = float(track_a_pred["pred_sales"].mean())
        mean_bl = float(baseline_pred["pred_sales"].mean())
        if mean_ta == 0:
            logger.warning("Track A 평균 예측이 0 → alpha=1.0 반환")
            return 1.0
        alpha = mean_bl / mean_ta
        logger.info(
            "스케일 보정: mean_track_a=%.4f, mean_baseline=%.4f, alpha=%.2f",
            mean_ta, mean_bl, alpha,
        )
        return alpha

    def apply_scale_factor(
        self,
        pred: pd.DataFrame,
        alpha: float,
    ) -> pd.DataFrame:
        """pred_sales에 alpha를 곱해 매장 수준으로 환산한 DataFrame을 반환한다."""
        out = pred.copy()
        out["pred_sales"] = out["pred_sales"] * alpha
        out["scale_factor"] = alpha
        return out

    # ------------------------------------------------------------------ #
    # 체크포인트                                                           #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self) -> None:
        """현재 체크포인트를 JSON 파일에 저장한다.

        신규 포맷:
            {
                "completed": {persona_id: {item_id: [weekly_qty...]}},
                "partial":   {"persona_id": ..., "last_batch": int, "data": {item_id: [weekly_qty...]}}
            }
        배치 완료마다 호출되므로 중간 충돌 시 손실 최소화.
        """
        if self._checkpoint_path is None:
            return
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed": self._checkpoint,
            "partial": self._partial,
        }
        self._checkpoint_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_raw_predictions_from_checkpoint(
        self,
        item_ids: list[str],
        n_weeks: int,
    ) -> dict[str, list[float]]:
        """체크포인트에서 50명 합산 주별 예측을 재구성한다.

        run_predict_loop()를 다시 실행하지 않고 체크포인트만으로 결과를 얻는다.
        completed 페르소나만 포함 (partial 제외).
        """
        weekly_sums: dict[str, list[float]] = {
            iid: [0.0] * n_weeks for iid in item_ids
        }
        for pid, item_preds in self._checkpoint.items():
            for iid, weekly_list in item_preds.items():
                if iid in weekly_sums:
                    for w, qty in enumerate(weekly_list):
                        if w < n_weeks:
                            weekly_sums[iid][w] += qty
        return weekly_sums

    # ------------------------------------------------------------------ #
    # 내부 유틸                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _date_to_week_idx(
        date: pd.Timestamp,
        week_contexts: list[WeekContext],
    ) -> int:
        """날짜가 속하는 주의 인덱스(0-based)를 반환한다."""
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
        for ctx in week_contexts:
            if ctx.date_start <= date_str <= ctx.date_end:
                return ctx.week_num - 1
        # 범위 초과면 마지막 주로 처리
        return len(week_contexts) - 1

    @staticmethod
    def _n_days_in_week(wk_idx: int, week_contexts: list[WeekContext]) -> int:
        """주 인덱스에 해당하는 날수를 반환한다."""
        ctx = week_contexts[wk_idx]
        start = pd.Timestamp(ctx.date_start)
        end = pd.Timestamp(ctx.date_end)
        return (end - start).days + 1

    @staticmethod
    def _get_wm_yr_wk_for_week(
        cold_test: pd.DataFrame,
        week_ctx: WeekContext,
    ) -> int | None:
        """cold_test에 wm_yr_wk 컬럼이 있으면 해당 주의 값을 반환한다."""
        if "wm_yr_wk" not in cold_test.columns:
            return None
        mask = cold_test["date"].dt.strftime("%Y-%m-%d") >= week_ctx.date_start
        subset = cold_test[mask]
        return int(subset["wm_yr_wk"].iloc[0]) if not subset.empty else None
