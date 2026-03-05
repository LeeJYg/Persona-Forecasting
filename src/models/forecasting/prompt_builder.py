"""Track A 예측 프롬프트 빌더.

M5 캘린더와 판매 가격 데이터를 활용해 주별 예측 프롬프트를 조립한다.
Condition A (Structured Only): persona description 제외, 구조화 필드만 사용.
"""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.models.persona.schema import Persona

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 시스템 프롬프트 (고정, Condition A)                                           #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_CONDITION_A = textwrap.dedent("""\
    You are a synthetic Walmart customer at Store CA_1 in California.
    Based solely on your shopping profile, predict how many units of each listed
    item you personally would purchase during the given week.

    Rules:
    - Predictions must be non-negative integers (0, 1, 2, ...).
    - Reflect your budget, visit frequency, and category preferences.
    - SNAP eligibility increases your food purchasing power on SNAP issuance days.
    - Higher price sensitivity means fewer purchases of expensive items.
    - Return ONLY the JSON object — no explanations or extra text.
""")

# --------------------------------------------------------------------------- #
# 데이터 클래스                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class WeekContext:
    """한 주의 달력 컨텍스트."""

    week_num: int           # 1-based (1..17)
    total_weeks: int
    date_start: str         # "2016-01-01"
    date_end: str           # "2016-01-07"
    month: int
    month_name: str
    season: str
    snap_days: int          # 해당 주 CA SNAP 지급일 수
    events: list[str]       # 이벤트 이름 목록


@dataclass
class ItemInfo:
    """프롬프트에 포함할 아이템 정보."""

    item_id: str
    cat_id: str
    dept_id: str
    avg_price: float | None     # USD, None이면 "N/A" 표시


# --------------------------------------------------------------------------- #
# 프롬프트 빌더                                                                 #
# --------------------------------------------------------------------------- #

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

_SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
    5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter",
}


class PromptBuilder:
    """주별 예측 프롬프트를 조립하는 클래스.

    Args:
        calendar_df: M5 calendar.csv 로드 결과.
        sell_prices_df: M5 sell_prices.csv 로드 결과.
        store_id: 대상 매장 ID (SNAP 컬럼 선택에 사용).
        condition: "A" (Structured Only), "B", "C".
    """

    def __init__(
        self,
        calendar_df: pd.DataFrame,
        sell_prices_df: pd.DataFrame,
        store_id: str = "CA_1",
        condition: str = "A",
    ) -> None:
        self._calendar = calendar_df.copy()
        self._calendar["date"] = pd.to_datetime(self._calendar["date"])
        self._prices = sell_prices_df
        self._store_id = store_id
        self._state = store_id.split("_")[0]       # "CA"
        self._snap_col = f"snap_{self._state}"     # "snap_CA"
        self._condition = condition

        # item_id → 최근 13주 평균 가격 (CA_1 매장 기준, 미리 계산)
        self._price_cache: dict[str, float] = {}

        if condition != "A":
            logger.warning(
                "PromptBuilder: condition=%s. "
                "Phase 1에서는 Condition A만 지원. 계속 진행하지만 결과를 확인하세요.",
                condition,
            )

    # ------------------------------------------------------------------ #
    # 공개 API                                                             #
    # ------------------------------------------------------------------ #

    def build_week_contexts(
        self,
        cold_test: pd.DataFrame,
    ) -> list[WeekContext]:
        """cold_test DataFrame에서 주별 컨텍스트 목록을 생성한다.

        cold_test는 (item_id, date, sales) 포함. 아이템마다 동일 날짜 범위를
        가지므로 첫 번째 아이템의 날짜만 사용해도 무방.

        Args:
            cold_test: cold-start 테스트 데이터프레임.

        Returns:
            WeekContext 리스트 (week_num 오름차순).
        """
        dates = (
            cold_test[["date"]]
            .drop_duplicates()
            .copy()
        )
        dates["date"] = pd.to_datetime(dates["date"])
        dates = dates.sort_values("date").reset_index(drop=True)

        # 7일 단위로 주 번호 부여 (남은 날짜는 마지막 주에 포함)
        dates["week_num"] = (dates.index // 7) + 1
        total_weeks = dates["week_num"].max()

        contexts: list[WeekContext] = []
        for wk, grp in dates.groupby("week_num"):
            d_start = grp["date"].min()
            d_end = grp["date"].max()
            cal_slice = self._calendar[
                (self._calendar["date"] >= d_start)
                & (self._calendar["date"] <= d_end)
            ]
            snap_days = int(cal_slice[self._snap_col].sum()) if self._snap_col in cal_slice else 0
            events = (
                cal_slice["event_name_1"].dropna().tolist()
                + cal_slice["event_name_2"].dropna().tolist()
            )
            events = [e for e in events if e]

            month = int(d_start.month)
            contexts.append(WeekContext(
                week_num=int(wk),
                total_weeks=int(total_weeks),
                date_start=d_start.strftime("%Y-%m-%d"),
                date_end=d_end.strftime("%Y-%m-%d"),
                month=month,
                month_name=_MONTH_NAMES[month],
                season=_SEASON_MAP[month],
                snap_days=snap_days,
                events=events,
            ))

        logger.info(
            "WeekContexts 생성 완료: %d주 (%s ~ %s)",
            len(contexts),
            contexts[0].date_start if contexts else "?",
            contexts[-1].date_end if contexts else "?",
        )
        return contexts

    def get_item_info(
        self,
        item_ids: list[str],
        cold_test: pd.DataFrame,
        forecast_wm_yr_wk: int | None = None,
    ) -> list[ItemInfo]:
        """아이템 ID 목록에 대한 ItemInfo를 생성한다.

        Args:
            item_ids: 조회할 아이템 ID 목록.
            cold_test: cat_id, dept_id 조회용 DataFrame.
            forecast_wm_yr_wk: 예측 시점 wm_yr_wk (가격 조회 기준).
                               None이면 가격을 "N/A"로 표시.

        Returns:
            ItemInfo 리스트.
        """
        meta = (
            cold_test[["item_id", "cat_id", "dept_id"]]
            .drop_duplicates("item_id")
            .set_index("item_id")
        )
        result: list[ItemInfo] = []
        for iid in item_ids:
            row = meta.loc[iid] if iid in meta.index else None
            avg_price = self._get_avg_price(iid, forecast_wm_yr_wk)
            result.append(ItemInfo(
                item_id=iid,
                cat_id=str(row["cat_id"]) if row is not None else "UNKNOWN",
                dept_id=str(row["dept_id"]) if row is not None else "UNKNOWN",
                avg_price=avg_price,
            ))
        return result

    def build_prediction_prompt(
        self,
        persona: Persona,
        item_infos: list[ItemInfo],
        week_ctx: WeekContext,
    ) -> tuple[str, str]:
        """(system_prompt, user_prompt) 튜플을 반환한다.

        Args:
            persona: 예측에 사용할 페르소나.
            item_infos: 이번 배치의 아이템 목록 (최대 10개).
            week_ctx: 예측 주의 달력 컨텍스트.

        Returns:
            (system_prompt, user_prompt)
        """
        system_prompt = SYSTEM_PROMPT_CONDITION_A  # Condition A 고정
        user_prompt = self._build_user_prompt(persona, item_infos, week_ctx)
        return system_prompt, user_prompt

    # ------------------------------------------------------------------ #
    # 내부 헬퍼                                                            #
    # ------------------------------------------------------------------ #

    def _build_user_prompt(
        self,
        persona: Persona,
        item_infos: list[ItemInfo],
        week_ctx: WeekContext,
    ) -> str:
        p = persona.profile

        # 카테고리 선호도 텍스트
        cat_pref_str = " | ".join(
            f"{cat} {pct*100:.0f}%"
            for cat, pct in sorted(p.category_preference.items(), key=lambda x: -x[1])
        )

        # 주간 컨텍스트 텍스트
        event_text = ", ".join(week_ctx.events) if week_ctx.events else "None"
        snap_text = (
            f"SNAP issuance day(s) this week: {week_ctx.snap_days} day(s)"
            if week_ctx.snap_days > 0
            else "No SNAP issuance this week"
        )

        # 아이템 테이블
        rows = []
        for info in item_infos:
            price_str = f"${info.avg_price:.2f}" if info.avg_price is not None else "N/A"
            rows.append(
                f"| {info.item_id:<22} | {info.cat_id:<9} | {info.dept_id:<11} | {price_str:>7} |"
            )
        item_table = (
            "| Item ID                | Category  | Department  | Avg Price |\n"
            "|------------------------|-----------|-------------|----------|\n"
            + "\n".join(rows)
        )

        # JSON 출력 형식 (빈 예시 포함)
        json_keys = "\n".join(
            f'    "{info.item_id}": <int>,' for info in item_infos
        )

        lines = [
            "## Your Shopping Profile",
            f"- Weekly Walmart budget: ${p.weekly_budget:.2f}",
            f"- SNAP eligible: {'Yes' if p.snap_eligible else 'No'}",
            f"- Economic status: {p.economic_status}",
            f"- Shopping motivation: {p.shopping_motivation}",
            f"- Category preference: {cat_pref_str}",
            f"- Price sensitivity: {p.price_sensitivity}",
            f"- Visit frequency: {p.visit_frequency}",
            f"- Preferred departments: {', '.join(p.preferred_departments)}",
            f"- Decision style: {p.decision_style}",
            f"- Brand loyalty: {p.brand_loyalty}",
            f"- Promotion sensitivity: {p.promotion_sensitivity}",
            "",
            "## Week Context",
            f"- Forecast week: {week_ctx.week_num} of {week_ctx.total_weeks}"
            f"  ({week_ctx.date_start} ~ {week_ctx.date_end})",
            f"- Month: {week_ctx.month_name}  |  Season: {week_ctx.season}",
            f"- {snap_text}",
            f"- Notable events: {event_text}",
            "",
            "## Items to Forecast",
            item_table,
            "",
            "## Output Format (return ONLY this JSON)",
            "{",
            '  "predictions": {',
            json_keys.rstrip(","),
            "  }",
            "}",
        ]
        return "\n".join(lines)

    def _get_avg_price(
        self,
        item_id: str,
        wm_yr_wk: int | None,
        lookback_weeks: int = 13,
    ) -> float | None:
        """sell_prices에서 아이템의 최근 평균 가격을 조회한다."""
        if item_id in self._price_cache:
            return self._price_cache[item_id]

        if self._prices is None or wm_yr_wk is None:
            return None

        item_prices = self._prices[
            (self._prices["store_id"] == self._store_id)
            & (self._prices["item_id"] == item_id)
            & (self._prices["wm_yr_wk"] <= wm_yr_wk)
        ].tail(lookback_weeks)

        if item_prices.empty:
            return None

        avg = float(item_prices["sell_price"].mean())
        self._price_cache[item_id] = avg
        return avg
