"""LLM 기반 합성 페르소나 생성기.

LLM에게 매장 데이터 제약 없이 자유롭게 50개의 다양한 소비 성향 페르소나를 생성시킨다.
배치(batch)마다 이전 배치 요약을 포함해 behavioral convergence를 방지한다.
"""
from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient
from src.models.persona.schema import Persona, PersonaProfile

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 시스템 프롬프트 (고정)                                                        #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a consumer behavior research expert generating synthetic customer profiles
    for a cold-start demand forecasting study. Your task is to create realistic,
    diverse Walmart customer personas that will be used to simulate purchase decisions
    for items with no sales history.

    ## Output Format
    Return a JSON object with a "personas" array. Each element MUST follow this schema exactly:

    {
      "personas": [
        {
          "persona_id": "CA_1_P001",
          "store_id": "CA_1",
          "profile": {
            "description": "<string>",
                // 2-3 sentences. CRITICAL: Show behavior, personality, and lifestyle
                // naturally through actions. Do NOT explicitly state JSON field values.
                // BAD: "She has high price sensitivity and a weekly budget of $50."
                // GOOD: "She always hunts for clearance tags and calculates her cart
                //        total before reaching the register, putting items back if needed."
            "weekly_budget": <float>,
                // Typical weekly spend at Walmart in USD (must match economic_status):
                //   budget-constrained → 30–80
                //   lower-middle       → 60–120
                //   middle-income      → 100–200
                //   upper-middle       → 180–300
                //   discretionary      → 250+
            "snap_eligible": <boolean>,
                // SNAP (food stamps) recipient. Must be false for middle-income and above.
            "shopping_motivation": "<string>",
                // Choose EXACTLY ONE from:
                // "stock-up trip" | "convenience fill-in" | "deal hunting" |
                // "impulse browsing" | "routine replenishment" | "special occasion" |
                // "health & wellness focus" | "hobby pursuit"
            "economic_status": "<string>",
                // Choose EXACTLY ONE from:
                // "budget-constrained" | "lower-middle" | "middle-income" |
                // "upper-middle" | "discretionary spender"
            "category_preference": {
                // MUST sum to exactly 1.0
                "FOODS": <float>,
                "HOBBIES": <float>,
                "HOUSEHOLD": <float>
            },
            "price_sensitivity": "<string>",
                // EXACTLY ONE: "low" | "medium" | "high"
            "visit_frequency": "<string>",
                // e.g., "daily", "2-3 times/week", "weekly", "bi-weekly"
            "preferred_departments": ["<string>"],
                // Use ONLY from: FOODS_1, FOODS_2, FOODS_3,
                //                HOBBIES_1, HOBBIES_2,
                //                HOUSEHOLD_1, HOUSEHOLD_2
            "decision_style": "<string>",
                // EXACTLY ONE: "impulsive" | "planned" | "habitual" | "deal_seeker"
            "brand_loyalty": "<string>",
                // EXACTLY ONE: "low" | "medium" | "high"
            "promotion_sensitivity": "<string>"
                // EXACTLY ONE: "low" | "medium" | "high"
          }
        }
      ]
    }

    ## Diversity Requirements
    Cross-vary ALL four dimensions within each batch to prevent behavioral convergence:
    1. Demographics       — age group (18-25, 26-35, 36-50, 51-65, 65+),
                            household type (single, couple, family+kids, senior)
    2. Lifestyle          — urban professional, suburban parent, rural resident,
                            student, retiree, gig worker
    3. Shopping motivation — MUST be unique per persona within the same batch
    4. Economic status    — distribute across all five levels within the batch
""")

# --------------------------------------------------------------------------- #
# 생성기 클래스                                                                 #
# --------------------------------------------------------------------------- #

class LLMPersonaGenerator:
    """LLM을 사용해 합성 페르소나를 배치 단위로 생성한다.

    Args:
        config: load_config()로 로드한 DotDict 설정 객체.
        llm_client: LLMClient 인스턴스.
    """

    def __init__(self, config: Any, llm_client: LLMClient) -> None:
        personas_cfg = config.experiment.personas
        self.store_id: str = personas_cfg.store_id
        self.n_personas: int = int(personas_cfg.n_personas)
        self.batch_size: int = int(personas_cfg.batch_size)
        self.llm = llm_client

    def generate(self, n_personas: int | None = None) -> list[Persona]:
        """페르소나를 배치 단위로 생성한다.

        Args:
            n_personas: 생성할 페르소나 수. None이면 config 값 사용.

        Returns:
            생성된 Persona 리스트.
        """
        total = n_personas if n_personas is not None else self.n_personas
        system_prompt = _SYSTEM_PROMPT

        all_personas: list[Persona] = []
        prev_summaries: list[str] = []

        idx = 1  # 1-indexed persona number
        batch_idx = 0

        while idx <= total:
            batch_size = min(self.batch_size, total - idx + 1)
            end_idx = idx + batch_size - 1

            logger.info(
                "배치 %d 생성 중: %s_P%03d ~ %s_P%03d (%d개)",
                batch_idx, self.store_id, idx, self.store_id, end_idx, batch_size,
            )

            user_prompt = self._build_user_prompt(
                batch_idx=batch_idx,
                batch_size=batch_size,
                start_idx=idx,
                prev_summaries=prev_summaries,
            )

            try:
                raw = self.llm.generate_json(system_prompt, user_prompt)
                batch = self._parse_batch(raw, start_idx=idx)
            except Exception as e:
                logger.error("배치 %d 생성 실패: %s", batch_idx, e)
                idx += batch_size
                batch_idx += 1
                continue

            # 다음 배치 프롬프트를 위한 요약 추출
            for p in batch:
                prev_summaries.append(self._summarize(p))

            all_personas.extend(batch)
            logger.info("배치 %d 완료: %d개 생성 (누계: %d개)", batch_idx, len(batch), len(all_personas))

            idx += batch_size
            batch_idx += 1

        logger.info("전체 생성 완료: %d / %d 페르소나", len(all_personas), total)
        return all_personas

    # --------------------------------------------------------------------- #
    # 프롬프트 구성                                                            #
    # --------------------------------------------------------------------- #

    def _build_user_prompt(
        self,
        batch_idx: int,
        batch_size: int,
        start_idx: int,
        prev_summaries: list[str],
    ) -> str:
        """배치별 사용자 프롬프트를 구성한다."""
        end_idx = start_idx + batch_size - 1
        id_range = ", ".join(
            f"{self.store_id}_P{i:03d}" for i in range(start_idx, end_idx + 1)
        )

        lines: list[str] = [
            "Store context:",
            "A Walmart Supercenter in California (store ID: CA_1) selling groceries (FOODS),",
            "household supplies (HOUSEHOLD), and hobby items (HOBBIES).",
            "",
            "Task:",
            f"Generate {batch_size} synthetic customer profiles.",
            f"Assign persona_ids sequentially: {id_range}.",
            "",
            "Constraints:",
            "- Each persona MUST have a UNIQUE shopping_motivation (no repeats within this batch).",
            "- category_preference values MUST sum to exactly 1.0.",
            "- Use ONLY valid department IDs listed in the schema.",
            "- weekly_budget MUST be consistent with economic_status per the budget guidance.",
            "- snap_eligible must be false for middle-income, upper-middle, and discretionary spender.",
        ]

        if prev_summaries:
            lines += [
                "",
                "Already generated personas (DO NOT repeat these archetypes):",
            ]
            for s in prev_summaries:
                lines.append(f"  • {s}")
            lines += [
                "",
                "Generate personas that differ meaningfully from ALL of the above in at least",
                "economic_status, shopping_motivation, AND lifestyle.",
            ]

        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # 파싱 및 검증                                                            #
    # --------------------------------------------------------------------- #

    def _parse_batch(self, raw: dict[str, Any], start_idx: int) -> list[Persona]:
        """LLM 응답을 Persona 리스트로 변환한다.

        유효성 검증 실패 시 해당 페르소나를 스킵하고 경고를 기록한다.
        """
        personas_data = raw.get("personas", [])
        if not isinstance(personas_data, list):
            logger.error("응답에 'personas' 배열이 없음: %s", raw)
            return []

        result: list[Persona] = []
        for i, pdata in enumerate(personas_data):
            expected_id = f"{self.store_id}_P{start_idx + i:03d}"
            try:
                profile = PersonaProfile.from_dict(pdata["profile"])
                errors = profile.validate()
                if errors:
                    logger.warning(
                        "%s 검증 실패 (스킵): %s", expected_id, " | ".join(errors)
                    )
                    continue

                # LLM이 할당한 persona_id 무시하고 순서 기반 ID 사용
                persona = Persona(
                    persona_id=expected_id,
                    store_id=self.store_id,
                    profile=profile,
                )
                result.append(persona)

            except (KeyError, TypeError, ValueError) as e:
                logger.warning("%s 파싱 실패 (스킵): %s", expected_id, e)

        return result

    @staticmethod
    def _summarize(persona: Persona) -> str:
        """다음 배치 프롬프트에 포함할 페르소나 요약 문자열을 생성한다."""
        p = persona.profile
        main_cat = max(p.category_preference, key=p.category_preference.get)  # type: ignore[arg-type]
        return (
            f"{persona.persona_id} — {p.economic_status}, "
            f"{main_cat}-focused, {p.price_sensitivity} price sensitivity, "
            f"{p.decision_style}, {p.shopping_motivation}"
        )

    # --------------------------------------------------------------------- #
    # 저장                                                                   #
    # --------------------------------------------------------------------- #

    def save(self, personas: list[Persona], output_dir: str | Path) -> None:
        """페르소나를 개별 JSON 파일과 통합 파일로 저장한다.

        Args:
            personas: 저장할 Persona 리스트.
            output_dir: 저장 디렉토리 경로.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 개별 파일
        for p in personas:
            fp = out / f"{p.persona_id}.json"
            fp.write_text(
                json.dumps(p.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # 통합 파일
        all_path = out / "all_personas.json"
        all_path.write_text(
            json.dumps(
                [p.to_dict() for p in personas],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # 메타데이터
        meta = {
            "generated_at": datetime.now().isoformat(),
            "n_personas": len(personas),
            "store_id": self.store_id,
            "distribution": self._compute_distribution(personas),
        }
        (out / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "저장 완료: %d개 파일 → %s", len(personas), out.resolve()
        )
        self._log_distribution(personas)

    @staticmethod
    def _compute_distribution(personas: list[Persona]) -> dict[str, Any]:
        """페르소나 속성 분포를 계산한다."""
        from collections import Counter

        profiles = [p.profile for p in personas]
        return {
            "price_sensitivity": dict(Counter(p.price_sensitivity for p in profiles)),
            "decision_style": dict(Counter(p.decision_style for p in profiles)),
            "economic_status": dict(Counter(p.economic_status for p in profiles)),
            "shopping_motivation": dict(Counter(p.shopping_motivation for p in profiles)),
            "snap_eligible": dict(Counter(str(p.snap_eligible) for p in profiles)),
            "brand_loyalty": dict(Counter(p.brand_loyalty for p in profiles)),
        }

    def _log_distribution(self, personas: list[Persona]) -> None:
        """생성된 페르소나 분포를 로그에 기록한다."""
        dist = self._compute_distribution(personas)
        logger.info("=== 페르소나 분포 ===")
        for attr, counts in dist.items():
            logger.info("  %s: %s", attr, counts)
