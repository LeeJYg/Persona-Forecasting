"""합성 페르소나 데이터 스키마.

PersonaProfile: 소비 성향 프로필 (LLM 생성)
Persona: 최종 페르소나 (profile + 메타데이터)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# 유효값 상수                                                                   #
# --------------------------------------------------------------------------- #

VALID_DEPARTMENTS: frozenset[str] = frozenset([
    "FOODS_1", "FOODS_2", "FOODS_3",
    "HOBBIES_1", "HOBBIES_2",
    "HOUSEHOLD_1", "HOUSEHOLD_2",
])

VALID_CATEGORIES: frozenset[str] = frozenset(["FOODS", "HOBBIES", "HOUSEHOLD"])

VALID_PRICE_SENSITIVITY: frozenset[str] = frozenset(["low", "medium", "high"])

VALID_DECISION_STYLE: frozenset[str] = frozenset([
    "impulsive", "planned", "habitual", "deal_seeker",
])

VALID_BRAND_LOYALTY: frozenset[str] = frozenset(["low", "medium", "high"])

VALID_PROMOTION_SENSITIVITY: frozenset[str] = frozenset(["low", "medium", "high"])

VALID_SHOPPING_MOTIVATION: frozenset[str] = frozenset([
    "stock-up trip",
    "convenience fill-in",
    "deal hunting",
    "impulse browsing",
    "routine replenishment",
    "special occasion",
    "health & wellness focus",
    "hobby pursuit",
])

VALID_ECONOMIC_STATUS: frozenset[str] = frozenset([
    "budget-constrained",
    "lower-middle",
    "middle-income",
    "upper-middle",
    "discretionary spender",
])

CATEGORY_PREFERENCE_TOLERANCE: float = 0.01  # category_preference 합계 허용 오차


# --------------------------------------------------------------------------- #
# 데이터클래스                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class PersonaProfile:
    """페르소나의 소비 성향 프로필.

    LLM이 자유롭게 생성하며, 매장 실데이터(M5)의 제약을 받지 않는다.
    description은 행동 서술 방식(Show, Don't Tell)을 따른다.
    """

    description: str
    weekly_budget: float                    # 주당 Walmart 지출 (USD)
    snap_eligible: bool                     # SNAP(푸드스탬프) 수급 여부
    shopping_motivation: str               # 구매 동기 (VALID_SHOPPING_MOTIVATION 중 하나)
    economic_status: str                   # 경제적 수준 (VALID_ECONOMIC_STATUS 중 하나)
    category_preference: dict[str, float]  # {"FOODS": 0.6, "HOBBIES": 0.1, "HOUSEHOLD": 0.3}
    price_sensitivity: str                 # "low" | "medium" | "high"
    visit_frequency: str                   # e.g., "2-3 times/week"
    preferred_departments: list[str]       # VALID_DEPARTMENTS 의 부분집합
    decision_style: str                    # "impulsive" | "planned" | "habitual" | "deal_seeker"
    brand_loyalty: str                     # "low" | "medium" | "high"
    promotion_sensitivity: str            # "low" | "medium" | "high"

    def validate(self) -> list[str]:
        """유효성 검증. 오류 메시지 목록을 반환한다 (빈 리스트면 통과)."""
        errors: list[str] = []

        # category_preference 합계
        pref_sum = sum(self.category_preference.values())
        if abs(pref_sum - 1.0) > CATEGORY_PREFERENCE_TOLERANCE:
            errors.append(
                f"category_preference 합계={pref_sum:.3f} (1.0 ±{CATEGORY_PREFERENCE_TOLERANCE} 허용)"
            )

        # 카테고리 키
        invalid_cats = set(self.category_preference) - VALID_CATEGORIES
        if invalid_cats:
            errors.append(f"유효하지 않은 카테고리: {invalid_cats}")

        # price_sensitivity
        if self.price_sensitivity not in VALID_PRICE_SENSITIVITY:
            errors.append(
                f"price_sensitivity='{self.price_sensitivity}' "
                f"(유효값: {sorted(VALID_PRICE_SENSITIVITY)})"
            )

        # preferred_departments
        invalid_depts = set(self.preferred_departments) - VALID_DEPARTMENTS
        if invalid_depts:
            errors.append(f"유효하지 않은 부서: {invalid_depts}")

        # decision_style
        if self.decision_style not in VALID_DECISION_STYLE:
            errors.append(
                f"decision_style='{self.decision_style}' "
                f"(유효값: {sorted(VALID_DECISION_STYLE)})"
            )

        # brand_loyalty
        if self.brand_loyalty not in VALID_BRAND_LOYALTY:
            errors.append(
                f"brand_loyalty='{self.brand_loyalty}' "
                f"(유효값: {sorted(VALID_BRAND_LOYALTY)})"
            )

        # promotion_sensitivity
        if self.promotion_sensitivity not in VALID_PROMOTION_SENSITIVITY:
            errors.append(
                f"promotion_sensitivity='{self.promotion_sensitivity}' "
                f"(유효값: {sorted(VALID_PROMOTION_SENSITIVITY)})"
            )

        # shopping_motivation
        if self.shopping_motivation not in VALID_SHOPPING_MOTIVATION:
            errors.append(
                f"shopping_motivation='{self.shopping_motivation}' "
                f"(유효값: {sorted(VALID_SHOPPING_MOTIVATION)})"
            )

        # economic_status
        if self.economic_status not in VALID_ECONOMIC_STATUS:
            errors.append(
                f"economic_status='{self.economic_status}' "
                f"(유효값: {sorted(VALID_ECONOMIC_STATUS)})"
            )

        # weekly_budget 양수 확인
        if self.weekly_budget <= 0:
            errors.append(f"weekly_budget={self.weekly_budget} (양수여야 함)")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict 반환."""
        return {
            "description": self.description,
            "weekly_budget": self.weekly_budget,
            "snap_eligible": self.snap_eligible,
            "shopping_motivation": self.shopping_motivation,
            "economic_status": self.economic_status,
            "category_preference": self.category_preference,
            "price_sensitivity": self.price_sensitivity,
            "visit_frequency": self.visit_frequency,
            "preferred_departments": self.preferred_departments,
            "decision_style": self.decision_style,
            "brand_loyalty": self.brand_loyalty,
            "promotion_sensitivity": self.promotion_sensitivity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonaProfile":
        """dict에서 PersonaProfile을 생성한다."""
        return cls(
            description=d["description"],
            weekly_budget=float(d["weekly_budget"]),
            snap_eligible=bool(d["snap_eligible"]),
            shopping_motivation=d["shopping_motivation"],
            economic_status=d["economic_status"],
            category_preference=d["category_preference"],
            price_sensitivity=d["price_sensitivity"],
            visit_frequency=d["visit_frequency"],
            preferred_departments=d["preferred_departments"],
            decision_style=d["decision_style"],
            brand_loyalty=d["brand_loyalty"],
            promotion_sensitivity=d["promotion_sensitivity"],
        )


@dataclass
class Persona:
    """합성 고객 페르소나.

    persona_id 형식: {store_id}_P{index:03d}  (예: CA_1_P001)
    purchase_history는 현재 빈 리스트로 유지 (예측 파이프라인에서 profile만 사용).
    """

    persona_id: str
    store_id: str
    profile: PersonaProfile
    purchase_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict 반환."""
        return {
            "persona_id": self.persona_id,
            "store_id": self.store_id,
            "profile": self.profile.to_dict(),
            "purchase_history": self.purchase_history,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Persona":
        """dict에서 Persona를 생성한다."""
        return cls(
            persona_id=d["persona_id"],
            store_id=d["store_id"],
            profile=PersonaProfile.from_dict(d["profile"]),
            purchase_history=d.get("purchase_history", []),
        )
