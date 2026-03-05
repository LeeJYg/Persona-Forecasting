"""LLM 페르소나 기반 수요 예측 모델 패키지.

Track A: GPT-4o-mini를 이용한 Naive 페르소나 예측
Track B: Qwen 2.5 32B hidden state + linear regression head
"""
from __future__ import annotations

from src.models.forecasting.persona_predictor import PersonaPredictor

__all__ = ["PersonaPredictor"]
