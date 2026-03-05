"""OpenAI LLM 클라이언트 래퍼.

OPENAI_API_KEY 환경 변수로 인증한다.
configs/config.yaml의 experiment.llm 섹션 설정을 사용한다.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import openai

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI API 클라이언트 래퍼.

    Args:
        config: load_config()로 로드한 DotDict 설정 객체.

    Raises:
        EnvironmentError: OPENAI_API_KEY 환경 변수가 없을 때.
    """

    def __init__(self, config: Any) -> None:
        llm_cfg = config.experiment.llm
        self.model: str = llm_cfg.model
        self.generation_temperature: float = float(llm_cfg.generation_temperature)
        self.prediction_temperature: float = float(llm_cfg.prediction_temperature)
        self.max_retries: int = int(llm_cfg.max_retries)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. "
                "프로젝트 루트의 .env 파일을 확인하거나 "
                "`export OPENAI_API_KEY=...`를 실행하세요."
            )
        self._client = openai.OpenAI(api_key=api_key)
        logger.info("LLMClient 초기화 완료: model=%s", self.model)

    def _resolve_temperature(self, override: float | None, default: float) -> float:
        """override가 None이면 default를 반환한다."""
        return override if override is not None else default

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """JSON 응답을 생성한다.

        response_format={"type": "json_object"}로 파싱 안정성을 높인다.

        Args:
            system_prompt: 시스템 프롬프트.
            user_prompt: 사용자 프롬프트.
            temperature: 온도. None이면 generation_temperature 사용.

        Returns:
            파싱된 JSON dict.

        Raises:
            RuntimeError: max_retries 초과 시.
        """
        temp = self._resolve_temperature(temperature, self.generation_temperature)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temp,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content or ""
                return json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON 파싱 실패 (시도 %d/%d): %s", attempt, self.max_retries, e
                )
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"{self.max_retries}회 시도 후 JSON 파싱 실패"
                    ) from e

            except openai.RateLimitError:
                wait = 2**attempt
                logger.warning("Rate limit 발생. %d초 후 재시도...", wait)
                time.sleep(wait)

            except openai.OpenAIError as e:
                logger.error(
                    "OpenAI API 오류 (시도 %d/%d): %s", attempt, self.max_retries, e
                )
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenAI API 오류: {e}") from e
                time.sleep(2)

        raise RuntimeError(f"{self.max_retries}회 시도 후 응답 생성 실패")

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        """텍스트 응답을 생성한다.

        Args:
            system_prompt: 시스템 프롬프트.
            user_prompt: 사용자 프롬프트.
            temperature: 온도. None이면 prediction_temperature 사용.

        Returns:
            생성된 텍스트 문자열.
        """
        temp = self._resolve_temperature(temperature, self.prediction_temperature)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content or ""

            except openai.RateLimitError:
                wait = 2**attempt
                logger.warning("Rate limit 발생. %d초 후 재시도...", wait)
                time.sleep(wait)

            except openai.OpenAIError as e:
                logger.error(
                    "OpenAI API 오류 (시도 %d/%d): %s", attempt, self.max_retries, e
                )
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenAI API 오류: {e}") from e
                time.sleep(2)

        raise RuntimeError(f"{self.max_retries}회 시도 후 텍스트 생성 실패")
