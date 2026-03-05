"""Track B: Qwen 2.5 Instruct hidden state 임베딩 추출기.

config.yaml의 track_b.model_name 하나만 바꿔 7B / 14B / 32B로 교체 가능.
양자화(4-bit / 8-bit)는 track_b.quantization 섹션에서 설정한다.

설계 원칙
----------
1. 입력 구성: concat(persona_structured_text, item_text) → 단일 시퀀스
2. forward pass → last hidden layer → mean-pool over all non-padding positions
3. x_item = mean_pool([emb(persona_i, item) for i in range(50)])
   → 50개 (persona, item) 쌍의 평균 → 아이템별 하나의 임베딩 벡터

GPU 메모리 참고 (bfloat16 / 4-bit, 24GB GPU 기준):
   7B  : ~14 GB / ~4 GB  → GPU 1개
   14B : ~28 GB / ~7 GB  → GPU 2개 / 1개        ← 기본값
   32B : ~64 GB / ~16 GB → GPU 3개 / 1개
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Condition A 입력 텍스트 템플릿 (description 제외)
_PERSONA_TEXT_TEMPLATE = """\
Customer profile:
- Weekly budget: ${weekly_budget:.2f}
- SNAP eligible: {snap_eligible}
- Economic status: {economic_status}
- Shopping motivation: {shopping_motivation}
- Category preference: {cat_pref}
- Price sensitivity: {price_sensitivity}
- Visit frequency: {visit_frequency}
- Preferred departments: {preferred_departments}
- Decision style: {decision_style}
- Brand loyalty: {brand_loyalty}
- Promotion sensitivity: {promotion_sensitivity}"""

_ITEM_TEXT_TEMPLATE = """\
Item: {item_id} | Department: {dept_id} | Category: {cat_id} | Average price: {price}"""

_COMBINED_TEMPLATE = "{persona_text}\n\n{item_text}"


def build_persona_text(profile: Any, condition: str = "A") -> str:
    """PersonaProfile을 임베딩 입력 텍스트로 변환한다.

    Condition A: structured fields only (description 제외).

    Args:
        profile: PersonaProfile 인스턴스.
        condition: "A" | "B" | "C".

    Returns:
        str: 텍스트 표현.
    """
    cat_pref = ", ".join(
        f"{cat}={pct:.2f}"
        for cat, pct in sorted(profile.category_preference.items(), key=lambda x: -x[1])
    )
    text = _PERSONA_TEXT_TEMPLATE.format(
        weekly_budget=profile.weekly_budget,
        snap_eligible=str(profile.snap_eligible).lower(),
        economic_status=profile.economic_status,
        shopping_motivation=profile.shopping_motivation,
        cat_pref=cat_pref,
        price_sensitivity=profile.price_sensitivity,
        visit_frequency=profile.visit_frequency,
        preferred_departments=", ".join(profile.preferred_departments),
        decision_style=profile.decision_style,
        brand_loyalty=profile.brand_loyalty,
        promotion_sensitivity=profile.promotion_sensitivity,
    )
    if condition == "B":
        text = f"Description: {profile.description}\n\n{text}"
    elif condition == "C":
        text = f"Description: {profile.description}"
    return text


def build_item_text(item_id: str, dept_id: str, cat_id: str, avg_price: float | None) -> str:
    """아이템 메타데이터를 임베딩 입력 텍스트로 변환한다."""
    price_str = f"${avg_price:.2f}" if avg_price is not None else "N/A"
    return _ITEM_TEXT_TEMPLATE.format(
        item_id=item_id,
        dept_id=dept_id,
        cat_id=cat_id,
        price=price_str,
    )


def build_combined_text(persona_text: str, item_text: str) -> str:
    """페르소나 텍스트 + 아이템 텍스트를 결합한다."""
    return _COMBINED_TEMPLATE.format(
        persona_text=persona_text,
        item_text=item_text,
    )


class QwenEmbedder:
    """config-driven Qwen Instruct 임베딩 추출기.

    model_name과 quantization 설정은 config.yaml의 track_b 섹션에서 가져온다.
    QwenEmbedder 인스턴스를 직접 생성할 때는 아래 인자를 전달한다.

    Args:
        model_name: HuggingFace 모델 ID.
                    (예: "Qwen/Qwen2.5-14B-Instruct")
        dtype: 파라미터 기본 dtype ("bfloat16" | "float16" | "float32").
               양자화 사용 시 비-양자화 레이어에 적용되며, 4-bit의 실제 연산 dtype은
               quantization["compute_dtype"]으로 별도 제어된다.
        quantization: 양자화 설정 dict 또는 None.
            {
              "mode": "none" | "4bit" | "8bit",
              "compute_dtype": "bfloat16",   # 4-bit 전용
              "quant_type": "nf4",           # 4-bit 전용 ("nf4" | "fp4")
              "double_quant": True,          # 4-bit 전용
            }
        device_map: transformers device_map ("auto" | "cuda" | "cpu" | ...).
                    멀티-GPU 자동 분산에는 "auto" 권장.
        batch_size: 한 번에 처리할 시퀀스 수 (GPU 메모리에 따라 조정).
        cache_dir: HuggingFace 모델 캐시 디렉토리.
        max_length: 토크나이저 최대 시퀀스 길이.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        dtype: str = "bfloat16",
        quantization: dict | None = None,
        device_map: str = "auto",
        batch_size: int = 4,
        cache_dir: str | Path | None = None,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._dtype_str = dtype
        self._quant_cfg: dict = quantization or {"mode": "none"}
        self._device_map = device_map
        self._batch_size = batch_size
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._max_length = max_length

        self._model = None
        self._tokenizer = None
        self._hidden_size: int | None = None

    # ------------------------------------------------------------------ #
    # 로드                                                                 #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """모델과 토크나이저를 로드한다.

        최초 호출 시에만 실제 로드가 수행된다 (이후 호출은 무시).
        양자화 설정이 있으면 BitsAndBytesConfig를 적용한다.
        """
        if self._model is not None:
            return

        import torch  # type: ignore[import-untyped]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

        quant_mode = str(self._quant_cfg.get("mode", "none")).lower()
        bnb_config = self._build_bnb_config(quant_mode)

        # torch_dtype: 양자화 사용 시 비-양자화 레이어용 dtype
        # 4-bit/8-bit 모드에서는 "auto"로 두는 것이 권장이나,
        # bfloat16 명시가 안전한 경우가 많으므로 유지
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

        load_kwargs: dict = {
            "device_map": self._device_map,
            "trust_remote_code": True,
            "output_hidden_states": True,
            "cache_dir": self._cache_dir,
        }

        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
            # 양자화 시 torch_dtype은 compute_dtype과 일치시킴
            compute_dtype_str = str(self._quant_cfg.get("compute_dtype", "bfloat16"))
            load_kwargs["torch_dtype"] = dtype_map.get(compute_dtype_str, torch.bfloat16)
            logger.info(
                "Qwen 모델 로드 중: %s [%s 양자화, compute=%s]",
                self._model_name, quant_mode, compute_dtype_str,
            )
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            logger.info(
                "Qwen 모델 로드 중: %s [full precision, dtype=%s]",
                self._model_name, self._dtype_str,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            cache_dir=self._cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            **load_kwargs,
        )
        self._model.eval()
        self._hidden_size = self._model.config.hidden_size
        logger.info(
            "모델 로드 완료: hidden_size=%d, quant_mode=%s, device_map=%s",
            self._hidden_size, quant_mode, self._device_map,
        )

    def _build_bnb_config(self, quant_mode: str):
        """quant_mode에 따라 BitsAndBytesConfig를 생성한다.

        Args:
            quant_mode: "none" | "4bit" | "8bit"

        Returns:
            BitsAndBytesConfig 인스턴스, 또는 None (quant_mode="none").
        """
        if quant_mode == "none":
            return None

        try:
            import torch  # type: ignore[import-untyped]
            from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "양자화를 사용하려면 bitsandbytes 패키지가 필요합니다.\n"
                "  pip install bitsandbytes>=0.43.0"
            ) from e

        if quant_mode == "4bit":
            compute_dtype_str = str(self._quant_cfg.get("compute_dtype", "bfloat16"))
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            compute_dtype = dtype_map.get(compute_dtype_str, torch.bfloat16)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=str(self._quant_cfg.get("quant_type", "nf4")),
                bnb_4bit_use_double_quant=bool(self._quant_cfg.get("double_quant", True)),
            )

        if quant_mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)

        raise ValueError(
            f"지원하지 않는 quantization mode: '{quant_mode}'. "
            "'none' | '4bit' | '8bit' 중 하나를 선택하세요."
        )

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트로부터 임베딩 행렬을 반환한다.

        last hidden layer의 non-padding 토큰들을 mean-pool해 각 시퀀스를
        하나의 벡터로 표현한다.

        Args:
            texts: 입력 텍스트 리스트.

        Returns:
            np.ndarray: shape (len(texts), hidden_size).
        """
        if self._model is None:
            raise RuntimeError("load()를 먼저 호출하세요.")

        import torch  # type: ignore[import-untyped]

        all_embeddings: list[np.ndarray] = []
        n = len(texts)

        for batch_start in range(0, n, self._batch_size):
            batch_texts = texts[batch_start: batch_start + self._batch_size]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )

            # 모델의 첫 번째 파라미터 device 감지
            input_ids = encoded["input_ids"].to(next(self._model.parameters()).device)
            attention_mask = encoded["attention_mask"].to(input_ids.device)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # last hidden layer: (batch, seq_len, hidden_size)
            last_hidden = outputs.hidden_states[-1]

            # mean-pool: non-padding 위치만 평균
            # attention_mask: 1=real token, 0=padding
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)     # (batch, hidden_size)
            lengths = mask.sum(dim=1)                    # (batch, 1)
            lengths = lengths.clamp(min=1e-9)
            mean_pooled = summed / lengths               # (batch, hidden_size)

            # float32 numpy로 변환 (GPU 메모리 즉시 해제)
            batch_embs = mean_pooled.float().cpu().numpy()
            all_embeddings.append(batch_embs)

            del outputs, last_hidden, mean_pooled
            torch.cuda.empty_cache()

            if (batch_start // self._batch_size + 1) % 10 == 0:
                logger.info(
                    "임베딩 추출: %d / %d",
                    min(batch_start + self._batch_size, n),
                    n,
                )

        return np.vstack(all_embeddings)  # (n, hidden_size)

    def build_item_embeddings(
        self,
        item_ids: list[str],
        item_meta: dict[str, dict],
        personas: list[Any],
        condition: str = "A",
    ) -> np.ndarray:
        """x_item = mean_pool([emb(persona_i, item) for i in range(50)]).

        각 (persona, item) 쌍에 대해 임베딩을 추출하고,
        같은 아이템의 모든 페르소나 임베딩을 평균내어 최종 아이템 임베딩을 만든다.

        Args:
            item_ids: 임베딩을 생성할 아이템 ID 목록.
            item_meta: {item_id: {"dept_id": ..., "cat_id": ..., "avg_price": ...}}
            personas: Persona 인스턴스 리스트.
            condition: "A" | "B" | "C".

        Returns:
            np.ndarray: shape (len(item_ids), hidden_size).
        """
        if self._model is None:
            raise RuntimeError("load()를 먼저 호출하세요.")

        n_items = len(item_ids)
        n_personas = len(personas)
        logger.info(
            "아이템 임베딩 구성 시작: %d items × %d personas = %d forward passes",
            n_items, n_personas, n_items * n_personas,
        )

        # 아이템 텍스트 미리 생성
        item_texts: dict[str, str] = {}
        for iid in item_ids:
            meta = item_meta.get(iid, {})
            item_texts[iid] = build_item_text(
                item_id=iid,
                dept_id=str(meta.get("dept_id", "UNKNOWN")),
                cat_id=str(meta.get("cat_id", "UNKNOWN")),
                avg_price=meta.get("avg_price"),
            )

        # 페르소나 텍스트 미리 생성
        persona_texts: list[str] = [
            build_persona_text(p.profile, condition=condition) for p in personas
        ]

        # (item_id, persona_idx) 순서로 모든 combined text 구성
        # order: item0×persona0, item0×persona1, ..., item0×persona49,
        #        item1×persona0, ...
        all_texts: list[str] = []
        for iid in item_ids:
            for p_text in persona_texts:
                all_texts.append(build_combined_text(p_text, item_texts[iid]))

        total = len(all_texts)
        logger.info("combined 텍스트 %d개 임베딩 추출 중...", total)
        all_embeddings = self.get_embeddings(all_texts)
        # shape: (n_items * n_personas, hidden_size)

        # 아이템별 mean-pool: (n_personas,) → (1,) per item
        item_embeddings = all_embeddings.reshape(n_items, n_personas, -1).mean(axis=1)
        # shape: (n_items, hidden_size)

        logger.info(
            "아이템 임베딩 완료: shape=%s", item_embeddings.shape
        )
        return item_embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        item_ids: list[str],
        save_path: Path,
    ) -> None:
        """임베딩 행렬을 npz 파일로 저장한다.

        Args:
            embeddings: shape (n_items, hidden_size).
            item_ids: 아이템 ID 목록 (embeddings 행 순서와 일치).
            save_path: 저장 경로 (.npz 확장자 권장).
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            embeddings=embeddings,
            item_ids=np.array(item_ids),
        )
        logger.info("임베딩 저장: %s (shape=%s)", save_path, embeddings.shape)

    @staticmethod
    def load_embeddings(load_path: Path) -> tuple[np.ndarray, list[str]]:
        """save_embeddings()로 저장된 npz 파일을 로드한다.

        Returns:
            (embeddings, item_ids)
        """
        data = np.load(load_path, allow_pickle=True)
        embeddings = data["embeddings"]
        item_ids = data["item_ids"].tolist()
        logger.info("임베딩 로드: %s (shape=%s)", load_path, embeddings.shape)
        return embeddings, item_ids
