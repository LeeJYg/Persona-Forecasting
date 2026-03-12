"""warm_train 전체 아이템에 대한 V3 임베딩 확장 스크립트.

기존 exp011_v3_pipeline에서 300개 warm 아이템 임베딩을 추출했으며,
이 스크립트는 나머지 2,649개 아이템의 임베딩을 동일한 V3 방식으로 추출한다.

V3 방식:
  - 입력: apply_chat_template(persona_text + item_text, role="user")
  - 레이어: hidden_states[-2] (second-to-last)
  - 풀링: last real-token position (sequence 마지막 non-padding 위치)

출력:
  - 체크포인트: experiments/exp011_v3_pipeline/embeddings/checkpoints_warm_expand/{item_id}.pt
      각 파일: shape (50, 5120), float32
  - 최종 병합: warm_raw_all.pt     shape (2949, 50, 5120)
               warm_raw_all_meta.csv  (item_id, cat_id, dept_id, is_cold, is_new)

재실행 시:
  이미 체크포인트가 있는 아이템은 자동으로 건너뜀 → 중단 후 재실행 가능.

실행법:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \\
  HF_HOME=/mnt/sdd1/jylee/huggingface_cache \\
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
  tmux new-session -d -s expand_emb \\
    "conda run --no-capture-output -n persona-forecasting \\
     python scripts/expand_warm_embeddings.py \\
     2>&1 | tee experiments/exp011_v3_pipeline/embeddings/expand_run.log"

  # merge만 수행 (추출 완료 후):
  python scripts/expand_warm_embeddings.py --merge-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.forecasting.qwen_embedder import (
    build_persona_text,
    build_item_text,
    build_combined_text,
)
from src.models.persona.schema import Persona

CFG_PATH = ROOT / "configs/exp_expand_embeddings.yaml"

# ──────────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_personas(cfg: dict) -> list[Any]:
    persona_path = ROOT / cfg["paths"]["personas_json"]
    with open(persona_path, encoding="utf-8") as f:
        raw = json.load(f)
    personas = [Persona.from_dict(p) for p in raw]
    logger.info("페르소나 로드: %d개", len(personas))
    return personas


def compute_avg_price(sell_prices_path: Path, store_id: str, lookback_weeks: int) -> dict[str, float]:
    """CA_1 아이템별 최근 N주 평균 가격 반환."""
    sp = pd.read_csv(sell_prices_path)
    sp_ca1 = sp[sp["store_id"] == store_id].copy()
    # 최근 N주: wm_yr_wk 기준 상위 N
    recent_weeks = sp_ca1["wm_yr_wk"].nlargest(lookback_weeks).unique()
    sp_recent = sp_ca1[sp_ca1["wm_yr_wk"].isin(recent_weeks)]
    avg = sp_recent.groupby("item_id")["sell_price"].mean().to_dict()
    return avg


def get_items_to_extract(cfg: dict) -> pd.DataFrame:
    """임베딩이 없는 warm 아이템 목록 반환."""
    wt = pd.read_csv(ROOT / cfg["paths"]["warm_train_csv"])
    existing_meta = pd.read_csv(ROOT / cfg["paths"]["existing_emb_dir"] / "item_meta.csv")
    already_embedded = set(existing_meta["item_id"])

    warm_items = (
        wt[["item_id", "dept_id", "cat_id"]]
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )
    new_items = warm_items[~warm_items["item_id"].isin(already_embedded)].reset_index(drop=True)
    logger.info(
        "warm 아이템 전체: %d | 기존 임베딩: %d | 추출 대상: %d",
        len(warm_items), len(already_embedded), len(new_items),
    )
    return new_items


# ──────────────────────────────────────────────────────────────
# V3 임베딩 추출 (chat_template + last-token + hidden_states[-2])
# ──────────────────────────────────────────────────────────────

def load_model(cfg: dict):
    """Qwen 모델과 토크나이저를 로드한다."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = cfg["model"]["model_name"]
    cache_dir = cfg["model"]["cache_dir"]
    quant = cfg["model"]["quantization"]
    max_length = cfg["model"]["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # right-padding: last real token = actual_length - 1
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=quant["quant_type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=cfg["model"]["device_map"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_hidden_states=True,
        cache_dir=cache_dir,
    )
    model.eval()
    hidden_size = model.config.hidden_size
    logger.info("모델 로드 완료: hidden_size=%d", hidden_size)
    return model, tokenizer, max_length


def extract_v3_batch(
    texts: list[str],
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """V3 방식으로 임베딩 추출.

    입력 텍스트를 chat template으로 감싸고,
    hidden_states[-2]의 last real-token 벡터를 반환한다.

    Args:
        texts: raw combined texts (persona + item).
        model: 로드된 Qwen 모델.
        tokenizer: 토크나이저.
        max_length: 최대 토큰 길이.
        batch_size: GPU 배치 크기.

    Returns:
        np.ndarray: shape (len(texts), hidden_size), float32.
    """
    # chat template 적용
    chat_texts = []
    for t in texts:
        messages = [{"role": "user", "content": t}]
        ct = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        chat_texts.append(ct)

    all_embs: list[np.ndarray] = []
    n = len(chat_texts)
    first_param_device = next(model.parameters()).device

    for start in range(0, n, batch_size):
        batch = chat_texts[start: start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(first_param_device)
        attention_mask = encoded["attention_mask"].to(first_param_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # hidden_states[-2]: second-to-last layer, shape (batch, seq_len, hidden)
        hidden = outputs.hidden_states[-2]  # (batch, seq_len, hidden)

        # last real-token position (right-padding: sum - 1)
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_size_actual = hidden.size(0)
        last_token_emb = hidden[
            torch.arange(batch_size_actual, device=hidden.device),
            seq_lengths,
        ]  # (batch, hidden)

        batch_embs = last_token_emb.float().cpu().numpy()
        all_embs.append(batch_embs)

        del outputs, hidden, last_token_emb
        torch.cuda.empty_cache()

    return np.vstack(all_embs)  # (n, hidden_size)


def extract_item_embedding(
    item_id: str,
    dept_id: str,
    cat_id: str,
    avg_price: float | None,
    personas: list[Any],
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    condition: str = "A",
) -> np.ndarray:
    """단일 아이템에 대한 V3 임베딩 추출.

    Returns:
        np.ndarray: shape (n_personas, hidden_size), float32.
    """
    item_text = build_item_text(item_id, dept_id, cat_id, avg_price)
    texts = []
    for p in personas:
        p_text = build_persona_text(p.profile, condition=condition)
        texts.append(build_combined_text(p_text, item_text))

    emb = extract_v3_batch(texts, model, tokenizer, max_length, batch_size)
    return emb  # (n_personas, hidden_size)


# ──────────────────────────────────────────────────────────────
# 체크포인트 관리
# ──────────────────────────────────────────────────────────────

def get_checkpoint_dir(cfg: dict) -> Path:
    ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_completed_items(ckpt_dir: Path) -> set[str]:
    """체크포인트 디렉토리에서 이미 완료된 아이템 ID 반환."""
    completed = set()
    for pt_file in ckpt_dir.glob("*.pt"):
        completed.add(pt_file.stem)
    return completed


def save_checkpoint(item_id: str, emb: np.ndarray, ckpt_dir: Path) -> None:
    """개별 아이템 임베딩을 체크포인트로 저장."""
    path = ckpt_dir / f"{item_id}.pt"
    torch.save(torch.tensor(emb, dtype=torch.float32), path)


# ──────────────────────────────────────────────────────────────
# 최종 병합
# ──────────────────────────────────────────────────────────────

def merge_checkpoints(cfg: dict) -> None:
    """체크포인트를 warm_raw_all.pt + meta CSV로 병합한다.

    warm_raw_all.pt: shape (2949, 50, 5120), float32
      - 기존 300개 (warm_raw.pt 순서)
      - 신규 2649개 (알파벳 순)
    """
    emb_dir = ROOT / cfg["paths"]["existing_emb_dir"]
    ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]

    # 기존 300개 로드
    existing_raw = torch.load(emb_dir / "warm_raw.pt", map_location="cpu", weights_only=True)
    existing_meta = pd.read_csv(emb_dir / "item_meta.csv")
    existing_warm_meta = existing_meta[existing_meta["is_cold"] == False].reset_index(drop=True)

    logger.info("기존 warm 임베딩: %s", tuple(existing_raw.shape))

    # 신규 체크포인트 로드
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    new_item_ids = [f.stem for f in ckpt_files]
    logger.info("신규 체크포인트: %d개", len(ckpt_files))

    if not ckpt_files:
        logger.warning("신규 체크포인트 없음 — 병합 중단")
        return

    new_tensors = []
    for f in ckpt_files:
        t = torch.load(f, map_location="cpu", weights_only=True)
        new_tensors.append(t.unsqueeze(0))  # (1, 50, hidden)

    new_raw = torch.cat(new_tensors, dim=0)  # (N_new, 50, hidden)
    logger.info("신규 임베딩: %s", tuple(new_raw.shape))

    # 병합
    all_raw = torch.cat([existing_raw, new_raw], dim=0)  # (2949, 50, hidden)
    out_path = emb_dir / "warm_raw_all.pt"
    torch.save(all_raw, out_path)
    logger.info("병합 저장: %s  shape=%s", out_path, tuple(all_raw.shape))

    # 메타 CSV
    wt = pd.read_csv(ROOT / cfg["paths"]["warm_train_csv"])
    item_lookup = wt.drop_duplicates("item_id").set_index("item_id")[["dept_id", "cat_id"]]

    new_meta_rows = []
    for iid in new_item_ids:
        row = item_lookup.loc[iid] if iid in item_lookup.index else {"dept_id": "UNK", "cat_id": "UNK"}
        new_meta_rows.append({
            "item_id": iid,
            "cat_id": row["cat_id"],
            "dept_id": row["dept_id"],
            "is_cold": False,
            "is_new": True,
        })

    existing_warm_meta["is_new"] = False
    new_meta_df = pd.DataFrame(new_meta_rows)
    all_meta = pd.concat([existing_warm_meta, new_meta_df], ignore_index=True)
    meta_path = emb_dir / "warm_raw_all_meta.csv"
    all_meta.to_csv(meta_path, index=False)
    logger.info("메타 저장: %s  rows=%d", meta_path, len(all_meta))
    logger.info("=== 병합 완료 ===")


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="warm 아이템 V3 임베딩 확장")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="체크포인트 병합만 수행 (추출 건너뜀)",
    )
    parser.add_argument(
        "--config",
        default=str(CFG_PATH),
        help="config yaml 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    log_path = ROOT / cfg["paths"]["log_dir"] / "expand_embeddings.log"
    setup_logging(log_path)

    logger.info("=== warm 임베딩 확장 시작 ===")
    logger.info("config: %s", CFG_PATH)

    if args.merge_only:
        logger.info("--merge-only: 병합만 수행")
        merge_checkpoints(cfg)
        return

    # ── 준비 ──────────────────────────────────────────────────
    personas = load_personas(cfg)
    n_personas = cfg["extraction"]["n_personas"]
    personas = personas[:n_personas]
    condition = cfg["extraction"]["condition"]
    save_every = cfg["extraction"]["save_every_n_items"]
    batch_size = cfg["model"]["batch_size"]
    max_length = cfg["model"]["max_length"]
    store_id = cfg["extraction"]["target_store"]
    lookback_weeks = cfg["extraction"]["price_lookback_weeks"]

    ckpt_dir = get_checkpoint_dir(cfg)
    completed = get_completed_items(ckpt_dir)
    logger.info("기존 체크포인트: %d개 완료", len(completed))

    # 추출 대상 아이템
    new_items_df = get_items_to_extract(cfg)
    todo_df = new_items_df[~new_items_df["item_id"].isin(completed)].reset_index(drop=True)
    logger.info("이번 실행에서 추출할 아이템: %d개", len(todo_df))

    if todo_df.empty:
        logger.info("추출할 아이템 없음 — 병합으로 진행")
        merge_checkpoints(cfg)
        return

    # avg_price 계산
    avg_price = compute_avg_price(
        ROOT / cfg["paths"]["sell_prices_csv"],
        store_id=store_id,
        lookback_weeks=lookback_weeks,
    )

    # ── 모델 로드 ──────────────────────────────────────────────
    logger.info("모델 로드 중...")
    model, tokenizer, max_length = load_model(cfg)

    # ── 추출 루프 ──────────────────────────────────────────────
    total = len(todo_df)
    for idx, row in todo_df.iterrows():
        item_id = row["item_id"]
        dept_id = row["dept_id"]
        cat_id = row["cat_id"]
        price = avg_price.get(item_id)

        logger.info(
            "[%d/%d] %s (%s, price=%s)",
            idx + 1, total, item_id, dept_id,
            f"${price:.2f}" if price else "N/A",
        )

        emb = extract_item_embedding(
            item_id=item_id,
            dept_id=dept_id,
            cat_id=cat_id,
            avg_price=price,
            personas=personas,
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            condition=condition,
        )  # (n_personas, hidden_size)

        save_checkpoint(item_id, emb, ckpt_dir)
        completed.add(item_id)

        # 주기적 상태 로그
        n_done = len(completed)
        n_total_new = len(new_items_df)
        logger.info(
            "  진행: %d / %d 완료 (%.1f%%)",
            n_done, n_total_new, 100 * n_done / n_total_new,
        )

        # N개마다 중간 병합 상태 기록 (merge는 하지 않음)
        if n_done % save_every == 0:
            logger.info("  [체크포인트 %d개 저장됨]", n_done)

    logger.info("=== 추출 완료: %d / %d ===", len(completed), len(new_items_df))

    # 전체 완료 시 자동 병합
    if len(completed) >= len(new_items_df):
        logger.info("전체 완료 → 자동 병합 시작")
        merge_checkpoints(cfg)
    else:
        remaining = len(new_items_df) - len(completed)
        logger.info("미완료 %d개 — 재실행 시 이어서 진행", remaining)


if __name__ == "__main__":
    main()
