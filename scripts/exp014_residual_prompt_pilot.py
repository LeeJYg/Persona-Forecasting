"""Exp014: Residual Scaling Law (D) + 프롬프트 파일럿 (E).

실험 D: exp013 실험 A와 동일한 train/test split + seed로 residual embedding 적용
        → 원본 vs residual scaling 비교 (MAE 정체 패턴이 같은지)
실험 E: V4-A(Persona-Heavy) / V4-B(Contrastive) 프롬프트 파일럿 (GPU)
        5개 아이템 × 50 페르소나 × 3 variants = 750 texts

실행:
    D만:    conda run -n persona-forecasting python scripts/exp014_residual_prompt_pilot.py
    D+E:    CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=/mnt/sdd1/jylee/huggingface_cache
            conda run -n persona-forecasting python scripts/exp014_residual_prompt_pilot.py --run-e

출력:
    docs/diagnosis/residual_scaling_and_prompt_report.md
    experiments/exp013_scaling_and_direction/figures/exp_d_*.png
    experiments/exp013_scaling_and_direction/figures/exp_e_*.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.optim import Adam

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 경로 ──────────────────────────────────────────────────────────────────
EMB_DIR     = ROOT / "experiments/exp011_v3_pipeline/embeddings"
FIG_DIR     = ROOT / "experiments/exp013_scaling_and_direction/figures"
REPORT_DIR  = ROOT / "docs/diagnosis"
DATA_DIR    = ROOT / "data/processed"
PERSONA_DIR = DATA_DIR / "personas"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── exp011 하이퍼파라미터 (동일) ─────────────────────────────────────────
HIDDEN_DIM   = 5120
N_WEEKS      = 17
BOTTLENECK   = 64
DROPOUT      = 0.5
EPOCHS       = 200
PATIENCE     = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-2
UNIFORM_ENT  = float(np.log(50))

# exp013 A 원본 결과 (run.log에서 추출)
EXP013_A_ORIG = {
    30:  {"mae": 71.94, "entropy": 0.1158},
    50:  {"mae": 65.59, "entropy": 0.2186},
    75:  {"mae": 59.66, "entropy": 0.2676},
    100: {"mae": 57.64, "entropy": 0.2168},
    150: {"mae": 59.63, "entropy": 0.2884},
    200: {"mae": 58.72, "entropy": 0.1251},
    250: {"mae": 62.53, "entropy": 0.0871},
}

# ─── 모델 ──────────────────────────────────────────────────────────────────

class AttnBottleneck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(HIDDEN_DIM, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, BOTTLENECK), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(BOTTLENECK, N_WEEKS),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(x).squeeze(-1)
        w = torch.softmax(scores, dim=-1)
        return self.head((x * w.unsqueeze(-1)).sum(1)), w


def train_model(X: torch.Tensor, y: torch.Tensor, seed: int) -> AttnBottleneck:
    torch.manual_seed(seed); np.random.seed(seed)
    m = AttnBottleneck()
    opt = Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, no_imp, state = float("inf"), 0, None
    m.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred, _ = m(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward(); opt.step()
        if loss.item() < best:
            best, no_imp = loss.item(), 0
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if state:
        m.load_state_dict(state)
    m.eval()
    return m


def eval_model(m: AttnBottleneck, X: torch.Tensor, y: np.ndarray,
               meta: pd.DataFrame) -> dict:
    m.eval()
    with torch.no_grad():
        pred, w = m(X)
    w_np = w.numpy(); pred_np = pred.numpy()
    mae = float(np.abs(pred_np - y).mean())
    def ent(v): v = np.clip(v, 1e-12, None); return -float(np.sum(v * np.log(v)))
    mean_ent  = float(np.mean([ent(w_np[i]) for i in range(len(w_np))]))
    p042_w    = float(w_np[:, 41].mean())
    top1      = w_np.argmax(axis=1)
    cat_vals  = meta["cat_id"].values
    modes = []
    cat_top1 = {}
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        mask = cat_vals == cat
        if not mask.any():
            modes.append(-1); cat_top1[cat] = {"top1_mode": -1, "consistency": 0.0}; continue
        t = top1[mask]; mode = int(np.bincount(t).argmax())
        cat_top1[cat] = {"top1_mode": mode, "consistency": float((t == mode).mean())}
        modes.append(mode)
    return {"mae": mae, "mean_entropy": mean_ent, "p042_weight": p042_w,
            "cat_top1": cat_top1, "cat_differentiated": len(set(modes)) == 3}

# ─── 데이터 로드 ──────────────────────────────────────────────────────────

def load_data() -> tuple[torch.Tensor, np.ndarray, pd.DataFrame]:
    logger.info("데이터 로드 중...")
    warm_raw  = torch.load(EMB_DIR / "warm_raw.pt",  weights_only=True)
    meta      = pd.read_csv(EMB_DIR / "item_meta.csv")
    warm_meta = meta[meta["is_cold"] == False].reset_index(drop=True)
    cold_test = pd.read_csv(DATA_DIR / "cold_start/cold_test.csv", parse_dates=["date"])
    warm_test = pd.read_csv(DATA_DIR / "cold_start/warm_test.csv", parse_dates=["date"])
    d0 = str(cold_test["date"].min().date()); d1 = str(cold_test["date"].max().date())
    warm_ids = warm_meta["item_id"].tolist()
    sub = warm_test[warm_test["item_id"].isin(warm_ids) &
                    (warm_test["date"] >= d0) & (warm_test["date"] <= d1)].copy()
    sub["week"] = pd.to_datetime(sub["date"]).dt.to_period("W")
    y = (sub.groupby(["item_id","week"])["sales"].sum()
           .unstack("week").reindex(warm_ids).fillna(0)).values.astype(np.float32)
    logger.info("warm_raw: %s  y: %s  mean=%.2f", tuple(warm_raw.shape), y.shape, y.mean())
    return warm_raw, y, warm_meta


def load_personas() -> list[dict]:
    return [json.loads(fp.read_text())["profile"]
            for fp in sorted(PERSONA_DIR.glob("CA_1_P*.json"))[:50]]


def make_test_pool_split(y_warm: np.ndarray) -> tuple[list[int], list[int]]:
    """exp013 A와 완전히 동일한 test/pool split (seed=0, stratified 50개)."""
    item_mean = y_warm.mean(axis=1)
    qlabels   = pd.qcut(pd.Series(item_mean), q=4,
                        labels=["Q1","Q2","Q3","Q4"]).to_numpy()
    rng = np.random.default_rng(0)
    test_idx = []
    for q in ["Q1","Q2","Q3","Q4"]:
        qi = np.where(qlabels == q)[0]
        test_idx.extend(rng.choice(qi, size=min(13, len(qi)), replace=False).tolist())
    test_idx = sorted(test_idx[:50])
    pool_idx = [i for i in range(300) if i not in set(test_idx)]
    return test_idx, pool_idx

# ─── 실험 D: Residual Scaling Law ─────────────────────────────────────────

def experiment_d(warm_raw: torch.Tensor, y_warm: np.ndarray,
                 warm_meta: pd.DataFrame) -> dict:
    logger.info("=== 실험 D: Residual Scaling Law ===")
    TRAIN_SIZES = [30, 50, 75, 100, 150, 200, 250]
    SEEDS       = [42, 123, 777]

    # Residual 생성
    item_mean    = warm_raw.mean(dim=1, keepdim=True)   # (300, 1, 5120)
    warm_residual = warm_raw - item_mean                  # (300, 50, 5120)

    test_idx, pool_idx = make_test_pool_split(y_warm)
    X_res_test = warm_residual[test_idx]
    y_test     = y_warm[test_idx]
    meta_test  = warm_meta.iloc[test_idx].reset_index(drop=True)
    logger.info("  테스트 %d개, 풀 %d개", len(test_idx), len(pool_idx))

    results: dict[int, list[dict]] = {n: [] for n in TRAIN_SIZES}
    for n_tr in TRAIN_SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            tr_idx = rng.choice(pool_idx, size=n_tr, replace=False).tolist()
            m = train_model(warm_residual[tr_idx], torch.tensor(y_warm[tr_idx]), seed)
            r = eval_model(m, X_res_test, y_test, meta_test)
            results[n_tr].append(r)
            logger.info("  [Residual] n=%d seed=%d | MAE=%.2f ent=%.4f P042=%.4f diff=%s",
                        n_tr, seed, r["mae"], r["mean_entropy"], r["p042_weight"],
                        r["cat_differentiated"])
    return results

# ─── 실험 E: 프롬프트 파일럿 (GPU) ────────────────────────────────────────

def _build_persona_struct(p: dict) -> str:
    cat_pref = ", ".join(f"{c}={v:.2f}" for c, v in
                         sorted(p["category_preference"].items(), key=lambda x: -x[1]))
    return (
        f"Customer profile:\n"
        f"- Weekly budget: ${p['weekly_budget']:.2f}\n"
        f"- SNAP eligible: {str(p['snap_eligible']).lower()}\n"
        f"- Economic status: {p['economic_status']}\n"
        f"- Shopping motivation: {p['shopping_motivation']}\n"
        f"- Category preference: {cat_pref}\n"
        f"- Price sensitivity: {p['price_sensitivity']}\n"
        f"- Visit frequency: {p['visit_frequency']}\n"
        f"- Preferred departments: {', '.join(p['preferred_departments'])}\n"
        f"- Decision style: {p['decision_style']}\n"
        f"- Brand loyalty: {p['brand_loyalty']}\n"
        f"- Promotion sensitivity: {p['promotion_sensitivity']}"
    )


def _v3(ps: str, dept: str, cat: str, price: float | None) -> str:
    pr = f"${price:.2f}" if price else "N/A"
    return (
        "<|im_start|>system\nYou are an expert consumer behavior analyst. "
        "Your task is to evaluate the match between a customer and a product.<|im_end|>\n"
        f"<|im_start|>user\n{ps}\n\nItem Profile:\n- Department: {dept}\n"
        f"- Category: {cat}\n- Average price: {pr}\n\n"
        "Task: Based on the customer profile, assess the probability of this customer "
        "purchasing this item. Answer with one of: Highly Likely, Likely, Neutral, "
        "Unlikely, Highly Unlikely.<|im_end|>\n<|im_start|>assistant\nThe answer is:"
    )


def _v4a(ps: str, desc: str, cat: str, price: float | None) -> str:
    """V4-A: 아이템 최소화 + 페르소나 서사 확장."""
    pr = f"${price:.2f}" if price else "N/A"
    return (
        "<|im_start|>system\nYou are an expert consumer behavior analyst. "
        "Your task is to evaluate how likely a specific customer is to purchase a product."
        "<|im_end|>\n"
        f"<|im_start|>user\nCustomer background: {desc}\n\n{ps}\n\n"
        f"Item Profile:\n- Category: {cat}\n- Average price: {pr}\n\n"
        "Task: Based on this customer's background and profile, assess the probability of "
        "this customer purchasing this item. Answer with one of: Highly Likely, Likely, "
        "Neutral, Unlikely, Highly Unlikely.<|im_end|>\n"
        "<|im_start|>assistant\nThe answer is:"
    )


def _v4b(ps: str, dept: str, cat: str, price: float | None) -> str:
    """V4-B: V3 + 대조 지시 추가."""
    pr = f"${price:.2f}" if price else "N/A"
    return (
        "<|im_start|>system\nYou are an expert consumer behavior analyst. "
        "Your task is to evaluate the match between a customer and a product.<|im_end|>\n"
        f"<|im_start|>user\n{ps}\n\nItem Profile:\n- Department: {dept}\n"
        f"- Category: {cat}\n- Average price: {pr}\n\n"
        "Task: Based on the customer profile, assess the probability of this customer "
        "purchasing this item. Answer with one of: Highly Likely, Likely, Neutral, "
        "Unlikely, Highly Unlikely. "
        "To ensure your response reflects this customer's unique identity, consider how "
        "their specific economic background and preferences distinguish their purchasing "
        "patterns from those of someone with entirely different circumstances."
        "<|im_end|>\n<|im_start|>assistant\nThe answer is:"
    )


def _quality(X: np.ndarray, personas: list[dict], label: str) -> dict:
    """(N_items, 50, 5120) → embedding quality metrics."""
    n_items = X.shape[0]
    cos_sims = []
    for i in range(n_items):
        v = X[i]; norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        vn = v / norms; sim = vn @ vn.T
        idx = np.triu_indices(50, k=1)
        cos_sims.append(sim[idx].mean())
    cos_mean = float(np.mean(cos_sims))

    budgets = np.array([p["weekly_budget"] for p in personas])
    si = np.argsort(budgets); lo = si[:10]; hi = si[-10:]
    l2_inc = [np.linalg.norm(X[i,hi].mean(0) - X[i,lo].mean(0)) for i in range(n_items)]
    l2_mean = float(np.mean(l2_inc))

    Xf = X.reshape(-1, HIDDEN_DIM)
    Xc = Xf - Xf.mean(0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    tot = (s**2).sum()
    top1  = float((s[0]**2) / tot)
    top10 = float((s[:10]**2).sum() / tot)

    logger.info("  [%s] cos=%.4f  incomeL2=%.2f  PCA top1=%.3f top10=%.3f",
                label, cos_mean, l2_mean, top1, top10)
    return {"cos_mean": cos_mean, "income_l2": l2_mean,
            "pca_top1": top1, "pca_top10": top10,
            "pca_curve": (np.cumsum(s**2) / tot)[:50].tolist()}


def experiment_e(warm_raw: torch.Tensor, warm_meta: pd.DataFrame,
                 personas: list[dict]) -> dict:
    """GPU 필요. 5개 파일럿 아이템 × 50 페르소나 × 3 variants."""
    import os
    logger.info("=== 실험 E: 프롬프트 파일럿 (GPU) ===")

    # 5개 파일럿 아이템 선택: FOODS×2, HOBBIES×2, HOUSEHOLD×1 (판매량 중위)
    item_mean_arr = warm_raw.mean(dim=(1, 2)).numpy()
    pilot_idx = []
    for cat, n in [("FOODS", 2), ("HOBBIES", 2), ("HOUSEHOLD", 1)]:
        mask = (warm_meta["cat_id"] == cat).values
        ci = np.where(mask)[0]
        sel = ci[np.argsort(np.abs(item_mean_arr[ci] - np.median(item_mean_arr[ci])))[:n]]
        pilot_idx.extend(sel.tolist())
    pilot_idx = pilot_idx[:5]
    pilot_meta = warm_meta.iloc[pilot_idx].reset_index(drop=True)
    logger.info("  파일럿 아이템: %s",
                pilot_meta[["item_id", "cat_id"]].to_dict("records"))

    # sell_prices (avg_price) — M5 원본 파일에서 CA_1 기준으로 계산
    sell_df   = pd.read_csv(ROOT / "m5-forecasting-accuracy/sell_prices.csv")
    price_map = (sell_df[sell_df["store_id"] == "CA_1"]
                 .groupby("item_id")["sell_price"].mean().to_dict())

    # V3: warm_raw.pt 슬라이스
    X_v3 = warm_raw[pilot_idx].numpy()   # (5, 50, 5120)
    q_v3 = _quality(X_v3, personas, "V3")

    # Qwen 모델 로드
    hf_home = os.environ.get("HF_HOME", "/mnt/sdd1/jylee/huggingface_cache")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    logger.info("  Qwen 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        quantization_config=bnb, device_map="auto",
        trust_remote_code=True, output_hidden_states=True,
        torch_dtype=torch.bfloat16, cache_dir=hf_home,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True, cache_dir=hf_home)
    model.eval()
    logger.info("  Qwen 로드 완료")

    def extract(texts: list[str], bs: int = 4, ml: int = 512) -> np.ndarray:
        """last-token, layer[-2] (V3 전략과 동일)."""
        embs = []
        for i in range(0, len(texts), bs):
            enc = tokenizer(texts[i:i+bs], padding=True, truncation=True,
                            max_length=ml, return_tensors="pt",
                            add_special_tokens=False)
            ids  = enc["input_ids"].to(model.device)
            mask = enc["attention_mask"].to(ids.device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask)
            sl = out.hidden_states[-2]   # (B, L, H)
            lens = mask.sum(dim=1) - 1
            for bi, l in enumerate(lens):
                embs.append(sl[bi, l].cpu().float().numpy())
            if (i // bs + 1) % 10 == 0:
                logger.info("    %d / %d texts done", i + len(texts[i:i+bs]), len(texts))
        return np.stack(embs)

    def build_and_extract(build_fn) -> np.ndarray:
        texts = []
        for row in pilot_meta.itertuples():
            price = price_map.get(row.item_id, None)
            for p in personas:
                texts.append(build_fn(p, row.dept_id, row.cat_id, price))
        logger.info("  텍스트 %d개 추출 시작...", len(texts))
        return extract(texts).reshape(5, 50, HIDDEN_DIM)

    # V4-A
    logger.info("  V4-A 추출...")
    X_v4a = build_and_extract(
        lambda p, dept, cat, price: _v4a(_build_persona_struct(p), p["description"], cat, price))
    q_v4a = _quality(X_v4a, personas, "V4-A")

    # V4-B
    logger.info("  V4-B 추출...")
    X_v4b = build_and_extract(
        lambda p, dept, cat, price: _v4b(_build_persona_struct(p), dept, cat, price))
    q_v4b = _quality(X_v4b, personas, "V4-B")

    # PCA curves
    return {
        "v3": q_v3, "v4a": q_v4a, "v4b": q_v4b,
        "pilot_items": pilot_meta[["item_id", "cat_id"]].to_dict("records"),
    }

# ─── 시각화 ────────────────────────────────────────────────────────────────

def plot_d(results_d: dict) -> None:
    TRAIN_SIZES = sorted(results_d.keys())
    orig_mae = [EXP013_A_ORIG[n]["mae"]     for n in TRAIN_SIZES]
    orig_ent = [EXP013_A_ORIG[n]["entropy"] for n in TRAIN_SIZES]
    res_mae  = [np.mean([r["mae"]          for r in results_d[n]]) for n in TRAIN_SIZES]
    res_mae_s= [np.std( [r["mae"]          for r in results_d[n]]) for n in TRAIN_SIZES]
    res_ent  = [np.mean([r["mean_entropy"] for r in results_d[n]]) for n in TRAIN_SIZES]
    res_ent_s= [np.std( [r["mean_entropy"] for r in results_d[n]]) for n in TRAIN_SIZES]
    res_p42  = [np.mean([r["p042_weight"]  for r in results_d[n]]) for n in TRAIN_SIZES]
    res_diff = [np.mean([r["cat_differentiated"] for r in results_d[n]]) for n in TRAIN_SIZES]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) MAE 비교
    axes[0].plot(TRAIN_SIZES, orig_mae, "o-", color="steelblue", label="Original (exp013 A)")
    axes[0].errorbar(TRAIN_SIZES, res_mae, yerr=res_mae_s, fmt="s--",
                     color="orangered", capsize=4, label="Residual (exp014 D)")
    axes[0].set_xlabel("Train set size"); axes[0].set_ylabel("Test MAE")
    axes[0].set_title("(D) Original vs Residual MAE")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # (2) Entropy 비교
    axes[1].plot(TRAIN_SIZES, orig_ent, "o-", color="steelblue", label="Original")
    axes[1].errorbar(TRAIN_SIZES, res_ent, yerr=res_ent_s, fmt="s--",
                     color="orangered", capsize=4, label="Residual")
    axes[1].axhline(UNIFORM_ENT, color="green", ls="--",
                    label=f"uniform={UNIFORM_ENT:.2f}")
    axes[1].set_xlabel("Train set size"); axes[1].set_ylabel("Mean attention entropy")
    axes[1].set_title("(D) Attention Entropy Comparison"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # (3) P042 + 카테고리 분화율
    ax3b = axes[2].twinx()
    axes[2].errorbar(TRAIN_SIZES, res_p42, fmt="^-", color="purple",
                     capsize=4, label="Residual P042 weight")
    axes[2].axhline(1/50, color="gray", ls="--", label="uniform=0.02", alpha=0.6)
    ax3b.plot(TRAIN_SIZES, res_diff, "D--", color="seagreen", label="Cat. diff. rate (right)")
    axes[2].set_xlabel("Train set size"); axes[2].set_ylabel("P042 weight", color="purple")
    ax3b.set_ylabel("Differentiation rate", color="seagreen")
    axes[2].set_title("(D) Residual: P042 & Category Differentiation")
    axes[2].legend(loc="upper left"); ax3b.legend(loc="upper right"); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_d_residual_scaling.png", dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", FIG_DIR / "exp_d_residual_scaling.png")


def plot_e(res_e: dict) -> None:
    variants = ["v3", "v4a", "v4b"]
    labels   = ["V3 (원본)", "V4-A (Persona-Heavy)", "V4-B (Contrastive)"]
    colors   = ["steelblue", "orangered", "seagreen"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Bar chart: cos sim + income L2
    x   = np.arange(3)
    cos = [res_e[v]["cos_mean"]  for v in variants]
    l2  = [res_e[v]["income_l2"] for v in variants]
    axes[0].bar(x - 0.2, cos, 0.4, label="Pairwise cosine sim", color=colors, alpha=0.8)
    ax0b = axes[0].twinx()
    ax0b.bar(x + 0.2, l2, 0.4, label="Income L2 distance",
             color=[c + "80" for c in ["#4472c4","#c0504d","#9bbb59"]], alpha=0.7,
             edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylabel("Pairwise cosine similarity"); ax0b.set_ylabel("Income L2 distance")
    axes[0].set_title("(E) 프롬프트 변형 품질 비교")
    axes[0].axhline(0.95, color="red", ls="--", lw=1, label="cos=0.95 threshold")
    axes[0].legend(loc="upper right")

    # (2) PCA cumulative curve
    for v, label, color in zip(variants, labels, colors):
        curve = res_e[v]["pca_curve"]
        axes[1].plot(range(1, len(curve)+1), curve, marker=".", label=label, color=color)
    axes[1].set_xlabel("Number of PCs"); axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("(E) PCA Cumulative Variance: V3 vs V4-A vs V4-B")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_e_prompt_pilot.png", dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", FIG_DIR / "exp_e_prompt_pilot.png")

# ─── 보고서 ────────────────────────────────────────────────────────────────

def write_report(results_d: dict, res_e: dict | None) -> None:
    TRAIN_SIZES = sorted(results_d.keys())

    lines = [
        "# Residual Scaling Law + 프롬프트 파일럿 보고서 (Exp014)",
        "",
        "**작성일:** 2026-03-10",
        "**배경:** exp013 C-1에서 MAE 동일 → 방향 정보 불필요(해석 1) vs 데이터 부족(해석 2) 구분",
        "",
        "---", "",
        "## 실험 D: Residual Embedding Scaling Law", "",
        "### 설계",
        "- exp013 실험 A와 동일한 train/test split (seed=0 stratified 50개)",
        "- 입력: warm_residual = warm_raw - mean(warm_raw, dim=personas)",
        "- 학습 크기: [30, 50, 75, 100, 150, 200, 250] × seeds [42, 123, 777]",
        "",
        "### 원본 vs Residual 비교 테이블 (mean ± std)",
        "",
        "| Train N | 원본 MAE | Residual MAE | 원본 Entropy | Residual Entropy | Res Cat.Diff |",
        "|---------|---------|-------------|-------------|-----------------|--------------|",
    ]
    for n in TRAIN_SIZES:
        orig_m  = EXP013_A_ORIG[n]["mae"]
        orig_e  = EXP013_A_ORIG[n]["entropy"]
        res_m   = np.mean([r["mae"]          for r in results_d[n]])
        res_ms  = np.std( [r["mae"]          for r in results_d[n]])
        res_ent = np.mean([r["mean_entropy"] for r in results_d[n]])
        res_ens = np.std( [r["mean_entropy"] for r in results_d[n]])
        diff    = np.mean([r["cat_differentiated"] for r in results_d[n]])
        lines.append(
            f"| {n:4d} | {orig_m:.1f} | {res_m:.1f}±{res_ms:.1f} | "
            f"{orig_e:.4f} | {res_ent:.4f}±{res_ens:.4f} | {diff:.2f} |"
        )

    # 판정 로직
    mae_100 = np.mean([r["mae"] for r in results_d[100]])
    mae_250 = np.mean([r["mae"] for r in results_d[250]])
    mae_30  = np.mean([r["mae"] for r in results_d[30]])
    plateau_threshold = (mae_100 - mae_30) * 0.2   # 100 이후 20% 이상 더 감소하면 "계속 감소"

    if mae_250 < mae_100 - abs(plateau_threshold):
        verdict_d = "해석 2 지지 — residual에서 N=100 이후에도 계속 개선 → warm 확장 + residual 조합 유망"
        warm_expand_expected = "warm 300→2949: MAE ~{:.0f} 예상 (현재 {:.1f}에서 ~{:.0f}% 감소)".format(
            mae_250 * 0.7, mae_250, 30)
    else:
        verdict_d = "해석 1 지지 — residual에서도 정체 → 방향 정보 자체가 예측에 불필요, embedding 정보량 한계"
        warm_expand_expected = "warm 확장의 기대 효과 제한적 (현재 {:.1f}에서 소폭 개선 예상)".format(mae_250)

    lines += [
        "",
        f"→ N=100: Residual MAE={mae_100:.1f}  N=250: {mae_250:.1f}  (감소 폭: {mae_100 - mae_250:.1f})",
        f"→ **판정: {verdict_d}**",
        "",
        "---", "",
    ]

    # 실험 E
    lines += ["## 실험 E: 프롬프트 파일럿 (V4-A / V4-B)", ""]
    if res_e is not None:
        pilot_str = ", ".join(
            f"{r['item_id']}({r['cat_id']})" for r in res_e["pilot_items"]
        )
        lines += [
            f"**파일럿 아이템 5개:** {pilot_str}",
            "",
            "| 지표 | V3 (기존) | V4-A (Persona-Heavy) | V4-B (Contrastive) |",
            "|------|----------|---------------------|-------------------|",
            f"| pairwise cosine sim | **{res_e['v3']['cos_mean']:.4f}** | "
            f"**{res_e['v4a']['cos_mean']:.4f}** | **{res_e['v4b']['cos_mean']:.4f}** |",
            f"| income L2 distance | {res_e['v3']['income_l2']:.2f} | "
            f"{res_e['v4a']['income_l2']:.2f} | {res_e['v4b']['income_l2']:.2f} |",
            f"| PCA top-1 PC | {res_e['v3']['pca_top1']:.3f} | "
            f"{res_e['v4a']['pca_top1']:.3f} | {res_e['v4b']['pca_top1']:.3f} |",
            f"| PCA top-10 PC | {res_e['v3']['pca_top10']:.3f} | "
            f"{res_e['v4a']['pca_top10']:.3f} | {res_e['v4b']['pca_top10']:.3f} |",
            "",
        ]
        best_v = "V4-A" if res_e["v4a"]["cos_mean"] < res_e["v4b"]["cos_mean"] else "V4-B"
        best_cos = min(res_e["v4a"]["cos_mean"], res_e["v4b"]["cos_mean"])
        v3_cos   = res_e["v3"]["cos_mean"]
        drop     = v3_cos - best_cos

        if best_cos < 0.95:
            verdict_e = f"{best_v}가 cosine sim을 {best_cos:.4f}로 낮춤 (V3 {v3_cos:.4f}에서 {drop:.4f} 감소) → 전체 재추출 가치 있음"
            prompt_expected = f"V3→{best_v} 재추출: cosine sim {v3_cos:.4f}→{best_cos:.4f} → attention market modeling 개선 기대"
        elif drop > 0.01:
            verdict_e = f"{best_v}가 일부 효과 있으나 0.95 임계 미달 ({best_cos:.4f}) → 제한적 개선"
            prompt_expected = f"V3→{best_v} 재추출: 제한적 개선 기대 (cosine {v3_cos:.4f}→{best_cos:.4f})"
        else:
            verdict_e = "V4-A/V4-B 모두 효과 미미 → 프롬프트 수준 변경 한계, 모델/레이어 변경 필요"
            prompt_expected = "프롬프트 변형 재추출: 효과 미미 — 투자 가치 낮음"
    else:
        lines += [
            "실험 E는 GPU가 필요합니다. `--run-e` 플래그로 재실행하세요:",
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=0,1,2,3 \\",
            "HF_HOME=/mnt/sdd1/jylee/huggingface_cache \\",
            "conda run --no-capture-output -n persona-forecasting \\",
            "  python scripts/exp014_residual_prompt_pilot.py --run-e \\",
            "  2>&1 | tee experiments/exp013_scaling_and_direction/exp014_run.log",
            "```",
            "",
        ]
        verdict_e  = "미실행"
        prompt_expected = "미실행 (GPU 필요)"

    lines += ["→ **판정: {}**".format(verdict_e), "", "---", ""]

    # 종합 판정 + 투자 대비 기대값
    lines += [
        "## 종합 판정 및 경로 선택",
        "",
        "### 두 경로의 기대 효과 (수치 기반)",
        "",
        "| 경로 | 설명 | 기대 MAE 개선 | 근거 |",
        "|------|------|--------------|------|",
        f"| **경로 A: warm 확장** (300→2,949) | residual 조합 | {warm_expand_expected} | 실험 D 결과 |",
        f"| **경로 B: 프롬프트 재추출** (V4) | V4-A 또는 V4-B | {prompt_expected} | 실험 E 결과 |",
        "",
    ]

    if res_e is None:
        lines.append("→ 실험 E 미실행으로 최종 경로 선택 불가. E 결과 후 재판정 필요.")
    elif best_cos < 0.95 and "해석 2" in verdict_d:
        lines += [
            "→ **두 경로 모두 유망**: warm 확장(경로 A)과 프롬프트 재추출(경로 B) 병행 권고",
            f"   - D: residual + warm 확장 → MAE 지속 개선 가능",
            f"   - E: {best_v} 재추출 → cosine sim 감소 → attention market modeling 강화",
            "→ **우선순위: 경로 A** (residual 후처리 비용 0 + warm 확장은 데이터 엔지니어링)",
        ]
    elif "해석 2" in verdict_d:
        lines += [
            "→ **경로 A 우선**: warm 확장이 기대 효과 더 높음 (프롬프트 변형 효과 제한적)",
        ]
    elif best_cos < 0.95:
        lines += [
            f"→ **경로 B 우선**: {best_v} 재추출이 기대 효과 더 높음 (warm 확장 효과 제한적)",
        ]
    else:
        lines += [
            "→ **두 경로 모두 제한적**: embedding 정보량 자체의 한계 → 모델/레이어 변경 검토 필요",
        ]

    lines += [
        "",
        "---",
        "",
        "**시각화:** `experiments/exp013_scaling_and_direction/figures/exp_d_*.png`, `exp_e_*.png`",
        "**스크립트:** `scripts/exp014_residual_prompt_pilot.py`",
    ]

    path = REPORT_DIR / "residual_scaling_and_prompt_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", path)

# ─── 메인 ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-e", action="store_true", help="실험 E (GPU) 실행")
    args = parser.parse_args()

    logger.info("=== Exp014: Residual Scaling + 프롬프트 파일럿 ===")

    warm_raw, y_warm, warm_meta = load_data()
    personas = load_personas()

    # D 먼저 실행 (GPU 불필요)
    results_d = experiment_d(warm_raw, y_warm, warm_meta)
    plot_d(results_d)

    # E (GPU, optional)
    res_e = None
    if args.run_e:
        res_e = experiment_e(warm_raw, warm_meta, personas)
        plot_e(res_e)

    write_report(results_d, res_e)

    # 터미널 요약
    TRAIN_SIZES = sorted(results_d.keys())
    print("\n" + "=" * 65)
    print("=== Exp014 결과 요약 ===")
    print()
    print(f"{'N':>4}  {'원본 MAE':>9}  {'Residual MAE':>13}  {'Res Entropy':>12}  {'Res Diff':>9}")
    for n in TRAIN_SIZES:
        res_m  = np.mean([r["mae"]          for r in results_d[n]])
        res_ms = np.std( [r["mae"]          for r in results_d[n]])
        res_e_ = np.mean([r["mean_entropy"] for r in results_d[n]])
        diff   = np.mean([r["cat_differentiated"] for r in results_d[n]])
        orig   = EXP013_A_ORIG[n]["mae"]
        print(f"{n:>4}  {orig:>9.1f}  {res_m:>9.1f}±{res_ms:<3.1f}  {res_e_:>12.4f}  {diff:>9.2f}")
    if res_e:
        print()
        print("E (프롬프트):")
        for v, label in [("v3","V3"), ("v4a","V4-A"), ("v4b","V4-B")]:
            print(f"  {label}: cos={res_e[v]['cos_mean']:.4f}  L2={res_e[v]['income_l2']:.2f}  "
                  f"PCA_top1={res_e[v]['pca_top1']:.3f}")
    print("=" * 65)
    print("보고서: docs/diagnosis/residual_scaling_and_prompt_report.md")


if __name__ == "__main__":
    main()
