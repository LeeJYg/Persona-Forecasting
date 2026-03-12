"""Exp013: Warm Scaling Law + 분포 불일치 + Embedding 방향 분리 검증.

실험 A: 학습 셋 크기별 Attention 분화 (Scaling Law)
실험 B: 분포 불일치 영향 정량화
실험 C-1: Item-Mean Subtraction (즉시 실행, GPU 불필요)
실험 C-2: 프롬프트 변형 파일럿 (GPU 필요, --run-c2 플래그)

사용 데이터:
    A/B/C-1: warm_raw.pt (300×50×5120), y_warm (재계산)
    C-2:     소규모 신규 임베딩 추출 (Qwen2.5-32B-Instruct, 4-bit, 5아이템×50페르소나)

실행:
    GPU 불필요: conda run -n persona-forecasting python scripts/exp013_scaling_and_direction.py
    C-2 포함:  CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=... python ... --run-c2

출력:
    scripts/exp013_scaling_and_direction.py
    experiments/exp013_scaling_and_direction/figures/
    docs/diagnosis/scaling_and_direction_report.md
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

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
EMB_DIR    = ROOT / "experiments/exp011_v3_pipeline/embeddings"
FIG_DIR    = ROOT / "experiments/exp013_scaling_and_direction/figures"
REPORT_DIR = ROOT / "docs/diagnosis"
DATA_DIR   = ROOT / "data/processed"
PERSONA_DIR = DATA_DIR / "personas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── exp011 Attention+Bottleneck 하이퍼파라미터 ──────────────────────────────
HIDDEN_DIM   = 5120
N_WEEKS      = 17
BOTTLENECK   = 64
DROPOUT      = 0.5
EPOCHS       = 200
PATIENCE     = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-2
UNIFORM_ENT  = float(np.log(50))   # log(50) ≈ 3.912

# ─── V3 프롬프트 (exp011과 동일) ─────────────────────────────────────────────

_PERSONA_STRUCT_TEMPLATE = """\
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


def _build_persona_struct(profile: dict) -> str:
    """Condition A: 구조화 필드만 (description 제외)."""
    cat_pref = ", ".join(
        f"{cat}={pct:.2f}"
        for cat, pct in sorted(
            profile["category_preference"].items(), key=lambda x: -x[1]
        )
    )
    return _PERSONA_STRUCT_TEMPLATE.format(
        weekly_budget=profile["weekly_budget"],
        snap_eligible=str(profile["snap_eligible"]).lower(),
        economic_status=profile["economic_status"],
        shopping_motivation=profile["shopping_motivation"],
        cat_pref=cat_pref,
        price_sensitivity=profile["price_sensitivity"],
        visit_frequency=profile["visit_frequency"],
        preferred_departments=", ".join(profile["preferred_departments"]),
        decision_style=profile["decision_style"],
        brand_loyalty=profile["brand_loyalty"],
        promotion_sensitivity=profile["promotion_sensitivity"],
    )


def build_v3_text(persona_struct: str, dept_id: str, cat_id: str,
                  avg_price: float | None) -> str:
    """V3 프롬프트 (exp011과 완전 동일)."""
    price_str = f"${avg_price:.2f}" if avg_price is not None else "N/A"
    return (
        "<|im_start|>system\n"
        "You are an expert consumer behavior analyst. "
        "Your task is to evaluate the match between a customer and a product."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{persona_struct}\n\n"
        f"Item Profile:\n"
        f"- Department: {dept_id}\n"
        f"- Category: {cat_id}\n"
        f"- Average price: {price_str}\n\n"
        "Task: Based on the customer profile, assess the probability of this customer "
        "purchasing this item. Answer with one of: Highly Likely, Likely, Neutral, "
        "Unlikely, Highly Unlikely."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "The answer is:"
    )


def build_v4a_text(persona_struct: str, description: str,
                   cat_id: str, avg_price: float | None) -> str:
    """V4-A (Persona-Heavy): 아이템 정보 최소화 + 페르소나 서사 확장.

    변경점 vs V3:
    - 아이템: 카테고리 + 가격만 (dept_id, item_id 제거)
    - 페르소나: description 서술형 추가 (structured fields 앞에 배치)
    - 지시: 구매 의향과 예상 수량 강조
    """
    price_str = f"${avg_price:.2f}" if avg_price is not None else "N/A"
    return (
        "<|im_start|>system\n"
        "You are an expert consumer behavior analyst. "
        "Your task is to evaluate how likely a specific customer is to purchase a product."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Customer background: {description}\n\n"
        f"{persona_struct}\n\n"
        f"Item Profile:\n"
        f"- Category: {cat_id}\n"
        f"- Average price: {price_str}\n\n"
        "Task: Based on this customer's background and profile, assess the probability of "
        "this customer purchasing this item. Answer with one of: Highly Likely, Likely, "
        "Neutral, Unlikely, Highly Unlikely."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "The answer is:"
    )


def build_v4b_text(persona_struct: str, dept_id: str, cat_id: str,
                   avg_price: float | None) -> str:
    """V4-B (Contrastive Prompt): V3 + 대조 지시 추가.

    변경점 vs V3:
    - 프롬프트 끝에 페르소나 차이 강조 지시 추가
    - 목적: LLM이 페르소나별 고유 판단을 더 강하게 활성화하도록 유도
    """
    price_str = f"${avg_price:.2f}" if avg_price is not None else "N/A"
    return (
        "<|im_start|>system\n"
        "You are an expert consumer behavior analyst. "
        "Your task is to evaluate the match between a customer and a product."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{persona_struct}\n\n"
        f"Item Profile:\n"
        f"- Department: {dept_id}\n"
        f"- Category: {cat_id}\n"
        f"- Average price: {price_str}\n\n"
        "Task: Based on the customer profile, assess the probability of this customer "
        "purchasing this item. Answer with one of: Highly Likely, Likely, Neutral, "
        "Unlikely, Highly Unlikely. "
        "To ensure your response reflects this customer's unique identity, consider how "
        "their specific economic background and preferences distinguish their purchasing "
        "patterns from those of someone with entirely different circumstances."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "The answer is:"
    )


# ─── 공통 모델 ────────────────────────────────────────────────────────────────

class AttnBottleneck(nn.Module):
    """exp011 3-4_attn_bottleneck와 동일한 구조."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM,
                 n_weeks: int = N_WEEKS,
                 bottleneck: int = BOTTLENECK,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(x).squeeze(-1)
        attn_w = torch.softmax(scores, dim=-1)
        fused  = (x * attn_w.unsqueeze(-1)).sum(1)
        return self.head(fused), attn_w


def train_attn_model(X_raw: torch.Tensor, y: torch.Tensor,
                     seed: int = 42) -> AttnBottleneck:
    torch.manual_seed(seed); np.random.seed(seed)
    model = AttnBottleneck()
    opt   = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss, no_imp, best_state = float("inf"), 0, None
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred, _ = model(X_raw)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def compute_metrics(model: AttnBottleneck, X_test: torch.Tensor,
                    y_test: np.ndarray, test_meta: pd.DataFrame,
                    p042_idx: int = 41) -> dict:
    model.eval()
    with torch.no_grad():
        pred, attn_w = model(X_test)
    attn_np = attn_w.cpu().numpy()
    pred_np = pred.cpu().numpy()
    mae = float(np.abs(pred_np - y_test).mean())
    def entropy(w): w = np.clip(w, 1e-12, None); return -float(np.sum(w * np.log(w)))
    mean_ent = float(np.mean([entropy(attn_np[i]) for i in range(len(attn_np))]))
    p042_w   = float(attn_np[:, p042_idx].mean())
    top1  = attn_np.argmax(axis=1)
    cats  = test_meta["cat_id"].values
    cat_top1 = {}
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        mask = cats == cat
        if mask.sum() == 0:
            cat_top1[cat] = {"top1_mode": -1, "consistency": 0.0}
            continue
        t = top1[mask]
        m = int(np.bincount(t).argmax())
        cat_top1[cat] = {"top1_mode": m, "consistency": float((t == m).mean())}
    modes = [cat_top1[c]["top1_mode"] for c in ["FOODS", "HOBBIES", "HOUSEHOLD"]]
    return {"mae": mae, "mean_entropy": mean_ent, "p042_weight": p042_w,
            "cat_top1": cat_top1, "cat_differentiated": len(set(modes)) == 3}


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data() -> tuple[torch.Tensor, np.ndarray, pd.DataFrame]:
    logger.info("데이터 로드 중...")
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", weights_only=True)
    meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    warm_meta = meta[meta["is_cold"] == False].reset_index(drop=True)
    cold_test = pd.read_csv(DATA_DIR / "cold_start/cold_test.csv", parse_dates=["date"])
    warm_test = pd.read_csv(DATA_DIR / "cold_start/warm_test.csv", parse_dates=["date"])
    d0 = str(cold_test["date"].min().date()); d1 = str(cold_test["date"].max().date())
    warm_ids = warm_meta["item_id"].tolist()
    sub = warm_test[warm_test["item_id"].isin(warm_ids) &
                    (warm_test["date"] >= d0) & (warm_test["date"] <= d1)].copy()
    sub["week"] = pd.to_datetime(sub["date"]).dt.to_period("W")
    y = (sub.groupby(["item_id", "week"])["sales"].sum()
           .unstack("week").reindex(warm_ids).fillna(0)).values.astype(np.float32)
    logger.info("warm_raw: %s  y_warm: %s  mean=%.2f", tuple(warm_raw.shape), y.shape, y.mean())
    return warm_raw, y, warm_meta


def load_personas(n: int = 50) -> list[dict]:
    files = sorted(PERSONA_DIR.glob("CA_1_P*.json"))[:n]
    return [json.loads(fp.read_text())["profile"] for fp in files]


# ─── 실험 A: Scaling Law ──────────────────────────────────────────────────────

def experiment_a(warm_raw: torch.Tensor, y_warm: np.ndarray,
                 warm_meta: pd.DataFrame) -> dict:
    logger.info("=== 실험 A: Scaling Law 시작 ===")
    TRAIN_SIZES = [30, 50, 75, 100, 150, 200, 250]
    SEEDS       = [42, 123, 777]

    item_mean = y_warm.mean(axis=1)
    qlabels   = pd.qcut(pd.Series(item_mean), q=4, labels=["Q1","Q2","Q3","Q4"]).to_numpy()
    rng_t = np.random.default_rng(0)
    test_idx = []
    for q in ["Q1","Q2","Q3","Q4"]:
        qi = np.where(qlabels == q)[0]
        test_idx.extend(rng_t.choice(qi, size=min(13, len(qi)), replace=False).tolist())
    test_idx = sorted(test_idx[:50])
    pool_idx = [i for i in range(300) if i not in set(test_idx)]
    X_test    = warm_raw[test_idx]
    y_test    = y_warm[test_idx]
    meta_test = warm_meta.iloc[test_idx].reset_index(drop=True)
    logger.info("  테스트 %d개, 풀 %d개 | 카테고리: %s",
                len(test_idx), len(pool_idx), dict(meta_test["cat_id"].value_counts()))

    results: dict[int, list[dict]] = {n: [] for n in TRAIN_SIZES}
    for n_tr in TRAIN_SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            tr_idx = rng.choice(pool_idx, size=n_tr, replace=False).tolist()
            model = train_attn_model(warm_raw[tr_idx], torch.tensor(y_warm[tr_idx]), seed=seed)
            m = compute_metrics(model, X_test, y_test, meta_test)
            results[n_tr].append(m)
            logger.info("  n=%d seed=%d | MAE=%.2f ent=%.4f P042=%.4f diff=%s",
                        n_tr, seed, m["mae"], m["mean_entropy"], m["p042_weight"],
                        m["cat_differentiated"])
    return {"results": results, "test_idx": test_idx, "pool_idx": pool_idx,
            "meta_test": meta_test}


# ─── 실험 B: 분포 불일치 ──────────────────────────────────────────────────────

def experiment_b(warm_raw: torch.Tensor, y_warm: np.ndarray,
                 warm_meta: pd.DataFrame) -> dict:
    logger.info("=== 실험 B: 분포 불일치 시작 ===")
    SEEDS  = [42, 123, 777]
    q_labs = pd.qcut(pd.Series(y_warm.mean(axis=1)), q=4,
                     labels=["Q1","Q2","Q3","Q4"]).to_numpy()
    q_idx  = {q: np.where(q_labs == q)[0] for q in ["Q1","Q2","Q3","Q4"]}
    results: dict[str, list[dict]] = {"B1": [], "B2": [], "B3": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        high = np.concatenate([q_idx["Q2"], q_idx["Q3"], q_idx["Q4"]])
        tr   = rng.choice(high, size=200, replace=False)
        X_tr = warm_raw[tr]; y_tr = torch.tensor(y_warm[tr])
        # B-1
        rem   = np.setdiff1d(high, tr)
        te_b1 = rng.choice(rem, size=min(50, len(rem)), replace=False)
        m1 = train_attn_model(X_tr, y_tr, seed=seed)
        results["B1"].append(compute_metrics(m1, warm_raw[te_b1], y_warm[te_b1],
                                             warm_meta.iloc[te_b1].reset_index(drop=True)))
        # B-2 (same model as B-1)
        te_b2 = q_idx["Q1"]
        results["B2"].append(compute_metrics(m1, warm_raw[te_b2], y_warm[te_b2],
                                             warm_meta.iloc[te_b2].reset_index(drop=True)))
        # B-3
        low   = np.concatenate([q_idx["Q1"], q_idx["Q2"], q_idx["Q3"]])
        tr_b3 = rng.choice(low, size=200, replace=False)
        te_b3 = q_idx["Q4"]
        m3 = train_attn_model(warm_raw[tr_b3], torch.tensor(y_warm[tr_b3]), seed=seed)
        results["B3"].append(compute_metrics(m3, warm_raw[te_b3], y_warm[te_b3],
                                             warm_meta.iloc[te_b3].reset_index(drop=True)))
        logger.info("  seed=%d | B1=%.2f B2=%.2f B3=%.2f",
                    seed, results["B1"][-1]["mae"], results["B2"][-1]["mae"],
                    results["B3"][-1]["mae"])
    return results


# ─── 실험 C-1: Item-Mean Subtraction ─────────────────────────────────────────

def experiment_c1(warm_raw: torch.Tensor, cold_raw: torch.Tensor,
                  y_warm: np.ndarray, warm_meta: pd.DataFrame,
                  personas: list[dict]) -> dict:
    """
    Item-mean subtraction: warm_raw[i] -= mean(warm_raw[i], dim=personas)
    → 순수 페르소나 간 변동 성분만 남김
    남긴 후 코사인 유사도, L2 distance, PCA, AttnBottleneck 학습 비교
    """
    logger.info("=== 실험 C-1: Item-Mean Subtraction ===")

    # Residual 생성
    X = warm_raw.clone()              # (300, 50, 5120)
    Xc = cold_raw.clone()            # (100, 50, 5120)
    item_mean_warm = X.mean(dim=1, keepdim=True)       # (300, 1, 5120)
    item_mean_cold = Xc.mean(dim=1, keepdim=True)      # (100, 1, 5120)
    warm_residual = X - item_mean_warm                  # (300, 50, 5120)
    cold_residual = Xc - item_mean_cold                 # (100, 50, 5120)

    def pairwise_cos_per_item(X_raw: torch.Tensor, n_sample: int = 100) -> np.ndarray:
        """아이템별 50개 페르소나 간 pairwise cosine similarity의 mean."""
        X_np = X_raw.numpy()
        n_items = min(X_np.shape[0], n_sample)
        scores = []
        for i in range(n_items):
            v = X_np[i]; norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            vn = v / norms; sim = vn @ vn.T
            idx = np.triu_indices(50, k=1)
            scores.append(sim[idx].mean())
        return np.array(scores)

    # (1) 코사인 유사도 비교
    logger.info("  (1) 코사인 유사도 분석...")
    cos_orig_warm = pairwise_cos_per_item(X, 300)
    cos_res_warm  = pairwise_cos_per_item(warm_residual, 300)
    cos_orig_cold = pairwise_cos_per_item(Xc, 100)
    cos_res_cold  = pairwise_cos_per_item(cold_residual, 100)
    logger.info("  원본  warm cos: mean=%.4f  residual warm: mean=%.4f",
                cos_orig_warm.mean(), cos_res_warm.mean())
    logger.info("  원본  cold cos: mean=%.4f  residual cold: mean=%.4f",
                cos_orig_cold.mean(), cos_res_cold.mean())

    # (2) income 기반 L2 distance (exp012 (b)와 동일)
    budgets  = np.array([p["weekly_budget"] for p in personas])
    sort_idx = np.argsort(budgets)
    low_idx  = sort_idx[:10]; high_idx = sort_idx[-10:]
    rng = np.random.default_rng(42)

    def income_vs_random_l2(X_raw: torch.Tensor) -> dict:
        X_np = X_raw.numpy(); n_items = X_np.shape[0]
        inc_l2 = []
        for i in range(n_items):
            d = X_np[i, high_idx].mean(0) - X_np[i, low_idx].mean(0)
            inc_l2.append(np.linalg.norm(d))
        inc_l2 = np.array(inc_l2)
        rand_l2 = []
        for _ in range(100):
            perm = rng.permutation(50)
            a, b = perm[:10], perm[10:20]
            trial = [np.linalg.norm(X_np[i, a].mean(0) - X_np[i, b].mean(0))
                     for i in range(n_items)]
            rand_l2.append(np.array(trial))
        rand_l2 = np.stack(rand_l2).mean(axis=0)
        t, p = stats.ttest_rel(inc_l2, rand_l2)
        return {"income_l2": float(inc_l2.mean()), "random_l2": float(rand_l2.mean()),
                "t": float(t), "p": float(p)}

    logger.info("  (2) Income L2 분석...")
    l2_orig = income_vs_random_l2(X)
    l2_res  = income_vs_random_l2(warm_residual)
    logger.info("  원본  income_l2=%.2f random=%.2f p=%.4f",
                l2_orig["income_l2"], l2_orig["random_l2"], l2_orig["p"])
    logger.info("  residual income_l2=%.2f random=%.2f p=%.4f",
                l2_res["income_l2"], l2_res["random_l2"], l2_res["p"])

    # (3) PCA: warm_mean (원본 vs residual)
    logger.info("  (3) PCA 분석...")
    warm_mean_orig = X.mean(dim=1).numpy()               # (300, 5120)
    warm_mean_res  = warm_residual.mean(dim=1).numpy()   # (300, 5120) (≈0 since mean of residual)

    # 페르소나 평균 residual (관점 전환: 각 페르소나의 아이템 간 평균)
    # 더 의미 있는 PCA: cold_residual을 각 아이템별로 평균 — 거의 0. 대신 개별 벡터 PCA
    # (300 × 50, 5120) 행렬 PCA
    def pca_variance(mat: np.ndarray, n_comp: int = 10) -> tuple[float, float]:
        """상위 n_comp PC의 누적 설명 분산 비율. (top-1, top-10)"""
        from numpy.linalg import svd
        mat_c = mat - mat.mean(axis=0, keepdims=True)
        n = min(mat_c.shape[0], n_comp + 5)
        _, s, _ = np.linalg.svd(mat_c, full_matrices=False)
        total = (s ** 2).sum()
        top1  = float((s[0] ** 2) / total)
        top10 = float((s[:10] ** 2).sum() / total)
        return top1, top10

    # warm_mean_orig PCA
    top1_orig_mean, top10_orig_mean = pca_variance(warm_mean_orig)
    # warm per-persona vectors PCA (300×50=15000 vectors)
    Xflat_orig = X.numpy().reshape(-1, HIDDEN_DIM)     # (15000, 5120)
    Xflat_res  = warm_residual.numpy().reshape(-1, HIDDEN_DIM)
    # 샘플 2000개로 근사
    rng2 = np.random.default_rng(0)
    idx_s = rng2.choice(len(Xflat_orig), size=2000, replace=False)
    top1_orig, top10_orig = pca_variance(Xflat_orig[idx_s])
    top1_res,  top10_res  = pca_variance(Xflat_res[idx_s])
    logger.info("  원본 PCA (per-persona): top-1=%.3f top-10=%.3f", top1_orig, top10_orig)
    logger.info("  residual PCA:          top-1=%.3f top-10=%.3f", top1_res, top10_res)

    # (4) Residual로 AttnBottleneck 학습 후 test MAE, entropy, P042
    logger.info("  (4) Residual AttnBottleneck 학습...")
    # 동일 stratified 테스트 셋 사용
    item_mean = y_warm.mean(axis=1)
    qlabels   = pd.qcut(pd.Series(item_mean), q=4, labels=["Q1","Q2","Q3","Q4"]).to_numpy()
    rng_t = np.random.default_rng(0)
    test_idx = []
    for q in ["Q1","Q2","Q3","Q4"]:
        qi = np.where(qlabels == q)[0]
        test_idx.extend(rng_t.choice(qi, size=min(13, len(qi)), replace=False).tolist())
    test_idx = sorted(test_idx[:50])
    pool_idx = [i for i in range(300) if i not in set(test_idx)]
    X_res_test  = warm_residual[test_idx]
    y_test      = y_warm[test_idx]
    meta_test   = warm_meta.iloc[test_idx].reset_index(drop=True)

    res_metrics = []
    orig_metrics = []
    for seed in [42, 123, 777]:
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(pool_idx, size=200, replace=False).tolist()
        # 원본
        m_orig = train_attn_model(X[tr_idx], torch.tensor(y_warm[tr_idx]), seed=seed)
        orig_metrics.append(compute_metrics(m_orig, X[test_idx], y_test, meta_test))
        # residual
        m_res = train_attn_model(warm_residual[tr_idx], torch.tensor(y_warm[tr_idx]), seed=seed)
        res_metrics.append(compute_metrics(m_res, X_res_test, y_test, meta_test))

    orig_mae = np.mean([m["mae"] for m in orig_metrics])
    res_mae  = np.mean([m["mae"] for m in res_metrics])
    orig_ent = np.mean([m["mean_entropy"] for m in orig_metrics])
    res_ent  = np.mean([m["mean_entropy"] for m in res_metrics])
    logger.info("  (4) 원본 MAE=%.2f entropy=%.4f  residual MAE=%.2f entropy=%.4f",
                orig_mae, orig_ent, res_mae, res_ent)

    return {
        "cos_orig_warm_mean": float(cos_orig_warm.mean()),
        "cos_res_warm_mean":  float(cos_res_warm.mean()),
        "cos_orig_cold_mean": float(cos_orig_cold.mean()),
        "cos_res_cold_mean":  float(cos_res_cold.mean()),
        "l2_orig":  l2_orig,
        "l2_res":   l2_res,
        "pca_orig_per_persona": {"top1": top1_orig, "top10": top10_orig},
        "pca_res_per_persona":  {"top1": top1_res,  "top10": top10_res},
        "attn_orig_mae":  orig_mae,
        "attn_res_mae":   res_mae,
        "attn_orig_ent":  orig_ent,
        "attn_res_ent":   res_ent,
        "orig_metrics": orig_metrics,
        "res_metrics":  res_metrics,
        "cos_orig_per_item":  cos_orig_warm,
        "cos_res_per_item":   cos_res_warm,
    }


# ─── 실험 C-2: 프롬프트 변형 파일럿 (GPU 필요) ───────────────────────────────

def experiment_c2(warm_raw: torch.Tensor, warm_meta: pd.DataFrame,
                  personas: list[dict]) -> dict:
    """
    V4-A (Persona-Heavy), V4-B (Contrastive) vs V3 (원본 warm_raw.pt 슬라이스).
    5개 아이템 × 50 페르소나 = 250 텍스트.
    GPU 필요.
    """
    import os
    logger.info("=== 실험 C-2: 프롬프트 변형 파일럿 (GPU) ===")

    # 5개 아이템 선택: FOODS×2, HOBBIES×2, HOUSEHOLD×1 (warm 중에서)
    item_mean = warm_raw.mean(dim=(1, 2)).numpy()
    pilot_idx = []
    for cat, n in [("FOODS", 2), ("HOBBIES", 2), ("HOUSEHOLD", 1)]:
        mask = (warm_meta["cat_id"] == cat).values
        cat_i = np.where(mask)[0]
        # 판매량 중위 근처 선택 (가장 대표적인 것)
        median_idx = cat_i[np.argsort(np.abs(item_mean[cat_i] - np.median(item_mean[cat_i])))[:n]]
        pilot_idx.extend(median_idx.tolist())
    pilot_idx = pilot_idx[:5]
    pilot_meta = warm_meta.iloc[pilot_idx].reset_index(drop=True)
    logger.info("  파일럿 5개 아이템: %s", pilot_meta[["item_id", "cat_id"]].to_dict("records"))

    # 아이템 메타 (dept_id, cat_id, avg_price 필요)
    # avg_price는 item_meta.csv에 없으므로 sell_prices에서 계산
    sell_prices = pd.read_csv(DATA_DIR / "cold_start/warm_train.csv",
                              usecols=["item_id", "sell_price"])
    price_map = sell_prices.groupby("item_id")["sell_price"].mean().to_dict()

    def get_price(item_id: str) -> float | None:
        return price_map.get(item_id, None)

    # V3 embedding: warm_raw.pt 슬라이스
    X_v3 = warm_raw[pilot_idx].numpy()  # (5, 50, 5120)
    logger.info("  V3 from warm_raw.pt: %s", X_v3.shape)

    # QwenEmbedder 로드
    hf_home = os.environ.get("HF_HOME", "/mnt/sdd1/jylee/huggingface_cache")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    logger.info("  Qwen 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, output_hidden_states=True,
        torch_dtype=torch.bfloat16, cache_dir=hf_home,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True, cache_dir=hf_home,
    )
    model.eval()
    logger.info("  Qwen 모델 로드 완료. hidden_size=%d", model.config.hidden_size)

    def extract_v3_style(texts: list[str], batch_size: int = 4,
                         max_length: int = 512) -> np.ndarray:
        """last-token, layer[-2] 추출 (V3 전략 동일)."""
        all_emb = []
        for bs in range(0, len(texts), batch_size):
            batch = texts[bs: bs + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt",
                            add_special_tokens=False)
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(input_ids.device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            sec_last = out.hidden_states[-2]  # (B, L, H)
            seq_lens = attn_mask.sum(dim=1) - 1
            for b_i, sl in enumerate(seq_lens):
                all_emb.append(sec_last[b_i, sl].cpu().float().numpy())
        return np.stack(all_emb)  # (N, H)

    def build_texts_for_variant(build_fn) -> np.ndarray:
        """250 텍스트 추출 → reshape (5, 50, 5120)."""
        texts = []
        for row in pilot_meta.itertuples():
            p_struct_list = [_build_persona_struct(p) for p in personas]
            for pi, p in enumerate(personas):
                p_struct = p_struct_list[pi]
                p_desc   = p.get("description", "")
                avg_price = get_price(row.item_id)
                texts.append(build_fn(p_struct, p_desc, row.dept_id, row.cat_id, avg_price))
        logger.info("  텍스트 %d개 추출 중...", len(texts))
        embs = extract_v3_style(texts)
        return embs.reshape(5, 50, HIDDEN_DIM)

    # V4-A
    logger.info("  V4-A 추출 중...")
    def build_v4a_wrapper(p_struct, p_desc, dept_id, cat_id, avg_price):
        return build_v4a_text(p_struct, p_desc, cat_id, avg_price)
    X_v4a = build_texts_for_variant(build_v4a_wrapper)

    # V4-B
    logger.info("  V4-B 추출 중...")
    def build_v4b_wrapper(p_struct, p_desc, dept_id, cat_id, avg_price):
        return build_v4b_text(p_struct, dept_id, cat_id, avg_price)
    X_v4b = build_texts_for_variant(build_v4b_wrapper)

    def analyze_embedding_quality(X: np.ndarray, name: str) -> dict:
        """(5, 50, 5120) → quality metrics."""
        cos_sims = []
        for i in range(5):
            v = X[i]; norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            vn = v / norms; sim = vn @ vn.T
            idx = np.triu_indices(50, k=1)
            cos_sims.append(sim[idx].mean())
        cos_mean = float(np.mean(cos_sims))
        # income L2
        budgets = np.array([p["weekly_budget"] for p in personas])
        si = np.argsort(budgets); low_i = si[:10]; high_i = si[-10:]
        l2_inc = [np.linalg.norm(X[i, high_i].mean(0) - X[i, low_i].mean(0)) for i in range(5)]
        # PCA
        Xf = X.reshape(-1, HIDDEN_DIM)
        _, s, _ = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False)
        total = (s**2).sum()
        top1  = float((s[0]**2) / total)
        top10 = float((s[:10]**2).sum() / total)
        logger.info("  [%s] cos=%.4f income_L2=%.2f PCA top-1=%.3f top-10=%.3f",
                    name, cos_mean, np.mean(l2_inc), top1, top10)
        return {"cos_mean": cos_mean, "income_l2_mean": float(np.mean(l2_inc)),
                "pca_top1": top1, "pca_top10": top10}

    q_v3  = analyze_embedding_quality(X_v3,  "V3 (원본)")
    q_v4a = analyze_embedding_quality(X_v4a, "V4-A")
    q_v4b = analyze_embedding_quality(X_v4b, "V4-B")

    # PCA curve (누적)
    def pca_cumulative(X_flat: np.ndarray, n_comp: int = 50) -> np.ndarray:
        _, s, _ = np.linalg.svd(X_flat - X_flat.mean(0), full_matrices=False)
        total = (s**2).sum()
        cum = np.cumsum(s**2) / total
        return cum[:n_comp]

    cum_v3  = pca_cumulative(X_v3.reshape(-1, HIDDEN_DIM))
    cum_v4a = pca_cumulative(X_v4a.reshape(-1, HIDDEN_DIM))
    cum_v4b = pca_cumulative(X_v4b.reshape(-1, HIDDEN_DIM))

    return {
        "v3":  q_v3,  "v4a": q_v4a, "v4b": q_v4b,
        "pca_curves": {"v3": cum_v3.tolist(), "v4a": cum_v4a.tolist(), "v4b": cum_v4b.tolist()},
        "pilot_items": pilot_meta[["item_id", "cat_id"]].to_dict("records"),
    }


# ─── 시각화 ───────────────────────────────────────────────────────────────────

def plot_all(results_a: dict, results_b: dict,
             rc1: dict, rc2: dict | None) -> None:
    TRAIN_SIZES = sorted(results_a.keys())

    def agg(d: dict, key: str) -> tuple[list, list]:
        means, stds = [], []
        for n in TRAIN_SIZES:
            v = [r[key] for r in d[n]]
            means.append(np.mean(v)); stds.append(np.std(v))
        return means, stds

    mae_m, mae_s = agg(results_a, "mae")
    ent_m, ent_s = agg(results_a, "mean_entropy")
    p42_m, p42_s = agg(results_a, "p042_weight")
    diff_r = [np.mean([r["cat_differentiated"] for r in results_a[n]]) for n in TRAIN_SIZES]

    # Figure 1: 실험 A
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    x = TRAIN_SIZES
    axes[0].errorbar(x, mae_m, yerr=mae_s, marker="o", color="steelblue", capsize=4)
    axes[0].set_title("(A) Test MAE vs Train Size"); axes[0].set_xlabel("N"); axes[0].grid(alpha=0.3)
    axes[1].errorbar(x, ent_m, yerr=ent_s, marker="s", color="orangered", capsize=4)
    axes[1].axhline(UNIFORM_ENT, color="green", ls="--", label=f"uniform={UNIFORM_ENT:.2f}")
    axes[1].set_title("(A) Attention Entropy vs Train Size"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].errorbar(x, p42_m, yerr=p42_s, marker="^", color="purple", capsize=4)
    axes[2].axhline(1/50, color="gray", ls="--", label="uniform=0.02")
    axes[2].set_title("(A) P042 Weight vs Train Size"); axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[3].plot(x, diff_r, marker="D", color="seagreen")
    axes[3].set_ylim(-0.05, 1.05); axes[3].set_title("(A) Category Differentiation Rate"); axes[3].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp_a_scaling_law.png", dpi=150); plt.close(fig)

    # Figure 2: 실험 B
    colors = ["steelblue", "orangered", "seagreen"]
    labels = ["B-1 matched", "B-2 cold-like", "B-3 reverse"]
    mae_b  = [np.mean([r["mae"] for r in results_b[v]]) for v in ["B1","B2","B3"]]
    mae_sb = [np.std([r["mae"]  for r in results_b[v]]) for v in ["B1","B2","B3"]]
    ent_b  = [np.mean([r["mean_entropy"] for r in results_b[v]]) for v in ["B1","B2","B3"]]
    p42_b  = [np.mean([r["p042_weight"]  for r in results_b[v]]) for v in ["B1","B2","B3"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(range(3), mae_b, yerr=mae_sb, color=colors, capsize=5, alpha=0.85, edgecolor="white")
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(labels); axes[0].set_title("(B) MAE by Variant")
    axes[1].bar(range(3), ent_b, color=colors, alpha=0.85, edgecolor="white")
    axes[1].axhline(UNIFORM_ENT, color="green", ls="--", label="uniform"); axes[1].set_title("(B) Entropy by Variant"); axes[1].legend()
    axes[2].bar(range(3), p42_b, color=colors, alpha=0.85, edgecolor="white")
    axes[2].axhline(1/50, color="gray", ls="--", label="uniform"); axes[2].set_title("(B) P042 Weight by Variant"); axes[2].legend()
    for ax in axes: ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp_b_distribution_mismatch.png", dpi=150); plt.close(fig)

    # Figure 3: C-1 코사인 유사도 분포
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rc1["cos_orig_per_item"], bins=25, alpha=0.6, color="steelblue",
                 label=f"원본 mean={rc1['cos_orig_warm_mean']:.4f}", edgecolor="white")
    axes[0].hist(rc1["cos_res_per_item"], bins=25, alpha=0.6, color="orangered",
                 label=f"residual mean={rc1['cos_res_warm_mean']:.4f}", edgecolor="white")
    axes[0].set_title("(C-1) 원본 vs Residual 페르소나 cosine similarity")
    axes[0].set_xlabel("Mean pairwise cosine sim"); axes[0].legend()
    axes[1].bar(["원본 MAE", "Residual MAE"], [rc1["attn_orig_mae"], rc1["attn_res_mae"]],
                color=["steelblue", "orangered"], alpha=0.85, edgecolor="white")
    axes[1].set_title(f"(C-1) AttnBottleneck MAE (N=200, 3 seeds)")
    axes[1].set_ylabel("MAE")
    for i, v in enumerate([rc1["attn_orig_mae"], rc1["attn_res_mae"]]):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp_c1_residual.png", dpi=150); plt.close(fig)

    # Figure 4: C-2 PCA curve (only if available)
    if rc2 is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, color in [("v3", "steelblue"), ("v4a", "orangered"), ("v4b", "seagreen")]:
            curve = rc2["pca_curves"][name]
            ax.plot(range(1, len(curve)+1), curve, marker=".", label=name.upper(), color=color)
        ax.set_xlabel("Number of PCs"); ax.set_ylabel("Cumulative explained variance")
        ax.set_title("(C-2) PCA Cumulative Variance: V3 vs V4-A vs V4-B")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG_DIR / "exp_c2_pca.png", dpi=150); plt.close(fig)


# ─── 보고서 ───────────────────────────────────────────────────────────────────

# 현재 V3 프롬프트 전문 (exp011 동일)
_V3_PROMPT_FULL = """\
### V3 프롬프트 전문 (exp011과 동일)

**Persona Text (Condition A — structured only):**
```
Customer profile:
- Weekly budget: $<weekly_budget>
- SNAP eligible: <snap_eligible>
- Economic status: <economic_status>
- Shopping motivation: <shopping_motivation>
- Category preference: FOODS=0.XX, HOBBIES=0.XX, HOUSEHOLD=0.XX
- Price sensitivity: <price_sensitivity>
- Visit frequency: <visit_frequency>
- Preferred departments: <preferred_departments>
- Decision style: <decision_style>
- Brand loyalty: <brand_loyalty>
- Promotion sensitivity: <promotion_sensitivity>
```

**Full Chat Template:**
```
<|im_start|>system
You are an expert consumer behavior analyst. Your task is to evaluate the match between a customer and a product.
<|im_end|>
<|im_start|>user
{persona_text}

Item Profile:
- Department: {dept_id}
- Category: {cat_id}
- Average price: {avg_price}

Task: Based on the customer profile, assess the probability of this customer purchasing this item. Answer with one of: Highly Likely, Likely, Neutral, Unlikely, Highly Unlikely.
<|im_end|>
<|im_start|>assistant
The answer is:
```

**Embedding 전략:** last-token, layer[-2] (second-to-last hidden state)
"""

_V4A_PROMPT_FULL = """\
### V4-A 변형 (Persona-Heavy)

**변경 원칙:** 아이템 정보 최소화 (카테고리 + 가격만) + 페르소나 서사(description) 추가

```
<|im_start|>system
You are an expert consumer behavior analyst. Your task is to evaluate how likely a specific customer is to purchase a product.
<|im_end|>
<|im_start|>user
Customer background: <description (자유 서술형 서사)>

{persona_struct}

Item Profile:
- Category: {cat_id}
- Average price: {avg_price}

Task: Based on this customer's background and profile, assess the probability of this customer purchasing this item. Answer with one of: Highly Likely, Likely, Neutral, Unlikely, Highly Unlikely.
<|im_end|>
<|im_start|>assistant
The answer is:
```
"""

_V4B_PROMPT_FULL = """\
### V4-B 변형 (Contrastive Prompt)

**변경 원칙:** V3와 동일한 구조 + 프롬프트 끝에 페르소나 고유성 강조 지시 추가

```
<|im_start|>system
You are an expert consumer behavior analyst. Your task is to evaluate the match between a customer and a product.
<|im_end|>
<|im_start|>user
{persona_text}

Item Profile:
- Department: {dept_id}
- Category: {cat_id}
- Average price: {avg_price}

Task: Based on the customer profile, assess the probability of this customer purchasing this item. Answer with one of: Highly Likely, Likely, Neutral, Unlikely, Highly Unlikely. To ensure your response reflects this customer's unique identity, consider how their specific economic background and preferences distinguish their purchasing patterns from those of someone with entirely different circumstances.
<|im_end|>
<|im_start|>assistant
The answer is:
```
"""


def write_report(results_a: dict, results_b: dict,
                 rc1: dict, rc2: dict | None) -> None:
    TRAIN_SIZES = sorted(results_a.keys())
    lines = [
        "# Scaling Law + 분포 불일치 + Embedding 방향 분리 검증 보고서 (Exp013)",
        "",
        "**작성일:** 2026-03-10",
        "**배경:** exp012에서 P042 지배 → 데이터 부족 vs embedding 방향 문제 구분",
        "",
        "---", "",
        "## 실험 A: Scaling Law", "",
        "**설계:** 고정 테스트 50개 (stratified) + 학습 풀 250개 / 학습 크기 [30,50,75,100,150,200,250] × 3 seeds", "",
        "| Train N | MAE | Entropy | Entropy/Uniform | P042 Weight | Cat. Diff. Rate |",
        "|---------|-----|---------|-----------------|-------------|-----------------|",
    ]
    for n in TRAIN_SIZES:
        mae_m = np.mean([r["mae"]          for r in results_a[n]])
        mae_s = np.std( [r["mae"]          for r in results_a[n]])
        ent_m = np.mean([r["mean_entropy"] for r in results_a[n]])
        ent_s = np.std( [r["mean_entropy"] for r in results_a[n]])
        p42_m = np.mean([r["p042_weight"]  for r in results_a[n]])
        p42_s = np.std( [r["p042_weight"]  for r in results_a[n]])
        diff  = np.mean([r["cat_differentiated"] for r in results_a[n]])
        lines.append(f"| {n:4d} | {mae_m:.1f}±{mae_s:.1f} | {ent_m:.4f}±{ent_s:.4f} | "
                     f"{ent_m/UNIFORM_ENT:.3f} | {p42_m:.4f}±{p42_s:.4f} | {diff:.2f} |")

    p42_250 = np.mean([r["p042_weight"] for r in results_a[250]])
    p42_30  = np.mean([r["p042_weight"] for r in results_a[30]])
    diff_250 = np.mean([r["cat_differentiated"] for r in results_a[250]])
    if p42_250 < p42_30 * 0.8 and diff_250 > 0.3:
        verdict_a = "데이터 부족이 주 원인 → warm 확장(경로 B) 정당화"
    elif p42_250 < p42_30 * 0.9:
        verdict_a = "데이터 부족이 부분 원인 → warm 확장 + embedding 개선 병행"
    else:
        verdict_a = "embedding 자체 문제 → 프롬프트/레이어 변경 필요"
    lines += ["", f"→ **판정: {verdict_a}**", "", "---", ""]

    # 실험 B
    lines += ["## 실험 B: 분포 불일치", "",
              "| 변형 | 학습 | 테스트 | MAE | Entropy | P042 Weight |",
              "|------|------|--------|-----|---------|-------------|"]
    vdesc = {"B1": ("Q2+Q3+Q4 200개", "Q2+Q3+Q4 50개 [matched]"),
             "B2": ("Q2+Q3+Q4 200개", "Q1 전체 [cold-like]"),
             "B3": ("Q1+Q2+Q3 200개", "Q4 전체 [reverse]")}
    b_mae = {}
    for v in ["B1","B2","B3"]:
        mae_m = np.mean([r["mae"]          for r in results_b[v]])
        mae_s = np.std( [r["mae"]          for r in results_b[v]])
        ent_m = np.mean([r["mean_entropy"] for r in results_b[v]])
        p42_m = np.mean([r["p042_weight"]  for r in results_b[v]])
        b_mae[v] = mae_m
        lines.append(f"| {v} | {vdesc[v][0]} | {vdesc[v][1]} | "
                     f"{mae_m:.1f}±{mae_s:.1f} | {ent_m:.4f} | {p42_m:.4f} |")
    ratio = b_mae["B2"] / (b_mae["B1"] + 1e-9)
    verdict_b = ("분포 불일치가 주 원인" if ratio > 1.5 else
                 "분포 불일치가 중간 영향" if ratio > 1.2 else
                 "분포 불일치는 주 원인 아님 → 다른 원인 탐색")
    lines += ["", f"- B-2/B-1 MAE 비율: **{ratio:.2f}x**", f"→ **판정: {verdict_b}**", "", "---", ""]

    # 실험 C-1
    lines += ["## 실험 C-1: Item-Mean Subtraction", "",
              "| 지표 | 원본 | Residual |",
              "|------|------|----------|",
              f"| 페르소나 pairwise cosine sim (warm 평균) | **{rc1['cos_orig_warm_mean']:.4f}** | **{rc1['cos_res_warm_mean']:.4f}** |",
              f"| 페르소나 pairwise cosine sim (cold 평균) | {rc1['cos_orig_cold_mean']:.4f} | {rc1['cos_res_cold_mean']:.4f} |",
              f"| income L2 distance | {rc1['l2_orig']['income_l2']:.2f} (p={rc1['l2_orig']['p']:.4f}) | {rc1['l2_res']['income_l2']:.2f} (p={rc1['l2_res']['p']:.4f}) |",
              f"| random L2 distance | {rc1['l2_orig']['random_l2']:.2f} | {rc1['l2_res']['random_l2']:.2f} |",
              f"| PCA top-1 PC (per-persona 벡터) | {rc1['pca_orig_per_persona']['top1']:.3f} | {rc1['pca_res_per_persona']['top1']:.3f} |",
              f"| PCA top-10 PC | {rc1['pca_orig_per_persona']['top10']:.3f} | {rc1['pca_res_per_persona']['top10']:.3f} |",
              f"| AttnBottleneck MAE (N=200, 3 seeds avg) | {rc1['attn_orig_mae']:.2f} | {rc1['attn_res_mae']:.2f} |",
              f"| Attention Entropy | {rc1['attn_orig_ent']:.4f} | {rc1['attn_res_ent']:.4f} |",
              ""]
    cos_drop = rc1["cos_orig_warm_mean"] - rc1["cos_res_warm_mean"]
    if cos_drop > 0.05 and rc1["attn_res_mae"] <= rc1["attn_orig_mae"] * 1.1:
        verdict_c1 = "후처리만으로 방향 분리 가능 → 가장 효율적인 경로"
    elif cos_drop > 0.05 and rc1["attn_res_mae"] > rc1["attn_orig_mae"] * 1.1:
        verdict_c1 = "방향 분리됐지만 MAE 악화 → 아이템 평균에 유의미한 페르소나 정보 혼재"
    else:
        verdict_c1 = "residual에서 방향 분리 미흡 → 후처리 부적합, 프롬프트 변경 필요"
    lines += [f"→ cosine sim 감소: {cos_drop:.4f}", f"→ **판정: {verdict_c1}**", "", "---", ""]

    # 실험 C-2
    lines += ["## 실험 C-2: 프롬프트 변형 파일럿 (GPU)", ""]
    if rc2 is not None:
        lines += ["| 지표 | V3 (원본) | V4-A (Persona-Heavy) | V4-B (Contrastive) |",
                  "|------|----------|---------------------|-------------------|",
                  f"| pairwise cosine sim | {rc2['v3']['cos_mean']:.4f} | {rc2['v4a']['cos_mean']:.4f} | {rc2['v4b']['cos_mean']:.4f} |",
                  f"| income L2 mean | {rc2['v3']['income_l2_mean']:.2f} | {rc2['v4a']['income_l2_mean']:.2f} | {rc2['v4b']['income_l2_mean']:.2f} |",
                  f"| PCA top-1 | {rc2['v3']['pca_top1']:.3f} | {rc2['v4a']['pca_top1']:.3f} | {rc2['v4b']['pca_top1']:.3f} |",
                  f"| PCA top-10 | {rc2['v3']['pca_top10']:.3f} | {rc2['v4a']['pca_top10']:.3f} | {rc2['v4b']['pca_top10']:.3f} |",
                  ""]
        best_v4 = "V4-A" if rc2["v4a"]["cos_mean"] < rc2["v4b"]["cos_mean"] else "V4-B"
        best_cos = min(rc2["v4a"]["cos_mean"], rc2["v4b"]["cos_mean"])
        if best_cos < 0.95:
            verdict_c2 = f"{best_v4}가 cosine sim을 0.95 이하로 낮춤 → 전체 재추출 가치 있음"
        else:
            verdict_c2 = "V4-A/B 모두 효과 미미 → 모델/레이어 변경 검토 필요"
        lines += [f"→ **판정: {verdict_c2}**", ""]
    else:
        lines += ["C-2는 GPU가 필요합니다. `--run-c2` 플래그로 실행하세요.", "",
                  "```bash", "CUDA_VISIBLE_DEVICES=0,1,2,3 \\",
                  f"HF_HOME=/mnt/sdd1/jylee/huggingface_cache \\",
                  "conda run --no-capture-output -n persona-forecasting \\",
                  "  python scripts/exp013_scaling_and_direction.py --run-c2",
                  "```", ""]
        verdict_c2 = "미실행"

    # 프롬프트 전문
    lines += ["---", "", "## 프롬프트 전문", "",
              _V3_PROMPT_FULL, "", _V4A_PROMPT_FULL, "", _V4B_PROMPT_FULL, "", "---", "",
              "## 종합 결론", "",
              "| 실험 | 질문 | 판정 |",
              "|------|------|------|",
              f"| A (Scaling Law) | P042 지배가 데이터 부족 때문인가? | {verdict_a} |",
              f"| B (Distribution) | 분포 불일치가 주 성능 저하 원인인가? | {verdict_b} |",
              f"| C-1 (Residual) | Item-mean subtraction으로 방향 분리 가능한가? | {verdict_c1} |",
              f"| C-2 (Prompt) | V4 프롬프트가 cosine sim을 낮추는가? | {verdict_c2} |",
              "",
              "---",
              "",
              "**시각화:** `experiments/exp013_scaling_and_direction/figures/`",
              "**스크립트:** `scripts/exp013_scaling_and_direction.py`",
    ]
    path = REPORT_DIR / "scaling_and_direction_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", path)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-c2", action="store_true", help="C-2 GPU 실험 실행")
    args = parser.parse_args()

    logger.info("=== Exp013: Scaling + Direction 검증 시작 ===")

    warm_raw, y_warm, warm_meta = load_data()
    cold_raw = torch.load(EMB_DIR / "cold_raw.pt", weights_only=True)
    personas = load_personas(n=50)

    # C-1 우선 실행 (우선순위 1)
    rc1 = experiment_c1(warm_raw, cold_raw, y_warm, warm_meta, personas)

    # A (우선순위 2)
    res_a_full = experiment_a(warm_raw, y_warm, warm_meta)
    results_a  = res_a_full["results"]

    # B (우선순위 3)
    results_b = experiment_b(warm_raw, y_warm, warm_meta)

    # C-2 (우선순위 4, optional)
    rc2 = None
    if args.run_c2:
        rc2 = experiment_c2(warm_raw, warm_meta, personas)

    plot_all(results_a, results_b, rc1, rc2)
    write_report(results_a, results_b, rc1, rc2)

    print("\n" + "=" * 65)
    print("=== Exp013 결과 요약 ===")
    print(f"\nC-1: cos 원본={rc1['cos_orig_warm_mean']:.4f} → residual={rc1['cos_res_warm_mean']:.4f}")
    print(f"     MAE 원본={rc1['attn_orig_mae']:.2f} → residual={rc1['attn_res_mae']:.2f}")
    TRAIN_SIZES = sorted(results_a.keys())
    print("\nA (Scaling Law):")
    print(f"  {'N':>4}  {'MAE':>7}  {'Entropy':>8}  {'P042':>7}  {'CatDiff':>8}")
    for n in TRAIN_SIZES:
        mae_m = np.mean([r["mae"]           for r in results_a[n]])
        ent_m = np.mean([r["mean_entropy"]  for r in results_a[n]])
        p42_m = np.mean([r["p042_weight"]   for r in results_a[n]])
        diff  = np.mean([r["cat_differentiated"] for r in results_a[n]])
        print(f"  {n:>4}  {mae_m:>7.2f}  {ent_m:>8.4f}  {p42_m:>7.4f}  {diff:>8.2f}")
    print("\nB (분포 불일치):")
    for v in ["B1","B2","B3"]:
        mae_m = np.mean([r["mae"] for r in results_b[v]])
        print(f"  {v}: MAE={mae_m:.2f}")
    if rc2:
        print(f"\nC-2: V3 cos={rc2['v3']['cos_mean']:.4f}  V4-A={rc2['v4a']['cos_mean']:.4f}  V4-B={rc2['v4b']['cos_mean']:.4f}")
    print("=" * 65)
    print("보고서: docs/diagnosis/scaling_and_direction_report.md")


if __name__ == "__main__":
    main()
