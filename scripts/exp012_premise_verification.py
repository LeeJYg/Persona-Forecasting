"""Exp012: 방법론 핵심 전제 3가지 독립 검증.

전제 1: Persona Sensitivity  — hidden state가 페르소나 속성에 따라 달라지는가?
전제 2: Attention as Market Modeling — Attention이 아이템-관련 페르소나를 선별하는가?
전제 3: Quantitative Transfer — 표현에서 판매량 수치를 예측할 수 있는가?

사용 데이터: exp011에서 저장된 hidden states + 학습된 모델 (새로 추출하지 않음)
주의: 기존 코드(exp011 등)를 수정하지 않음. 분석 스크립트 신규 작성만.

출력:
    experiments/exp012_premise_verification/figures/
    docs/diagnosis/premise_verification_report.md
"""
from __future__ import annotations

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
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

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
MODEL_DIR  = ROOT / "experiments/exp011_v3_pipeline/models"
FIG_DIR    = ROOT / "experiments/exp012_premise_verification/figures"
REPORT_DIR = ROOT / "docs/diagnosis"
DATA_DIR   = ROOT / "data/processed"
PERSONA_DIR = DATA_DIR / "personas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_embeddings() -> dict[str, torch.Tensor]:
    logger.info("임베딩 로드 중...")
    return {
        "warm_raw":  torch.load(EMB_DIR / "warm_raw.pt",  weights_only=True),
        "cold_raw":  torch.load(EMB_DIR / "cold_raw.pt",  weights_only=True),
        "warm_mean": torch.load(EMB_DIR / "warm_mean.pt", weights_only=True),
        "cold_mean": torch.load(EMB_DIR / "cold_mean.pt", weights_only=True),
    }


def load_item_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    """item_meta.csv → cold_meta, warm_meta (순서는 embedding 행 순서와 동일)."""
    meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_meta = meta[meta["is_cold"] == True].reset_index(drop=True)
    warm_meta = meta[meta["is_cold"] == False].reset_index(drop=True)
    return cold_meta, warm_meta


def load_personas(n: int = 50) -> list[dict]:
    """페르소나 JSON 로드 (알파벳 순 첫 n개)."""
    files = sorted(PERSONA_DIR.glob("CA_1_P*.json"))[:n]
    out = []
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        out.append(d["profile"])
    logger.info("페르소나 로드: %d개", len(out))
    return out


def agg_weekly(sales_df: pd.DataFrame, item_ids: list[str],
               date_start: str, date_end: str) -> np.ndarray:
    """exp011과 동일한 agg_weekly — pd.to_period('W') 기준."""
    sub = sales_df[
        sales_df["item_id"].isin(item_ids) &
        (sales_df["date"] >= date_start) &
        (sales_df["date"] <= date_end)
    ].copy()
    sub["week"] = pd.to_datetime(sub["date"]).dt.to_period("W")
    w = (sub.groupby(["item_id", "week"])["sales"]
           .sum()
           .unstack("week")
           .reindex(item_ids)
           .fillna(0))
    return w.values.astype(np.float32)


def load_sales_targets(warm_meta: pd.DataFrame) -> np.ndarray:
    """warm items의 실제 주간 판매량 y_warm (300, 17)."""
    cold_test  = pd.read_csv(DATA_DIR / "cold_start/cold_test.csv",  parse_dates=["date"])
    warm_test  = pd.read_csv(DATA_DIR / "cold_start/warm_test.csv",  parse_dates=["date"])
    date_start = str(cold_test["date"].min().date())
    date_end   = str(cold_test["date"].max().date())
    warm_ids   = warm_meta["item_id"].tolist()
    y_warm = agg_weekly(warm_test, warm_ids, date_start, date_end)
    logger.info("y_warm shape: %s  mean=%.2f", y_warm.shape, y_warm.mean())
    return y_warm


# ─── 검증 1: Persona Sensitivity ─────────────────────────────────────────────

def verify_persona_sensitivity(
    cold_raw: torch.Tensor,         # (100, 50, 5120)
    cold_meta: pd.DataFrame,
    personas: list[dict],
) -> dict:
    """
    전제 1: 동일 아이템에 대해 페르소나 속성(income)이 다르면 hidden state가 달라지는가?

    (a) 아이템별 페르소나 간 pairwise cosine similarity 분포
    (b) income 상위/하위 10명 그룹 간 L2 distance vs 랜덤 분리
    (c) 통계적 유의성 판정
    """
    logger.info("=== 검증 1: Persona Sensitivity ===")

    X = cold_raw.numpy()                     # (100, 50, 5120)
    n_items, n_personas, hidden = X.shape

    # (a) 아이템별 페르소나 간 cosine similarity ─────────────────────────────
    # 10개 아이템 샘플: 고판매 5개 + 저판매 5개 (차후 추가; 지금은 무작위 10개)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_items, size=min(10, n_items), replace=False)

    item_het_scores = []
    for i in range(n_items):
        vecs = X[i]                          # (50, 5120)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs_n = vecs / norms
        sim = vecs_n @ vecs_n.T              # (50, 50)
        # 상삼각 (대각 제외)
        idx = np.triu_indices(n_personas, k=1)
        pairwise_cos = sim[idx]
        item_het_scores.append(pairwise_cos.mean())

    item_het_scores = np.array(item_het_scores)
    logger.info("  (a) 아이템별 페르소나 pairwise cosine sim: mean=%.4f std=%.4f",
                item_het_scores.mean(), item_het_scores.std())

    # (b) income 분리: weekly_budget 기준 상위 10명 / 하위 10명 ───────────────
    budgets = np.array([p["weekly_budget"] for p in personas])
    sorted_idx = np.argsort(budgets)
    low_idx  = sorted_idx[:10]
    high_idx = sorted_idx[-10:]

    # 아이템별 L2 distance (income 분리)
    income_l2 = []
    for i in range(n_items):
        low_mean  = X[i, low_idx].mean(axis=0)
        high_mean = X[i, high_idx].mean(axis=0)
        income_l2.append(np.linalg.norm(high_mean - low_mean))
    income_l2 = np.array(income_l2)

    # 랜덤 분리 100회 반복
    random_l2_all = []
    for _ in range(100):
        perm = rng.permutation(n_personas)
        a_idx = perm[:10]; b_idx = perm[10:20]
        trial = []
        for i in range(n_items):
            a_mean = X[i, a_idx].mean(axis=0)
            b_mean = X[i, b_idx].mean(axis=0)
            trial.append(np.linalg.norm(a_mean - b_mean))
        random_l2_all.append(np.array(trial))
    random_l2_all = np.stack(random_l2_all)          # (100, n_items)
    random_l2_mean = random_l2_all.mean(axis=0)       # (n_items,)

    # 통계적 유의성: income_l2 > random 분포?
    t_stat, p_val = stats.ttest_rel(income_l2, random_l2_mean)
    effect_size = (income_l2.mean() - random_l2_mean.mean()) / random_l2_mean.std()

    logger.info("  (b) income_l2: mean=%.2f  random_l2_mean: mean=%.2f",
                income_l2.mean(), random_l2_mean.mean())
    logger.info("      t=%.3f  p=%.4f  effect_size=%.3f", t_stat, p_val, effect_size)

    # 판정
    if p_val < 0.05 and income_l2.mean() > random_l2_mean.mean():
        verdict = "지지"
    elif p_val < 0.05:
        verdict = "부분적 지지"
    else:
        verdict = "실패"

    # 시각화: income_l2 vs random_l2_mean histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(item_het_scores, bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(item_het_scores.mean(), color="red", linestyle="--",
                    label=f"mean={item_het_scores.mean():.4f}")
    axes[0].set_title("(a) 아이템별 페르소나 pairwise cosine similarity")
    axes[0].set_xlabel("Mean pairwise cosine sim"); axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].hist(random_l2_mean, bins=20, alpha=0.6, color="gray", label="random split", edgecolor="white")
    axes[1].hist(income_l2, bins=20, alpha=0.7, color="orangered", label="income split", edgecolor="white")
    axes[1].axvline(random_l2_mean.mean(), color="gray", linestyle="--")
    axes[1].axvline(income_l2.mean(), color="orangered", linestyle="--")
    axes[1].set_title(f"(b) income vs random split L2  (p={p_val:.4f})")
    axes[1].set_xlabel("L2 distance (high_income_mean - low_income_mean)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "premise1_persona_sensitivity.png", dpi=150)
    plt.close(fig)

    return {
        "pairwise_cos_mean": float(item_het_scores.mean()),
        "pairwise_cos_std":  float(item_het_scores.std()),
        "income_l2_mean":    float(income_l2.mean()),
        "random_l2_mean":    float(random_l2_mean.mean()),
        "t_stat":            float(t_stat),
        "p_val":             float(p_val),
        "effect_size":       float(effect_size),
        "verdict":           verdict,
    }


# ─── 검증 2: Attention as Market Modeling ────────────────────────────────────

def verify_attention_market(
    cold_raw:  torch.Tensor,         # (100, 50, 5120)
    warm_raw:  torch.Tensor,         # (300, 50, 5120)
    cold_meta: pd.DataFrame,
    warm_meta: pd.DataFrame,
    personas:  list[dict],
) -> dict:
    """
    전제 2: Attention+Bottleneck head가 아이템-관련 페르소나에 높은 가중치를 부여하는가?

    (a) attention weight 추출 (cold + warm items)
    (b) 카테고리별 상위 페르소나 분석
    (c) attention entropy 분석
    """
    logger.info("=== 검증 2: Attention as Market Modeling ===")

    state = torch.load(MODEL_DIR / "attn_bottleneck/model.pt",
                       weights_only=True, map_location="cpu")
    attn_w = state["attn.weight"]           # (1, 5120)

    def get_attn_weights(X_raw: torch.Tensor) -> np.ndarray:
        """X_raw: (N, P, H) → attention weights (N, P) after softmax."""
        scores = (X_raw @ attn_w.T).squeeze(-1)   # (N, P)
        weights = torch.softmax(scores, dim=-1)    # (N, P)
        return weights.numpy()

    # (a) attention weight 추출 ────────────────────────────────────────────────
    cold_attn = get_attn_weights(cold_raw)    # (100, 50)
    warm_attn = get_attn_weights(warm_raw)    # (300, 50)
    all_attn  = np.vstack([cold_attn, warm_attn])  # (400, 50)
    all_meta  = pd.concat([cold_meta, warm_meta], ignore_index=True)

    # (b) 카테고리별 상위 페르소나 ────────────────────────────────────────────
    categories = all_meta["cat_id"].unique()
    cat_top5 = {}
    for cat in sorted(categories):
        mask = (all_meta["cat_id"] == cat).values
        cat_weights = all_attn[mask].mean(axis=0)  # (50,)
        top5_idx = np.argsort(cat_weights)[::-1][:5]
        top5_info = []
        for pi in top5_idx:
            p = personas[pi]
            top5_info.append({
                "persona_idx":      int(pi),
                "weight":           float(cat_weights[pi]),
                "weekly_budget":    p["weekly_budget"],
                "economic_status":  p["economic_status"],
                "category_pref":    p["category_preference"],
                "decision_style":   p["decision_style"],
            })
        cat_top5[cat] = top5_info
        logger.info("  [%s] top5 personas: %s", cat,
                    [(t["persona_idx"], round(t["weight"], 4)) for t in top5_info])

    # (c) Attention entropy 분석 ──────────────────────────────────────────────
    UNIFORM_ENTROPY = float(np.log(50))   # ≈ 3.912

    def entropy(w: np.ndarray) -> float:
        w = np.clip(w, 1e-12, None)
        return float(-np.sum(w * np.log(w)))

    item_entropy = np.array([entropy(all_attn[i]) for i in range(len(all_attn))])
    mean_ent = float(item_entropy.mean())
    std_ent  = float(item_entropy.std())

    logger.info("  (c) Attention entropy: mean=%.4f  std=%.4f  uniform=%.4f",
                mean_ent, std_ent, UNIFORM_ENTROPY)

    # 판정
    ent_ratio = mean_ent / UNIFORM_ENTROPY
    if ent_ratio < 0.8:
        ent_verdict = "집중적 (sparse)"
    elif ent_ratio < 0.95:
        ent_verdict = "약간 집중"
    else:
        ent_verdict = "uniform에 가까움"

    # 카테고리별 상위 페르소나가 도메인 지식과 일치하는가? (수동 확인용 출력)
    logger.info("  카테고리별 상위 페르소나 속성 (도메인 일치 여부는 수동 확인):")
    for cat, top5 in cat_top5.items():
        logger.info("    [%s]:", cat)
        for t in top5:
            logger.info("      P%03d weight=%.4f budget=$%.0f status=%s foods_pref=%.1f",
                        t["persona_idx"] + 1, t["weight"], t["weekly_budget"],
                        t["economic_status"], t["category_pref"].get("FOODS", 0))

    # 판정: 엔트로피 기준
    if ent_ratio < 0.85:
        verdict = "지지"
    elif ent_ratio < 0.95:
        verdict = "부분적 지지"
    else:
        verdict = "실패"

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) Attention entropy histogram
    axes[0].hist(item_entropy, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(mean_ent, color="red", linestyle="--",
                    label=f"mean={mean_ent:.3f}")
    axes[0].axvline(UNIFORM_ENTROPY, color="green", linestyle="--",
                    label=f"uniform={UNIFORM_ENTROPY:.3f}")
    axes[0].set_title("Attention Entropy per Item")
    axes[0].set_xlabel("Entropy"); axes[0].legend()

    # (2) 카테고리별 평균 attention weight per persona (bar)
    cat_colors = {"FOODS": "tomato", "HOUSEHOLD": "steelblue", "HOBBIES": "seagreen"}
    for cat in sorted(categories):
        mask = (all_meta["cat_id"] == cat).values
        mean_w = all_attn[mask].mean(axis=0)
        axes[1].plot(range(50), mean_w, label=cat, alpha=0.8,
                     color=cat_colors.get(cat, "gray"))
    axes[1].set_title("카테고리별 평균 Attention Weight (per persona)")
    axes[1].set_xlabel("Persona index"); axes[1].set_ylabel("Mean attention weight")
    axes[1].legend(); axes[1].axhline(1/50, color="black", linestyle="--", alpha=0.5, label="uniform")

    # (3) warm vs cold entropy distribution
    cold_ent = np.array([entropy(cold_attn[i]) for i in range(len(cold_attn))])
    warm_ent = np.array([entropy(warm_attn[i]) for i in range(len(warm_attn))])
    axes[2].hist(cold_ent, bins=20, alpha=0.6, color="orangered", label=f"cold (n=100)", edgecolor="white")
    axes[2].hist(warm_ent, bins=20, alpha=0.6, color="steelblue", label=f"warm (n=300)", edgecolor="white")
    axes[2].axvline(UNIFORM_ENTROPY, color="green", linestyle="--", label=f"uniform")
    axes[2].set_title("Cold vs Warm Attention Entropy")
    axes[2].set_xlabel("Entropy"); axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "premise2_attention_market.png", dpi=150)
    plt.close(fig)

    return {
        "mean_entropy":    mean_ent,
        "std_entropy":     std_ent,
        "uniform_entropy": UNIFORM_ENTROPY,
        "entropy_ratio":   ent_ratio,
        "entropy_verdict": ent_verdict,
        "cat_top5":        cat_top5,
        "verdict":         verdict,
    }


# ─── 검증 3: Quantitative Transfer ───────────────────────────────────────────

def verify_quantitative_transfer(
    warm_mean: torch.Tensor,         # (300, 5120)
    y_warm:    np.ndarray,            # (300, 17)
    warm_meta: pd.DataFrame,
) -> dict:
    """
    전제 3: warm 내부에서 embedding이 아이템별 판매량 차이를 예측할 수 있는가?

    (a) warm 5-fold CV로 아이템별 예측값 vs 실제값 수집
    (b) 판매량 분위별 MAE
    (c) Pearson / Spearman 상관계수
    """
    logger.info("=== 검증 3: Quantitative Transfer ===")

    X = warm_mean.numpy()      # (300, 5120)
    y = y_warm                 # (300, 17)
    n_items, n_weeks = y.shape

    # 5-fold CV: 아이템별 예측값 수집 ─────────────────────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    item_pred_mean = np.zeros(n_items)
    item_true_mean = y.mean(axis=1)           # (300,) 아이템별 평균 실제 판매량

    # per-item CV prediction (average over weeks)
    item_preds_all = np.zeros((n_items, n_weeks))
    for tr_idx, val_idx in kf.split(X):
        pred_val = np.zeros((len(val_idx), n_weeks))
        for w in range(n_weeks):
            mdl = Ridge(alpha=1.0)
            mdl.fit(X[tr_idx], y[tr_idx, w])
            pred_val[:, w] = mdl.predict(X[val_idx])
        item_preds_all[val_idx] = pred_val

    item_pred_mean = item_preds_all.mean(axis=1)   # (300,) 아이템별 평균 예측

    # (a) 전체 MAE (warm CV)
    cv_mae = float(np.abs(y - item_preds_all).mean())
    logger.info("  (a) warm 5-fold CV MAE = %.4f", cv_mae)

    # (b) 판매량 분위별 MAE ────────────────────────────────────────────────────
    quartile_mae = {}
    quartiles = pd.qcut(pd.Series(item_true_mean), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        mask = (quartiles == q).to_numpy()
        q_mae = float(np.abs(y[mask] - item_preds_all[mask]).mean())
        q_true_mean = float(item_true_mean[mask].mean())
        quartile_mae[q] = {"mae": q_mae, "true_mean": q_true_mean, "n": int(mask.sum())}
        logger.info("  (b) %s: true_mean=%.2f  MAE=%.2f  n=%d", q, q_true_mean, q_mae, mask.sum())

    # (c) Pearson / Spearman 상관계수 ─────────────────────────────────────────
    pearson_r, pearson_p = stats.pearsonr(item_true_mean, item_pred_mean)
    spearman_r, spearman_p = stats.spearmanr(item_true_mean, item_pred_mean)
    logger.info("  (c) Pearson r=%.4f (p=%.4f)  Spearman r=%.4f (p=%.4f)",
                pearson_r, pearson_p, spearman_r, spearman_p)

    # 판정
    if spearman_r > 0.5 and pearson_r > 0.3:
        verdict = "부분적 지지"
    elif spearman_r > 0.3:
        verdict = "부분적 지지 (약)"
    else:
        verdict = "실패"

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) Scatter: 실제 vs 예측 (아이템별 평균)
    axes[0].scatter(item_true_mean, item_pred_mean, alpha=0.5, s=20, color="steelblue")
    lim_max = max(item_true_mean.max(), item_pred_mean.max()) * 1.05
    axes[0].plot([0, lim_max], [0, lim_max], "r--", lw=1, label="y=x (perfect)")
    axes[0].axhline(item_pred_mean.mean(), color="orange", linestyle="--", alpha=0.7,
                    label=f"pred mean={item_pred_mean.mean():.1f}")
    axes[0].set_xlabel("실제 주간 판매량 (아이템별 평균)")
    axes[0].set_ylabel("예측 주간 판매량 (아이템별 평균)")
    axes[0].set_title(f"(c) Scatter  Pearson={pearson_r:.3f}  Spearman={spearman_r:.3f}")
    axes[0].legend()

    # (2) 분위별 MAE bar chart
    qs = list(quartile_mae.keys())
    mae_vals = [quartile_mae[q]["mae"] for q in qs]
    true_means = [quartile_mae[q]["true_mean"] for q in qs]
    x = np.arange(len(qs))
    ax2b = axes[1].twinx()
    axes[1].bar(x - 0.2, mae_vals, 0.4, label="MAE", color="tomato", alpha=0.8)
    ax2b.bar(x + 0.2, true_means, 0.4, label="True mean", color="steelblue", alpha=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(qs)
    axes[1].set_ylabel("MAE", color="tomato")
    ax2b.set_ylabel("True mean weekly sales", color="steelblue")
    axes[1].set_title("(b) 분위별 MAE")
    axes[1].legend(loc="upper left"); ax2b.legend(loc="upper right")

    # (3) 예측값 분포 vs 실제값 분포
    axes[2].hist(item_true_mean, bins=30, alpha=0.6, color="steelblue",
                 label=f"실제 (mean={item_true_mean.mean():.1f})", edgecolor="white")
    axes[2].hist(item_pred_mean, bins=30, alpha=0.6, color="orangered",
                 label=f"예측 (mean={item_pred_mean.mean():.1f})", edgecolor="white")
    axes[2].set_title("(a) 아이템별 평균 판매량 분포")
    axes[2].set_xlabel("주간 판매량"); axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "premise3_quantitative_transfer.png", dpi=150)
    plt.close(fig)

    return {
        "cv_mae":             cv_mae,
        "quartile_mae":       quartile_mae,
        "pearson_r":          float(pearson_r),
        "pearson_p":          float(pearson_p),
        "spearman_r":         float(spearman_r),
        "spearman_p":         float(spearman_p),
        "item_true_mean_mean": float(item_true_mean.mean()),
        "item_pred_mean_mean": float(item_pred_mean.mean()),
        "verdict":             verdict,
    }


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(r1: dict, r2: dict, r3: dict) -> None:
    lines = []
    lines += [
        "# 방법론 핵심 전제 독립 검증 보고서 (Exp012)",
        "",
        f"**작성일:** 2026-03-10",
        f"**사용 데이터:** exp011 (V3 임베딩, warm_raw/cold_raw.pt + attn_bottleneck 모델)",
        "",
        "---",
        "",
        "## 데이터 가용성 확인",
        "",
        "| 필요 데이터 | 경로 | 존재 여부 |",
        "|------------|------|----------|",
        f"| warm per-persona embeddings | experiments/exp011_v3_pipeline/embeddings/warm_raw.pt | ✓ (300, 50, 5120) |",
        f"| cold per-persona embeddings | experiments/exp011_v3_pipeline/embeddings/cold_raw.pt | ✓ (100, 50, 5120) |",
        f"| Attention+Bottleneck 모델 | experiments/exp011_v3_pipeline/models/attn_bottleneck/model.pt | ✓ |",
        f"| 페르소나 JSON | data/processed/personas/CA_1_P*.json | ✓ (50개) |",
        "",
        "→ **모든 필요 데이터 존재. 전제 1~3 모두 직접 검증 가능.**",
        "",
        "---",
        "",
    ]

    # 전제 1
    lines += [
        "## 전제 1: Persona Sensitivity",
        "",
        "**질문:** 페르소나 속성(income)에 따라 동일 아이템에 대한 hidden state가 달라지는가?",
        "",
        "### (a) 아이템별 페르소나 간 pairwise cosine similarity",
        "",
        f"- 전체 100개 cold items에 대한 50개 페르소나 pairwise cosine similarity",
        f"- mean = **{r1['pairwise_cos_mean']:.4f}**  std = {r1['pairwise_cos_std']:.4f}",
        "",
        f"→ 1.0에 매우 가까움: 동일 아이템에 대해 모든 페르소나의 hidden state가 거의 동일한 방향을 가짐.",
        f"→ 페르소나 간 코사인 유사도가 높더라도 **L2 크기 차이는 존재할 수 있음** (검증 (b) 참조).",
        "",
        "### (b) income 기반 분리 vs 랜덤 분리 L2 distance",
        "",
        f"| 비교 | L2 distance (평균) |",
        f"|------|-------------------|",
        f"| income 상위 10명 vs 하위 10명 | **{r1['income_l2_mean']:.2f}** |",
        f"| 랜덤 10명 vs 10명 (100회 평균) | {r1['random_l2_mean']:.2f} |",
        "",
        f"- t-statistic = {r1['t_stat']:.3f}  p-value = {r1['p_val']:.4f}  effect size = {r1['effect_size']:.3f}",
        "",
    ]
    if r1['p_val'] < 0.05:
        lines.append(f"→ income 기반 분리 L2 > 랜덤 분리 L2 (통계적으로 유의, p<0.05).")
    else:
        lines.append(f"→ income 기반 분리 L2가 랜덤 분리와 통계적으로 유의한 차이 없음 (p={r1['p_val']:.4f}).")
    lines += ["", f"### 판정: **{r1['verdict']}**", "", "---", ""]

    # 전제 2
    lines += [
        "## 전제 2: Attention as Market Modeling",
        "",
        "**질문:** Attention head가 아이템-관련 페르소나에 선택적으로 높은 가중치를 부여하는가?",
        "",
        "### (c) Attention Entropy 분석",
        "",
        f"| 지표 | 값 |",
        f"|------|---|",
        f"| 평균 entropy | **{r2['mean_entropy']:.4f}** |",
        f"| std entropy | {r2['std_entropy']:.4f} |",
        f"| uniform entropy (log 50) | {r2['uniform_entropy']:.4f} |",
        f"| entropy ratio (mean/uniform) | **{r2['entropy_ratio']:.4f}** |",
        f"| 분류 | **{r2['entropy_verdict']}** |",
        "",
    ]
    if r2['entropy_ratio'] < 0.85:
        lines.append("→ 평균 entropy가 uniform의 85% 미만: Attention이 특정 페르소나에 집중적으로 가중치 부여.")
    elif r2['entropy_ratio'] < 0.95:
        lines.append("→ 평균 entropy가 uniform의 85~95% 수준: 약한 집중 경향.")
    else:
        lines.append("→ 평균 entropy가 uniform에 근접: Attention이 페르소나를 거의 구분하지 못함.")
    lines += [""]

    # 카테고리별 상위 페르소나
    lines += ["### (b) 카테고리별 상위 5개 페르소나", ""]
    for cat, top5 in sorted(r2["cat_top5"].items()):
        lines.append(f"**[{cat}]**")
        lines.append("| Rank | Persona | Weight | Budget | Economic Status | FOODS pref |")
        lines.append("|------|---------|--------|--------|-----------------|------------|")
        for rank, t in enumerate(top5, 1):
            lines.append(
                f"| {rank} | P{t['persona_idx']+1:03d} | {t['weight']:.4f} | "
                f"${t['weekly_budget']:.0f} | {t['economic_status']} | "
                f"{t['category_pref'].get('FOODS', 0):.1f} |"
            )
        lines.append("")

    lines += [f"### 판정: **{r2['verdict']}**", "", "---", ""]

    # 전제 3
    q_mae = r3["quartile_mae"]
    lines += [
        "## 전제 3: Quantitative Transfer",
        "",
        "**질문:** warm items 내부에서 embedding이 아이템별 판매량 차이를 포착하는가?",
        "",
        "### (c) 상관계수",
        "",
        f"| 지표 | 값 | p-value |",
        f"|------|---|---------|",
        f"| Pearson r | **{r3['pearson_r']:.4f}** | {r3['pearson_p']:.4f} |",
        f"| Spearman ρ | **{r3['spearman_r']:.4f}** | {r3['spearman_p']:.4f} |",
        "",
        f"- 실제 아이템별 평균 weekly sales: {r3['item_true_mean_mean']:.2f}",
        f"- 예측 아이템별 평균 weekly sales: {r3['item_pred_mean_mean']:.2f}",
        "",
        "### (b) 판매량 분위별 MAE (5-fold CV)",
        "",
        "| 분위 | 실제 판매량 (평균) | MAE | n |",
        "|------|-------------------|-----|---|",
    ]
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qd = q_mae[q]
        lines.append(f"| {q} | {qd['true_mean']:.2f} | {qd['mae']:.2f} | {qd['n']} |")
    lines += [
        "",
        f"- warm 5-fold CV MAE (전체): {r3['cv_mae']:.2f}",
        "",
    ]
    q1_mae = q_mae["Q1"]["mae"]
    q4_mae = q_mae["Q4"]["mae"]
    if q1_mae > q4_mae * 1.5:
        lines.append(f"→ Q1(저판매) MAE({q1_mae:.1f}) >> Q4(고판매) MAE({q4_mae:.1f}): head가 평균으로 수렴하는 경향 확인.")
    else:
        lines.append(f"→ Q1(저판매) MAE({q1_mae:.1f}) vs Q4(고판매) MAE({q4_mae:.1f}): 분위별 MAE 차이가 제한적.")
    lines += ["", f"### 판정: **{r3['verdict']}**", "", "---", ""]

    # 종합
    lines += [
        "## 종합 판정",
        "",
        "| 전제 | 내용 | 판정 |",
        "|------|------|------|",
        f"| 전제 1: Persona Sensitivity | 페르소나 속성이 hidden state에 인코딩됨 | **{r1['verdict']}** |",
        f"| 전제 2: Attention Market Modeling | Attention이 관련 페르소나를 선별함 | **{r2['verdict']}** |",
        f"| 전제 3: Quantitative Transfer | Warm에서 판매량 순위를 예측할 수 있음 | **{r3['verdict']}** |",
        "",
        "### 함의",
        "",
    ]

    verdicts = [r1["verdict"], r2["verdict"], r3["verdict"]]
    n_fail = sum(1 for v in verdicts if v == "실패")
    n_partial = sum(1 for v in verdicts if "부분" in v)

    if n_fail == 0 and n_partial == 0:
        lines.append("세 전제 모두 지지됨. 방법론의 기본 전제 검증 완료.")
    elif n_fail == 0:
        lines.append("부분적 지지가 있으나 전제가 완전히 기각되지는 않음. 개선 방향 탐색 필요.")
    elif n_fail >= 2:
        lines.append("핵심 전제가 2개 이상 실패. Phase 2 전에 방법론 재설계 필요.")
    else:
        lines.append("1개 전제 실패. 해당 전제를 보강하는 추가 실험 필요.")

    lines += [
        "",
        "### 다음 단계 권고",
        "",
        "- 전제 1 결과에 따라: 프롬프트 변경(더 강한 persona conditioning) 또는 대조 학습 도입 검토",
        "- 전제 2 결과에 따라: attention weight 시각화로 카테고리별 일치 여부 수동 확인",
        "- 전제 3 결과에 따라: Phase 2 (warm 2,949개 확장)로 샘플 수 증가 후 재검증",
        "",
        "---",
        "",
        "**시각화 파일:** `experiments/exp012_premise_verification/figures/`",
        "**분석 스크립트:** `scripts/exp012_premise_verification.py`",
    ]

    report_path = REPORT_DIR / "premise_verification_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", report_path)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Exp012: 방법론 전제 독립 검증 시작 ===")

    # 데이터 로드
    embs      = load_embeddings()
    cold_meta, warm_meta = load_item_meta()
    personas  = load_personas(n=50)
    y_warm    = load_sales_targets(warm_meta)

    # 검증 실행
    r1 = verify_persona_sensitivity(
        embs["cold_raw"], cold_meta, personas
    )
    r2 = verify_attention_market(
        embs["cold_raw"], embs["warm_raw"],
        cold_meta, warm_meta, personas
    )
    r3 = verify_quantitative_transfer(
        embs["warm_mean"], y_warm, warm_meta
    )

    # 보고서 작성
    write_report(r1, r2, r3)

    # 터미널 요약
    print("\n" + "=" * 60)
    print("=== Exp012 결과 요약 ===")
    print(f"전제 1 (Persona Sensitivity):       {r1['verdict']}")
    print(f"  income L2={r1['income_l2_mean']:.2f} vs random={r1['random_l2_mean']:.2f}  p={r1['p_val']:.4f}")
    print(f"전제 2 (Attention Market Modeling): {r2['verdict']}")
    print(f"  entropy={r2['mean_entropy']:.4f} (uniform={r2['uniform_entropy']:.4f}, ratio={r2['entropy_ratio']:.3f})")
    print(f"전제 3 (Quantitative Transfer):     {r3['verdict']}")
    print(f"  Spearman={r3['spearman_r']:.4f}  Pearson={r3['pearson_r']:.4f}  CV_MAE={r3['cv_mae']:.2f}")
    print("=" * 60)
    print(f"보고서: docs/diagnosis/premise_verification_report.md")
    print(f"시각화: experiments/exp012_premise_verification/figures/")


if __name__ == "__main__":
    main()
