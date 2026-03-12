"""Exp013: Warm 내부 Scaling Law 및 분포 불일치 검증.

실험 A: 학습 셋 크기별 Attention 분화 (Scaling Law)
    - 학습 풀 250개, 고정 테스트 50개 (stratified sampling)
    - 학습 셋 크기: [30, 50, 75, 100, 150, 200, 250]
    - 각 크기에서 3 seeds 반복
    - 측정: test MAE, attention entropy, 카테고리별 top-1 분화, P042 weight

실험 B: 분포 불일치 영향 정량화
    - B-1 (matched):    Q2+Q3+Q4 → Q2+Q3+Q4 (50개)
    - B-2 (cold-like):  Q2+Q3+Q4 → Q1 전체 (75개)
    - B-3 (reverse):    Q1+Q2+Q3 → Q4 전체 (75개)

사용 데이터: warm_raw.pt (300×50×5120), y_warm 재계산
기존 코드 수정 없음. 분석 스크립트 신규 작성.

출력:
    scripts/exp013_scaling_law.py
    experiments/exp013_scaling_law/figures/
    docs/diagnosis/scaling_law_report.md
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
import torch.nn as nn
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
EMB_DIR   = ROOT / "experiments/exp011_v3_pipeline/embeddings"
FIG_DIR   = ROOT / "experiments/exp013_scaling_law/figures"
REPORT_DIR = ROOT / "docs/diagnosis"
DATA_DIR  = ROOT / "data/processed"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── exp011 Attention+Bottleneck 구조 (동일 하이퍼파라미터) ──────────────────
HIDDEN_DIM   = 5120
N_WEEKS      = 17
BOTTLENECK   = 64
DROPOUT      = 0.5
EPOCHS       = 200
PATIENCE     = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-2


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
        """x: (N, P, H) → pred (N, W), attn_w (N, P)."""
        scores = self.attn(x).squeeze(-1)          # (N, P)
        attn_w = torch.softmax(scores, dim=-1)     # (N, P)
        fused  = (x * attn_w.unsqueeze(-1)).sum(1) # (N, H)
        return self.head(fused), attn_w


def train_attn_model(
    X_raw: torch.Tensor,   # (N, P, H)
    y: torch.Tensor,       # (N, W)
    device: str = "cpu",
    seed: int = 42,
) -> AttnBottleneck:
    """exp011과 동일한 설정으로 AttnBottleneck 전체 학습 (5-fold 없이 단일 학습)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AttnBottleneck().to(device)
    opt   = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    X_d   = X_raw.to(device)
    y_d   = y.to(device)

    best_loss = float("inf")
    no_imp    = 0
    best_state = None

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred, _ = model(X_d)
        loss = nn.functional.mse_loss(pred, y_d)
        loss.backward()
        opt.step()
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


def get_attn_weights(model: AttnBottleneck, X_raw: torch.Tensor) -> np.ndarray:
    """X_raw: (N, P, H) → attention weights (N, P) [numpy]."""
    model.eval()
    with torch.no_grad():
        _, attn_w = model(X_raw)
    return attn_w.cpu().numpy()


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data() -> tuple[torch.Tensor, np.ndarray, pd.DataFrame]:
    """warm_raw (300, 50, 5120), y_warm (300, 17), warm_meta."""
    logger.info("데이터 로드 중...")
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", weights_only=True)   # (300, 50, 5120)
    meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    warm_meta = meta[meta["is_cold"] == False].reset_index(drop=True)

    cold_test = pd.read_csv(DATA_DIR / "cold_start/cold_test.csv",  parse_dates=["date"])
    warm_test = pd.read_csv(DATA_DIR / "cold_start/warm_test.csv",  parse_dates=["date"])
    date_start = str(cold_test["date"].min().date())
    date_end   = str(cold_test["date"].max().date())
    warm_ids   = warm_meta["item_id"].tolist()

    sub = warm_test[
        warm_test["item_id"].isin(warm_ids) &
        (warm_test["date"] >= date_start) &
        (warm_test["date"] <= date_end)
    ].copy()
    sub["week"] = pd.to_datetime(sub["date"]).dt.to_period("W")
    w = (sub.groupby(["item_id", "week"])["sales"]
           .sum()
           .unstack("week")
           .reindex(warm_ids)
           .fillna(0))
    y_warm = w.values.astype(np.float32)

    logger.info("warm_raw: %s  y_warm: %s  mean=%.2f",
                tuple(warm_raw.shape), y_warm.shape, y_warm.mean())
    return warm_raw, y_warm, warm_meta


# ─── 분석 유틸리티 ────────────────────────────────────────────────────────────

def compute_metrics(model: AttnBottleneck,
                    X_test: torch.Tensor,
                    y_test: np.ndarray,
                    test_meta: pd.DataFrame,
                    p042_idx: int = 41) -> dict:
    """테스트 셋에 대한 지표 계산."""
    attn_w = get_attn_weights(model, X_test)   # (N_test, 50)

    # MAE
    with torch.no_grad():
        pred, _ = model(X_test)
    mae = float(np.abs(pred.cpu().numpy() - y_test).mean())

    # Attention entropy
    def entropy(w: np.ndarray) -> float:
        w = np.clip(w, 1e-12, None)
        return float(-np.sum(w * np.log(w)))
    item_ent = np.array([entropy(attn_w[i]) for i in range(len(attn_w))])
    mean_ent = float(item_ent.mean())

    # P042 평균 weight
    p042_w = float(attn_w[:, p042_idx].mean())

    # 카테고리별 top-1 페르소나 분화 분석
    top1_per_item = attn_w.argmax(axis=1)    # (N_test,)
    cats = test_meta["cat_id"].values
    cat_top1 = {}
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        mask = cats == cat
        if mask.sum() == 0:
            cat_top1[cat] = {"top1_mode": -1, "consistency": 0.0}
            continue
        top1_cat = top1_per_item[mask]
        mode_val = int(np.bincount(top1_cat).argmax())
        consistency = float((top1_cat == mode_val).mean())
        cat_top1[cat] = {"top1_mode": mode_val, "consistency": consistency}

    # 3개 카테고리 top-1이 서로 다른지
    modes = [cat_top1[c]["top1_mode"] for c in ["FOODS", "HOBBIES", "HOUSEHOLD"]]
    cat_differentiated = len(set(modes)) == 3

    return {
        "mae":               mae,
        "mean_entropy":      mean_ent,
        "p042_weight":       p042_w,
        "cat_top1":          cat_top1,
        "cat_differentiated": cat_differentiated,
    }


# ─── 실험 A: Scaling Law ──────────────────────────────────────────────────────

def experiment_a(warm_raw: torch.Tensor, y_warm: np.ndarray,
                 warm_meta: pd.DataFrame) -> dict:
    """
    학습 셋 크기별 [30, 50, 75, 100, 150, 200, 250] × 3 seeds.
    고정 테스트 50개: 분위별 균등 stratified sampling.
    """
    logger.info("=== 실험 A: Scaling Law 시작 ===")

    TRAIN_SIZES = [30, 50, 75, 100, 150, 200, 250]
    SEEDS       = [42, 123, 777]

    # 고정 테스트 50개: stratified (분위별 12~13개)
    item_mean   = y_warm.mean(axis=1)
    quartile_labels = pd.qcut(
        pd.Series(item_mean), q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).to_numpy()

    rng_test = np.random.default_rng(0)
    test_idx = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        q_idx = np.where(quartile_labels == q)[0]
        chosen = rng_test.choice(q_idx, size=min(13, len(q_idx)), replace=False)
        test_idx.extend(chosen.tolist())
    test_idx = sorted(test_idx[:50])

    pool_idx = [i for i in range(300) if i not in set(test_idx)]  # 250개

    X_test  = warm_raw[test_idx]
    y_test  = y_warm[test_idx]
    meta_test = warm_meta.iloc[test_idx].reset_index(drop=True)

    logger.info("  테스트 셋 구성: %d개  풀: %d개", len(test_idx), len(pool_idx))
    logger.info("  테스트 카테고리 분포: %s",
                dict(meta_test["cat_id"].value_counts()))

    results_a: dict[int, list[dict]] = {n: [] for n in TRAIN_SIZES}

    for n_train in TRAIN_SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            tr_idx = rng.choice(pool_idx, size=n_train, replace=False).tolist()

            X_tr = warm_raw[tr_idx]
            y_tr = torch.tensor(y_warm[tr_idx])

            model = train_attn_model(X_tr, y_tr, seed=seed)
            metrics = compute_metrics(model, X_test, y_test, meta_test)
            results_a[n_train].append(metrics)

            logger.info(
                "  n=%d seed=%d | MAE=%.2f entropy=%.4f P042=%.4f diff=%s",
                n_train, seed, metrics["mae"], metrics["mean_entropy"],
                metrics["p042_weight"], metrics["cat_differentiated"],
            )

    return {"results": results_a, "test_idx": test_idx, "pool_idx": pool_idx,
            "meta_test": meta_test}


# ─── 실험 B: 분포 불일치 ──────────────────────────────────────────────────────

def experiment_b(warm_raw: torch.Tensor, y_warm: np.ndarray,
                 warm_meta: pd.DataFrame) -> dict:
    """
    B-1: Q2+Q3+Q4(200개 학습) → Q2+Q3+Q4(50개 테스트)  [matched]
    B-2: Q2+Q3+Q4(200개 학습) → Q1 전체 (75개 테스트)   [cold-like]
    B-3: Q1+Q2+Q3(200개 학습) → Q4 전체 (75개 테스트)   [reverse]
    각 변형에서 3 seeds 반복.
    """
    logger.info("=== 실험 B: 분포 불일치 시작 ===")

    SEEDS = [42, 123, 777]
    item_mean = y_warm.mean(axis=1)
    q_labels = pd.qcut(
        pd.Series(item_mean), q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).to_numpy()

    q_idx = {q: np.where(q_labels == q)[0] for q in ["Q1", "Q2", "Q3", "Q4"]}
    logger.info("  분위별 아이템 수: %s", {q: len(v) for q, v in q_idx.items()})

    results_b: dict[str, list[dict]] = {"B1": [], "B2": [], "B3": []}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        # B-1: matched (Q2+Q3+Q4 → Q2+Q3+Q4 50개)
        high_pool = np.concatenate([q_idx["Q2"], q_idx["Q3"], q_idx["Q4"]])
        tr_b1 = rng.choice(high_pool, size=200, replace=False)
        remaining = np.setdiff1d(high_pool, tr_b1)
        te_b1 = rng.choice(remaining, size=min(50, len(remaining)), replace=False)
        X_tr_b1 = warm_raw[tr_b1]; y_tr_b1 = torch.tensor(y_warm[tr_b1])
        X_te_b1 = warm_raw[te_b1]; y_te_b1 = y_warm[te_b1]
        meta_te_b1 = warm_meta.iloc[te_b1].reset_index(drop=True)
        m_b1 = train_attn_model(X_tr_b1, y_tr_b1, seed=seed)
        metrics_b1 = compute_metrics(m_b1, X_te_b1, y_te_b1, meta_te_b1)
        results_b["B1"].append(metrics_b1)

        # B-2: cold-like (Q2+Q3+Q4 → Q1 전체)
        te_b2 = q_idx["Q1"]
        X_te_b2 = warm_raw[te_b2]; y_te_b2 = y_warm[te_b2]
        meta_te_b2 = warm_meta.iloc[te_b2].reset_index(drop=True)
        m_b2 = train_attn_model(X_tr_b1, y_tr_b1, seed=seed)  # 동일 학습셋
        metrics_b2 = compute_metrics(m_b2, X_te_b2, y_te_b2, meta_te_b2)
        results_b["B2"].append(metrics_b2)

        # B-3: reverse (Q1+Q2+Q3 → Q4 전체)
        low_pool = np.concatenate([q_idx["Q1"], q_idx["Q2"], q_idx["Q3"]])
        tr_b3 = rng.choice(low_pool, size=200, replace=False)
        te_b3 = q_idx["Q4"]
        X_tr_b3 = warm_raw[tr_b3]; y_tr_b3 = torch.tensor(y_warm[tr_b3])
        X_te_b3 = warm_raw[te_b3]; y_te_b3 = y_warm[te_b3]
        meta_te_b3 = warm_meta.iloc[te_b3].reset_index(drop=True)
        m_b3 = train_attn_model(X_tr_b3, y_tr_b3, seed=seed)
        metrics_b3 = compute_metrics(m_b3, X_te_b3, y_te_b3, meta_te_b3)
        results_b["B3"].append(metrics_b3)

        logger.info(
            "  seed=%d | B1_MAE=%.2f  B2_MAE=%.2f  B3_MAE=%.2f",
            seed, metrics_b1["mae"], metrics_b2["mae"], metrics_b3["mae"],
        )

    return results_b


# ─── 시각화 ───────────────────────────────────────────────────────────────────

def plot_experiment_a(results_a: dict) -> None:
    TRAIN_SIZES = sorted(results_a.keys())
    UNIFORM_ENT = np.log(50)

    def agg(metric_key: str) -> tuple[list[float], list[float]]:
        means, stds = [], []
        for n in TRAIN_SIZES:
            vals = [r[metric_key] for r in results_a[n]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return means, stds

    mae_m, mae_s = agg("mae")
    ent_m, ent_s = agg("mean_entropy")
    p42_m, p42_s = agg("p042_weight")

    # 카테고리 분화율 (3 seeds 중 cat_differentiated == True 비율)
    diff_rates = []
    for n in TRAIN_SIZES:
        rate = np.mean([r["cat_differentiated"] for r in results_a[n]])
        diff_rates.append(rate)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    x = TRAIN_SIZES

    # (1) MAE
    axes[0].errorbar(x, mae_m, yerr=mae_s, marker="o", color="steelblue", capsize=4)
    axes[0].set_xlabel("Train set size"); axes[0].set_ylabel("Test MAE")
    axes[0].set_title("(A) Test MAE vs Train Size")
    axes[0].grid(alpha=0.3)

    # (2) Attention entropy
    axes[1].errorbar(x, ent_m, yerr=ent_s, marker="s", color="orangered", capsize=4)
    axes[1].axhline(UNIFORM_ENT, color="green", linestyle="--",
                    label=f"uniform={UNIFORM_ENT:.2f}")
    axes[1].set_xlabel("Train set size"); axes[1].set_ylabel("Mean attention entropy")
    axes[1].set_title("(A) Attention Entropy vs Train Size")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # (3) P042 weight
    axes[2].errorbar(x, p42_m, yerr=p42_s, marker="^", color="purple", capsize=4)
    axes[2].axhline(1/50, color="gray", linestyle="--", label="uniform=0.02")
    axes[2].set_xlabel("Train set size"); axes[2].set_ylabel("P042 mean weight")
    axes[2].set_title("(A) P042 Dominance vs Train Size")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    # (4) 카테고리 분화율
    axes[3].plot(x, diff_rates, marker="D", color="seagreen")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel("Train set size"); axes[3].set_ylabel("Differentiation rate")
    axes[3].set_title("(A) Category Differentiation Rate")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_a_scaling_law.png", dpi=150)
    plt.close(fig)
    logger.info("실험 A 시각화 저장: %s", FIG_DIR / "exp_a_scaling_law.png")


def plot_experiment_b(results_b: dict) -> None:
    variants = ["B1", "B2", "B3"]
    labels   = ["B-1 matched", "B-2 cold-like", "B-3 reverse"]
    colors   = ["steelblue", "orangered", "seagreen"]

    mae_means   = [np.mean([r["mae"] for r in results_b[v]]) for v in variants]
    mae_stds    = [np.std([r["mae"]  for r in results_b[v]]) for v in variants]
    ent_means   = [np.mean([r["mean_entropy"] for r in results_b[v]]) for v in variants]
    p042_means  = [np.mean([r["p042_weight"]  for r in results_b[v]]) for v in variants]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.arange(3)
    axes[0].bar(x, mae_means, yerr=mae_stds, color=colors, capsize=5, alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Test MAE"); axes[0].set_title("(B) Test MAE by Variant")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, ent_means, color=colors, alpha=0.85, edgecolor="white")
    axes[1].axhline(np.log(50), color="green", linestyle="--", label="uniform")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean attention entropy")
    axes[1].set_title("(B) Attention Entropy by Variant")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, p042_means, color=colors, alpha=0.85, edgecolor="white")
    axes[2].axhline(1/50, color="gray", linestyle="--", label="uniform=0.02")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("P042 mean weight")
    axes[2].set_title("(B) P042 Dominance by Variant")
    axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_b_distribution_mismatch.png", dpi=150)
    plt.close(fig)
    logger.info("실험 B 시각화 저장: %s", FIG_DIR / "exp_b_distribution_mismatch.png")


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(results_a: dict, results_b: dict,
                 warm_meta: pd.DataFrame) -> None:
    TRAIN_SIZES  = sorted(results_a.keys())
    UNIFORM_ENT  = np.log(50)

    lines = [
        "# Warm 내부 Scaling Law 및 분포 불일치 검증 보고서 (Exp013)",
        "",
        "**작성일:** 2026-03-10",
        "**배경:** exp012에서 P042가 모든 카테고리에서 지배적 → 데이터 부족 vs embedding 문제 구분 필요",
        "",
        "---",
        "",
        "## 실험 설계",
        "",
        "- **공통 모델:** AttnBottleneck (exp011 동일: hidden=5120, bottleneck=64, dropout=0.5,",
        "  epochs=200, patience=20, lr=1e-3, weight_decay=1e-2)",
        "- **데이터:** warm_raw.pt (300×50×5120), y_warm (300×17 주간 판매량)",
        "- **P042 = P042 (index 41):** weekly_budget=$260, FOODS pref=0.7",
        "",
        "---",
        "",
        "## 실험 A: Scaling Law (학습 셋 크기별 Attention 분화)",
        "",
        "**설계:** 고정 테스트 50개(분위별 stratified) + 학습 풀 250개",
        "",
        "### 결과 테이블 (mean ± std, 3 seeds)",
        "",
        "| Train N | MAE | Entropy | Entropy/Uniform | P042 Weight | Cat. Diff. Rate |",
        "|---------|-----|---------|-----------------|-------------|-----------------|",
    ]

    for n in TRAIN_SIZES:
        mae_m  = np.mean([r["mae"]          for r in results_a[n]])
        mae_s  = np.std( [r["mae"]          for r in results_a[n]])
        ent_m  = np.mean([r["mean_entropy"] for r in results_a[n]])
        ent_s  = np.std( [r["mean_entropy"] for r in results_a[n]])
        p42_m  = np.mean([r["p042_weight"]  for r in results_a[n]])
        p42_s  = np.std( [r["p042_weight"]  for r in results_a[n]])
        diff_r = np.mean([r["cat_differentiated"] for r in results_a[n]])
        lines.append(
            f"| {n:4d} | {mae_m:.1f}±{mae_s:.1f} | "
            f"{ent_m:.4f}±{ent_s:.4f} | "
            f"{ent_m/UNIFORM_ENT:.3f} | "
            f"{p42_m:.4f}±{p42_s:.4f} | "
            f"{diff_r:.2f} |"
        )

    # 판단
    p42_250 = np.mean([r["p042_weight"] for r in results_a[250]])
    p42_30  = np.mean([r["p042_weight"] for r in results_a[30]])
    diff_250 = np.mean([r["cat_differentiated"] for r in results_a[250]])
    ent_250  = np.mean([r["mean_entropy"] for r in results_a[250]])

    lines += [""]
    if p42_250 < p42_30 * 0.8 and diff_250 > 0.3:
        verdict_a = "데이터 부족이 주 원인 → warm 확장(경로 B) 정당화"
        lines.append(f"→ N=250에서 P042 weight({p42_250:.4f}) < N=30({p42_30:.4f}) × 0.8 "
                     f"AND 카테고리 분화율({diff_250:.2f}) > 0.3")
        lines.append(f"→ **판정: {verdict_a}**")
    elif p42_250 < p42_30 * 0.9:
        verdict_a = "데이터 부족이 부분 원인 (완전 해소 불가) → warm 확장 + embedding 개선 병행 필요"
        lines.append(f"→ N=250에서 P042 감소 경향 있으나 카테고리 분화 미흡")
        lines.append(f"→ **판정: {verdict_a}**")
    else:
        verdict_a = "embedding 자체의 문제 → 프롬프트/레이어 변경 필요"
        lines.append(f"→ N=250에서도 P042 지배 해소되지 않음 ({p42_250:.4f})")
        lines.append(f"→ **판정: {verdict_a}**")

    lines += [
        "",
        "### 카테고리별 top-1 페르소나 상세 (N=250, 첫 번째 seed)",
        "",
        "| 카테고리 | top-1 모드 | 일관성 |",
        "|----------|-----------|--------|",
    ]
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        top1_info = results_a[250][0]["cat_top1"].get(cat, {})
        lines.append(f"| {cat} | P{top1_info.get('top1_mode', -1)+1:03d} | "
                     f"{top1_info.get('consistency', 0):.2f} |")

    lines += [
        "",
        "---",
        "",
        "## 실험 B: 분포 불일치 영향 정량화",
        "",
        "| 변형 | 학습 셋 | 테스트 셋 | MAE | Entropy | P042 Weight |",
        "|------|---------|----------|-----|---------|-------------|",
    ]
    variant_desc = {
        "B1": ("Q2+Q3+Q4 (200개)", "Q2+Q3+Q4 (50개) [matched]"),
        "B2": ("Q2+Q3+Q4 (200개)", "Q1 전체 (75개) [cold-like]"),
        "B3": ("Q1+Q2+Q3 (200개)", "Q4 전체 (75개) [reverse]"),
    }
    b_mae = {}
    for v, (tr_desc, te_desc) in variant_desc.items():
        mae_m  = np.mean([r["mae"]          for r in results_b[v]])
        mae_s  = np.std( [r["mae"]          for r in results_b[v]])
        ent_m  = np.mean([r["mean_entropy"] for r in results_b[v]])
        p42_m  = np.mean([r["p042_weight"]  for r in results_b[v]])
        b_mae[v] = mae_m
        lines.append(
            f"| {v} | {tr_desc} | {te_desc} | "
            f"{mae_m:.1f}±{mae_s:.1f} | {ent_m:.4f} | {p42_m:.4f} |"
        )

    lines += [""]
    ratio = b_mae["B2"] / (b_mae["B1"] + 1e-9)
    if ratio > 1.5:
        verdict_b = "분포 불일치가 성능 저하의 주 원인 → warm 확장 시 cold와 유사한 판매량 아이템 포함 필수"
    elif ratio > 1.2:
        verdict_b = "분포 불일치가 성능에 영향 (중간 수준) → cold-range 아이템 포함 권고"
    else:
        verdict_b = "분포 불일치는 주 원인 아님 → 다른 원인 탐색 필요"

    lines += [
        f"- B-2/B-1 MAE 비율: **{ratio:.2f}x** (B-1={b_mae['B1']:.1f}, B-2={b_mae['B2']:.1f})",
        f"- **판정: {verdict_b}**",
        "",
        "---",
        "",
        "## 종합 결론",
        "",
        f"| 실험 | 질문 | 판정 |",
        f"|------|------|------|",
        f"| A (Scaling Law) | P042 지배가 데이터 부족 때문인가? | {verdict_a} |",
        f"| B (Distribution) | 분포 불일치가 주 성능 저하 원인인가? | {verdict_b} |",
        "",
        "### Phase 2 경로 결정 시사점",
        "",
    ]

    if "warm 확장" in verdict_a:
        lines.append("- 실험 A: warm 데이터 확장이 카테고리 분화 개선에 효과적일 것으로 예상")
    else:
        lines.append("- 실험 A: warm 확장만으로는 불충분 — 프롬프트 또는 레이어 변경 병행 필요")

    if "주 원인" in verdict_b and "아님" not in verdict_b:
        lines.append("- 실험 B: warm 확장 시 저판매(Q1) 아이템을 충분히 포함해야 cold 전이 성능 확보 가능")
    else:
        lines.append("- 실험 B: 분포 불일치 외 다른 요인 탐색 필요")

    lines += [
        "",
        "---",
        "",
        "**시각화:** `experiments/exp013_scaling_law/figures/`",
        "**스크립트:** `scripts/exp013_scaling_law.py`",
    ]

    report_path = REPORT_DIR / "scaling_law_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", report_path)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Exp013: Scaling Law + 분포 불일치 검증 시작 ===")

    warm_raw, y_warm, warm_meta = load_data()

    # 실험 A
    res_a_full = experiment_a(warm_raw, y_warm, warm_meta)
    results_a  = res_a_full["results"]
    plot_experiment_a(results_a)

    # 실험 B
    results_b = experiment_b(warm_raw, y_warm, warm_meta)
    plot_experiment_b(results_b)

    # 보고서
    write_report(results_a, results_b, warm_meta)

    # 터미널 요약
    print("\n" + "=" * 65)
    print("=== Exp013 결과 요약 ===")
    print()
    print("실험 A (Scaling Law):")
    TRAIN_SIZES = sorted(results_a.keys())
    print(f"  {'N':>4}  {'MAE':>7}  {'Entropy':>8}  {'P042':>7}  {'CatDiff':>8}")
    UNIFORM_ENT = np.log(50)
    for n in TRAIN_SIZES:
        mae_m = np.mean([r["mae"]          for r in results_a[n]])
        ent_m = np.mean([r["mean_entropy"] for r in results_a[n]])
        p42_m = np.mean([r["p042_weight"]  for r in results_a[n]])
        diff_r = np.mean([r["cat_differentiated"] for r in results_a[n]])
        print(f"  {n:>4}  {mae_m:>7.2f}  {ent_m:>8.4f}  {p42_m:>7.4f}  {diff_r:>8.2f}")
    print()
    print("실험 B (분포 불일치):")
    for v in ["B1", "B2", "B3"]:
        mae_m = np.mean([r["mae"] for r in results_b[v]])
        ent_m = np.mean([r["mean_entropy"] for r in results_b[v]])
        p42_m = np.mean([r["p042_weight"]  for r in results_b[v]])
        print(f"  {v}: MAE={mae_m:.2f}  Entropy={ent_m:.4f}  P042={p42_m:.4f}")
    print("=" * 65)
    print("보고서: docs/diagnosis/scaling_law_report.md")
    print("시각화: experiments/exp013_scaling_law/figures/")


if __name__ == "__main__":
    main()
