"""Exp015: 유사 아이템 기반 파이프라인 핵심 가정 검증.

검증 1: Residual embedding 유사성 ↔ 판매량 유사성 상관
검증 2: K-NN 기반 LOO 예측 (embedding 유사성 vs 카테고리+가격)
검증 3: 유사 아이템 기반 Attention Head (item-specific head + rescale)

데이터: exp011 warm_raw.pt (300×50×5120), y_warm, sell_prices.csv
출력:
    docs/diagnosis/similar_item_verification.md
    experiments/exp015_similar_item/figures/
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 경로 ─────────────────────────────────────────────────────────────────────
EMB_DIR    = ROOT / "experiments/exp011_v3_pipeline/embeddings"
MODEL_DIR  = ROOT / "experiments/exp011_v3_pipeline/models"
FIG_DIR    = ROOT / "experiments/exp015_similar_item/figures"
REPORT_DIR = ROOT / "docs/diagnosis"
DATA_DIR   = ROOT / "data/processed/cold_start"
M5_DIR     = ROOT / "m5-forecasting-accuracy"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float]]:
    """warm_raw (300,50,5120), y_warm (300,17), warm_meta, price_map."""
    logger.info("데이터 로드 중...")
    warm_raw  = torch.load(EMB_DIR / "warm_raw.pt",  weights_only=True).numpy()
    warm_mean = torch.load(EMB_DIR / "warm_mean.pt", weights_only=True).numpy()

    meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_meta = meta[meta["is_cold"] == True].reset_index(drop=True)
    warm_meta = meta[meta["is_cold"] == False].reset_index(drop=True)

    # y_warm 계산 (cold_test 날짜 범위 사용)
    cold_test = pd.read_csv(DATA_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_test = pd.read_csv(DATA_DIR / "warm_test.csv",  parse_dates=["date"])
    date_start = str(cold_test["date"].min().date())
    date_end   = str(cold_test["date"].max().date())
    warm_ids   = warm_meta["item_id"].tolist()
    sub = warm_test[
        warm_test["item_id"].isin(warm_ids) &
        (warm_test["date"] >= date_start) &
        (warm_test["date"] <= date_end)
    ].copy()
    sub["week"] = pd.to_datetime(sub["date"]).dt.to_period("W")
    y_warm = (sub.groupby(["item_id", "week"])["sales"]
                .sum()
                .unstack("week")
                .reindex(warm_ids)
                .fillna(0)
                .values.astype(np.float32))

    # 가격 정보 (CA_1 기준 평균)
    sell_df   = pd.read_csv(M5_DIR / "sell_prices.csv")
    price_map = (sell_df[sell_df["store_id"] == "CA_1"]
                 .groupby("item_id")["sell_price"].mean().to_dict())

    logger.info("warm_raw=%s  y_warm=%s  mean_sales=%.2f",
                warm_raw.shape, y_warm.shape, y_warm.mean())
    return warm_raw, y_warm, warm_meta, price_map


# ─── 표현 계산 ────────────────────────────────────────────────────────────────

class Representations(NamedTuple):
    mean_pooled: np.ndarray   # (300, 5120)
    flat_pca:    np.ndarray   # (300, 64)  raw flat → PCA
    resid_pca:   np.ndarray   # (300, 64)  residual flat → PCA


def compute_representations(warm_raw: np.ndarray, pca_dim: int = 64) -> Representations:
    """3가지 아이템 표현 계산."""
    n_items, n_personas, hidden = warm_raw.shape
    logger.info("표현 계산 중 (n=%d, p=%d, h=%d)...", n_items, n_personas, hidden)

    # (a) mean-pooled: (300, 5120)
    mean_pooled = warm_raw.mean(axis=1)

    # (b) flat + PCA(64): raw → (300, 256000) → PCA
    logger.info("  (b) flat PCA 계산 중...")
    flat = warm_raw.reshape(n_items, -1)                    # (300, 256000)
    pca_b = PCA(n_components=pca_dim, svd_solver="randomized", random_state=42)
    flat_pca = pca_b.fit_transform(flat)
    logger.info("    flat PCA explained_var: %.3f", pca_b.explained_variance_ratio_.sum())

    # (c) residual flat + PCA(64): (raw - item_mean) → (300, 256000) → PCA
    logger.info("  (c) residual PCA 계산 중...")
    item_mean = warm_raw.mean(axis=1, keepdims=True)         # (300, 1, 5120)
    residual  = (warm_raw - item_mean).reshape(n_items, -1)  # (300, 256000)
    pca_c = PCA(n_components=pca_dim, svd_solver="randomized", random_state=42)
    resid_pca = pca_c.fit_transform(residual)
    logger.info("    residual PCA explained_var: %.3f", pca_c.explained_variance_ratio_.sum())

    return Representations(mean_pooled, flat_pca, resid_pca)


# ─── 유틸: 거리 행렬 ──────────────────────────────────────────────────────────

def cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine distance matrix (300×300)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    sim = Xn @ Xn.T
    return np.clip(1.0 - sim, 0.0, 2.0)


def cat_price_dist_matrix(warm_meta: pd.DataFrame,
                          price_map: dict[str, float]) -> np.ndarray:
    """카테고리+가격 기반 거리 행렬.

    거리 = |price_i - price_j| / price_scale + 10 * (cat_i != cat_j)
    → 다른 카테고리는 사실상 무한대로 취급.
    """
    n = len(warm_meta)
    cats   = warm_meta["cat_id"].values
    prices = np.array([price_map.get(iid, 0.0)
                       for iid in warm_meta["item_id"]])
    price_scale = prices.std() + 1e-6

    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            same_cat = (cats[i] == cats[j])
            pd_ij    = abs(prices[i] - prices[j]) / price_scale
            dist[i, j] = pd_ij if same_cat else (pd_ij + 10.0)
    return dist


# ─── 검증 1: Embedding 유사성 ↔ 판매량 유사성 ────────────────────────────────

def verify_similarity_sales_corr(
    reprs: Representations,
    y_warm: np.ndarray,
    warm_meta: pd.DataFrame,
    price_map: dict[str, float],
) -> dict:
    """K-NN 판매량 std가 전체 std보다 낮은지 확인."""
    logger.info("=== 검증 1: Embedding 유사성 ↔ 판매량 유사성 ===")

    item_sales_mean = y_warm.mean(axis=1)   # (300,)
    global_std = float(item_sales_mean.std())
    sales_diff_mat = np.abs(item_sales_mean[:, None] - item_sales_mean[None, :])  # (300,300)

    results: dict = {"global_std": global_std, "representations": {}}

    repr_dict = {
        "mean_pooled":  reprs.mean_pooled,
        "flat_pca64":   reprs.flat_pca,
        "residual_pca64": reprs.resid_pca,
    }
    # 카테고리+가격 포함
    cat_price_d = cat_price_dist_matrix(warm_meta, price_map)
    repr_dict["cat_price"] = None   # sentinel

    Ks = [5, 10, 20]
    fig, axes = plt.subplots(1, len(repr_dict), figsize=(5 * len(repr_dict), 4))

    mantel_rows = []
    for ax_idx, (name, X) in enumerate(repr_dict.items()):
        if X is not None:
            dist_mat = cosine_dist_matrix(X)
        else:
            dist_mat = cat_price_d

        # Mantel-like: upper triangle correlation
        tri_idx = np.triu_indices(len(warm_meta), k=1)
        dist_tri  = dist_mat[tri_idx]
        sales_tri = sales_diff_mat[tri_idx]
        r, p = stats.spearmanr(dist_tri, sales_tri)
        mantel_rows.append({"repr": name, "spearman_r": float(r), "p": float(p)})
        logger.info("  [%s] Mantel spearman r=%.4f  p=%.4f", name, r, p)

        # K-NN 판매량 std
        np.fill_diagonal(dist_mat, np.inf)
        knn_stds: dict[int, float] = {}
        for K in Ks:
            knn_idx    = np.argsort(dist_mat, axis=1)[:, :K]       # (300, K)
            knn_sales  = item_sales_mean[knn_idx]                   # (300, K)
            knn_std    = float(knn_sales.std(axis=1).mean())
            knn_stds[K] = knn_std
            logger.info("    K=%2d: K-NN sales std=%.2f (global %.2f)", K, knn_std, global_std)

        results["representations"][name] = {
            "mantel_r": float(r), "mantel_p": float(p), "knn_stds": knn_stds
        }

        ax = axes[ax_idx] if len(repr_dict) > 1 else axes
        ax.bar([f"K={k}" for k in Ks], [knn_stds[k] for k in Ks],
               color="steelblue", alpha=0.8)
        ax.axhline(global_std, color="red", linestyle="--", label=f"global std={global_std:.1f}")
        ax.set_title(f"{name}\nMantel r={r:.3f}")
        ax.set_ylabel("K-NN sales std")
        ax.legend()
        np.fill_diagonal(dist_mat, 0.0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "v1_knn_sales_std.png", dpi=150)
    plt.close(fig)

    results["mantel_table"] = mantel_rows
    return results


# ─── 검증 2: K-NN LOO 예측 ────────────────────────────────────────────────────

def knn_loo_predict(dist_mat: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """LOO K-NN weighted prediction. y: (n, n_weeks). returns pred (n, n_weeks)."""
    n = len(dist_mat)
    preds = np.zeros_like(y)
    D = dist_mat.copy()
    np.fill_diagonal(D, np.inf)
    for i in range(n):
        idx = np.argsort(D[i])[:K]
        dists = D[i][idx] + 1e-12
        weights = 1.0 / dists
        weights /= weights.sum()
        preds[i] = (y[idx] * weights[:, None]).sum(axis=0)
    return preds


def verify_knn_prediction(
    reprs: Representations,
    y_warm: np.ndarray,
    warm_meta: pd.DataFrame,
    price_map: dict[str, float],
) -> dict:
    """K × 유사성 기준별 LOO MAE."""
    logger.info("=== 검증 2: K-NN LOO 예측 ===")

    # 비교 기준값 (exp011/014 결과)
    GLOBAL_MEAN_MAE   = 96.23
    GLOBAL_ATTN_ORIG  = 62.5
    GLOBAL_ATTN_RESID = 55.8

    dist_mats = {
        "mean_pooled":    cosine_dist_matrix(reprs.mean_pooled),
        "flat_pca64":     cosine_dist_matrix(reprs.flat_pca),
        "residual_pca64": cosine_dist_matrix(reprs.resid_pca),
        "cat_price":      cat_price_dist_matrix(warm_meta, price_map),
    }

    Ks = [5, 10, 20, 50]
    rows = []
    for sim_name, D in dist_mats.items():
        for K in Ks:
            preds = knn_loo_predict(D, y_warm, K)
            mae   = float(np.abs(y_warm - preds).mean())
            rows.append({"sim": sim_name, "K": K, "mae": mae})
            logger.info("  [%s] K=%2d → MAE=%.2f", sim_name, K, mae)

    # 테이블 출력
    df = pd.DataFrame(rows).pivot(index="K", columns="sim", values="mae")
    logger.info("\n%s", df.to_string())

    # 시각화
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"mean_pooled": "o", "flat_pca64": "s",
               "residual_pca64": "^", "cat_price": "D"}
    colors  = {"mean_pooled": "steelblue", "flat_pca64": "darkorange",
               "residual_pca64": "seagreen", "cat_price": "purple"}
    for sim_name in dist_mats:
        sub = [r for r in rows if r["sim"] == sim_name]
        ax.plot([r["K"] for r in sub], [r["mae"] for r in sub],
                marker=markers[sim_name], label=sim_name,
                color=colors[sim_name])
    ax.axhline(GLOBAL_MEAN_MAE,   color="gray",   linestyle=":", label=f"Global mean {GLOBAL_MEAN_MAE}")
    ax.axhline(GLOBAL_ATTN_ORIG,  color="black",  linestyle="--", label=f"Attn+BN orig {GLOBAL_ATTN_ORIG}")
    ax.axhline(GLOBAL_ATTN_RESID, color="red",    linestyle="--", label=f"Attn+BN resid {GLOBAL_ATTN_RESID}")
    ax.set_xlabel("K"); ax.set_ylabel("LOO MAE")
    ax.set_title("V2: K-NN LOO Prediction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_knn_loo_mae.png", dpi=150)
    plt.close(fig)

    return {"rows": rows, "df_table": df.to_dict()}


# ─── 검증 3: Item-specific Head + Rescale ─────────────────────────────────────

class AttnBottleneck(torch.nn.Module):
    """exp011과 동일한 Attention+Bottleneck head."""
    def __init__(self, hidden: int = 5120, bottleneck: int = 64,
                 n_weeks: int = 17, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = torch.nn.Linear(hidden, 1, bias=False)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, bottleneck), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, P, H) → pred (N, n_weeks), attn_w (N, P)."""
        scores = self.attn(x).squeeze(-1)                    # (N, P)
        attn_w = torch.softmax(scores, dim=-1)               # (N, P)
        ctx    = (x * attn_w.unsqueeze(-1)).sum(dim=1)       # (N, H)
        return self.head(ctx), attn_w


def train_attn_head(X_raw: np.ndarray, y: np.ndarray,
                    n_epochs: int = 300, lr: float = 1e-3) -> AttnBottleneck | None:
    """K개 아이템으로 AttnBottleneck 학습. 불안정하면 None 반환."""
    n_items, n_personas, hidden = X_raw.shape
    n_weeks = y.shape[1]

    model = AttnBottleneck(hidden=hidden, n_weeks=n_weeks)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.from_numpy(X_raw)     # (K, P, H)
    y_t = torch.from_numpy(y)         # (K, n_weeks)

    model.train()
    for epoch in range(n_epochs):
        opt.zero_grad()
        pred, _ = model(X_t)
        loss = torch.nn.functional.l1_loss(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if torch.isnan(loss):
            return None

    return model


def verify_item_specific_head(
    warm_raw:  np.ndarray,    # (300, 50, 5120)
    resid_pca: np.ndarray,    # (300, 64)
    y_warm:    np.ndarray,    # (300, 17)
) -> dict:
    """LOO item-specific head + rescale 방식."""
    logger.info("=== 검증 3: Item-specific Head + Rescale ===")

    n_items = len(warm_raw)
    item_sales_mean = y_warm.mean(axis=1)     # (300,) 아이템별 평균 판매량
    global_sales_mean = float(item_sales_mean.mean())

    dist_mat = cosine_dist_matrix(resid_pca)
    np.fill_diagonal(dist_mat, np.inf)

    # K별 결과 저장
    Ks = [10, 20, 50]
    results = {}

    # 글로벌 head 학습 (전체 300개)
    logger.info("  글로벌 head 학습 중 (n=300)...")
    global_model = train_attn_head(warm_raw, y_warm, n_epochs=500)
    if global_model is None:
        logger.warning("  글로벌 head 학습 불안정 — rescale only")
    else:
        with torch.no_grad():
            global_pred, _ = global_model(torch.from_numpy(warm_raw))
        global_pred_np = global_pred.numpy()

    for K in Ks:
        logger.info("  K=%d 처리 중...", K)
        preds_specific = np.zeros_like(y_warm)
        preds_rescale  = np.zeros_like(y_warm)
        unstable_count = 0

        for i in range(n_items):
            knn_idx = np.argsort(dist_mat[i])[:K]
            X_knn   = warm_raw[knn_idx]          # (K, 50, 5120)
            y_knn   = y_warm[knn_idx]             # (K, 17)
            knn_mean_sales = float(item_sales_mean[knn_idx].mean())

            # (1) item-specific head
            model = train_attn_head(X_knn, y_knn, n_epochs=300)
            if model is None:
                unstable_count += 1
                # fallback: K-NN weighted mean
                dists = dist_mat[i][knn_idx] + 1e-12
                w = (1.0 / dists); w /= w.sum()
                preds_specific[i] = (y_knn * w[:, None]).sum(axis=0)
            else:
                with torch.no_grad():
                    xi = torch.from_numpy(warm_raw[i:i+1])   # (1, 50, 5120)
                    pred_i, _ = model(xi)
                preds_specific[i] = pred_i.numpy()[0]

            # (2) rescale: global_pred * (knn_mean_sales / global_sales_mean)
            if global_model is not None:
                scale = knn_mean_sales / (global_sales_mean + 1e-6)
                preds_rescale[i] = global_pred_np[i] * scale
            else:
                preds_rescale[i] = preds_specific[i]

        mae_specific = float(np.abs(y_warm - preds_specific).mean())
        mae_rescale  = float(np.abs(y_warm - preds_rescale).mean())
        results[K] = {
            "mae_specific": mae_specific,
            "mae_rescale":  mae_rescale,
            "unstable_count": unstable_count,
        }
        logger.info("    item-specific MAE=%.2f  rescale MAE=%.2f  unstable=%d/%d",
                    mae_specific, mae_rescale, unstable_count, n_items)

    # 시각화
    Ks_list = sorted(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(Ks_list, [results[k]["mae_specific"] for k in Ks_list],
            marker="o", label="item-specific head", color="steelblue")
    ax.plot(Ks_list, [results[k]["mae_rescale"] for k in Ks_list],
            marker="s", label="global+rescale", color="orangered")
    ax.axhline(55.8, color="red",   linestyle="--", label="Global Attn+BN resid 55.8")
    ax.axhline(62.5, color="black", linestyle="--", label="Global Attn+BN orig 62.5")
    ax.set_xlabel("K"); ax.set_ylabel("LOO MAE")
    ax.set_title("V3: Item-specific Head vs Rescale")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 불안정 카운트
    ax2 = axes[1]
    ax2.bar(Ks_list, [results[k]["unstable_count"] for k in Ks_list],
            color="salmon")
    ax2.set_xlabel("K"); ax2.set_ylabel("Unstable heads count")
    ax2.set_title("V3: Unstable Head Count (fallback to K-NN)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3_item_specific_head.png", dpi=150)
    plt.close(fig)

    # 글로벌 head MAE (참고)
    if global_model is not None:
        global_mae = float(np.abs(y_warm - global_pred_np).mean())
        logger.info("  글로벌 head train MAE (not CV): %.2f", global_mae)
        results["global_head_train_mae"] = global_mae

    return results


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(r1: dict, r2: dict, r3: dict) -> None:
    GLOBAL_MEAN_MAE   = 96.23
    GLOBAL_ATTN_ORIG  = 62.5
    GLOBAL_ATTN_RESID = 55.8

    lines: list[str] = []
    lines += [
        "# 유사 아이템 기반 파이프라인 핵심 가정 검증 (Exp015)",
        "",
        "**작성일:** 2026-03-10",
        "**기반 데이터:** exp011 warm_raw.pt (300×50×5120)",
        "",
        "**참조 기준값:**",
        f"- Global mean MAE: {GLOBAL_MEAN_MAE}",
        f"- Global Attn+Bottleneck (원본): {GLOBAL_ATTN_ORIG}",
        f"- Global Attn+Bottleneck (residual, N=250): {GLOBAL_ATTN_RESID}",
        "",
        "---",
        "",
        "## 검증 1: Embedding 유사성 ↔ 판매량 유사성 상관",
        "",
        f"**전체 판매량 std:** {r1['global_std']:.2f}",
        "",
        "### Mantel 검정 (embedding 거리 vs 판매량 차이 Spearman 상관)",
        "",
        "| 표현 방식 | Spearman r | p-value |",
        "|----------|-----------|---------|",
    ]
    for row in r1["mantel_table"]:
        lines.append(f"| {row['repr']} | {row['spearman_r']:.4f} | {row['p']:.4f} |")
    lines += [""]

    lines += ["### K-NN 판매량 std (전체 std 대비 낮을수록 좋음)", ""]
    headers = ["표현 방식", "K=5", "K=10", "K=20"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for name, v in r1["representations"].items():
        row = f"| {name} | " + " | ".join(
            f"{v['knn_stds'][k]:.2f}" for k in [5, 10, 20]
        ) + " |"
        lines.append(row)

    # 판정
    best_r = max(r1["mantel_table"], key=lambda x: abs(x["spearman_r"]))
    if abs(best_r["spearman_r"]) > 0.1 and best_r["p"] < 0.05:
        v1_verdict = "부분적 지지"
        v1_note = f"{best_r['repr']} 표현이 판매량 유사성과 유의한 상관(r={best_r['spearman_r']:.3f})."
    else:
        v1_verdict = "실패"
        v1_note = "embedding 거리와 판매량 유사성 간 유의한 상관 없음."
    lines += ["", f"### 판정: **{v1_verdict}**", f"> {v1_note}", "", "---", ""]

    # V2
    lines += [
        "## 검증 2: K-NN LOO 예측 MAE",
        "",
        "| K | mean_pooled | flat_pca64 | residual_pca64 | cat_price |",
        "|---|------------|-----------|----------------|-----------|",
    ]
    rows_df = pd.DataFrame(r2["rows"])
    for K in [5, 10, 20, 50]:
        sub = rows_df[rows_df["K"] == K]
        def get(s): return f"{sub[sub['sim']==s]['mae'].values[0]:.2f}" if len(sub[sub['sim']==s]) else "—"
        lines.append(f"| {K} | {get('mean_pooled')} | {get('flat_pca64')} | {get('residual_pca64')} | {get('cat_price')} |")

    best_row = rows_df.loc[rows_df["mae"].idxmin()]
    lines += [
        "",
        f"- **최저 MAE:** {best_row['mae']:.2f} (sim={best_row['sim']}, K={best_row['K']})",
        f"- Global Attn+BN residual: {GLOBAL_ATTN_RESID}",
        f"- Global mean: {GLOBAL_MEAN_MAE}",
        "",
    ]
    if best_row["mae"] < GLOBAL_ATTN_RESID:
        v2_verdict = "지지"
        v2_note = f"K-NN 최저 MAE({best_row['mae']:.2f}) < Global Attn+BN resid({GLOBAL_ATTN_RESID}) → 유사 아이템 접근 유망."
    elif best_row["mae"] < GLOBAL_ATTN_ORIG:
        v2_verdict = "부분적 지지"
        v2_note = f"K-NN이 원본 Global Attn+BN보다 나으나 residual 버전은 못 이김."
    else:
        v2_verdict = "실패"
        v2_note = "K-NN이 글로벌 베이스라인을 넘지 못함."
    lines += [f"### 판정: **{v2_verdict}**", f"> {v2_note}", "", "---", ""]

    # V3
    lines += [
        "## 검증 3: Item-specific Head + Rescale",
        "",
        "| K | item-specific MAE | rescale MAE | unstable count |",
        "|---|-------------------|------------|----------------|",
    ]
    for K, v in sorted((k, v) for k, v in r3.items() if isinstance(k, int)):
        if not isinstance(v, dict):
            continue
        lines.append(f"| {K} | {v['mae_specific']:.2f} | {v['mae_rescale']:.2f} | {v['unstable_count']}/300 |")

    if "global_head_train_mae" in r3:
        lines += ["", f"- 글로벌 head train MAE (참고, not CV): {r3['global_head_train_mae']:.2f}"]

    # 판정
    all_spec = [v["mae_specific"] for k, v in r3.items() if isinstance(k, int)]
    all_resc = [v["mae_rescale"]  for k, v in r3.items() if isinstance(k, int)]
    best_spec = min(all_spec) if all_spec else 9999
    best_resc = min(all_resc) if all_resc else 9999
    best_v3   = min(best_spec, best_resc)
    lines += [""]
    if best_v3 < GLOBAL_ATTN_RESID:
        v3_verdict = "지지"
        v3_note = f"item-specific 접근 MAE({best_v3:.2f}) < Global Attn+BN resid({GLOBAL_ATTN_RESID})."
    elif best_v3 < GLOBAL_ATTN_ORIG:
        v3_verdict = "부분적 지지"
        v3_note = f"원본 global보다 나으나 residual 버전보다 못함."
    else:
        v3_verdict = "실패"
        v3_note = "item-specific head가 글로벌 베이스라인을 이기지 못함."
    lines += [f"### 판정: **{v3_verdict}**", f"> {v3_note}", "", "---", ""]

    # 종합
    all_v = [v1_verdict, v2_verdict, v3_verdict]
    n_fail = sum(1 for v in all_v if v == "실패")
    lines += [
        "## 종합 판정",
        "",
        "| 검증 | 내용 | 판정 |",
        "|------|------|------|",
        f"| 검증 1 | Embedding 유사성이 판매량 유사성을 반영하는가? | **{v1_verdict}** |",
        f"| 검증 2 | K-NN LOO 예측이 글로벌 head보다 나은가? | **{v2_verdict}** |",
        f"| 검증 3 | Item-specific head/rescale이 글로벌보다 나은가? | **{v3_verdict}** |",
        "",
        "### 결론: 유사 아이템 기반 접근 cold items에 적용 가치가 있는가?",
        "",
    ]
    if n_fail == 0:
        lines.append("**YES** — 세 검증 모두 지지. 유사 아이템 기반 cold-start 예측 파이프라인 구현 권장.")
    elif n_fail == 1:
        lines.append("**조건부** — 일부 검증 실패. 가장 효과적인 유사성 기준과 K값을 고정하여 cold-start에 적용 가능.")
    elif n_fail == 2:
        lines.append("**신중** — 대부분 검증 실패. 카테고리+가격 기준이 더 나을 수 있으므로 embedding 유사성 의존 최소화 권장.")
    else:
        lines.append("**NO** — 모든 검증 실패. 유사 아이템 기반 접근보다 글로벌 Attn+BN residual(55.8)을 유지하는 것이 낫다.")

    lines += [
        "",
        "---",
        "",
        "**시각화:** `experiments/exp015_similar_item/figures/`",
        "**스크립트:** `scripts/exp015_similar_item.py`",
    ]

    path = REPORT_DIR / "similar_item_verification.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", path)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Exp015: 유사 아이템 기반 가정 검증 시작 ===")

    warm_raw, y_warm, warm_meta, price_map = load_data()
    reprs = compute_representations(warm_raw)

    r1 = verify_similarity_sales_corr(reprs, y_warm, warm_meta, price_map)
    r2 = verify_knn_prediction(reprs, y_warm, warm_meta, price_map)
    r3 = verify_item_specific_head(warm_raw, reprs.resid_pca, y_warm)

    write_report(r1, r2, r3)

    print("\n" + "=" * 60)
    print("=== Exp015 결과 요약 ===")
    print(f"V1 최고 Mantel r: {max(abs(x['spearman_r']) for x in r1['mantel_table']):.4f}")
    rows_df = pd.DataFrame(r2["rows"])
    best = rows_df.loc[rows_df["mae"].idxmin()]
    print(f"V2 최저 K-NN LOO MAE: {best['mae']:.2f}  (sim={best['sim']}, K={best['K']})")
    all_v3 = [v for k, v in r3.items() if isinstance(v, dict)]
    if all_v3:
        print(f"V3 최저 MAE: spec={min(v['mae_specific'] for v in all_v3):.2f}  "
              f"rescale={min(v['mae_rescale'] for v in all_v3):.2f}")
    print(f"보고서: docs/diagnosis/similar_item_verification.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
