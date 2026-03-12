"""Exp017: G1 최선 원인 규명 + Attention 심층 분석.

실험 H: G1 안정성 검증 (10 seeds)
실험 I: Rescale 악화 원인 분석 (factor 분포, K-NN 품질, G1 스케일)
실험 J: Attention Score 심층 분석 (카테고리별 패턴, 페르소나 속성 상관)
실험 K: 유사 아이템 학습 재검토 (warm Q1 분석, cold K-range)

출력:
    docs/diagnosis/g1_analysis_report.md
    experiments/exp017_g1_analysis/figures/
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import KFold

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_weekly

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EMB_DIR     = ROOT / "experiments/exp011_v3_pipeline/embeddings"
FIG_DIR     = ROOT / "experiments/exp017_g1_analysis/figures"
REPORT_DIR  = ROOT / "docs/diagnosis"
CS_DIR      = ROOT / "data/processed/cold_start"
M5_DIR      = ROOT / "m5-forecasting-accuracy"
PERSONA_DIR = ROOT / "data/processed/personas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# exp016 결과 상수
G3_COLD_MAE = 8.9784
G5_COLD_MAE = 8.9286
TRAIN_EPOCHS = 500
K_ABLATION   = 50


# ─── exp016에서 복사한 공통 함수 ──────────────────────────────────────────────

def _to_weekly_iso(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"]  = df["date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    group_cols = [c for c in ["item_id", "store_id", "cat_id", "dept_id", "state_id",
                               "iso_year", "iso_week"] if c in df.columns]
    return (
        df.groupby(group_cols)
        .agg(**{"sales": ("sales", "sum"), "date": ("week_start", "first")})
        .reset_index()
    )


def _complete_weeks(df_daily: pd.DataFrame) -> set:
    df_daily = df_daily.copy()
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily["iso_year"] = df_daily["date"].dt.isocalendar().year.astype(int)
    df_daily["iso_week"]  = df_daily["date"].dt.isocalendar().week.astype(int)
    days = (df_daily.groupby(["iso_year", "iso_week"])["date"]
            .nunique().reset_index(name="n_days"))
    return set(zip(days[days["n_days"] == 7]["iso_year"],
                   days[days["n_days"] == 7]["iso_week"]))


class AttnBottleneck(torch.nn.Module):
    def __init__(self, hidden=5120, bottleneck=64, n_weeks=16, dropout=0.1):
        super().__init__()
        self.attn = torch.nn.Linear(hidden, 1, bias=False)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, bottleneck), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        attn_w = torch.softmax(scores, dim=-1)
        ctx    = (x * attn_w.unsqueeze(-1)).sum(dim=1)
        return self.head(ctx)

    def forward_with_attn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(x).squeeze(-1)
        attn_w = torch.softmax(scores, dim=-1)
        ctx    = (x * attn_w.unsqueeze(-1)).sum(dim=1)
        return self.head(ctx), attn_w


def train_head(X_raw: np.ndarray, y: np.ndarray, n_epochs=TRAIN_EPOCHS,
               lr=1e-3, seed: int = 42) -> AttnBottleneck:
    torch.manual_seed(seed)
    n_weeks = y.shape[1]
    model = AttnBottleneck(n_weeks=n_weeks)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X_t   = torch.from_numpy(X_raw.astype(np.float32))
    y_t   = torch.from_numpy(y.astype(np.float32))
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(model(X_t), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def predict_head(model: AttnBottleneck, X_raw: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_raw.astype(np.float32))).numpy()


def get_attn_weights(model: AttnBottleneck, X_raw: np.ndarray) -> np.ndarray:
    """attention weights (N, P) 추출."""
    model.eval()
    with torch.no_grad():
        _, attn_w = model.forward_with_attn(torch.from_numpy(X_raw.astype(np.float32)))
    return attn_w.numpy()


def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.clip(1.0 - An @ Bn.T, 0.0, 2.0)


def preds_to_df(preds, item_ids, cat_ids, week_list, week_dates):
    records = []
    for i, item_id in enumerate(item_ids):
        for t, wk in enumerate(week_list):
            date = week_dates.get(wk)
            if date is None:
                continue
            records.append({"item_id": item_id, "store_id": "CA_1",
                            "cat_id": cat_ids[i], "date": date,
                            "pred_sales": max(0.0, float(preds[i, t]))})
    return pd.DataFrame(records)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data() -> dict:
    logger.info("데이터 로드 중...")
    warm_raw  = torch.load(EMB_DIR / "warm_raw.pt",  weights_only=True).numpy()
    cold_raw  = torch.load(EMB_DIR / "cold_raw.pt",  weights_only=True).numpy()
    warm_mean = torch.load(EMB_DIR / "warm_mean.pt", weights_only=True).numpy()
    cold_mean = torch.load(EMB_DIR / "cold_mean.pt", weights_only=True).numpy()

    meta      = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_meta = meta[meta["is_cold"] == True].reset_index(drop=True)
    warm_meta = meta[meta["is_cold"] == False].reset_index(drop=True)

    cold_test_raw  = pd.read_csv(CS_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_test_raw  = pd.read_csv(CS_DIR / "warm_test.csv",  parse_dates=["date"])
    warm_train_raw = pd.read_csv(CS_DIR / "warm_train.csv", parse_dates=["date"])
    sell_prices    = pd.read_csv(M5_DIR / "sell_prices.csv")

    complete_set = _complete_weeks(cold_test_raw)
    week_list    = sorted(complete_set)

    cold_test_weekly = _to_weekly_iso(cold_test_raw)
    cold_test_weekly = cold_test_weekly[
        cold_test_weekly.apply(lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)]
    warm_train_weekly = _to_weekly_iso(warm_train_raw)

    week_idx = {wk: i for i, wk in enumerate(week_list)}
    week_dates = (cold_test_weekly[["iso_year", "iso_week", "date"]].drop_duplicates()
                  .assign(wk=lambda r: list(zip(r["iso_year"], r["iso_week"])))
                  .set_index("wk")["date"].to_dict())

    cold_ids = cold_meta["item_id"].tolist()
    warm_ids = warm_meta["item_id"].tolist()

    def build_y(df_raw, ids):
        wkly = _to_weekly_iso(df_raw)
        wkly = wkly[wkly.apply(lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)]
        y = np.zeros((len(ids), len(week_list)), dtype=np.float32)
        for _, row in wkly.iterrows():
            ci = ids.index(row["item_id"]) if row["item_id"] in ids else -1
            wi = week_idx.get((row["iso_year"], row["iso_week"]), -1)
            if ci >= 0 and wi >= 0:
                y[ci, wi] = row["sales"]
        return y

    y_cold = build_y(cold_test_raw, cold_ids)
    y_warm = build_y(warm_test_raw, warm_ids)

    price_map = (sell_prices[sell_prices["store_id"] == "CA_1"]
                 .groupby("item_id")["sell_price"].mean().to_dict())

    # residual
    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)

    logger.info("  y_cold=%s mean=%.2f  y_warm=%s mean=%.2f",
                y_cold.shape, y_cold.mean(), y_warm.shape, y_warm.mean())

    return dict(
        warm_raw=warm_raw, cold_raw=cold_raw,
        warm_mean=warm_mean, cold_mean=cold_mean,
        warm_residual=warm_residual, cold_residual=cold_residual,
        cold_meta=cold_meta, warm_meta=warm_meta,
        cold_test_weekly=cold_test_weekly,
        warm_train_weekly=warm_train_weekly,
        y_cold=y_cold, y_warm=y_warm,
        week_list=week_list, week_dates=week_dates,
        price_map=price_map,
    )


def load_personas() -> list[dict]:
    files = sorted(PERSONA_DIR.glob("CA_1_P*.json"))[:50]
    return [json.loads(fp.read_text())["profile"] for fp in files]


# ─── 실험 H: G1 안정성 검증 ──────────────────────────────────────────────────

def experiment_h(data: dict) -> dict:
    """10 seeds로 G1 cold MAE 분포 확인."""
    logger.info("=== 실험 H: G1 안정성 검증 (10 seeds) ===")
    seeds = list(range(10))
    maes, daccs = [], []

    for seed in seeds:
        model = train_head(data["warm_residual"], data["y_warm"], seed=seed)
        preds = predict_head(model, data["cold_residual"])
        pred_df = preds_to_df(preds, data["cold_meta"]["item_id"].tolist(),
                              data["cold_meta"]["cat_id"].tolist(),
                              data["week_list"], data["week_dates"])
        ev = evaluate_weekly(data["cold_test_weekly"], pred_df,
                             data["warm_train_weekly"], model_name=f"G1_seed{seed}")
        maes.append(ev["mae"]); daccs.append(ev["direction_accuracy"])
        logger.info("  seed=%d → MAE=%.4f  DirAcc=%.3f", seed, ev["mae"], ev["direction_accuracy"])

    maes_arr = np.array(maes)
    mean_mae = float(maes_arr.mean()); std_mae = float(maes_arr.std())

    # G3=8.9784, G5=8.9286 vs G1 분포: one-sample t-test
    t_g3, p_g3 = stats.ttest_1samp(maes_arr, G3_COLD_MAE)
    t_g5, p_g5 = stats.ttest_1samp(maes_arr, G5_COLD_MAE)
    logger.info("  G1: mean=%.4f ± %.4f", mean_mae, std_mae)
    logger.info("  vs G3(%.4f): t=%.3f p=%.4f", G3_COLD_MAE, t_g3, p_g3)
    logger.info("  vs G5(%.4f): t=%.3f p=%.4f", G5_COLD_MAE, t_g5, p_g5)

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(maes_arr, bins=10, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean_mae, color="blue", linestyle="--", label=f"G1 mean={mean_mae:.3f}")
    ax.axvline(G3_COLD_MAE, color="orange", linestyle="--", label=f"G3={G3_COLD_MAE:.3f}")
    ax.axvline(G5_COLD_MAE, color="green",  linestyle="--", label=f"G5={G5_COLD_MAE:.3f}")
    ax.set_xlabel("Cold MAE"); ax.set_title("(H) G1 MAE 분포 (10 seeds)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "h_g1_stability.png", dpi=150); plt.close(fig)

    return {"maes": maes, "daccs": daccs, "mean": mean_mae, "std": std_mae,
            "t_vs_g3": t_g3, "p_vs_g3": p_g3, "t_vs_g5": t_g5, "p_vs_g5": p_g5}


# ─── 실험 I: Rescale 악화 원인 분석 ──────────────────────────────────────────

def experiment_i(data: dict, g1_model: AttnBottleneck) -> dict:
    """I-1: Rescale factor 분포, I-2: K-NN 품질, I-3: G1 예측 스케일."""
    logger.info("=== 실험 I: Rescale 악화 원인 분석 ===")

    cold_residual = data["cold_residual"]
    cold_mean     = data["cold_mean"]
    warm_mean     = data["warm_mean"]
    y_warm        = data["y_warm"]
    y_cold        = data["y_cold"]
    cold_meta     = data["cold_meta"]
    warm_meta     = data["warm_meta"]
    price_map     = data["price_map"]

    item_sales_mean_warm = y_warm.mean(axis=1)
    global_warm_mean     = float(item_sales_mean_warm.mean())
    item_sales_mean_cold = y_cold.mean(axis=1)

    dist_cw = cosine_dist(cold_mean, warm_mean)  # (100, 300)

    # ── I-1: Rescale factor 분포 ─────────────────────────────────────────────
    factors = []
    for i in range(len(cold_meta)):
        knn_idx = np.argsort(dist_cw[i])[:K_ABLATION]
        knn_mean = float(item_sales_mean_warm[knn_idx].mean())
        factors.append(knn_mean / (global_warm_mean + 1e-6))
    factors = np.array(factors)

    g1_preds = predict_head(g1_model, cold_residual)   # (100, 16)
    g1_pred_mean = g1_preds.mean(axis=1)               # (100,)

    logger.info("  I-1 factor: mean=%.3f std=%.3f min=%.3f max=%.3f",
                factors.mean(), factors.std(), factors.min(), factors.max())

    # factor 극단값 아이템 MAE 비교
    low_mask  = factors < np.percentile(factors, 33)
    high_mask = factors > np.percentile(factors, 67)
    g2_preds = g1_preds * factors[:, None]
    mae_g1_low  = float(np.abs(y_cold[low_mask]  - g1_preds[low_mask]).mean())
    mae_g1_high = float(np.abs(y_cold[high_mask] - g1_preds[high_mask]).mean())
    mae_g2_low  = float(np.abs(y_cold[low_mask]  - g2_preds[low_mask]).mean())
    mae_g2_high = float(np.abs(y_cold[high_mask] - g2_preds[high_mask]).mean())
    logger.info("  I-1 low factor: G1 MAE=%.2f  G2 MAE=%.2f", mae_g1_low, mae_g2_low)
    logger.info("  I-1 high factor: G1 MAE=%.2f  G2 MAE=%.2f", mae_g1_high, mae_g2_high)

    # ── I-2: K-NN 품질 (category/dept match rate) ────────────────────────────
    cold_cats  = cold_meta["cat_id"].values
    cold_depts = cold_meta["dept_id"].values
    warm_cats  = warm_meta["cat_id"].values
    warm_depts = warm_meta["dept_id"].values
    cold_prices = np.array([price_map.get(iid, np.nan) for iid in cold_meta["item_id"]])
    warm_prices = np.array([price_map.get(iid, np.nan) for iid in warm_meta["item_id"]])

    cat_match_rates, dept_match_rates, sales_ratios = [], [], []
    for i in range(len(cold_meta)):
        knn_idx = np.argsort(dist_cw[i])[:K_ABLATION]
        cat_match  = float((warm_cats[knn_idx] == cold_cats[i]).mean())
        dept_match = float((warm_depts[knn_idx] == cold_depts[i]).mean())
        knn_sales_mean = float(item_sales_mean_warm[knn_idx].mean())
        cold_sales     = float(item_sales_mean_cold[i])
        ratio = knn_sales_mean / (cold_sales + 1e-6) if cold_sales > 0 else np.nan
        cat_match_rates.append(cat_match)
        dept_match_rates.append(dept_match)
        sales_ratios.append(ratio)

    sales_ratios_arr = np.array([r for r in sales_ratios if not np.isnan(r)])
    logger.info("  I-2 cat_match_rate: mean=%.3f  dept_match: mean=%.3f",
                np.mean(cat_match_rates), np.mean(dept_match_rates))
    logger.info("  I-2 K-NN/cold sales ratio: mean=%.2f std=%.2f",
                sales_ratios_arr.mean(), sales_ratios_arr.std())

    # 10개 아이템 상세 (고판매 3, 중판매 4, 저판매 3)
    cold_sorted = np.argsort(item_sales_mean_cold)
    sample_idx  = np.concatenate([cold_sorted[:3], cold_sorted[len(cold_sorted)//4:len(cold_sorted)//4+4], cold_sorted[-3:]])
    detail_rows = []
    for i in sample_idx:
        knn_idx = np.argsort(dist_cw[i])[:5]
        detail_rows.append({
            "cold_item": cold_meta["item_id"].iloc[i],
            "cold_cat":  cold_cats[i],
            "cold_dept": cold_depts[i],
            "cold_price": round(float(cold_prices[i]), 2) if not np.isnan(cold_prices[i]) else None,
            "cold_sales_mean": round(float(item_sales_mean_cold[i]), 2),
            "top5_items": warm_meta["item_id"].iloc[knn_idx].tolist(),
            "top5_cats":  warm_cats[knn_idx].tolist(),
            "top5_sales": [round(float(item_sales_mean_warm[j]), 2) for j in knn_idx],
            "top5_prices": [round(float(warm_prices[j]), 2) if not np.isnan(warm_prices[j]) else None for j in knn_idx],
        })
    logger.info("  I-2 상세 10개 아이템 완료")

    # ── I-3: G1 예측 스케일 ──────────────────────────────────────────────────
    pred_mean_all = float(g1_pred_mean.mean())
    pred_std_all  = float(g1_pred_mean.std())
    cold_mean_all = float(item_sales_mean_cold.mean())
    cold_std_all  = float(item_sales_mean_cold.std())
    pearson_r, _ = stats.pearsonr(item_sales_mean_cold, g1_pred_mean)
    logger.info("  I-3 G1 pred mean=%.2f std=%.2f  actual mean=%.2f std=%.2f  r=%.3f",
                pred_mean_all, pred_std_all, cold_mean_all, cold_std_all, pearson_r)

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # I-1: factor histogram
    axes[0].hist(factors, bins=20, color="tomato", edgecolor="white", alpha=0.85)
    axes[0].axvline(1.0, color="black", linestyle="--", label="factor=1 (no change)")
    axes[0].axvline(factors.mean(), color="red", linestyle="--",
                    label=f"mean={factors.mean():.3f}")
    axes[0].set_title("I-1: Rescale Factor 분포 (K=50)"); axes[0].set_xlabel("factor")
    axes[0].legend(fontsize=8)

    # I-2: category match rate distribution
    axes[1].hist(cat_match_rates, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(cat_match_rates), color="red", linestyle="--",
                    label=f"mean={np.mean(cat_match_rates):.2f}")
    axes[1].set_title("I-2: K=50 이웃의 Category Match Rate"); axes[1].set_xlabel("rate")
    axes[1].legend(fontsize=8)

    # I-3: scatter actual vs G1 pred
    axes[2].scatter(item_sales_mean_cold, g1_pred_mean, alpha=0.6, s=25, color="steelblue")
    lim = max(item_sales_mean_cold.max(), g1_pred_mean.max()) * 1.05
    axes[2].plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
    axes[2].axhline(pred_mean_all, color="orange", linestyle="--",
                    label=f"pred mean={pred_mean_all:.1f}")
    axes[2].axvline(cold_mean_all, color="green", linestyle="--",
                    label=f"actual mean={cold_mean_all:.1f}")
    axes[2].set_xlabel("실제 cold 주간 평균 판매량"); axes[2].set_ylabel("G1 예측 주간 평균")
    axes[2].set_title(f"I-3: G1 vs Actual  r={pearson_r:.3f}"); axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "i_rescale_analysis.png", dpi=150); plt.close(fig)

    return {
        "factors": factors.tolist(), "factor_mean": float(factors.mean()),
        "factor_std": float(factors.std()), "factor_min": float(factors.min()),
        "factor_max": float(factors.max()),
        "mae_g1_low": mae_g1_low, "mae_g2_low": mae_g2_low,
        "mae_g1_high": mae_g1_high, "mae_g2_high": mae_g2_high,
        "cat_match_rate_mean": float(np.mean(cat_match_rates)),
        "dept_match_rate_mean": float(np.mean(dept_match_rates)),
        "sales_ratio_mean": float(sales_ratios_arr.mean()),
        "sales_ratio_std": float(sales_ratios_arr.std()),
        "pred_mean": pred_mean_all, "pred_std": pred_std_all,
        "actual_mean": cold_mean_all, "actual_std": cold_std_all,
        "pearson_r": pearson_r, "detail_rows": detail_rows,
    }


# ─── 실험 J: Attention 심층 분석 ──────────────────────────────────────────────

def experiment_j(data: dict, g1_model: AttnBottleneck,
                 personas: list[dict]) -> dict:
    """J-1~J-4: attention weight 분석."""
    logger.info("=== 실험 J: Attention Score 심층 분석 ===")

    cold_residual = data["cold_residual"]   # (100, 50, 5120)
    cold_meta     = data["cold_meta"]
    y_cold        = data["y_cold"]

    # J-1: cold attention weights
    attn_w = get_attn_weights(g1_model, cold_residual)   # (100, 50)

    # J-2: 카테고리별 attention profile ──────────────────────────────────────
    cats = cold_meta["cat_id"].values
    cat_profiles = {}
    for cat in sorted(np.unique(cats)):
        mask = (cats == cat)
        cat_profiles[cat] = attn_w[mask].mean(axis=0)   # (50,)

    # top-5 per category
    cat_top5 = {}
    for cat, profile in cat_profiles.items():
        top5_idx = np.argsort(profile)[::-1][:5]
        cat_top5[cat] = [(int(pi), float(profile[pi]),
                          personas[pi].get("weekly_budget", 0),
                          personas[pi].get("economic_status", ""),
                          personas[pi].get("category_preference", {}).get("FOODS", 0),
                          personas[pi].get("category_preference", {}).get(cat, 0))
                         for pi in top5_idx]
        logger.info("  [%s] top-5: %s", cat, [(r[0], round(r[1], 4)) for r in cat_top5[cat]])

    top1_per_cat = {cat: cat_top5[cat][0][0] for cat in cat_top5}
    unique_top1 = len(set(top1_per_cat.values())) == len(top1_per_cat)
    logger.info("  카테고리별 top-1 다름: %s (%s)", unique_top1, top1_per_cat)

    # J-3: entropy vs MAE ─────────────────────────────────────────────────────
    UNIFORM_H = float(np.log(50))
    item_entropy = np.array([-np.sum(np.clip(w, 1e-12, None) * np.log(np.clip(w, 1e-12, None)))
                             for w in attn_w])
    g1_preds = predict_head(g1_model, cold_residual)
    item_mae  = np.abs(y_cold - g1_preds).mean(axis=1)

    # 상/하 50분위 비교
    med_ent = np.median(item_entropy)
    sparse_mask = item_entropy <= med_ent
    diffuse_mask = ~sparse_mask
    mae_sparse  = float(item_mae[sparse_mask].mean())
    mae_diffuse = float(item_mae[diffuse_mask].mean())
    r_ent, p_ent = stats.spearmanr(item_entropy, item_mae)
    logger.info("  J-3 entropy: mean=%.3f (uniform=%.3f)", item_entropy.mean(), UNIFORM_H)
    logger.info("  J-3 sparse MAE=%.2f  diffuse MAE=%.2f  spearman r=%.3f p=%.4f",
                mae_sparse, mae_diffuse, r_ent, p_ent)

    # J-4: 페르소나 속성과 attention weight 상관 ─────────────────────────────
    mean_attn_per_persona = attn_w.mean(axis=0)   # (50,)
    budgets  = np.array([p["weekly_budget"] for p in personas])
    foods_pref = np.array([p["category_preference"].get("FOODS", 0) for p in personas])
    snap     = np.array([float(p["snap_eligible"]) for p in personas])

    r_budget, p_budget = stats.spearmanr(mean_attn_per_persona, budgets)
    r_foods,  p_foods  = stats.spearmanr(mean_attn_per_persona, foods_pref)
    r_snap,   p_snap   = stats.spearmanr(mean_attn_per_persona, snap)
    logger.info("  J-4 attn vs budget: r=%.3f p=%.4f", r_budget, p_budget)
    logger.info("  J-4 attn vs FOODS pref: r=%.3f p=%.4f", r_foods, p_foods)
    logger.info("  J-4 attn vs snap: r=%.3f p=%.4f", r_snap, p_snap)

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # J-2: 카테고리별 attention profile (bar)
    ax = axes[0, 0]
    x = np.arange(50)
    cat_colors_map = {"FOODS": "tomato", "HOUSEHOLD": "steelblue", "HOBBIES": "seagreen"}
    for cat, profile in cat_profiles.items():
        ax.plot(x, profile, label=cat, color=cat_colors_map.get(cat, "gray"), alpha=0.85)
    ax.axhline(1/50, color="black", linestyle="--", alpha=0.4, label="uniform")
    ax.set_title("J-2: 카테고리별 mean attention weight (cold)")
    ax.set_xlabel("Persona index"); ax.set_ylabel("Attention weight")
    ax.legend(fontsize=8)

    # J-2 heatmap: (3 categories × 50 personas)
    ax2 = axes[0, 1]
    profile_mat = np.array([cat_profiles[c] for c in sorted(cat_profiles.keys())])
    im = ax2.imshow(profile_mat, aspect="auto", cmap="hot")
    ax2.set_yticks(range(len(cat_profiles)))
    ax2.set_yticklabels(sorted(cat_profiles.keys()))
    ax2.set_xlabel("Persona index"); ax2.set_title("J-2: Attention Heatmap")
    plt.colorbar(im, ax=ax2)

    # J-3: entropy vs item MAE scatter
    axes[1, 0].scatter(item_entropy, item_mae, alpha=0.6, s=20, color="steelblue")
    axes[1, 0].axvline(UNIFORM_H, color="green", linestyle="--",
                       label=f"uniform={UNIFORM_H:.2f}", alpha=0.7)
    axes[1, 0].set_xlabel("Attention entropy"); axes[1, 0].set_ylabel("Item MAE")
    axes[1, 0].set_title(f"J-3: Entropy vs MAE  spearman r={r_ent:.3f}")
    axes[1, 0].legend(fontsize=8)

    # J-4: 페르소나 속성 bar chart
    ax4 = axes[1, 1]
    sorted_idx = np.argsort(mean_attn_per_persona)[::-1]
    top20_idx  = sorted_idx[:20]
    ax4.bar(range(20), mean_attn_per_persona[top20_idx], color="darkorange", alpha=0.85)
    ax4.set_xticks(range(20))
    ax4.set_xticklabels([f"P{i+1:02d}" for i in top20_idx], rotation=45, ha="right", fontsize=7)
    ax4.set_ylabel("Mean attention weight")
    ax4.set_title("J-4: Top-20 personas by attention weight (cold)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "j_attention_analysis.png", dpi=150); plt.close(fig)

    return {
        "attn_weights": attn_w.tolist(),
        "cat_top5": cat_top5,
        "unique_top1_per_cat": unique_top1,
        "top1_per_cat": top1_per_cat,
        "entropy_mean": float(item_entropy.mean()),
        "uniform_entropy": UNIFORM_H,
        "mae_sparse": mae_sparse, "mae_diffuse": mae_diffuse,
        "spearman_entropy_mae": float(r_ent), "p_entropy_mae": float(p_ent),
        "r_budget": r_budget, "p_budget": p_budget,
        "r_foods": r_foods,  "p_foods": p_foods,
        "r_snap":  r_snap,   "p_snap":  p_snap,
    }


# ─── 실험 K: 유사 아이템 학습 재검토 ────────────────────────────────────────

def experiment_k(data: dict, g1_model: AttnBottleneck) -> dict:
    """K-1: warm Q1 G1/G2/G3 비교, K-2: cold K-range rescale MAE."""
    logger.info("=== 실험 K: 유사 아이템 학습 재검토 ===")

    warm_residual = data["warm_residual"]
    warm_mean     = data["warm_mean"]
    cold_residual = data["cold_residual"]
    cold_mean     = data["cold_mean"]
    y_warm        = data["y_warm"]
    y_cold        = data["y_cold"]
    cold_meta     = data["cold_meta"]

    item_sales_mean_warm = y_warm.mean(axis=1)
    global_warm_mean     = float(item_sales_mean_warm.mean())
    dist_cold_warm       = cosine_dist(cold_mean, warm_mean)   # (100, 300)

    # K-1: Warm Q1 분석 ─────────────────────────────────────────────────────
    logger.info("  K-1: Warm Q1 분석...")
    q1_thresh = np.percentile(item_sales_mean_warm, 25)
    q1_mask   = item_sales_mean_warm <= q1_thresh
    q1_idx    = np.where(q1_mask)[0]
    logger.info("  Q1: %d 아이템, mean sales=%.2f", q1_mask.sum(),
                item_sales_mean_warm[q1_idx].mean())

    # warm 내부 5-fold CV (Q1만 평가)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_warm = len(warm_residual)
    g1_warm_preds = np.zeros_like(y_warm)
    g2_warm_preds = np.zeros_like(y_warm)
    g3_warm_preds = np.zeros_like(y_warm)

    for tr, val in kf.split(np.arange(n_warm)):
        dist_vt = cosine_dist(warm_mean[val], warm_mean[tr])
        mdl = train_head(warm_residual[tr], y_warm[tr], n_epochs=TRAIN_EPOCHS_CV, seed=42)
        base = predict_head(mdl, warm_residual[val])    # (|val|, 16)
        tr_sales = item_sales_mean_warm[tr]

        g1_warm_preds[val] = base
        for j, i in enumerate(val):
            scale = float(item_sales_mean_warm[np.argsort(dist_vt[j])[:K_ABLATION]].mean()) \
                    / (global_warm_mean + 1e-6)
            g2_warm_preds[i] = base[j] * scale
            g3_warm_preds[i] = y_warm[tr][np.argsort(dist_vt[j])[:K_ABLATION]].mean(axis=0)

    mae_g1_q1 = float(np.abs(y_warm[q1_mask] - g1_warm_preds[q1_mask]).mean())
    mae_g2_q1 = float(np.abs(y_warm[q1_mask] - g2_warm_preds[q1_mask]).mean())
    mae_g3_q1 = float(np.abs(y_warm[q1_mask] - g3_warm_preds[q1_mask]).mean())
    logger.info("  K-1 warm Q1: G1=%.2f  G2=%.2f  G3=%.2f", mae_g1_q1, mae_g2_q1, mae_g3_q1)

    # K-2: Cold K-range rescale MAE ─────────────────────────────────────────
    logger.info("  K-2: Cold K-range [5,10,20,50,100,150,200,300]...")
    cold_base = predict_head(g1_model, cold_residual)
    Ks = [5, 10, 20, 50, 100, 150, 200, 300]
    k_mae_rows = []
    for K in Ks:
        preds = np.zeros_like(cold_base)
        for i in range(len(cold_meta)):
            idx = np.argsort(dist_cold_warm[i])[:K]
            scale = float(item_sales_mean_warm[idx].mean()) / (global_warm_mean + 1e-6)
            preds[i] = cold_base[i] * scale
        mae = float(np.abs(y_cold - preds).mean())
        k_mae_rows.append({"K": K, "mae": mae})
        logger.info("  K-2 K=%3d → cold MAE=%.4f", K, mae)

    # K=300 (=전체 warm 평균, rescale=1) 확인
    g1_cold_mae = float(np.abs(y_cold - cold_base).mean())
    logger.info("  K-2 G1 (no rescale): %.4f  K=300: %.4f", g1_cold_mae,
                k_mae_rows[-1]["mae"])

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # K-1 bar chart
    labels = ["G1\n(head only)", "G2\n(rescale K=50)", "G3\n(K-NN mean)"]
    maes   = [mae_g1_q1, mae_g2_q1, mae_g3_q1]
    colors = ["steelblue", "tomato", "seagreen"]
    bars = axes[0].bar(labels, maes, color=colors, alpha=0.85)
    axes[0].set_title("K-1: Warm Q1 아이템 G1/G2/G3 MAE")
    axes[0].set_ylabel("MAE")
    for bar, val in zip(bars, maes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    # K-2 line chart
    ks_arr = [r["K"] for r in k_mae_rows]
    mae_arr = [r["mae"] for r in k_mae_rows]
    axes[1].plot(ks_arr, mae_arr, marker="o", color="tomato", label="K-NN rescale MAE")
    axes[1].axhline(g1_cold_mae, color="blue", linestyle="--", label=f"G1 no rescale={g1_cold_mae:.3f}")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Cold MAE")
    axes[1].set_title("K-2: Cold K별 rescale MAE")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "k_similarity_recheck.png", dpi=150); plt.close(fig)

    return {
        "q1_count": int(q1_mask.sum()),
        "q1_mean_sales": float(item_sales_mean_warm[q1_idx].mean()),
        "mae_g1_q1": mae_g1_q1, "mae_g2_q1": mae_g2_q1, "mae_g3_q1": mae_g3_q1,
        "k2_rows": k_mae_rows, "g1_cold_mae": g1_cold_mae,
    }


TRAIN_EPOCHS_CV = 200


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(rh: dict, ri: dict, rj: dict, rk: dict) -> None:
    lines: list[str] = [
        "# G1 최선 원인 규명 + Attention 심층 분석 (Exp017)",
        "",
        "**작성일:** 2026-03-10",
        "**참조 기준:** G3 cold MAE=8.9784, G5 cold MAE=8.9286, G1 (exp016) 8.77",
        "",
        "---",
        "",
        "## 실험 H: G1 안정성 검증 (10 seeds)",
        "",
        f"| 지표 | 값 |",
        f"|------|---|",
        f"| G1 MAE mean ± std | **{rh['mean']:.4f} ± {rh['std']:.4f}** |",
        f"| G1 MAE min/max | {min(rh['maes']):.4f} / {max(rh['maes']):.4f} |",
        f"| G3 MAE (deterministic) | {G3_COLD_MAE} |",
        f"| G5 MAE (deterministic) | {G5_COLD_MAE} |",
        f"| t-test G1 vs G3: t={rh['t_vs_g3']:.3f} p={rh['p_vs_g3']:.4f} | {'유의' if rh['p_vs_g3'] < 0.05 else '비유의'} |",
        f"| t-test G1 vs G5: t={rh['t_vs_g5']:.3f} p={rh['p_vs_g5']:.4f} | {'유의' if rh['p_vs_g5'] < 0.05 else '비유의'} |",
        "",
    ]
    if rh['p_vs_g3'] < 0.05 and rh['mean'] < G3_COLD_MAE:
        lines.append("→ G1이 G3보다 통계적으로 유의하게 나음 (p<0.05). embedding 기여 확인.")
    elif rh['p_vs_g3'] >= 0.05:
        lines.append(f"→ G1과 G3의 차이 비유의 (p={rh['p_vs_g3']:.3f}). embedding의 독립 기여 불명확.")
    lines += ["", "---", "",
              "## 실험 I: Rescale 악화 원인 분석", "",
              "### I-1: Rescale Factor 분포 (K=50)", "",
              f"| factor mean | {ri['factor_mean']:.3f} |",
              f"| factor std  | {ri['factor_std']:.3f} |",
              f"| factor min/max | {ri['factor_min']:.3f} / {ri['factor_max']:.3f} |",
              "",
              f"- factor mean={ri['factor_mean']:.3f}: K-NN 이웃의 판매량이 전체 warm 평균의 {ri['factor_mean']*100:.0f}%",
              "",
              "**factor 극단값 아이템 MAE 비교:**",
              "",
              "| 그룹 | G1 MAE | G2 MAE (rescale) | 변화 |",
              "|------|--------|------------------|------|",
              f"| factor < 33분위 (작은 rescale) | {ri['mae_g1_low']:.2f} | {ri['mae_g2_low']:.2f} | {'악화' if ri['mae_g2_low'] > ri['mae_g1_low'] else '개선'} |",
              f"| factor > 67분위 (큰 rescale) | {ri['mae_g1_high']:.2f} | {ri['mae_g2_high']:.2f} | {'악화' if ri['mae_g2_high'] > ri['mae_g1_high'] else '개선'} |",
              "",
              "### I-2: K-NN 품질 분석", "",
              f"- Category match rate (mean): **{ri['cat_match_rate_mean']:.3f}**",
              f"- Department match rate (mean): **{ri['dept_match_rate_mean']:.3f}**",
              f"- K-NN mean sales / cold item sales (ratio): mean={ri['sales_ratio_mean']:.2f} ± {ri['sales_ratio_std']:.2f}",
              "",
              f"→ K-NN 이웃의 평균 판매량이 cold item 판매량의 **{ri['sales_ratio_mean']:.1f}배**. " +
              ("이 불일치가 rescale 악화의 직접 원인." if ri['sales_ratio_mean'] > 1.3 or ri['sales_ratio_mean'] < 0.7 else "비교적 유사한 스케일."),
              "",
              "### I-3: G1 예측 스케일 확인", "",
              f"| 지표 | 값 |",
              f"|------|---|",
              f"| G1 pred mean | **{ri['pred_mean']:.2f}** |",
              f"| Actual cold mean | **{ri['actual_mean']:.2f}** |",
              f"| G1 pred std | {ri['pred_std']:.2f} |",
              f"| Actual cold std | {ri['actual_std']:.2f} |",
              f"| Pearson r (pred vs actual) | {ri['pearson_r']:.3f} |",
              ""]

    if abs(ri['pred_mean'] - ri['actual_mean']) < ri['actual_mean'] * 0.2:
        lines.append("→ G1 예측 스케일이 이미 cold 스케일에 가까움 → rescale 불필요.")
    else:
        lines.append(f"→ G1 pred mean({ri['pred_mean']:.1f}) ≠ actual mean({ri['actual_mean']:.1f}) → 스케일 미스매치 존재.")

    lines += ["", "---", "",
              "## 실험 J: Attention 심층 분석", "",
              "### J-2: 카테고리별 top-5 Attention 페르소나", ""]
    for cat, top5 in sorted(rj["cat_top5"].items()):
        lines.append(f"**[{cat}]**")
        lines.append("| Rank | P# | Weight | Budget | Economic | FOODS pref | CAT pref |")
        lines.append("|------|-----|--------|--------|----------|------------|----------|")
        for rank, (pi, w, budget, econ, foods_p, cat_p) in enumerate(top5, 1):
            lines.append(f"| {rank} | P{pi+1:02d} | {w:.4f} | ${budget:.0f} | {econ} | {foods_p:.1f} | {cat_p:.1f} |")
        lines.append("")
    lines += [f"- 카테고리별 top-1 페르소나 다름: **{rj['unique_top1_per_cat']}** ({rj['top1_per_cat']})", ""]

    lines += [
        "### J-3: Attention Entropy vs MAE", "",
        f"| 지표 | 값 |",
        f"|------|---|",
        f"| Attention entropy mean | {rj['entropy_mean']:.4f} |",
        f"| Uniform entropy (log 50) | {rj['uniform_entropy']:.4f} |",
        f"| Sparse items MAE | {rj['mae_sparse']:.2f} |",
        f"| Diffuse items MAE | {rj['mae_diffuse']:.2f} |",
        f"| Spearman r (entropy vs MAE) | {rj['spearman_entropy_mae']:.3f} (p={rj['p_entropy_mae']:.4f}) |",
        "",
    ]
    lines += [
        "### J-4: 페르소나 속성 vs Attention Weight", "",
        "| 속성 | Spearman r | p-value |",
        "|------|-----------|---------|",
        f"| weekly_budget | {rj['r_budget']:.3f} | {rj['p_budget']:.4f} |",
        f"| FOODS category pref | {rj['r_foods']:.3f} | {rj['p_foods']:.4f} |",
        f"| snap_eligible | {rj['r_snap']:.3f} | {rj['p_snap']:.4f} |",
        "", "---", "",
        "## 실험 K: 유사 아이템 학습 재검토", "",
        "### K-1: Warm Q1 아이템 분석", "",
        f"- Q1 기준 (<=25분위): {rk['q1_count']}개, mean weekly sales={rk['q1_mean_sales']:.2f}",
        f"  (cold items mean sales=10.26과 유사한 판매량 그룹)",
        "",
        "| 변형 | Warm Q1 MAE |",
        "|------|------------|",
        f"| G1 (head only) | {rk['mae_g1_q1']:.2f} |",
        f"| G2 (rescale K=50) | {rk['mae_g2_q1']:.2f} |",
        f"| G3 (K-NN mean) | {rk['mae_g3_q1']:.2f} |",
        "",
    ]
    q1_rescale_verdict = "개선" if rk['mae_g2_q1'] < rk['mae_g1_q1'] else "악화"
    if rk['mae_g2_q1'] < rk['mae_g1_q1']:
        lines.append("→ Warm Q1에서도 rescale이 개선됨 → warm/cold 판매량 분포 차이가 문제.")
    else:
        lines.append(f"→ Warm Q1에서도 rescale이 {q1_rescale_verdict} → 저판매 아이템에 rescale 자체가 적합하지 않음.")

    lines += ["", "### K-2: Cold K별 Rescale MAE", "",
              "| K | Cold MAE |",
              "|---|----------|"]
    for row in rk["k2_rows"]:
        lines.append(f"| {row['K']} | {row['mae']:.4f} |")
    lines += [f"| G1 no rescale | {rk['g1_cold_mae']:.4f} |", ""]

    best_k2 = min(rk["k2_rows"], key=lambda r: r["mae"])
    lines.append(f"→ 최적 K={best_k2['K']} (MAE={best_k2['mae']:.4f}). "
                 + ("K 증가할수록 G1에 수렴하는지: " +
                    ("예 — K=300 ≈ G1" if abs(rk['k2_rows'][-1]['mae'] - rk['g1_cold_mae']) < 0.1
                     else f"K=300 MAE={rk['k2_rows'][-1]['mae']:.4f} ≠ G1={rk['g1_cold_mae']:.4f}")))

    lines += ["", "---", "",
              "## 종합 결론", "",
              "### 핵심 발견 요약", "",
              f"1. **G1 안정성**: MAE {rh['mean']:.3f}±{rh['std']:.3f}  vs G3={G3_COLD_MAE:.3f} "
              + ("(유의한 차이)" if rh['p_vs_g3'] < 0.05 else "(비유의)"),
              f"2. **Rescale 악화 원인**: factor mean={ri['factor_mean']:.2f}, "
              f"K-NN/cold sales ratio={ri['sales_ratio_mean']:.2f} → "
              + ("warm과 cold의 판매량 스케일 불일치가 주원인" if abs(ri['sales_ratio_mean'] - 1.0) > 0.3
                 else "스케일은 유사, 다른 원인"),
              f"3. **G1 pred mean={ri['pred_mean']:.2f} vs actual={ri['actual_mean']:.2f}**: "
              + ("head가 이미 cold 스케일을 학습 → rescale 불필요" if abs(ri['pred_mean'] - ri['actual_mean']) < 2.0
                 else "스케일 미스매치 → head 개선 필요"),
              f"4. **Attention 카테고리 분리**: {'분리됨' if rj['unique_top1_per_cat'] else 'P042 등 특정 페르소나 지배'}",
              f"5. **Warm Q1 rescale**: {q1_rescale_verdict} → 저판매 아이템에서 rescale 방향성 확인",
              "",
              "---",
              "",
              "**시각화:** `experiments/exp017_g1_analysis/figures/`",
              "**스크립트:** `scripts/exp017_g1_analysis.py`"]

    path = REPORT_DIR / "g1_analysis_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", path)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Exp017: G1 분석 시작 ===")

    data     = load_data()
    personas = load_personas()

    # G1 기준 모델 (seed=0, H에서 재사용)
    logger.info("기준 G1 모델 학습 (seed=0)...")
    g1_model = train_head(data["warm_residual"], data["y_warm"], seed=0)

    rh = experiment_h(data)               # seed 0은 H에서 다시 학습됨
    ri = experiment_i(data, g1_model)
    rj = experiment_j(data, g1_model, personas)
    rk = experiment_k(data, g1_model)

    write_report(rh, ri, rj, rk)

    print("\n" + "=" * 60)
    print("=== Exp017 결과 요약 ===")
    print(f"H: G1 MAE = {rh['mean']:.4f} ± {rh['std']:.4f}")
    print(f"   vs G3(p={rh['p_vs_g3']:.3f})  vs G5(p={rh['p_vs_g5']:.3f})")
    print(f"I: rescale factor mean={ri['factor_mean']:.2f}  K-NN/cold ratio={ri['sales_ratio_mean']:.2f}")
    print(f"   G1 pred mean={ri['pred_mean']:.2f}  actual mean={ri['actual_mean']:.2f}")
    print(f"J: entropy={rj['entropy_mean']:.3f} (uniform={rj['uniform_entropy']:.3f})")
    print(f"   cat top-1 unique: {rj['unique_top1_per_cat']}")
    print(f"K: warm Q1 G1={rk['mae_g1_q1']:.2f} G2={rk['mae_g2_q1']:.2f} G3={rk['mae_g3_q1']:.2f}")
    best = min(rk["k2_rows"], key=lambda r: r["mae"])
    print(f"   cold best K={best['K']} MAE={best['mae']:.4f}  G1={rk['g1_cold_mae']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
