"""exp020: 판단 재료 수집 — Residual vs Raw MAE 원인 규명 + 아이템별 성적표.

파트 A: Raw vs Residual 4가지 변형 비교 (동일 평가 함수, 16주)
파트 B: 아이템별 성적표 (Track B G1 + lightgbm + knn)
파트 C: 평가 함수 소스, config 요약, 수치 일관성

출력:
    docs/diagnosis/error_analysis_materials.md
    experiments/exp020_error_analysis/item_scorecard.csv
    experiments/exp020_error_analysis/pred_comparison.csv
    experiments/exp020_error_analysis/figures/
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    evaluate_weekly, mae, rmse, wrmsse, direction_accuracy_weekly, _merge,
    _compute_category_scales,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 경로 상수 ────────────────────────────────────────────────────────────────
EMB_DIR  = ROOT / "experiments/exp011_v3_pipeline/embeddings"
COMP_DIR = ROOT / "experiments/exp006_competitors"
CS_DIR   = ROOT / "data/processed/cold_start"
M5_DIR   = ROOT / "m5-forecasting-accuracy"
OUT_DIR  = ROOT / "experiments/exp020_error_analysis"
FIG_DIR  = OUT_DIR / "figures"
REPORT   = ROOT / "docs/diagnosis/error_analysis_materials.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── exp016 G1 설정 (Track B 기준선) ─────────────────────────────────────────
G1_CFG = dict(epochs=500, lr=1e-3, weight_decay=1e-4,
              bottleneck=64, dropout=0.1, seed=42)
# exp011 3-4_attn_bottleneck 설정 (exp013_scaling_law.py 주석 "exp011과 동일")
E11_CFG = dict(epochs=200, patience=20, lr=1e-3, weight_decay=1e-2,
               bottleneck=64, dropout=0.5, seed=42)


# ─── 모델 정의 ────────────────────────────────────────────────────────────────

class AttnBottleneck_G1(nn.Module):
    """exp016 G1 구조: dropout=0.1, 출력 n_weeks."""
    def __init__(self, hidden=5120, bottleneck=64, n_weeks=16, dropout=0.1):
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, bottleneck), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, n_weeks),
        )
    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        ctx = (x * w.unsqueeze(-1)).sum(dim=1)
        return self.head(ctx)


class AttnBottleneck_E11(nn.Module):
    """exp011 3-4_attn_bottleneck 구조: dropout=0.5, 출력 n_weeks."""
    def __init__(self, hidden=5120, bottleneck=64, n_weeks=17, dropout=0.5):
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, bottleneck), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, n_weeks),
        )
    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        w = torch.softmax(scores, dim=-1)
        ctx = (x * w.unsqueeze(-1)).sum(dim=1)
        return self.head(ctx)


def train_g1(X: np.ndarray, y: np.ndarray) -> AttnBottleneck_G1:
    """exp016 G1 학습: MAE loss, weight_decay=1e-4, clip_grad=1.0, 500ep."""
    torch.manual_seed(G1_CFG["seed"])
    n_weeks = y.shape[1]
    model = AttnBottleneck_G1(hidden=X.shape[2], bottleneck=G1_CFG["bottleneck"],
                               n_weeks=n_weeks, dropout=G1_CFG["dropout"])
    opt = Adam(model.parameters(), lr=G1_CFG["lr"],
               weight_decay=G1_CFG["weight_decay"])
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(G1_CFG["epochs"]):
        model.train(); opt.zero_grad()
        loss = F.l1_loss(model(Xt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


def train_e11(X: np.ndarray, y: np.ndarray) -> AttnBottleneck_E11:
    """exp011 학습: MSE loss, weight_decay=1e-2, early stopping(patience=20), 200ep."""
    torch.manual_seed(E11_CFG["seed"])
    n_weeks = y.shape[1]
    model = AttnBottleneck_E11(hidden=X.shape[2], bottleneck=E11_CFG["bottleneck"],
                                n_weeks=n_weeks, dropout=E11_CFG["dropout"])
    opt = Adam(model.parameters(), lr=E11_CFG["lr"],
               weight_decay=E11_CFG["weight_decay"])
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    best_loss, no_imp, best_state = float("inf"), 0, None
    for _ in range(E11_CFG["epochs"]):
        model.train(); opt.zero_grad()
        loss = F.mse_loss(model(Xt), yt)
        loss.backward()
        opt.step()
        v = loss.item()
        if v < best_loss:
            best_loss = v; no_imp = 0
            best_state = {k: v2.clone() for k, v2 in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= E11_CFG["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_np(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return np.clip(p, 0, None)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data() -> dict:
    logger.info("데이터 로드 중...")
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", map_location="cpu",
                          weights_only=False).numpy()   # (300,50,5120)
    cold_raw = torch.load(EMB_DIR / "cold_raw.pt", map_location="cpu",
                          weights_only=False).numpy()   # (100,50,5120)
    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)

    item_meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_ids = item_meta[item_meta["is_cold"] == True]["item_id"].tolist()
    warm_ids = item_meta[item_meta["is_cold"] == False]["item_id"].tolist()

    cold_test_raw  = pd.read_csv(CS_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_train_raw = pd.read_csv(CS_DIR / "warm_train.csv", parse_dates=["date"])
    warm_test_raw  = pd.read_csv(CS_DIR / "warm_test.csv",  parse_dates=["date"])

    def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
        df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
        df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
        gcols = [c for c in ["item_id","store_id","cat_id","dept_id","iso_year","iso_week"]
                 if c in df.columns]
        return (df.groupby(gcols)
                .agg(sales=("sales","sum"), date=("week_start","first"))
                .reset_index())

    cold_weekly  = to_weekly(cold_test_raw)
    warm_train_w = to_weekly(warm_train_raw)
    warm_test_w  = to_weekly(warm_test_raw)

    # 완전한 16주
    day_cnt = (cold_test_raw.assign(
        iso_year=cold_test_raw["date"].dt.isocalendar().year.astype(int),
        iso_week=cold_test_raw["date"].dt.isocalendar().week.astype(int))
        .groupby(["iso_year","iso_week"])["date"].nunique())
    complete_set = set(zip(day_cnt[day_cnt==7].index.get_level_values(0),
                           day_cnt[day_cnt==7].index.get_level_values(1)))
    cold_weekly_16 = cold_weekly[cold_weekly.apply(
        lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)].copy()
    wl16 = sorted(complete_set)

    wk2date = {}
    for _, r in cold_weekly_16.iterrows():
        wk = (r["iso_year"], r["iso_week"])
        if wk not in wk2date:
            wk2date[wk] = r["date"]
    wd16 = [wk2date.get(wk, pd.Timestamp("2016-01-01")) for wk in wl16]

    def build_y(ids, weekly, week_list):
        wi = {wk: i for i, wk in enumerate(week_list)}
        ii = {iid: i for i, iid in enumerate(ids)}
        y = np.zeros((len(ids), len(week_list)), dtype=np.float32)
        for _, r in weekly.iterrows():
            wk = (r["iso_year"], r["iso_week"])
            if wk in wi and r["item_id"] in ii:
                y[ii[r["item_id"]], wi[wk]] = r["sales"]
        return y

    y_warm_16 = build_y(warm_ids, warm_test_w, wl16)
    y_cold_16 = build_y(cold_ids, cold_weekly_16, wl16)
    # exp011은 17주로 학습 → y_warm/cold_17도 구성
    all_weeks_set = set(zip(
        cold_weekly["iso_year"], cold_weekly["iso_week"]))
    wl17 = sorted(all_weeks_set)
    y_warm_17 = build_y(warm_ids, warm_test_w, wl17)
    y_cold_17 = build_y(cold_ids, cold_weekly, wl17)

    cold_meta_df = (cold_test_raw[["item_id","cat_id","dept_id"]]
                    .drop_duplicates("item_id").set_index("item_id"))
    cold_stats = pd.read_csv(CS_DIR / "cold_item_stats.csv").set_index("item_id")

    logger.info("  warm_raw=%s cold_raw=%s", warm_raw.shape, cold_raw.shape)
    logger.info("  y_warm_16=%s y_cold_16=%s  y_warm_17=%s",
                y_warm_16.shape, y_cold_16.shape, y_warm_17.shape)
    return {
        "warm_raw": warm_raw, "cold_raw": cold_raw,
        "warm_residual": warm_residual, "cold_residual": cold_residual,
        "cold_ids": cold_ids, "warm_ids": warm_ids,
        "cold_weekly_16": cold_weekly_16,
        "cold_weekly_17": cold_weekly,
        "warm_train_weekly": warm_train_w,
        "warm_test_weekly": warm_test_w,
        "y_warm_16": y_warm_16, "y_cold_16": y_cold_16,
        "y_warm_17": y_warm_17, "y_cold_17": y_cold_17,
        "week_list_16": wl16, "week_dates_16": wd16,
        "week_list_17": wl17,
        "cold_meta": cold_meta_df,
        "cold_stats": cold_stats,
    }


def preds_to_df(preds, cold_ids, cat_ids, week_list, week_dates):
    rows = []
    for ii, iid in enumerate(cold_ids):
        for wi, wk in enumerate(week_list):
            rows.append({
                "item_id": iid, "store_id": "CA_1",
                "cat_id": cat_ids[ii] if cat_ids else "",
                "iso_year": wk[0], "iso_week": wk[1],
                "date": week_dates[wi],
                "pred_sales": float(preds[ii, wi]),
            })
    return pd.DataFrame(rows)


def full_eval(pred_df, cold_weekly, warm_train, name):
    ev = evaluate_weekly(cold_weekly, pred_df, warm_train, name)
    return {
        "mae": float(ev["mae"]),
        "rmse": float(ev["rmse"]),
        "wrmsse": float(ev["wrmsse"]),
        "dir_acc": float(ev["direction_accuracy"]),
    }


# ─── Part A ───────────────────────────────────────────────────────────────────

def part_a(data: dict) -> dict:
    logger.info("=== Part A: Raw vs Residual 4가지 변형 비교 ===")
    cold_ids = data["cold_ids"]
    cat_ids  = [data["cold_meta"]["cat_id"].get(i, "") for i in cold_ids]
    cold_weekly_16 = data["cold_weekly_16"]
    warm_train     = data["warm_train_weekly"]
    wl16, wd16     = data["week_list_16"], data["week_dates_16"]

    results = {}
    preds_all = {}

    # A-1a: raw embedding, G1 train_head (MAE loss), 16주
    logger.info("  A-1a: raw + G1 (MAE loss, 500ep, 16주)...")
    m = train_g1(data["warm_raw"], data["y_warm_16"])
    p = predict_np(m, data["cold_raw"])
    pdf = preds_to_df(p, cold_ids, cat_ids, wl16, wd16)
    results["A-1a_raw_G1"] = full_eval(pdf, cold_weekly_16, warm_train, "A1a")
    preds_all["A-1a"] = p
    logger.info("    MAE=%.4f", results["A-1a_raw_G1"]["mae"])

    # A-1b: residual embedding, G1 train_head
    logger.info("  A-1b: residual + G1 (MAE loss, 500ep, 16주)...")
    m = train_g1(data["warm_residual"], data["y_warm_16"])
    p = predict_np(m, data["cold_residual"])
    pdf = preds_to_df(p, cold_ids, cat_ids, wl16, wd16)
    results["A-1b_res_G1"] = full_eval(pdf, cold_weekly_16, warm_train, "A1b")
    preds_all["A-1b"] = p
    logger.info("    MAE=%.4f", results["A-1b_res_G1"]["mae"])

    # A-1c: raw embedding, exp011 train_head (MSE loss, 17주 학습 → 16주만 평가)
    logger.info("  A-1c: raw + E11 (MSE loss, early-stop, 17주→16주 평가)...")
    m = train_e11(data["warm_raw"], data["y_warm_17"])
    p17 = predict_np(m, data["cold_raw"])
    # 17주 예측에서 16주에 해당하는 열만 추출
    wl17 = data["week_list_17"]
    idx_16 = [i for i, wk in enumerate(wl17) if wk in set(wl16)]
    p16 = p17[:, idx_16]
    pdf = preds_to_df(p16, cold_ids, cat_ids, wl16, wd16)
    results["A-1c_raw_E11"] = full_eval(pdf, cold_weekly_16, warm_train, "A1c")
    preds_all["A-1c"] = p16
    logger.info("    MAE=%.4f", results["A-1c_raw_E11"]["mae"])

    # A-1d: residual embedding, exp011 train_head
    logger.info("  A-1d: residual + E11 (MSE loss, early-stop, 17주→16주 평가)...")
    m = train_e11(data["warm_residual"], data["y_warm_17"])
    p17 = predict_np(m, data["cold_residual"])
    p16 = p17[:, idx_16]
    pdf = preds_to_df(p16, cold_ids, cat_ids, wl16, wd16)
    results["A-1d_res_E11"] = full_eval(pdf, cold_weekly_16, warm_train, "A1d")
    preds_all["A-1d"] = p16
    logger.info("    MAE=%.4f", results["A-1d_res_E11"]["mae"])

    # A-2: 예측값 분포
    y_cold = data["y_cold_16"]
    y_warm = data["y_warm_16"]
    distrib = {
        "y_cold_actual": {"mean": float(y_cold.mean()), "std": float(y_cold.std()),
                          "min": float(y_cold.min()), "max": float(y_cold.max())},
        "y_warm_actual": {"mean": float(y_warm.mean()), "std": float(y_warm.std()),
                          "min": float(y_warm.min()), "max": float(y_warm.max())},
    }
    for k, p in preds_all.items():
        distrib[k] = {"mean": float(p.mean()), "std": float(p.std()),
                      "min": float(p.min()), "max": float(p.max())}

    # A-3: 아이템별 scatter 데이터 (100개)
    item_actual_mean = y_cold.mean(axis=1)
    rows = []
    for ii, iid in enumerate(cold_ids):
        rows.append({
            "item_id": iid,
            "actual_mean": float(item_actual_mean[ii]),
            "pred_raw_G1_mean": float(preds_all["A-1a"][ii].mean()),
            "pred_res_G1_mean": float(preds_all["A-1b"][ii].mean()),
            "pred_raw_E11_mean": float(preds_all["A-1c"][ii].mean()),
            "pred_res_E11_mean": float(preds_all["A-1d"][ii].mean()),
        })
    scatter_df = pd.DataFrame(rows)
    scatter_df.to_csv(OUT_DIR / "pred_comparison.csv", index=False)
    logger.info("  pred_comparison.csv 저장")

    # A 시각화: 예측값 분포 histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 300, 50)
    axes[0].hist(preds_all["A-1a"].flatten(), bins=bins, alpha=0.6,
                 label="raw+G1 (MAE)", color="steelblue", density=True)
    axes[0].hist(preds_all["A-1b"].flatten(), bins=bins, alpha=0.6,
                 label="res+G1 (MAE)", color="tomato", density=True)
    axes[0].hist(y_cold.flatten(), bins=bins, alpha=0.4,
                 label="actual cold", color="green", density=True)
    axes[0].set_title("G1 학습 방식: raw vs residual 예측 분포")
    axes[0].set_xlabel("weekly sales"); axes[0].legend(fontsize=8); axes[0].set_xlim(0, 300)

    axes[1].hist(preds_all["A-1c"].flatten(), bins=bins, alpha=0.6,
                 label="raw+E11 (MSE)", color="steelblue", density=True)
    axes[1].hist(preds_all["A-1d"].flatten(), bins=bins, alpha=0.6,
                 label="res+E11 (MSE)", color="tomato", density=True)
    axes[1].hist(y_cold.flatten(), bins=bins, alpha=0.4,
                 label="actual cold", color="green", density=True)
    axes[1].set_title("E11 학습 방식: raw vs residual 예측 분포")
    axes[1].set_xlabel("weekly sales"); axes[1].legend(fontsize=8); axes[1].set_xlim(0, 300)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "a_pred_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("  a_pred_distribution.png 저장")

    return {"metrics": results, "distrib": distrib, "scatter_df": scatter_df}


# ─── Part B ───────────────────────────────────────────────────────────────────

def _item_level_metrics(pred_df, cold_weekly, warm_train, cold_ids):
    """아이템별 MAE, WRMSSE, DirAcc 계산."""
    merged = cold_weekly.merge(
        pred_df[["item_id","iso_year","iso_week","pred_sales"]],
        on=["item_id","iso_year","iso_week"], how="inner")

    # 카테고리별 WRMSSE 스케일
    cat_scales = _compute_category_scales(warm_train, lag=4)

    rows = {}
    for iid, grp in merged.groupby("item_id"):
        cat = grp["cat_id"].iloc[0]
        actual = grp["sales"].values
        pred   = grp["pred_sales"].values
        mae_v  = float(np.abs(actual - pred).mean())

        scale = cat_scales.get(str(cat), 1.0) or 1.0
        mse_i = float(np.mean((actual - pred) ** 2))
        wrmsse_v = float(np.sqrt(mse_i / scale))

        # DirAcc (flat 포함)
        sub = grp.sort_values(["iso_year","iso_week"])
        act_dir = np.sign(np.diff(sub["sales"].values))
        prd_dir = np.sign(np.diff(sub["pred_sales"].values))
        da_v    = float((act_dir == prd_dir).mean()) if len(act_dir) > 0 else float("nan")
        # DirAcc (flat 제외)
        m = act_dir != 0
        da_nf = float((act_dir[m] == prd_dir[m]).mean()) if m.sum() > 0 else float("nan")

        rows[iid] = {"mae": mae_v, "wrmsse": wrmsse_v,
                     "dir_acc": da_v, "dir_acc_noflat": da_nf}
    return rows


def part_b(data: dict) -> dict:
    logger.info("=== Part B: 아이템별 성적표 ===")
    cold_ids = data["cold_ids"]
    cat_ids  = [data["cold_meta"]["cat_id"].get(i, "") for i in cold_ids]
    cold_weekly = data["cold_weekly_16"]
    warm_train  = data["warm_train_weekly"]
    wl16, wd16  = data["week_list_16"], data["week_dates_16"]
    cold_meta   = data["cold_meta"]
    cold_stats  = data["cold_stats"]

    # ─ Track B G1 예측 (exp016 G1: residual, 16주, seed=42)
    logger.info("  Track B G1 예측 중...")
    m_g1 = train_g1(data["warm_residual"], data["y_warm_16"])
    p_g1 = predict_np(m_g1, data["cold_residual"])
    pdf_g1 = preds_to_df(p_g1, cold_ids, cat_ids, wl16, wd16)
    tb_ev = full_eval(pdf_g1, cold_weekly, warm_train, "trackb_g1")
    logger.info("  Track B G1: MAE=%.4f DirAcc=%.4f", tb_ev["mae"], tb_ev["dir_acc"])
    tb_item = _item_level_metrics(pdf_g1, cold_weekly, warm_train, cold_ids)

    # ─ competitor 예측 로드
    def load_comp(model_name):
        p = COMP_DIR / model_name / "predictions" / f"{model_name}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, parse_dates=["date"])
        df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
        df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
        return df

    wl16_set = set(wl16)
    lgbm_df  = load_comp("lightgbm_proxy_lags")
    knn_df   = load_comp("knn_analog")

    def filter16(df):
        return df[df.apply(
            lambda r: (r["iso_year"],r["iso_week"]) in wl16_set, axis=1)].copy()

    lgbm_item, knn_item = {}, {}
    if lgbm_df is not None:
        lgbm_item = _item_level_metrics(filter16(lgbm_df), cold_weekly, warm_train, cold_ids)
        logger.info("  lgbm 아이템별 지표 계산 완료: %d items", len(lgbm_item))
    if knn_df is not None:
        knn_item = _item_level_metrics(filter16(knn_df), cold_weekly, warm_train, cold_ids)
        logger.info("  knn 아이템별 지표 계산 완료: %d items", len(knn_item))

    # ─ 평균 가격 조회
    prices_df = pd.read_csv(M5_DIR / "sell_prices.csv")
    cal_df    = pd.read_csv(M5_DIR / "calendar.csv")
    # cold test 기간의 wm_yr_wk 확인
    cold_test_raw = pd.read_csv(CS_DIR / "cold_test.csv", parse_dates=["date"])
    test_dates = cold_test_raw["date"].dt.normalize().unique()
    cal_sub    = cal_df[pd.to_datetime(cal_df["date"]).isin(test_dates)]["wm_yr_wk"].unique()
    price_avg  = (prices_df[(prices_df["store_id"]=="CA_1") &
                             (prices_df["wm_yr_wk"].isin(cal_sub))]
                  .groupby("item_id")["sell_price"].mean())

    # ─ 아이템별 실제 판매 통계 (16주)
    actual_stats = (cold_weekly.groupby("item_id").agg(
        actual_weekly_mean=("sales","mean"),
        actual_weekly_std=("sales","std"),
        actual_nonzero_week_ratio=("sales", lambda x: (x>0).mean()),
    ).reset_index())
    actual_stats["actual_cv"] = (actual_stats["actual_weekly_std"] /
                                  actual_stats["actual_weekly_mean"].clip(lower=1e-6))
    # Track B 예측 주간 평균
    tb_pred_mean = (pdf_g1.groupby("item_id")["pred_sales"].mean()
                   .rename("trackb_pred_weekly_mean"))

    # ─ 성적표 조합
    scorecard_rows = []
    for iid in cold_ids:
        tb  = tb_item.get(iid, {})
        lg  = lgbm_item.get(iid, {})
        knn = knn_item.get(iid, {})
        ast = actual_stats[actual_stats["item_id"]==iid].iloc[0] if iid in actual_stats["item_id"].values else {}
        row = {
            "item_id": iid,
            "cat_id": cold_meta.loc[iid, "cat_id"] if iid in cold_meta.index else "",
            "dept_id": cold_meta.loc[iid, "dept_id"] if iid in cold_meta.index else "",
            "sales_tier": cold_stats.loc[iid, "sales_tier"] if iid in cold_stats.index else "",
            "avg_price": float(price_avg.get(iid, float("nan"))),
            "actual_weekly_mean":          float(ast.get("actual_weekly_mean", float("nan"))),
            "actual_weekly_std":           float(ast.get("actual_weekly_std",  float("nan"))),
            "actual_nonzero_week_ratio":   float(ast.get("actual_nonzero_week_ratio", float("nan"))),
            "actual_cv":                   float(ast.get("actual_cv", float("nan"))),
            "trackb_pred_weekly_mean":     float(tb_pred_mean.get(iid, float("nan"))),
            "trackb_MAE":                  float(tb.get("mae",          float("nan"))),
            "trackb_WRMSSE":               float(tb.get("wrmsse",       float("nan"))),
            "trackb_DirAcc":               float(tb.get("dir_acc",      float("nan"))),
            "trackb_DirAcc_noflat":        float(tb.get("dir_acc_noflat", float("nan"))),
            "lgbm_MAE":                    float(lg.get("mae",          float("nan"))),
            "lgbm_WRMSSE":                 float(lg.get("wrmsse",       float("nan"))),
            "lgbm_DirAcc":                 float(lg.get("dir_acc",      float("nan"))),
            "knn_MAE":                     float(knn.get("mae",         float("nan"))),
            "knn_WRMSSE":                  float(knn.get("wrmsse",      float("nan"))),
            "knn_DirAcc":                  float(knn.get("dir_acc",     float("nan"))),
        }
        row["trackb_wins_lgbm_WRMSSE"] = (
            row["trackb_WRMSSE"] < row["lgbm_WRMSSE"]
            if not (np.isnan(row["trackb_WRMSSE"]) or np.isnan(row["lgbm_WRMSSE"]))
            else float("nan")
        )
        scorecard_rows.append(row)

    scorecard = pd.DataFrame(scorecard_rows)
    scorecard.to_csv(OUT_DIR / "item_scorecard.csv", index=False)
    logger.info("  item_scorecard.csv 저장 (%d rows)", len(scorecard))

    # ─ B-2: 부서별 통계
    warm_train_full = pd.read_csv(CS_DIR / "warm_train.csv", parse_dates=["date"])
    # CA_1 전체 (warm_train에서 CA_1만)
    ca1_train = warm_train_full[warm_train_full["store_id"]=="CA_1"].copy()
    ca1_train["nonzero"] = (ca1_train["sales"] > 0).astype(int)

    dept_stats = ca1_train.groupby(["dept_id","item_id"]).agg(
        daily_mean=("sales","mean"),
        daily_std=("sales","std"),
        nonzero_ratio=("nonzero","mean"),
    ).reset_index()
    dept_stats["cv"] = dept_stats["daily_std"] / dept_stats["daily_mean"].clip(lower=1e-6)

    dept_summary = dept_stats.groupby("dept_id").agg(
        n_items=("item_id","nunique"),
        daily_mean_sales=("daily_mean","mean"),
        nonzero_day_ratio=("nonzero_ratio","mean"),
        sales_cv=("cv","mean"),
    ).reset_index()

    # 가격 평균 (CA_1, 전체 기간)
    price_dept = (prices_df[prices_df["store_id"]=="CA_1"]
                  .merge(dept_stats[["item_id","dept_id"]].drop_duplicates(),
                         on="item_id", how="left")
                  .groupby("dept_id")["sell_price"].mean()
                  .rename("avg_price"))
    dept_summary = dept_summary.merge(price_dept, on="dept_id", how="left")

    # cold_count per dept
    cold_dept = (scorecard.groupby("dept_id")["item_id"].count()
                 .reset_index().rename(columns={"item_id":"cold_count"}))
    dept_summary = dept_summary.merge(cold_dept, on="dept_id", how="left")
    dept_summary["cold_count"] = dept_summary["cold_count"].fillna(0).astype(int)

    # ─ WRMSSE 분포 시각화 (Track B vs lightgbm)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    tb_wrmsse  = scorecard["trackb_WRMSSE"].dropna()
    lgbm_wrmsse = scorecard["lgbm_WRMSSE"].dropna()
    bins2 = np.linspace(0, 10, 40)
    axes[0].hist(tb_wrmsse, bins=bins2, alpha=0.7, label="Track B (G1)", color="steelblue", edgecolor="white")
    axes[0].hist(lgbm_wrmsse, bins=bins2, alpha=0.7, label="lightgbm", color="tomato", edgecolor="white")
    axes[0].set_xlabel("아이템별 WRMSSE"); axes[0].set_title("Track B vs lightgbm WRMSSE 분포")
    axes[0].legend(fontsize=9)

    tb_mae = scorecard["trackb_MAE"].dropna()
    lg_mae = scorecard["lgbm_MAE"].dropna()
    bins3  = np.linspace(0, 30, 40)
    axes[1].hist(tb_mae, bins=bins3, alpha=0.7, label="Track B (G1)", color="steelblue", edgecolor="white")
    axes[1].hist(lg_mae, bins=bins3, alpha=0.7, label="lightgbm", color="tomato", edgecolor="white")
    axes[1].set_xlabel("아이템별 MAE"); axes[1].set_title("Track B vs lightgbm MAE 분포")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "b_wrmsse_mae_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("  b_wrmsse_mae_distribution.png 저장")

    return {
        "tb_eval": tb_ev,
        "scorecard": scorecard,
        "dept_summary": dept_summary,
        "lgbm_path": str(COMP_DIR / "lightgbm_proxy_lags/predictions/lightgbm_proxy_lags.csv"),
        "knn_path":  str(COMP_DIR / "knn_analog/predictions/knn_analog.csv"),
    }


# ─── Part C ───────────────────────────────────────────────────────────────────

def part_c_mae_discrepancy(data: dict) -> dict:
    """C-3: exp018 MAE=8.39 vs exp019 MAE=9.67 차이 원인."""
    logger.info("=== Part C-3: MAE 수치 일관성 확인 ===")

    # exp018 최적: residual, 17주, epochs=500, lr=1e-3, bn=128, lambda_dir=0.5, T=2.0
    # exp019 supplement: residual, 16주, epochs=500, lr=1e-3, bn=128, lambda_dir=0.5, T=2.0 (seed 0-4)
    # 동일 모델, 다른 평가 기간(17w vs 16w) + seed
    # 여기서는 seed=42 고정으로 17w와 16w 각각 단일 실행

    # 이미 Part A에서 A-1b (residual+G1+16주) 계산함
    # 추가로 residual+G1+17주 계산
    cold_ids = data["cold_ids"]
    cat_ids  = [data["cold_meta"]["cat_id"].get(i, "") for i in cold_ids]
    warm_train = data["warm_train_weekly"]
    wl17 = data["week_list_17"]

    # 17주 week_dates
    cold_w17 = data["cold_weekly_17"]
    wk2date17 = {}
    for _, r in cold_w17.iterrows():
        wk = (r["iso_year"], r["iso_week"])
        if wk not in wk2date17:
            wk2date17[wk] = r["date"]
    wd17 = [wk2date17.get(wk, pd.Timestamp("2016-01-01")) for wk in wl17]

    logger.info("  residual+G1 17주 예측 계산 중...")
    m17 = train_g1(data["warm_residual"], data["y_warm_17"])
    p17 = predict_np(m17, data["cold_residual"])
    pdf17 = preds_to_df(p17, cold_ids, cat_ids, wl17, wd17)
    ev17 = full_eval(pdf17, cold_w17, warm_train, "g1_res_17w")
    logger.info("  residual+G1 17주: MAE=%.4f", ev17["mae"])

    # 16주는 Part A A-1b에서 계산됨 (재계산)
    wl16, wd16 = data["week_list_16"], data["week_dates_16"]
    m16 = train_g1(data["warm_residual"], data["y_warm_16"])
    p16 = predict_np(m16, data["cold_residual"])
    pdf16 = preds_to_df(p16, cold_ids, cat_ids, wl16, wd16)
    ev16 = full_eval(pdf16, data["cold_weekly_16"], warm_train, "g1_res_16w")
    logger.info("  residual+G1 16주: MAE=%.4f", ev16["mae"])

    return {
        "g1_res_17w": ev17,
        "g1_res_16w": ev16,
        "note": (
            "exp018 보고 MAE=8.39: residual, 17주, bn=128, lambda_dir=0.5, T=2.0, "
            "3 seed 평균 (실험 당시 측정값). "
            "exp019 supplement MAE=9.67: 동일 설정(bn=128, lambda_dir=0.5, T=2.0)이나 "
            "16주 기준, seed 0-4(5개) 평균 — 16주 기준 변경 + seed 분산이 주된 요인. "
            f"이번 재계산(seed=42 고정): 16주={ev16['mae']:.4f}, 17주={ev17['mae']:.4f}"
        ),
    }


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(a: dict, b: dict, c3: dict) -> None:
    logger.info("보고서 작성 중...")
    sc = b["scorecard"]

    def fmt(v, d=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):.{d}f}"

    # C-1: evaluate_weekly 소스코드 (핵심 부분만 인용)
    eval_src = dedent("""\
    ```python
    # src/evaluation/metrics.py (핵심 발췌)

    def evaluate_weekly(cold_test, pred, warm_train, model_name="unknown"):
        return evaluate(cold_test, pred, warm_train, model_name, wrmsse_lag=4)

    def evaluate(cold_test, pred, warm_train, model_name, wrmsse_lag=28):
        merged = _merge(cold_test, pred)
        result = {
            "mae":                mae(merged["sales"], merged["pred_sales"]),
            "rmse":               rmse(merged["sales"], merged["pred_sales"]),
            "wrmsse":             wrmsse(merged, warm_train, lag=wrmsse_lag),
            "direction_accuracy": direction_accuracy_weekly(merged),
        }
        return result

    def mae(actual, predicted):
        # 전체 flattened 평균 (아이템 평균 후 재평균이 아님)
        return float(np.abs(actual.values - predicted.values).mean())

    def direction_accuracy_weekly(merged):
        # (item_id, iso_year, iso_week) 단위 집계 → 전주 대비 방향 sign 비교
        # flat(sign=0) 포함: actual_dir=0 ↔ pred_dir=0 이면 일치로 처리
        ...
        correct = (weekly["actual_dir"] == weekly["pred_dir"]).sum()
        return float(correct / total)

    def wrmsse(merged, warm_train, lag=4):
        # 1. warm 아이템 카테고리별 스케일: lag-week diff MSE 평균
        # 2. cold 아이템별 RMSSE_i = sqrt(MSE_i / scale_cat)
        # 3. 가중 평균: w_i = 테스트 기간 실제 판매량 합
        ...
    ```""")

    lines = [
        "# 판단 재료 수집 보고서 (Exp020)",
        "",
        "**작성일:** 2026-03-11",
        "**목적:** 수치와 코드에서 확인된 사실만 보고. 판정 없음.",
        "",
        "---",
        "",
        "## 파트 C-2: 실험 설정 요약",
        "",
        "### Track B G1 설정 (exp016 G1, 이번 실험 기준선)",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| embedding | residual (raw - per-item-persona-mean) |",
        "| n_weeks | 16 (완전한 주만) |",
        "| epochs | 500 |",
        "| lr | 1e-3 |",
        "| weight_decay | 1e-4 |",
        "| bottleneck | 64 |",
        "| dropout | 0.1 |",
        "| loss | MAE (l1_loss) |",
        "| clip_grad_norm | 1.0 |",
        "| lambda_dir | 0 (없음) |",
        "| seed | 42 |",
        "",
        "### exp011 3-4_attn_bottleneck 설정",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| embedding | raw (no residual) |",
        "| n_weeks | 17 (부분 주 포함) |",
        "| epochs | 200 (early stopping, patience=20) |",
        "| lr | 1e-3 |",
        "| weight_decay | 1e-2 |",
        "| bottleneck | 64 |",
        "| dropout | 0.5 |",
        "| loss | MSE (mse_loss) |",
        "| clip_grad_norm | 없음 |",
        "| post-hoc scaling | α=warm_pred_mean/cold_pred_mean (~0.135) |",
        "| seed | 42 |",
        "",
        "### Competitor 예측값 경로",
        "",
        f"- lightgbm: `{b['lgbm_path']}`",
        f"- knn:      `{b['knn_path']}`",
        "- 평가 기간 필터: 16주 완전한 주(iso_year, iso_week 기준)",
        "",
        "---",
        "",
        "## 파트 A: Raw vs Residual 4가지 변형 비교",
        "",
        "### A-1: 동일 평가 함수(evaluate_weekly), 동일 16주 기준 결과",
        "",
        "| 변형 | Embedding | 학습 코드 | Loss | MAE | WRMSSE | DirAcc |",
        "|------|-----------|---------|------|-----|--------|--------|",
    ]

    for key, ev in a["metrics"].items():
        emb = "raw" if "raw" in key else "residual"
        code = "G1 (MAE,500ep,wd=1e-4)" if "G1" in key else "E11 (MSE,200ep,wd=1e-2)"
        loss = "MAE" if "G1" in key else "MSE"
        lines.append(f"| {key} | {emb} | {code} | {loss} | "
                     f"{fmt(ev['mae'])} | {fmt(ev['wrmsse'])} | {fmt(ev['dir_acc'])} |")

    lines += [
        "",
        "### A-2: 예측값 분포",
        "",
        "**비교 기준:**",
        f"- y_cold actual: mean={a['distrib']['y_cold_actual']['mean']:.4f}, "
        f"std={a['distrib']['y_cold_actual']['std']:.4f}, "
        f"max={a['distrib']['y_cold_actual']['max']:.1f}",
        f"- y_warm actual: mean={a['distrib']['y_warm_actual']['mean']:.4f}, "
        f"std={a['distrib']['y_warm_actual']['std']:.4f}, "
        f"max={a['distrib']['y_warm_actual']['max']:.1f}",
        "",
        "| 변형 | pred mean | pred std | pred min | pred max |",
        "|------|-----------|---------|---------|---------|",
    ]
    for k in ["A-1a","A-1b","A-1c","A-1d"]:
        d = a["distrib"].get(k, {})
        lines.append(f"| {k} | {fmt(d.get('mean'))} | {fmt(d.get('std'))} | "
                     f"{fmt(d.get('min'))} | {fmt(d.get('max'))} |")

    lines += [
        "",
        "### A-3: 아이템별 scatter 데이터 → `experiments/exp020_error_analysis/pred_comparison.csv`",
        "",
        "상위 10개 (actual_mean 기준 내림차순):",
        "",
        "| item_id | actual_mean | pred_raw_G1 | pred_res_G1 | pred_raw_E11 | pred_res_E11 |",
        "|---------|------------|------------|------------|-------------|-------------|",
    ]
    top10 = a["scatter_df"].sort_values("actual_mean", ascending=False).head(10)
    for _, r in top10.iterrows():
        lines.append(f"| {r['item_id']} | {r['actual_mean']:.2f} | "
                     f"{r['pred_raw_G1_mean']:.2f} | {r['pred_res_G1_mean']:.2f} | "
                     f"{r['pred_raw_E11_mean']:.2f} | {r['pred_res_E11_mean']:.2f} |")

    lines += [
        "",
        "**시각화:** `experiments/exp020_error_analysis/figures/a_pred_distribution.png`",
        "",
        "---",
        "",
        "## 파트 B: 아이템별 성적표",
        "",
        "### B-1: Track B G1 전체 지표",
        "",
        f"| 지표 | Track B G1 |",
        "|------|-----------|",
        f"| MAE | {fmt(b['tb_eval']['mae'])} |",
        f"| RMSE | {fmt(b['tb_eval']['rmse'])} |",
        f"| WRMSSE | {fmt(b['tb_eval']['wrmsse'])} |",
        f"| DirAcc | {fmt(b['tb_eval']['dir_acc'])} |",
        "",
        "**아이템별 성적표 → `experiments/exp020_error_analysis/item_scorecard.csv`**",
        "",
        "상위 20개 고오차 아이템 (trackb_MAE 기준):",
        "",
        "| item_id | cat_id | tier | actual_mean | trackb_MAE | lgbm_MAE | knn_MAE | tb<lgbm |",
        "|---------|--------|------|------------|-----------|---------|--------|---------|",
    ]
    top20 = sc.nlargest(20, "trackb_MAE")
    for _, r in top20.iterrows():
        tb_win = str(r.get("trackb_wins_lgbm_WRMSSE","")) if not pd.isna(r.get("trackb_wins_lgbm_WRMSSE",float("nan"))) else "—"
        lines.append(f"| {r['item_id']} | {r['cat_id']} | {r['sales_tier']} | "
                     f"{fmt(r['actual_weekly_mean'],2)} | {fmt(r['trackb_MAE'],2)} | "
                     f"{fmt(r['lgbm_MAE'],2)} | {fmt(r['knn_MAE'],2)} | {tb_win} |")

    # TrackB wins 통계
    wins = sc["trackb_wins_lgbm_WRMSSE"].dropna()
    n_win  = int(wins.sum())
    n_lose = int((~wins.astype(bool)).sum())
    lines += [
        "",
        f"**Track B WRMSSE < lightgbm WRMSSE:** {n_win}/{len(wins)} 아이템",
        f"**Track B WRMSSE ≥ lightgbm WRMSSE:** {n_lose}/{len(wins)} 아이템",
        "",
        "### B-2: 부서별 통계 (CA_1, warm_train 기준)",
        "",
        "| dept_id | n_items | daily_mean | nonzero_ratio | sales_cv | avg_price | cold_count |",
        "|---------|---------|-----------|--------------|---------|-----------|------------|",
    ]
    for _, r in b["dept_summary"].sort_values("dept_id").iterrows():
        lines.append(f"| {r['dept_id']} | {int(r['n_items'])} | "
                     f"{fmt(r['daily_mean_sales'],3)} | "
                     f"{fmt(r['nonzero_day_ratio'],3)} | "
                     f"{fmt(r['sales_cv'],2)} | "
                     f"{fmt(r.get('avg_price', float('nan')),2)} | "
                     f"{int(r['cold_count'])} |")

    lines += [
        "",
        "### B-3: M5 부서 상품군 정보",
        "",
        "M5 Walmart 데이터셋은 상품군 정보가 공개되지 않음. 부서명은 익명화됨.",
        "Kaggle 대회 설명 및 원본 논문(Makridakis et al. 2022)에 따르면:",
        "- FOODS_1, FOODS_2, FOODS_3: 식품류 3개 부서 (세부 품목 불공개)",
        "- HOBBIES_1, HOBBIES_2: 취미/스포츠용품류 2개 부서",
        "- HOUSEHOLD_1, HOUSEHOLD_2: 가정용품류 2개 부서",
        "항목 수준의 상품명 정보는 데이터셋에 포함되지 않음.",
        "",
        "---",
        "",
        "## 파트 C: 맥락 검증 자료",
        "",
        "### C-1: evaluate_weekly 함수 소스코드",
        "",
        eval_src,
        "",
        "### C-3: exp018 MAE=8.39 vs exp019 MAE=9.67 차이 원인",
        "",
        c3["note"],
        "",
        "| 설정 | MAE |",
        "|------|-----|",
        "| exp018 보고값 (residual,17주,bn=128,λ_dir=0.5,T=2.0,3-seed평균) | 8.3860 |",
        "| exp019 supplement (동일 설정,16주,seed 0-4,5개 평균) | 9.6651 |",
        f"| 이번 재계산 G1 residual,16주,seed=42 | {fmt(c3['g1_res_16w']['mae'])} |",
        f"| 이번 재계산 G1 residual,17주,seed=42 | {fmt(c3['g1_res_17w']['mae'])} |",
        "",
        "차이 요인:",
        "1. **평가 기간**: 17주(부분 주 포함) vs 16주(완전한 주만) — seed=42 기준 "
        f"Δ={abs(c3['g1_res_17w']['mae']-c3['g1_res_16w']['mae']):.4f}",
        "2. **bottleneck 크기**: exp018 best=128, G1=64 — 모델 용량 차이",
        "3. **lambda_dir**: exp018 best=0.5, G1=0 — 방향성 loss 추가 여부",
        "4. **T (온도 스케일링)**: exp018 best=2.0, G1=없음",
        "5. **seed 분산**: exp019 supplement에서 seed=4가 MAE=11.29로 이상치 발생",
        "",
        "---",
        "",
        "**시각화:** `experiments/exp020_error_analysis/figures/`",
        "**스크립트:** `scripts/exp020_error_analysis.py`",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", REPORT)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Exp020: 판단 재료 수집 ===")
    data = load_data()

    a_result = part_a(data)
    b_result = part_b(data)
    c3_result = part_c_mae_discrepancy(data)

    write_report(a_result, b_result, c3_result)

    print("\n" + "="*70)
    print("[Exp020 요약]")
    print("=== Part A ===")
    for key, ev in a_result["metrics"].items():
        print(f"  {key}: MAE={ev['mae']:.4f}  WRMSSE={ev['wrmsse']:.4f}  "
              f"DirAcc={ev['dir_acc']:.4f}")
    print("=== Part B ===")
    print(f"  Track B G1: MAE={b_result['tb_eval']['mae']:.4f}  "
          f"WRMSSE={b_result['tb_eval']['wrmsse']:.4f}")
    sc = b_result["scorecard"]
    wins = sc["trackb_wins_lgbm_WRMSSE"].dropna()
    print(f"  Track B < lgbm (WRMSSE): {int(wins.sum())}/{len(wins)} items")
    print("=== Part C-3 ===")
    print(c3_result["note"])
    print("="*70)


if __name__ == "__main__":
    main()
