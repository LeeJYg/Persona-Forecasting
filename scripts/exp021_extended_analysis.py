"""exp021: 예측값 검증 + 선택품/필수품 확장 분석 + 복수 Cold Set 타당성.

파트 0: exp011 원본 head raw 예측값 확인
파트 1: Threshold Sensitivity (선택품/필수품 기준)
파트 2: HOBBIES_1 심층 분석
파트 3: 아이템 세부 정보 (주별 시퀀스)
파트 4: 복수 Cold Set 타당성 보고
파트 5: Low 티어 아이템 특성

출력:
    docs/diagnosis/extended_analysis_materials.md
    scripts/exp021_extended_analysis.py
"""
from __future__ import annotations

import logging
import random
import sys
import warnings
from pathlib import Path

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

from src.evaluation.metrics import _compute_category_scales

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EMB_DIR  = ROOT / "experiments/exp011_v3_pipeline/embeddings"
COMP_DIR = ROOT / "experiments/exp006_competitors"
CS_DIR   = ROOT / "data/processed/cold_start"
M5_DIR   = ROOT / "m5-forecasting-accuracy"
SC_PATH  = ROOT / "experiments/exp020_error_analysis/item_scorecard.csv"
OUT_DIR  = ROOT / "experiments/exp021_extended_analysis"
FIG_DIR  = OUT_DIR / "figures"
REPORT   = ROOT / "docs/diagnosis/extended_analysis_materials.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── 모델 정의 ────────────────────────────────────────────────────────────────

class AttnBottleneckE11(nn.Module):
    """exp011 3-4_attn_bottleneck: dropout=0.5, n_weeks=17."""
    def __init__(self, hidden=5120, bottleneck=64, n_weeks=17, dropout=0.5):
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


class AttnBottleneckG1(nn.Module):
    """exp016 G1: dropout=0.1, n_weeks=16."""
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


def train_g1(X, y, seed=42):
    torch.manual_seed(seed)
    m = AttnBottleneckG1(hidden=X.shape[2], n_weeks=y.shape[1])
    opt = Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(500):
        m.train(); opt.zero_grad()
        loss = F.l1_loss(m(Xt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    return m


def predict_np(m, X):
    with torch.no_grad():
        p = m(torch.tensor(X, dtype=torch.float32)).numpy()
    return np.clip(p, 0, None)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data():
    logger.info("데이터 로드 중...")
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", map_location="cpu",
                          weights_only=False).numpy()
    cold_raw = torch.load(EMB_DIR / "cold_raw.pt", map_location="cpu",
                          weights_only=False).numpy()
    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)

    item_meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_ids = item_meta[item_meta["is_cold"] == True]["item_id"].tolist()
    warm_ids = item_meta[item_meta["is_cold"] == False]["item_id"].tolist()

    cold_test_raw  = pd.read_csv(CS_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_train_raw = pd.read_csv(CS_DIR / "warm_train.csv", parse_dates=["date"])
    warm_test_raw  = pd.read_csv(CS_DIR / "warm_test.csv",  parse_dates=["date"])

    def to_weekly(df):
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

    # 16주 (완전한 주)
    day_cnt = (cold_test_raw.assign(
        iso_year=cold_test_raw["date"].dt.isocalendar().year.astype(int),
        iso_week=cold_test_raw["date"].dt.isocalendar().week.astype(int))
        .groupby(["iso_year","iso_week"])["date"].nunique())
    complete_set = set(zip(day_cnt[day_cnt==7].index.get_level_values(0),
                           day_cnt[day_cnt==7].index.get_level_values(1)))
    cold_weekly_16 = cold_weekly[cold_weekly.apply(
        lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)].copy()
    wl16 = sorted(complete_set)
    wk2date16 = {}
    for _, r in cold_weekly_16.iterrows():
        wk = (r["iso_year"], r["iso_week"])
        if wk not in wk2date16:
            wk2date16[wk] = r["date"]
    wd16 = [wk2date16.get(wk, pd.Timestamp("2016-01-01")) for wk in wl16]

    # 17주 (모든 주)
    all_wks = set(zip(cold_weekly["iso_year"], cold_weekly["iso_week"]))
    wl17 = sorted(all_wks)
    wk2date17 = {}
    for _, r in cold_weekly.iterrows():
        wk = (r["iso_year"], r["iso_week"])
        if wk not in wk2date17:
            wk2date17[wk] = r["date"]
    wd17 = [wk2date17.get(wk, pd.Timestamp("2016-01-01")) for wk in wl17]

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
    y_warm_17 = build_y(warm_ids, warm_test_w, wl17)
    y_cold_17 = build_y(cold_ids, cold_weekly, wl17)

    cold_meta = (cold_test_raw[["item_id","cat_id","dept_id"]]
                 .drop_duplicates("item_id").set_index("item_id"))

    logger.info("  cold_raw=%s warm_raw=%s", cold_raw.shape, warm_raw.shape)
    logger.info("  wl16=%d weeks  wl17=%d weeks", len(wl16), len(wl17))

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
        "week_list_17": wl17, "week_dates_17": wd17,
        "cold_meta": cold_meta,
    }


def preds_to_weekly_dict(preds, cold_ids, week_list):
    """(N, T) → {item_id: {(yr,wk): pred}}"""
    result = {}
    for ii, iid in enumerate(cold_ids):
        result[iid] = {wk: float(preds[ii, wi]) for wi, wk in enumerate(week_list)}
    return result


def load_comp_weekly(model_name, wl_set):
    p = COMP_DIR / model_name / "predictions" / f"{model_name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    df = df[df.apply(lambda r: (r["iso_year"], r["iso_week"]) in wl_set, axis=1)]
    return df


def item_level_metrics(pred_df, cold_weekly, warm_train, cold_ids):
    """아이템별 WRMSSE, DirAcc 계산."""
    cat_scales = _compute_category_scales(warm_train, lag=4)
    merged = cold_weekly.merge(
        pred_df[["item_id","iso_year","iso_week","pred_sales"]],
        on=["item_id","iso_year","iso_week"], how="inner")
    rows = {}
    for iid, grp in merged.groupby("item_id"):
        cat = grp["cat_id"].iloc[0]
        actual, pred = grp["sales"].values, grp["pred_sales"].values
        mae_v = float(np.abs(actual - pred).mean())
        scale = cat_scales.get(str(cat), 1.0) or 1.0
        wrmsse_v = float(np.sqrt(np.mean((actual - pred) ** 2) / scale))
        sub = grp.sort_values(["iso_year","iso_week"])
        ad = np.sign(np.diff(sub["sales"].values))
        pd_ = np.sign(np.diff(sub["pred_sales"].values))
        da_v = float((ad == pd_).mean()) if len(ad) > 0 else float("nan")
        m = ad != 0
        da_nf = float((ad[m] == pd_[m]).mean()) if m.sum() > 0 else float("nan")
        rows[iid] = {"mae": mae_v, "wrmsse": wrmsse_v,
                     "dir_acc": da_v, "dir_acc_noflat": da_nf,
                     "pred_mean": float(pred.mean())}
    return rows


# ─── Part 0 ───────────────────────────────────────────────────────────────────

def part0(data):
    logger.info("=== Part 0: exp011 raw 예측값 확인 ===")
    cold_ids = data["cold_ids"]
    cold_raw = data["cold_raw"]
    y_cold_17 = data["y_cold_17"]
    y_cold_16 = data["y_cold_16"]
    wl16 = data["week_list_16"]
    wl17 = data["week_list_17"]

    # 0-2: exp011 모델 로드 + forward pass
    sd = torch.load(ROOT / "experiments/exp011_v3_pipeline/models/attn_bottleneck/model.pt",
                    map_location="cpu", weights_only=False)
    m11 = AttnBottleneckE11(hidden=5120, bottleneck=64, n_weeks=17, dropout=0.5)
    m11.load_state_dict(sd)
    m11.eval()

    with torch.no_grad():
        raw11 = m11(torch.tensor(cold_raw, dtype=torch.float32)).numpy()  # (100, 17)
    logger.info("  exp011 raw shape: %s", raw11.shape)

    # 0-3: 예측값 분포
    dist11 = {"mean": float(raw11.mean()), "std": float(raw11.std()),
               "min": float(raw11.min()), "max": float(raw11.max())}
    logger.info("  exp011 raw: mean=%.4f std=%.4f min=%.4f max=%.4f",
                dist11["mean"], dist11["std"], dist11["min"], dist11["max"])

    # 0-4: element-wise MAE vs y_cold_17
    mae11_raw = float(np.abs(raw11 - y_cold_17).mean())
    logger.info("  exp011 raw MAE vs y_cold_17: %.4f  (보고값: ~67)", mae11_raw)

    # 0-5: exp016 G1 학습 + raw 예측값
    logger.info("  G1 학습 중...")
    m_g1 = train_g1(data["warm_residual"], data["y_warm_16"])
    raw_g1 = predict_np(m_g1, data["cold_residual"])  # (100, 16)
    dist_g1 = {"mean": float(raw_g1.mean()), "std": float(raw_g1.std()),
                "min": float(raw_g1.min()), "max": float(raw_g1.max())}
    # 16주 기준 MAE
    mae_g1_raw = float(np.abs(raw_g1 - y_cold_16).mean())
    logger.info("  G1 raw: mean=%.4f std=%.4f  MAE vs y_cold_16=%.4f",
                dist_g1["mean"], dist_g1["std"], mae_g1_raw)

    # 16주 기준으로 exp011 raw도 잘라서 비교
    idx_16in17 = [i for i, wk in enumerate(wl17) if wk in set(wl16)]
    raw11_16 = raw11[:, idx_16in17]
    mae11_16 = float(np.abs(raw11_16 - y_cold_16).mean())
    logger.info("  exp011 raw (16주 기준): MAE=%.4f", mae11_16)

    # 0-6: 고판매 5개 + 저판매 5개 시퀀스 (actual vs exp011 vs G1, 16주)
    actual_means = y_cold_16.mean(axis=1)
    sorted_idx   = np.argsort(actual_means)
    low5_idx  = sorted_idx[:5].tolist()
    high5_idx = sorted_idx[-5:].tolist()
    sample_idx = high5_idx + low5_idx

    sequences = []
    for ci in sample_idx:
        iid = cold_ids[ci]
        for wi, wk in enumerate(wl16):
            sequences.append({
                "item_id": iid,
                "group": "high" if ci in high5_idx else "low",
                "actual_mean_cat": float(actual_means[ci]),
                "week_idx": wi + 1,
                "iso_year": wk[0], "iso_week": wk[1],
                "actual": float(y_cold_16[ci, wi]),
                "exp011_pred": float(raw11_16[ci, wi]),
                "g1_pred": float(raw_g1[ci, wi]),
            })
    seq_df = pd.DataFrame(sequences)

    # 0-7: scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    actual_item_mean = y_cold_16.mean(axis=1)
    e11_item_mean    = raw11_16.mean(axis=1)
    g1_item_mean     = raw_g1.mean(axis=1)
    ax.scatter(actual_item_mean, e11_item_mean, alpha=0.6, s=25,
               color="steelblue", label="exp011 raw")
    ax.scatter(actual_item_mean, g1_item_mean, alpha=0.6, s=25,
               color="tomato", label="G1 (residual,16주)")
    mx = max(actual_item_mean.max(), e11_item_mean.max(), g1_item_mean.max()) * 1.1
    ax.plot([0, mx], [0, mx], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("actual_weekly_mean"); ax.set_ylabel("predicted_weekly_mean")
    ax.set_title("Actual vs Predicted (아이템 평균, 16주)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p0_actual_vs_pred_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("  scatter plot 저장")

    return {
        "dist11": dist11, "dist_g1": dist_g1,
        "mae11_raw_17w": mae11_raw, "mae11_raw_16w": mae11_16,
        "mae_g1_raw_16w": mae_g1_raw,
        "seq_df": seq_df,
        "raw11_16": raw11_16, "raw_g1": raw_g1,
        "m_g1": m_g1,
    }


# ─── Part 1 ───────────────────────────────────────────────────────────────────

def part1(sc):
    logger.info("=== Part 1: Threshold Sensitivity ===")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rows_lgbm, rows_knn = [], []
    for thr in thresholds:
        optional = sc[sc["actual_nonzero_week_ratio"] < thr]
        essential = sc[sc["actual_nonzero_week_ratio"] >= thr]

        def stats(grp, comp_wrmsse_col, comp_dir_col):
            tb_w  = grp["trackb_WRMSSE"].dropna()
            co_w  = grp[comp_wrmsse_col].dropna()
            tb_d  = grp["trackb_DirAcc"].dropna()
            co_d  = grp[comp_dir_col].dropna()
            both  = grp[[comp_wrmsse_col, "trackb_WRMSSE"]].dropna()
            win_n = int((both["trackb_WRMSSE"] < both[comp_wrmsse_col]).sum())
            return {
                "n": len(grp),
                "tb_wrmsse": float(tb_w.mean()) if len(tb_w) else float("nan"),
                "co_wrmsse": float(co_w.mean()) if len(co_w) else float("nan"),
                "tb_win_pct": win_n / len(both) if len(both) else float("nan"),
                "tb_diracc": float(tb_d.mean()) if len(tb_d) else float("nan"),
                "co_diracc": float(co_d.mean()) if len(co_d) else float("nan"),
            }

        o_lg = stats(optional, "lgbm_WRMSSE", "lgbm_DirAcc")
        e_lg = stats(essential, "lgbm_WRMSSE", "lgbm_DirAcc")
        o_kn = stats(optional, "knn_WRMSSE", "knn_DirAcc")
        e_kn = stats(essential, "knn_WRMSSE", "knn_DirAcc")

        rows_lgbm.append({"threshold": thr,
                           "opt_n": o_lg["n"], "ess_n": e_lg["n"],
                           "opt_tb_wrmsse": o_lg["tb_wrmsse"], "opt_lgbm_wrmsse": o_lg["co_wrmsse"],
                           "opt_tb_win": o_lg["tb_win_pct"],
                           "opt_tb_diracc": o_lg["tb_diracc"], "opt_lgbm_diracc": o_lg["co_diracc"],
                           "ess_tb_wrmsse": e_lg["tb_wrmsse"], "ess_lgbm_wrmsse": e_lg["co_wrmsse"],
                           "ess_tb_win": e_lg["tb_win_pct"],
                           "ess_tb_diracc": e_lg["tb_diracc"], "ess_lgbm_diracc": e_lg["co_diracc"]})
        rows_knn.append({"threshold": thr,
                          "opt_n": o_kn["n"], "ess_n": e_kn["n"],
                          "opt_tb_wrmsse": o_kn["tb_wrmsse"], "opt_knn_wrmsse": o_kn["co_wrmsse"],
                          "opt_tb_win": o_kn["tb_win_pct"],
                          "opt_tb_diracc": o_kn["tb_diracc"], "opt_knn_diracc": o_kn["co_diracc"],
                          "ess_tb_wrmsse": e_kn["tb_wrmsse"], "ess_knn_wrmsse": e_kn["co_wrmsse"],
                          "ess_tb_win": e_kn["tb_win_pct"],
                          "ess_tb_diracc": e_kn["tb_diracc"], "ess_knn_diracc": e_kn["co_diracc"]})

    return {"lgbm": pd.DataFrame(rows_lgbm), "knn": pd.DataFrame(rows_knn)}


# ─── Part 2 ───────────────────────────────────────────────────────────────────

def part2(sc):
    logger.info("=== Part 2: HOBBIES_1 심층 분석 ===")
    h1 = sc[sc["dept_id"] == "HOBBIES_1"].copy()
    h1["tb_wins"] = h1["trackb_wins_lgbm_WRMSSE"]

    win_grp  = h1[h1["tb_wins"] == True]
    lose_grp = h1[h1["tb_wins"] == False]

    def grp_stats(grp, label):
        return {
            "group": label, "n": len(grp),
            "avg_price": float(grp["avg_price"].mean()),
            "actual_weekly_mean": float(grp["actual_weekly_mean"].mean()),
            "nonzero_week_ratio": float(grp["actual_nonzero_week_ratio"].mean()),
            "actual_cv": float(grp["actual_cv"].mean()),
            "mean_wrmsse_tb": float(grp["trackb_WRMSSE"].mean()),
            "mean_wrmsse_lgbm": float(grp["lgbm_WRMSSE"].mean()),
            "mean_diracc_tb": float(grp["trackb_DirAcc"].mean()),
            "mean_diracc_lgbm": float(grp["lgbm_DirAcc"].mean()),
        }

    comparison = [grp_stats(win_grp, "TB wins"), grp_stats(lose_grp, "TB loses")]

    # nonzero_week_ratio 분포
    h1_nonzero_dist = {
        "mean": float(h1["actual_nonzero_week_ratio"].mean()),
        "std":  float(h1["actual_nonzero_week_ratio"].std()),
        "min":  float(h1["actual_nonzero_week_ratio"].min()),
        "max":  float(h1["actual_nonzero_week_ratio"].max()),
        "ge_0.9_count": int((h1["actual_nonzero_week_ratio"] >= 0.9).sum()),
        "lt_0.5_count": int((h1["actual_nonzero_week_ratio"] < 0.5).sum()),
    }

    logger.info("  HOBBIES_1 n=%d  wins=%d  loses=%d",
                len(h1), len(win_grp), len(lose_grp))
    return {
        "h1_df": h1, "comparison": pd.DataFrame(comparison),
        "nonzero_dist": h1_nonzero_dist
    }


# ─── Part 3 ───────────────────────────────────────────────────────────────────

def part3(sc, data, raw_g1, lgbm_df, knn_df):
    logger.info("=== Part 3: 아이템 세부 정보 ===")
    cold_ids = data["cold_ids"]
    cold_weekly = data["cold_weekly_16"]
    wl16 = data["week_list_16"]
    wl16_set = set(wl16)

    # G1 예측 weekly dict
    g1_dict = preds_to_weekly_dict(raw_g1, cold_ids, wl16)

    # lgbm / knn weekly dict
    def comp_dict(df):
        if df is None:
            return {}
        d = {}
        for _, r in df.iterrows():
            wk = (int(r["iso_year"]), int(r["iso_week"]))
            if wk not in wl16_set:
                continue
            iid = r["item_id"]
            if iid not in d:
                d[iid] = {}
            d[iid][wk] = float(r["pred_sales"])
        return d

    lgbm_dict = comp_dict(lgbm_df)
    knn_dict  = comp_dict(knn_df)

    # 가격 변동 (sell_prices)
    prices_df = pd.read_csv(M5_DIR / "sell_prices.csv")
    price_stats = (prices_df[prices_df["store_id"]=="CA_1"]
                   .groupby("item_id")["sell_price"]
                   .agg(avg_price="mean", price_std="std").reset_index()
                   .set_index("item_id"))

    # 실제 주별 판매량 dict
    actual_dict = {}
    for _, r in cold_weekly.iterrows():
        iid = r["item_id"]
        wk  = (int(r["iso_year"]), int(r["iso_week"]))
        if iid not in actual_dict:
            actual_dict[iid] = {}
        actual_dict[iid][wk] = float(r["sales"])

    def item_detail(iid):
        actual_seq = [actual_dict.get(iid, {}).get(wk, 0.0) for wk in wl16]
        g1_seq     = [g1_dict.get(iid, {}).get(wk, 0.0)     for wk in wl16]
        lgbm_seq   = [lgbm_dict.get(iid, {}).get(wk, 0.0)   for wk in wl16]
        knn_seq    = [knn_dict.get(iid, {}).get(wk, 0.0)    for wk in wl16]
        act_arr    = np.array(actual_seq)
        return {
            "item_id": iid,
            "dept_id": data["cold_meta"].loc[iid, "dept_id"] if iid in data["cold_meta"].index else "",
            "cat_id":  data["cold_meta"].loc[iid, "cat_id"]  if iid in data["cold_meta"].index else "",
            "avg_price": float(price_stats.loc[iid, "avg_price"]) if iid in price_stats.index else float("nan"),
            "price_std": float(price_stats.loc[iid, "price_std"]) if iid in price_stats.index else float("nan"),
            "actual_weekly_mean": float(act_arr.mean()),
            "actual_weekly_std":  float(act_arr.std()),
            "nonzero_week_ratio": float((act_arr > 0).mean()),
            "actual_seq":   [round(v, 2) for v in actual_seq],
            "g1_seq":       [round(v, 2) for v in g1_seq],
            "lgbm_seq":     [round(v, 2) for v in lgbm_seq],
            "knn_seq":      [round(v, 2) for v in knn_seq],
            "max_sales_week": int(np.argmax(act_arr)) + 1,
            "min_sales_week": int(np.argmin(act_arr)) + 1,
            "zero_weeks":     int((act_arr == 0).sum()),
        }

    rng = random.Random(42)

    # 선택품 (nonzero < 0.5), 10개
    optional_ids = sc[sc["actual_nonzero_week_ratio"] < 0.5]["item_id"].tolist()
    sample_opt = rng.sample(optional_ids, min(10, len(optional_ids)))
    opt_details = [item_detail(iid) for iid in sample_opt]
    logger.info("  선택품 %d개 샘플", len(opt_details))

    # 필수품 (nonzero >= 0.9), 10개
    essential_ids = sc[sc["actual_nonzero_week_ratio"] >= 0.9]["item_id"].tolist()
    sample_ess = rng.sample(essential_ids, min(10, len(essential_ids)))
    ess_details = [item_detail(iid) for iid in sample_ess]
    logger.info("  필수품 %d개 샘플", len(ess_details))

    # HOBBIES_1 패배 아이템, 5개
    h1_lose_ids = sc[(sc["dept_id"]=="HOBBIES_1") &
                      (sc["trackb_wins_lgbm_WRMSSE"]==False)]["item_id"].tolist()
    sample_h1_lose = rng.sample(h1_lose_ids, min(5, len(h1_lose_ids)))
    h1_lose_details = [item_detail(iid) for iid in sample_h1_lose]
    logger.info("  HOBBIES_1 패배 %d개 샘플", len(h1_lose_details))

    return {
        "optional": opt_details,
        "essential": ess_details,
        "h1_lose": h1_lose_details,
    }


# ─── Part 4 ───────────────────────────────────────────────────────────────────

def part4():
    logger.info("=== Part 4: 복수 Cold Set 타당성 ===")
    # CA_1 전체 아이템 수
    sales = pd.read_csv(M5_DIR / "sales_train_evaluation.csv")
    ca1_items = sales[sales["store_id"]=="CA_1"]["item_id"].nunique()
    current_cold = 100
    current_warm = 300
    remaining = ca1_items - current_cold - current_warm

    # exp011 embedding 속도: 20,000 텍스트 ≈ 20시간
    # → 1,000 텍스트/시간 = ~14.4분 / 200 텍스트
    texts_per_item = 50  # 50 personas
    hours_per_10k_texts = 10  # exp011 로그에서 15,000 텍스트 ≈ ~14시간
    secs_per_text = (hours_per_10k_texts * 3600) / 10000

    def estimate_hours(n_items):
        n_texts = n_items * texts_per_item
        return n_texts * secs_per_text / 3600

    return {
        "ca1_total_items": ca1_items,
        "current_cold": current_cold,
        "current_warm": current_warm,
        "remaining_ca1": remaining,
        "texts_per_item": texts_per_item,
        "est_hours_200": estimate_hours(200),
        "est_hours_1000": estimate_hours(1000),
        "lgbm_no_embedding": True,
        "note": (
            f"CA_1 전체 {ca1_items}개 아이템 중 현재 사용 {current_cold+current_warm}개. "
            f"나머지 {remaining}개 중 추가 cold 추출 가능. "
            f"200개 추가 cold = {texts_per_item*200:,}개 텍스트 ≈ {estimate_hours(200):.1f}시간. "
            f"seed 5개(총 1,000개) = {texts_per_item*1000:,}개 텍스트 ≈ {estimate_hours(1000):.1f}시간. "
            f"LightGBM은 embedding 불필요 → 추가 cold에 대한 LightGBM 비교 예측은 즉시 생성 가능."
        )
    }


# ─── Part 5 ───────────────────────────────────────────────────────────────────

def part5(sc):
    logger.info("=== Part 5: Low 티어 아이템 ===")
    low = sc[sc["sales_tier"] == "Low"].copy()

    overlap_05 = int((low["actual_nonzero_week_ratio"] < 0.5).sum())
    overlap_09 = int((low["actual_nonzero_week_ratio"] >= 0.9).sum())

    logger.info("  Low tier n=%d  nonzero<0.5: %d  nonzero>=0.9: %d",
                len(low), overlap_05, overlap_09)
    return {
        "low_df": low,
        "overlap_05": overlap_05,
        "overlap_09": overlap_09,
    }


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def fmt(v, d=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{float(v):.{d}f}"


def write_detail_block(detail, lines):
    lines += [
        f"#### {detail['item_id']}",
        f"- dept: {detail['dept_id']}  cat: {detail['cat_id']}",
        f"- avg_price: {fmt(detail['avg_price'],2)}  price_std: {fmt(detail['price_std'],2)}",
        f"- actual_weekly_mean: {fmt(detail['actual_weekly_mean'],2)}  "
        f"actual_weekly_std: {fmt(detail['actual_weekly_std'],2)}",
        f"- nonzero_week_ratio: {fmt(detail['nonzero_week_ratio'],3)}",
        f"- max_sales_week: {detail['max_sales_week']}  "
        f"min_sales_week: {detail['min_sales_week']}  "
        f"zero_weeks: {detail['zero_weeks']}/16",
        f"- actual:  {detail['actual_seq']}",
        f"- G1 pred: {detail['g1_seq']}",
        f"- lgbm:    {detail['lgbm_seq']}",
        "",
    ]


def write_report(p0, p1, p2, p3, p4, p5):
    logger.info("보고서 작성 중...")
    lines = [
        "# 예측값 검증 + 선택품/필수품 확장 분석 보고서 (Exp021)",
        "",
        "**작성일:** 2026-03-11",
        "**목적:** 수치와 사실만 보고. 판정 없음.",
        "**Track B 기준선:** exp016 G1 (residual, 16주, MAE loss, 500ep, bn=64, seed=42)",
        "",
        "---",
        "",
        "## 파트 0: exp011 원본 head raw 예측값 확인",
        "",
        "### 0-1~0-2: 모델 및 forward pass",
        "- 체크포인트: `experiments/exp011_v3_pipeline/models/attn_bottleneck/model.pt`",
        "- 구조: hidden=5120, bottleneck=64, n_weeks=17, dropout=0.5",
        "- 입력: `cold_raw.pt` (100×50×5120)",
        "- 출력 shape: (100, 17)",
        "",
        "### 0-3: 예측값 분포",
        "",
        "| 모델 | mean | std | min | max |",
        "|------|------|-----|-----|-----|",
        f"| y_cold_actual (16주) | — | — | — | — |",
        f"| exp011 raw (17주 출력) | {fmt(p0['dist11']['mean'])} | "
        f"{fmt(p0['dist11']['std'])} | {fmt(p0['dist11']['min'])} | "
        f"{fmt(p0['dist11']['max'])} |",
        f"| G1 raw (16주 출력) | {fmt(p0['dist_g1']['mean'])} | "
        f"{fmt(p0['dist_g1']['std'])} | {fmt(p0['dist_g1']['min'])} | "
        f"{fmt(p0['dist_g1']['max'])} |",
        "",
        "### 0-4~0-5: element-wise MAE",
        "",
        "| 모델 | 평가 기간 | MAE vs y_cold |",
        "|------|---------|--------------|",
        f"| exp011 raw | 17주 | {fmt(p0['mae11_raw_17w'])} |",
        f"| exp011 raw | 16주 (17주에서 추출) | {fmt(p0['mae11_raw_16w'])} |",
        f"| G1 raw | 16주 | {fmt(p0['mae_g1_raw_16w'])} |",
        "",
        "### 0-6: 고판매/저판매 아이템 주별 시퀀스 (16주)",
        "",
    ]

    seq_df = p0["seq_df"]
    for iid in seq_df["item_id"].unique():
        sub = seq_df[seq_df["item_id"] == iid].sort_values("week_idx")
        grp = sub["group"].iloc[0]
        act_mean = sub["actual_mean_cat"].iloc[0]
        lines += [
            f"**{iid}** (group={grp}, actual_mean={act_mean:.2f})",
            "",
            "| week | actual | exp011_pred | g1_pred |",
            "|------|--------|------------|---------|",
        ]
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['week_idx'])} | {r['actual']:.2f} | "
                         f"{r['exp011_pred']:.2f} | {r['g1_pred']:.2f} |")
        lines.append("")

    lines += [
        "**시각화:** `experiments/exp021_extended_analysis/figures/p0_actual_vs_pred_scatter.png`",
        "",
        "---",
        "",
        "## 파트 1: Threshold Sensitivity",
        "",
        "### 1-1: Track B vs LightGBM",
        "",
        "| thr | 선택품n | 필수품n | 선택품 TB_WRMSSE | 선택품 LGB_WRMSSE | 선택품 TB승률 | 선택품 TB_DA | 선택품 LGB_DA | 필수품 TB_WRMSSE | 필수품 LGB_WRMSSE | 필수품 TB승률 | 필수품 TB_DA | 필수품 LGB_DA |",
        "|-----|---------|---------|----------------|------------------|------------|------------|------------|----------------|------------------|------------|------------|------------|",
    ]
    for _, r in p1["lgbm"].iterrows():
        lines.append(
            f"| {r['threshold']} | {int(r['opt_n'])} | {int(r['ess_n'])} | "
            f"{fmt(r['opt_tb_wrmsse'],3)} | {fmt(r['opt_lgbm_wrmsse'],3)} | "
            f"{fmt(r['opt_tb_win'],3)} | {fmt(r['opt_tb_diracc'],3)} | {fmt(r['opt_lgbm_diracc'],3)} | "
            f"{fmt(r['ess_tb_wrmsse'],3)} | {fmt(r['ess_lgbm_wrmsse'],3)} | "
            f"{fmt(r['ess_tb_win'],3)} | {fmt(r['ess_tb_diracc'],3)} | {fmt(r['ess_lgbm_diracc'],3)} |"
        )

    lines += [
        "",
        "### 1-2: Track B vs KNN",
        "",
        "| thr | 선택품n | 필수품n | 선택품 TB_WRMSSE | 선택품 KNN_WRMSSE | 선택품 TB승률 | 선택품 TB_DA | 선택품 KNN_DA | 필수품 TB_WRMSSE | 필수품 KNN_WRMSSE | 필수품 TB승률 | 필수품 TB_DA | 필수품 KNN_DA |",
        "|-----|---------|---------|----------------|------------------|------------|------------|------------|----------------|------------------|------------|------------|------------|",
    ]
    for _, r in p1["knn"].iterrows():
        lines.append(
            f"| {r['threshold']} | {int(r['opt_n'])} | {int(r['ess_n'])} | "
            f"{fmt(r['opt_tb_wrmsse'],3)} | {fmt(r['opt_knn_wrmsse'],3)} | "
            f"{fmt(r['opt_tb_win'],3)} | {fmt(r['opt_tb_diracc'],3)} | {fmt(r['opt_knn_diracc'],3)} | "
            f"{fmt(r['ess_tb_wrmsse'],3)} | {fmt(r['ess_knn_wrmsse'],3)} | "
            f"{fmt(r['ess_tb_win'],3)} | {fmt(r['ess_tb_diracc'],3)} | {fmt(r['ess_knn_diracc'],3)} |"
        )

    # Part 2
    h1 = p2["h1_df"]
    lines += [
        "",
        "---",
        "",
        "## 파트 2: HOBBIES_1 심층 분석",
        "",
        "### 2-1: HOBBIES_1 28개 전체 리스트",
        "",
        "| item_id | tier | avg_price | actual_mean | nonzero_ratio | actual_cv | "
        "TB_WRMSSE | lgbm_WRMSSE | TB_DA | lgbm_DA | TB_wins |",
        "|---------|------|-----------|------------|--------------|---------|"
        "---------|------------|------|--------|---------|",
    ]
    for _, r in h1.sort_values("trackb_WRMSSE", ascending=False).iterrows():
        lines.append(
            f"| {r['item_id']} | {r['sales_tier']} | {fmt(r['avg_price'],2)} | "
            f"{fmt(r['actual_weekly_mean'],2)} | {fmt(r['actual_nonzero_week_ratio'],3)} | "
            f"{fmt(r['actual_cv'],2)} | {fmt(r['trackb_WRMSSE'],3)} | "
            f"{fmt(r['lgbm_WRMSSE'],3)} | {fmt(r['trackb_DirAcc'],3)} | "
            f"{fmt(r['lgbm_DirAcc'],3)} | {r['tb_wins']} |"
        )

    comp = p2["comparison"]
    lines += [
        "",
        "### 2-2: TB wins vs TB loses 비교",
        "",
        "| 그룹 | n | avg_price | actual_mean | nonzero_ratio | actual_cv | TB_WRMSSE | LGBM_WRMSSE | TB_DA | LGBM_DA |",
        "|------|---|-----------|------------|--------------|---------|---------|-----------|------|--------|",
    ]
    for _, r in comp.iterrows():
        lines.append(
            f"| {r['group']} | {int(r['n'])} | {fmt(r['avg_price'],2)} | "
            f"{fmt(r['actual_weekly_mean'],2)} | {fmt(r['nonzero_week_ratio'],3)} | "
            f"{fmt(r['actual_cv'],2)} | {fmt(r['mean_wrmsse_tb'],3)} | "
            f"{fmt(r['mean_wrmsse_lgbm'],3)} | {fmt(r['mean_diracc_tb'],3)} | "
            f"{fmt(r['mean_diracc_lgbm'],3)} |"
        )

    nd = p2["nonzero_dist"]
    lines += [
        "",
        "### 2-3: HOBBIES_1 nonzero_week_ratio 분포",
        "",
        f"- mean={fmt(nd['mean'],3)}  std={fmt(nd['std'],3)}  "
        f"min={fmt(nd['min'],3)}  max={fmt(nd['max'],3)}",
        f"- nonzero_ratio ≥ 0.9: {nd['ge_0.9_count']}개 / 28개",
        f"- nonzero_ratio < 0.5: {nd['lt_0.5_count']}개 / 28개",
        "",
        "---",
        "",
        "## 파트 3: 아이템 세부 정보 (주별 시퀀스)",
        "",
        "### 3-1: 선택품 (nonzero < 0.5) 랜덤 10개",
        "",
    ]
    for d in p3["optional"]:
        write_detail_block(d, lines)

    lines += ["### 3-2: 필수품 (nonzero ≥ 0.9) 랜덤 10개", ""]
    for d in p3["essential"]:
        write_detail_block(d, lines)

    lines += ["### 3-3: HOBBIES_1 TB 패배 아이템 랜덤 5개", ""]
    for d in p3["h1_lose"]:
        write_detail_block(d, lines)

    lines += [
        "---",
        "",
        "## 파트 4: 복수 Cold Set 타당성",
        "",
        f"- CA_1 전체 아이템: {p4['ca1_total_items']}개",
        f"- 현재 사용: warm {p4['current_warm']}개 + cold {p4['current_cold']}개",
        f"- 잔여 후보: {p4['remaining_ca1']}개",
        "",
        "| 시나리오 | 추가 cold 수 | 텍스트 수 | 추정 소요 시간 |",
        "|---------|------------|---------|--------------|",
        f"| seed 1개 | 200개 | {200*p4['texts_per_item']:,} | "
        f"≈ {p4['est_hours_200']:.1f}시간 |",
        f"| seed 5개 | 1,000개 | {1000*p4['texts_per_item']:,} | "
        f"≈ {p4['est_hours_1000']:.1f}시간 |",
        "",
        f"- LightGBM: embedding 불필요 → 추가 cold 예측 즉시 생성 가능",
        f"- embedding 속도 기준: exp011에서 10,000 텍스트 ≈ 10시간 추정",
        f"- data leakage 방지: 기존 warm 300개는 cold 재분류 불가",
        "",
        "---",
        "",
        "## 파트 5: Low 티어 아이템",
        "",
        "### 5-1: Low 티어 24개 전체 리스트",
        "",
        "| item_id | dept_id | avg_price | actual_mean | nonzero_ratio | TB_WRMSSE | lgbm_WRMSSE | TB_DA | lgbm_DA |",
        "|---------|---------|-----------|------------|--------------|---------|------------|------|--------|",
    ]
    low = p5["low_df"].sort_values("trackb_WRMSSE", ascending=False)
    for _, r in low.iterrows():
        lines.append(
            f"| {r['item_id']} | {r['dept_id']} | {fmt(r['avg_price'],2)} | "
            f"{fmt(r['actual_weekly_mean'],2)} | {fmt(r['actual_nonzero_week_ratio'],3)} | "
            f"{fmt(r['trackb_WRMSSE'],3)} | {fmt(r['lgbm_WRMSSE'],3)} | "
            f"{fmt(r['trackb_DirAcc'],3)} | {fmt(r['lgbm_DirAcc'],3)} |"
        )

    lines += [
        "",
        "### 5-2: Low 티어와 선택품(nonzero<0.5)의 겹침",
        "",
        f"- Low 티어 24개 중 nonzero_week_ratio < 0.5: **{p5['overlap_05']}개**",
        f"- Low 티어 24개 중 nonzero_week_ratio ≥ 0.9: **{p5['overlap_09']}개**",
        "",
        "---",
        "",
        "**시각화:** `experiments/exp021_extended_analysis/figures/`",
        "**스크립트:** `scripts/exp021_extended_analysis.py`",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", REPORT)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Exp021 시작 ===")
    data = load_data()
    sc   = pd.read_csv(SC_PATH)

    # Part 0
    p0 = part0(data)
    raw_g1 = p0["raw_g1"]

    # Competitor 로드 (16주 필터)
    wl16_set = set(data["week_list_16"])
    lgbm_df = load_comp_weekly("lightgbm_proxy_lags", wl16_set)
    knn_df  = load_comp_weekly("knn_analog",          wl16_set)

    # Part 1
    p1 = part1(sc)

    # Part 2
    p2 = part2(sc)

    # Part 3
    p3 = part3(sc, data, raw_g1, lgbm_df, knn_df)

    # Part 4
    p4 = part4()

    # Part 5
    p5 = part5(sc)

    write_report(p0, p1, p2, p3, p4, p5)

    print("\n" + "="*70)
    print("[Exp021 요약]")
    print(f"Part 0 — exp011 raw MAE(17w)={p0['mae11_raw_17w']:.4f}  "
          f"G1 raw MAE(16w)={p0['mae_g1_raw_16w']:.4f}")
    print(f"Part 2 — HOBBIES_1: wins={p2['h1_df']['tb_wins'].sum()}, "
          f"loses={(~p2['h1_df']['tb_wins'].astype(bool)).sum()}")
    print(f"Part 5 — Low tier: nonzero<0.5={p5['overlap_05']}/24")
    print(f"보고서: {REPORT}")
    print("="*70)


if __name__ == "__main__":
    main()
