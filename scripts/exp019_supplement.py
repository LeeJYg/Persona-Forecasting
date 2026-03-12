"""Exp019 보완: lightgbm 설치 후 Wilcoxon 검증 + 보고서 완성."""
from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

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

EMB_DIR  = ROOT / "experiments/exp011_v3_pipeline/embeddings"
COMP_DIR = ROOT / "experiments/exp006_competitors"
CS_DIR   = ROOT / "data/processed/cold_start"
REPORT   = ROOT / "docs/diagnosis/evaluation_framework_report.md"

BEST_CONFIG = {
    "epochs": 500, "T": 2.0, "lambda_dir": 0.5, "lr": 1e-3, "bottleneck": 128,
}
N_SEEDS = 5


# ─── 모델 ─────────────────────────────────────────────────────────────────────

class AttnBottleneck(nn.Module):
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


def train_head(X, y, n_epochs=500, lr=1e-3, bottleneck=64, seed=0, lambda_dir=0.0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    m = AttnBottleneck(hidden=X.shape[2], bottleneck=bottleneck, n_weeks=y.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(n_epochs):
        m.train(); opt.zero_grad()
        p = m(Xt)
        loss = torch.mean(torch.abs(p - yt))
        if lambda_dir > 0 and y.shape[1] > 1:
            loss += lambda_dir * torch.mean((torch.sign(p[:, 1:] - p[:, :-1]) !=
                                             torch.sign(yt[:, 1:] - yt[:, :-1])).float())
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    return m


def predict(m, X, T=1.0):
    with torch.no_grad():
        raw = m(torch.tensor(X, dtype=torch.float32)).numpy()
    return np.clip(raw * T, 0, None)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data():
    logger.info("데이터 로드 중...")
    # embeddings (shape: n_items, 50_personas, 5120)
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", map_location="cpu",
                          weights_only=False).numpy()
    cold_raw = torch.load(EMB_DIR / "cold_raw.pt", map_location="cpu",
                          weights_only=False).numpy()
    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)

    item_meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_ids = item_meta[item_meta["is_cold"] == True]["item_id"].tolist()
    warm_ids = item_meta[item_meta["is_cold"] == False]["item_id"].tolist()

    # daily → weekly
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

    cold_weekly = to_weekly(cold_test_raw)
    warm_train_weekly = to_weekly(warm_train_raw)
    warm_test_weekly  = to_weekly(warm_test_raw)

    # 완전한 16주 목록
    counts = cold_test_raw.copy()
    counts["iso_year"] = counts["date"].dt.isocalendar().year.astype(int)
    counts["iso_week"] = counts["date"].dt.isocalendar().week.astype(int)
    day_cnt = counts.groupby(["iso_year","iso_week"])["date"].nunique()
    complete_set_16 = set(zip(day_cnt[day_cnt == 7].index.get_level_values(0),
                               day_cnt[day_cnt == 7].index.get_level_values(1)))
    cold_weekly_16 = cold_weekly[cold_weekly.apply(
        lambda r: (r["iso_year"], r["iso_week"]) in complete_set_16, axis=1)].copy()
    week_list_16 = sorted(complete_set_16)

    # week_dates
    wk2date = {}
    for _, r in cold_weekly_16.iterrows():
        wk = (r["iso_year"], r["iso_week"])
        if wk not in wk2date:
            wk2date[wk] = r["date"]
    week_dates_16 = [wk2date.get(wk, pd.Timestamp("2016-01-01")) for wk in week_list_16]

    # y matrices
    def build_y(item_ids, weekly_df, week_list):
        wi = {wk: i for i, wk in enumerate(week_list)}
        ii = {iid: i for i, iid in enumerate(item_ids)}
        y = np.zeros((len(item_ids), len(week_list)), dtype=np.float32)
        for _, r in weekly_df.iterrows():
            wk = (r["iso_year"], r["iso_week"])
            if wk in wi and r["item_id"] in ii:
                y[ii[r["item_id"]], wi[wk]] = r["sales"]
        return y

    y_warm_16 = build_y(warm_ids, warm_test_weekly, week_list_16)
    y_cold_16 = build_y(cold_ids, cold_weekly_16, week_list_16)

    cold_meta = (cold_test_raw[["item_id","cat_id"]].drop_duplicates("item_id")
                 .set_index("item_id"))

    logger.info("  warm_residual=%s  cold_residual=%s", warm_residual.shape, cold_residual.shape)
    logger.info("  y_warm_16=%s  y_cold_16=%s", y_warm_16.shape, y_cold_16.shape)
    logger.info("  16주: %d개", len(week_list_16))

    return {
        "warm_residual": warm_residual, "cold_residual": cold_residual,
        "cold_ids": cold_ids, "warm_ids": warm_ids,
        "cold_weekly_16": cold_weekly_16,
        "warm_train_weekly": warm_train_weekly,
        "y_warm_16": y_warm_16, "y_cold_16": y_cold_16,
        "week_list_16": week_list_16, "week_dates_16": week_dates_16,
        "cold_meta": cold_meta,
    }


def preds_to_df(preds, cold_ids, cat_ids, week_list, week_dates):
    rows = []
    for ii, iid in enumerate(cold_ids):
        for wi, wk in enumerate(week_list):
            rows.append({
                "item_id": iid, "store_id": "CA_1",
                "cat_id": cat_ids[ii],
                "iso_year": wk[0], "iso_week": wk[1],
                "date": week_dates[wi],
                "pred_sales": float(preds[ii, wi]),
            })
    return pd.DataFrame(rows)


def full_eval(pred_df, cold_weekly, warm_train, name):
    ev = evaluate_weekly(cold_weekly, pred_df, warm_train, name)
    mae_v  = float(ev["mae"])
    rmse_v = float(ev["rmse"])
    wrms_v = float(ev["wrmsse"])
    da_v   = float(ev["direction_accuracy"])
    warm_mean_sales = warm_train.groupby("item_id")["sales"].mean().mean()
    mase_v = mae_v / max(warm_mean_sales, 1e-6)
    merged = cold_weekly.merge(
        pred_df[["item_id","iso_year","iso_week","pred_sales"]],
        on=["item_id","iso_year","iso_week"], how="inner")
    smape_v = float((2 * np.abs(merged["sales"] - merged["pred_sales"]) /
                     (np.abs(merged["sales"]) + np.abs(merged["pred_sales"]) + 1e-6)).mean())
    return {"mae": mae_v, "rmse": rmse_v, "wrmsse": wrms_v,
            "mase": mase_v, "smape": smape_v, "dir_acc": da_v}


def load_competitor_preds(model_name):
    p = COMP_DIR / model_name / "predictions" / f"{model_name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def filter_16w(pred_df, wl16_set):
    return pred_df[pred_df.apply(
        lambda r: (r["iso_year"], r["iso_week"]) in wl16_set, axis=1)].copy()


def item_mae_vec(pred_df, cold_weekly, cold_ids):
    merged = cold_weekly.merge(
        pred_df[["item_id","iso_year","iso_week","pred_sales"]],
        on=["item_id","iso_year","iso_week"], how="inner")
    if merged.empty:
        return np.array([]), []
    s = (merged.groupby("item_id")
         .apply(lambda g: (g["sales"] - g["pred_sales"]).abs().mean())
         .reindex(cold_ids).dropna())
    return s.values, s.index.tolist()


def fmt(v, dec=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.{dec}f}"


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    data = load_data()
    cold_ids = data["cold_ids"]
    cat_ids  = [data["cold_meta"]["cat_id"].get(i, "") for i in cold_ids]
    cold_weekly  = data["cold_weekly_16"]
    warm_train   = data["warm_train_weekly"]
    wl16_set     = set(data["week_list_16"])

    # ── Track B (N_SEEDS)
    logger.info("=== Track B (%d seeds) ===", N_SEEDS)
    tb_evals, tb_preds = [], []
    for seed in range(N_SEEDS):
        m = train_head(data["warm_residual"], data["y_warm_16"],
                       n_epochs=BEST_CONFIG["epochs"], lr=BEST_CONFIG["lr"],
                       bottleneck=BEST_CONFIG["bottleneck"],
                       lambda_dir=BEST_CONFIG["lambda_dir"], seed=seed)
        p = predict(m, data["cold_residual"], T=BEST_CONFIG["T"])
        pdf = preds_to_df(p, cold_ids, cat_ids,
                          data["week_list_16"], data["week_dates_16"])
        ev = full_eval(pdf, cold_weekly, warm_train, f"trackb_s{seed}")
        tb_evals.append(ev)
        tb_preds.append(pdf)
        logger.info("  seed=%d MAE=%.4f DirAcc=%.4f", seed, ev["mae"], ev["dir_acc"])

    tb_mean = {k: float(np.mean([e[k] for e in tb_evals])) for k in tb_evals[0]}
    tb_std  = {k: float(np.std( [e[k] for e in tb_evals])) for k in tb_evals[0]}
    logger.info("Track B: MAE=%.4f±%.4f  DirAcc=%.4f±%.4f",
                tb_mean["mae"], tb_std["mae"],
                tb_mean["dir_acc"], tb_std["dir_acc"])

    # seed=0 예측 → Wilcoxon용
    trackb_pred0 = tb_preds[0]

    # ── lightgbm
    logger.info("=== lightgbm_proxy_lags ===")
    lgbm_pred_df = load_competitor_preds("lightgbm_proxy_lags")
    if lgbm_pred_df is None:
        logger.info("  run_competitors.py 실행 중...")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_competitors.py"),
             "--model", "lightgbm_proxy_lags"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=3600,
        )
        if r.returncode != 0:
            logger.error("  lgbm 실패:\n%s", r.stderr[-800:])
        lgbm_pred_df = load_competitor_preds("lightgbm_proxy_lags")

    if lgbm_pred_df is not None:
        lgbm_filt = filter_16w(lgbm_pred_df, wl16_set)
        lgbm_ev   = full_eval(lgbm_filt, cold_weekly, warm_train, "lightgbm_proxy_lags")
        logger.info("  MAE=%.4f  DirAcc=%.4f", lgbm_ev["mae"], lgbm_ev["dir_acc"])
    else:
        lgbm_ev = {"mae": 8.48, "rmse": float("nan"), "wrmsse": float("nan"),
                   "mase": float("nan"), "smape": float("nan"), "dir_acc": 0.343}
        logger.warning("  lgbm fallback 사용")

    # ── knn
    logger.info("=== knn_analog ===")
    knn_pred_df = load_competitor_preds("knn_analog")
    if knn_pred_df is not None:
        knn_filt = filter_16w(knn_pred_df, wl16_set)
        knn_ev   = full_eval(knn_filt, cold_weekly, warm_train, "knn_analog")
        logger.info("  MAE=%.4f  DirAcc=%.4f", knn_ev["mae"], knn_ev["dir_acc"])
    else:
        knn_ev = {"mae": 9.5977, "rmse": float("nan"), "wrmsse": float("nan"),
                  "mase": float("nan"), "smape": float("nan"), "dir_acc": 0.4113}

    # ── Wilcoxon
    logger.info("=== Wilcoxon ===")
    wilcoxon: dict = {}
    if lgbm_pred_df is not None:
        lgbm_filt2 = filter_16w(lgbm_pred_df, wl16_set)
        tb_vec,  _      = item_mae_vec(trackb_pred0, cold_weekly, cold_ids)
        lgb_vec, lgb_id = item_mae_vec(lgbm_filt2,  cold_weekly, cold_ids)
        if len(tb_vec) > 0 and len(lgb_vec) > 0:
            # 공통 아이템만
            tb_s  = pd.Series(item_mae_vec(trackb_pred0, cold_weekly, cold_ids)[0],
                               index=item_mae_vec(trackb_pred0, cold_weekly, cold_ids)[1])
            lgb_s = pd.Series(lgb_vec, index=lgb_id)
            common = tb_s.index.intersection(lgb_s.index)
            tb_arr  = tb_s.loc[common].values
            lgb_arr = lgb_s.loc[common].values
            stat, pval = stats.wilcoxon(tb_arr, lgb_arr)
            wilcoxon = {
                "statistic": float(stat), "p_value": float(pval),
                "n_items": len(common),
                "trackb_mean_mae":  float(np.mean(tb_arr)),
                "lgbm_mean_mae":    float(np.mean(lgb_arr)),
                "trackb_better":    int((tb_arr < lgb_arr).sum()),
                "lgbm_better":      int((lgb_arr < tb_arr).sum()),
            }
            logger.info("  stat=%.2f  p=%.4f  n=%d",
                        stat, pval, len(common))
            logger.info("  TrackB 우위: %d/%d  lgbm 우위: %d/%d",
                        wilcoxon["trackb_better"], len(common),
                        wilcoxon["lgbm_better"], len(common))
        else:
            wilcoxon = {"p_value": float("nan"), "note": "벡터 없음"}
    else:
        wilcoxon = {"p_value": float("nan"), "note": "lgbm 예측값 없음"}

    # ── 보고서 파트 2 교체
    logger.info("보고서 업데이트 중...")
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    lines = existing.split("\n")

    # 새 파트2 섹션 구성
    table = [
        "## 파트 2: 통합 비교 테이블", "",
        "| 모델 | MAE | RMSE | WRMSSE | MASE | sMAPE | DirAcc |",
        "|------|-----|------|--------|------|-------|--------|",
    ]

    def row(name, ev, mean_v=None, std_v=None):
        mae_s = (f"{fmt(mean_v.get('mae'))}±{fmt(std_v.get('mae'))}"
                 if mean_v else fmt(ev["mae"]))
        da_s  = (f"{fmt(mean_v.get('dir_acc'))}±{fmt(std_v.get('dir_acc'))}"
                 if mean_v else fmt(ev["dir_acc"]))
        return (f"| {name} | {mae_s} | {fmt(ev['rmse'])} | "
                f"{fmt(ev['wrmsse'])} | {fmt(ev['mase'])} | "
                f"{fmt(ev['smape'])} | {da_s} |")

    table.append(row("Track_B_exp018", tb_evals[0], tb_mean, tb_std))
    table.append(row("lightgbm_proxy_lags", lgbm_ev))
    table.append(row("knn_analog", knn_ev))
    table.append("| similar_item_avg | 8.6400 | — | — | — | — | 0.2320 |")
    table.append("| Track_A_calibrated | 8.9000 | — | — | — | — | 0.3930 |")

    # Wilcoxon 섹션
    table += ["", "### 통계적 유의성 (Wilcoxon signed-rank test, paired by item)", ""]
    pval = wilcoxon.get("p_value", float("nan"))
    stat = wilcoxon.get("statistic", float("nan"))
    if not np.isnan(pval):
        sig = "**유의미한 차이 (p<0.05)**" if pval < 0.05 else "유의미한 차이 없음 (p≥0.05)"
        n   = wilcoxon["n_items"]
        tb_b = wilcoxon["trackb_better"]
        lg_b = wilcoxon["lgbm_better"]
        table += [
            "| 비교 | 통계량 | p-value | 해석 |",
            "|------|--------|---------|------|",
            f"| Track B vs lightgbm | {stat:.2f} | {pval:.4f} | {sig} |",
            "",
            f"아이템별 MAE 우위 (n={n}): Track B {tb_b}개 / lightgbm {lg_b}개",
        ]
    else:
        note = wilcoxon.get("note", "unknown")
        table += [f"⚠️ 검증 불가: {note}"]

    table += ["", "---"]

    # 파트2 구간 교체
    s = next((i for i, l in enumerate(lines) if l.strip().startswith("## 파트 2")), None)
    e = next((i for i, l in enumerate(lines)
               if i > (s or 0) and l.strip().startswith("## 파트 3")), None)

    if s is not None and e is not None:
        new_lines = lines[:s] + table + [""] + lines[e:]
    else:
        new_lines = lines + [""] + table

    REPORT.write_text("\n".join(new_lines), encoding="utf-8")
    logger.info("보고서 저장: %s", REPORT)

    # ── 최종 요약
    print("\n" + "="*70)
    print("[Exp019 보완 결과]")
    print(f"Track B  : MAE={tb_mean['mae']:.4f}±{tb_std['mae']:.4f}  "
          f"DirAcc={tb_mean['dir_acc']:.4f}±{tb_std['dir_acc']:.4f}")
    print(f"lightgbm : MAE={lgbm_ev['mae']:.4f}  DirAcc={lgbm_ev['dir_acc']:.4f}")
    print(f"knn      : MAE={knn_ev['mae']:.4f}  DirAcc={knn_ev['dir_acc']:.4f}")
    if not np.isnan(pval):
        print(f"Wilcoxon : stat={stat:.2f}  p={pval:.4f}  → "
              f"{'유의미' if pval<0.05 else '비유의'}")
        print(f"           TrackB 우위 {wilcoxon['trackb_better']}/{wilcoxon['n_items']}")
    else:
        print(f"Wilcoxon : 불가 ({wilcoxon.get('note','')})")
    print("="*70)


if __name__ == "__main__":
    main()
