"""Exp019: DirAcc 재현 검증 + 공정 평가 체계 구축.

파트 1: exp011(17주, DirAcc=0.550) vs exp016/018(16주, DirAcc=0.402) 원인 규명
파트 2: 전체 평가 지표 (MAE, RMSE, WRMSSE, MASE, sMAPE, DirAcc) 통합 비교
파트 3: 아이템별 오차 분석 + 오차 분해 + Competitor 비교

출력:
    docs/diagnosis/evaluation_framework_report.md
    experiments/exp019_evaluation_framework/figures/
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

EMB_DIR     = ROOT / "experiments/exp011_v3_pipeline/embeddings"
MODEL11_DIR = ROOT / "experiments/exp011_v3_pipeline/models/attn_bottleneck"
FIG_DIR     = ROOT / "experiments/exp019_evaluation_framework/figures"
REPORT_DIR  = ROOT / "docs/diagnosis"
CS_DIR      = ROOT / "data/processed/cold_start"
COMP_DIR    = ROOT / "experiments/exp006_competitors"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
COMP_DIR.mkdir(parents=True, exist_ok=True)

# exp006에서 확인된 competitor 결과 (MAE, DirAcc 기준)
COMPETITOR_REF = {
    "lightgbm_proxy_lags": {"mae": 8.48, "dir_acc": 0.343},
    "similar_item_avg":    {"mae": 8.64, "dir_acc": 0.232},
    "Track_A_calibrated":  {"mae": 8.90, "dir_acc": 0.393},
    "knn_analog":          {"mae": 9.57, "dir_acc": 0.412},
}

# exp018 최적 구성
BEST_CONFIG = {
    "emb": "residual", "weeks": "17", "epochs": 500,
    "T": 2.0, "lambda_dir": 0.5, "lr": 1e-3, "bottleneck": 128,
}


# ─── 모델 정의 ────────────────────────────────────────────────────────────────

class AttnBottleneck(nn.Module):
    """Attention+Bottleneck head."""
    def __init__(self, hidden: int = 5120, bottleneck: int = 64,
                 n_weeks: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, bottleneck), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        attn_w = torch.softmax(scores, dim=-1)
        ctx    = (x * attn_w.unsqueeze(-1)).sum(dim=1)
        return self.head(ctx)

    def get_attn_weights(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        return torch.softmax(scores, dim=-1)


def train_head(
    X: np.ndarray, y: np.ndarray,
    n_epochs: int = 500, lr: float = 1e-3, bottleneck: int = 64,
    seed: int = 0, lambda_dir: float = 0.0,
) -> AttnBottleneck:
    """AttnBottleneck 학습."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_weeks = y.shape[1]
    model = AttnBottleneck(bottleneck=bottleneck, n_weeks=n_weeks)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X_t   = torch.from_numpy(X.astype(np.float32))
    y_t   = torch.from_numpy(y.astype(np.float32))

    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        pred = model(X_t)
        mae_loss = nn.functional.l1_loss(pred, y_t)
        if lambda_dir > 0:
            pd_ = pred[:, 1:] - pred[:, :-1]
            td_ = y_t[:, 1:]  - y_t[:, :-1]
            sign_match = torch.tanh(pd_ * 10) * torch.tanh(td_ * 10)
            dir_loss = -sign_match.mean()
            loss = mae_loss + lambda_dir * dir_loss
        else:
            loss = mae_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def _to_weekly_iso(df: pd.DataFrame) -> pd.DataFrame:
    """daily → weekly ISO 집계."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    group_cols = [c for c in ["item_id", "store_id", "cat_id", "dept_id",
                               "state_id", "iso_year", "iso_week"] if c in df.columns]
    return (
        df.groupby(group_cols)
        .agg(**{"sales": ("sales", "sum"), "date": ("week_start", "first")})
        .reset_index()
    )


def _complete_weeks(df_daily: pd.DataFrame) -> set:
    """7일이 모두 있는 완전한 주 집합 반환."""
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    counts = (
        df.groupby(["iso_year", "iso_week"])["date"]
        .nunique().reset_index(name="n_days")
    )
    complete = counts[counts["n_days"] == 7]
    return set(zip(complete["iso_year"], complete["iso_week"]))


def load_data() -> dict:
    """데이터 및 임베딩 로드."""
    logger.info("데이터 로드 중...")

    # embeddings
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", map_location="cpu",
                          weights_only=False).numpy()   # (300,50,5120)
    cold_raw = torch.load(EMB_DIR / "cold_raw.pt", map_location="cpu",
                          weights_only=False).numpy()   # (100,50,5120)

    # residual embeddings
    warm_item_mean = warm_raw.mean(axis=1, keepdims=True)   # (300,1,5120)
    cold_item_mean = cold_raw.mean(axis=1, keepdims=True)
    warm_residual  = warm_raw - warm_item_mean
    cold_residual  = cold_raw - cold_item_mean

    # daily data
    cold_test_raw  = pd.read_csv(CS_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_train_raw = pd.read_csv(CS_DIR / "warm_train.csv", parse_dates=["date"])
    warm_test_raw  = pd.read_csv(CS_DIR / "warm_test.csv",  parse_dates=["date"])

    # item_meta.csv에서 warm/cold 아이템 순서 로드 (embedding 행 순서와 일치)
    item_meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_ids = item_meta[item_meta["is_cold"] == True]["item_id"].tolist()   # 100개, embedding 순서
    warm_ids = item_meta[item_meta["is_cold"] == False]["item_id"].tolist()  # 300개, embedding 순서

    # 완전한 주 (16주)
    complete_set_16 = _complete_weeks(cold_test_raw)
    logger.info("  완전한 주(16w): %d개", len(complete_set_16))

    # 모든 주 (17주 — 부분 포함)
    cold_test_raw2 = cold_test_raw.copy()
    cold_test_raw2["iso_year"] = cold_test_raw2["date"].dt.isocalendar().year.astype(int)
    cold_test_raw2["iso_week"] = cold_test_raw2["date"].dt.isocalendar().week.astype(int)
    all_weeks = set(zip(cold_test_raw2["iso_year"], cold_test_raw2["iso_week"]))
    week_list_17 = sorted(all_weeks)
    logger.info("  전체 주(17w): %d개", len(week_list_17))

    cold_test_weekly = _to_weekly_iso(cold_test_raw)
    warm_train_weekly = _to_weekly_iso(warm_train_raw)
    warm_test_weekly  = _to_weekly_iso(warm_test_raw)

    # 16w 필터링
    cold_weekly_16 = cold_test_weekly[
        cold_test_weekly.apply(lambda r: (r["iso_year"], r["iso_week"]) in complete_set_16, axis=1)
    ]
    # 17w (모든 주)
    cold_weekly_17 = cold_test_weekly.copy()

    week_list_16 = sorted(complete_set_16)

    def build_y_matrix(cold_weekly, week_list, cold_ids):
        week_idx = {wk: i for i, wk in enumerate(week_list)}
        y = np.zeros((len(cold_ids), len(week_list)), dtype=np.float32)
        cold_id_idx = {cid: i for i, cid in enumerate(cold_ids)}
        for _, row in cold_weekly.iterrows():
            wk = (row["iso_year"], row["iso_week"])
            if wk in week_idx and row["item_id"] in cold_id_idx:
                y[cold_id_idx[row["item_id"]], week_idx[wk]] = row["sales"]
        return y

    def build_y_warm(warm_test_weekly, week_list, warm_ids):
        week_idx = {wk: i for i, wk in enumerate(week_list)}
        y = np.zeros((len(warm_ids), len(week_list)), dtype=np.float32)
        warm_id_idx = {wid: i for i, wid in enumerate(warm_ids)}
        for _, row in warm_test_weekly.iterrows():
            wk = (row["iso_year"], row["iso_week"])
            if wk in week_idx and row["item_id"] in warm_id_idx:
                y[warm_id_idx[row["item_id"]], week_idx[wk]] = row["sales"]
        return y

    y_cold_16 = build_y_matrix(cold_weekly_16, week_list_16, cold_ids)
    y_cold_17 = build_y_matrix(cold_weekly_17, week_list_17, cold_ids)
    y_warm_16 = build_y_warm(warm_test_weekly, week_list_16, warm_ids)
    y_warm_17 = build_y_warm(warm_test_weekly, week_list_17, warm_ids)
    logger.info("  y_cold_16=%s  y_cold_17=%s", y_cold_16.shape, y_cold_17.shape)
    logger.info("  y_warm_16=%s  y_warm_17=%s", y_warm_16.shape, y_warm_17.shape)

    # week start dates
    def week_dates(cold_weekly, week_list):
        wk2date = {}
        for _, r in cold_weekly.iterrows():
            wk = (r["iso_year"], r["iso_week"])
            if wk in set(week_list):
                wk2date[wk] = r["date"]
        return [wk2date.get(wk, pd.Timestamp("2016-01-01")) for wk in week_list]

    wd_16 = week_dates(cold_weekly_16, week_list_16)
    wd_17 = week_dates(cold_weekly_17, week_list_17)

    # cold item metadata
    cold_meta = (
        cold_test_raw[["item_id", "cat_id", "dept_id"]].drop_duplicates("item_id")
        .set_index("item_id").loc[cold_ids].reset_index()
    )

    # sell_prices for cold items (for MASE/sMAPE context)
    try:
        sell_prices = pd.read_csv(
            ROOT / "m5-forecasting-accuracy/sell_prices.csv"
        )
        price_avg = (
            sell_prices[sell_prices["store_id"] == "CA_1"]
            .groupby("item_id")["sell_price"].mean()
        )
        cold_meta["price"] = cold_meta["item_id"].map(price_avg).fillna(
            price_avg.mean()
        )
    except Exception:
        cold_meta["price"] = np.nan

    return {
        "warm_raw": warm_raw, "cold_raw": cold_raw,
        "warm_residual": warm_residual, "cold_residual": cold_residual,
        "cold_ids": cold_ids, "warm_ids": warm_ids,
        "cold_test_weekly": cold_test_weekly,
        "cold_weekly_16": cold_weekly_16,
        "cold_weekly_17": cold_weekly_17,
        "warm_train_weekly": warm_train_weekly,
        "y_cold_16": y_cold_16, "y_cold_17": y_cold_17,
        "y_warm_16": y_warm_16, "y_warm_17": y_warm_17,
        "week_list_16": week_list_16, "week_list_17": week_list_17,
        "week_dates_16": wd_16, "week_dates_17": wd_17,
        "cold_meta": cold_meta,
    }


# ─── 예측 → DataFrame 변환 ────────────────────────────────────────────────────

def preds_to_df(
    preds: np.ndarray,
    item_ids: list[str],
    cat_ids: list[str],
    week_list: list[tuple],
    week_dates: list,
    store_id: str = "CA_1",
) -> pd.DataFrame:
    """(N, T) 예측 배열 → evaluate_weekly용 long DataFrame."""
    rows = []
    for i, (iid, cid) in enumerate(zip(item_ids, cat_ids)):
        for t, (wk, wd) in enumerate(zip(week_list, week_dates)):
            rows.append({
                "item_id": iid, "store_id": store_id, "cat_id": cid,
                "iso_year": wk[0], "iso_week": wk[1],
                "date": wd,
                "pred_sales": float(preds[i, t]),
            })
    return pd.DataFrame(rows)


def predict(model: AttnBottleneck, X: np.ndarray, T: float = 1.0) -> np.ndarray:
    """모델 예측. T: temperature scaling."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X.astype(np.float32))
        if T != 1.0:
            scores = model.attn(x_t).squeeze(-1) / T
            attn_w = torch.softmax(scores, dim=-1)
            ctx    = (x_t * attn_w.unsqueeze(-1)).sum(dim=1)
            preds  = model.head(ctx)
        else:
            preds  = model(x_t)
    return preds.numpy()


# ─── 추가 평가 지표 계산 ──────────────────────────────────────────────────────

def compute_extra_metrics(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    warm_train_weekly: pd.DataFrame,
) -> dict[str, float]:
    """MASE, sMAPE를 추가로 계산.

    actual_df: cold_weekly (iso_year, iso_week, item_id, sales)
    pred_df:   preds_to_df 결과 (item_id, iso_year, iso_week, pred_sales)
    """
    merged = actual_df.merge(
        pred_df[["item_id", "iso_year", "iso_week", "pred_sales"]],
        on=["item_id", "iso_year", "iso_week"], how="inner",
    )
    if merged.empty:
        return {"mase": float("nan"), "smape": float("nan"), "rmse": float("nan")}

    # RMSE
    rmse = float(np.sqrt(np.mean((merged["sales"] - merged["pred_sales"]) ** 2)))

    # sMAPE = 2|actual-pred| / (|actual|+|pred|)
    denom = merged["sales"].abs() + merged["pred_sales"].abs()
    smape_vals = np.where(denom > 0, 2 * (merged["sales"] - merged["pred_sales"]).abs() / denom, 0.0)
    smape = float(smape_vals.mean())

    # MASE: naive = category mean weekly sales (from warm train)
    cat_weekly_mean = (
        warm_train_weekly.groupby("cat_id")["sales"].mean().to_dict()
    )
    cold_meta_cat = merged[["item_id"]].drop_duplicates()
    # merge cat info via actual_df
    if "cat_id" in actual_df.columns:
        cold_meta_cat = actual_df[["item_id", "cat_id"]].drop_duplicates()
        merged = merged.merge(cold_meta_cat, on="item_id", how="left", suffixes=("", "_r"))
        if "cat_id_r" in merged.columns:
            merged["cat_id"] = merged["cat_id"].fillna(merged["cat_id_r"])
            merged = merged.drop(columns=["cat_id_r"])
    else:
        merged["cat_id"] = "UNKNOWN"

    merged["naive_mae"] = merged["cat_id"].map(cat_weekly_mean).fillna(merged["sales"].mean())
    mae_naive = float(merged["naive_mae"].mean())
    mae_actual = float((merged["sales"] - merged["pred_sales"]).abs().mean())
    mase = mae_actual / mae_naive if mae_naive > 0 else float("nan")

    return {"rmse": rmse, "smape": smape, "mase": mase}


def full_eval(
    pred_df: pd.DataFrame,
    cold_weekly: pd.DataFrame,
    warm_train_weekly: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    """evaluate_weekly + RMSE/MASE/sMAPE를 통합 계산."""
    ev = evaluate_weekly(cold_weekly, pred_df, warm_train_weekly, model_name)
    extra = compute_extra_metrics(pred_df, cold_weekly, warm_train_weekly)
    return {
        "mae": ev["mae"],
        "rmse": extra["rmse"],
        "wrmsse": ev["wrmsse"],
        "mase": extra["mase"],
        "smape": extra["smape"],
        "dir_acc": ev["direction_accuracy"],
    }


# ─── Part 1: DirAcc 재현 검증 ─────────────────────────────────────────────────

def dir_acc_weekly_exp009(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """exp009 원본 DirAcc 계산: flat 실제 변화 제외, 비-flat 전환만 평가.

    y_true, y_pred: (N, T) 주간 판매량 배열
    """
    if y_true.shape[1] < 2:
        return float("nan")
    td  = np.sign(np.diff(y_true, 1, axis=1))   # (N, T-1)
    pd_ = np.sign(np.diff(y_pred, 1, axis=1))
    m   = (td != 0)                               # non-flat actual mask
    if m.sum() > 0:
        return float((td[m] == pd_[m]).mean())
    return float("nan")


def part1_diracc_verify(data: dict) -> dict:
    """exp011(17주, 원본 embedding) DirAcc 재현 및 평가 기준 비교."""
    logger.info("=== Part 1: DirAcc 재현 검증 ===")

    # 1-1: exp011 코드 추적 — 모델 구조 확인
    model11_path = MODEL11_DIR / "model.pt"
    state_dict = torch.load(model11_path, map_location="cpu", weights_only=False)
    head_out_shape = state_dict["head.3.weight"].shape  # [n_weeks, bottleneck]
    n_weeks_11 = head_out_shape[0]
    logger.info("  exp011 모델 출력 weeks: %d", n_weeks_11)
    logger.info("  DirAcc 계산 함수: src.evaluation.metrics.direction_accuracy_weekly")
    logger.info("  flat 처리: sign 비교 (sign(0)==0 → flat vs flat는 일치)")

    # exp011 모델 로드
    model11 = AttnBottleneck(hidden=5120, bottleneck=64, n_weeks=n_weeks_11)
    model11.load_state_dict(state_dict)
    model11.eval()

    # 원본 embedding으로 예측
    cold_raw  = data["cold_raw"]   # (100,50,5120)
    cold_ids  = data["cold_ids"]
    cold_meta = data["cold_meta"]
    cat_ids   = cold_meta["cat_id"].tolist()

    with torch.no_grad():
        preds11 = model11(torch.from_numpy(cold_raw.astype(np.float32))).numpy()  # (100,17)

    logger.info("  exp011 예측 shape: %s  mean=%.2f  std=%.2f",
                preds11.shape, preds11.mean(), preds11.std())

    # exp009 원본 DirAcc 방식 (flat 제외) — y_cold_17과 비교
    da_exp009_17 = dir_acc_weekly_exp009(data["y_cold_17"], preds11)
    logger.info("  exp009 방식 DirAcc / 17w (flat 제외): %.4f", da_exp009_17)
    da_exp009_16 = dir_acc_weekly_exp009(data["y_cold_16"], preds11[:, [i for i, wk in enumerate(data["week_list_17"]) if wk in set(data["week_list_16"])]])
    logger.info("  exp009 방식 DirAcc / 16w (flat 제외): %.4f", da_exp009_16)

    # ─ 1-3: 17주 기준 평가 (exp011 원래 평가)
    pred_df_17 = preds_to_df(
        preds11, cold_ids, cat_ids,
        data["week_list_17"], data["week_dates_17"],
    )
    ev11_17 = evaluate_weekly(
        data["cold_weekly_17"], pred_df_17,
        data["warm_train_weekly"], "exp011_17w",
    )
    logger.info("  exp011 / 17w → MAE=%.4f  DirAcc(현재방식)=%.4f",
                ev11_17["mae"], ev11_17["direction_accuracy"])

    # ─ 1-3: 동일 예측을 16주 기준으로 재평가
    # 17주 예측에서 16주에 해당하는 열만 추출
    wl17 = data["week_list_17"]
    wl16 = data["week_list_16"]
    idx_16in17 = [i for i, wk in enumerate(wl17) if wk in set(wl16)]
    preds11_16 = preds11[:, idx_16in17]  # (100,16)

    pred_df_11_16 = preds_to_df(
        preds11_16, cold_ids, cat_ids,
        data["week_list_16"], data["week_dates_16"],
    )
    ev11_16 = evaluate_weekly(
        data["cold_weekly_16"], pred_df_11_16,
        data["warm_train_weekly"], "exp011_16w",
    )
    logger.info("  exp011 / 16w → MAE=%.4f  DirAcc=%.4f",
                ev11_16["mae"], ev11_16["direction_accuracy"])

    # ─ exp018 최적 모델 (17주, residual)
    X_res_warm = data["warm_residual"]
    X_res_cold = data["cold_residual"]
    y_warm_17  = data["y_warm_17"]

    model18 = train_head(
        X_res_warm, y_warm_17,
        n_epochs=BEST_CONFIG["epochs"], lr=BEST_CONFIG["lr"],
        bottleneck=BEST_CONFIG["bottleneck"],
        lambda_dir=BEST_CONFIG["lambda_dir"], seed=0,
    )
    preds18_17 = predict(model18, X_res_cold, T=BEST_CONFIG["T"])
    pred_df_18_17 = preds_to_df(
        preds18_17, cold_ids, cat_ids,
        data["week_list_17"], data["week_dates_17"],
    )
    ev18_17 = evaluate_weekly(
        data["cold_weekly_17"], pred_df_18_17,
        data["warm_train_weekly"], "exp018_17w",
    )
    logger.info("  exp018 / 17w → MAE=%.4f  DirAcc=%.4f",
                ev18_17["mae"], ev18_17["direction_accuracy"])

    # 16주 기준
    y_warm_16 = data["y_warm_16"]
    model18_16 = train_head(
        X_res_warm, y_warm_16,
        n_epochs=BEST_CONFIG["epochs"], lr=BEST_CONFIG["lr"],
        bottleneck=BEST_CONFIG["bottleneck"],
        lambda_dir=BEST_CONFIG["lambda_dir"], seed=0,
    )
    preds18_16 = predict(model18_16, X_res_cold, T=BEST_CONFIG["T"])
    pred_df_18_16 = preds_to_df(
        preds18_16, cold_ids, cat_ids,
        data["week_list_16"], data["week_dates_16"],
    )
    ev18_16 = evaluate_weekly(
        data["cold_weekly_16"], pred_df_18_16,
        data["warm_train_weekly"], "exp018_16w",
    )
    logger.info("  exp018 / 16w → MAE=%.4f  DirAcc=%.4f",
                ev18_16["mae"], ev18_16["direction_accuracy"])

    result = {
        "n_weeks_exp011": n_weeks_11,
        "da_exp009_method_17w": da_exp009_17,   # exp009 원본: flat 제외, 17주
        "da_exp009_method_16w": da_exp009_16,   # exp009 원본: flat 제외, 16주
        "exp011_17w": {"mae": ev11_17["mae"],    "dir_acc": ev11_17["direction_accuracy"]},
        "exp011_16w": {"mae": ev11_16["mae"],    "dir_acc": ev11_16["direction_accuracy"]},
        "exp018_17w": {"mae": ev18_17["mae"],    "dir_acc": ev18_17["direction_accuracy"]},
        "exp018_16w": {"mae": ev18_16["mae"],    "dir_acc": ev18_16["direction_accuracy"]},
        "diracc_diff_eval_period": ev11_17["direction_accuracy"] - ev11_16["direction_accuracy"],
        # 저장용 pred_df (Part 2/3 재활용)
        "_pred_df_18_16": pred_df_18_16,
        "_model18_16": model18_16,
    }

    # 판정: DirAcc 0.55 vs 0.40 원인
    diff_method = abs(da_exp009_17 - ev11_17["direction_accuracy"])
    diff_eval   = abs(ev11_17["direction_accuracy"] - ev11_16["direction_accuracy"])
    diff_model  = abs(ev11_17["direction_accuracy"] - ev18_17["direction_accuracy"])

    if diff_method > 0.05:
        verdict = (
            f"계산 방식 차이가 주된 원인: "
            f"exp009방식(flat제외)={da_exp009_17:.4f} vs 현재방식(flat포함)={ev11_17['direction_accuracy']:.4f} "
            f"(Δ={diff_method:+.4f})"
        )
    elif diff_eval > 0.05:
        verdict = f"평가 기간 차이가 주된 원인: 17w={ev11_17['direction_accuracy']:.4f} vs 16w={ev11_16['direction_accuracy']:.4f}"
    elif diff_model > 0.05:
        verdict = f"모델/임베딩 차이가 주된 원인: exp011={ev11_17['direction_accuracy']:.4f} vs exp018={ev18_17['direction_accuracy']:.4f}"
    else:
        verdict = "DirAcc 차이 미미 — 계산 방식/기간/모델 모두 유사"
    result["verdict"] = verdict
    logger.info("  판정: %s", verdict)

    return result


# ─── Part 2: 공정 평가 체계 ───────────────────────────────────────────────────

def _run_competitor(model_name: str) -> bool:
    """run_competitors.py 실행. 성공 여부 반환."""
    pred_path = COMP_DIR / model_name / "predictions" / f"{model_name}.csv"
    if pred_path.exists():
        logger.info("  [%s] 기존 예측값 사용", model_name)
        return True
    logger.info("  [%s] run_competitors.py 실행 중...", model_name)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_competitors.py"), "--model", model_name],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        logger.warning("  [%s] 실행 실패: %s", model_name, result.stderr[-500:])
        return False
    return pred_path.exists()


def _load_competitor_preds(model_name: str) -> pd.DataFrame | None:
    """Saved competitor 예측값 로드."""
    pred_path = COMP_DIR / model_name / "predictions" / f"{model_name}.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path, parse_dates=["date"])
        df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
        df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
        return df
    return None


def _eval_competitor_from_pred(
    pred_weekly_df: pd.DataFrame,
    cold_weekly: pd.DataFrame,
    warm_train_weekly: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    """Competitor 예측 DataFrame을 evaluate_weekly에 맞게 변환 후 평가."""
    # pred_weekly_df: item_id, iso_year, iso_week, pred_sales
    # 16w 기준 필터링
    wl16_set = set(zip(
        cold_weekly["iso_year"], cold_weekly["iso_week"]
    ))
    pred_filt = pred_weekly_df[
        pred_weekly_df.apply(lambda r: (r["iso_year"], r["iso_week"]) in wl16_set, axis=1)
    ].copy()
    if pred_filt.empty:
        return {k: float("nan") for k in ["mae","rmse","wrmsse","mase","smape","dir_acc"]}

    # evaluate_weekly가 기대하는 포맷으로 맞추기
    if "date" not in pred_filt.columns:
        pred_filt = pred_filt.merge(
            cold_weekly[["iso_year", "iso_week", "date"]].drop_duplicates(),
            on=["iso_year", "iso_week"], how="left",
        )
    if "pred_sales" not in pred_filt.columns and "pred" in pred_filt.columns:
        pred_filt = pred_filt.rename(columns={"pred": "pred_sales"})

    return full_eval(pred_filt, cold_weekly, warm_train_weekly, model_name)


def part2_fair_eval(data: dict, track_b_pred_df: pd.DataFrame) -> dict:
    """Part 2: 전체 지표 통합 비교."""
    logger.info("=== Part 2: 공정 평가 체계 ===")

    cold_weekly = data["cold_weekly_16"]
    warm_train  = data["warm_train_weekly"]

    results: dict[str, dict] = {}

    # ─ Track B (exp018 최적, 10 seeds)
    logger.info("  Track B (exp018 최적) 평가 중...")
    track_b_seeds = []
    for seed in range(10):
        m = train_head(
            data["warm_residual"], data["y_warm_16"],
            n_epochs=BEST_CONFIG["epochs"], lr=BEST_CONFIG["lr"],
            bottleneck=BEST_CONFIG["bottleneck"],
            lambda_dir=BEST_CONFIG["lambda_dir"], seed=seed,
        )
        p = predict(m, data["cold_residual"], T=BEST_CONFIG["T"])
        pdf = preds_to_df(
            p, data["cold_ids"], data["cold_meta"]["cat_id"].tolist(),
            data["week_list_16"], data["week_dates_16"],
        )
        ev = full_eval(pdf, cold_weekly, warm_train, f"trackb_s{seed}")
        track_b_seeds.append(ev)
        logger.info("    seed=%d MAE=%.4f DirAcc=%.4f", seed, ev["mae"], ev["dir_acc"])

    for k in ["mae", "rmse", "wrmsse", "mase", "smape", "dir_acc"]:
        vals = [s[k] for s in track_b_seeds if not np.isnan(s.get(k, np.nan))]
        results["Track_B_exp018"] = results.get("Track_B_exp018", {})
        results["Track_B_exp018"][k] = float(np.mean(vals)) if vals else float("nan")
        results["Track_B_exp018"][f"{k}_std"] = float(np.std(vals)) if vals else float("nan")

    # Keep all seed predictions for Wilcoxon test
    results["Track_B_exp018"]["_seed_preds"] = track_b_seeds

    # ─ Competitors
    competitor_models = ["knn_analog", "lightgbm_proxy_lags"]

    for model_name in competitor_models:
        logger.info("  Competitor [%s] 처리 중...", model_name)
        success = _run_competitor(model_name)
        if success:
            pred_df_comp = _load_competitor_preds(model_name)
            if pred_df_comp is not None:
                ev = _eval_competitor_from_pred(pred_df_comp, cold_weekly, warm_train, model_name)
                results[model_name] = ev
                logger.info("    MAE=%.4f  DirAcc=%.4f", ev["mae"], ev["dir_acc"])
            else:
                # Fallback to reference constants
                results[model_name] = {**COMPETITOR_REF.get(model_name, {}),
                                       "rmse": float("nan"), "wrmsse": float("nan"),
                                       "mase": float("nan"), "smape": float("nan")}
        else:
            # Fallback
            results[model_name] = {**COMPETITOR_REF.get(model_name, {}),
                                   "rmse": float("nan"), "wrmsse": float("nan"),
                                   "mase": float("nan"), "smape": float("nan")}

    # Reference-only competitors (no predictions available)
    for model_name, ref in COMPETITOR_REF.items():
        if model_name not in results:
            results[model_name] = {**ref,
                                   "rmse": float("nan"), "wrmsse": float("nan"),
                                   "mase": float("nan"), "smape": float("nan")}

    # ─ Wilcoxon test: Track B vs lightgbm_proxy_lags (item-level MAE)
    logger.info("  통계적 유의성 검증 중...")
    wilcoxon = {}

    trackb_pred_merged = _compute_item_mae_vector(
        track_b_pred_df, cold_weekly, data["cold_ids"]
    )

    lgbm_pred_df = _load_competitor_preds("lightgbm_proxy_lags")
    if lgbm_pred_df is not None:
        lgbm_pred_merged = _compute_item_mae_vector(
            _filter_to_weekly_preds(lgbm_pred_df, cold_weekly),
            cold_weekly, data["cold_ids"],
        )
        if len(trackb_pred_merged) == len(lgbm_pred_merged) > 0:
            stat, pval = stats.wilcoxon(trackb_pred_merged, lgbm_pred_merged)
            wilcoxon["trackb_vs_lgbm"] = {
                "statistic": float(stat), "p_value": float(pval),
                "trackb_mean_mae": float(np.mean(trackb_pred_merged)),
                "lgbm_mean_mae": float(np.mean(lgbm_pred_merged)),
            }
            logger.info("  Wilcoxon TrackB vs lgbm: p=%.4f", pval)
        else:
            wilcoxon["trackb_vs_lgbm"] = {"p_value": float("nan"), "note": "예측값 없음"}
    else:
        wilcoxon["trackb_vs_lgbm"] = {"p_value": float("nan"), "note": "lgbm 예측값 없음"}

    results["wilcoxon"] = wilcoxon
    return results


def _filter_to_weekly_preds(pred_df: pd.DataFrame, cold_weekly: pd.DataFrame) -> pd.DataFrame:
    """Competitor 예측을 16w cold_weekly 기준으로 필터링."""
    wl16_set = set(zip(cold_weekly["iso_year"], cold_weekly["iso_week"]))
    return pred_df[
        pred_df.apply(lambda r: (r.get("iso_year", 0), r.get("iso_week", 0)) in wl16_set, axis=1)
    ].copy()


def _compute_item_mae_vector(
    pred_df: pd.DataFrame,
    cold_weekly: pd.DataFrame,
    cold_ids: list[str],
) -> np.ndarray:
    """item_id별 MAE 벡터 반환."""
    if pred_df is None or pred_df.empty:
        return np.array([])
    merged = cold_weekly.merge(
        pred_df[["item_id", "iso_year", "iso_week", "pred_sales"]],
        on=["item_id", "iso_year", "iso_week"], how="inner",
    )
    if merged.empty:
        return np.array([])
    item_mae = merged.groupby("item_id").apply(
        lambda g: (g["sales"] - g["pred_sales"]).abs().mean()
    ).reindex(cold_ids).dropna()
    return item_mae.values


# ─── Part 3: 아이템별 오차 분석 ──────────────────────────────────────────────

def part3_error_analysis(data: dict, track_b_pred_df: pd.DataFrame) -> dict:
    """Part 3: 아이템별 오차 분석 + 오차 분해 + Competitor 비교."""
    logger.info("=== Part 3: 아이템별 오차 분석 ===")

    cold_weekly = data["cold_weekly_16"]
    cold_ids    = data["cold_ids"]
    cold_meta   = data["cold_meta"].set_index("item_id")

    # Track B item-level MAE
    merged = cold_weekly.merge(
        track_b_pred_df[["item_id", "iso_year", "iso_week", "pred_sales"]],
        on=["item_id", "iso_year", "iso_week"], how="inner",
    )

    item_stats = merged.groupby("item_id").apply(lambda g: pd.Series({
        "actual_mean": g["sales"].mean(),
        "pred_mean":   g["pred_sales"].mean(),
        "mae":         (g["sales"] - g["pred_sales"]).abs().mean(),
        "scale_err":   abs(g["sales"].mean() - g["pred_sales"].mean()),
        "n_weeks":     len(g),
    })).reset_index()

    item_stats["pattern_err"] = (item_stats["mae"] - item_stats["scale_err"]).clip(lower=0)
    item_stats["cat_id"]  = item_stats["item_id"].map(lambda x: cold_meta.get("cat_id", {}).get(x, "") if isinstance(cold_meta, dict) else cold_meta["cat_id"].get(x, ""))
    item_stats["cat_id"]  = item_stats["item_id"].map(
        cold_meta["cat_id"].to_dict() if "cat_id" in cold_meta.columns else {}
    )
    item_stats["price"] = item_stats["item_id"].map(
        cold_meta["price"].to_dict() if "price" in cold_meta.columns else {}
    )

    # 상위 20개 고오차 아이템
    top20 = item_stats.nlargest(20, "mae")
    logger.info("  Top-20 고오차 아이템 (MAE): min=%.2f max=%.2f",
                top20["mae"].min(), top20["mae"].max())

    # 카테고리 분포
    cat_dist_all  = item_stats["cat_id"].value_counts().to_dict()
    cat_dist_top20 = top20["cat_id"].value_counts().to_dict()
    logger.info("  전체 카테고리: %s", cat_dist_all)
    logger.info("  Top-20 카테고리: %s", cat_dist_top20)

    # 스케일 vs 패턴 오차
    scale_pct = float((item_stats["scale_err"] / item_stats["mae"].clip(lower=1e-6)).mean())
    pattern_pct = 1.0 - scale_pct
    logger.info("  스케일 오차 비율: %.1f%%  패턴 오차 비율: %.1f%%",
                scale_pct * 100, pattern_pct * 100)

    # Track B vs lightgbm per-item
    lgbm_pred_df = _load_competitor_preds("lightgbm_proxy_lags")
    lgbm_vs_trackb = {}
    if lgbm_pred_df is not None:
        lgbm_filt = _filter_to_weekly_preds(lgbm_pred_df, cold_weekly)
        lgbm_merged = cold_weekly.merge(
            lgbm_filt[["item_id", "iso_year", "iso_week", "pred_sales"]],
            on=["item_id", "iso_year", "iso_week"], how="inner",
        )
        lgbm_item_mae = lgbm_merged.groupby("item_id").apply(
            lambda g: (g["sales"] - g["pred_sales"]).abs().mean()
        ).rename("lgbm_mae").reset_index()

        compare = item_stats[["item_id", "mae", "cat_id", "actual_mean", "price"]].merge(
            lgbm_item_mae, on="item_id", how="inner",
        )
        compare["trackb_better"] = compare["mae"] < compare["lgbm_mae"]

        n_trackb_better = int(compare["trackb_better"].sum())
        n_lgbm_better   = int((~compare["trackb_better"]).sum())
        logger.info("  Track B better: %d/100  lgbm better: %d/100",
                    n_trackb_better, n_lgbm_better)

        # 특성 비교
        trackb_better_cat = compare[compare["trackb_better"]]["cat_id"].value_counts().to_dict()
        lgbm_better_cat   = compare[~compare["trackb_better"]]["cat_id"].value_counts().to_dict()

        lgbm_vs_trackb = {
            "n_trackb_better": n_trackb_better,
            "n_lgbm_better": n_lgbm_better,
            "trackb_better_mean_actual": float(compare[compare["trackb_better"]]["actual_mean"].mean()),
            "lgbm_better_mean_actual": float(compare[~compare["trackb_better"]]["actual_mean"].mean()),
            "trackb_better_cat": trackb_better_cat,
            "lgbm_better_cat": lgbm_better_cat,
        }

    return {
        "item_stats": item_stats.to_dict(orient="records"),
        "top20": top20[["item_id","cat_id","actual_mean","pred_mean","mae","scale_err","pattern_err"]].to_dict(orient="records"),
        "cat_dist_all": cat_dist_all,
        "cat_dist_top20": cat_dist_top20,
        "scale_pct": scale_pct,
        "pattern_pct": pattern_pct,
        "lgbm_vs_trackb": lgbm_vs_trackb,
        "overall_mae": float(item_stats["mae"].mean()),
        "overall_scale_err": float(item_stats["scale_err"].mean()),
    }


# ─── 시각화 ───────────────────────────────────────────────────────────────────

def plot_p1(p1: dict) -> None:
    """Part 1: 평가 기간별 DirAcc 비교 bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["exp011/17w\n(원래)", "exp011/16w\n(재평가)", "exp018/17w", "exp018/16w"]
    da_vals = [
        p1["exp011_17w"]["dir_acc"], p1["exp011_16w"]["dir_acc"],
        p1["exp018_17w"]["dir_acc"], p1["exp018_16w"]["dir_acc"],
    ]
    colors = ["steelblue", "steelblue", "tomato", "tomato"]
    bars = ax.bar(labels, da_vals, color=colors, alpha=0.8)
    ax.axhline(0.412, color="green", linestyle="--", alpha=0.7, label="Target DirAcc=0.412")
    ax.set_ylabel("DirAcc"); ax.set_title("Part 1: DirAcc by Evaluation Period and Model")
    ax.set_ylim(0, 0.65); ax.legend(fontsize=9)
    for b, v in zip(bars, da_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_diracc_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: p1_diracc_comparison.png")


def plot_p2(p2: dict) -> None:
    """Part 2: 전체 모델 MAE + DirAcc 비교."""
    models = [k for k in p2.keys() if not k.startswith("_") and k != "wilcoxon"]
    maes   = [p2[m].get("mae", float("nan")) for m in models]
    daccs  = [p2[m].get("dir_acc", float("nan")) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Part 2: 통합 비교 (MAE & DirAcc)", fontsize=13)

    x = np.arange(len(models))
    axes[0].bar(x, maes, color=["tomato" if "Track_B" in m else "steelblue" for m in models], alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("MAE"); axes[0].set_title("Cold MAE (↓ better)")
    axes[0].axhline(8.48, color="gray", linestyle=":", alpha=0.7, label="lgbm ref")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, daccs, color=["tomato" if "Track_B" in m else "steelblue" for m in models], alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("DirAcc"); axes[1].set_title("DirAcc (↑ better)")
    axes[1].axhline(0.412, color="green", linestyle="--", alpha=0.7, label="target")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "p2_full_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: p2_full_comparison.png")


def plot_p3(p3: dict) -> None:
    """Part 3: 아이템별 MAE 분포 + 스케일/패턴 오차 분해."""
    item_df = pd.DataFrame(p3["item_stats"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Part 3: 아이템별 오차 분석", fontsize=13)

    # MAE 히스토그램
    axes[0].hist(item_df["mae"].dropna(), bins=20, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Item MAE"); axes[0].set_ylabel("Count")
    axes[0].set_title("Track B Item MAE 분포")

    # 스케일 vs 패턴 오차
    axes[1].scatter(item_df["scale_err"], item_df["pattern_err"], alpha=0.5, s=20)
    axes[1].set_xlabel("Scale Error"); axes[1].set_ylabel("Pattern Error")
    axes[1].set_title("Scale vs Pattern 오차")

    # actual vs pred scatter
    axes[2].scatter(item_df["actual_mean"], item_df["pred_mean"], alpha=0.5, s=20)
    max_val = max(item_df["actual_mean"].max(), item_df["pred_mean"].max()) * 1.1
    axes[2].plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")
    axes[2].set_xlabel("Actual Mean Sales"); axes[2].set_ylabel("Pred Mean Sales")
    axes[2].set_title("Actual vs Predicted (item mean)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "p3_item_error_analysis.png", dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: p3_item_error_analysis.png")


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(p1: dict, p2: dict, p3: dict) -> None:
    """docs/diagnosis/evaluation_framework_report.md 작성."""
    lines = [
        "# 평가 체계 구축 보고서 (Exp019)",
        "",
        "**작성일:** 2026-03-11",
        "**목표:** DirAcc 0.55 재현 검증 + 공정 평가 체계 확립 + 개선 방향 도출",
        "",
        "---",
        "",
        "## 파트 1: DirAcc 재현 검증",
        "",
        "### exp011 모델 정보",
        f"- 출력 주 수: **{p1['n_weeks_exp011']}주** (17주 포함)",
        "- 원래 계산 함수: `dir_acc_weekly()` (run_exp009.py) — **flat 제외**",
        "- 현재 함수: `direction_accuracy_weekly()` (metrics.py) — **flat 포함**",
        "",
        "### DirAcc 계산 방식 차이 (핵심)",
        "",
        "| 방식 | flat 처리 | exp011/17w DirAcc |",
        "|------|----------|-----------------|",
        f"| **exp009 원본** (flat 제외, m=td≠0) | 비-flat만 평가 | **{p1['da_exp009_method_17w']:.4f}** |",
        f"| **현재 metrics.py** (flat 포함) | 모든 전환 평가 | {p1['exp011_17w']['dir_acc']:.4f} |",
        "",
        "### 평가 기간별 DirAcc 비교 (현재 metrics.py 방식)",
        "",
        "| 모델 | 임베딩 | 평가 기간 | MAE | DirAcc |",
        "|------|--------|---------|-----|--------|",
        f"| exp011 (원래) | 원본 raw | 17주 | {p1['exp011_17w']['mae']:.4f} | {p1['exp011_17w']['dir_acc']:.4f} |",
        f"| exp011 (재평가) | 원본 raw | 16주 | {p1['exp011_16w']['mae']:.4f} | {p1['exp011_16w']['dir_acc']:.4f} |",
        f"| exp018 최적 | residual | 17주 | {p1['exp018_17w']['mae']:.4f} | {p1['exp018_17w']['dir_acc']:.4f} |",
        f"| exp018 최적 | residual | 16주 | {p1['exp018_16w']['mae']:.4f} | {p1['exp018_16w']['dir_acc']:.4f} |",
        "",
        f"**평가 기간(17w→16w)으로 인한 DirAcc 변화 (exp011 기준):** Δ={p1['diracc_diff_eval_period']:+.4f}",
        "",
        f"### 판정: **{p1['verdict']}**",
        "",
        "---",
        "",
        "## 파트 2: 통합 비교 테이블",
        "",
        "| 모델 | MAE | RMSE | WRMSSE | MASE | sMAPE | DirAcc |",
        "|------|-----|------|--------|------|-------|--------|",
    ]

    def fmt(v: Any, dec: int = 4) -> str:
        if isinstance(v, float) and np.isnan(v):
            return "—"
        try:
            return f"{float(v):.{dec}f}"
        except Exception:
            return "—"

    for model, metrics in p2.items():
        if model.startswith("_") or model == "wilcoxon":
            continue
        lines.append(
            f"| {model} | {fmt(metrics.get('mae'))} | {fmt(metrics.get('rmse'))} | "
            f"{fmt(metrics.get('wrmsse'))} | {fmt(metrics.get('mase'))} | "
            f"{fmt(metrics.get('smape'))} | {fmt(metrics.get('dir_acc'))} |"
        )

    wil = p2.get("wilcoxon", {}).get("trackb_vs_lgbm", {})
    pval = wil.get("p_value", float("nan"))
    lines += [
        "",
        "### 통계적 유의성 (Wilcoxon signed-rank test, paired by item)",
        "",
        f"| 비교 | 통계량 | p-value | 해석 |",
        "|------|--------|---------|------|",
    ]
    if not np.isnan(pval):
        interp = "Track B 유의하게 우수 (p<0.05)" if pval < 0.05 else "유의한 차이 없음 (p≥0.05)"
        lines.append(f"| Track B vs lightgbm | {fmt(wil.get('statistic', float('nan')))} | {pval:.4f} | {interp} |")
    else:
        lines.append("| Track B vs lightgbm | — | — | 예측값 없어 검증 불가 |")

    lines += [
        "",
        "---",
        "",
        "## 파트 3: 아이템별 오차 분석",
        "",
        f"**Track B 전체 MAE:** {p3['overall_mae']:.4f}",
        f"**스케일 오차 비율:** {p3['scale_pct']*100:.1f}%  패턴 오차 비율: {p3['pattern_pct']*100:.1f}%",
        "",
        "### 전체 카테고리 분포",
        f"> {p3['cat_dist_all']}",
        "",
        "### 상위 20개 고오차 아이템의 카테고리 분포",
        f"> {p3['cat_dist_top20']}",
        "",
        "### Top-20 고오차 아이템",
        "",
        "| item_id | cat | actual_mean | pred_mean | MAE | scale_err | pattern_err |",
        "|---------|-----|------------|----------|-----|-----------|-------------|",
    ]
    for row in p3["top20"][:20]:
        lines.append(
            f"| {row['item_id']} | {row.get('cat_id','—')} | "
            f"{fmt(row['actual_mean'],2)} | {fmt(row['pred_mean'],2)} | "
            f"{fmt(row['mae'],2)} | {fmt(row['scale_err'],2)} | {fmt(row['pattern_err'],2)} |"
        )

    lgbm_cmp = p3.get("lgbm_vs_trackb", {})
    if lgbm_cmp:
        lines += [
            "",
            "### Track B vs lightgbm_proxy_lags 아이템별 비교",
            "",
            f"- Track B가 더 좋은 아이템: **{lgbm_cmp.get('n_trackb_better', '—')}/100**",
            f"- lgbm이 더 좋은 아이템: **{lgbm_cmp.get('n_lgbm_better', '—')}/100**",
            f"- Track B 우세 아이템 평균 판매량: {fmt(lgbm_cmp.get('trackb_better_mean_actual', float('nan')), 2)}",
            f"- lgbm 우세 아이템 평균 판매량: {fmt(lgbm_cmp.get('lgbm_better_mean_actual', float('nan')), 2)}",
            f"- Track B 우세 카테고리: {lgbm_cmp.get('trackb_better_cat', {})}",
            f"- lgbm 우세 카테고리: {lgbm_cmp.get('lgbm_better_cat', {})}",
        ]

    lines += [
        "",
        "---",
        "",
        "## 종합 결론",
        "",
        f"1. **DirAcc 0.55 재현 판정:** {p1['verdict']}",
        f"2. **Track B MAE 우위:** {p2.get('Track_B_exp018', {}).get('mae', float('nan')):.4f} vs lightgbm {p2.get('lightgbm_proxy_lags', {}).get('mae', 8.48):.4f}",
        "3. **개선 우선순위:** 스케일 오차({:.1f}%)가 패턴 오차보다 큼 → 판매량 스케일 예측 개선 필요".format(p3["scale_pct"]*100),
        "",
        "**시각화:** `experiments/exp019_evaluation_framework/figures/`",
        "**스크립트:** `scripts/exp019_evaluation_framework.py`",
    ]

    path = REPORT_DIR / "evaluation_framework_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", path)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """exp019 전체 실행."""
    logger.info("=== Exp019: DirAcc 재현 검증 + 공정 평가 체계 ===")

    data = load_data()

    # Part 1
    p1 = part1_diracc_verify(data)
    track_b_pred_df = p1.pop("_pred_df_18_16")
    p1.pop("_model18_16", None)
    plot_p1(p1)

    # Part 2
    p2 = part2_fair_eval(data, track_b_pred_df)
    plot_p2(p2)

    # Part 3
    p3 = part3_error_analysis(data, track_b_pred_df)
    plot_p3(p3)

    # Report
    write_report(p1, p2, p3)

    logger.info("=== Exp019 완료 ===")

    print("\n" + "=" * 70)
    print("[Part 1] DirAcc 재현:")
    print(f"  exp011/17w: MAE={p1['exp011_17w']['mae']:.4f}  DirAcc={p1['exp011_17w']['dir_acc']:.4f}")
    print(f"  exp011/16w: MAE={p1['exp011_16w']['mae']:.4f}  DirAcc={p1['exp011_16w']['dir_acc']:.4f}")
    print(f"  exp018/17w: MAE={p1['exp018_17w']['mae']:.4f}  DirAcc={p1['exp018_17w']['dir_acc']:.4f}")
    print(f"  exp018/16w: MAE={p1['exp018_16w']['mae']:.4f}  DirAcc={p1['exp018_16w']['dir_acc']:.4f}")
    print(f"  판정: {p1['verdict']}")

    tb = p2.get("Track_B_exp018", {})
    print(f"\n[Part 2] Track B: MAE={tb.get('mae', 0):.4f}  DirAcc={tb.get('dir_acc', 0):.4f}")

    wil = p2.get("wilcoxon", {}).get("trackb_vs_lgbm", {})
    print(f"  Wilcoxon vs lgbm: p={wil.get('p_value', float('nan')):.4f}")

    print(f"\n[Part 3] 스케일 오차 {p3['scale_pct']*100:.1f}%  패턴 오차 {p3['pattern_pct']*100:.1f}%")
    lgbm_cmp = p3.get("lgbm_vs_trackb", {})
    if lgbm_cmp:
        print(f"  Track B 우세: {lgbm_cmp.get('n_trackb_better',0)}/100 아이템")

    print(f"\n보고서: docs/diagnosis/evaluation_framework_report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
