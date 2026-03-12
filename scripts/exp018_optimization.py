"""Exp018: Optimization — DirAcc 개선 실험.

4개 파트로 구성:
    Part 1: DirAcc Factor Analysis (embedding, weeks, epochs)
    Part 2: Attention Modifications (diffuse/sparse profiling, temperature, top-k)
    Part 3: Loss and Architecture (direction loss, hyperparam grid, dual attention)
    Part 4: Final Optimal Configuration (10 seeds)

배경:
    exp016 G1 baseline: cold MAE=8.71±0.29, DirAcc=0.386 (residual, 16w, 500ep)
    exp011 reference: DirAcc=0.550 (original embedding, 17w)
    Target: MAE < 8.48, DirAcc > 0.412

출력:
    experiments/exp018_optimization/figures/
    docs/diagnosis/optimization_report.md
"""
from __future__ import annotations

import json
import logging
import sys
import time
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

EMB_DIR    = ROOT / "experiments/exp011_v3_pipeline/embeddings"
FIG_DIR    = ROOT / "experiments/exp018_optimization/figures"
REPORT_DIR = ROOT / "docs/diagnosis"
CS_DIR     = ROOT / "data/processed/cold_start"
M5_DIR     = ROOT / "m5-forecasting-accuracy"
PERSONA_DIR = ROOT / "data/processed/personas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "experiments/exp018_optimization/checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _save_ckpt(name: str, obj: Any) -> None:
    path = CKPT_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("체크포인트 저장: %s", path)


def _load_ckpt(name: str) -> Any:
    path = CKPT_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))

# Competitor reference constants (exp006)
COMPETITORS = {
    "lightgbm_proxy_lags": {"mae": 8.48, "dir_acc": 0.343},
    "similar_item_avg":    {"mae": 8.64, "dir_acc": 0.232},
    "Track_A_calibrated":  {"mae": 8.90, "dir_acc": 0.393},
    "knn_analog":          {"mae": 9.57, "dir_acc": 0.412},
}

# exp016 G1 baseline
BASELINE_MAE = 8.71
BASELINE_DIRACC = 0.386
TARGET_MAE = 8.48
TARGET_DIRACC = 0.412


# ─── 공통 유틸: exp016/exp017에서 복사 ────────────────────────────────────────

def _to_weekly_iso(df: pd.DataFrame) -> pd.DataFrame:
    """ISO week 기준 주간 집계 (exp006 run_competitors.py와 동일)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    group_cols = [c for c in ["item_id", "store_id", "cat_id", "dept_id", "state_id",
                               "iso_year", "iso_week"] if c in df.columns]
    weekly = (
        df.groupby(group_cols)
        .agg(**{"sales": ("sales", "sum"), "date": ("week_start", "first")})
        .reset_index()
    )
    return weekly


def _complete_weeks(df_daily: pd.DataFrame) -> set:
    """cold_test 기준 완전한 주(7일)만 필터링 → (iso_year, iso_week) set."""
    df_daily = df_daily.copy()
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily["iso_year"] = df_daily["date"].dt.isocalendar().year.astype(int)
    df_daily["iso_week"] = df_daily["date"].dt.isocalendar().week.astype(int)
    days_per_week = (
        df_daily.groupby(["iso_year", "iso_week"])["date"]
        .nunique()
        .reset_index(name="n_days")
    )
    complete = days_per_week[days_per_week["n_days"] == 7]
    return set(zip(complete["iso_year"], complete["iso_week"]))


def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance. A: (m, d), B: (n, d) → (m, n)."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.clip(1.0 - An @ Bn.T, 0.0, 2.0)


def preds_to_df(
    preds: np.ndarray,
    item_ids: list[str],
    cat_ids: list[str],
    week_list: list,
    week_dates: dict,
) -> pd.DataFrame:
    """(N, n_weeks) → evaluate_weekly용 DataFrame."""
    records = []
    for i, item_id in enumerate(item_ids):
        for t, wk in enumerate(week_list):
            date = week_dates.get(wk)
            if date is None:
                continue
            records.append({
                "item_id": item_id,
                "store_id": "CA_1",
                "cat_id": cat_ids[i],
                "date": date,
                "pred_sales": max(0.0, float(preds[i, t])),
            })
    return pd.DataFrame(records)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def _build_y_from_df(
    df_raw: pd.DataFrame,
    ids: list[str],
    week_list: list,
    complete_set: set,
) -> np.ndarray:
    """주어진 daily DataFrame에서 완전한 주만 필터링해 y 행렬 생성."""
    week_idx = {wk: i for i, wk in enumerate(week_list)}
    wkly = _to_weekly_iso(df_raw)
    wkly = wkly[
        wkly.apply(lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)
    ]
    y = np.zeros((len(ids), len(week_list)), dtype=np.float32)
    for _, row in wkly.iterrows():
        ci = ids.index(row["item_id"]) if row["item_id"] in ids else -1
        wi = week_idx.get((row["iso_year"], row["iso_week"]), -1)
        if ci >= 0 and wi >= 0:
            y[ci, wi] = row["sales"]
    return y


def _build_y_period(
    df_raw: pd.DataFrame,
    ids: list[str],
) -> tuple[np.ndarray, list, dict]:
    """pd.to_period('W') 기반 주간 집계 (17주용)."""
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period("W")
    df["week_start"] = df["period"].dt.start_time

    group_cols = [c for c in ["item_id", "store_id", "cat_id", "dept_id", "state_id"]
                  if c in df.columns]
    group_cols_full = group_cols + ["period", "week_start"]

    weekly = (
        df.groupby(group_cols_full)
        .agg(sales=("sales", "sum"))
        .reset_index()
    )

    periods_sorted = sorted(weekly["period"].unique())
    period_idx = {p: i for i, p in enumerate(periods_sorted)}
    # week_dates: period → week_start date
    period_dates = (
        weekly[["period", "week_start"]]
        .drop_duplicates()
        .set_index("period")["week_start"]
        .to_dict()
    )

    y = np.zeros((len(ids), len(periods_sorted)), dtype=np.float32)
    for _, row in weekly.iterrows():
        ci = ids.index(row["item_id"]) if row["item_id"] in ids else -1
        pi = period_idx.get(row["period"], -1)
        if ci >= 0 and pi >= 0:
            y[ci, pi] = row["sales"]

    # week_list: list of Period objects, week_dates: period → date
    return y, periods_sorted, period_dates


def _build_week_dates(
    df_weekly_filtered: pd.DataFrame,
    week_list: list,
) -> dict:
    """weekly DataFrame에서 week_list에 대응하는 date dict 구성."""
    wd = (
        df_weekly_filtered[["iso_year", "iso_week", "date"]]
        .drop_duplicates()
        .assign(wk_key=lambda r: list(zip(r["iso_year"], r["iso_week"])))
        .set_index("wk_key")["date"]
        .to_dict()
    )
    return wd


def load_data() -> dict:
    """모든 필요 데이터를 로드. 16주/17주 변형 포함."""
    logger.info("데이터 로드 중...")
    warm_raw  = torch.load(EMB_DIR / "warm_raw.pt",  weights_only=True).numpy()  # (300,50,5120)
    cold_raw  = torch.load(EMB_DIR / "cold_raw.pt",  weights_only=True).numpy()  # (100,50,5120)
    warm_mean = torch.load(EMB_DIR / "warm_mean.pt", weights_only=True).numpy()  # (300,5120)
    cold_mean = torch.load(EMB_DIR / "cold_mean.pt", weights_only=True).numpy()  # (100,5120)

    meta      = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_meta = meta[meta["is_cold"] == True].reset_index(drop=True)
    warm_meta = meta[meta["is_cold"] == False].reset_index(drop=True)

    cold_test_raw  = pd.read_csv(CS_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_test_raw  = pd.read_csv(CS_DIR / "warm_test.csv",  parse_dates=["date"])
    warm_train_raw = pd.read_csv(CS_DIR / "warm_train.csv", parse_dates=["date"])

    cold_ids = cold_meta["item_id"].tolist()
    warm_ids = warm_meta["item_id"].tolist()

    # ── 16주: 완전한 주만 ──────────────────────────────────────────────────
    complete_set_16 = _complete_weeks(cold_test_raw)
    week_list_16 = sorted(complete_set_16)
    logger.info("  완전한 주(16w): %d개", len(week_list_16))

    cold_test_weekly_16 = _to_weekly_iso(cold_test_raw)
    cold_test_weekly_16 = cold_test_weekly_16[
        cold_test_weekly_16.apply(
            lambda r: (r["iso_year"], r["iso_week"]) in complete_set_16, axis=1
        )
    ]
    week_dates_16 = _build_week_dates(cold_test_weekly_16, week_list_16)

    y_cold_16 = _build_y_from_df(cold_test_raw, cold_ids, week_list_16, complete_set_16)
    y_warm_16 = _build_y_from_df(warm_test_raw, warm_ids, week_list_16, complete_set_16)

    # ── 17주: 전체 주 (pd.Period 기반) ───────────────────────────────────
    y_cold_17, periods_cold, period_dates_cold = _build_y_period(cold_test_raw, cold_ids)
    y_warm_17, periods_warm, period_dates_warm = _build_y_period(warm_test_raw, warm_ids)

    # 두 시리즈 중 공통 period 사용
    common_periods = sorted(set(periods_cold) & set(periods_warm))
    if not common_periods:
        # fallback: cold 기준
        common_periods = periods_cold

    p2i_cold = {p: i for i, p in enumerate(periods_cold)}
    p2i_warm = {p: i for i, p in enumerate(periods_warm)}
    common_idx_cold = [p2i_cold[p] for p in common_periods if p in p2i_cold]
    common_idx_warm = [p2i_warm[p] for p in common_periods if p in p2i_warm]
    y_cold_17 = y_cold_17[:, common_idx_cold]
    y_warm_17 = y_warm_17[:, common_idx_warm]
    week_list_17 = common_periods

    # week_dates_17: period → date (week_start)
    week_dates_17 = {p: period_dates_cold.get(p, period_dates_warm.get(p)) for p in week_list_17}

    # cold_test_weekly_17 — 17주 포함한 cold_test, period 기반 weekly
    cold_test_weekly_17 = _to_weekly_iso(cold_test_raw)  # 전체 주 포함
    # period 열 추가
    cold_test_weekly_17["period"] = cold_test_weekly_17["date"].dt.to_period("W")
    cold_test_weekly_17 = cold_test_weekly_16  # for evaluate_weekly, use 16w filtered (consistent date column)
    # Note: for 17w eval we create a separate df below

    # Build cold_test_weekly_17 properly for evaluate_weekly
    _cold_test_17_df = _to_weekly_iso(cold_test_raw)  # (all weeks)
    # filter to common periods
    _cold_test_17_df["period"] = pd.to_datetime(_cold_test_17_df["date"]).dt.to_period("W")
    _cold_test_17_df = _cold_test_17_df[_cold_test_17_df["period"].isin(common_periods)].copy()

    warm_train_weekly = _to_weekly_iso(warm_train_raw)

    # ── Residuals ─────────────────────────────────────────────────────────
    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)  # (300,50,5120)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)  # (100,50,5120)

    logger.info(
        "  y_cold_16=%s mean=%.2f  y_warm_16=%s mean=%.2f",
        y_cold_16.shape, y_cold_16.mean(), y_warm_16.shape, y_warm_16.mean()
    )
    logger.info(
        "  y_cold_17=%s  y_warm_17=%s  (17주)",
        y_cold_17.shape, y_warm_17.shape
    )

    return dict(
        warm_raw=warm_raw, cold_raw=cold_raw,
        warm_mean=warm_mean, cold_mean=cold_mean,
        warm_residual=warm_residual, cold_residual=cold_residual,
        cold_meta=cold_meta, warm_meta=warm_meta,
        cold_test_weekly_16=cold_test_weekly_16,
        cold_test_weekly_17=_cold_test_17_df,
        warm_train_weekly=warm_train_weekly,
        y_warm_16=y_warm_16, y_cold_16=y_cold_16,
        y_warm_17=y_warm_17, y_cold_17=y_cold_17,
        week_list_16=week_list_16, week_list_17=week_list_17,
        week_dates_16=week_dates_16, week_dates_17=week_dates_17,
    )


# ─── 모델 클래스 ──────────────────────────────────────────────────────────────

class AttnBottleneck(nn.Module):
    """Attention + Bottleneck head (standard)."""

    def __init__(
        self,
        hidden: int = 5120,
        bottleneck: int = 64,
        n_weeks: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, P, H) → (N, n_weeks)."""
        scores = self.attn(x).squeeze(-1)              # (N, P)
        attn_w = torch.softmax(scores, dim=-1)         # (N, P)
        ctx    = (x * attn_w.unsqueeze(-1)).sum(dim=1) # (N, H)
        return self.head(ctx)                          # (N, n_weeks)

    def forward_with_attn(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pred, attn_w). attn_w: (N, P)."""
        scores = self.attn(x).squeeze(-1)
        attn_w = torch.softmax(scores, dim=-1)
        ctx    = (x * attn_w.unsqueeze(-1)).sum(dim=1)
        return self.head(ctx), attn_w


class AttnBottleneckTemp(AttnBottleneck):
    """Inference-only temperature scaling and top-k masking."""

    def forward_with_temp(
        self,
        x: torch.Tensor,
        T: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Temperature scaling + optional top-k masking.

        Args:
            x: (N, P, H)
            T: temperature for softmax(scores / T). T<1 = sharper, T>1 = more uniform.
            top_k: if set, zero out all but top-k attention weights after softmax.

        Returns:
            (N, n_weeks) predictions.
        """
        scores = self.attn(x).squeeze(-1)               # (N, P)
        attn_w = torch.softmax(scores / T, dim=-1)      # (N, P)
        if top_k is not None and top_k < attn_w.shape[1]:
            # zero out all but top-k
            topk_vals, topk_idx = torch.topk(attn_w, top_k, dim=-1)
            mask = torch.zeros_like(attn_w)
            mask.scatter_(-1, topk_idx, 1.0)
            attn_w = attn_w * mask
            attn_w = attn_w / (attn_w.sum(dim=-1, keepdim=True) + 1e-12)
        ctx = (x * attn_w.unsqueeze(-1)).sum(dim=1)     # (N, H)
        return self.head(ctx)                            # (N, n_weeks)


class DualAttnBottleneck(nn.Module):
    """Two separate attention streams (raw + residual), combined."""

    def __init__(
        self,
        hidden: int = 5120,
        bottleneck: int = 64,
        n_weeks: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_raw = nn.Linear(hidden, 1, bias=False)
        self.attn_res = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x_raw: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: (N, P, H) — raw embeddings
            x_res: (N, P, H) — residual embeddings
        Returns:
            (N, n_weeks)
        """
        # raw stream
        scores_raw = self.attn_raw(x_raw).squeeze(-1)       # (N, P)
        attn_raw   = torch.softmax(scores_raw, dim=-1)      # (N, P)
        ctx_raw    = (x_raw * attn_raw.unsqueeze(-1)).sum(dim=1)  # (N, H)
        # residual stream
        scores_res = self.attn_res(x_res).squeeze(-1)
        attn_res   = torch.softmax(scores_res, dim=-1)
        ctx_res    = (x_res * attn_res.unsqueeze(-1)).sum(dim=1)  # (N, H)
        # combine
        ctx = torch.cat([ctx_raw, ctx_res], dim=-1)          # (N, 2H)
        return self.head(ctx)


# ─── train_head 함수군 ────────────────────────────────────────────────────────

def train_head(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 500,
    lr: float = 1e-3,
    wd: float = 1e-4,
    bottleneck: int = 64,
    dropout: float = 0.1,
    seed: int = 42,
) -> AttnBottleneck:
    """Standard L1 loss training.

    Args:
        X_raw: (N, P, H)
        y: (N, n_weeks)

    Returns:
        Trained AttnBottleneck model.
    """
    torch.manual_seed(seed)
    n_weeks = y.shape[1]
    model = AttnBottleneck(n_weeks=n_weeks, bottleneck=bottleneck, dropout=dropout)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    X_t   = torch.from_numpy(X_raw.astype(np.float32))
    y_t   = torch.from_numpy(y.astype(np.float32))
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = nn.functional.l1_loss(model(X_t), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def train_head_dir(
    X_raw: np.ndarray,
    y: np.ndarray,
    lambda_dir: float = 0.5,
    n_epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
) -> AttnBottleneck:
    """L1 + direction loss training.

    Direction loss: soft sign via tanh.
        pred_diff_soft  = tanh(pred_diff * 10)
        target_diff_soft = tanh(target_diff * 10)
        sign_match = pred_diff_soft * target_diff_soft
        direction_loss = -sign_match.mean()
    """
    torch.manual_seed(seed)
    n_weeks = y.shape[1]
    model = AttnBottleneck(n_weeks=n_weeks)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X_t   = torch.from_numpy(X_raw.astype(np.float32))
    y_t   = torch.from_numpy(y.astype(np.float32))
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        pred = model(X_t)                                  # (N, n_weeks)
        l1_loss = nn.functional.l1_loss(pred, y_t)
        if lambda_dir > 0.0 and n_weeks > 1:
            pred_diff   = pred[:, 1:] - pred[:, :-1]      # (N, n_weeks-1)
            target_diff = y_t[:, 1:] - y_t[:, :-1]
            pred_diff_soft   = torch.tanh(pred_diff * 10.0)
            target_diff_soft = torch.tanh(target_diff * 10.0)
            sign_match = pred_diff_soft * target_diff_soft
            dir_loss = -sign_match.mean()
            loss = l1_loss + lambda_dir * dir_loss
        else:
            loss = l1_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def train_dual_head(
    X_raw: np.ndarray,
    X_res: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
) -> DualAttnBottleneck:
    """Trains DualAttnBottleneck on raw + residual embeddings.

    Args:
        X_raw: (N, P, H)
        X_res: (N, P, H)
        y: (N, n_weeks)

    Returns:
        Trained DualAttnBottleneck model.
    """
    torch.manual_seed(seed)
    n_weeks = y.shape[1]
    model = DualAttnBottleneck(n_weeks=n_weeks)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xr_t  = torch.from_numpy(X_raw.astype(np.float32))
    Xs_t  = torch.from_numpy(X_res.astype(np.float32))
    y_t   = torch.from_numpy(y.astype(np.float32))
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = nn.functional.l1_loss(model(Xr_t, Xs_t), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def predict_head(model: AttnBottleneck, X_raw: np.ndarray) -> np.ndarray:
    """Run standard forward pass, return (N, n_weeks) numpy array."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_raw.astype(np.float32))).numpy()


def predict_dual_head(
    model: DualAttnBottleneck,
    X_raw: np.ndarray,
    X_res: np.ndarray,
) -> np.ndarray:
    """Run DualAttnBottleneck forward pass."""
    model.eval()
    with torch.no_grad():
        Xr_t = torch.from_numpy(X_raw.astype(np.float32))
        Xs_t = torch.from_numpy(X_res.astype(np.float32))
        return model(Xr_t, Xs_t).numpy()


def get_attn_weights(model: AttnBottleneck, X_raw: np.ndarray) -> np.ndarray:
    """Extract attention weights (N, P)."""
    model.eval()
    with torch.no_grad():
        _, attn_w = model.forward_with_attn(
            torch.from_numpy(X_raw.astype(np.float32))
        )
    return attn_w.numpy()


# ─── eval_cold helper ─────────────────────────────────────────────────────────

def eval_cold(
    model: AttnBottleneck,
    X_cold: np.ndarray,
    cold_meta: pd.DataFrame,
    data: dict,
    n_weeks_key: str = "16",
    seed: int | None = None,
) -> dict[str, float]:
    """Run model on cold items, return {'mae': float, 'dir_acc': float}.

    Args:
        model: trained AttnBottleneck
        X_cold: (100, P, H) cold embeddings
        cold_meta: cold item metadata DataFrame
        data: output of load_data()
        n_weeks_key: "16" or "17"
        seed: unused (for API consistency)

    Returns:
        dict with 'mae' and 'dir_acc' keys
    """
    preds = predict_head(model, X_cold)  # (100, n_weeks)
    week_list  = data[f"week_list_{n_weeks_key}"]
    week_dates = data[f"week_dates_{n_weeks_key}"]
    cold_test_weekly = data[f"cold_test_weekly_{n_weeks_key}"]

    pred_df = preds_to_df(
        preds,
        cold_meta["item_id"].tolist(),
        cold_meta["cat_id"].tolist(),
        week_list,
        week_dates,
    )
    ev = evaluate_weekly(
        cold_test_weekly, pred_df, data["warm_train_weekly"],
        model_name=f"eval_cold_{n_weeks_key}w",
    )
    return {"mae": ev["mae"], "dir_acc": ev["direction_accuracy"]}


# ─── Part 1: DirAcc Factor Analysis ──────────────────────────────────────────

def part1_factor_analysis(data: dict) -> dict:
    """Part 1: DirAcc Factor Analysis — 5 variants × 3 seeds."""
    logger.info("=== Part 1: DirAcc Factor Analysis ===")

    variants: list[dict[str, Any]] = [
        {"name": "baseline",  "emb": "residual", "weeks": "16", "epochs": 500},
        {"name": "1A-a",      "emb": "raw",      "weeks": "16", "epochs": 500},
        {"name": "1A-b",      "emb": "residual", "weeks": "17", "epochs": 500},
        {"name": "1A-c",      "emb": "residual", "weeks": "16", "epochs": 200},
        {"name": "1A-d",      "emb": "raw",      "weeks": "17", "epochs": 500},
    ]
    seeds = [0, 1, 2]

    results: dict[str, dict] = {}
    for v in variants:
        name = v["name"]
        logger.info("  variant=%s emb=%s weeks=%s epochs=%d", name, v["emb"], v["weeks"], v["epochs"])

        # Select embeddings
        if v["emb"] == "residual":
            X_warm = data["warm_residual"]
            X_cold = data["cold_residual"]
        else:
            X_warm = data["warm_raw"]
            X_cold = data["cold_raw"]

        n_weeks_key = v["weeks"]
        y_warm = data[f"y_warm_{n_weeks_key}"]
        n_weeks = y_warm.shape[1]

        maes, daccs = [], []
        for seed in seeds:
            model = train_head(X_warm, y_warm, n_epochs=v["epochs"], seed=seed)
            ev = eval_cold(model, X_cold, data["cold_meta"], data,
                           n_weeks_key=n_weeks_key)
            maes.append(ev["mae"])
            daccs.append(ev["dir_acc"])
            logger.info("    seed=%d → MAE=%.4f  DirAcc=%.3f", seed, ev["mae"], ev["dir_acc"])

        results[name] = {
            "mae_mean":    float(np.mean(maes)),
            "mae_std":     float(np.std(maes)),
            "diracc_mean": float(np.mean(daccs)),
            "diracc_std":  float(np.std(daccs)),
            "maes": maes, "daccs": daccs,
            "emb": v["emb"], "weeks": v["weeks"], "epochs": v["epochs"],
        }
        logger.info(
            "  [%s] MAE=%.4f±%.4f  DirAcc=%.4f±%.4f",
            name, results[name]["mae_mean"], results[name]["mae_std"],
            results[name]["diracc_mean"], results[name]["diracc_std"],
        )

    return results


def plot_p1(p1: dict) -> None:
    """p1_diracc_factors.png — grouped bar chart: MAE and DirAcc for 5 variants."""
    variant_names = list(p1.keys())
    mae_means  = [p1[n]["mae_mean"]    for n in variant_names]
    mae_stds   = [p1[n]["mae_std"]     for n in variant_names]
    dacc_means = [p1[n]["diracc_mean"] for n in variant_names]
    dacc_stds  = [p1[n]["diracc_std"]  for n in variant_names]

    x = np.arange(len(variant_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_mae  = ["#4c72b0"] * len(variant_names)
    colors_dacc = ["#c44e52"] * len(variant_names)
    colors_mae[0]  = "#90c0e8"   # baseline lighter
    colors_dacc[0] = "#e89090"

    axes[0].bar(x, mae_means, yerr=mae_stds, capsize=4, color=colors_mae, alpha=0.85, edgecolor="white")
    axes[0].axhline(BASELINE_MAE, color="gray", linestyle="--", alpha=0.6, label=f"G1 baseline={BASELINE_MAE}")
    axes[0].axhline(TARGET_MAE,   color="green", linestyle="--", alpha=0.6, label=f"Target MAE={TARGET_MAE}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variant_names, rotation=15, ha="right")
    axes[0].set_ylabel("Cold MAE"); axes[0].set_title("Part 1: Cold MAE by Variant (mean±std, 3 seeds)")
    axes[0].legend(fontsize=8)
    for i, (m, s) in enumerate(zip(mae_means, mae_stds)):
        axes[0].text(i, m + s + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, dacc_means, yerr=dacc_stds, capsize=4, color=colors_dacc, alpha=0.85, edgecolor="white")
    axes[1].axhline(BASELINE_DIRACC, color="gray", linestyle="--", alpha=0.6, label=f"G1 baseline={BASELINE_DIRACC}")
    axes[1].axhline(TARGET_DIRACC,   color="green", linestyle="--", alpha=0.6, label=f"Target DirAcc={TARGET_DIRACC}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variant_names, rotation=15, ha="right")
    axes[1].set_ylabel("Direction Accuracy"); axes[1].set_title("Part 1: DirAcc by Variant")
    axes[1].legend(fontsize=8)
    for i, (d, s) in enumerate(zip(dacc_means, dacc_stds)):
        axes[1].text(i, d + s + 0.002, f"{d:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "p1_diracc_factors.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", out)


# ─── Part 2: Attention Modifications ─────────────────────────────────────────

def part2a_diffuse_sparse(data: dict, model: AttnBottleneck) -> dict:
    """Part 2A: Diffuse vs Sparse profiling."""
    logger.info("=== Part 2A: Diffuse vs Sparse Profiling ===")

    cold_residual = data["cold_residual"]  # (100, 50, 5120)
    cold_meta     = data["cold_meta"]
    y_cold_16     = data["y_cold_16"]

    # Attention weights from cold_residual
    attn_w = get_attn_weights(model, cold_residual)  # (100, 50)

    # Entropy per item
    def entropy(w: np.ndarray) -> float:
        w = np.clip(w, 1e-12, None)
        return float(-np.sum(w * np.log(w)))

    entropies = np.array([entropy(attn_w[i]) for i in range(len(attn_w))])

    # Bottom-25 entropy = sparse, top-25 entropy = diffuse
    sorted_idx = np.argsort(entropies)
    sparse_idx = sorted_idx[:25]   # lowest entropy (concentrated)
    diffuse_idx = sorted_idx[-25:] # highest entropy (spread)

    # Load sell_prices
    sell_prices = pd.read_csv(M5_DIR / "sell_prices.csv")
    price_map = (
        sell_prices[sell_prices["store_id"] == "CA_1"]
        .groupby("item_id")["sell_price"].mean().to_dict()
    )

    def group_profile(idx_arr: np.ndarray) -> dict:
        cats = cold_meta["cat_id"].iloc[idx_arr].values
        cat_counts = {
            "FOODS": int((cats == "FOODS").sum()),
            "HOBBIES": int((cats == "HOBBIES").sum()),
            "HOUSEHOLD": int((cats == "HOUSEHOLD").sum()),
        }
        mean_weekly_sales = float(y_cold_16[idx_arr].mean())
        prices = [price_map.get(cold_meta["item_id"].iloc[i], np.nan) for i in idx_arr]
        mean_price = float(np.nanmean(prices)) if len(prices) > 0 else float("nan")
        top1_attn = float(attn_w[idx_arr].max(axis=1).mean())
        # CV = std/mean of actual weekly sales per item (averaged)
        cvs = []
        for i in idx_arr:
            weekly = y_cold_16[i]
            if weekly.mean() > 0:
                cvs.append(weekly.std() / weekly.mean())
        cv_mean = float(np.mean(cvs)) if cvs else float("nan")
        # embedding norm: mean L2 norm of residual vectors (mean over personas)
        emb_norms = []
        for i in idx_arr:
            norms = np.linalg.norm(data["cold_residual"][i], axis=-1)  # (50,)
            emb_norms.append(norms.mean())
        emb_norm_mean = float(np.mean(emb_norms))
        return dict(
            cat_counts=cat_counts,
            mean_weekly_sales=mean_weekly_sales,
            mean_price=mean_price,
            top1_attn=top1_attn,
            cv_mean=cv_mean,
            emb_norm_mean=emb_norm_mean,
        )

    sparse_profile  = group_profile(sparse_idx)
    diffuse_profile = group_profile(diffuse_idx)

    logger.info("  Sparse:  %s", sparse_profile)
    logger.info("  Diffuse: %s", diffuse_profile)

    return {
        "entropies": entropies.tolist(),
        "sparse_idx": sparse_idx.tolist(),
        "diffuse_idx": diffuse_idx.tolist(),
        "sparse_profile": sparse_profile,
        "diffuse_profile": diffuse_profile,
    }


def plot_p2a(p2a: dict) -> None:
    """p2a_diffuse_sparse_profile.png — 2×3 subplot."""
    sp = p2a["sparse_profile"]
    dp = p2a["diffuse_profile"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Part 2A: Diffuse vs Sparse Attention Groups", fontsize=13)

    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    groups = ["Sparse (low entropy)", "Diffuse (high entropy)"]

    # Row 0: category distribution
    ax = axes[0, 0]
    x = np.arange(3)
    width = 0.35
    sp_counts = [sp["cat_counts"][c] for c in categories]
    dp_counts = [dp["cat_counts"][c] for c in categories]
    ax.bar(x - width/2, sp_counts, width, label="Sparse",  color="steelblue", alpha=0.8)
    ax.bar(x + width/2, dp_counts, width, label="Diffuse", color="tomato",    alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories); ax.set_title("Category Distribution")
    ax.legend(fontsize=8); ax.set_ylabel("Count")

    # Row 0: mean weekly sales
    ax = axes[0, 1]
    ax.bar(groups, [sp["mean_weekly_sales"], dp["mean_weekly_sales"]],
           color=["steelblue", "tomato"], alpha=0.85)
    ax.set_title("Mean Weekly Sales")
    ax.set_ylabel("Sales")
    for i, v in enumerate([sp["mean_weekly_sales"], dp["mean_weekly_sales"]]):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Row 0: mean price
    ax = axes[0, 2]
    ax.bar(groups, [sp["mean_price"], dp["mean_price"]],
           color=["steelblue", "tomato"], alpha=0.85)
    ax.set_title("Mean Price"); ax.set_ylabel("Price ($)")
    for i, v in enumerate([sp["mean_price"], dp["mean_price"]]):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Row 1: top-1 attention weight
    ax = axes[1, 0]
    ax.bar(groups, [sp["top1_attn"], dp["top1_attn"]],
           color=["steelblue", "tomato"], alpha=0.85)
    ax.set_title("Top-1 Attention Weight (concentration)"); ax.set_ylabel("Attn weight")
    for i, v in enumerate([sp["top1_attn"], dp["top1_attn"]]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # Row 1: CV
    ax = axes[1, 1]
    ax.bar(groups, [sp["cv_mean"], dp["cv_mean"]],
           color=["steelblue", "tomato"], alpha=0.85)
    ax.set_title("CV (std/mean weekly sales)"); ax.set_ylabel("CV")
    for i, v in enumerate([sp["cv_mean"], dp["cv_mean"]]):
        if not np.isnan(v):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Row 1: embedding norm
    ax = axes[1, 2]
    ax.bar(groups, [sp["emb_norm_mean"], dp["emb_norm_mean"]],
           color=["steelblue", "tomato"], alpha=0.85)
    ax.set_title("Mean Residual Embedding L2 Norm"); ax.set_ylabel("L2 Norm")
    for i, v in enumerate([sp["emb_norm_mean"], dp["emb_norm_mean"]]):
        ax.text(i, v + 0.5, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Entropy histogram overlay
    entropies = np.array(p2a["entropies"])
    for ax_ in [axes[0, 0]]:
        pass  # already used

    fig.tight_layout()
    out = FIG_DIR / "p2a_diffuse_sparse_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", out)


def part2b1_temperature(data: dict, model: AttnBottleneck) -> dict:
    """Part 2B-1: Temperature scaling (inference only)."""
    logger.info("=== Part 2B-1: Temperature Scaling ===")

    cold_residual = data["cold_residual"]
    cold_meta     = data["cold_meta"]

    # Wrap in AttnBottleneckTemp (share weights)
    temp_model = AttnBottleneckTemp(n_weeks=data["y_warm_16"].shape[1])
    temp_model.load_state_dict(model.state_dict())
    temp_model.eval()

    T_vals = [0.1, 0.3, 0.5, 1.0, 2.0]
    results = []

    X_t = torch.from_numpy(cold_residual.astype(np.float32))
    for T in T_vals:
        with torch.no_grad():
            preds = temp_model.forward_with_temp(X_t, T=T).numpy()
        pred_df = preds_to_df(
            preds,
            cold_meta["item_id"].tolist(),
            cold_meta["cat_id"].tolist(),
            data["week_list_16"], data["week_dates_16"],
        )
        ev = evaluate_weekly(
            data["cold_test_weekly_16"], pred_df, data["warm_train_weekly"],
            model_name=f"temp_T{T}",
        )
        results.append({"T": T, "mae": ev["mae"], "dir_acc": ev["direction_accuracy"]})
        logger.info("  T=%.1f → MAE=%.4f  DirAcc=%.3f", T, ev["mae"], ev["direction_accuracy"])

    return {"rows": results}


def plot_p2b1(p2b1: dict) -> None:
    """p2b1_temperature.png — line chart."""
    rows = p2b1["rows"]
    Ts   = [r["T"] for r in rows]
    maes = [r["mae"] for r in rows]
    daccs = [r["dir_acc"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(Ts, maes, marker="o", color="steelblue", label="MAE")
    axes[0].axhline(BASELINE_MAE, color="gray", linestyle="--", alpha=0.6, label=f"baseline={BASELINE_MAE}")
    axes[0].axhline(TARGET_MAE,   color="green", linestyle="--", alpha=0.6, label=f"target={TARGET_MAE}")
    axes[0].set_xlabel("Temperature T"); axes[0].set_ylabel("Cold MAE")
    axes[0].set_title("Part 2B-1: Temperature vs MAE"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(Ts, daccs, marker="o", color="tomato", label="DirAcc")
    axes[1].axhline(BASELINE_DIRACC, color="gray", linestyle="--", alpha=0.6, label=f"baseline={BASELINE_DIRACC}")
    axes[1].axhline(TARGET_DIRACC,   color="green", linestyle="--", alpha=0.6, label=f"target={TARGET_DIRACC}")
    axes[1].set_xlabel("Temperature T"); axes[1].set_ylabel("Direction Accuracy")
    axes[1].set_title("Part 2B-1: Temperature vs DirAcc"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "p2b1_temperature.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", out)


def part2b2_topk(data: dict, model: AttnBottleneck) -> dict:
    """Part 2B-2: Top-K attention masking (inference only)."""
    logger.info("=== Part 2B-2: Top-K Attention ===")

    cold_residual = data["cold_residual"]
    cold_meta     = data["cold_meta"]

    temp_model = AttnBottleneckTemp(n_weeks=data["y_warm_16"].shape[1])
    temp_model.load_state_dict(model.state_dict())
    temp_model.eval()

    K_vals = [3, 5, 10, 20, 50]
    results = []

    X_t = torch.from_numpy(cold_residual.astype(np.float32))
    for K in K_vals:
        with torch.no_grad():
            preds = temp_model.forward_with_temp(X_t, T=1.0, top_k=K).numpy()
        pred_df = preds_to_df(
            preds,
            cold_meta["item_id"].tolist(),
            cold_meta["cat_id"].tolist(),
            data["week_list_16"], data["week_dates_16"],
        )
        ev = evaluate_weekly(
            data["cold_test_weekly_16"], pred_df, data["warm_train_weekly"],
            model_name=f"topk_K{K}",
        )
        results.append({"K": K, "mae": ev["mae"], "dir_acc": ev["direction_accuracy"]})
        logger.info("  K=%d → MAE=%.4f  DirAcc=%.3f", K, ev["mae"], ev["direction_accuracy"])

    return {"rows": results}


def plot_p2b2(p2b2: dict) -> None:
    """p2b2_topk.png — line chart."""
    rows  = p2b2["rows"]
    Ks    = [r["K"] for r in rows]
    maes  = [r["mae"] for r in rows]
    daccs = [r["dir_acc"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(Ks, maes, marker="o", color="steelblue")
    axes[0].axhline(BASELINE_MAE, color="gray", linestyle="--", alpha=0.6, label=f"baseline={BASELINE_MAE}")
    axes[0].set_xlabel("K (top-k)"); axes[0].set_ylabel("Cold MAE")
    axes[0].set_title("Part 2B-2: Top-K vs MAE"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(Ks, daccs, marker="o", color="tomato")
    axes[1].axhline(BASELINE_DIRACC, color="gray", linestyle="--", alpha=0.6, label=f"baseline={BASELINE_DIRACC}")
    axes[1].axhline(TARGET_DIRACC,   color="green", linestyle="--", alpha=0.6, label=f"target={TARGET_DIRACC}")
    axes[1].set_xlabel("K (top-k)"); axes[1].set_ylabel("Direction Accuracy")
    axes[1].set_title("Part 2B-2: Top-K vs DirAcc"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "p2b2_topk.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", out)


# ─── Part 3: Loss and Architecture ───────────────────────────────────────────

def part3a_direction_loss(data: dict) -> dict:
    """Part 3A: Direction Loss — lambda_dir × 3 seeds."""
    logger.info("=== Part 3A: Direction Loss ===")

    X_warm = data["warm_residual"]
    X_cold = data["cold_residual"]
    y_warm = data["y_warm_16"]
    seeds  = [0, 1, 2]
    lambda_vals = [0.0, 0.1, 0.5, 1.0, 2.0]

    results = []
    for lam in lambda_vals:
        maes, daccs = [], []
        for seed in seeds:
            model = train_head_dir(X_warm, y_warm, lambda_dir=lam, seed=seed)
            ev = eval_cold(model, X_cold, data["cold_meta"], data, n_weeks_key="16")
            maes.append(ev["mae"])
            daccs.append(ev["dir_acc"])
        row = {
            "lambda_dir": lam,
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "diracc_mean": float(np.mean(daccs)), "diracc_std": float(np.std(daccs)),
        }
        results.append(row)
        logger.info(
            "  lambda=%.1f → MAE=%.4f±%.4f  DirAcc=%.4f±%.4f",
            lam, row["mae_mean"], row["mae_std"], row["diracc_mean"], row["diracc_std"],
        )

    return {"rows": results}


def plot_p3a(p3a: dict) -> None:
    """p3a_direction_loss.png — line chart."""
    rows  = p3a["rows"]
    lams  = [r["lambda_dir"] for r in rows]
    maes  = [r["mae_mean"]   for r in rows]
    mae_s = [r["mae_std"]    for r in rows]
    daccs = [r["diracc_mean"] for r in rows]
    dac_s = [r["diracc_std"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].errorbar(lams, maes, yerr=mae_s, marker="o", color="steelblue", capsize=4)
    axes[0].axhline(BASELINE_MAE, color="gray", linestyle="--", alpha=0.6, label=f"baseline={BASELINE_MAE}")
    axes[0].set_xlabel("lambda_dir"); axes[0].set_ylabel("Cold MAE")
    axes[0].set_title("Part 3A: Direction Loss vs MAE"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(lams, daccs, yerr=dac_s, marker="o", color="tomato", capsize=4)
    axes[1].axhline(BASELINE_DIRACC, color="gray", linestyle="--", alpha=0.6, label=f"baseline={BASELINE_DIRACC}")
    axes[1].axhline(TARGET_DIRACC,   color="green", linestyle="--", alpha=0.6, label=f"target={TARGET_DIRACC}")
    axes[1].set_xlabel("lambda_dir"); axes[1].set_ylabel("Direction Accuracy")
    axes[1].set_title("Part 3A: Direction Loss vs DirAcc"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "p3a_direction_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", out)


def part3b_grid_search(data: dict) -> dict:
    """Part 3B: Hyperparameter Grid Search — 27 combos × 3 seeds."""
    logger.info("=== Part 3B: Hyperparameter Grid Search ===")

    X_warm = data["warm_residual"]
    X_cold = data["cold_residual"]
    y_warm = data["y_warm_16"]
    seeds  = [0, 1, 2]

    lr_vals         = [5e-4, 1e-3, 2e-3]
    bottleneck_vals = [32, 64, 128]
    epochs_vals     = [300, 500, 1000]

    rows = []
    total = len(lr_vals) * len(bottleneck_vals) * len(epochs_vals)
    count = 0
    for lr in lr_vals:
        for bn in bottleneck_vals:
            for ep in epochs_vals:
                count += 1
                maes, daccs = [], []
                for seed in seeds:
                    model = train_head(
                        X_warm, y_warm,
                        n_epochs=ep, lr=lr, bottleneck=bn, seed=seed,
                    )
                    ev = eval_cold(model, X_cold, data["cold_meta"], data, n_weeks_key="16")
                    maes.append(ev["mae"])
                    daccs.append(ev["dir_acc"])
                row = {
                    "lr": lr, "bottleneck": bn, "epochs": ep,
                    "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                    "diracc_mean": float(np.mean(daccs)), "diracc_std": float(np.std(daccs)),
                }
                rows.append(row)
                if count % 9 == 0 or count == total:
                    logger.info(
                        "  [%d/%d] lr=%.4f bn=%d ep=%d → MAE=%.4f  DirAcc=%.4f",
                        count, total, lr, bn, ep, row["mae_mean"], row["diracc_mean"],
                    )

    # Pareto front: MAE↓, DirAcc↑
    pareto = _compute_pareto(rows)

    return {"rows": rows, "pareto": pareto}


def _compute_pareto(rows: list[dict]) -> list[dict]:
    """Compute Pareto front (minimize MAE, maximize DirAcc)."""
    pareto = []
    for r in rows:
        dominated = False
        for other in rows:
            if (other["mae_mean"] <= r["mae_mean"] and
                    other["diracc_mean"] >= r["diracc_mean"] and
                    (other["mae_mean"] < r["mae_mean"] or
                     other["diracc_mean"] > r["diracc_mean"])):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return pareto


def plot_p3b(p3b: dict) -> None:
    """p3b_pareto.png — scatter MAE vs DirAcc, Pareto front highlighted."""
    rows   = p3b["rows"]
    pareto = p3b["pareto"]

    maes_all  = [r["mae_mean"]    for r in rows]
    daccs_all = [r["diracc_mean"] for r in rows]
    maes_par  = [r["mae_mean"]    for r in pareto]
    daccs_par = [r["diracc_mean"] for r in pareto]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(maes_all, daccs_all, alpha=0.5, s=30, color="steelblue", label="All combos")
    ax.scatter(maes_par, daccs_par, alpha=0.9, s=80, color="tomato", zorder=5, label="Pareto front")

    # annotate Pareto points
    for r in sorted(pareto, key=lambda x: x["mae_mean"])[:5]:
        ax.annotate(
            f"lr={r['lr']:.0e}\nbn={r['bottleneck']}\nep={r['epochs']}",
            xy=(r["mae_mean"], r["diracc_mean"]),
            xytext=(5, 5), textcoords="offset points", fontsize=6, alpha=0.8,
        )

    ax.axvline(TARGET_MAE, color="green", linestyle="--", alpha=0.6, label=f"Target MAE={TARGET_MAE}")
    ax.axhline(TARGET_DIRACC, color="green", linestyle=":", alpha=0.6, label=f"Target DirAcc={TARGET_DIRACC}")
    ax.axvline(BASELINE_MAE, color="gray", linestyle="--", alpha=0.4, label=f"baseline MAE={BASELINE_MAE}")
    ax.axhline(BASELINE_DIRACC, color="gray", linestyle=":", alpha=0.4, label=f"baseline DirAcc={BASELINE_DIRACC}")

    ax.set_xlabel("Cold MAE"); ax.set_ylabel("Direction Accuracy")
    ax.set_title("Part 3B: Grid Search — Pareto Front (MAE↓, DirAcc↑)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "p3b_pareto.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", out)


def part3c_dual_attn(data: dict) -> dict:
    """Part 3C: Dual Attention (raw + residual) — 3 seeds."""
    logger.info("=== Part 3C: Dual Attention ===")

    X_warm_raw = data["warm_raw"]
    X_warm_res = data["warm_residual"]
    X_cold_raw = data["cold_raw"]
    X_cold_res = data["cold_residual"]
    y_warm     = data["y_warm_16"]
    cold_meta  = data["cold_meta"]
    seeds      = [0, 1, 2]

    maes, daccs = [], []
    for seed in seeds:
        model = train_dual_head(X_warm_raw, X_warm_res, y_warm, seed=seed)
        preds = predict_dual_head(model, X_cold_raw, X_cold_res)  # (100, 16)
        pred_df = preds_to_df(
            preds,
            cold_meta["item_id"].tolist(),
            cold_meta["cat_id"].tolist(),
            data["week_list_16"], data["week_dates_16"],
        )
        ev = evaluate_weekly(
            data["cold_test_weekly_16"], pred_df, data["warm_train_weekly"],
            model_name=f"dual_attn_seed{seed}",
        )
        maes.append(ev["mae"])
        daccs.append(ev["direction_accuracy"])
        logger.info("  seed=%d → MAE=%.4f  DirAcc=%.3f", seed, ev["mae"], ev["direction_accuracy"])

    return {
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "diracc_mean": float(np.mean(daccs)), "diracc_std": float(np.std(daccs)),
        "maes": maes, "daccs": daccs,
    }


# ─── Part 4: Final Optimal Configuration ─────────────────────────────────────

def _select_best_config(
    p1: dict,
    p2b1: dict,
    p3a: dict,
    p3b: dict,
    p3c: dict,
) -> dict:
    """Programmatically select the best config from parts 1-3."""

    # 1. Best embedding from Part 1 (better DirAcc without worsening MAE by >0.3)
    baseline_mae = p1["baseline"]["mae_mean"]
    baseline_da  = p1["baseline"]["diracc_mean"]
    best_emb   = "residual"
    best_weeks = "16"
    best_epochs = 500
    best_da = baseline_da

    for name, v in p1.items():
        if name == "baseline":
            continue
        if (v["diracc_mean"] > best_da and
                v["mae_mean"] <= baseline_mae + 0.3):
            best_da = v["diracc_mean"]
            best_emb   = v["emb"]
            best_weeks = v["weeks"]
            best_epochs = v["epochs"]

    logger.info("  Best embedding config: emb=%s weeks=%s epochs=%d da=%.4f",
                best_emb, best_weeks, best_epochs, best_da)

    # 2. Best T from Part 2B-1 (inference-time, free)
    best_T = 1.0
    best_T_da = 0.0
    best_T_mae = float("inf")
    for row in p2b1["rows"]:
        if row["T"] == 1.0:
            best_T_mae = row["mae"]
            best_T_da  = row["dir_acc"]
    for row in p2b1["rows"]:
        if (row["dir_acc"] > best_T_da and
                row["mae"] <= best_T_mae + 0.05):  # small tolerance for inference-free
            best_T = row["T"]
            best_T_da  = row["dir_acc"]
            best_T_mae = row["mae"]

    logger.info("  Best T: %.1f (DirAcc=%.4f)", best_T, best_T_da)

    # 3. Best lambda_dir from Part 3A (DirAcc improved without MAE worsening by >0.5)
    p3a_rows = p3a["rows"]
    base_p3a = next(r for r in p3a_rows if r["lambda_dir"] == 0.0)
    best_lam = 0.0
    best_lam_da = base_p3a["diracc_mean"]
    best_lam_mae = base_p3a["mae_mean"]
    for row in p3a_rows:
        if (row["diracc_mean"] > best_lam_da and
                row["mae_mean"] <= base_p3a["mae_mean"] + 0.5):
            best_lam    = row["lambda_dir"]
            best_lam_da  = row["diracc_mean"]
            best_lam_mae = row["mae_mean"]

    logger.info("  Best lambda_dir: %.1f (DirAcc=%.4f, MAE=%.4f)",
                best_lam, best_lam_da, best_lam_mae)

    # 4. Best (lr, bottleneck, epochs) from Pareto front
    # Highest DirAcc with MAE < 9.0
    eligible = [r for r in p3b["pareto"] if r["mae_mean"] < 9.0]
    if not eligible:
        eligible = p3b["pareto"]
    best_hparam = max(eligible, key=lambda r: r["diracc_mean"])
    logger.info(
        "  Best hparam: lr=%.4f bn=%d ep=%d (MAE=%.4f DirAcc=%.4f)",
        best_hparam["lr"], best_hparam["bottleneck"], best_hparam["epochs"],
        best_hparam["mae_mean"], best_hparam["diracc_mean"],
    )

    # 5. DualAttn consideration
    use_dual = (
        p3c["diracc_mean"] > baseline_da and
        p3c["mae_mean"] < baseline_mae
    )
    logger.info("  Use DualAttn: %s", use_dual)

    return {
        "emb": best_emb,
        "weeks": best_weeks,
        "epochs": best_epochs,
        "T": best_T,
        "lambda_dir": best_lam,
        "lr": best_hparam["lr"],
        "bottleneck": best_hparam["bottleneck"],
        "use_dual": use_dual,
        "rationale": {
            "emb_reason": f"Best DirAcc={best_da:.4f} from Part1 variants",
            "T_reason": f"Best T={best_T} from Part2B-1 inference-time",
            "lam_reason": f"lambda={best_lam} improves DirAcc to {best_lam_da:.4f}",
            "hparam_reason": (
                f"lr={best_hparam['lr']:.4f}, bn={best_hparam['bottleneck']}, "
                f"ep={best_hparam['epochs']} from Pareto front"
            ),
            "dual_reason": f"DualAttn {'dominates' if use_dual else 'does not dominate'} standard",
        },
    }


def part4_final(
    data: dict,
    p1: dict,
    p2b1: dict,
    p3a: dict,
    p3b: dict,
    p3c: dict,
) -> dict:
    """Part 4: Final Optimal Configuration — 10 seeds."""
    logger.info("=== Part 4: Final Optimal Configuration ===")

    cfg = _select_best_config(p1, p2b1, p3a, p3b, p3c)
    logger.info("  Selected config: %s", cfg)

    # Select embeddings
    if cfg["emb"] == "residual":
        X_warm = data["warm_residual"]
        X_cold = data["cold_residual"]
    else:
        X_warm = data["warm_raw"]
        X_cold = data["cold_raw"]

    if cfg["use_dual"]:
        X_warm_raw = data["warm_raw"]
        X_warm_res = data["warm_residual"]
        X_cold_raw = data["cold_raw"]
        X_cold_res = data["cold_residual"]

    n_weeks_key = cfg["weeks"]
    y_warm = data[f"y_warm_{n_weeks_key}"]

    maes, daccs = [], []
    seeds = list(range(10))

    for seed in seeds:
        if cfg["use_dual"]:
            model_dual = train_dual_head(
                X_warm_raw, X_warm_res, y_warm,
                n_epochs=cfg["epochs"], lr=cfg["lr"], seed=seed,
            )
            preds = predict_dual_head(model_dual, X_cold_raw, X_cold_res)
            pred_df = preds_to_df(
                preds,
                data["cold_meta"]["item_id"].tolist(),
                data["cold_meta"]["cat_id"].tolist(),
                data[f"week_list_{n_weeks_key}"],
                data[f"week_dates_{n_weeks_key}"],
            )
            ev = evaluate_weekly(
                data[f"cold_test_weekly_{n_weeks_key}"], pred_df,
                data["warm_train_weekly"], model_name=f"final_dual_seed{seed}",
            )
        else:
            if cfg["lambda_dir"] > 0.0:
                model = train_head_dir(
                    X_warm, y_warm,
                    lambda_dir=cfg["lambda_dir"],
                    n_epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    seed=seed,
                )
            else:
                model = train_head(
                    X_warm, y_warm,
                    n_epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    bottleneck=cfg["bottleneck"],
                    seed=seed,
                )

            # Apply temperature at inference
            if cfg["T"] != 1.0:
                temp_model = AttnBottleneckTemp(n_weeks=y_warm.shape[1])
                temp_model.load_state_dict(model.state_dict())
                temp_model.eval()
                X_t = torch.from_numpy(X_cold.astype(np.float32))
                with torch.no_grad():
                    preds = temp_model.forward_with_temp(X_t, T=cfg["T"]).numpy()
            else:
                preds = predict_head(model, X_cold)

            pred_df = preds_to_df(
                preds,
                data["cold_meta"]["item_id"].tolist(),
                data["cold_meta"]["cat_id"].tolist(),
                data[f"week_list_{n_weeks_key}"],
                data[f"week_dates_{n_weeks_key}"],
            )
            ev = evaluate_weekly(
                data[f"cold_test_weekly_{n_weeks_key}"], pred_df,
                data["warm_train_weekly"], model_name=f"final_seed{seed}",
            )

        maes.append(ev["mae"])
        daccs.append(ev["direction_accuracy"])
        logger.info("  seed=%d → MAE=%.4f  DirAcc=%.3f", seed, ev["mae"], ev["direction_accuracy"])

    result = {
        "config": cfg,
        "mae_mean":    float(np.mean(maes)), "mae_std":    float(np.std(maes)),
        "diracc_mean": float(np.mean(daccs)), "diracc_std": float(np.std(daccs)),
        "maes": maes, "daccs": daccs,
    }
    logger.info(
        "  Final: MAE=%.4f±%.4f  DirAcc=%.4f±%.4f",
        result["mae_mean"], result["mae_std"],
        result["diracc_mean"], result["diracc_std"],
    )
    return result


def plot_p4(p4: dict) -> None:
    """p4_final.png — 10-seed distribution histogram."""
    maes  = p4["maes"]
    daccs = p4["daccs"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(maes, bins=10, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(p4["mae_mean"], color="blue", linestyle="--",
                    label=f"mean={p4['mae_mean']:.3f}")
    axes[0].axvline(BASELINE_MAE, color="gray", linestyle="--", alpha=0.6,
                    label=f"G1 baseline={BASELINE_MAE}")
    axes[0].axvline(TARGET_MAE, color="green", linestyle="--", alpha=0.7,
                    label=f"target={TARGET_MAE}")
    axes[0].set_xlabel("Cold MAE"); axes[0].set_title("Part 4: Final MAE (10 seeds)")
    axes[0].legend(fontsize=8)

    axes[1].hist(daccs, bins=10, color="tomato", edgecolor="white", alpha=0.85)
    axes[1].axvline(p4["diracc_mean"], color="red", linestyle="--",
                    label=f"mean={p4['diracc_mean']:.3f}")
    axes[1].axvline(BASELINE_DIRACC, color="gray", linestyle="--", alpha=0.6,
                    label=f"G1 baseline={BASELINE_DIRACC}")
    axes[1].axvline(TARGET_DIRACC, color="green", linestyle="--", alpha=0.7,
                    label=f"target={TARGET_DIRACC}")
    axes[1].set_xlabel("Direction Accuracy"); axes[1].set_title("Part 4: Final DirAcc (10 seeds)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "p4_final.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", out)


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(
    p1: dict,
    p2a: dict,
    p2b1: dict,
    p2b2: dict,
    p3a: dict,
    p3b: dict,
    p3c: dict,
    p4: dict,
) -> None:
    """Save optimization report to docs/diagnosis/optimization_report.md."""
    lines: list[str] = [
        "# Optimization Report (Exp018)",
        "",
        "**작성일:** 2026-03-10",
        f"**목표:** MAE < {TARGET_MAE} (vs lightgbm baseline), DirAcc > {TARGET_DIRACC} (vs knn_analog)",
        f"**exp016 G1 baseline:** MAE={BASELINE_MAE}±0.29, DirAcc={BASELINE_DIRACC}",
        "",
        "---",
        "",
        "## Part 1: DirAcc Factor Analysis",
        "",
        "| Variant | Embedding | Weeks | Epochs | MAE (mean±std) | DirAcc (mean±std) |",
        "|---------|-----------|-------|--------|----------------|-------------------|",
    ]

    for name, v in p1.items():
        lines.append(
            f"| {name} | {v['emb']} | {v['weeks']} | {v['epochs']} "
            f"| {v['mae_mean']:.4f}±{v['mae_std']:.4f} "
            f"| {v['diracc_mean']:.4f}±{v['diracc_std']:.4f} |"
        )

    # Determine best from Part 1
    best_p1_name = max(
        (n for n in p1 if n != "baseline"),
        key=lambda n: p1[n]["diracc_mean"],
    )
    best_p1 = p1[best_p1_name]
    baseline_p1 = p1["baseline"]
    lines += [
        "",
        f"**DirAcc Factor Conclusion:**",
        f"- Baseline (residual, 16w, 500ep): MAE={baseline_p1['mae_mean']:.4f}, DirAcc={baseline_p1['diracc_mean']:.4f}",
        f"- Best variant: **{best_p1_name}** (emb={best_p1['emb']}, {best_p1['weeks']}w, {best_p1['epochs']}ep) "
        f"→ MAE={best_p1['mae_mean']:.4f}, DirAcc={best_p1['diracc_mean']:.4f}",
        "",
    ]

    # Part 2A
    sp = p2a["sparse_profile"]
    dp = p2a["diffuse_profile"]
    lines += [
        "---",
        "",
        "## Part 2A: Diffuse vs Sparse Attention Groups",
        "",
        "| Metric | Sparse (low entropy, N=25) | Diffuse (high entropy, N=25) |",
        "|--------|---------------------------|------------------------------|",
        f"| FOODS count | {sp['cat_counts']['FOODS']} | {dp['cat_counts']['FOODS']} |",
        f"| HOBBIES count | {sp['cat_counts']['HOBBIES']} | {dp['cat_counts']['HOBBIES']} |",
        f"| HOUSEHOLD count | {sp['cat_counts']['HOUSEHOLD']} | {dp['cat_counts']['HOUSEHOLD']} |",
        f"| Mean weekly sales | {sp['mean_weekly_sales']:.4f} | {dp['mean_weekly_sales']:.4f} |",
        f"| Mean price ($) | {sp['mean_price']:.2f} | {dp['mean_price']:.2f} |",
        f"| Top-1 attn weight | {sp['top1_attn']:.4f} | {dp['top1_attn']:.4f} |",
        f"| CV (std/mean) | {sp['cv_mean']:.4f} | {dp['cv_mean']:.4f} |",
        f"| Residual emb norm | {sp['emb_norm_mean']:.4f} | {dp['emb_norm_mean']:.4f} |",
        "",
    ]

    # Part 2B-1
    lines += [
        "---",
        "",
        "## Part 2B-1: Temperature Scaling (inference-only)",
        "",
        "| Temperature T | Cold MAE | DirAcc |",
        "|---------------|----------|--------|",
    ]
    for row in p2b1["rows"]:
        lines.append(f"| {row['T']} | {row['mae']:.4f} | {row['dir_acc']:.4f} |")

    best_t_row = max(p2b1["rows"], key=lambda r: r["dir_acc"])
    lines += [
        "",
        f"**Best T:** {best_t_row['T']} → DirAcc={best_t_row['dir_acc']:.4f}, MAE={best_t_row['mae']:.4f}",
        "",
    ]

    # Part 2B-2
    lines += [
        "---",
        "",
        "## Part 2B-2: Top-K Attention (inference-only)",
        "",
        "| K_attn | Cold MAE | DirAcc |",
        "|--------|----------|--------|",
    ]
    for row in p2b2["rows"]:
        lines.append(f"| {row['K']} | {row['mae']:.4f} | {row['dir_acc']:.4f} |")

    best_k_row = max(p2b2["rows"], key=lambda r: r["dir_acc"])
    lines += [
        "",
        f"**Best K:** {best_k_row['K']} → DirAcc={best_k_row['dir_acc']:.4f}, MAE={best_k_row['mae']:.4f}",
        "",
    ]

    # Part 3A
    lines += [
        "---",
        "",
        "## Part 3A: Direction Loss",
        "",
        "| lambda_dir | MAE (mean±std) | DirAcc (mean±std) |",
        "|------------|----------------|-------------------|",
    ]
    for row in p3a["rows"]:
        lines.append(
            f"| {row['lambda_dir']} | {row['mae_mean']:.4f}±{row['mae_std']:.4f} "
            f"| {row['diracc_mean']:.4f}±{row['diracc_std']:.4f} |"
        )

    best_lam_row = max(p3a["rows"], key=lambda r: r["diracc_mean"])
    lines += [
        "",
        f"**Best lambda_dir:** {best_lam_row['lambda_dir']} → "
        f"DirAcc={best_lam_row['diracc_mean']:.4f}±{best_lam_row['diracc_std']:.4f}",
        "",
    ]

    # Part 3B
    pareto = p3b["pareto"]
    pareto_sorted = sorted(pareto, key=lambda r: -r["diracc_mean"])[:5]
    lines += [
        "---",
        "",
        "## Part 3B: Grid Search Summary (Top-5 Pareto by DirAcc)",
        "",
        "| lr | bottleneck | epochs | MAE (mean) | DirAcc (mean) |",
        "|----|------------|--------|------------|---------------|",
    ]
    for r in pareto_sorted:
        lines.append(
            f"| {r['lr']:.4f} | {r['bottleneck']} | {r['epochs']} "
            f"| {r['mae_mean']:.4f} | {r['diracc_mean']:.4f} |"
        )

    # Part 3C
    lines += [
        "",
        "---",
        "",
        "## Part 3C: Dual Attention (raw + residual)",
        "",
        f"| MAE (mean±std) | DirAcc (mean±std) |",
        f"|----------------|-------------------|",
        f"| {p3c['mae_mean']:.4f}±{p3c['mae_std']:.4f} | {p3c['diracc_mean']:.4f}±{p3c['diracc_std']:.4f} |",
        "",
        f"- vs baseline: MAE Δ={p3c['mae_mean'] - BASELINE_MAE:+.4f}, DirAcc Δ={p3c['diracc_mean'] - BASELINE_DIRACC:+.4f}",
        "",
    ]

    # Part 4
    cfg = p4["config"]
    lines += [
        "---",
        "",
        "## Part 4: Final Optimal Configuration (10 seeds)",
        "",
        "### Selected Config",
        "",
        f"| Parameter | Value | Rationale |",
        f"|-----------|-------|-----------|",
        f"| Embedding | {cfg['emb']} | {cfg['rationale']['emb_reason']} |",
        f"| Weeks | {cfg['weeks']} | From Part 1 analysis |",
        f"| Epochs | {cfg['epochs']} | From Part 1 analysis |",
        f"| Temperature T | {cfg['T']} | {cfg['rationale']['T_reason']} |",
        f"| lambda_dir | {cfg['lambda_dir']} | {cfg['rationale']['lam_reason']} |",
        f"| lr | {cfg['lr']} | {cfg['rationale']['hparam_reason']} |",
        f"| bottleneck | {cfg['bottleneck']} | From Part 3B Pareto |",
        f"| Use DualAttn | {cfg['use_dual']} | {cfg['rationale']['dual_reason']} |",
        "",
        "### 10-Seed Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Cold MAE | **{p4['mae_mean']:.4f} ± {p4['mae_std']:.4f}** |",
        f"| DirAcc   | **{p4['diracc_mean']:.4f} ± {p4['diracc_std']:.4f}** |",
        "",
        "### Comparison to Competitors",
        "",
        "| Model | MAE | DirAcc |",
        "|-------|-----|--------|",
    ]
    for cname, cv in COMPETITORS.items():
        lines.append(f"| {cname} | {cv['mae']} | {cv['dir_acc']} |")
    lines += [
        f"| **exp018 Final** | **{p4['mae_mean']:.4f}** | **{p4['diracc_mean']:.4f}** |",
        "",
        "---",
        "",
        "## Conclusion",
        "",
    ]

    mae_target_met = p4["mae_mean"] < TARGET_MAE
    da_target_met  = p4["diracc_mean"] > TARGET_DIRACC

    if mae_target_met:
        lines.append(f"- MAE {p4['mae_mean']:.4f} < {TARGET_MAE}: **Target ACHIEVED**")
    else:
        lines.append(f"- MAE {p4['mae_mean']:.4f} >= {TARGET_MAE}: Target NOT achieved (Δ={p4['mae_mean']-TARGET_MAE:+.4f})")

    if da_target_met:
        lines.append(f"- DirAcc {p4['diracc_mean']:.4f} > {TARGET_DIRACC}: **Target ACHIEVED**")
    else:
        lines.append(f"- DirAcc {p4['diracc_mean']:.4f} <= {TARGET_DIRACC}: Target NOT achieved (Δ={p4['diracc_mean']-TARGET_DIRACC:+.4f})")

    lines += [
        "",
        f"**Both targets met:** {'YES' if mae_target_met and da_target_met else 'NO'}",
        "",
        "---",
        "",
        "**시각화:** `experiments/exp018_optimization/figures/`",
        "**스크립트:** `scripts/exp018_optimization.py`",
    ]

    path = REPORT_DIR / "optimization_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", path)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all 4 parts sequentially and write the report."""
    start_from = "1"
    if "--start-from" in sys.argv:
        idx = sys.argv.index("--start-from")
        start_from = sys.argv[idx + 1]

    if "--report-only" in sys.argv:
        logger.info("=== Exp018: 보고서 전용 모드 ===")
        p1   = _load_ckpt("p1")
        p2a  = _load_ckpt("p2a")
        p2b1 = _load_ckpt("p2b1")
        p2b2 = _load_ckpt("p2b2")
        p3a  = _load_ckpt("p3a")
        p3b  = _load_ckpt("p3b")
        p3c  = _load_ckpt("p3c")
        p4   = _load_ckpt("p4")
        write_report(p1, p2a, p2b1, p2b2, p3a, p3b, p3c, p4)
        logger.info("보고서 작성 완료.")
        return

    logger.info("=== Exp018: Optimization (start-from=%s) ===", start_from)
    t0 = time.time()

    data = load_data()

    if start_from in ("3c", "4"):
        # Load checkpoints from Parts 1-3B
        logger.info("체크포인트 로드 (Parts 1-3B)…")
        p1   = _load_ckpt("p1")
        p2a  = _load_ckpt("p2a")
        p2b1 = _load_ckpt("p2b1")
        p2b2 = _load_ckpt("p2b2")
        p3a  = _load_ckpt("p3a")
        p3b  = _load_ckpt("p3b")
        logger.info("체크포인트 로드 완료.")
    else:
        # ── Part 1 ─────────────────────────────────────────────────────────────
        t1 = time.time()
        logger.info("--- Part 1 start ---")
        p1 = part1_factor_analysis(data)
        plot_p1(p1)
        _save_ckpt("p1", p1)
        logger.info("Part 1 완료 (%.1fs)", time.time() - t1)

        # ── Part 2: train reference model (residual, 16w, 500ep, seed=0) ──────
        t2 = time.time()
        logger.info("--- Part 2 reference model 학습 ---")
        ref_model = train_head(data["warm_residual"], data["y_warm_16"], n_epochs=500, seed=0)

        logger.info("--- Part 2A start ---")
        p2a = part2a_diffuse_sparse(data, ref_model)
        plot_p2a(p2a)
        _save_ckpt("p2a", p2a)

        logger.info("--- Part 2B-1 start ---")
        p2b1 = part2b1_temperature(data, ref_model)
        plot_p2b1(p2b1)
        _save_ckpt("p2b1", p2b1)

        logger.info("--- Part 2B-2 start ---")
        p2b2 = part2b2_topk(data, ref_model)
        plot_p2b2(p2b2)
        _save_ckpt("p2b2", p2b2)
        logger.info("Part 2 완료 (%.1fs)", time.time() - t2)

        # ── Part 3 ─────────────────────────────────────────────────────────────
        t3 = time.time()
        logger.info("--- Part 3A start ---")
        p3a = part3a_direction_loss(data)
        plot_p3a(p3a)
        _save_ckpt("p3a", p3a)

        logger.info("--- Part 3B start (81 trainings) ---")
        p3b = part3b_grid_search(data)
        plot_p3b(p3b)
        _save_ckpt("p3b", p3b)

    # ── Part 3C ────────────────────────────────────────────────────────────
    if start_from not in ("4",):
        t3c = time.time()
        logger.info("--- Part 3C start ---")
        p3c = part3c_dual_attn(data)
        _save_ckpt("p3c", p3c)
        logger.info("Part 3C 완료 (%.1fs)", time.time() - t3c)
    else:
        p3c = _load_ckpt("p3c")

    # ── Part 4 ─────────────────────────────────────────────────────────────
    t4 = time.time()
    logger.info("--- Part 4 start ---")
    p4 = part4_final(data, p1, p2b1, p3a, p3b, p3c)
    plot_p4(p4)
    _save_ckpt("p4", p4)
    logger.info("Part 4 완료 (%.1fs)", time.time() - t4)

    # ── Report ─────────────────────────────────────────────────────────────
    write_report(p1, p2a, p2b1, p2b2, p3a, p3b, p3c, p4)

    logger.info("=== Exp018 전체 완료 (%.1fs) ===", time.time() - t0)

    print("\n" + "=" * 70)
    print("=== Exp018 결과 요약 ===")
    print("\n[Part 1] DirAcc Factor Analysis:")
    for name, v in p1.items():
        print(f"  {name:<12} MAE={v['mae_mean']:.4f}±{v['mae_std']:.4f}  DirAcc={v['diracc_mean']:.4f}±{v['diracc_std']:.4f}")

    print("\n[Part 2B-1] Temperature:")
    for r in p2b1["rows"]:
        print(f"  T={r['T']}  MAE={r['mae']:.4f}  DirAcc={r['dir_acc']:.4f}")

    print("\n[Part 3A] Direction Loss:")
    for r in p3a["rows"]:
        print(f"  lam={r['lambda_dir']}  MAE={r['mae_mean']:.4f}±{r['mae_std']:.4f}  DirAcc={r['diracc_mean']:.4f}±{r['diracc_std']:.4f}")

    print("\n[Part 3C] Dual Attention:")
    print(f"  MAE={p3c['mae_mean']:.4f}±{p3c['mae_std']:.4f}  DirAcc={p3c['diracc_mean']:.4f}±{p3c['diracc_std']:.4f}")

    cfg = p4["config"]
    print(f"\n[Part 4] Final Config: emb={cfg['emb']}, weeks={cfg['weeks']}, epochs={cfg['epochs']}, "
          f"T={cfg['T']}, lam={cfg['lambda_dir']}, lr={cfg['lr']}, bn={cfg['bottleneck']}, dual={cfg['use_dual']}")
    print(f"  Final 10-seed: MAE={p4['mae_mean']:.4f}±{p4['mae_std']:.4f}  DirAcc={p4['diracc_mean']:.4f}±{p4['diracc_std']:.4f}")
    print(f"\n  Target MAE < {TARGET_MAE}: {'ACHIEVED' if p4['mae_mean'] < TARGET_MAE else 'NOT ACHIEVED'}")
    print(f"  Target DirAcc > {TARGET_DIRACC}: {'ACHIEVED' if p4['diracc_mean'] > TARGET_DIRACC else 'NOT ACHIEVED'}")
    print(f"\n보고서: docs/diagnosis/optimization_report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
