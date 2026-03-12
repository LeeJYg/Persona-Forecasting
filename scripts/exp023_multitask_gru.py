"""exp023: Multi-Task GRU Cold-Start Forecaster.

핵심 변경:
  AttnBottleneck → ColdStartForecaster (GRU + Direction Head + Magnitude Head)
  week embedding으로 시간 정보 주입
  log(1+sales) target
  price + log(1+cat_avg) scale anchors

3-Phase Training: direction only → magnitude only → joint fine-tuning
6 Ablation: full / no_direction / no_gru / no_anchor / anchor_only / baseline
3 Seeds: 42, 123, 777

출력:
    docs/diagnosis/multitask_gru_report.md
    experiments/exp023_multitask_gru/figures/
"""
from __future__ import annotations

import logging
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
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

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
CS_DIR   = ROOT / "data/processed/cold_start"
M5_DIR   = ROOT / "m5-forecasting-accuracy"
SC_PATH  = ROOT / "experiments/exp020_error_analysis/item_scorecard.csv"
OUT_DIR  = ROOT / "experiments/exp023_multitask_gru"
FIG_DIR  = OUT_DIR / "figures"
REPORT   = ROOT / "docs/diagnosis/multitask_gru_report.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "docs/diagnosis").mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 777]

# ─── 모델 정의 ────────────────────────────────────────────────────────────────

class ColdStartForecaster(nn.Module):
    """Multi-Task GRU: Direction Head + Magnitude Head."""

    def __init__(self, persona_dim: int = 5120, gru_hidden_dim: int = 64,
                 week_emb_dim: int = 16, dropout: float = 0.3) -> None:
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim

        # 1. Persona Attention
        self.attn_weights      = nn.Linear(persona_dim, 1)
        self.persona_projector = nn.Linear(persona_dim, gru_hidden_dim)

        # 2. Temporal Components
        self.week_embeds = nn.Embedding(16, week_emb_dim)
        self.gru         = nn.GRU(week_emb_dim, gru_hidden_dim, batch_first=True)

        # 3. Direction Head (3-class: UP=0, DOWN=1, FLAT=2)
        self.direction_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 3),
        )

        # 4. Magnitude Head (gru_out + dir_logits + scale_anchors → scalar)
        self.magnitude_head = nn.Sequential(
            nn.Linear(gru_hidden_dim + 3 + 2, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, persona_embeddings: torch.Tensor,
                scale_anchors: torch.Tensor,
                noise_scale: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        persona_embeddings: (N, P, H)
        scale_anchors:      (N, 2)  [price, log1p_cat_avg]
        Returns: dir_logits (N,16,3), mag_preds (N,16)
        """
        batch_size = persona_embeddings.size(0)

        if self.training and noise_scale > 0:
            persona_embeddings = persona_embeddings + \
                torch.randn_like(persona_embeddings) * noise_scale

        attn_scores = F.softmax(self.attn_weights(persona_embeddings), dim=1)
        context = torch.sum(attn_scores * persona_embeddings, dim=1)  # (N, H)

        h0 = self.persona_projector(context).unsqueeze(0)  # (1, N, gru_hidden)

        week_idx    = torch.arange(16, device=persona_embeddings.device).repeat(batch_size, 1)
        week_inputs = self.week_embeds(week_idx)             # (N, 16, week_emb_dim)

        gru_out, _ = self.gru(week_inputs, h0)              # (N, 16, gru_hidden)

        dir_logits = self.direction_head(gru_out)            # (N, 16, 3)

        anchors_exp = scale_anchors.unsqueeze(1).repeat(1, 16, 1)  # (N, 16, 2)
        mag_input   = torch.cat([gru_out, dir_logits, anchors_exp], dim=-1)
        mag_preds   = self.magnitude_head(mag_input).squeeze(-1)   # (N, 16)

        return dir_logits, mag_preds


class AttnBottleneckBaseline(nn.Module):
    """기존 G1: AttnBottleneck, raw target, 16주 출력."""

    def __init__(self, hidden: int = 5120, bottleneck: int = 64,
                 n_weeks: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, bottleneck), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w   = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        ctx = (x * w.unsqueeze(-1)).sum(1)
        return self.head(ctx)


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"]      = pd.to_datetime(df["date"])
    df["iso_year"]  = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"]  = df["date"].dt.isocalendar().week.astype(int)
    df["week_start"]= df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    gcols = [c for c in ["item_id","store_id","cat_id","dept_id","iso_year","iso_week"]
             if c in df.columns]
    return (df.groupby(gcols)
            .agg(sales=("sales","sum"), date=("week_start","first"))
            .reset_index())


def load_data() -> dict[str, Any]:
    logger.info("데이터 로드 중...")
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", map_location="cpu",
                          weights_only=False).numpy()   # (300, 50, 5120)
    cold_raw = torch.load(EMB_DIR / "cold_raw.pt", map_location="cpu",
                          weights_only=False).numpy()   # (100, 50, 5120)
    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)

    item_meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_meta = item_meta[item_meta["is_cold"] == True].reset_index(drop=True)
    warm_meta = item_meta[item_meta["is_cold"] == False].reset_index(drop=True)
    cold_ids  = cold_meta["item_id"].tolist()
    warm_ids  = warm_meta["item_id"].tolist()

    cold_test_raw  = pd.read_csv(CS_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_test_raw  = pd.read_csv(CS_DIR / "warm_test.csv",  parse_dates=["date"])
    warm_train_raw = pd.read_csv(CS_DIR / "warm_train.csv", parse_dates=["date"])

    cold_weekly     = _to_weekly(cold_test_raw)
    warm_test_w     = _to_weekly(warm_test_raw)
    warm_train_weekly = _to_weekly(warm_train_raw)

    # 완전한 16주
    day_cnt = (cold_test_raw
               .assign(iso_year=cold_test_raw["date"].dt.isocalendar().year.astype(int),
                       iso_week=cold_test_raw["date"].dt.isocalendar().week.astype(int))
               .groupby(["iso_year","iso_week"])["date"].nunique())
    complete = set(zip(day_cnt[day_cnt==7].index.get_level_values(0),
                       day_cnt[day_cnt==7].index.get_level_values(1)))
    cold_weekly_16 = cold_weekly[cold_weekly.apply(
        lambda r: (r["iso_year"],r["iso_week"]) in complete, axis=1)].copy()
    warm_test_w16  = warm_test_w[warm_test_w.apply(
        lambda r: (r["iso_year"],r["iso_week"]) in complete, axis=1)].copy()
    wl16 = sorted(complete)

    # week_dates: (iso_year, iso_week) → date
    wk2date = {}
    for _, r in cold_weekly_16.iterrows():
        wk = (r["iso_year"], r["iso_week"])
        if wk not in wk2date:
            wk2date[wk] = r["date"]

    def build_y(ids: list[str], weekly: pd.DataFrame, wl: list) -> np.ndarray:
        wi = {wk: i for i, wk in enumerate(wl)}
        ii = {iid: i for i, iid in enumerate(ids)}
        y  = np.zeros((len(ids), len(wl)), dtype=np.float32)
        for _, r in weekly.iterrows():
            wk = (r["iso_year"], r["iso_week"])
            if wk in wi and r["item_id"] in ii:
                y[ii[r["item_id"]], wi[wk]] = r["sales"]
        return y

    y_warm = build_y(warm_ids, warm_test_w16, wl16)   # (300, 16)
    y_cold = build_y(cold_ids, cold_weekly_16, wl16)   # (100, 16)

    logger.info("  warm_raw=%s cold_raw=%s y_warm=%s y_cold=%s",
                warm_raw.shape, cold_raw.shape, y_warm.shape, y_cold.shape)

    # ── sell_prices: 테스트 기간 평균 가격 ──────────────────────────────────
    # 테스트 기간 wm_yr_wk 범위: calendar.csv에서 2016-01-01~2016-04-24에 해당하는 주차
    calendar = pd.read_csv(M5_DIR / "calendar.csv", parse_dates=["date"])
    test_dates = pd.date_range("2016-01-01", "2016-04-24")
    test_wks   = set(calendar[calendar["date"].isin(test_dates)]["wm_yr_wk"].unique())

    prices = pd.read_csv(M5_DIR / "sell_prices.csv")
    prices_test = prices[(prices["store_id"] == "CA_1") &
                          (prices["wm_yr_wk"].isin(test_wks))]
    avg_price_map = (prices_test.groupby("item_id")["sell_price"].mean().to_dict())

    warm_prices = np.array([avg_price_map.get(iid, 1.0) for iid in warm_ids], dtype=np.float32)
    cold_prices = np.array([avg_price_map.get(iid, 1.0) for iid in cold_ids], dtype=np.float32)

    # ── cat_avg: warm 아이템만으로 카테고리 평균 계산 ───────────────────────
    cat_ids_warm = warm_meta["cat_id"].tolist()
    cat_avg: dict[str, float] = {}
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        mask = [i for i, c in enumerate(cat_ids_warm) if c == cat]
        if mask:
            cat_avg[cat] = float(y_warm[mask].mean())
        else:
            cat_avg[cat] = 1.0
    logger.info("  cat_avg: %s", cat_avg)

    # scale_anchors: [price, log(1+cat_avg)]
    cat_ids_warm_arr = warm_meta["cat_id"].tolist()
    cat_ids_cold_arr = cold_meta["cat_id"].tolist()

    anchors_warm = np.stack([
        warm_prices,
        np.array([np.log1p(cat_avg.get(c, 1.0)) for c in cat_ids_warm_arr], dtype=np.float32),
    ], axis=1)  # (300, 2)

    anchors_cold = np.stack([
        cold_prices,
        np.array([np.log1p(cat_avg.get(c, 1.0)) for c in cat_ids_cold_arr], dtype=np.float32),
    ], axis=1)  # (100, 2)

    # ── direction labels (warm, method A) ──────────────────────────────────
    y_w_t = torch.from_numpy(y_warm)
    diff  = y_w_t[:, 1:] - y_w_t[:, :-1]      # (300, 15)
    pct   = diff / (y_w_t[:, :-1] + 1e-6)
    gt_dir = torch.zeros_like(diff, dtype=torch.long)  # default FLAT=2
    gt_dir[pct > 0.05]            = 0   # UP
    gt_dir[pct < -0.05]           = 1   # DOWN
    gt_dir[torch.abs(pct) <= 0.05]= 2   # FLAT
    logger.info("  direction label dist: UP=%d DOWN=%d FLAT=%d",
                (gt_dir==0).sum().item(), (gt_dir==1).sum().item(), (gt_dir==2).sum().item())

    return dict(
        warm_raw=warm_raw, cold_raw=cold_raw,
        warm_residual=warm_residual, cold_residual=cold_residual,
        warm_ids=warm_ids, cold_ids=cold_ids,
        cold_meta=cold_meta, warm_meta=warm_meta,
        y_warm=y_warm, y_cold=y_cold,
        anchors_warm=anchors_warm, anchors_cold=anchors_cold,
        gt_dir=gt_dir,
        cold_weekly_16=cold_weekly_16,
        warm_train_weekly=warm_train_weekly,
        week_list=wl16,
        week_dates=wk2date,
    )


# ─── 학습 함수 ────────────────────────────────────────────────────────────────

def train_full(X: np.ndarray, y_warm: np.ndarray, anchors: np.ndarray,
               gt_dir: torch.Tensor, seed: int) -> ColdStartForecaster:
    """3-Phase Training."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ColdStartForecaster(persona_dim=X.shape[2])
    Xt  = torch.tensor(X, dtype=torch.float32)
    at  = torch.tensor(anchors, dtype=torch.float32)
    yt  = torch.log1p(torch.tensor(y_warm, dtype=torch.float32))

    ce = nn.CrossEntropyLoss()

    # Phase 1: Direction only (100 ep)
    opt1 = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for ep in range(100):
        opt1.zero_grad()
        dir_l, _ = model(Xt, at, noise_scale=0.01)
        loss = ce(dir_l[:, 1:].reshape(-1, 3), gt_dir.reshape(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt1.step()

    # Phase 2: Magnitude only, direction_head frozen (200 ep)
    for p in model.direction_head.parameters():
        p.requires_grad = False
    opt2 = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=1e-4)
    model.train()
    for ep in range(200):
        opt2.zero_grad()
        _, mag = model(Xt, at, noise_scale=0.01)
        loss = F.l1_loss(mag, yt)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()

    # Phase 3: Joint fine-tune (100 ep)
    for p in model.parameters():
        p.requires_grad = True
    opt3 = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    model.train()
    for ep in range(100):
        opt3.zero_grad()
        dir_l, mag = model(Xt, at, noise_scale=0.01)
        dir_loss = ce(dir_l[:, 1:].reshape(-1, 3), gt_dir.reshape(-1))
        mag_loss = F.l1_loss(mag, yt)
        loss = 0.4 * dir_loss + 0.6 * mag_loss
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt3.step()

    model.eval()
    return model


def train_no_direction(X: np.ndarray, y_warm: np.ndarray, anchors: np.ndarray,
                       seed: int) -> ColdStartForecaster:
    """no_direction: magnitude head만 학습 (direction head 사용 안 함)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ColdStartForecaster(persona_dim=X.shape[2])
    Xt = torch.tensor(X, dtype=torch.float32)
    at = torch.tensor(anchors, dtype=torch.float32)
    yt = torch.log1p(torch.tensor(y_warm, dtype=torch.float32))

    # direction_head를 freeze하고 magnitude만 학습
    for p in model.direction_head.parameters():
        p.requires_grad = False
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()),
               lr=1e-3, weight_decay=1e-4)
    model.train()
    for ep in range(300):
        opt.zero_grad()
        _, mag = model(Xt, at, noise_scale=0.01)
        loss = F.l1_loss(mag, yt)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


class NoGRUModel(nn.Module):
    """no_gru: AttnBottleneck + scale_anchor + log target."""

    def __init__(self, hidden: int = 5120, bottleneck: int = 64,
                 n_weeks: int = 16, dropout: float = 0.1,
                 anchor_dim: int = 2) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden + anchor_dim, bottleneck), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        w   = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        ctx = (x * w.unsqueeze(-1)).sum(1)
        ctx_a = torch.cat([ctx, anchors], dim=-1)
        return self.head(ctx_a)


def train_no_gru(X: np.ndarray, y_warm: np.ndarray, anchors: np.ndarray,
                 seed: int) -> NoGRUModel:
    torch.manual_seed(seed); np.random.seed(seed)
    model = NoGRUModel(hidden=X.shape[2])
    Xt = torch.tensor(X, dtype=torch.float32)
    at = torch.tensor(anchors, dtype=torch.float32)
    yt = torch.log1p(torch.tensor(y_warm, dtype=torch.float32))

    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for ep in range(500):
        opt.zero_grad()
        loss = F.l1_loss(model(Xt, at), yt)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


def train_no_anchor(X: np.ndarray, y_warm: np.ndarray, anchors: np.ndarray,
                    gt_dir: torch.Tensor, seed: int) -> ColdStartForecaster:
    """no_anchor: scale_anchors를 zeros로 교체."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ColdStartForecaster(persona_dim=X.shape[2])
    Xt  = torch.tensor(X, dtype=torch.float32)
    at0 = torch.zeros(X.shape[0], 2, dtype=torch.float32)   # anchors=0
    yt  = torch.log1p(torch.tensor(y_warm, dtype=torch.float32))
    ce  = nn.CrossEntropyLoss()

    opt1 = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for ep in range(100):
        opt1.zero_grad()
        dir_l, _ = model(Xt, at0, noise_scale=0.01)
        loss = ce(dir_l[:, 1:].reshape(-1, 3), gt_dir.reshape(-1))
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt1.step()

    for p in model.direction_head.parameters(): p.requires_grad = False
    opt2 = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=1e-4)
    for ep in range(200):
        opt2.zero_grad()
        _, mag = model(Xt, at0, noise_scale=0.01)
        loss = F.l1_loss(mag, yt)
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt2.step()

    for p in model.parameters(): p.requires_grad = True
    opt3 = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    for ep in range(100):
        opt3.zero_grad()
        dir_l, mag = model(Xt, at0, noise_scale=0.01)
        loss = 0.4 * ce(dir_l[:, 1:].reshape(-1, 3), gt_dir.reshape(-1)) + \
               0.6 * F.l1_loss(mag, yt)
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt3.step()

    model.eval()
    return model


def train_anchor_only(X_shape: tuple, y_warm: np.ndarray, anchors: np.ndarray,
                      gt_dir: torch.Tensor, seed: int) -> ColdStartForecaster:
    """anchor_only: persona_embeddings를 zeros로 대체."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ColdStartForecaster(persona_dim=X_shape[2])
    Xz  = torch.zeros(X_shape[0], X_shape[1], X_shape[2], dtype=torch.float32)
    at  = torch.tensor(anchors, dtype=torch.float32)
    yt  = torch.log1p(torch.tensor(y_warm, dtype=torch.float32))
    ce  = nn.CrossEntropyLoss()

    opt1 = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for ep in range(100):
        opt1.zero_grad()
        dir_l, _ = model(Xz, at, noise_scale=0.0)
        loss = ce(dir_l[:, 1:].reshape(-1, 3), gt_dir.reshape(-1))
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt1.step()

    for p in model.direction_head.parameters(): p.requires_grad = False
    opt2 = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=1e-4)
    for ep in range(200):
        opt2.zero_grad()
        _, mag = model(Xz, at, noise_scale=0.0)
        loss = F.l1_loss(mag, yt)
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt2.step()

    for p in model.parameters(): p.requires_grad = True
    opt3 = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    for ep in range(100):
        opt3.zero_grad()
        dir_l, mag = model(Xz, at, noise_scale=0.0)
        loss = 0.4 * ce(dir_l[:, 1:].reshape(-1, 3), gt_dir.reshape(-1)) + \
               0.6 * F.l1_loss(mag, yt)
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt3.step()

    model.eval()
    return model


def train_baseline(X: np.ndarray, y_warm: np.ndarray, seed: int) -> AttnBottleneckBaseline:
    """기존 G1: raw target, 500 epoch, MAE loss."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = AttnBottleneckBaseline(hidden=X.shape[2])
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y_warm, dtype=torch.float32)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for ep in range(500):
        opt.zero_grad()
        loss = F.l1_loss(model(Xt), yt)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


# ─── 추론 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_gru(model: ColdStartForecaster,
                X: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Returns (N, 16) clipped sales predictions."""
    Xt = torch.tensor(X, dtype=torch.float32)
    at = torch.tensor(anchors, dtype=torch.float32)
    _, log_p = model(Xt, at)
    return torch.expm1(log_p).clamp(min=0).numpy()


@torch.no_grad()
def predict_no_gru(model: NoGRUModel,
                   X: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32)
    at = torch.tensor(anchors, dtype=torch.float32)
    log_p = model(Xt, at)
    return torch.expm1(log_p).clamp(min=0).numpy()


@torch.no_grad()
def predict_baseline(model: AttnBottleneckBaseline, X: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32)
    return model(Xt).clamp(min=0).numpy()


@torch.no_grad()
def dir_accuracy_warm(model: ColdStartForecaster, X: np.ndarray,
                      anchors: np.ndarray, gt_dir: torch.Tensor) -> float:
    """Direction head의 warm validation 3-class accuracy."""
    Xt = torch.tensor(X, dtype=torch.float32)
    at = torch.tensor(anchors, dtype=torch.float32)
    dir_l, _ = model(Xt, at)
    pred_dir = dir_l[:, 1:].argmax(dim=-1)  # (N, 15)
    correct  = (pred_dir == gt_dir).float().mean().item()
    return correct


# ─── 평가용 DataFrame 변환 ────────────────────────────────────────────────────

def preds_to_df(preds: np.ndarray, item_ids: list[str], cat_ids: list[str],
                week_list: list, week_dates: dict) -> pd.DataFrame:
    records = []
    for i, iid in enumerate(item_ids):
        for t, wk in enumerate(week_list):
            d = week_dates.get(wk)
            if d is None:
                continue
            records.append(dict(
                item_id=iid, store_id="CA_1", cat_id=cat_ids[i],
                date=d, pred_sales=max(0.0, float(preds[i, t])),
            ))
    return pd.DataFrame(records)


def run_evaluation(preds: np.ndarray, data: dict, model_name: str) -> dict:
    cat_ids = data["cold_meta"]["cat_id"].tolist()
    pred_df = preds_to_df(preds, data["cold_ids"], cat_ids,
                          data["week_list"], data["week_dates"])
    ev = evaluate_weekly(data["cold_weekly_16"], pred_df,
                         data["warm_train_weekly"], model_name=model_name)
    return ev


# ─── 서브그룹 분석 ────────────────────────────────────────────────────────────

def subgroup_eval(preds: np.ndarray, data: dict,
                  model_name: str) -> dict[str, dict]:
    """선택품(nonzero<0.5) / 필수품(nonzero≥0.9) 서브그룹 평가."""
    sc = pd.read_csv(SC_PATH)
    ids = data["cold_ids"]
    y   = data["y_cold"]

    results: dict[str, dict] = {}
    for label, cond in [("elective", lambda r: r["actual_nonzero_week_ratio"] < 0.5),
                         ("essential", lambda r: r["actual_nonzero_week_ratio"] >= 0.9)]:
        sub_ids = sc[sc.apply(cond, axis=1)]["item_id"].tolist()
        sub_idx = [i for i, iid in enumerate(ids) if iid in set(sub_ids)]
        if not sub_idx:
            continue
        sub_ids_ord = [ids[i] for i in sub_idx]
        sub_cats    = [data["cold_meta"].loc[data["cold_meta"]["item_id"] == iid,
                                             "cat_id"].values[0] for iid in sub_ids_ord]
        sub_pred    = preds[sub_idx]
        pred_df     = preds_to_df(sub_pred, sub_ids_ord, sub_cats,
                                  data["week_list"], data["week_dates"])
        cold_sub    = data["cold_weekly_16"][
            data["cold_weekly_16"]["item_id"].isin(set(sub_ids_ord))]
        ev = evaluate_weekly(cold_sub, pred_df, data["warm_train_weekly"],
                             model_name=f"{model_name}_{label}")
        results[label] = dict(
            n=len(sub_idx),
            wrmsse=ev["wrmsse"],
            dir_acc=ev["direction_accuracy"],
            mae=ev["mae"],
            actual_mean=float(y[sub_idx].mean()),
            pred_mean=float(sub_pred.mean()),
        )
    return results


# ─── 시각화 ───────────────────────────────────────────────────────────────────

def plot_bar_chart(all_results: dict[str, dict], seeds: list[int]) -> None:
    """6개 변형 WRMSSE / DirAcc bar chart."""
    variants = ["full", "no_direction", "no_gru", "no_anchor", "anchor_only", "baseline"]
    wr_means, wr_stds, da_means, da_stds = [], [], [], []

    for var in variants:
        wrs = [all_results[var][s]["wrmsse"] for s in seeds if s in all_results[var]]
        das = [all_results[var][s]["direction_accuracy"] for s in seeds if s in all_results[var]]
        wr_means.append(np.mean(wrs)); wr_stds.append(np.std(wrs))
        da_means.append(np.mean(das)); da_stds.append(np.std(das))

    x   = np.arange(len(variants))
    w   = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(x, wr_means, w, yerr=wr_stds, capsize=4, color="steelblue", label="WRMSSE (lower=better)")
    ax1.set_xticks(x); ax1.set_xticklabels(variants, rotation=30, ha="right")
    ax1.set_ylabel("WRMSSE"); ax1.set_title("WRMSSE by Variant (mean±std, 3 seeds)")
    for i, (m, s) in enumerate(zip(wr_means, wr_stds)):
        ax1.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=8)

    ax2.bar(x, da_means, w, yerr=da_stds, capsize=4, color="coral", label="DirAcc (higher=better)")
    ax2.set_xticks(x); ax2.set_xticklabels(variants, rotation=30, ha="right")
    ax2.set_ylabel("DirAcc"); ax2.set_title("DirAcc by Variant (mean±std, 3 seeds)")
    for i, (m, s) in enumerate(zip(da_means, da_stds)):
        ax2.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "bar_wrmsse_diracc.png", dpi=150)
    plt.close(fig)
    logger.info("  bar_wrmsse_diracc.png 저장")


def plot_line_10items(full_preds: np.ndarray, baseline_preds: np.ndarray,
                      data: dict) -> None:
    """full 모델의 10개 아이템 꺾은선 (exp022와 동일 선정 기준)."""
    sc  = pd.read_csv(SC_PATH)
    y   = data["y_cold"]
    ids = data["cold_ids"]
    ii  = {iid: i for i, iid in enumerate(ids)}

    elective  = sc[sc["actual_nonzero_week_ratio"] < 0.5]
    essential = sc[sc["actual_nonzero_week_ratio"] >= 0.9]
    items_10  = (
        [(iid, "선택품_좋음") for iid in elective.nsmallest(3, "trackb_WRMSSE")["item_id"]] +
        [(iid, "선택품_나쁨") for iid in elective.nlargest(2, "trackb_WRMSSE")["item_id"]] +
        [(iid, "필수품_TB승") for iid in
         (essential[essential["trackb_wins_lgbm_WRMSSE"]==True]
          .nsmallest(3, "trackb_WRMSSE")["item_id"] if len(essential[essential["trackb_wins_lgbm_WRMSSE"]==True]) >= 3
          else essential.nsmallest(3, "trackb_WRMSSE")["item_id"])] +
        [(iid, "필수품_TB패") for iid in essential.nlargest(2, "trackb_WRMSSE")["item_id"]]
    )

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    for ax_i, (iid, grp) in enumerate(items_10):
        idx = ii.get(iid, -1)
        if idx < 0:
            axes.flatten()[ax_i].set_visible(False)
            continue
        ax = axes.flatten()[ax_i]
        wks = range(1, 17)
        ax.plot(wks, y[idx],             "k-",  lw=2, label="actual")
        ax.plot(wks, full_preds[idx],    "r--", lw=1.5, label="full")
        ax.plot(wks, baseline_preds[idx],"b--", lw=1.5, label="baseline")
        r = sc[sc["item_id"] == iid].iloc[0]
        ax.set_title(f"{iid}\n{r['dept_id']} | nzr={r['actual_nonzero_week_ratio']:.2f}\n[{grp}]",
                     fontsize=7)
        if ax_i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("분석 3-2: full vs baseline (10 items)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "line_10items.png", dpi=150)
    plt.close(fig)
    logger.info("  line_10items.png 저장")


def plot_pred_histogram(full_preds: np.ndarray, baseline_preds: np.ndarray,
                        y_cold: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bins = 40
    for ax, arr, label, color in [
        (axes[0], y_cold.flatten(),         "actual",   "black"),
        (axes[1], full_preds.flatten(),     "full",     "red"),
        (axes[2], baseline_preds.flatten(), "baseline", "blue"),
    ]:
        ax.hist(arr, bins=bins, color=color, alpha=0.7, edgecolor="k")
        ax.set_title(f"{label} (mean={arr.mean():.2f})")
        ax.set_xlabel("Weekly Sales")
    fig.suptitle("분석 3-3: 예측 분포 비교")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pred_histogram.png", dpi=150)
    plt.close(fig)
    logger.info("  pred_histogram.png 저장")


def plot_cv_histogram(full_preds: np.ndarray, baseline_preds: np.ndarray,
                      y_cold: np.ndarray) -> None:
    def cv_ratios(preds, y):
        ratios = []
        for i in range(len(y)):
            am = y[i].mean(); pm = preds[i].mean()
            if am < 1e-9 or pm < 1e-9:
                continue
            ac = y[i].std() / am
            pc = preds[i].std() / pm
            if ac > 1e-9:
                ratios.append(pc / ac)
        return np.array(ratios)

    full_cv = cv_ratios(full_preds, y_cold)
    base_cv = cv_ratios(baseline_preds, y_cold)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cv, label, color in [
        (axes[0], full_cv,   "full",     "red"),
        (axes[1], base_cv,   "baseline", "blue"),
    ]:
        ax.hist(cv, bins=30, color=color, alpha=0.7, edgecolor="k")
        ax.axvline(1.0, color="k", linestyle="--", label="ratio=1")
        ax.set_title(f"{label}: G1_CV/actual_CV\nmean={cv.mean():.3f} <0.1:{(cv<0.1).sum()}")
        ax.set_xlabel("CV ratio"); ax.legend(fontsize=8)
    fig.suptitle("분석 3-4: CV 비율 histogram")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv_histogram.png", dpi=150)
    plt.close(fig)
    logger.info("  cv_histogram.png 저장")


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(all_results: dict, subgroup_all: dict, dir_acc_warm: dict,
                 data: dict) -> None:
    variants = ["full", "no_direction", "no_gru", "no_anchor", "anchor_only", "baseline"]
    seeds    = SEEDS

    def mean_std(vals: list[float]) -> str:
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

    lines: list[str] = [
        "# Multi-Task GRU 실험 보고서 (Exp023)",
        "",
        f"**작성일:** 2026-03-11",
        "**목적:** AttnBottleneck → Multi-Task GRU (Direction + Magnitude Head)",
        "**기존 데이터:** warm_raw.pt(300×50×5120), cold_raw.pt(100×50×5120) 재사용",
        "",
        "## 실험 설정 (Hyperparameters)",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| persona_dim | 5120 |",
        "| gru_hidden_dim | 64 |",
        "| week_emb_dim | 16 |",
        "| dropout | 0.3 |",
        "| Phase1 (direction) | 100 epoch, lr=1e-3, wd=1e-4 |",
        "| Phase2 (magnitude) | 200 epoch, lr=1e-3, wd=1e-4 |",
        "| Phase3 (joint) | 100 epoch, lr=1e-5, wd=1e-4, λ_dir=0.4 λ_mag=0.6 |",
        "| noise_scale | 0.01 (training) |",
        "| direction threshold | 5% |",
        "| seeds | 42, 123, 777 |",
        "| baseline | AttnBottleneck G1: 500ep, lr=1e-3, wd=1e-4, MAE, raw target |",
        "",
        "---",
        "",
        "## 6-1: 전체 지표 (mean±std over 3 seeds)",
        "",
        "| 변형 | WRMSSE | DirAcc | MAE | pred_mean | actual_mean |",
        "|------|--------|--------|-----|-----------|-------------|",
    ]

    y_actual_mean = data["y_cold"].mean()
    for var in variants:
        if var not in all_results:
            continue
        res_list = [all_results[var][s] for s in seeds if s in all_results[var]]
        wrs  = [r["wrmsse"] for r in res_list]
        das  = [r["direction_accuracy"] for r in res_list]
        maes = [r["mae"] for r in res_list]
        pms  = [r.get("pred_mean", float("nan")) for r in res_list]
        lines.append(
            f"| {var} | {mean_std(wrs)} | {mean_std(das)} | "
            f"{mean_std(maes)} | {np.mean(pms):.2f} | {y_actual_mean:.2f} |"
        )

    lines += [
        "",
        "## 6-2: 선택품 / 필수품 서브그룹 (mean over 3 seeds)",
        "",
        "| 변형 | 그룹 | n | WRMSSE | DirAcc | MAE | actual_mean | pred_mean |",
        "|------|------|---|--------|--------|-----|-------------|-----------|",
    ]

    for var in variants:
        if var not in subgroup_all:
            continue
        for grp_key, glabel in [("elective", "선택품(nonzero<0.5)"),
                                  ("essential", "필수품(nonzero≥0.9)")]:
            sg_list = [subgroup_all[var][s][grp_key]
                       for s in seeds
                       if s in subgroup_all[var] and grp_key in subgroup_all[var][s]]
            if not sg_list:
                continue
            n   = sg_list[0]["n"]
            wrs = [r["wrmsse"] for r in sg_list]
            das = [r["dir_acc"] for r in sg_list]
            ms  = [r["mae"] for r in sg_list]
            am  = sg_list[0]["actual_mean"]
            pms = [r["pred_mean"] for r in sg_list]
            lines.append(
                f"| {var} | {glabel} | {n} | {mean_std(wrs)} | "
                f"{mean_std(das)} | {mean_std(ms)} | {am:.2f} | {np.mean(pms):.2f} |"
            )

    lines += [
        "",
        "## Direction Head warm 3-class accuracy",
        "",
        "| 변형 | seed=42 | seed=123 | seed=777 | mean±std |",
        "|------|---------|----------|----------|----------|",
    ]
    for var in ["full", "no_anchor", "anchor_only"]:
        if var not in dir_acc_warm:
            continue
        vals = [dir_acc_warm[var].get(s, float("nan")) for s in seeds]
        valid = [v for v in vals if not np.isnan(v)]
        lines.append(
            f"| {var} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | "
            f"{mean_std(valid)} |"
        )

    lines += [
        "",
        "## 시각화",
        "",
        "- `experiments/exp023_multitask_gru/figures/bar_wrmsse_diracc.png`",
        "- `experiments/exp023_multitask_gru/figures/line_10items.png`",
        "- `experiments/exp023_multitask_gru/figures/pred_histogram.png`",
        "- `experiments/exp023_multitask_gru/figures/cv_histogram.png`",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", REPORT)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Exp023 Multi-Task GRU 시작 ===")
    data = load_data()

    warm_res  = data["warm_residual"]   # (300, 50, 5120)
    cold_res  = data["cold_residual"]   # (100, 50, 5120)
    y_warm    = data["y_warm"]           # (300, 16)
    y_cold    = data["y_cold"]           # (100, 16)
    aw        = data["anchors_warm"]     # (300, 2)
    ac        = data["anchors_cold"]     # (100, 2)
    gt_dir    = data["gt_dir"]           # (300, 15)

    all_results:   dict[str, dict[int, dict]] = {}
    subgroup_all:  dict[str, dict[int, dict]] = {}
    dir_acc_warm_d: dict[str, dict[int, float]] = {}

    VARIANT_FUNCS = {
        "full":         lambda s: train_full(warm_res, y_warm, aw, gt_dir, s),
        "no_direction": lambda s: train_no_direction(warm_res, y_warm, aw, s),
        "no_gru":       lambda s: train_no_gru(warm_res, y_warm, aw, s),
        "no_anchor":    lambda s: train_no_anchor(warm_res, y_warm, aw, gt_dir, s),
        "anchor_only":  lambda s: train_anchor_only(warm_res.shape, y_warm, aw, gt_dir, s),
        "baseline":     lambda s: train_baseline(warm_res, y_warm, s),
    }

    for var, train_fn in VARIANT_FUNCS.items():
        logger.info("=== 변형: %s ===", var)
        all_results[var]   = {}
        subgroup_all[var]  = {}
        if var in ("full", "no_anchor", "anchor_only"):
            dir_acc_warm_d[var] = {}

        for seed in SEEDS:
            logger.info("  seed=%d 학습 중...", seed)
            model = train_fn(seed)

            # 추론
            if var == "no_gru":
                preds = predict_no_gru(model, cold_res, ac)
            elif var == "baseline":
                preds = predict_baseline(model, cold_res)
            elif var == "anchor_only":
                Xz = np.zeros_like(cold_res)
                ac0 = ac
                preds = predict_gru(model, Xz, ac0)
            elif var == "no_anchor":
                preds = predict_gru(model, cold_res,
                                    np.zeros((cold_res.shape[0], 2), dtype=np.float32))
            else:
                preds = predict_gru(model, cold_res, ac)

            ev = run_evaluation(preds, data, f"{var}_seed{seed}")
            ev["pred_mean"] = float(preds.mean())
            all_results[var][seed] = ev

            sg = subgroup_eval(preds, data, f"{var}_seed{seed}")
            subgroup_all[var][seed] = sg

            # direction head accuracy (warm)
            if var in ("full", "no_anchor") and isinstance(model, ColdStartForecaster):
                if var == "no_anchor":
                    aw_eval = np.zeros((warm_res.shape[0], 2), dtype=np.float32)
                else:
                    aw_eval = aw
                da = dir_accuracy_warm(model, warm_res, aw_eval, gt_dir)
                dir_acc_warm_d[var][seed] = da
                logger.info("    dir_acc_warm=%.4f", da)
            elif var == "anchor_only" and isinstance(model, ColdStartForecaster):
                Xw_z = np.zeros_like(warm_res)
                da = dir_accuracy_warm(model, Xw_z, aw, gt_dir)
                dir_acc_warm_d[var][seed] = da
                logger.info("    dir_acc_warm=%.4f", da)

            logger.info("    WRMSSE=%.4f DirAcc=%.4f MAE=%.4f pred_mean=%.2f",
                        ev["wrmsse"], ev["direction_accuracy"], ev["mae"], ev["pred_mean"])

    # ── 시각화 ──────────────────────────────────────────────────────────────
    logger.info("시각화 생성 중...")

    # full/baseline seed=42 대표 예측값
    full_m    = train_full(warm_res, y_warm, aw, gt_dir, 42)
    full_p    = predict_gru(full_m, cold_res, ac)
    base_m    = train_baseline(warm_res, y_warm, 42)
    base_p    = predict_baseline(base_m, cold_res)

    plot_bar_chart(all_results, SEEDS)
    plot_line_10items(full_p, base_p, data)
    plot_pred_histogram(full_p, base_p, y_cold)
    plot_cv_histogram(full_p, base_p, y_cold)

    # ── 보고서 ──────────────────────────────────────────────────────────────
    write_report(all_results, subgroup_all, dir_acc_warm_d, data)

    # ── 요약 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[Exp023 요약]")
    for var in ["full", "no_direction", "no_gru", "no_anchor", "anchor_only", "baseline"]:
        if var not in all_results: continue
        wrs = [all_results[var][s]["wrmsse"] for s in SEEDS if s in all_results[var]]
        das = [all_results[var][s]["direction_accuracy"] for s in SEEDS if s in all_results[var]]
        print(f"  {var:15s}: WRMSSE={np.mean(wrs):.4f}±{np.std(wrs):.4f}  "
              f"DirAcc={np.mean(das):.4f}±{np.std(das):.4f}")
    print(f"보고서: {REPORT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
