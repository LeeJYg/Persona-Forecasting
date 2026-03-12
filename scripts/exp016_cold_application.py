"""Exp016: Cold Items 적용 + Rescale Ablation.

실험 F: warm 300개 전체로 global head 학습 → cold 100개에 K-NN rescale 적용
실험 G: 5가지 변형으로 rescale 기여 분리 (warm 5-fold CV + cold 평가)

평가 기준: exp006과 동일한 16-week ISO-week 기준 (완전한 주만)
비교 대상 (exp006):
    lightgbm_proxy_lags  MAE=8.48  DirAcc=0.343
    similar_item_avg     MAE=8.64  DirAcc=0.232
    Track A calibrated   MAE=8.90  DirAcc=0.393
    knn_analog           MAE=9.57  DirAcc=0.412

출력:
    docs/diagnosis/cold_application_report.md
    experiments/exp016_cold_application/figures/
"""
from __future__ import annotations

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

EMB_DIR    = ROOT / "experiments/exp011_v3_pipeline/embeddings"
FIG_DIR    = ROOT / "experiments/exp016_cold_application/figures"
REPORT_DIR = ROOT / "docs/diagnosis"
CS_DIR     = ROOT / "data/processed/cold_start"
M5_DIR     = ROOT / "m5-forecasting-accuracy"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── exp006 경쟁자 결과 (비교용 상수) ─────────────────────────────────────────
COMPETITORS = {
    "lightgbm_proxy_lags": {"mae": 8.48, "dir_acc": 0.343},
    "similar_item_avg":    {"mae": 8.64, "dir_acc": 0.232},
    "Track_A_calibrated":  {"mae": 8.90, "dir_acc": 0.393},
    "knn_analog":          {"mae": 9.57, "dir_acc": 0.412},
}

K_ABLATION = 50      # G 실험 고정 K
TRAIN_EPOCHS_FULL = 500   # cold용 전체 warm 학습
TRAIN_EPOCHS_CV   = 200   # warm CV용


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def _to_weekly_iso(df: pd.DataFrame) -> pd.DataFrame:
    """exp006 run_competitors.py의 _to_weekly와 동일 (ISO-week 기준)."""
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


def _complete_weeks(df_daily: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """cold_test 기준 완전한 주만 필터링 (7일 미만 제거). → (weekly_df, complete_week_set)."""
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
    complete_set = set(zip(complete["iso_year"], complete["iso_week"]))
    return complete_set


def load_data():
    """모든 필요 데이터 로드."""
    logger.info("데이터 로드 중...")
    warm_raw  = torch.load(EMB_DIR / "warm_raw.pt",  weights_only=True).numpy()   # (300,50,5120)
    cold_raw  = torch.load(EMB_DIR / "cold_raw.pt",  weights_only=True).numpy()   # (100,50,5120)
    warm_mean = torch.load(EMB_DIR / "warm_mean.pt", weights_only=True).numpy()   # (300,5120)
    cold_mean = torch.load(EMB_DIR / "cold_mean.pt", weights_only=True).numpy()   # (100,5120)

    meta      = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_meta = meta[meta["is_cold"] == True].reset_index(drop=True)
    warm_meta = meta[meta["is_cold"] == False].reset_index(drop=True)

    cold_test_raw  = pd.read_csv(CS_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_test_raw  = pd.read_csv(CS_DIR / "warm_test.csv",  parse_dates=["date"])
    warm_train_raw = pd.read_csv(CS_DIR / "warm_train.csv", parse_dates=["date"])

    # 완전한 주 필터링
    complete_set = _complete_weeks(cold_test_raw)
    logger.info("  완전한 주: %d개 (16주 기준)", len(complete_set))

    # cold_test_weekly (16주, exp006 평가 기준)
    cold_test_weekly = _to_weekly_iso(cold_test_raw)
    cold_test_weekly = cold_test_weekly[
        cold_test_weekly.apply(lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)
    ]
    logger.info("  cold_test_weekly: %d rows (%d items × %d weeks)",
                len(cold_test_weekly),
                cold_test_weekly["item_id"].nunique(),
                cold_test_weekly["iso_week"].nunique())

    # warm_train_weekly (WRMSSE 계산용)
    warm_train_weekly = _to_weekly_iso(warm_train_raw)

    # y_cold: cold items의 실제 주간 판매량 (16주, 배열 형태)
    cold_ids = cold_meta["item_id"].tolist()
    weeks_order = sorted(complete_set)               # [(iso_year, iso_week), ...]
    y_cold_df = (
        cold_test_weekly
        .assign(wk_key=lambda r: list(zip(r["iso_year"], r["iso_week"])))
    )
    # (n_cold, 16) 행렬
    week_list = sorted(complete_set)
    y_cold = np.zeros((len(cold_ids), len(week_list)), dtype=np.float32)
    week_idx = {wk: i for i, wk in enumerate(week_list)}
    for _, row in cold_test_weekly.iterrows():
        ci = cold_ids.index(row["item_id"]) if row["item_id"] in cold_ids else -1
        wi = week_idx.get((row["iso_year"], row["iso_week"]), -1)
        if ci >= 0 and wi >= 0:
            y_cold[ci, wi] = row["sales"]

    # y_warm: warm items의 실제 주간 판매량 (같은 16주)
    warm_ids = warm_meta["item_id"].tolist()
    warm_test_weekly = _to_weekly_iso(warm_test_raw)
    warm_test_weekly = warm_test_weekly[
        warm_test_weekly.apply(lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)
    ]
    y_warm = np.zeros((len(warm_ids), len(week_list)), dtype=np.float32)
    for _, row in warm_test_weekly.iterrows():
        wi_i = warm_ids.index(row["item_id"]) if row["item_id"] in warm_ids else -1
        wk_i = week_idx.get((row["iso_year"], row["iso_week"]), -1)
        if wi_i >= 0 and wk_i >= 0:
            y_warm[wi_i, wk_i] = row["sales"]

    logger.info("  y_cold=%s  mean=%.2f  y_warm=%s  mean=%.2f",
                y_cold.shape, y_cold.mean(), y_warm.shape, y_warm.mean())

    # cold_test_weekly date 열이 필요: week_start dates for pred DataFrame
    week_dates = (
        cold_test_weekly[["iso_year", "iso_week", "date"]]
        .drop_duplicates()
        .assign(wk_key=lambda r: list(zip(r["iso_year"], r["iso_week"])))
        .set_index("wk_key")["date"]
        .to_dict()
    )

    return dict(
        warm_raw=warm_raw, cold_raw=cold_raw,
        warm_mean=warm_mean, cold_mean=cold_mean,
        cold_meta=cold_meta, warm_meta=warm_meta,
        cold_test_weekly=cold_test_weekly,
        warm_train_weekly=warm_train_weekly,
        y_cold=y_cold, y_warm=y_warm,
        week_list=week_list, week_dates=week_dates,
    )


# ─── 모델 ─────────────────────────────────────────────────────────────────────

class AttnBottleneck(torch.nn.Module):
    """Attention+Bottleneck head (exp011/013/014와 동일 구조)."""
    def __init__(self, hidden: int = 5120, bottleneck: int = 64,
                 n_weeks: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = torch.nn.Linear(hidden, 1, bias=False)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, bottleneck), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)                # (N, P)
        attn_w = torch.softmax(scores, dim=-1)           # (N, P)
        ctx    = (x * attn_w.unsqueeze(-1)).sum(dim=1)   # (N, H)
        return self.head(ctx)                            # (N, n_weeks)


def train_head(X_raw: np.ndarray, y: np.ndarray, n_epochs: int = 500,
               lr: float = 1e-3) -> AttnBottleneck:
    """AttnBottleneck 학습. X_raw: (N, P, H), y: (N, n_weeks)."""
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


# ─── 유틸: 거리 / K-NN ────────────────────────────────────────────────────────

def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance. A: (m, d), B: (n, d) → (m, n)."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.clip(1.0 - An @ Bn.T, 0.0, 2.0)


def knn_rescale_factor(dist_row: np.ndarray, y_source: np.ndarray, K: int,
                       global_mean: float) -> float:
    """거리 행렬의 한 행으로부터 K-NN rescale 인수를 계산."""
    knn_idx = np.argsort(dist_row)[:K]
    knn_mean = float(y_source[knn_idx].mean())
    return knn_mean / (global_mean + 1e-6)


def knn_weighted_pred(dist_row: np.ndarray, y_source: np.ndarray, K: int) -> np.ndarray:
    """거리 역수 가중 K-NN 예측 (y_source: (N, n_weeks))."""
    knn_idx   = np.argsort(dist_row)[:K]
    dists     = dist_row[knn_idx] + 1e-12
    weights   = 1.0 / dists
    weights  /= weights.sum()
    return (y_source[knn_idx] * weights[:, None]).sum(axis=0)


# ─── 예측 → pred DataFrame 변환 ──────────────────────────────────────────────

def preds_to_df(preds: np.ndarray, item_ids: list[str], cat_ids: list[str],
                week_list: list, week_dates: dict) -> pd.DataFrame:
    """(N, n_weeks) → evaluate_weekly용 DataFrame."""
    records = []
    for i, item_id in enumerate(item_ids):
        for t, wk in enumerate(week_list):
            date = week_dates.get(wk)
            if date is None:
                continue
            records.append({
                "item_id":   item_id,
                "store_id":  "CA_1",
                "cat_id":    cat_ids[i],
                "date":      date,
                "pred_sales": max(0.0, float(preds[i, t])),
            })
    return pd.DataFrame(records)


# ─── 실험 F: Cold Items 적용 ──────────────────────────────────────────────────

def experiment_f(data: dict) -> dict:
    """Global head + K-NN rescale을 cold에 적용. K=[5,10,20,50]."""
    logger.info("=== 실험 F: Cold Items 적용 ===")

    warm_raw  = data["warm_raw"]    # (300, 50, 5120)
    cold_raw  = data["cold_raw"]    # (100, 50, 5120)
    warm_mean = data["warm_mean"]   # (300, 5120)
    cold_mean = data["cold_mean"]   # (100, 5120)
    y_warm    = data["y_warm"]      # (300, 16)
    cold_meta = data["cold_meta"]
    warm_meta = data["warm_meta"]

    # residual 생성
    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)  # (300, 50, 5120)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)  # (100, 50, 5120)

    # 글로벌 head 학습 (warm_residual)
    logger.info("  글로벌 head 학습 중 (n=300, %d epochs)...", TRAIN_EPOCHS_FULL)
    model = train_head(warm_residual, y_warm, n_epochs=TRAIN_EPOCHS_FULL)
    cold_pred_base = predict_head(model, cold_residual)    # (100, 16)

    # cold→warm cosine distance (mean_pooled 기준)
    dist_cold_warm = cosine_dist(cold_mean, warm_mean)     # (100, 300)

    # warm 판매량 통계
    item_sales_mean = y_warm.mean(axis=1)     # (300,)
    global_warm_mean = float(item_sales_mean.mean())
    logger.info("  warm mean weekly sales: %.2f", global_warm_mean)

    results = {}
    Ks = [5, 10, 20, 50]
    for K in Ks:
        preds = np.zeros_like(cold_pred_base)
        for i in range(len(cold_mean)):
            scale = knn_rescale_factor(dist_cold_warm[i], item_sales_mean, K, global_warm_mean)
            preds[i] = cold_pred_base[i] * scale

        pred_df = preds_to_df(
            preds,
            cold_meta["item_id"].tolist(),
            cold_meta["cat_id"].tolist(),
            data["week_list"], data["week_dates"],
        )
        ev = evaluate_weekly(
            data["cold_test_weekly"], pred_df, data["warm_train_weekly"],
            model_name=f"TrackB_rescale_K{K}"
        )
        results[K] = {"mae": ev["mae"], "dir_acc": ev["direction_accuracy"],
                      "wrmsse": ev["wrmsse"]}
        logger.info("  K=%2d → MAE=%.4f  DirAcc=%.3f  WRMSSE=%.4f",
                    K, ev["mae"], ev["direction_accuracy"], ev["wrmsse"])

    # K=0: global head only (no rescale), baseline for F
    pred_df0 = preds_to_df(
        cold_pred_base,
        cold_meta["item_id"].tolist(),
        cold_meta["cat_id"].tolist(),
        data["week_list"], data["week_dates"],
    )
    ev0 = evaluate_weekly(
        data["cold_test_weekly"], pred_df0, data["warm_train_weekly"],
        model_name="TrackB_global_head_only"
    )
    results["no_rescale"] = {"mae": ev0["mae"], "dir_acc": ev0["direction_accuracy"],
                             "wrmsse": ev0["wrmsse"]}
    logger.info("  No rescale → MAE=%.4f  DirAcc=%.3f", ev0["mae"], ev0["direction_accuracy"])

    return results


# ─── 실험 G: Rescale Ablation ─────────────────────────────────────────────────

def _warm_cv_predict(warm_residual: np.ndarray, y_warm: np.ndarray,
                     warm_mean: np.ndarray, item_sales_mean: np.ndarray,
                     global_warm_mean: float, K: int,
                     variant: str, rng_seed: int = 42) -> np.ndarray:
    """5-fold CV로 warm items에 대한 G 변형 예측값 반환. shape: (300, 16)."""
    n_items = len(warm_residual)
    n_weeks = y_warm.shape[1]
    preds = np.zeros((n_items, n_weeks), dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=rng_seed)

    for fold, (tr, val) in enumerate(kf.split(np.arange(n_items))):
        X_tr  = warm_residual[tr]
        y_tr  = y_warm[tr]
        X_val = warm_residual[val]

        # 거리 행렬 (val→tr)
        dist_val_tr = cosine_dist(warm_mean[val], warm_mean[tr])   # (|val|, |tr|)

        if variant in ("G1", "G2", "G4"):
            if variant == "G4":
                # 랜덤 embedding 생성 (같은 shape)
                rng = np.random.default_rng(rng_seed + fold)
                X_tr_input  = rng.standard_normal(X_tr.shape).astype(np.float32)
                X_val_input = rng.standard_normal(X_val.shape).astype(np.float32)
                # G4의 K-NN도 랜덤 거리로
                dist_val_tr_g4 = cosine_dist(
                    X_val_input.mean(axis=1), X_tr_input.mean(axis=1)
                )
            else:
                X_tr_input  = X_tr
                X_val_input = X_val
                dist_val_tr_g4 = dist_val_tr

            model = train_head(X_tr_input, y_tr, n_epochs=TRAIN_EPOCHS_CV)
            base_pred = predict_head(model, X_val_input)   # (|val|, 16)
        else:
            base_pred = None
            dist_val_tr_g4 = dist_val_tr

        for j, i in enumerate(val):
            tr_sales_mean = item_sales_mean[tr]
            if variant == "G1":
                preds[i] = base_pred[j]
            elif variant == "G2":
                scale = knn_rescale_factor(dist_val_tr[j], tr_sales_mean, K, global_warm_mean)
                preds[i] = base_pred[j] * scale
            elif variant == "G3":
                # K-NN 판매량 평균 (head 없음)
                knn_idx = np.argsort(dist_val_tr[j])[:K]
                preds[i] = y_tr[knn_idx].mean(axis=0)
            elif variant == "G4":
                scale = knn_rescale_factor(dist_val_tr_g4[j], tr_sales_mean, K, global_warm_mean)
                preds[i] = base_pred[j] * scale
            elif variant == "G5":
                preds[i] = knn_weighted_pred(dist_val_tr[j], y_tr, K)

    return preds


def experiment_g(data: dict) -> dict:
    """5개 변형으로 rescale 기여 분리. warm 5-fold CV + cold 평가."""
    logger.info("=== 실험 G: Rescale Ablation (K=%d) ===", K_ABLATION)

    warm_raw  = data["warm_raw"]
    cold_raw  = data["cold_raw"]
    warm_mean = data["warm_mean"]
    cold_mean = data["cold_mean"]
    y_warm    = data["y_warm"]
    cold_meta = data["cold_meta"]

    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)

    item_sales_mean  = y_warm.mean(axis=1)
    global_warm_mean = float(item_sales_mean.mean())

    dist_cold_warm = cosine_dist(cold_mean, warm_mean)   # (100, 300)

    results = {}

    # ── 공통: 전체 warm으로 글로벌 head 학습 (cold 평가용) ──
    logger.info("  글로벌 head 학습 (full warm, %d epochs)...", TRAIN_EPOCHS_FULL)
    global_model = train_head(warm_residual, y_warm, n_epochs=TRAIN_EPOCHS_FULL)
    cold_base    = predict_head(global_model, cold_residual)   # (100, 16)

    # G-4용 랜덤 head (3회 반복)
    logger.info("  G-4 랜덤 head 학습 (3회)...")
    random_cold_preds = []
    rng_master = np.random.default_rng(0)
    for rep in range(3):
        rng = np.random.default_rng(rep)
        rand_warm = rng.standard_normal(warm_residual.shape).astype(np.float32)
        rand_cold = rng.standard_normal(cold_residual.shape).astype(np.float32)
        rand_model = train_head(rand_warm, y_warm, n_epochs=TRAIN_EPOCHS_FULL)
        rand_cold_pred = predict_head(rand_model, rand_cold)   # (100, 16)
        # G-4의 K-NN도 랜덤 mean_pooled로
        rand_dist = cosine_dist(rand_cold.mean(axis=1), rand_warm.mean(axis=1))
        g4_cold = np.zeros_like(cold_base)
        for i in range(len(cold_meta)):
            scale = knn_rescale_factor(rand_dist[i], item_sales_mean, K_ABLATION, global_warm_mean)
            g4_cold[i] = rand_cold_pred[i] * scale
        random_cold_preds.append(g4_cold)
    g4_cold_avg = np.mean(random_cold_preds, axis=0)

    # ── Cold 평가 (5개 변형) ──────────────────────────────────────────────────
    variants_cold: dict[str, np.ndarray] = {}

    # G-1: global head only
    variants_cold["G1_head_only"] = cold_base.copy()

    # G-2: global head + rescale
    g2 = np.zeros_like(cold_base)
    for i in range(len(cold_meta)):
        scale = knn_rescale_factor(dist_cold_warm[i], item_sales_mean, K_ABLATION, global_warm_mean)
        g2[i] = cold_base[i] * scale
    variants_cold["G2_head_rescale"] = g2

    # G-3: K-NN mean sales (head 없음)
    g3 = np.zeros_like(cold_base)
    for i in range(len(cold_meta)):
        knn_idx = np.argsort(dist_cold_warm[i])[:K_ABLATION]
        g3[i] = y_warm[knn_idx].mean(axis=0)
    variants_cold["G3_knn_mean_sales"] = g3

    # G-4: random head + rescale (3회 평균)
    variants_cold["G4_random_rescale"] = g4_cold_avg

    # G-5: K-NN weighted avg
    g5 = np.zeros_like(cold_base)
    for i in range(len(cold_meta)):
        g5[i] = knn_weighted_pred(dist_cold_warm[i], y_warm, K_ABLATION)
    variants_cold["G5_knn_weighted"] = g5

    cold_results = {}
    for vname, preds in variants_cold.items():
        pred_df = preds_to_df(
            preds,
            cold_meta["item_id"].tolist(),
            cold_meta["cat_id"].tolist(),
            data["week_list"], data["week_dates"],
        )
        ev = evaluate_weekly(
            data["cold_test_weekly"], pred_df, data["warm_train_weekly"],
            model_name=vname
        )
        cold_results[vname] = {"mae": ev["mae"], "dir_acc": ev["direction_accuracy"],
                               "wrmsse": ev["wrmsse"]}
        logger.info("  [cold] %-20s MAE=%.4f  DirAcc=%.3f", vname, ev["mae"], ev["direction_accuracy"])

    # ── Warm 5-fold CV 평가 ────────────────────────────────────────────────────
    logger.info("  Warm 5-fold CV (G1~G5)...")
    warm_results = {}
    for variant in ["G1", "G2", "G3", "G4", "G5"]:
        logger.info("    %s warm CV 중...", variant)
        if variant == "G4":
            # 3회 반복 평균
            preds_list = [
                _warm_cv_predict(warm_residual, y_warm, warm_mean, item_sales_mean,
                                 global_warm_mean, K_ABLATION, variant, rng_seed=seed)
                for seed in [42, 123, 777]
            ]
            warm_pred = np.mean(preds_list, axis=0)
        else:
            warm_pred = _warm_cv_predict(
                warm_residual, y_warm, warm_mean, item_sales_mean,
                global_warm_mean, K_ABLATION, variant, rng_seed=42
            )
        warm_mae = float(np.abs(y_warm - warm_pred).mean())
        warm_results[variant] = warm_mae
        logger.info("    %s warm MAE=%.2f", variant, warm_mae)

    results["cold"] = cold_results
    results["warm"] = warm_results
    return results


# ─── 시각화 ───────────────────────────────────────────────────────────────────

def plot_f(f_results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # MAE bar chart: competitors + Track B K variants
    labels = list(COMPETITORS.keys()) + [f"TrackB_K{k}" for k in [5, 10, 20, 50]]
    mae_vals = [COMPETITORS[k]["mae"] for k in COMPETITORS] + \
               [f_results[k]["mae"] for k in [5, 10, 20, 50]]
    colors = ["#7fbbdd"] * len(COMPETITORS) + ["#e06c5b"] * 4
    ax = axes[0]
    bars = ax.bar(labels, mae_vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Weekly MAE"); ax.set_title("(F) Cold MAE: Track B vs Competitors")
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # DirAcc bar chart
    da_vals = [COMPETITORS[k]["dir_acc"] for k in COMPETITORS] + \
              [f_results[k]["dir_acc"] for k in [5, 10, 20, 50]]
    bars2 = axes[1].bar(labels, da_vals, color=colors)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Direction Accuracy"); axes[1].set_title("(F) DirAcc: Track B vs Competitors")
    for bar, val in zip(bars2, da_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_f_cold_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", FIG_DIR / "exp_f_cold_comparison.png")


def plot_g(g_results: dict) -> None:
    vnames = ["G1_head_only", "G2_head_rescale", "G3_knn_mean_sales",
              "G4_random_rescale", "G5_knn_weighted"]
    short  = ["G1", "G2", "G3", "G4", "G5"]

    cold_mae = [g_results["cold"][v]["mae"] for v in vnames]
    warm_mae = [g_results["warm"][v.split("_")[0]] for v in vnames]

    x = np.arange(len(vnames))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Grouped bar: warm vs cold MAE
    axes[0].bar(x - width/2, warm_mae, width, label="Warm 5-fold CV MAE",
                color="steelblue", alpha=0.85)
    axes[0].bar(x + width/2, cold_mae, width, label="Cold MAE",
                color="tomato", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(short)
    axes[0].set_ylabel("MAE"); axes[0].set_title("(G) Ablation: Warm CV vs Cold MAE")
    axes[0].legend()
    for i, (wm, cm) in enumerate(zip(warm_mae, cold_mae)):
        axes[0].text(i - width/2, wm + 0.3, f"{wm:.1f}", ha="center", fontsize=8)
        axes[0].text(i + width/2, cm + 0.3, f"{cm:.2f}", ha="center", fontsize=8)

    # Cold DirAcc
    cold_da = [g_results["cold"][v]["dir_acc"] for v in vnames]
    axes[1].bar(short, cold_da, color="seagreen", alpha=0.85)
    axes[1].set_ylabel("Direction Accuracy"); axes[1].set_title("(G) Cold DirAcc by Variant")
    for i, da in enumerate(cold_da):
        axes[1].text(i, da + 0.005, f"{da:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp_g_ablation.png", dpi=150)
    plt.close(fig)
    logger.info("시각화 저장: %s", FIG_DIR / "exp_g_ablation.png")


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(f_results: dict, g_results: dict) -> None:
    vnames = ["G1_head_only", "G2_head_rescale", "G3_knn_mean_sales",
              "G4_random_rescale", "G5_knn_weighted"]
    v_desc = {
        "G1_head_only":      "Global head only (no rescale)",
        "G2_head_rescale":   "Global head + K-NN rescale (우리 방법)",
        "G3_knn_mean_sales": "K-NN mean sales only (head 없음)",
        "G4_random_rescale": "Random embedding head + rescale (3회 평균)",
        "G5_knn_weighted":   "K-NN distance-weighted avg",
    }

    # 판정 로직
    g2_mae = g_results["cold"]["G2_head_rescale"]["mae"]
    g3_mae = g_results["cold"]["G3_knn_mean_sales"]["mae"]
    g4_mae = g_results["cold"]["G4_random_rescale"]["mae"]
    best_f_mae = min(f_results[k]["mae"] for k in [5, 10, 20, 50])

    embedding_contributes = (g2_mae < g3_mae * 0.97)   # G-2 << G-3
    embedding_over_random = (g4_mae > g2_mae * 1.03)   # G-4 >> G-2
    beats_best_competitor = (best_f_mae < min(v["mae"] for v in COMPETITORS.values()))

    lines: list[str] = [
        "# Cold Items 적용 + Rescale Ablation 보고서 (Exp016)",
        "",
        "**작성일:** 2026-03-10",
        "**평가 기준:** 16주 ISO-week (완전한 주만, exp006과 동일)",
        "",
        "---",
        "",
        "## 실험 F: Cold Items 적용 (K별 MAE/DirAcc)",
        "",
        "### Track B (rescale) vs Competitors",
        "",
        "| 모델 | MAE | DirAcc | WRMSSE |",
        "|------|-----|--------|--------|",
    ]
    for name, v in COMPETITORS.items():
        lines.append(f"| {name} | {v['mae']:.4f} | {v['dir_acc']:.3f} | — |")
    lines.append(f"| Global head only (no rescale) | {f_results['no_rescale']['mae']:.4f} | {f_results['no_rescale']['dir_acc']:.3f} | {f_results['no_rescale']['wrmsse']:.4f} |")
    for K in [5, 10, 20, 50]:
        v = f_results[K]
        lines.append(f"| TrackB_rescale_K={K} | {v['mae']:.4f} | {v['dir_acc']:.3f} | {v['wrmsse']:.4f} |")

    best_k = min([5, 10, 20, 50], key=lambda k: f_results[k]["mae"])
    lines += [
        "",
        f"- **Track B 최저 MAE:** {best_f_mae:.4f} (K={best_k})",
        f"- **최우수 competitor:** lightgbm_proxy_lags MAE=8.48",
        "",
    ]
    if beats_best_competitor:
        lines.append("→ Track B가 모든 competitor를 능가.")
    else:
        ratio = best_f_mae / 8.48
        lines.append(f"→ Track B 최저 MAE = {best_f_mae:.4f} (best competitor의 {ratio:.2f}배). 아직 competitor에 미치지 못함.")

    lines += ["", "---", "", "## 실험 G: Rescale Ablation (K=50)",
              "",
              "### Cold 평가",
              "",
              "| 변형 | 설명 | Cold MAE | DirAcc | WRMSSE |",
              "|------|------|----------|--------|--------|"]
    for vname in vnames:
        v = g_results["cold"][vname]
        lines.append(f"| {vname.split('_')[0]} | {v_desc[vname]} | {v['mae']:.4f} | {v['dir_acc']:.3f} | {v['wrmsse']:.4f} |")

    lines += ["",
              "### Warm 5-fold CV MAE",
              "",
              "| 변형 | Warm MAE |",
              "|------|----------|"]
    for vname in vnames:
        vk = vname.split("_")[0]
        lines.append(f"| {vk} | {g_results['warm'][vk]:.2f} |")

    lines += ["", "### 핵심 질문: embedding의 기여가 있는가?", ""]

    lines.append(f"| 비교 | G-2 MAE | G-3 MAE | G-4 MAE | 해석 |")
    lines.append("|------|---------|---------|---------|------|")
    lines.append(f"| G-2 vs G-3 | {g2_mae:.4f} | {g3_mae:.4f} | — | "
                 + ("G-2 < G-3: head가 K-NN 위에 추가 정보 제공" if embedding_contributes
                    else "G-2 ≈ G-3: head의 기여 없음, K-NN 판매량이 전부") + " |")
    lines.append(f"| G-2 vs G-4 | {g2_mae:.4f} | — | {g4_mae:.4f} | "
                 + ("G-4 > G-2: 실제 embedding이 random보다 나음" if embedding_over_random
                    else "G-4 ≈ G-2: embedding 품질 무관, rescale이 전부") + " |")

    lines += [""]

    # 종합 판정
    if embedding_contributes and embedding_over_random:
        emb_verdict = "**LLM persona embedding이 실질적으로 기여한다 (YES)**"
        emb_note    = "G-2 < G-3(head 기여 확인) + G-4 > G-2(embedding 품질 기여 확인)."
    elif embedding_contributes:
        emb_verdict = "**부분적 기여 — head는 기여하나 embedding 품질 효과는 불명확**"
        emb_note    = "G-2 < G-3이나 G-4 ≈ G-2 → head 구조는 필요하지만 LLM embedding 자체의 기여는 제한적."
    elif embedding_over_random:
        emb_verdict = "**부분적 기여 — embedding 품질은 기여하나 K-NN 대비 head 추가 이점 없음**"
        emb_note    = "G-4 > G-2(embedding 품질 효과)이나 G-2 ≈ G-3 → K-NN 판매량 평균이 주요 기여."
    else:
        emb_verdict = "**LLM persona embedding의 독립적 기여 없음 (NO)**"
        emb_note    = "G-2 ≈ G-3 ≈ G-4 → rescale(K-NN 판매량 통계)이 전부. LLM embedding은 기여 없음."

    lines += [
        "## 종합 판정",
        "",
        f"### {emb_verdict}",
        f"> {emb_note}",
        "",
        "### 실험 F 종합",
        f"> Track B 최저 cold MAE = {best_f_mae:.4f} (K={best_k}), best competitor(8.48) 대비 {'상회' if beats_best_competitor else '미달'}.",
        "",
        "### 다음 단계 권고",
        "",
    ]
    if not embedding_contributes:
        lines += [
            "- G-2 ≈ G-3: head를 제거하고 단순 K-NN mean sales만으로 cold-start 예측 가능",
            "- LLM persona embedding의 기여를 높이려면 더 강한 persona conditioning 또는 contrastive learning 필요",
        ]
    else:
        lines += [
            "- 현재 파이프라인(global head + K-NN rescale) 방향 유효",
            "- warm 확장(2,949개)으로 head 품질 개선 후 재평가 권장",
        ]
    lines += [
        "- DirAcc 개선을 위해 방향성 loss term 추가 검토",
        "",
        "---",
        "",
        "**시각화:** `experiments/exp016_cold_application/figures/`",
        "**스크립트:** `scripts/exp016_cold_application.py`",
    ]

    path = REPORT_DIR / "cold_application_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", path)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Exp016: Cold Application + Rescale Ablation ===")
    data = load_data()
    f_results = experiment_f(data)
    g_results = experiment_g(data)
    plot_f(f_results)
    plot_g(g_results)
    write_report(f_results, g_results)

    print("\n" + "=" * 65)
    print("=== Exp016 결과 요약 ===")
    print(f"\n[F] Cold MAE (K별):")
    for K in [5, 10, 20, 50]:
        print(f"  K={K:2d}: MAE={f_results[K]['mae']:.4f}  DirAcc={f_results[K]['dir_acc']:.3f}")
    print(f"  Best competitor: lightgbm_proxy_lags MAE=8.48")
    print(f"\n[G] Ablation (cold, K={K_ABLATION}):")
    for vname in ["G1_head_only", "G2_head_rescale", "G3_knn_mean_sales",
                  "G4_random_rescale", "G5_knn_weighted"]:
        v = g_results["cold"][vname]
        print(f"  {vname:<25} MAE={v['mae']:.4f}  DirAcc={v['dir_acc']:.3f}")
    print(f"\n[G] Warm CV MAE:")
    for k in ["G1", "G2", "G3", "G4", "G5"]:
        print(f"  {k}: {g_results['warm'][k]:.2f}")
    print(f"\n보고서: docs/diagnosis/cold_application_report.md")
    print("=" * 65)


if __name__ == "__main__":
    main()
