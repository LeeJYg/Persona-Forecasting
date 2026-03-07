"""Exp009: Per-Persona Raw Embeddings + Attention Head 실험.

Steps
-----
1. Raw per-persona 임베딩 추출 (+ mean-pooled 검증용, item-only ablation)
2. Shape assert → 실패 시 즉시 종료
3. 6개 Head 학습 및 5-fold CV 평가
4. 비교 테이블 생성

출력: experiments/exp009_attention_head/
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.forecasting.qwen_embedder import (
    QwenEmbedder,
    build_combined_text,
    build_item_text,
    build_persona_text,
)
from src.models.persona.schema import Persona

# ─── 로깅 설정 ────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )

logger = logging.getLogger(__name__)

# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def agg_weekly(sales_df: pd.DataFrame, item_ids: list[str],
               date_start: str, date_end: str) -> np.ndarray:
    sub = sales_df[
        sales_df["item_id"].isin(item_ids) &
        (sales_df["date"] >= date_start) &
        (sales_df["date"] <= date_end)
    ].copy()
    sub["week"] = pd.to_datetime(sub["date"]).dt.to_period("W")
    w = (sub.groupby(["item_id", "week"])["sales"].sum()
           .unstack("week").reindex(item_ids).fillna(0))
    return w.values.astype(np.float32)  # (n_items, n_weeks)


def dir_acc_weekly(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[1] < 2:
        return float("nan")
    td = np.sign(np.diff(y_true, axis=1))
    pd_ = np.sign(np.diff(y_pred, axis=1))
    m = td != 0
    return float((td[m] == pd_[m]).mean()) if m.sum() > 0 else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


# weekly → daily 일별 예측 DataFrame 변환
def weekly_to_daily_df(
    y_weekly: np.ndarray,
    item_ids: list[str],
    cold_test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for i, iid in enumerate(item_ids):
        item_rows = cold_test[cold_test["item_id"] == iid].sort_values("date")
        dates = item_rows["date"].tolist()
        week_periods = pd.to_datetime(dates).to_period("W")
        unique_weeks = sorted(set(week_periods))
        n_weeks = y_weekly.shape[1]
        wp_map = {wp: float(y_weekly[i, j]) for j, wp in enumerate(unique_weeks) if j < n_weeks}
        n_days = item_rows.groupby(week_periods).size()
        for date, wp in zip(dates, week_periods):
            rows.append({"item_id": iid, "date": date,
                         "pred_sales": wp_map.get(wp, 0.0) / n_days.get(wp, 7)})
    return pd.DataFrame(rows)


# ─── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_base_data(config):
    cold_dir = ROOT / config.paths.cold_start_dir
    logger.info("[데이터] 로드 중...")
    cold_test  = pd.read_csv(cold_dir / "cold_test.csv",  parse_dates=["date"])
    warm_train = pd.read_csv(cold_dir / "warm_train.csv", parse_dates=["date"])
    warm_test  = pd.read_csv(cold_dir / "warm_test.csv",  parse_dates=["date"])
    sell_prices = pd.read_csv(ROOT / config.paths.sell_prices)
    return cold_test, warm_train, warm_test, sell_prices


def load_personas(personas_dir: Path, n: int) -> list[Persona]:
    files = sorted(personas_dir.glob("CA_1_P*.json"))[:n]
    out = []
    for fp in files:
        try:
            out.append(Persona.from_dict(json.loads(fp.read_text(encoding="utf-8"))))
        except Exception as e:
            logger.warning("페르소나 로드 실패 %s: %s", fp.name, e)
    return out


def build_item_meta(item_ids, sales_df, sell_prices, store_id, lookback=13):
    meta = (sales_df[sales_df["item_id"].isin(item_ids)][["item_id", "dept_id", "cat_id"]]
            .drop_duplicates("item_id").set_index("item_id"))
    price = (sell_prices[(sell_prices["store_id"] == store_id) &
                         (sell_prices["item_id"].isin(item_ids))]
             .groupby("item_id")
             .apply(lambda g: g.nlargest(lookback, "wm_yr_wk")["sell_price"].mean())
             .rename("avg_price"))
    out = {}
    for iid in item_ids:
        row = meta.loc[iid] if iid in meta.index else None
        out[iid] = {
            "dept_id": str(row["dept_id"]) if row is not None else "UNKNOWN",
            "cat_id":  str(row["cat_id"])  if row is not None else "UNKNOWN",
            "avg_price": float(price.loc[iid]) if iid in price.index else None,
        }
    return out


def sample_warm_items(warm_train, cold_ids_set, n, seed):
    warm = (warm_train[~warm_train["item_id"].isin(cold_ids_set)]
            [["item_id", "cat_id"]].drop_duplicates("item_id"))
    cats = sorted(warm["cat_id"].unique())
    per_cat = n // len(cats); rem = n % len(cats)
    rng = np.random.default_rng(seed); sampled = []
    for i, cat in enumerate(cats):
        items = warm[warm["cat_id"] == cat]["item_id"].tolist()
        k = per_cat + (1 if i < rem else 0)
        sampled.extend(rng.choice(items, size=min(k, len(items)), replace=False).tolist())
    return sampled


# ─── Step 1: Raw + Mean + Item-Only 임베딩 추출 ────────────────────────────────

def step1_extract_embeddings(
    embedder: QwenEmbedder,
    cold_ids: list[str],
    warm_ids: list[str],
    item_meta: dict,
    personas: list[Persona],
    out_dir: Path,
) -> dict[str, Path]:
    """Raw per-persona, mean-pooled, item-only 임베딩을 추출해 .pt로 저장."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    n_personas = len(personas)
    hidden_dim = embedder._hidden_size

    persona_texts = [build_persona_text(p.profile, condition="A") for p in personas]

    def extract_raw_and_mean(item_ids: list[str], tag: str):
        item_texts = {
            iid: build_item_text(
                item_id=iid,
                dept_id=item_meta[iid]["dept_id"],
                cat_id=item_meta[iid]["cat_id"],
                avg_price=item_meta[iid]["avg_price"],
            )
            for iid in item_ids
        }
        # all_texts: (n_items × n_personas,) — item 순서, persona 순서
        all_texts = [
            build_combined_text(pt, item_texts[iid])
            for iid in item_ids
            for pt in persona_texts
        ]
        logger.info("[%s] combined 텍스트 %d개 임베딩 추출 중...", tag, len(all_texts))
        t0 = time.time()
        all_emb = embedder.get_embeddings(all_texts)  # (n_items*n_personas, hidden)
        elapsed = time.time() - t0
        logger.info("[%s] 추출 완료: %.1f초", tag, elapsed)

        raw  = all_emb.reshape(len(item_ids), n_personas, hidden_dim)  # (N, 50, 5120)
        mean = raw.mean(axis=1)                                          # (N, 5120)
        return (torch.tensor(raw,  dtype=torch.float32),
                torch.tensor(mean, dtype=torch.float32))

    def extract_item_only(item_ids: list[str], tag: str):
        texts = [
            build_item_text(iid, item_meta[iid]["dept_id"],
                            item_meta[iid]["cat_id"], item_meta[iid]["avg_price"])
            for iid in item_ids
        ]
        logger.info("[%s] item-only %d개 추출 중...", tag, len(texts))
        emb = embedder.get_embeddings(texts)
        return torch.tensor(emb, dtype=torch.float32)

    # warm
    logger.info("=== warm raw+mean 추출 시작 (%s) ===", ts())
    warm_raw, warm_mean = extract_raw_and_mean(warm_ids, "warm")

    # cold
    logger.info("=== cold raw+mean 추출 시작 (%s) ===", ts())
    cold_raw, cold_mean = extract_raw_and_mean(cold_ids, "cold")

    # item-only (warm + cold)
    logger.info("=== item-only 추출 시작 (%s) ===", ts())
    all_ids = warm_ids + cold_ids
    item_only_all = extract_item_only(all_ids, "item_only")
    warm_item_only = item_only_all[:len(warm_ids)]
    cold_item_only = item_only_all[len(warm_ids):]

    # 저장
    paths = {
        "warm_raw":       out_dir / "warm_raw.pt",
        "cold_raw":       out_dir / "cold_raw.pt",
        "warm_mean":      out_dir / "warm_mean.pt",
        "cold_mean":      out_dir / "cold_mean.pt",
        "warm_item_only": out_dir / "warm_item_only.pt",
        "cold_item_only": out_dir / "cold_item_only.pt",
    }
    for key, tensor in [
        ("warm_raw", warm_raw), ("cold_raw", cold_raw),
        ("warm_mean", warm_mean), ("cold_mean", cold_mean),
        ("warm_item_only", warm_item_only), ("cold_item_only", cold_item_only),
    ]:
        torch.save(tensor, paths[key])
        logger.info("저장: %s %s", paths[key].name, tuple(tensor.shape))

    return paths


def step1_assert(paths, n_warm, n_cold, n_personas, hidden_dim):
    import torch
    logger.info("=== Step 1 Shape Assert ===")
    expected = {
        "warm_raw":       (n_warm, n_personas, hidden_dim),
        "cold_raw":       (n_cold, n_personas, hidden_dim),
        "warm_mean":      (n_warm, hidden_dim),
        "cold_mean":      (n_cold, hidden_dim),
        "warm_item_only": (n_warm, hidden_dim),
        "cold_item_only": (n_cold, hidden_dim),
    }
    for key, exp_shape in expected.items():
        t = torch.load(paths[key], weights_only=True)
        assert tuple(t.shape) == exp_shape, (
            f"FAIL {key}: expected {exp_shape}, got {tuple(t.shape)}"
        )
        logger.info("  OK  %s  %s", key, tuple(t.shape))
    logger.info("All shapes OK.")


# ─── Ridge CV 헬퍼 ────────────────────────────────────────────────────────────

def ridge_cv_eval(X_warm, y_warm, X_cold, y_cold_weekly, alpha=1.0, cv=5):
    """5-fold CV warm val MAE + cold MAE / RMSE / DirAcc 반환."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    n_weeks = y_warm.shape[1]
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    val_maes = []
    for tr_idx, val_idx in kf.split(X_warm):
        wk_val_maes = []
        for w in range(n_weeks):
            mdl = Ridge(alpha=alpha)
            mdl.fit(X_warm[tr_idx], y_warm[tr_idx, w])
            pred = mdl.predict(X_warm[val_idx])
            wk_val_maes.append(float(np.abs(y_warm[val_idx, w] - pred).mean()))
        val_maes.append(np.mean(wk_val_maes))
    warm_val_mae = float(np.mean(val_maes))

    cold_pred = np.zeros_like(y_cold_weekly)
    for w in range(n_weeks):
        mdl = Ridge(alpha=alpha)
        mdl.fit(X_warm, y_warm[:, w])
        cold_pred[:, w] = mdl.predict(X_cold)

    return dict(
        warm_val_mae=warm_val_mae,
        cold_mae=mae(y_cold_weekly, cold_pred),
        cold_rmse=rmse(y_cold_weekly, cold_pred),
        cold_dir_acc=dir_acc_weekly(y_cold_weekly, cold_pred),
    )


# ─── Attention Head 모델 정의 ─────────────────────────────────────────────────

def build_attn_model(hidden_dim: int, n_weeks: int, bottleneck: int | None = None,
                     dropout: float = 0.5):
    import torch.nn as nn
    class AttnHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(hidden_dim, 1, bias=False)
            if bottleneck:
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim, bottleneck),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(bottleneck, n_weeks),
                )
            else:
                self.head = nn.Linear(hidden_dim, n_weeks)

        def forward(self, x):  # x: (batch, n_personas, hidden)
            scores  = self.attn(x)                          # (batch, n_personas, 1)
            weights = scores.softmax(dim=1)                 # (batch, n_personas, 1)
            agg     = (weights * x).sum(dim=1)              # (batch, hidden)
            return self.head(agg)                           # (batch, n_weeks)

    return AttnHead()


def attn_cv_eval(
    X_warm_raw: np.ndarray,   # (300, 50, 5120)
    y_warm: np.ndarray,       # (300, n_weeks)
    X_cold_raw: np.ndarray,   # (100, 50, 5120)
    y_cold: np.ndarray,
    bottleneck: int | None,
    dropout: float,
    model_save_dir: Path,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    cv: int = 5,
) -> dict:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from sklearn.model_selection import KFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_items, n_personas, hidden_dim = X_warm_raw.shape
    n_weeks = y_warm.shape[1]

    X_warm_t = torch.tensor(X_warm_raw, dtype=torch.float32)
    X_cold_t = torch.tensor(X_cold_raw, dtype=torch.float32)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    fold_val_maes = []

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_warm_t)):
        model = build_attn_model(hidden_dim, n_weeks, bottleneck, dropout).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        X_tr = X_warm_t[tr_idx].to(device)
        y_tr = torch.tensor(y_warm[tr_idx], dtype=torch.float32, device=device)
        X_v  = X_warm_t[val_idx].to(device)
        y_v  = torch.tensor(y_warm[val_idx], dtype=torch.float32, device=device)

        best_val, no_imp, best_state = float("inf"), 0, None
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_tr), y_tr)
            loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                vl = criterion(model(X_v), y_v).item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v).cpu().numpy()
        fold_val_maes.append(mae(y_warm[val_idx], val_pred))
        logger.info("  fold %d/%d  val_MAE=%.4f  (best_val_loss=%.4f)",
                    fold_i + 1, cv, fold_val_maes[-1], best_val)

    warm_val_mae = float(np.mean(fold_val_maes))

    # 전체 warm으로 재학습 → cold 예측
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model = build_attn_model(hidden_dim, n_weeks, bottleneck, dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    X_all = X_warm_t.to(device)
    y_all = torch.tensor(y_warm, dtype=torch.float32, device=device)

    best_loss, no_imp, best_state = float("inf"), 0, None
    for ep in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(X_all), y_all)
        loss.backward(); optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience: break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_save_dir / "model.pt")

    model.eval()
    with torch.no_grad():
        cold_pred = model(X_cold_t.to(device)).cpu().numpy()  # (100, n_weeks)

    return dict(
        warm_val_mae=warm_val_mae,
        cold_mae=mae(y_cold, cold_pred),
        cold_rmse=rmse(y_cold, cold_pred),
        cold_dir_acc=dir_acc_weekly(y_cold, cold_pred),
    )


# ─── Step 3: 6 Heads ──────────────────────────────────────────────────────────

def step3_train_all_heads(paths, warm_ids, cold_ids,
                          y_warm, y_cold, models_dir, results_dir):
    import torch
    results_dir.mkdir(parents=True, exist_ok=True)

    X_warm_mean = torch.load(paths["warm_mean"],      weights_only=True).numpy()
    X_cold_mean = torch.load(paths["cold_mean"],      weights_only=True).numpy()
    X_warm_io   = torch.load(paths["warm_item_only"], weights_only=True).numpy()
    X_cold_io   = torch.load(paths["cold_item_only"], weights_only=True).numpy()
    X_warm_raw  = torch.load(paths["warm_raw"],       weights_only=True).numpy()
    X_cold_raw  = torch.load(paths["cold_raw"],       weights_only=True).numpy()

    n_personas  = X_warm_raw.shape[1]
    hidden_dim  = X_warm_raw.shape[2]
    n_weeks     = y_warm.shape[1]

    results = {}

    # 3-1. Ridge (mean-pooled)
    logger.info("\n--- 3-1. Ridge (mean-pooled) [%s] ---", ts())
    results["3-1_ridge_mean"] = ridge_cv_eval(X_warm_mean, y_warm, X_cold_mean, y_cold)
    logger.info("  warm_val=%.4f  cold=%.4f  dir=%.4f",
                results["3-1_ridge_mean"]["warm_val_mae"],
                results["3-1_ridge_mean"]["cold_mae"],
                results["3-1_ridge_mean"]["cold_dir_acc"])

    # 3-2. Ridge (item-only)
    logger.info("\n--- 3-2. Ridge (item-only) [%s] ---", ts())
    results["3-2_ridge_item_only"] = ridge_cv_eval(X_warm_io, y_warm, X_cold_io, y_cold)
    logger.info("  warm_val=%.4f  cold=%.4f  dir=%.4f",
                results["3-2_ridge_item_only"]["warm_val_mae"],
                results["3-2_ridge_item_only"]["cold_mae"],
                results["3-2_ridge_item_only"]["cold_dir_acc"])

    # 3-3. Attention + Linear
    logger.info("\n--- 3-3. Attention + Linear [%s] ---", ts())
    results["3-3_attn_linear"] = attn_cv_eval(
        X_warm_raw, y_warm, X_cold_raw, y_cold,
        bottleneck=None, dropout=0.0,
        model_save_dir=models_dir / "attn_linear",
    )
    logger.info("  warm_val=%.4f  cold=%.4f  dir=%.4f",
                results["3-3_attn_linear"]["warm_val_mae"],
                results["3-3_attn_linear"]["cold_mae"],
                results["3-3_attn_linear"]["cold_dir_acc"])

    # 3-4. Attention + Bottleneck
    logger.info("\n--- 3-4. Attention + Bottleneck [%s] ---", ts())
    results["3-4_attn_bottleneck"] = attn_cv_eval(
        X_warm_raw, y_warm, X_cold_raw, y_cold,
        bottleneck=64, dropout=0.5,
        model_save_dir=models_dir / "attn_bottleneck",
    )
    logger.info("  warm_val=%.4f  cold=%.4f  dir=%.4f",
                results["3-4_attn_bottleneck"]["warm_val_mae"],
                results["3-4_attn_bottleneck"]["cold_mae"],
                results["3-4_attn_bottleneck"]["cold_dir_acc"])

    # 3-5. Per-Persona Variance + Ridge (mean + std → 10240)
    logger.info("\n--- 3-5. Variance + Ridge [%s] ---", ts())
    X_warm_var = np.concatenate(
        [X_warm_raw.mean(axis=1), X_warm_raw.std(axis=1)], axis=1)  # (300, 10240)
    X_cold_var = np.concatenate(
        [X_cold_raw.mean(axis=1), X_cold_raw.std(axis=1)], axis=1)
    results["3-5_variance_ridge"] = ridge_cv_eval(X_warm_var, y_warm, X_cold_var, y_cold)
    logger.info("  warm_val=%.4f  cold=%.4f  dir=%.4f",
                results["3-5_variance_ridge"]["warm_val_mae"],
                results["3-5_variance_ridge"]["cold_mae"],
                results["3-5_variance_ridge"]["cold_dir_acc"])

    # 3-6. Variance + PCA(64) + Ridge
    logger.info("\n--- 3-6. Variance + PCA(64) + Ridge [%s] ---", ts())
    from sklearn.decomposition import PCA
    pca = PCA(n_components=64, random_state=42).fit(X_warm_var)
    X_warm_pca = pca.transform(X_warm_var)
    X_cold_pca = pca.transform(X_cold_var)
    results["3-6_variance_pca_ridge"] = ridge_cv_eval(X_warm_pca, y_warm, X_cold_pca, y_cold)
    logger.info("  warm_val=%.4f  cold=%.4f  dir=%.4f",
                results["3-6_variance_pca_ridge"]["warm_val_mae"],
                results["3-6_variance_pca_ridge"]["cold_mae"],
                results["3-6_variance_pca_ridge"]["cold_dir_acc"])

    return results


# ─── 최종 비교 테이블 ─────────────────────────────────────────────────────────

PARAMS_MAP = {
    "3-1_ridge_mean":        5121,
    "3-2_ridge_item_only":   5121,
    "3-3_attn_linear":       5120 + 1 + 5120,  # attn + out (per week head)
    "3-4_attn_bottleneck":   5120 + 1 + 5120 * 64 + 64 + 64,
    "3-5_variance_ridge":    10241,
    "3-6_variance_pca_ridge": 65,
}
HEAD_NAMES = {
    "3-1_ridge_mean":        "3-1. Ridge (mean-pooled)",
    "3-2_ridge_item_only":   "3-2. Ridge (item-only)",
    "3-3_attn_linear":       "3-3. Attention + Linear",
    "3-4_attn_bottleneck":   "3-4. Attention + Bottleneck",
    "3-5_variance_ridge":    "3-5. Variance + Ridge",
    "3-6_variance_pca_ridge":"3-6. Variance + PCA(64) + Ridge",
}


def print_comparison_table(results: dict, results_dir: Path) -> None:
    rows = []
    for key, r in results.items():
        rows.append({
            "Head": HEAD_NAMES.get(key, key),
            "warm_val_MAE": round(r["warm_val_mae"], 4),
            "cold_MAE":     round(r["cold_mae"], 4),
            "cold_RMSE":    round(r["cold_rmse"], 4),
            "cold_DirAcc":  round(r["cold_dir_acc"], 4),
            "params":       PARAMS_MAP.get(key, "?"),
        })

    df = pd.DataFrame(rows)
    logger.info("\n=== Exp009 Head 비교 테이블 ===")
    logger.info("\n%s", df.to_string(index=False))

    # 기존 exp005 참조값 추가
    ref_rows = [
        {"source": "exp005 Ridge(5120)", "cold_MAE": 68.30, "cold_DirAcc": 0.536,
         "note": "weekly 단위 DirAcc"},
        {"source": "exp006 LightGBM lags", "cold_MAE": 8.48, "cold_DirAcc": 0.343,
         "note": "exp006 없음(N/A)"},
        {"source": "exp006 Track A calib", "cold_MAE": 8.90, "cold_DirAcc": 0.393,
         "note": "exp006 없음(N/A)"},
    ]
    logger.info("\n참고값 (exp005/006):")
    for r in ref_rows:
        logger.info("  %-28s  cold_MAE=%-7s  DirAcc=%-6s  [%s]",
                    r["source"], r["cold_MAE"], r["cold_DirAcc"], r["note"])

    df.to_csv(results_dir / "comparison.csv", index=False)
    (results_dir / "metrics_per_head.json").write_text(
        json.dumps({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()} for k, v in results.items()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("\n결과 저장: %s", results_dir)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    config  = load_config()

    exp_dir     = ROOT / "experiments/exp009_attention_head"
    emb_dir     = exp_dir / "embeddings"
    models_dir  = exp_dir / "models"
    results_dir = exp_dir / "results"

    setup_logging(exp_dir / "run.log")
    logger.info("=== Exp009 시작 (%s) ===", ts())

    # ── 디스크 여유 확인 ──────────────────────────────────────
    import shutil
    free_gb = shutil.disk_usage(ROOT).free / 1024 ** 3
    logger.info("디스크 여유: %.1f GB (루트)", free_gb)
    assert free_gb > 1.0, f"루트 디스크 여유 부족: {free_gb:.1f} GB"
    logger.info("디스크 여유 OK")

    # ── 기본 데이터 로드 ──────────────────────────────────────
    cold_test, warm_train, warm_test, sell_prices = load_base_data(config)
    cold_ids = sorted(cold_test["item_id"].unique().tolist())
    warm_ids = sample_warm_items(
        warm_train, set(cold_ids),
        int(config.experiment.track_b.n_warm_items_for_training),
        int(config.experiment.seed),
    )
    personas = load_personas(
        ROOT / config.paths.personas_dir,
        int(config.experiment.track_b.n_personas_for_embedding),
    )
    assert personas, "페르소나 없음"
    logger.info("cold=%d  warm=%d  personas=%d", len(cold_ids), len(warm_ids), len(personas))

    all_item_ids = cold_ids + warm_ids
    item_meta = build_item_meta(
        all_item_ids,
        pd.concat([cold_test, warm_train], ignore_index=True),
        sell_prices,
        str(config.experiment.cold_start.target_store),
    )

    # ── item_meta.csv ─────────────────────────────────────────
    emb_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = [{"item_id": iid, "cat_id": item_meta[iid]["cat_id"],
                  "dept_id": item_meta[iid]["dept_id"],
                  "is_cold": iid in set(cold_ids)}
                 for iid in all_item_ids]
    pd.DataFrame(meta_rows).to_csv(emb_dir / "item_meta.csv", index=False)

    # ── Step 1: 임베딩 추출 ───────────────────────────────────
    tb_cfg = config.experiment.track_b
    embedder = QwenEmbedder(
        model_name=str(tb_cfg.model_name),
        dtype=str(tb_cfg.embedding_dtype),
        quantization=dict(tb_cfg.quantization),
        device_map="auto",
        batch_size=int(tb_cfg.embedding_batch_size),
        cache_dir="/mnt/sdd1/jylee/huggingface_cache",
    )
    embedder.load()

    paths = step1_extract_embeddings(
        embedder, cold_ids, warm_ids, item_meta, personas, emb_dir
    )

    # ── Step 1 Shape Assert ───────────────────────────────────
    hidden_dim = embedder._hidden_size
    try:
        step1_assert(paths, len(warm_ids), len(cold_ids), len(personas), hidden_dim)
    except AssertionError as e:
        logger.error("Step 1 FAILED: %s", e)
        sys.exit(1)
    logger.info("=== Step 1 완료 (%s) ===", ts())

    # 모델 메모리 해제
    del embedder
    import torch, gc
    torch.cuda.empty_cache(); gc.collect()

    # ── 판매량 집계 ───────────────────────────────────────────
    date_start = str(cold_test["date"].min().date())
    date_end   = str(cold_test["date"].max().date())
    y_warm = agg_weekly(warm_test, warm_ids, date_start, date_end)
    y_cold = agg_weekly(cold_test, cold_ids, date_start, date_end)
    logger.info("y_warm=%s  y_cold=%s", y_warm.shape, y_cold.shape)

    # ── Step 3: Head 학습 ─────────────────────────────────────
    logger.info("=== Step 3: Head 학습 시작 (%s) ===", ts())
    results = step3_train_all_heads(
        paths, warm_ids, cold_ids, y_warm, y_cold, models_dir, results_dir
    )

    # ── 비교 테이블 ───────────────────────────────────────────
    print_comparison_table(results, results_dir)

    elapsed = (time.time() - t_start) / 60
    logger.info("=== Exp009 완료 (%s) | 총 %.1f분 ===", ts(), elapsed)


if __name__ == "__main__":
    main()
