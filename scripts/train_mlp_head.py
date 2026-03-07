"""Track B: MLP Regression Head 학습 및 평가.

기존 mean-pooled 임베딩(exp005)을 재사용하고 head만 교체한다.
raw per-persona 임베딩(300×50×5120)이 저장되지 않아 Attention Head는 생략.

출력: experiments/exp005_track_b_embedding/mlp_head/
  - mlp_head.pt          (학습된 모델 가중치)
  - training_curve.png   (train/val loss 곡선)
  - evaluation_results.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# MLP Head 정의
# ──────────────────────────────────────────────────────────────

def build_mlp(input_dim: int = 5120):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ──────────────────────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────────────────────

def train_one_week(
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[list[float], list[float], int]:
    """단일 주차 타겟에 대해 MLP를 학습한다.

    Returns:
        (train_losses, val_losses, best_epoch)
    """
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32, device=device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr_t)
        loss = criterion(pred, y_tr_t)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(criterion(val_pred, y_val_t).item())
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            best_epoch    = epoch
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, best_epoch


# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────

def aggregate_weekly(sales_df: pd.DataFrame, item_ids: list[str],
                     date_start: str, date_end: str) -> np.ndarray:
    sub = sales_df[
        sales_df["item_id"].isin(item_ids) &
        (sales_df["date"] >= date_start) &
        (sales_df["date"] <= date_end)
    ].copy()
    sub["week"] = pd.to_datetime(sub["date"]).dt.to_period("W")
    weekly = (
        sub.groupby(["item_id", "week"])["sales"]
        .sum()
        .unstack("week")
        .reindex(item_ids)
        .fillna(0)
    )
    return weekly.values  # (n_items, n_weeks)


def direction_accuracy_weekly(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[1] < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true, axis=1))
    pred_dir = np.sign(np.diff(y_pred, axis=1))
    mask = true_dir != 0
    return float((true_dir[mask] == pred_dir[mask]).mean()) if mask.sum() > 0 else float("nan")


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config()
    exp_dir     = ROOT / "experiments/exp005_track_b_embedding"
    out_dir     = exp_dir / "mlp_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    cold_dir    = ROOT / config.paths.cold_start_dir

    # ── 데이터 로드 ──────────────────────────────────────────
    logger.info("데이터 로드 중...")
    cold_test  = pd.read_csv(cold_dir / "cold_test.csv",  parse_dates=["date"])
    warm_test  = pd.read_csv(cold_dir / "warm_test.csv",  parse_dates=["date"])

    d_warm = np.load(exp_dir / "embeddings/item_emb_warm.npz", allow_pickle=True)
    d_cold = np.load(exp_dir / "embeddings/item_emb_cold.npz", allow_pickle=True)
    X_warm = d_warm["embeddings"].astype(np.float32)
    X_cold = d_cold["embeddings"].astype(np.float32)
    warm_ids = d_warm["item_ids"].tolist()
    cold_ids = d_cold["item_ids"].tolist()

    date_start = str(cold_test["date"].min().date())
    date_end   = str(cold_test["date"].max().date())

    y_warm_weekly = aggregate_weekly(warm_test, warm_ids, date_start, date_end)
    y_cold_weekly = aggregate_weekly(cold_test, cold_ids, date_start, date_end)
    n_items, n_weeks = y_warm_weekly.shape
    logger.info("warm=%d items × %d weeks, cold=%d items",
                n_items, n_weeks, len(cold_ids))

    # ── 80/20 train/val split ────────────────────────────────
    rng = np.random.default_rng(int(config.experiment.seed))
    idx = rng.permutation(n_items)
    n_train = int(n_items * 0.8)
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    X_tr,  X_val  = X_warm[tr_idx],       X_warm[val_idx]
    y_tr_w, y_val_w = y_warm_weekly[tr_idx], y_warm_weekly[val_idx]
    logger.info("train=%d  val=%d", len(tr_idx), len(val_idx))

    # ── 주차별 MLP 학습 ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    y_cold_pred = np.zeros_like(y_cold_weekly, dtype=float)
    val_maes, all_train_curves, all_val_curves = [], [], []

    for w in range(n_weeks):
        model = build_mlp(input_dim=X_warm.shape[1])
        tr_losses, val_losses, best_ep = train_one_week(
            model,
            X_tr,  y_tr_w[:, w],
            X_val, y_val_w[:, w],
        )
        all_train_curves.append(tr_losses)
        all_val_curves.append(val_losses)

        # val MAE
        model.eval().to(device)
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, device=device)
            val_pred = model(X_val_t).cpu().numpy().flatten()
        val_maes.append(float(np.abs(y_val_w[:, w] - val_pred).mean()))

        # cold 예측
        with torch.no_grad():
            X_cold_t = torch.tensor(X_cold, device=device)
            cold_pred = model(X_cold_t).cpu().numpy().flatten()
        y_cold_pred[:, w] = cold_pred

        if (w + 1) % 4 == 0 or w == n_weeks - 1:
            logger.info("week %2d/%d  best_ep=%3d  val_MAE=%.4f",
                        w + 1, n_weeks, best_ep, val_maes[-1])

    # ── 저장: 대표 주차 learning curve ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for w_plot, ax in zip([0, n_weeks // 2], axes):
        ax.plot(all_train_curves[w_plot], label="train")
        ax.plot(all_val_curves[w_plot],   label="val")
        ax.set_title(f"Week {w_plot+1} Learning Curve")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 평가 ─────────────────────────────────────────────────
    # cold MAE (일별)
    pred_rows = []
    for i, iid in enumerate(cold_ids):
        item_rows = cold_test[cold_test["item_id"] == iid].copy().sort_values("date")
        dates = item_rows["date"].tolist()
        week_periods = pd.to_datetime(dates).to_period("W")
        unique_weeks = sorted(set(week_periods))
        week_pred_map = {wp: float(y_cold_pred[i, j])
                         for j, wp in enumerate(unique_weeks) if j < n_weeks}
        n_days_per_week = item_rows.groupby(week_periods).size()
        for date, wp in zip(dates, week_periods):
            n_days = n_days_per_week.get(wp, 7)
            pred_rows.append({
                "item_id": iid,
                "date": date,
                "pred_sales": week_pred_map.get(wp, 0.0) / n_days,
            })

    pred_df = pd.DataFrame(pred_rows)
    merged  = cold_test.merge(pred_df, on=["item_id", "date"], how="left")
    merged["pred_sales"] = merged["pred_sales"].fillna(0.0)

    cold_mae  = float(np.abs(merged["sales"] - merged["pred_sales"]).mean())
    warm_val_mae = float(np.mean(val_maes))

    dir_acc = direction_accuracy_weekly(y_cold_weekly, y_cold_pred)

    # 카테고리별 cold MAE
    cat_map = (
        cold_test[["item_id", "cat_id"]].drop_duplicates("item_id")
        .set_index("item_id")["cat_id"].to_dict()
    )
    by_cat: dict[str, dict] = {}
    for cat, grp in merged.groupby("cat_id"):
        by_cat[cat] = {
            "mae":  float(np.abs(grp["sales"] - grp["pred_sales"]).mean()),
        }

    logger.info("=" * 60)
    logger.info("=== MLP Head 결과 ===")
    logger.info("warm val MAE (CV) = %.4f", warm_val_mae)
    logger.info("cold test MAE     = %.4f", cold_mae)
    logger.info("cold test DirAcc  = %.4f", dir_acc)
    logger.info("카테고리별:")
    for cat, m in by_cat.items():
        logger.info("  %s: MAE=%.4f", cat, m["mae"])
    logger.info("베이스라인(Ridge):  warm_val=61.24  cold=68.30  DirAcc=0.372")
    logger.info("=" * 60)

    # 결과 저장
    results = {
        "head": "MLP(5120→256→64→1)",
        "warm_val_mae": warm_val_mae,
        "cold_test_mae": cold_mae,
        "cold_test_dir_acc": dir_acc,
        "by_category": by_cat,
        "n_weeks": n_weeks,
        "train_items": len(tr_idx),
        "val_items": len(val_idx),
    }
    (out_dir / "evaluation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pred_df.to_csv(out_dir / "mlp_pred.csv", index=False)
    logger.info("결과 저장 완료: %s", out_dir)


if __name__ == "__main__":
    main()
