"""Track B 결과 심층 분석 스크립트.

1. 스케일 보정 (global mean ratio alpha)
2. Ridge train vs validation MAE (overfitting 진단)
3. 대안 회귀 모델 비교 (Lasso, RandomForest, PCA+Linear)
4. t-SNE 시각화 (hidden states 품질)
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
# 공통 유틸
# ──────────────────────────────────────────────────────────────

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """주간 단위 방향 정확도 (연속된 주 간 증감 일치율)."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    if y_true.shape[1] < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true, axis=1))
    pred_dir = np.sign(np.diff(y_pred, axis=1))
    mask = true_dir != 0
    if mask.sum() == 0:
        return float("nan")
    return float((true_dir[mask] == pred_dir[mask]).mean())


def aggregate_weekly(sales_df: pd.DataFrame, item_ids: list[str],
                     date_start: str, date_end: str) -> np.ndarray:
    """(n_items, n_weeks) 주간 판매량 행렬 반환."""
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


# ──────────────────────────────────────────────────────────────
# 0. 데이터 로드
# ──────────────────────────────────────────────────────────────

def load_data(config):
    exp_dir = ROOT / "experiments/exp005_track_b_embedding"
    cold_start_dir = ROOT / config.paths.cold_start_dir

    logger.info("데이터 로드 중...")
    cold_test  = pd.read_csv(cold_start_dir / "cold_test.csv",  parse_dates=["date"])
    warm_train = pd.read_csv(cold_start_dir / "warm_train.csv", parse_dates=["date"])
    warm_test  = pd.read_csv(cold_start_dir / "warm_test.csv",  parse_dates=["date"])
    pred_df    = pd.read_csv(exp_dir / "predictions/track_b_pred.csv", parse_dates=["date"])

    emb_cold, cold_ids = _load_npz(exp_dir / "embeddings/item_emb_cold.npz")
    emb_warm, warm_ids = _load_npz(exp_dir / "embeddings/item_emb_warm.npz")

    date_start = str(cold_test["date"].min().date())
    date_end   = str(cold_test["date"].max().date())

    cat_map = (
        pd.concat([cold_test, warm_train])[["item_id", "cat_id"]]
        .drop_duplicates("item_id")
        .set_index("item_id")["cat_id"]
        .to_dict()
    )

    logger.info("cold_ids=%d  warm_ids=%d  emb_cold=%s  emb_warm=%s",
                len(cold_ids), len(warm_ids), emb_cold.shape, emb_warm.shape)
    return dict(
        cold_test=cold_test, warm_train=warm_train, warm_test=warm_test,
        pred_df=pred_df, emb_cold=emb_cold, emb_warm=emb_warm,
        cold_ids=cold_ids, warm_ids=warm_ids,
        date_start=date_start, date_end=date_end, cat_map=cat_map,
    )


def _load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["item_ids"].tolist()


# ──────────────────────────────────────────────────────────────
# 1. 스케일 보정 (global mean ratio alpha)
# ──────────────────────────────────────────────────────────────

def analysis_scale_correction(d: dict) -> dict:
    logger.info("\n" + "="*60)
    logger.info("1. 스케일 보정 (global mean ratio alpha)")

    cold_test = d["cold_test"]
    pred_df   = d["pred_df"]

    # 실제 cold 판매량 평균
    true_mean = cold_test["sales"].mean()
    # raw 예측 평균
    pred_mean = pred_df["pred_sales"].mean()

    alpha = true_mean / pred_mean if pred_mean != 0 else 1.0
    logger.info("true mean=%.4f  pred mean=%.4f  alpha=%.4f", true_mean, pred_mean, alpha)

    # 보정 후 MAE
    merged = cold_test.merge(pred_df[["item_id", "date", "pred_sales"]], on=["item_id", "date"], how="left")
    raw_mae     = mean_absolute_error(merged["sales"].values, merged["pred_sales"].values)
    scaled_mae  = mean_absolute_error(merged["sales"].values, merged["pred_sales"].values * alpha)

    logger.info("Raw MAE    = %.4f", raw_mae)
    logger.info("Scaled MAE = %.4f  (×%.3f)", scaled_mae, alpha)

    # 카테고리별
    results = {"alpha": alpha, "raw_mae": raw_mae, "scaled_mae": scaled_mae, "by_category": {}}
    for cat, grp in merged.groupby("cat_id"):
        raw_c    = mean_absolute_error(grp["sales"].values, grp["pred_sales"].values)
        scaled_c = mean_absolute_error(grp["sales"].values, grp["pred_sales"].values * alpha)
        results["by_category"][cat] = {"raw_mae": raw_c, "scaled_mae": scaled_c}
        logger.info("  %s: raw=%.4f  scaled=%.4f", cat, raw_c, scaled_c)

    return results


# ──────────────────────────────────────────────────────────────
# 2. Ridge train vs validation MAE (overfitting 진단)
# ──────────────────────────────────────────────────────────────

def analysis_ridge_overfit(d: dict) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error as sk_mae

    logger.info("\n" + "="*60)
    logger.info("2. Ridge train vs validation MAE (overfitting 진단)")

    X = d["emb_warm"]
    y_weekly = aggregate_weekly(d["warm_test"], d["warm_ids"],
                                d["date_start"], d["date_end"])
    # (n_items, n_weeks) → flatten 주차별로 각각 학습 평가
    n_weeks = y_weekly.shape[1]
    train_maes, val_maes = [], []

    for w in range(n_weeks):
        y = y_weekly[:, w]
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        train_pred = model.predict(X)
        train_maes.append(sk_mae(y, train_pred))
        scores = cross_val_score(model, X, y, cv=5,
                                 scoring="neg_mean_absolute_error")
        val_maes.append(-scores.mean())

    train_mae_mean = float(np.mean(train_maes))
    val_mae_mean   = float(np.mean(val_maes))
    ratio = val_mae_mean / train_mae_mean if train_mae_mean > 0 else float("inf")

    logger.info("Train MAE (mean across weeks) = %.4f", train_mae_mean)
    logger.info("Val   MAE (5-fold CV, mean)   = %.4f", val_mae_mean)
    logger.info("Val/Train ratio = %.2f  (%s)",
                ratio,
                "Overfitting 의심" if ratio > 2.0 else
                "둘 다 나쁨 → hidden states 정보 부족" if val_mae_mean > 5.0
                else "정상 범위")

    result = {
        "train_mae": train_mae_mean,
        "val_mae_cv5": val_mae_mean,
        "ratio": ratio,
        "per_week": {"train": train_maes, "val": val_maes},
    }
    return result


# ──────────────────────────────────────────────────────────────
# 3. 대안 회귀 모델 비교
# ──────────────────────────────────────────────────────────────

def analysis_alt_regressors(d: dict) -> dict:
    from sklearn.linear_model import Ridge, Lasso, LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error as sk_mae

    logger.info("\n" + "="*60)
    logger.info("3. 대안 회귀 모델 비교")

    X_warm = d["emb_warm"]
    X_cold = d["emb_cold"]
    cold_ids = d["cold_ids"]

    y_warm_weekly = aggregate_weekly(d["warm_test"], d["warm_ids"],
                                     d["date_start"], d["date_end"])
    y_cold_weekly = aggregate_weekly(d["cold_test"], cold_ids,
                                     d["date_start"], d["date_end"])

    n_weeks = y_warm_weekly.shape[1]

    models = {
        "Ridge(alpha=1)":     Ridge(alpha=1.0),
        "Lasso(alpha=0.01)":  Lasso(alpha=0.01, max_iter=5000),
        "RandomForest":       RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "PCA50+Linear":       Pipeline([("pca", PCA(n_components=50)),
                                        ("lr",  LinearRegression())]),
    }

    results = {}
    for name, model in models.items():
        warm_val_maes, cold_pred_maes = [], []
        y_cold_pred_all = np.zeros_like(y_cold_weekly, dtype=float)

        for w in range(n_weeks):
            yw = y_warm_weekly[:, w]
            yc = y_cold_weekly[:, w]

            # warm validation MAE (5-fold CV)
            scores = cross_val_score(model, X_warm, yw, cv=5,
                                     scoring="neg_mean_absolute_error")
            warm_val_maes.append(-scores.mean())

            # cold test MAE (warm 전체로 학습 후 cold 예측)
            model.fit(X_warm, yw)
            y_cold_pred = model.predict(X_cold)
            y_cold_pred_all[:, w] = y_cold_pred
            cold_pred_maes.append(sk_mae(yc, y_cold_pred))

        warm_val = float(np.mean(warm_val_maes))
        cold_test_mae = float(np.mean(cold_pred_maes))
        results[name] = {"warm_val_mae": warm_val, "cold_test_mae": cold_test_mae}
        logger.info("%-22s  warm_val_MAE=%.4f  cold_test_MAE=%.4f",
                    name, warm_val, cold_test_mae)

    return results


# ──────────────────────────────────────────────────────────────
# 4. t-SNE 시각화
# ──────────────────────────────────────────────────────────────

def analysis_tsne(d: dict, out_dir: Path) -> None:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("\n" + "="*60)
    logger.info("4. t-SNE 시각화 (warm items, 카테고리별 클러스터)")

    X = d["emb_warm"]
    cats = [d["cat_map"].get(iid, "UNKNOWN") for iid in d["warm_ids"]]
    cat_labels = np.array(cats)

    # 5120차원 → PCA 50 → t-SNE 2D
    logger.info("PCA 50 차원 축소 중...")
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X)

    logger.info("t-SNE 2D 투영 중 (perplexity=30)...")
    X_2d = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca").fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"FOODS": "#e74c3c", "HOBBIES": "#3498db", "HOUSEHOLD": "#2ecc71"}
    for cat, color in colors.items():
        mask = cat_labels == cat
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=cat, alpha=0.7, s=40, edgecolors="none")

    ax.set_title("t-SNE of Warm Item Embeddings (Qwen2.5-32B 4-bit)\n"
                 "Colored by Category", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Category")
    ax.grid(alpha=0.3)

    out_path = out_dir / "tsne_warm_embeddings.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("t-SNE 저장: %s", out_path)


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main():
    config   = load_config()
    out_dir  = ROOT / "experiments/exp005_track_b_embedding"
    (out_dir / "analysis").mkdir(exist_ok=True)

    d = load_data(config)

    r1 = analysis_scale_correction(d)
    r2 = analysis_ridge_overfit(d)
    r3 = analysis_alt_regressors(d)
    analysis_tsne(d, out_dir / "analysis")

    # 결과 저장
    summary = {"scale_correction": r1, "ridge_overfit": r2, "alt_regressors": r3}
    out_path = out_dir / "analysis/analysis_results.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("\n전체 분석 완료. 결과: %s", out_path)

    # 최종 요약 출력
    logger.info("\n" + "="*60)
    logger.info("=== 분석 요약 ===")
    logger.info("[1] 스케일 보정 alpha=%.3f  raw MAE=%.4f → scaled MAE=%.4f",
                r1["alpha"], r1["raw_mae"], r1["scaled_mae"])
    logger.info("[2] Ridge  train=%.4f  val(CV5)=%.4f  ratio=%.2f",
                r2["train_mae"], r2["val_mae_cv5"], r2["ratio"])
    logger.info("[3] 모델별 cold MAE:")
    for name, v in r3.items():
        logger.info("    %-22s  warm_val=%.4f  cold=%.4f", name, v["warm_val_mae"], v["cold_test_mae"])
    logger.info("[4] t-SNE → experiments/exp005_track_b_embedding/analysis/tsne_warm_embeddings.png")


if __name__ == "__main__":
    main()
