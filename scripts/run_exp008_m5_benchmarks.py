"""exp008: M5 공식 벤치마크 (Naive, sNaive) 검증.

두 가지 비교를 수행한다:

[비교 1] M5 공식 validation 기간 (d_1886~d_1913) — submission 파일 활용
  - Naive.csv / sNaive.csv에서 CA_1 300 warm items 추출
  - sales_train_evaluation.csv의 실제값으로 WRMSSE 계산
  - 목적: 우리 WRMSSE 평가 코드가 올바른지 검증
  - 기대: 우리가 계산한 값이 M5 공식 전체 수치(Naive=1.752, sNaive=0.847)와
          같은 방향성을 보여야 함. CA_1 subset이므로 수치가 다를 수 있음.

[비교 2] exp007 동일 기간 (d_1771~d_1798, 2015-12-04 ~ 2015-12-31)
  - warm_train.csv 마지막 28일을 테스트로, 그 이전을 훈련으로
  - Naive/sNaive를 직접 구현해 WRMSSE 계산
  - 목적: LightGBM(WRMSSE=0.911)과 동일 기간 직접 비교
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import wrmsse, mae, rmse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 설정
# ============================================================
TEST_DAYS = 28
TARGET_STORE = "CA_1"
EXP_DIR = ROOT / "experiments" / "exp008_m5_benchmarks"
EXP_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT / "data" / "processed" / "cold_start"
M5_DIR = ROOT / "m5-forecasting-accuracy"
VERIF_DIR = ROOT / "docs" / "verification"

# M5 validation 기간: d_1886 = 2016-04-25 (calendar.csv로 확인 필요)
# warm_train.csv 기간: d_1 ~ d_1798 (2011-01-29 ~ 2015-12-31)
# exp007 test 기간: d_1771 ~ d_1798 = last 28 days of warm_train


# ============================================================
# 헬퍼: warm items 목록 로드
# ============================================================
def load_warm_item_ids() -> set[str]:
    """warm_train.csv에서 실제 300 warm item_id 목록을 반환."""
    # warm_train.csv는 300개 warm items만 포함하므로 직접 읽음
    # 파일이 크므로 item_id 컬럼만 읽어 unique 값 추출
    item_ids: set[str] = set()
    for chunk in pd.read_csv(DATA_DIR / "warm_train.csv",
                             usecols=["item_id", "store_id"],
                             chunksize=500_000):
        ca1 = chunk[chunk["store_id"] == TARGET_STORE]
        item_ids.update(ca1["item_id"].unique().tolist())
    logger.info("warm item_id 로드 완료: %d items", len(item_ids))
    return item_ids


# ============================================================
# 비교 1: M5 공식 validation 기간 (submission 파일 활용)
# ============================================================
def run_comparison_1(warm_item_ids: set[str]) -> list[dict]:
    """M5 공식 validation 기간(d_1886~d_1913)에서 WRMSSE 계산."""
    logger.info("\n=== 비교 1: M5 공식 validation 기간 (d_1886~d_1913) ===")

    # 1-a. 달력에서 d_1886~d_1913 날짜 확인
    cal = pd.read_csv(M5_DIR / "calendar.csv", parse_dates=["date"])
    val_cal = cal[(cal["d"] >= "d_1886") & (cal["d"] <= "d_1913")].sort_values("d")
    val_dates = val_cal["date"].dt.strftime("%Y-%m-%d").tolist()
    val_d_cols = [f"d_{i}" for i in range(1886, 1914)]
    logger.info("validation 기간: %s ~ %s", val_dates[0], val_dates[-1])

    # 1-b. CA_1 warm items의 실제값 추출 (sales_train_evaluation.csv)
    logger.info("sales_train_evaluation.csv 로드 중 (CA_1 필터)...")
    sales_cols = ["id", "item_id", "store_id", "cat_id"] + val_d_cols
    # 파일이 크므로 CA_1만 필터
    chunks = []
    for chunk in pd.read_csv(
        M5_DIR / "sales_train_evaluation.csv",
        usecols=lambda c: c in sales_cols or c in ["id", "item_id", "store_id", "cat_id"] + val_d_cols,
        chunksize=5000,
    ):
        ca1_chunk = chunk[
            (chunk["store_id"] == TARGET_STORE)
            & (chunk["item_id"].isin(warm_item_ids))
        ]
        if len(ca1_chunk) > 0:
            chunks.append(ca1_chunk)
    actual_wide = pd.concat(chunks, ignore_index=True)
    logger.info("CA_1 warm items: %d rows", len(actual_wide))

    # 1-c. wide → long 변환
    actual_long = actual_wide.melt(
        id_vars=["item_id", "store_id", "cat_id"],
        value_vars=val_d_cols,
        var_name="d",
        value_name="sales",
    )
    # d 컬럼을 날짜로 변환
    d_to_date = dict(zip(val_cal["d"].tolist(), val_cal["date"].tolist()))
    actual_long["date"] = actual_long["d"].map(d_to_date)
    actual_long = actual_long.drop(columns=["d"]).sort_values(["item_id", "date"]).reset_index(drop=True)

    # 1-d. 학습 데이터 (scale 계산용) — d_1858~d_1885 (validation 직전 28일)
    pre_d_cols = [f"d_{i}" for i in range(1858, 1886)]
    pre_chunks = []
    for chunk in pd.read_csv(
        M5_DIR / "sales_train_evaluation.csv",
        usecols=lambda c: c in ["item_id", "store_id", "cat_id"] + pre_d_cols,
        chunksize=5000,
    ):
        ca1_chunk = chunk[
            (chunk["store_id"] == TARGET_STORE)
            & (chunk["item_id"].isin(warm_item_ids))
        ]
        if len(ca1_chunk) > 0:
            pre_chunks.append(ca1_chunk)
    scale_wide = pd.concat(pre_chunks, ignore_index=True)
    pre_cal = cal[(cal["d"] >= "d_1858") & (cal["d"] <= "d_1885")].sort_values("d")
    pre_d_to_date = dict(zip(pre_cal["d"].tolist(), pre_cal["date"].tolist()))
    scale_long = scale_wide.melt(
        id_vars=["item_id", "store_id", "cat_id"],
        value_vars=pre_d_cols,
        var_name="d",
        value_name="sales",
    )
    scale_long["date"] = scale_long["d"].map(pre_d_to_date)
    scale_long = scale_long.drop(columns=["d"])

    # 1-e. M5 공식 WRMSSE 계산용 item-own scale (lag-28 MSE)
    # scale 기간: validation 직전 전체 학습 기간 (d_1 ~ d_1885)에서 lag-28 MSE 계산
    # 효율을 위해 직전 365일(d_1521~d_1885)만 사용
    logger.info("item-own scale 계산 중 (d_1858~d_1885 lag-28 사용)...")
    # pre_d_cols: d_1858~d_1885 (scale window = 28일)
    # 이미 scale_long에 있음 (lag-28 MSE 계산은 metrics.wrmsse 내부에서 수행)
    # → 우리 metrics.wrmsse(merged, scale_long, lag=28)가 이미 lag-28 MSE 사용
    # 추가로 M5 공식 방식(item-own lag-1 MSE, 전체 train)도 계산
    def m5_wrmsse_item_own(pred_df: pd.DataFrame, actual_df: pd.DataFrame,
                           full_train_wide: pd.DataFrame, d_cols_train: list[str]) -> float:
        """M5 공식 WRMSSE: item별 lag-1 MSE로 scale, test sales로 weight."""
        # scale_i: lag-1 차분의 MSE (전체 train 기간)
        item_scales: dict[str, float] = {}
        item_weights: dict[str, float] = {}
        for _, row in full_train_wide.iterrows():
            vals = row[d_cols_train].values.astype(float)
            diffs = vals[1:] - vals[:-1]
            scale = float(np.mean(diffs ** 2)) if len(diffs) > 0 else 1.0
            item_scales[row["item_id"]] = max(scale, 1e-8)
        # weight_i: test period actual sales
        for item_id, grp in actual_df.groupby("item_id"):
            item_weights[item_id] = float(grp["sales"].sum())
        # WRMSSE
        merged_pred = actual_df.merge(pred_df[["item_id", "date", "pred_sales"]],
                                      on=["item_id", "date"], how="left")
        merged_pred["pred_sales"] = merged_pred["pred_sales"].fillna(0.0)
        total_w = sum(item_weights.values())
        if total_w == 0:
            return float("nan")
        wrmsse_sum = 0.0
        for item_id, grp in merged_pred.groupby("item_id"):
            mse = float(np.mean((grp["sales"].values - grp["pred_sales"].values) ** 2))
            scale = item_scales.get(item_id, 1.0)
            rmsse = np.sqrt(mse / scale)
            w = item_weights.get(item_id, 0.0)
            wrmsse_sum += w * rmsse
        return wrmsse_sum / total_w

    # 비교 1용 full train (d_1~d_1885) 로드
    logger.info("item-own scale용 full train 로드 중 (CA_1 warm items)...")
    all_d_cols = [f"d_{i}" for i in range(1, 1886)]
    # 파일이 크므로 CA_1 warm items만 필터 (wide 포맷 유지)
    full_train_chunks = []
    for chunk in pd.read_csv(
        M5_DIR / "sales_train_evaluation.csv",
        usecols=lambda c: c in ["item_id", "store_id"] + all_d_cols,
        chunksize=3000,
    ):
        ca1_chunk = chunk[
            (chunk["store_id"] == TARGET_STORE)
            & (chunk["item_id"].isin(warm_item_ids))
        ]
        if len(ca1_chunk) > 0:
            full_train_chunks.append(ca1_chunk)
    full_train_wide = pd.concat(full_train_chunks, ignore_index=True)
    logger.info("full train wide: %d items", len(full_train_wide))

    # 1-e. submission 파일 → WRMSSE 계산
    results = []
    for csv_name, model_name in [("Naive.csv", "naive_official"), ("sNaive.csv", "snaive_official")]:
        sub = pd.read_csv(VERIF_DIR / csv_name)
        # _validation 행만, CA_1 warm items만
        sub_val = sub[sub["id"].str.endswith("_CA_1_validation")].copy()
        sub_val["item_id"] = sub_val["id"].str.replace("_CA_1_validation", "", regex=False)
        sub_val = sub_val[sub_val["item_id"].isin(warm_item_ids)]
        logger.info("[%s] CA_1 warm items: %d rows", model_name, len(sub_val))

        # wide → long (F1..F28 → date)
        f_cols = [f"F{i}" for i in range(1, 29)]
        sub_long = sub_val.melt(
            id_vars=["item_id"],
            value_vars=f_cols,
            var_name="f",
            value_name="pred_sales",
        )
        sub_long["f_idx"] = sub_long["f"].str[1:].astype(int) - 1
        sub_long["date"] = sub_long["f_idx"].apply(lambda i: val_dates[i] if i < len(val_dates) else None)
        sub_long["date"] = pd.to_datetime(sub_long["date"])
        sub_long["store_id"] = TARGET_STORE
        sub_long["pred_sales"] = sub_long["pred_sales"].clip(lower=0.0)

        # actual과 merge
        merged = actual_long.merge(
            sub_long[["item_id", "date", "pred_sales"]],
            on=["item_id", "date"],
            how="left",
        )
        merged["pred_sales"] = merged["pred_sales"].fillna(0.0)

        mae_val = mae(merged["sales"], merged["pred_sales"])
        rmse_val = rmse(merged["sales"], merged["pred_sales"])
        wrmsse_val = wrmsse(merged, scale_long, lag=28)
        wrmsse_official = m5_wrmsse_item_own(sub_long, actual_long, full_train_wide, all_d_cols)

        logger.info(
            "[%s] WRMSSE(cat-mean-lag28)=%.4f  WRMSSE(item-own-lag1)=%.4f  MAE=%.4f  RMSE=%.4f",
            model_name, wrmsse_val, wrmsse_official, mae_val, rmse_val,
        )
        results.append({
            "model": model_name,
            "period": "M5_validation (d_1886~d_1913)",
            "wrmsse_cat_mean_lag28": wrmsse_val,
            "wrmsse_item_own_lag1": wrmsse_official,
            "mae": mae_val,
            "rmse": rmse_val,
            "n_items": merged["item_id"].nunique(),
        })

    return results


# ============================================================
# 비교 2: exp007 동일 기간 (직접 구현)
# ============================================================
def load_exp007_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """warm_train.csv 로드 후 exp007과 동일하게 마지막 28일을 test로 분할."""
    cold_ids_raw = pd.read_csv(DATA_DIR / "cold_ids.csv")["id"].tolist()
    cold_ids = set("_".join(x.split("_")[:3]) for x in cold_ids_raw)

    logger.info("warm_train.csv 로드 중 (CA_1 warm items)...")
    chunks = []
    for chunk in pd.read_csv(DATA_DIR / "warm_train.csv", parse_dates=["date"], chunksize=500_000):
        filtered = chunk[
            (chunk["store_id"] == TARGET_STORE) & (~chunk["item_id"].isin(cold_ids))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)
    warm = pd.concat(chunks, ignore_index=True)

    cutoff = warm["date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = warm[warm["date"] < cutoff].copy()
    test = warm[warm["date"] >= cutoff].copy()
    logger.info(
        "exp007 분할: train ~ %s, test %s ~ %s",
        train["date"].max().date(),
        test["date"].min().date(),
        test["date"].max().date(),
    )
    return train, test


def predict_naive(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Naive: 학습 데이터 마지막 28일을 반복."""
    results = []
    for item_id, item_train in train.sort_values("date").groupby("item_id"):
        last_28 = item_train.tail(TEST_DAYS)["sales"].values
        item_test = test[test["item_id"] == item_id].sort_values("date").copy()
        n = len(item_test)
        item_test["pred_sales"] = np.tile(last_28, n // TEST_DAYS + 1)[:n]
        results.append(item_test)
    return pd.concat(results, ignore_index=True)


def predict_snaive(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """sNaive: 52주(364일) 전 같은 기간 값 사용."""
    results = []
    for item_id, item_train in train.sort_values("date").groupby("item_id"):
        date_to_sales = item_train.set_index("date")["sales"].to_dict()
        item_test = test[test["item_id"] == item_id].sort_values("date").copy()
        preds = []
        for d in item_test["date"]:
            for offset in [364, 365, 363, 357, 371]:
                past = d - pd.Timedelta(days=offset)
                if past in date_to_sales:
                    preds.append(float(date_to_sales[past]))
                    break
            else:
                preds.append(float(item_train.tail(28)["sales"].mean()))
        item_test["pred_sales"] = preds
        results.append(item_test)
    return pd.concat(results, ignore_index=True)


def evaluate(name: str, pred_df: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame) -> dict:
    merged = test.merge(
        pred_df[["item_id", "store_id", "date", "pred_sales"]],
        on=["item_id", "store_id", "date"],
        how="left",
    )
    merged["pred_sales"] = merged["pred_sales"].fillna(0.0)
    mae_val = mae(merged["sales"], merged["pred_sales"])
    rmse_val = rmse(merged["sales"], merged["pred_sales"])
    wrmsse_val = wrmsse(merged, train, lag=28)
    logger.info("[%s] WRMSSE=%.4f  MAE=%.4f  RMSE=%.4f", name, wrmsse_val, mae_val, rmse_val)
    return {
        "model": name,
        "period": "exp007 동일 기간 (d_1771~d_1798)",
        "wrmsse": wrmsse_val,
        "mae": mae_val,
        "rmse": rmse_val,
        "n_items": merged["item_id"].nunique(),
    }


def run_comparison_2() -> list[dict]:
    logger.info("\n=== 비교 2: exp007 동일 기간 (d_1771~d_1798) ===")
    train, test = load_exp007_data()
    results = []
    results.append(evaluate("naive_exp007", predict_naive(train, test), test, train))
    results.append(evaluate("snaive_exp007", predict_snaive(train, test), test, train))
    return results


# ============================================================
# main
# ============================================================
def main() -> None:
    logger.info("=== exp008: M5 벤치마크 검증 ===")

    warm_item_ids = load_warm_item_ids()

    all_results = []

    # 비교 1
    results1 = run_comparison_1(warm_item_ids)
    all_results.extend(results1)

    # 비교 2
    results2 = run_comparison_2()
    all_results.extend(results2)

    # 최종 비교 표
    logger.info("\n=== 최종 결과 비교 ===")
    logger.info("%-35s %-30s %8s %12s %8s %8s",
                "Model", "Period", "WRMSSE*", "WRMSSE(M5)", "MAE", "RMSE")
    logger.info("-" * 100)
    for r in all_results:
        wrmsse_main = r.get("wrmsse_cat_mean_lag28", r.get("wrmsse", float("nan")))
        wrmsse_m5 = r.get("wrmsse_item_own_lag1", float("nan"))
        logger.info("%-35s %-30s %8.4f %12.4f %8.4f %8.4f",
                    r["model"], r["period"], wrmsse_main, wrmsse_m5, r["mae"], r["rmse"])
    logger.info("* WRMSSE* = cat-mean-lag28 scale (our metric)")
    logger.info("%-35s %-30s %8.4f %8.4f %8s",
                "lightgbm (exp007)", "d_1771~d_1798", 0.9118, 1.0272, "2.1451")
    logger.info("")
    logger.info("참고 — M5 전체 (30,490 items, Scores and Ranks.xlsx):")
    logger.info("  Naive  avg WRMSSE = 1.752,  sNaive avg WRMSSE = 0.847")

    # 저장
    out_path = EXP_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
