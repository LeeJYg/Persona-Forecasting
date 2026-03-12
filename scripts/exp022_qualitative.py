"""exp022: 정성적 평가 (Head 예측 경향 + LLM Rationale + 시간축 분석).

분석 1: Head 예측 경향 시각화 (GPU 불필요)
분석 2: LLM 실제 답변 + Rationale (GPU 필요)
분석 3: 시간 축 분석 (GPU 불필요)

출력:
    docs/diagnosis/qualitative_analysis.md
    experiments/exp022_qualitative/figures/
"""
from __future__ import annotations

import json
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

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
SC_PATH  = ROOT / "experiments/exp020_error_analysis/item_scorecard.csv"
PERSONA_PATH = ROOT / "data/processed/personas/all_personas.json"
OUT_DIR  = ROOT / "experiments/exp022_qualitative"
FIG_DIR  = OUT_DIR / "figures"
REPORT   = ROOT / "docs/diagnosis/qualitative_analysis.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "docs/diagnosis").mkdir(parents=True, exist_ok=True)


# ─── 모델 ─────────────────────────────────────────────────────────────────────

class AttnBottleneckG1(nn.Module):
    """exp016 G1: attention → bottleneck head, n_weeks=16, dropout=0.1."""

    def __init__(self, hidden: int = 5120, bottleneck: int = 64,
                 n_weeks: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, bottleneck), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, P, H) → (N, n_weeks)."""
        scores  = self.attn(x).squeeze(-1)           # (N, P)
        attn_w  = torch.softmax(scores, dim=-1)       # (N, P)
        ctx     = (x * attn_w.unsqueeze(-1)).sum(1)  # (N, H)
        return self.head(ctx)

    def forward_with_attn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pred, attn_weights). x: (N, P, H)."""
        scores  = self.attn(x).squeeze(-1)           # (N, P)
        attn_w  = torch.softmax(scores, dim=-1)       # (N, P)
        ctx     = (x * attn_w.unsqueeze(-1)).sum(1)  # (N, H)
        return self.head(ctx), attn_w


def train_g1(X: np.ndarray, y: np.ndarray, seed: int = 42) -> AttnBottleneckG1:
    """G1 설정: MAE loss, lr=1e-3, wd=1e-4, clip_grad=1.0, 500 epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttnBottleneckG1(hidden=X.shape[2], n_weeks=y.shape[1])
    opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt    = torch.tensor(X, dtype=torch.float32)
    yt    = torch.tensor(y, dtype=torch.float32)
    model.train()
    for ep in range(500):
        opt.zero_grad()
        loss = F.l1_loss(model(Xt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (ep + 1) % 100 == 0:
            logger.info("  G1 epoch %d/500 loss=%.4f", ep + 1, loss.item())
    model.eval()
    return model


def predict_np(model: AttnBottleneckG1, X: np.ndarray) -> np.ndarray:
    """(N, P, H) → (N, n_weeks), clipped ≥ 0."""
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return np.clip(p, 0, None)


def get_attn_weights(model: AttnBottleneckG1, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred (N,T), attn_weights (N,P)), clipped."""
    with torch.no_grad():
        pred, attn_w = model.forward_with_attn(torch.tensor(X, dtype=torch.float32))
    return np.clip(pred.numpy(), 0, None), attn_w.numpy()


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    gcols = [c for c in ["item_id", "store_id", "cat_id", "dept_id",
                          "iso_year", "iso_week"] if c in df.columns]
    return (df.groupby(gcols)
            .agg(sales=("sales", "sum"), date=("week_start", "first"))
            .reset_index())


def load_data() -> dict[str, Any]:
    logger.info("데이터 로드 중...")
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", map_location="cpu",
                          weights_only=False).numpy()
    cold_raw = torch.load(EMB_DIR / "cold_raw.pt", map_location="cpu",
                          weights_only=False).numpy()
    warm_residual = warm_raw - warm_raw.mean(axis=1, keepdims=True)
    cold_residual = cold_raw - cold_raw.mean(axis=1, keepdims=True)

    item_meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_ids  = item_meta[item_meta["is_cold"] == True]["item_id"].tolist()
    warm_ids  = item_meta[item_meta["is_cold"] == False]["item_id"].tolist()

    cold_test_raw  = pd.read_csv(CS_DIR / "cold_test.csv",  parse_dates=["date"])
    warm_test_raw  = pd.read_csv(CS_DIR / "warm_test.csv",  parse_dates=["date"])

    cold_weekly = _to_weekly(cold_test_raw)
    warm_test_w = _to_weekly(warm_test_raw)

    # 완전한 주 (16주) 계산
    day_cnt = (
        cold_test_raw
        .assign(iso_year=cold_test_raw["date"].dt.isocalendar().year.astype(int),
                iso_week=cold_test_raw["date"].dt.isocalendar().week.astype(int))
        .groupby(["iso_year", "iso_week"])["date"].nunique()
    )
    complete_set = set(zip(day_cnt[day_cnt == 7].index.get_level_values(0),
                           day_cnt[day_cnt == 7].index.get_level_values(1)))
    cold_weekly_16 = cold_weekly[cold_weekly.apply(
        lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)].copy()
    wl16 = sorted(complete_set)

    def build_y(ids: list[str], weekly: pd.DataFrame, week_list: list) -> np.ndarray:
        wi = {wk: i for i, wk in enumerate(week_list)}
        ii = {iid: i for i, iid in enumerate(ids)}
        y  = np.zeros((len(ids), len(week_list)), dtype=np.float32)
        for _, r in weekly.iterrows():
            wk = (r["iso_year"], r["iso_week"])
            if wk in wi and r["item_id"] in ii:
                y[ii[r["item_id"]], wi[wk]] = r["sales"]
        return y

    y_warm = build_y(warm_ids, warm_test_w, wl16)
    y_cold = build_y(cold_ids, cold_weekly_16, wl16)

    # lgbm / knn 예측값 로드
    lgbm_df = pd.read_csv(
        COMP_DIR / "lightgbm_proxy_lags/predictions/lightgbm_proxy_lags.csv",
        parse_dates=["date"])
    lgbm_df = lgbm_df[lgbm_df.apply(
        lambda r: (r["iso_year"], r["iso_week"]) in complete_set, axis=1)]

    # lgbm 예측 행렬: (100, 16)
    lgbm_pred = np.zeros((len(cold_ids), len(wl16)), dtype=np.float32)
    wi = {wk: i for i, wk in enumerate(wl16)}
    ii = {iid: i for i, iid in enumerate(cold_ids)}
    for _, r in lgbm_df.iterrows():
        if r["item_id"] in ii:
            wk = (r["iso_year"], r["iso_week"])
            if wk in wi:
                lgbm_pred[ii[r["item_id"]], wi[wk]] = r["pred_sales"]

    # item scorecard
    scorecard = pd.read_csv(SC_PATH)

    logger.info("  cold_raw=%s  y_cold=%s  wl16=%d",
                cold_raw.shape, y_cold.shape, len(wl16))

    return dict(
        warm_raw=warm_raw, cold_raw=cold_raw,
        warm_residual=warm_residual, cold_residual=cold_residual,
        cold_ids=cold_ids, warm_ids=warm_ids,
        y_warm=y_warm, y_cold=y_cold,
        cold_weekly_16=cold_weekly_16,
        week_list=wl16,
        lgbm_pred=lgbm_pred,
        scorecard=scorecard,
        item_meta=item_meta,
    )


# ─── 분석 1: Head 예측 경향 시각화 ───────────────────────────────────────────

def analysis1(data: dict, g1_pred: np.ndarray) -> dict[str, Any]:
    """10개 아이템 꺾은선 그래프 + CV 비율 분석."""
    logger.info("=== 분석 1: Head 예측 경향 시각화 ===")
    sc  = data["scorecard"].copy()
    y   = data["y_cold"]          # (100, 16)
    lgb = data["lgbm_pred"]       # (100, 16)
    ids = data["cold_ids"]        # list[str]
    ii  = {iid: i for i, iid in enumerate(ids)}

    # 아이템 선정
    elective   = sc[sc["actual_nonzero_week_ratio"] < 0.5].copy()
    essential  = sc[sc["actual_nonzero_week_ratio"] >= 0.9].copy()
    elective_good = elective.nsmallest(3, "trackb_WRMSSE")["item_id"].tolist()
    elective_bad  = elective.nlargest(2, "trackb_WRMSSE")["item_id"].tolist()
    essential_win = (essential[essential["trackb_wins_lgbm_WRMSSE"] == True]
                     .nsmallest(3, "trackb_WRMSSE")["item_id"].tolist())
    essential_lose = (essential[essential["trackb_wins_lgbm_WRMSSE"] == False]
                      .nlargest(2, "trackb_WRMSSE")["item_id"].tolist())

    # 10개가 안 되는 경우 보완
    if len(essential_win) < 3:
        essential_win = essential.nsmallest(3, "trackb_WRMSSE")["item_id"].tolist()
    if len(essential_lose) < 2:
        essential_lose = essential.nlargest(2, "trackb_WRMSSE")["item_id"].tolist()

    items_10 = (
        [(iid, "선택품_좋음") for iid in elective_good] +
        [(iid, "선택품_나쁨") for iid in elective_bad] +
        [(iid, "필수품_TB승") for iid in essential_win] +
        [(iid, "필수품_TB패") for iid in essential_lose]
    )

    logger.info("  선택품 good=%s bad=%s  필수품 win=%s lose=%s",
                elective_good, elective_bad, essential_win, essential_lose)

    # 1-1: 2×5 subplot 꺾은선 그래프
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes_flat = axes.flatten()
    cv_records = []

    for ax_i, (iid, grp) in enumerate(items_10):
        idx = ii.get(iid, -1)
        if idx < 0:
            axes_flat[ax_i].set_visible(False)
            continue
        act = y[idx]        # (16,)
        g1  = g1_pred[idx]  # (16,)
        lgb_i = lgb[idx]    # (16,)
        wks = range(1, 17)

        ax = axes_flat[ax_i]
        ax.plot(wks, act,   "k-",  linewidth=2, label="actual")
        ax.plot(wks, g1,    "r--", linewidth=1.5, label="G1 pred")
        ax.plot(wks, lgb_i, "b--", linewidth=1.5, label="LightGBM")
        ax.set_xlabel("Week")
        ax.set_ylabel("Weekly Sales")

        sc_row = sc[sc["item_id"] == iid].iloc[0]
        dept = sc_row["dept_id"]
        amean = sc_row["actual_weekly_mean"]
        nzr   = sc_row["actual_nonzero_week_ratio"]
        ax.set_title(f"{iid}\n{dept} | mean={amean:.1f} | nzr={nzr:.2f}\n[{grp}]",
                     fontsize=8)
        if ax_i == 0:
            ax.legend(fontsize=7)

        # CV 계산
        act_mean = act.mean(); act_std = act.std()
        g1_mean  = g1.mean();  g1_std  = g1.std()
        act_cv = act_std / act_mean if act_mean > 1e-9 else np.nan
        g1_cv  = g1_std  / g1_mean  if g1_mean  > 1e-9 else np.nan
        ratio  = g1_cv / act_cv if (act_cv and not np.isnan(act_cv) and act_cv > 1e-9) else np.nan
        cv_records.append(dict(item_id=iid, group=grp,
                               act_mean=act_mean, act_cv=act_cv,
                               g1_cv=g1_cv, cv_ratio=ratio))

    fig.suptitle("분석 1-1: actual vs G1 pred vs LightGBM (10 items)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "a1_linechart_10items.png", dpi=150)
    plt.close(fig)
    logger.info("  a1_linechart_10items.png 저장")

    # 1-2: 전체 100개 G1_CV / actual_CV histogram
    all_cv = []
    for i, iid in enumerate(ids):
        act = y[i]; g1 = g1_pred[i]
        am = act.mean(); gm = g1.mean()
        ac = act.std() / am if am > 1e-9 else np.nan
        gc = g1.std()  / gm if gm > 1e-9 else np.nan
        ratio = gc / ac if (ac and not np.isnan(ac) and ac > 1e-9) else np.nan
        all_cv.append(ratio)

    all_cv_arr = np.array([v for v in all_cv if v is not None and not np.isnan(v)])
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(all_cv_arr, bins=30, edgecolor="k")
    ax2.axvline(1.0, color="r", linestyle="--", label="ratio=1 (perfect)")
    ax2.set_xlabel("G1_CV / actual_CV")
    ax2.set_ylabel("Count")
    ax2.set_title(f"분석 1-2: G1 CV비율 분포 (100 cold items)\n"
                  f"mean={all_cv_arr.mean():.3f}  median={np.median(all_cv_arr):.3f}  "
                  f"<0.1: {(all_cv_arr < 0.1).sum()}")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "a1_cv_ratio_histogram.png", dpi=150)
    plt.close(fig2)
    logger.info("  a1_cv_ratio_histogram.png 저장")

    return dict(
        items_10=items_10,
        cv_table=cv_records,
        all_cv=all_cv_arr,
    )


# ─── 분석 3: 시간 축 분석 ─────────────────────────────────────────────────────

def analysis3(data: dict, g1_pred: np.ndarray) -> dict[str, Any]:
    """주별 평균 예측 vs 실제 그래프 + 4주 head 설계."""
    logger.info("=== 분석 3: 시간 축 분석 ===")
    y   = data["y_cold"]          # (100, 16)
    ids = data["cold_ids"]

    # 3-1: 주별 평균
    actual_weekly_mean = y.mean(axis=0)           # (16,)
    g1_weekly_mean     = g1_pred.mean(axis=0)     # (16,)

    fig, ax = plt.subplots(figsize=(10, 5))
    wks = range(1, 17)
    ax.plot(wks, actual_weekly_mean, "k-o", linewidth=2, label="actual (100-item mean)")
    ax.plot(wks, g1_weekly_mean,     "r--s", linewidth=2, label="G1 pred (100-item mean)")
    ax.set_xlabel("Week (ISO)")
    ax.set_ylabel("Weekly Sales (mean over 100 items)")
    ax.set_title("분석 3-1: 주별 평균 예측 vs 실제 (전체 cold 100개)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "a3_weekly_mean.png", dpi=150)
    plt.close(fig)
    logger.info("  a3_weekly_mean.png 저장")

    # 주별 G1 예측 분산 (어느 주에 분산이 큰지)
    g1_weekly_std = g1_pred.std(axis=0)
    actual_weekly_std = y.std(axis=0)

    # G1 flat 정도: std of g1_weekly_mean / mean of g1_weekly_mean
    g1_temporal_cv = g1_weekly_mean.std() / g1_weekly_mean.mean() if g1_weekly_mean.mean() > 1e-9 else np.nan
    act_temporal_cv = actual_weekly_mean.std() / actual_weekly_mean.mean() if actual_weekly_mean.mean() > 1e-9 else np.nan

    logger.info("  G1 temporal CV=%.4f  actual temporal CV=%.4f",
                g1_temporal_cv, act_temporal_cv)

    return dict(
        actual_weekly_mean=actual_weekly_mean,
        g1_weekly_mean=g1_weekly_mean,
        g1_weekly_std=g1_weekly_std,
        actual_weekly_std=actual_weekly_std,
        g1_temporal_cv=g1_temporal_cv,
        act_temporal_cv=act_temporal_cv,
    )


# ─── 분석 2: LLM 실제 답변 + Rationale ───────────────────────────────────────

def _build_persona_text(p_profile: dict) -> str:
    """exp014/016과 동일한 구조화 persona text."""
    cat_pref = ", ".join(
        f"{c}={v:.2f}" for c, v in
        sorted(p_profile["category_preference"].items(), key=lambda x: -x[1])
    )
    return (
        f"Customer profile:\n"
        f"- Weekly budget: ${p_profile['weekly_budget']:.2f}\n"
        f"- SNAP eligible: {str(p_profile['snap_eligible']).lower()}\n"
        f"- Economic status: {p_profile['economic_status']}\n"
        f"- Shopping motivation: {p_profile['shopping_motivation']}\n"
        f"- Category preference: {cat_pref}\n"
        f"- Price sensitivity: {p_profile['price_sensitivity']}\n"
        f"- Visit frequency: {p_profile['visit_frequency']}\n"
        f"- Preferred departments: {', '.join(p_profile['preferred_departments'])}\n"
        f"- Decision style: {p_profile['decision_style']}\n"
        f"- Brand loyalty: {p_profile['brand_loyalty']}\n"
        f"- Promotion sensitivity: {p_profile['promotion_sensitivity']}"
    )


def _build_rationale_prompt(persona_text: str, dept: str, cat: str,
                             avg_price: float | None) -> str:
    """V3 + rationale 요청 프롬프트."""
    pr = f"${avg_price:.2f}" if avg_price and not np.isnan(avg_price) else "N/A"
    return (
        "<|im_start|>system\nYou are an expert consumer behavior analyst. "
        "Your task is to evaluate the match between a customer and a product."
        "<|im_end|>\n"
        f"<|im_start|>user\n{persona_text}\n\n"
        f"Item Profile:\n- Department: {dept}\n- Category: {cat}\n"
        f"- Average price: {pr}\n\n"
        "Task: Based on the customer profile, assess the probability of this customer "
        "purchasing this item.\n"
        "1. First, explain your reasoning in 2-3 sentences.\n"
        "2. Then, give your final answer as one of: Highly Likely, Likely, Neutral, "
        "Unlikely, Highly Unlikely."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def analysis2(data: dict, g1_model: AttnBottleneckG1) -> dict[str, Any]:
    """LLM rationale 생성 (GPU 필요)."""
    logger.info("=== 분석 2: LLM 답변 + Rationale ===")
    sc  = data["scorecard"].copy()
    ids = data["cold_ids"]
    ii  = {iid: i for i, iid in enumerate(ids)}

    # 2-1: 선택품 WRMSSE 상위(좋음) 3 + 하위(나쁨) 3
    elective = sc[sc["actual_nonzero_week_ratio"] < 0.5].copy()
    top3_ids = elective.nsmallest(3, "trackb_WRMSSE")["item_id"].tolist()
    bot3_ids = elective.nlargest(3, "trackb_WRMSSE")["item_id"].tolist()
    items_6  = [(iid, "선택품_좋음") for iid in top3_ids] + \
               [(iid, "선택품_나쁨") for iid in bot3_ids]
    logger.info("  분석 2 대상: %s", [iid for iid, _ in items_6])

    # 2-2: attention weight 추출
    cold_res = data["cold_residual"]  # (100, 50, 5120)
    item_attn: dict[str, np.ndarray] = {}
    for iid, _ in items_6:
        idx = ii.get(iid, -1)
        if idx < 0:
            continue
        x = torch.tensor(cold_res[idx:idx+1], dtype=torch.float32)
        with torch.no_grad():
            _, aw = g1_model.forward_with_attn(x)
        item_attn[iid] = aw[0].numpy()  # (50,)

    # 페르소나 로드
    with open(PERSONA_PATH) as f:
        all_personas: list[dict] = json.load(f)
    # persona_id → profile 매핑
    persona_map = {p["persona_id"]: p["profile"] for p in all_personas}
    persona_ids = [p["persona_id"] for p in all_personas]  # 순서 유지

    # 2-3: LLM 로드
    logger.info("  LLM 로드 중 (Qwen2.5-32B-Instruct 4-bit)...")
    try:
        import os
        os.environ.setdefault("HF_HOME", "/mnt/sdd1/jylee/huggingface_cache")
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_id = "Qwen/Qwen2.5-32B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        llm_model.eval()
        gpu_available = True
        logger.info("  LLM 로드 완료")
    except Exception as e:
        logger.warning("  LLM 로드 실패: %s — 분석 2 스킵", e)
        gpu_available = False

    if not gpu_available:
        return dict(items_6=items_6, item_attn=item_attn,
                    llm_results=[], gpu_available=False)

    # 2-4: 18개 (6 items × top-3 personas) LLM 답변 생성
    llm_results: list[dict] = []
    for iid, grp in items_6:
        idx = ii.get(iid, -1)
        if idx < 0 or iid not in item_attn:
            continue
        attn_w = item_attn[iid]  # (50,)
        top3_idx = np.argsort(attn_w)[::-1][:3]

        sc_row = sc[sc["item_id"] == iid].iloc[0]
        dept = sc_row["dept_id"]
        cat  = sc_row["cat_id"]
        avg_price = sc_row.get("avg_price", None)
        if pd.isna(avg_price):
            avg_price = None

        logger.info("  아이템 %s [%s] top3_personas=%s", iid, grp,
                    [persona_ids[k] for k in top3_idx])

        for rank, p_idx in enumerate(top3_idx):
            if p_idx >= len(persona_ids):
                continue
            p_id = persona_ids[p_idx]
            p_profile = persona_map.get(p_id, {})
            aw_score = float(attn_w[p_idx])

            persona_text = _build_persona_text(p_profile)
            prompt = _build_rationale_prompt(persona_text, dept, cat, avg_price)

            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
                with torch.no_grad():
                    out = llm_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                # 생성된 부분만 디코딩
                gen_ids = out[0][inputs["input_ids"].shape[-1]:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            except Exception as e:
                response = f"[ERROR: {e}]"

            llm_results.append(dict(
                item_id=iid, group=grp,
                dept_id=dept, cat_id=cat, avg_price=avg_price,
                persona_id=p_id, attn_weight=aw_score,
                rank=rank + 1,
                weekly_budget=p_profile.get("weekly_budget"),
                economic_status=p_profile.get("economic_status"),
                response=response,
            ))
            logger.info("    [%s × %s rank=%d] 생성 완료 (len=%d)",
                        iid, p_id, rank + 1, len(response))

    return dict(items_6=items_6, item_attn=item_attn,
                llm_results=llm_results, gpu_available=True)


# ─── 보고서 작성 ──────────────────────────────────────────────────────────────

def write_report(a1: dict, a2: dict, a3: dict, data: dict) -> None:
    sc = data["scorecard"]
    ids = data["cold_ids"]

    lines: list[str] = [
        "# 정성적 평가 보고서 (Exp022)",
        "",
        f"**작성일:** 2026-03-11",
        "**목적:** 수치가 아닌 개별 케이스의 행동을 직접 확인. 판정 없음.",
        "**Track B 기준선:** exp016 G1 (residual, 16주, MAE loss, 500ep, bn=64, seed=42)",
        "",
        "---",
        "",
        "## 분석 1: Head 예측 경향 시각화",
        "",
        "### 1-1: 10개 아이템 꺾은선 그래프",
        "",
        "**대상 아이템 선정 기준:**",
        "- 선택품(nonzero_week_ratio < 0.5): TB WRMSSE 좋은 3개, 나쁜 2개",
        "- 필수품(nonzero_week_ratio ≥ 0.9): TB가 LightGBM을 이긴 3개, 진 2개",
        "",
        "| # | item_id | group | actual_mean | nonzero_ratio | TB_WRMSSE | LGBM_WRMSSE |",
        "|---|---------|-------|------------|---------------|-----------|-------------|",
    ]

    for iid, grp in a1["items_10"]:
        row = sc[sc["item_id"] == iid]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"| | {iid} | {grp} | {r['actual_weekly_mean']:.2f} | "
            f"{r['actual_nonzero_week_ratio']:.2f} | "
            f"{r['trackb_WRMSSE']:.4f} | {r['lgbm_WRMSSE']:.4f} |"
        )

    lines += [
        "",
        f"시각화: `experiments/exp022_qualitative/figures/a1_linechart_10items.png`",
        "",
        "### 1-2: G1 예측의 주간 변동 분석 (CV 비율)",
        "",
        "**CV 정의:** std / mean. G1_CV / actual_CV → 1이면 변동 패턴 포착, 0이면 flat.",
        "",
        "| item_id | group | actual_mean | actual_CV | G1_CV | CV_ratio |",
        "|---------|-------|------------|-----------|-------|----------|",
    ]

    for r in a1["cv_table"]:
        act_cv = f"{r['act_cv']:.4f}" if r["act_cv"] is not None and not np.isnan(r["act_cv"]) else "N/A"
        g1_cv  = f"{r['g1_cv']:.4f}"  if r["g1_cv"]  is not None and not np.isnan(r["g1_cv"])  else "N/A"
        ratio  = f"{r['cv_ratio']:.4f}" if r["cv_ratio"] is not None and not np.isnan(r["cv_ratio"]) else "N/A"
        lines.append(
            f"| {r['item_id']} | {r['group']} | {r['act_mean']:.2f} | "
            f"{act_cv} | {g1_cv} | {ratio} |"
        )

    cv = a1["all_cv"]
    lines += [
        "",
        f"**전체 100개 G1_CV / actual_CV 분포:**",
        f"- 유효 샘플: {len(cv)}개",
        f"- mean={cv.mean():.4f}  median={np.median(cv):.4f}  std={cv.std():.4f}",
        f"- <0.1 (거의 flat): {(cv < 0.1).sum()}개",
        f"- 0.1~0.5: {((cv >= 0.1) & (cv < 0.5)).sum()}개",
        f"- ≥0.5: {(cv >= 0.5).sum()}개",
        "",
        f"시각화: `experiments/exp022_qualitative/figures/a1_cv_ratio_histogram.png`",
        "",
        "---",
        "",
        "## 분석 2: LLM 실제 답변 + Rationale",
        "",
    ]

    if not a2.get("gpu_available"):
        lines += [
            "**[주의]** LLM 로드 실패 — GPU 없이 실행되어 분석 2 생략.",
            "",
        ]
    else:
        lines += [
            "**대상:** 선택품(nonzero<0.5) TB WRMSSE 상위 3개 + 하위 3개 = 총 6개 아이템",
            "**방법:** 각 아이템의 attention weight 상위 3개 페르소나 × 6 아이템 = 18 조합",
            "**프롬프트:** V3 + rationale 요청 (2-3문장 reasoning → 최종 답변)",
            "**max_new_tokens:** 200",
            "",
            "### 2-2: 아이템별 top-3 페르소나 attention weight",
            "",
            "| item_id | group | rank | persona_id | attn_weight | budget | economic_status |",
            "|---------|-------|------|------------|-------------|--------|-----------------|",
        ]

        for r in a2["llm_results"]:
            bgt = f"${r['weekly_budget']:.0f}" if r.get("weekly_budget") else "N/A"
            lines.append(
                f"| {r['item_id']} | {r['group']} | {r['rank']} | "
                f"{r['persona_id']} | {r['attn_weight']:.4f} | "
                f"{bgt} | {r.get('economic_status', 'N/A')} |"
            )

        lines += ["", "### 2-3~2-4: LLM 생성 답변 전문", ""]

        cur_item = None
        for r in a2["llm_results"]:
            if r["item_id"] != cur_item:
                cur_item = r["item_id"]
                sc_row = sc[sc["item_id"] == cur_item]
                am = sc_row["actual_weekly_mean"].values[0] if not sc_row.empty else "N/A"
                nzr = sc_row["actual_nonzero_week_ratio"].values[0] if not sc_row.empty else "N/A"
                tw  = sc_row["trackb_WRMSSE"].values[0] if not sc_row.empty else "N/A"
                lines += [
                    "",
                    f"=== ITEM: {cur_item} [{r['group']}] ===",
                    f"dept={r['dept_id']} | cat={r['cat_id']} | "
                    f"avg_price=${r['avg_price']:.2f}" if r.get("avg_price") else
                    f"dept={r['dept_id']} | cat={r['cat_id']}",
                    f"actual_mean={am:.2f} | nonzero_ratio={nzr:.2f} | TB_WRMSSE={tw:.4f}" if isinstance(am, float) else "",
                    "",
                ]
            bgt = f"${r['weekly_budget']:.0f}" if r.get("weekly_budget") else "N/A"
            lines += [
                f"--- Persona {r['persona_id']} "
                f"(attention weight: {r['attn_weight']:.4f}, "
                f"budget: {bgt}, {r.get('economic_status', '')}) ---",
                "LLM 답변:",
                f'"{r["response"]}"',
                "",
            ]

    lines += [
        "---",
        "",
        "## 분석 3: 시간 축 분석",
        "",
        "### 3-1: 주별 평균 예측 vs 실제 (전체 100개 cold 아이템)",
        "",
        "| Week | actual_mean | G1_pred_mean | G1_pred_std | actual_std |",
        "|------|------------|--------------|-------------|------------|",
    ]

    for wk in range(16):
        am   = a3["actual_weekly_mean"][wk]
        gm   = a3["g1_weekly_mean"][wk]
        gs   = a3["g1_weekly_std"][wk]
        as_  = a3["actual_weekly_std"][wk]
        lines.append(f"| {wk+1} | {am:.2f} | {gm:.2f} | {gs:.2f} | {as_:.2f} |")

    g_tcv = a3["g1_temporal_cv"]
    a_tcv = a3["act_temporal_cv"]
    lines += [
        "",
        f"**G1 주별 평균의 temporal CV (std/mean):** {g_tcv:.4f}",
        f"**actual 주별 평균의 temporal CV:** {a_tcv:.4f}",
        "",
        "- G1 temporal CV가 0에 가까울수록 head가 시간 패턴 없이 flat 예측",
        "- actual temporal CV 대비 G1 temporal CV 비율: "
        f"{g_tcv/a_tcv:.3f}" if a_tcv > 1e-9 else "N/A",
        "",
        f"시각화: `experiments/exp022_qualitative/figures/a3_weekly_mean.png`",
        "",
        "### 3-2: 4주 단위 head 실험 설계 (실행하지 않음 — 설계 검토만)",
        "",
        "**설계 개요:**",
        "- 현재: 300 warm × 16주 출력 = 샘플 300개, 파라미터 ~330K",
        "- 제안: 300 warm × 4구간(각 4주 예측) = 샘플 1,200개, 파라미터 ~82K",
        "",
        "**파라미터 수 계산:**",
        "- 현재 head: Linear(5120→64) + Linear(64→16) = 5120×64+64 + 64×16+16 = 329,872",
        "- attention: Linear(5120→1) = 5,120",
        "- 합계 ≈ 335K",
        "",
        "- 4주 head: Linear(5120→64) + Linear(64→4) = 5120×64+64 + 64×4+4 = 328,196",
        "  attention 동일 → 합계 ≈ 333K (거의 동일)",
        "  ※ 파라미터 자체는 크게 줄지 않음 (출력 차원만 16→4로 감소)",
        "",
        "**샘플 증가 효과:**",
        "- 현재: N_train=300 (warm items)",
        "- 4주 분할: 300 × 4 = 1,200 (실질적 학습 샘플 4배 증가)",
        "- 단, 4구간은 서로 상관이 있으므로 독립 샘플 가정 불성립",
        "",
        "**장점:**",
        "- 학습 샘플 수 증가 → 정규화 효과",
        "- head 과적합 위험 감소",
        "- 짧은 예측 구간 → 분포 폭 감소",
        "",
        "**단점:**",
        "- 구간 경계에서 DirAcc 측정 불가 (구간 내 방향성만 측정)",
        "- 구간별 독립 예측 → 구간 간 연속성 미보장",
        "- 현재 평가 기준(16주 전체 WRMSSE)과 불일치",
        "- 데이터 증강이 실질적 정보 추가 없이 과신 유발 가능",
        "",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("보고서 저장: %s", REPORT)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Exp022 시작 ===")

    # 데이터 로드
    data = load_data()

    # G1 학습
    logger.info("G1 모델 학습 중 (warm 300개, 500 epoch)...")
    warm_res = data["warm_residual"]  # (300, 50, 5120)
    y_warm   = data["y_warm"]          # (300, 16)
    g1_model = train_g1(warm_res, y_warm, seed=42)

    # cold G1 예측
    cold_res = data["cold_residual"]  # (100, 50, 5120)
    g1_pred  = predict_np(g1_model, cold_res)  # (100, 16)
    logger.info("  G1 cold 예측 완료: mean=%.4f MAE=%.4f",
                g1_pred.mean(), np.abs(g1_pred - data["y_cold"]).mean())

    # 분석 실행
    a1 = analysis1(data, g1_pred)
    a3 = analysis3(data, g1_pred)
    a2 = analysis2(data, g1_model)

    # 보고서 작성
    logger.info("보고서 작성 중...")
    write_report(a1, a2, a3, data)

    print("\n" + "=" * 70)
    print("[Exp022 요약]")
    print(f"분석 1 — 10개 아이템 시각화 완료")
    print(f"         CV ratio 전체 mean={a1['all_cv'].mean():.4f}  "
          f"<0.1(flat): {(a1['all_cv'] < 0.1).sum()}/100")
    print(f"분석 2 — GPU_available={a2.get('gpu_available')}  "
          f"LLM 결과 {len(a2.get('llm_results', []))}개")
    print(f"분석 3 — G1 temporal CV={a3['g1_temporal_cv']:.4f}  "
          f"actual temporal CV={a3['act_temporal_cv']:.4f}")
    print(f"보고서: {REPORT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
