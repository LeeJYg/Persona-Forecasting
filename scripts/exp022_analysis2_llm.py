"""exp022 분석 2: LLM 실제 답변 + Rationale (GPU 필요, 오프라인 모드).

분석 1, 3은 exp022_qualitative.py에서 완료됨.
이 스크립트는 분석 2만 실행 후 qualitative_analysis.md에 추가.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# 오프라인 모드: 로컬 캐시에서만 로드
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "/mnt/sdd1/jylee/huggingface_cache")

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EMB_DIR      = ROOT / "experiments/exp011_v3_pipeline/embeddings"
CS_DIR       = ROOT / "data/processed/cold_start"
SC_PATH      = ROOT / "experiments/exp020_error_analysis/item_scorecard.csv"
PERSONA_PATH = ROOT / "data/processed/personas/all_personas.json"
REPORT       = ROOT / "docs/diagnosis/qualitative_analysis.md"
OUT_DIR      = ROOT / "experiments/exp022_qualitative"


# ─── 모델 ─────────────────────────────────────────────────────────────────────

class AttnBottleneckG1(nn.Module):
    def __init__(self, hidden=5120, bottleneck=64, n_weeks=16, dropout=0.1):
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, bottleneck), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, n_weeks),
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        return self.head((x * w.unsqueeze(-1)).sum(1))

    def forward_with_attn(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        return self.head((x * w.unsqueeze(-1)).sum(1)), w


def train_g1(X, y, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    m = AttnBottleneckG1(hidden=X.shape[2], n_weeks=y.shape[1])
    opt = Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    m.train()
    for ep in range(500):
        opt.zero_grad()
        loss = F.l1_loss(m(Xt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (ep + 1) % 100 == 0:
            logger.info("  G1 epoch %d/500 loss=%.4f", ep + 1, loss.item())
    m.eval()
    return m


# ─── 데이터 로드 (최소) ───────────────────────────────────────────────────────

def load_minimal():
    logger.info("데이터 로드 중...")
    warm_raw = torch.load(EMB_DIR / "warm_raw.pt", map_location="cpu",
                          weights_only=False).numpy()
    cold_raw = torch.load(EMB_DIR / "cold_raw.pt", map_location="cpu",
                          weights_only=False).numpy()
    warm_res = warm_raw - warm_raw.mean(axis=1, keepdims=True)
    cold_res = cold_raw - cold_raw.mean(axis=1, keepdims=True)

    item_meta = pd.read_csv(EMB_DIR / "item_meta.csv")
    cold_ids  = item_meta[item_meta["is_cold"] == True]["item_id"].tolist()
    warm_ids  = item_meta[item_meta["is_cold"] == False]["item_id"].tolist()

    cold_test_raw = pd.read_csv(CS_DIR / "cold_test.csv", parse_dates=["date"])
    warm_test_raw = pd.read_csv(CS_DIR / "warm_test.csv", parse_dates=["date"])

    def to_weekly(df):
        df = df.copy()
        df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
        df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
        df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
        gc = [c for c in ["item_id","store_id","cat_id","dept_id","iso_year","iso_week"]
              if c in df.columns]
        return df.groupby(gc).agg(sales=("sales","sum"), date=("week_start","first")).reset_index()

    cold_w = to_weekly(cold_test_raw)
    warm_w = to_weekly(warm_test_raw)

    day_cnt = (cold_test_raw
               .assign(iso_year=cold_test_raw["date"].dt.isocalendar().year.astype(int),
                       iso_week=cold_test_raw["date"].dt.isocalendar().week.astype(int))
               .groupby(["iso_year","iso_week"])["date"].nunique())
    complete = set(zip(day_cnt[day_cnt==7].index.get_level_values(0),
                       day_cnt[day_cnt==7].index.get_level_values(1)))
    cold_w16 = cold_w[cold_w.apply(lambda r: (r["iso_year"],r["iso_week"]) in complete, axis=1)]
    wl16 = sorted(complete)

    def build_y(ids, weekly, wl):
        wi = {wk:i for i,wk in enumerate(wl)}
        ii = {iid:i for i,iid in enumerate(ids)}
        y = np.zeros((len(ids), len(wl)), dtype=np.float32)
        for _, r in weekly.iterrows():
            wk=(r["iso_year"],r["iso_week"])
            if wk in wi and r["item_id"] in ii:
                y[ii[r["item_id"]], wi[wk]] = r["sales"]
        return y

    y_warm = build_y(warm_ids, warm_w, wl16)
    y_cold = build_y(cold_ids, cold_w16, wl16)
    sc = pd.read_csv(SC_PATH)

    return dict(warm_res=warm_res, cold_res=cold_res,
                cold_ids=cold_ids, warm_ids=warm_ids,
                y_warm=y_warm, y_cold=y_cold, scorecard=sc)


# ─── 페르소나 텍스트 ──────────────────────────────────────────────────────────

def _build_persona_text(p: dict) -> str:
    cat_pref = ", ".join(
        f"{c}={v:.2f}" for c, v in
        sorted(p["category_preference"].items(), key=lambda x: -x[1])
    )
    return (
        f"Customer profile:\n"
        f"- Weekly budget: ${p['weekly_budget']:.2f}\n"
        f"- SNAP eligible: {str(p['snap_eligible']).lower()}\n"
        f"- Economic status: {p['economic_status']}\n"
        f"- Shopping motivation: {p['shopping_motivation']}\n"
        f"- Category preference: {cat_pref}\n"
        f"- Price sensitivity: {p['price_sensitivity']}\n"
        f"- Visit frequency: {p['visit_frequency']}\n"
        f"- Preferred departments: {', '.join(p['preferred_departments'])}\n"
        f"- Decision style: {p['decision_style']}\n"
        f"- Brand loyalty: {p['brand_loyalty']}\n"
        f"- Promotion sensitivity: {p['promotion_sensitivity']}"
    )


def _build_rationale_prompt(persona_text: str, dept: str, cat: str,
                             avg_price: float | None) -> str:
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


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Exp022 분석 2: LLM Rationale ===")
    data = load_minimal()

    # G1 학습
    logger.info("G1 모델 학습 중...")
    g1 = train_g1(data["warm_res"], data["y_warm"], seed=42)

    # 6개 아이템 선정
    sc  = data["scorecard"]
    ids = data["cold_ids"]
    ii  = {iid: i for i, iid in enumerate(ids)}
    elective = sc[sc["actual_nonzero_week_ratio"] < 0.5].copy()
    top3_ids = elective.nsmallest(3, "trackb_WRMSSE")["item_id"].tolist()
    bot3_ids = elective.nlargest(3, "trackb_WRMSSE")["item_id"].tolist()
    items_6  = [(iid, "선택품_좋음") for iid in top3_ids] + \
               [(iid, "선택품_나쁨") for iid in bot3_ids]
    logger.info("  대상 아이템: %s", [x[0] for x in items_6])

    # attention weights 추출
    item_attn: dict[str, np.ndarray] = {}
    for iid, _ in items_6:
        idx = ii.get(iid, -1)
        if idx < 0: continue
        x = torch.tensor(data["cold_res"][idx:idx+1], dtype=torch.float32)
        with torch.no_grad():
            _, aw = g1.forward_with_attn(x)
        item_attn[iid] = aw[0].numpy()

    # 페르소나 로드
    with open(PERSONA_PATH) as f:
        all_p = json.load(f)
    p_map  = {p["persona_id"]: p["profile"] for p in all_p}
    p_ids  = [p["persona_id"] for p in all_p]

    # LLM 로드 (오프라인)
    logger.info("LLM 로드 중 (오프라인 모드)...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model_id = "Qwen/Qwen2.5-32B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    )
    llm.eval()
    logger.info("LLM 로드 완료")

    # 18개 생성
    results: list[dict] = []
    for iid, grp in items_6:
        if iid not in item_attn: continue
        aw = item_attn[iid]
        top3_idx = np.argsort(aw)[::-1][:3]

        sr = sc[sc["item_id"] == iid].iloc[0]
        dept = sr["dept_id"]; cat = sr["cat_id"]
        avg_price = sr.get("avg_price") if not pd.isna(sr.get("avg_price", float("nan"))) else None

        for rank, p_idx in enumerate(top3_idx):
            if p_idx >= len(p_ids): continue
            p_id = p_ids[p_idx]
            p_profile = p_map.get(p_id, {})
            aw_score  = float(aw[p_idx])

            prompt = _build_rationale_prompt(_build_persona_text(p_profile),
                                             dept, cat, avg_price)
            try:
                inp = tok(prompt, return_tensors="pt").to(llm.device)
                with torch.no_grad():
                    out = llm.generate(
                        **inp, max_new_tokens=200, do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                gen = out[0][inp["input_ids"].shape[-1]:]
                response = tok.decode(gen, skip_special_tokens=True).strip()
            except Exception as e:
                response = f"[ERROR: {e}]"

            results.append(dict(
                item_id=iid, group=grp, dept_id=dept, cat_id=cat, avg_price=avg_price,
                persona_id=p_id, attn_weight=aw_score, rank=rank+1,
                weekly_budget=p_profile.get("weekly_budget"),
                economic_status=p_profile.get("economic_status"),
                response=response,
            ))
            logger.info("  [%s × %s rank=%d aw=%.4f] len=%d",
                        iid, p_id, rank+1, aw_score, len(response))

    # 결과를 JSON으로 저장
    out_json = OUT_DIR / "analysis2_llm_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("  결과 저장: %s", out_json)

    # 보고서 업데이트 (분석 2 섹션 교체)
    _update_report(results, sc, item_attn, p_ids)
    logger.info("보고서 업데이트 완료: %s", REPORT)

    print("\n" + "="*60)
    print(f"[분석 2 완료] LLM 결과 {len(results)}개")
    for r in results:
        print(f"  {r['item_id']} × {r['persona_id']} (rank={r['rank']} aw={r['attn_weight']:.4f}): {r['response'][:80]}...")
    print("="*60)


def _update_report(results, sc, item_attn, p_ids):
    """기존 보고서의 분석 2 섹션을 LLM 결과로 교체."""
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""

    new_sec: list[str] = [
        "## 분석 2: LLM 실제 답변 + Rationale",
        "",
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
    for r in results:
        bgt = f"${r['weekly_budget']:.0f}" if r.get("weekly_budget") else "N/A"
        new_sec.append(
            f"| {r['item_id']} | {r['group']} | {r['rank']} | "
            f"{r['persona_id']} | {r['attn_weight']:.4f} | "
            f"{bgt} | {r.get('economic_status','N/A')} |"
        )

    new_sec += ["", "### 2-3~2-4: LLM 생성 답변 전문", ""]
    cur_item = None
    for r in results:
        if r["item_id"] != cur_item:
            cur_item = r["item_id"]
            row = sc[sc["item_id"] == cur_item]
            am  = f"{row['actual_weekly_mean'].values[0]:.2f}" if not row.empty else "N/A"
            nzr = f"{row['actual_nonzero_week_ratio'].values[0]:.2f}" if not row.empty else "N/A"
            tw  = f"{row['trackb_WRMSSE'].values[0]:.4f}" if not row.empty else "N/A"
            new_sec += [
                "",
                f"=== ITEM: {cur_item} [{r['group']}] ===",
                f"dept={r['dept_id']} | cat={r['cat_id']} | "
                f"avg_price=${r['avg_price']:.2f}" if r.get("avg_price") else
                f"dept={r['dept_id']} | cat={r['cat_id']}",
                f"actual_mean={am} | nonzero_ratio={nzr} | TB_WRMSSE={tw}",
                "",
            ]
        bgt = f"${r['weekly_budget']:.0f}" if r.get("weekly_budget") else "N/A"
        new_sec += [
            f"--- Persona {r['persona_id']} "
            f"(attention weight: {r['attn_weight']:.4f}, "
            f"budget: {bgt}, {r.get('economic_status','')}) ---",
            "LLM 답변:",
            f'"{r["response"]}"',
            "",
        ]

    # 기존 보고서에서 분석 2 섹션 찾아서 교체
    marker_start = "## 분석 2: LLM 실제 답변 + Rationale"
    marker_end   = "---\n\n## 분석 3:"
    if marker_start in existing:
        before = existing[:existing.index(marker_start)]
        after_part = existing[existing.index(marker_start):]
        if "---\n\n## 분석 3:" in after_part:
            after = "---\n\n## 분석 3:" + after_part.split("---\n\n## 분석 3:", 1)[1]
        else:
            after = ""
        new_content = before + "\n".join(new_sec) + "\n\n---\n\n" + after.replace("---\n\n## 분석 3:", "## 분석 3:", 1)
    else:
        new_content = existing + "\n\n" + "\n".join(new_sec)

    REPORT.write_text(new_content, encoding="utf-8")


if __name__ == "__main__":
    main()
