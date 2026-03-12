"""exp022 분석 2-1: G1 attention weights 추출 (GPU 불필요).

LLM 답변 생성 없이 attention weight만 추출하여 보고서에 저장.
LLM 부분은 GPU 가용 시 exp022_analysis2_llm.py로 실행.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

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
            logger.info("  epoch %d/500 loss=%.4f", ep + 1, loss.item())
    m.eval()
    return m


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

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
    sc = pd.read_csv(SC_PATH)

    return dict(warm_res=warm_res, cold_res=cold_res,
                cold_ids=cold_ids, warm_ids=warm_ids,
                y_warm=y_warm, scorecard=sc)


# ─── persona 텍스트 빌더 ──────────────────────────────────────────────────────

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


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Exp022 분석 2-1: Attention Weights ===")
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

    # 페르소나 로드
    with open(PERSONA_PATH) as f:
        all_p = json.load(f)
    p_map  = {p["persona_id"]: p["profile"] for p in all_p}
    p_ids  = [p["persona_id"] for p in all_p]

    # attention weights 추출
    item_attn_records: list[dict] = []
    for iid, grp in items_6:
        idx = ii.get(iid, -1)
        if idx < 0: continue
        x = torch.tensor(data["cold_res"][idx:idx+1], dtype=torch.float32)
        with torch.no_grad():
            _, aw = g1.forward_with_attn(x)
        aw_np = aw[0].numpy()  # (50,)

        top5_idx = np.argsort(aw_np)[::-1][:5]
        sc_row = sc[sc["item_id"] == iid].iloc[0]
        for rank, p_idx in enumerate(top5_idx):
            pid = p_ids[p_idx] if p_idx < len(p_ids) else f"P{p_idx:03d}"
            pp  = p_map.get(pid, {})
            item_attn_records.append(dict(
                item_id=iid, group=grp, rank=rank+1,
                persona_id=pid, attn_weight=float(aw_np[p_idx]),
                weekly_budget=pp.get("weekly_budget"),
                economic_status=pp.get("economic_status"),
                shopping_motivation=pp.get("shopping_motivation"),
                price_sensitivity=pp.get("price_sensitivity"),
                persona_text=_build_persona_text(pp) if pp else "",
            ))

        logger.info("  %s [%s]: top3=%s",
                    iid, grp, [(p_ids[k] if k < len(p_ids) else k,
                                f"{aw_np[k]:.4f}") for k in top5_idx[:3]])

    # JSON 저장
    out_json = OUT_DIR / "analysis2_attn_weights.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(item_attn_records, f, ensure_ascii=False, indent=2)
    logger.info("attn_weights 저장: %s", out_json)

    # 보고서 분석 2 섹션 업데이트
    _update_report(item_attn_records, items_6, sc)


def _update_report(records: list[dict], items_6: list[tuple], sc: pd.DataFrame):
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""

    lines: list[str] = [
        "## 분석 2: LLM 실제 답변 + Rationale",
        "",
        "**대상:** 선택품(nonzero<0.5) TB WRMSSE 상위 3개 + 하위 3개 = 총 6개 아이템",
        "**방법:** 각 아이템의 G1 attention weight 상위 3개 페르소나 × 6 아이템 = 18 조합",
        "**LLM 답변:** GPU 자원 미확보로 현재 미실행. GPU 가용 시 exp022_analysis2_llm.py 실행.",
        "",
        "### 2-1: 선정 아이템 요약",
        "",
        "| item_id | group | actual_mean | nonzero_ratio | TB_WRMSSE | LGBM_WRMSSE |",
        "|---------|-------|------------|---------------|-----------|-------------|",
    ]
    for iid, grp in items_6:
        r = sc[sc["item_id"] == iid]
        if r.empty: continue
        r = r.iloc[0]
        lines.append(
            f"| {iid} | {grp} | {r['actual_weekly_mean']:.2f} | "
            f"{r['actual_nonzero_week_ratio']:.2f} | "
            f"{r['trackb_WRMSSE']:.4f} | {r['lgbm_WRMSSE']:.4f} |"
        )

    lines += [
        "",
        "### 2-2: 아이템별 top-5 페르소나 attention weight",
        "",
        "| item_id | group | rank | persona_id | attn_weight | budget | economic_status | shopping_motivation |",
        "|---------|-------|------|------------|-------------|--------|-----------------|---------------------|",
    ]
    for r in records:
        bgt = f"${r['weekly_budget']:.0f}" if r.get("weekly_budget") else "N/A"
        lines.append(
            f"| {r['item_id']} | {r['group']} | {r['rank']} | "
            f"{r['persona_id']} | {r['attn_weight']:.4f} | "
            f"{bgt} | {r.get('economic_status','N/A')} | "
            f"{r.get('shopping_motivation','N/A')} |"
        )

    lines += [
        "",
        "### 2-3: top-3 페르소나 프로필 전문 (LLM 답변 대상)",
        "",
    ]
    top3_records = [r for r in records if r["rank"] <= 3]
    cur_item = None
    for r in top3_records:
        if r["item_id"] != cur_item:
            cur_item = r["item_id"]
            sc_r = sc[sc["item_id"] == cur_item]
            am   = sc_r["actual_weekly_mean"].values[0] if not sc_r.empty else "N/A"
            nzr  = sc_r["actual_nonzero_week_ratio"].values[0] if not sc_r.empty else "N/A"
            tw   = sc_r["trackb_WRMSSE"].values[0] if not sc_r.empty else "N/A"
            lines += [
                "",
                f"=== ITEM: {cur_item} [{r['group']}] ===",
                f"dept={sc_r['dept_id'].values[0] if not sc_r.empty else 'N/A'} | "
                f"cat={sc_r['cat_id'].values[0] if not sc_r.empty else 'N/A'} | "
                f"actual_mean={am:.2f} | nonzero_ratio={nzr:.2f} | TB_WRMSSE={tw:.4f}" if isinstance(am, float) else "",
                "",
            ]
        bgt = f"${r['weekly_budget']:.0f}" if r.get("weekly_budget") else "N/A"
        lines += [
            f"--- Persona {r['persona_id']} "
            f"(rank={r['rank']}, attn_weight={r['attn_weight']:.4f}, "
            f"budget={bgt}, {r.get('economic_status','')}) ---",
            "Persona text:",
            r.get("persona_text", "N/A"),
            "",
            "**LLM 답변: GPU 가용 시 exp022_analysis2_llm.py로 생성 예정.**",
            "",
        ]

    # 기존 보고서에서 분석 2 섹션 교체
    marker = "## 분석 2: LLM 실제 답변 + Rationale"
    sep     = "\n---\n\n## 분석 3:"
    if marker in existing:
        before = existing[:existing.index(marker)]
        rest   = existing[existing.index(marker):]
        if "\n---\n\n## 분석 3:" in rest:
            after = "\n---\n\n## 분석 3:" + rest.split("\n---\n\n## 분석 3:", 1)[1]
        else:
            after = ""
        new_content = before + "\n".join(lines) + after
    else:
        new_content = existing + "\n\n" + "\n".join(lines)

    REPORT.write_text(new_content, encoding="utf-8")
    logger.info("보고서 업데이트: %s", REPORT)

    print("\n" + "="*60)
    print("[분석 2-1 완료] Attention Weights 추출")
    for iid, grp in {(r["item_id"], r["group"]) for r in records}:
        top3 = [r for r in records if r["item_id"] == iid and r["rank"] <= 3]
        print(f"  {iid} [{grp}]:")
        for r in top3:
            print(f"    rank{r['rank']}: {r['persona_id']} aw={r['attn_weight']:.4f} "
                  f"budget=${r.get('weekly_budget',0):.0f} {r.get('economic_status','')}")
    print("="*60)


if __name__ == "__main__":
    main()
