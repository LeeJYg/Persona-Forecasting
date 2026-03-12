"""Save exp018 Part 1-3B checkpoints from log data."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
CKPT_DIR = ROOT / "experiments/exp018_optimization/checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def save_ckpt(name, obj):
    path = CKPT_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


# ── Part 1 ─────────────────────────────────────────────────────────────────
variants = {
    "baseline": {"emb": "residual", "weeks": "16", "epochs": 500,
                 "maes": [8.5096, 8.4395, 8.3171], "daccs": [0.384, 0.377, 0.399]},
    "1A-a":    {"emb": "raw",      "weeks": "16", "epochs": 500,
                 "maes": [8.6536, 8.7670, 8.6210], "daccs": [0.385, 0.389, 0.380]},
    "1A-b":    {"emb": "residual", "weeks": "17", "epochs": 500,
                 "maes": [8.3459, 9.2210, 8.1090], "daccs": [0.418, 0.401, 0.398]},
    "1A-c":    {"emb": "residual", "weeks": "16", "epochs": 200,
                 "maes": [8.6408, 8.6118, 8.3914], "daccs": [0.383, 0.373, 0.389]},
    "1A-d":    {"emb": "raw",      "weeks": "17", "epochs": 500,
                 "maes": [8.5782, 8.8691, 8.6596], "daccs": [0.415, 0.391, 0.407]},
}
p1 = {}
for name, v in variants.items():
    maes = v["maes"]
    daccs = v["daccs"]
    p1[name] = {
        "emb": v["emb"], "weeks": v["weeks"], "epochs": v["epochs"],
        "maes": maes, "daccs": daccs,
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "diracc_mean": float(np.mean(daccs)), "diracc_std": float(np.std(daccs)),
    }
save_ckpt("p1", p1)

# ── Part 2A ─────────────────────────────────────────────────────────────────
# Part 2A (diffuse vs sparse) — no numeric results logged, use placeholder
p2a = {"sparse": {}, "diffuse": {}, "note": "profiling only — no numeric rows"}
save_ckpt("p2a", p2a)

# ── Part 2B-1 ───────────────────────────────────────────────────────────────
p2b1 = {"rows": [
    {"T": 0.1, "mae": 9.6609, "dir_acc": 0.389},
    {"T": 0.3, "mae": 9.3620, "dir_acc": 0.391},
    {"T": 0.5, "mae": 8.9677, "dir_acc": 0.389},
    {"T": 1.0, "mae": 8.5096, "dir_acc": 0.384},
    {"T": 2.0, "mae": 8.5310, "dir_acc": 0.389},
]}
save_ckpt("p2b1", p2b1)

# ── Part 2B-2 ───────────────────────────────────────────────────────────────
p2b2 = {"rows": [
    {"K": 3,  "mae": 8.4998, "dir_acc": 0.386},
    {"K": 5,  "mae": 8.5143, "dir_acc": 0.384},
    {"K": 10, "mae": 8.5096, "dir_acc": 0.384},
    {"K": 20, "mae": 8.5096, "dir_acc": 0.384},
    {"K": 50, "mae": 8.5096, "dir_acc": 0.384},
]}
save_ckpt("p2b2", p2b2)

# ── Part 3A ─────────────────────────────────────────────────────────────────
p3a = {"rows": [
    {"lambda_dir": 0.0, "mae_mean": 8.4221, "mae_std": 0.0795, "diracc_mean": 0.3867, "diracc_std": 0.0094},
    {"lambda_dir": 0.1, "mae_mean": 8.4868, "mae_std": 0.0570, "diracc_mean": 0.3869, "diracc_std": 0.0056},
    {"lambda_dir": 0.5, "mae_mean": 8.4437, "mae_std": 0.0143, "diracc_mean": 0.4016, "diracc_std": 0.0053},
    {"lambda_dir": 1.0, "mae_mean": 8.4135, "mae_std": 0.2777, "diracc_mean": 0.3916, "diracc_std": 0.0065},
    {"lambda_dir": 2.0, "mae_mean": 8.6201, "mae_std": 0.2106, "diracc_mean": 0.3933, "diracc_std": 0.0103},
]}
save_ckpt("p3a", p3a)

# ── Part 3B ─────────────────────────────────────────────────────────────────
# Reconstruct 27 combos from 81 eval_cold_16w log lines
# Order: for lr in [5e-4, 1e-3, 2e-3], for bn in [32, 64, 128], for ep in [300, 500, 1000], for seed in [0,1,2]
lr_vals = [5e-4, 1e-3, 2e-3]
bn_vals = [32, 64, 128]
ep_vals = [300, 500, 1000]

# 81 raw eval results in order (MAE, DirAcc) from Part 3B log
raw_evals = [
    (8.7169, 0.4007), (8.9093, 0.3893), (8.9714, 0.3987),  # combo 1: lr=5e-4 bn=32 ep=300
    (8.7380, 0.3993), (8.9434, 0.3920), (8.8351, 0.3913),  # combo 2: lr=5e-4 bn=32 ep=500
    (8.7298, 0.3973), (8.9212, 0.3913), (8.6702, 0.3753),  # combo 3: lr=5e-4 bn=32 ep=1000
    (8.5457, 0.3920), (8.8775, 0.3687), (8.1281, 0.3913),  # combo 4: lr=5e-4 bn=64 ep=300
    (8.4596, 0.3913), (8.6607, 0.3647), (8.1050, 0.3867),  # combo 5: lr=5e-4 bn=64 ep=500
    (8.7585, 0.3893), (8.6114, 0.3860), (8.2053, 0.3940),  # combo 6: lr=5e-4 bn=64 ep=1000
    (8.8709, 0.3807), (8.9784, 0.3973), (8.3866, 0.3913),  # combo 7: lr=5e-4 bn=128 ep=300
    (8.7381, 0.3840), (9.3338, 0.3940), (8.3159, 0.3773),  # combo 8: lr=5e-4 bn=128 ep=500
    (8.6217, 0.3960), (9.4373, 0.3973), (8.2864, 0.3960),  # combo 9: lr=5e-4 bn=128 ep=1000
    (8.7423, 0.3847), (8.5670, 0.3800), (8.5611, 0.3827),  # combo 10: lr=1e-3 bn=32 ep=300
    (8.9794, 0.3893), (8.3553, 0.3673), (8.7586, 0.3660),  # combo 11: lr=1e-3 bn=32 ep=500
    (9.0130, 0.3940), (8.9056, 0.3747), (8.9838, 0.3713),  # combo 12: lr=1e-3 bn=32 ep=1000
    (8.5054, 0.3953), (8.3582, 0.3660), (8.3988, 0.3867),  # combo 13: lr=1e-3 bn=64 ep=300
    (8.5096, 0.3840), (8.4395, 0.3767), (8.3171, 0.3993),  # combo 14: lr=1e-3 bn=64 ep=500
    (9.1047, 0.3653), (8.5427, 0.3933), (8.4592, 0.4027),  # combo 15: lr=1e-3 bn=64 ep=1000
    (8.4815, 0.3973), (8.7361, 0.3767), (8.5569, 0.3860),  # combo 16: lr=1e-3 bn=128 ep=300
    (8.4966, 0.4140), (8.7500, 0.3967), (8.4921, 0.4053),  # combo 17: lr=1e-3 bn=128 ep=500
    (8.4777, 0.3660), (8.5108, 0.3813), (8.4889, 0.3760),  # combo 18: lr=1e-3 bn=128 ep=1000
    (8.9776, 0.3800), (9.1895, 0.3640), (8.5785, 0.3653),  # combo 19: lr=2e-3 bn=32 ep=300
    (8.7165, 0.3833), (9.2343, 0.3660), (8.7643, 0.3940),  # combo 20: lr=2e-3 bn=32 ep=500
    (8.8573, 0.3740), (9.2568, 0.3847), (8.6623, 0.3740),  # combo 21: lr=2e-3 bn=32 ep=1000
    (8.8558, 0.3900), (8.7381, 0.3833), (8.4740, 0.3973),  # combo 22: lr=2e-3 bn=64 ep=300
    (8.2938, 0.3840), (8.4950, 0.3700), (8.4188, 0.3893),  # combo 23: lr=2e-3 bn=64 ep=500
    (8.7109, 0.3893), (8.6462, 0.3727), (8.3995, 0.3933),  # combo 24: lr=2e-3 bn=64 ep=1000
    (8.5170, 0.3913), (8.4500, 0.3920), (8.6507, 0.3853),  # combo 25: lr=2e-3 bn=128 ep=300
    (8.4062, 0.3853), (8.5565, 0.3893), (8.4661, 0.3953),  # combo 26: lr=2e-3 bn=128 ep=500
    (8.4161, 0.3887), (8.7181, 0.3980), (8.6117, 0.3927),  # combo 27: lr=2e-3 bn=128 ep=1000
]

assert len(raw_evals) == 81, f"Expected 81, got {len(raw_evals)}"

rows = []
idx = 0
for lr in lr_vals:
    for bn in bn_vals:
        for ep in ep_vals:
            seed_maes = [raw_evals[idx][0], raw_evals[idx+1][0], raw_evals[idx+2][0]]
            seed_daccs = [raw_evals[idx][1], raw_evals[idx+1][1], raw_evals[idx+2][1]]
            rows.append({
                "lr": lr, "bottleneck": bn, "epochs": ep,
                "mae_mean": float(np.mean(seed_maes)), "mae_std": float(np.std(seed_maes)),
                "diracc_mean": float(np.mean(seed_daccs)), "diracc_std": float(np.std(seed_daccs)),
            })
            idx += 3

# Compute Pareto front
def compute_pareto(rows):
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

pareto = compute_pareto(rows)
print(f"Part 3B: {len(rows)} rows, {len(pareto)} Pareto points")
for r in pareto:
    print(f"  lr={r['lr']:.4f} bn={r['bottleneck']} ep={r['epochs']} MAE={r['mae_mean']:.4f} DirAcc={r['diracc_mean']:.4f}")

p3b = {"rows": rows, "pareto": pareto}
save_ckpt("p3b", p3b)

print("\nAll checkpoints saved successfully.")
