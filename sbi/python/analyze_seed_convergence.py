"""
analyze_seed_convergence.py

Assesses how the number of seed realizations (k = 1..K) affects the stability
of simulated minirhizotron observations.

For each k, the running mean over the first k seeds is compared to the full
K-seed mean (used as the reference). The relative Frobenius error

    err(k) = ||mean_{1..k} - mean_{1..K}||_F / ||mean_{1..K}||_F

is computed per sample, then averaged across all N samples.

Supports two dataset versions:
  --k7   : uses tensors/      (K=7,  original)        [default if tensors_k20 absent]
  --k20  : uses tensors_k20/  (K=20, extended dataset)

Run with K=20 after extend_multiseed_dataset.jl / run_extend_multiseed_all_configs.sh
to see the full convergence curve and whether K=7 was sufficient.

Outputs:
  sbi/figures/seed_convergence[_k20].png          — main figure (3 panels)
  sbi/figures/seed_convergence_quintile[_k20].png — per-g_axial quintile
  (stdout)                                         — summary table
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS   = ["A", "B2", "C", "D2"]
CHANNELS  = ["root_length_cm", "has_root", "crossing_angle"]
FIG_DIR   = os.path.join(REPO_ROOT, "sbi", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Mode: K=7 (default) or K=20 (--k20 flag) ─────────────────────────────────

USE_K20      = "--k20" in sys.argv
TENSORS_SUBDIR = "tensors_k20" if USE_K20 else "tensors"
FIG_SUFFIX     = "_k20"       if USE_K20 else ""
K              = 20            if USE_K20 else 7

if USE_K20:
    print("Mode: K=20 (tensors_k20/)")
else:
    print("Mode: K=7  (tensors/)  — pass --k20 to use extended dataset")

# ── 1. Load all configs ───────────────────────────────────────────────────────

all_X   = []
all_meta = []

for cfg in CONFIGS:
    cfg_dir  = os.path.join(REPO_ROOT, "sbi", "data", "multi_seed", f"config_{cfg}")
    npz_path = os.path.join(cfg_dir, TENSORS_SUBDIR, "X_raw.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found.\n"
            + ("Run sbi/run_extend_multiseed_all_configs.sh first."
               if USE_K20 else
               "Run sbi/run_multiseed_all_configs.sh first.")
        )
    X   = np.load(npz_path)["X"]   # (N, K, T, ns, na, 3)
    df  = pd.read_csv(os.path.join(cfg_dir, "samples_filtered.csv"))
    df["config"] = cfg
    all_X.append(X)
    all_meta.append(df)

X_all  = np.concatenate(all_X, axis=0)
df_all = pd.concat(all_meta, ignore_index=True)

N, K_actual, T, ns, na, nc = X_all.shape
assert K_actual == K, f"Expected K={K}, got {K_actual}"
print(f"Loaded {N} samples × K={K} seeds  shape={X_all.shape}")

# ── 2. Compute per-sample relative Frobenius error for k=1..K ────────────────

ref    = X_all.mean(axis=1)   # (N, T, ns, na, 3)  — K=7 reference

# errors[i, k, c] = relative error for sample i, first k seeds, channel c
errors = np.zeros((N, K, nc), dtype=np.float32)

for k in range(1, K + 1):
    mean_k = X_all[:, :k].mean(axis=1)   # (N, T, ns, na, 3)
    diff   = mean_k - ref                  # (N, T, ns, na, 3)
    for c in range(nc):
        # Frobenius norm over (T, ns, na), normalized by K=7 reference
        norm_diff = np.sqrt((diff[..., c] ** 2).mean(axis=(1, 2, 3)))   # (N,)
        norm_ref  = np.sqrt((ref[..., c]  ** 2).mean(axis=(1, 2, 3))) + 1e-8
        errors[:, k - 1, c] = norm_diff / norm_ref

# ── 3. Also compute absolute RMSE for root_length_cm ─────────────────────────

abs_rmse = np.zeros((N, K), dtype=np.float32)
for k in range(1, K + 1):
    mean_k = X_all[:, :k, ..., 0].mean(axis=1)    # (N, T, ns, na)
    diff   = mean_k - ref[..., 0]
    abs_rmse[:, k - 1] = np.sqrt((diff ** 2).mean(axis=(1, 2, 3)))

# ── 4. g_axial quintiles ──────────────────────────────────────────────────────

df_all["g_axial_quintile"] = pd.qcut(
    df_all["g_axial"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
)
quintile_idx = {q: np.where(df_all["g_axial_quintile"] == q)[0]
                for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]}

# ── 5. Summary statistics ─────────────────────────────────────────────────────

ks = np.arange(1, K + 1)
print("\n── Relative Frobenius error vs k (mean ± std across N samples) ──")
print(f"{'k':>3}  {'root_len (rel)':>16}  {'has_root (rel)':>16}  {'angle (rel)':>13}  {'abs RMSE (cm)':>15}")
for k in range(K):
    vals = [f"{errors[:, k, c].mean():.4f} ± {errors[:, k, c].std():.4f}"
            for c in range(nc)]
    print(f"{k+1:>3}  {vals[0]:>16}  {vals[1]:>16}  {vals[2]:>13}  "
          f"{abs_rmse[:, k].mean():.4f} ± {abs_rmse[:, k].std():.4f}")

# Percentage improvement k=1 → k=7
for c, ch in enumerate(CHANNELS):
    e1, e7 = errors[:, 0, c].mean(), errors[:, -1, c].mean()
    print(f"\n{ch}: k=1 err={e1:.4f} → k=7 err={e7:.4f}  "
          f"({(e1-e7)/e1*100:.1f}% improvement)")

# ── 6. Main figure ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

Q_COLORS = {"Q1": "#4393C3", "Q2": "#92C5DE", "Q3": "#B8D97C",
             "Q4": "#F4A582", "Q5": "#D6604D"}
CH_COLORS = ["#2E6DA4", "#2CA02C", "#D95F02"]
CH_LABELS = ["Root length (cm)", "Root presence", "Crossing angle"]
CH_MARKERS = ["o", "s", "^"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                          gridspec_kw={"wspace": 0.35})
fig.patch.set_facecolor("white")

# ── Panel (a): relative error by channel ──────────────────────────────────────
tick_step_main = 2 if K > 10 else 1
ax = axes[0]
for c in range(nc):
    m = errors[:, :, c].mean(axis=0)
    s = errors[:, :, c].std(axis=0)
    ax.plot(ks, m, color=CH_COLORS[c], marker=CH_MARKERS[c],
            markersize=5, lw=1.5, label=CH_LABELS[c])
    ax.fill_between(ks, m - s, m + s, color=CH_COLORS[c], alpha=0.12)

ax.set_xticks(ks[::tick_step_main])
ax.set_xlabel("Number of seeds (k)", fontsize=9)
ax.set_ylabel(f"Relative error vs K={K} reference", fontsize=9)
ax.set_title("(a) Convergence by channel", loc="left", fontsize=9, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.axhline(0, color="0.5", lw=0.6, ls="--")

# ── Panel (b): absolute RMSE for root length ──────────────────────────────────
ax = axes[1]
m = abs_rmse.mean(axis=0)
s = abs_rmse.std(axis=0)
p10 = np.percentile(abs_rmse, 10, axis=0)
p90 = np.percentile(abs_rmse, 90, axis=0)

ax.plot(ks, m, color=CH_COLORS[0], marker="o", markersize=5, lw=1.5,
        label="Mean ± SD")
ax.fill_between(ks, m - s, m + s, color=CH_COLORS[0], alpha=0.15)
ax.fill_between(ks, p10, p90, color=CH_COLORS[0], alpha=0.07,
                label="P10–P90")

ax.set_xticks(ks[::tick_step_main])
ax.set_xlabel("Number of seeds (k)", fontsize=9)
ax.set_ylabel("RMSE of mean root length (cm)", fontsize=9)
ax.set_title("(b) Absolute error — root length", loc="left", fontsize=9, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.axhline(0, color="0.5", lw=0.6, ls="--")

# ── Panel (c): relative error by g_axial quintile ────────────────────────────
ax = axes[2]
for q, idx in quintile_idx.items():
    if len(idx) == 0:
        continue
    m = errors[idx, :, 0].mean(axis=0)
    ax.plot(ks, m, color=Q_COLORS[q], marker="o", markersize=4,
            lw=1.4, label=f"{q} (n={len(idx)})")

ax.set_xticks(ks[::tick_step_main])
ax.set_xlabel("Number of seeds (k)", fontsize=9)
ax.set_ylabel(f"Relative error vs K={K} (root length)", fontsize=9)
ax.set_title("(c) Convergence by g_axial quintile", loc="left", fontsize=9, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, title="g_axial (Q1=slow → Q5=fast)",
          title_fontsize=7)
ax.axhline(0, color="0.5", lw=0.6, ls="--")

fig.suptitle(
    f"Seed convergence analysis: N={N} samples × K={K} seeds  "
    "(all 4 configs pooled)",
    y=1.01, fontsize=10.5, fontweight="bold"
)

out_main = os.path.join(FIG_DIR, f"seed_convergence{FIG_SUFFIX}.png")
plt.savefig(out_main, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {out_main}")
plt.close()

# ── 7. Per-quintile detail figure ────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.15})
fig2.patch.set_facecolor("white")

for qi, (q, idx) in enumerate(quintile_idx.items()):
    ax = axes2[qi]
    if len(idx) == 0:
        ax.set_visible(False)
        continue

    sub = errors[idx, :, 0]   # (n_q, K)
    m   = sub.mean(axis=0)
    s   = sub.std(axis=0)
    p25 = np.percentile(sub, 25, axis=0)
    p75 = np.percentile(sub, 75, axis=0)

    ax.plot(ks, m, color=Q_COLORS[q], marker="o", markersize=5, lw=1.5)
    ax.fill_between(ks, m - s, m + s, color=Q_COLORS[q], alpha=0.18)
    ax.fill_between(ks, p25, p75, color=Q_COLORS[q], alpha=0.25)

    # Individual sample traces (thinned)
    for i in idx[::max(1, len(idx) // 30)]:
        ax.plot(ks, errors[i, :, 0], color=Q_COLORS[q], lw=0.3, alpha=0.3)

    ax.set_title(f"{q}  (n={len(idx)})", fontsize=9, fontweight="bold",
                 color=Q_COLORS[q])
    tick_step = 2 if K > 10 else 1
    ax.set_xticks(ks[::tick_step])
    ax.set_xlabel("k (seeds)", fontsize=8)
    if qi == 0:
        ax.set_ylabel(f"Relative error vs K={K}", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.tick_params(labelsize=7)

    # Annotation: mean at k=1 and k=K
    ax.annotate(f"k=1: {m[0]:.3f}", xy=(1, m[0]), xytext=(1.4, m[0]*1.05),
                fontsize=6.5, color=Q_COLORS[q])
    ax.annotate(f"k={K}: {m[-1]:.3f}", xy=(K, m[-1]),
                xytext=(K - K*0.18, m[-1] + max(m)*0.12),
                fontsize=6.5, color=Q_COLORS[q])

fig2.suptitle(
    "Seed convergence by g_axial quintile — root length channel  "
    f"(N={N} total, K={K})",
    y=1.02, fontsize=10, fontweight="bold"
)

out_q = os.path.join(FIG_DIR, f"seed_convergence_quintile{FIG_SUFFIX}.png")
plt.savefig(out_q, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_q}")
plt.close()

print("\nDone.")
