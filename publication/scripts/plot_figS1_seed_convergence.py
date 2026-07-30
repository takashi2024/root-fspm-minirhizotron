"""
plot_figS1_seed_convergence.py

Figure S1 — Seed convergence: how many observation seeds are needed?
Two-panel layout:
  Left:  CV of k-seed mean vs k for 3 channels (root length, presence, angle)
         with P10–P90 empirical band. Reference lines at 10% and 20%.
  Right: CV by g_axial quintile (root-length channel only).

Method (from analyze_seed_convergence_variance.py):
  rSE_base[i,c] = ||s[i,c]||_F / ||mu[i,c]||_F  (relative std at k=1)
  CV at k seeds = rSE_base / sqrt(k)   (iid assumption)

Run from project root:
  python3 publication/scripts/plot_figS1_seed_convergence.py
Output: publication/figures/figS1_seed_convergence.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SBI_DATA = os.path.join(ROOT, "sbi", "data", "multi_seed")
FIG_OUT  = os.path.join(ROOT, "publication", "figures", "figS1_seed_convergence.png")

CONFIGS  = ["A", "B2", "C", "D2"]
CHANNELS = ["Root length", "Root presence", "Crossing angle"]
K        = 20
K_SHOW   = [1, 7, 14, 20]   # tick marks to annotate

# ── Load data ─────────────────────────────────────────────────────────────────

all_X, all_meta = [], []
for cfg in CONFIGS:
    p = os.path.join(SBI_DATA, f"config_{cfg}", "tensors_k20", "X_raw.npz")
    X = np.load(p)["X"]                                        # (N, K, T, ns, na, nc)
    df = pd.read_csv(os.path.join(SBI_DATA, f"config_{cfg}", "samples_filtered.csv"))
    df["config"] = cfg
    all_X.append(X)
    all_meta.append(df)

X_all  = np.concatenate(all_X, axis=0)   # (N, K, T, ns, na, nc)
df_all = pd.concat(all_meta, ignore_index=True).reset_index(drop=True)

N = X_all.shape[0]

# ── Per-sample stats ──────────────────────────────────────────────────────────

mu = X_all.mean(axis=1)                        # (N, T, ns, na, nc)
s  = X_all.std(axis=1, ddof=1)

norm_mu = np.sqrt((mu**2).mean(axis=(1, 2, 3)))    # (N, nc)
norm_s  = np.sqrt((s**2).mean(axis=(1, 2, 3)))     # (N, nc)

rSE_base = norm_s / (norm_mu + 1e-8)              # (N, nc)  — CV at k=1

ks        = np.arange(1, K + 1)
ks_extrap = np.arange(K + 1, 51)

# rSE_k[i, c, ki] = rSE_base[i,c] / sqrt(k)
rSE_k      = rSE_base[:, :, np.newaxis] / np.sqrt(ks)        # (N, nc, K)
rSE_extrap = rSE_base[:, :, np.newaxis] / np.sqrt(ks_extrap) # (N, nc, 30)

# ── g_axial quintile membership ───────────────────────────────────────────────

if "g_axial_quintile" in df_all.columns:
    quintile_col = df_all["g_axial_quintile"].values
else:
    # compute from g_axial
    q = pd.qcut(df_all["g_axial"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    quintile_col = q.values

# ── Plot ──────────────────────────────────────────────────────────────────────

CHAN_COLORS = {
    "Root length":  "#1a6faf",
    "Root presence": "#2ca02c",
    "Crossing angle": "#d62728",
}
Q_COLORS = {
    "Q1": "#1a9641", "Q2": "#a6d96a", "Q3": "#ffffbf", "Q4": "#fdae61", "Q5": "#d7191c"
}
Q_LABELS = {"Q1": "Q1 (slow)", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4", "Q5": "Q5 (fast)"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.18, wspace=0.32)

x_all_k = np.concatenate([ks, ks_extrap])

# ── Left panel: 3 channels ────────────────────────────────────────────────────

ax = axes[0]
for ci, chan in enumerate(CHANNELS):
    color = CHAN_COLORS[chan]

    # empirical percentiles at each k (observed range k=1..20)
    p10 = np.percentile(rSE_k[:, ci, :], 10, axis=0)
    p25 = np.percentile(rSE_k[:, ci, :], 25, axis=0)
    p75 = np.percentile(rSE_k[:, ci, :], 75, axis=0)
    p90 = np.percentile(rSE_k[:, ci, :], 90, axis=0)
    med = np.median(rSE_k[:, ci, :], axis=0)

    # extrapolation (dashed)
    med_ex = np.median(rSE_extrap[:, ci, :], axis=0)

    ax.fill_between(ks, p10, p90, color=color, alpha=0.10)
    ax.fill_between(ks, p25, p75, color=color, alpha=0.18)
    ax.plot(ks, med, color=color, lw=2.0, label=chan)
    ax.plot(ks_extrap, med_ex, color=color, lw=1.5, ls="--", alpha=0.6)

# reference lines
for ref, lbl in [(0.20, "20%"), (0.10, "10%")]:
    ax.axhline(ref, color="0.55", lw=0.8, ls=":", zorder=1)
    ax.text(50.5, ref, lbl, va="center", fontsize=7.5, color="0.45")

# annotate k=7 and k=14
for k_ann in [7, 14]:
    ax.axvline(k_ann, color="0.70", lw=0.7, ls="--", zorder=0)
    ax.text(k_ann + 0.4, 1.45, f"k={k_ann}", fontsize=7.5, color="0.45", va="top")

ax.set_xlim(1, 50)
ax.set_ylim(0, 1.5)
ax.set_xticks([1, 7, 14, 20, 30, 40, 50])
ax.set_xlabel("Number of observation seeds ($k$)", fontsize=10)
ax.set_ylabel("CV of $k$-seed mean  ($s_i / \\sqrt{k} \\cdot \\mu_i^{-1}$)", fontsize=9.5)
ax.set_title("Convergence by feature channel", fontsize=10, fontweight="bold", pad=5)
ax.tick_params(labelsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", lw=0.35, alpha=0.5, color="0.75")

legend_handles = [Line2D([0], [0], color=CHAN_COLORS[c], lw=2.0, label=c)
                  for c in CHANNELS]
legend_handles += [
    matplotlib.patches.Patch(facecolor="0.6", alpha=0.28, label="P25–P75 band"),
    Line2D([0], [0], color="0.5", lw=1.5, ls="--", label="Extrapolation (k>20)"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
          framealpha=0.9, handlelength=1.5)

# ── Right panel: root length CV by g_axial quintile ──────────────────────────

ax2 = axes[1]
for qi, q_label in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"]):
    mask = quintile_col == q_label
    if mask.sum() == 0:
        continue
    color = Q_COLORS[q_label]
    rSE_q = rSE_k[mask, 0, :]          # root-length channel only
    rSE_q_ex = rSE_extrap[mask, 0, :]

    med    = np.median(rSE_q, axis=0)
    med_ex = np.median(rSE_q_ex, axis=0)
    p25    = np.percentile(rSE_q, 25, axis=0)
    p75    = np.percentile(rSE_q, 75, axis=0)

    ax2.fill_between(ks, p25, p75, color=color, alpha=0.15)
    ax2.plot(ks, med, color=color, lw=2.0, label=Q_LABELS[q_label])
    ax2.plot(ks_extrap, med_ex, color=color, lw=1.5, ls="--", alpha=0.6)

for ref, lbl in [(0.20, "20%"), (0.10, "10%")]:
    ax2.axhline(ref, color="0.55", lw=0.8, ls=":", zorder=1)
    ax2.text(50.5, ref, lbl, va="center", fontsize=7.5, color="0.45")

for k_ann in [7, 14]:
    ax2.axvline(k_ann, color="0.70", lw=0.7, ls="--", zorder=0)
    ax2.text(k_ann + 0.4, 1.38, f"k={k_ann}", fontsize=7.5, color="0.45", va="top")

ax2.set_xlim(1, 50)
ax2.set_ylim(0, 1.45)
ax2.set_xticks([1, 7, 14, 20, 30, 40, 50])
ax2.set_xlabel("Number of observation seeds ($k$)", fontsize=10)
ax2.set_ylabel("CV of $k$-seed mean  (root length)", fontsize=9.5)
ax2.set_title("Root length CV by $g_{axial}$ quintile", fontsize=10, fontweight="bold", pad=5)
ax2.tick_params(labelsize=8.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", lw=0.35, alpha=0.5, color="0.75")
ax2.legend(fontsize=8, loc="upper right", framealpha=0.9, handlelength=1.5,
           title="$g_{axial}$ quintile", title_fontsize=8)

fig.suptitle(
    f"Figure S1 — Seed convergence: variance of the $k$-seed mean  "
    f"(N={N} samples × K={K} seeds)",
    fontsize=10.5, y=0.93
)

fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
