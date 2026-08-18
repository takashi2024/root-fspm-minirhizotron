"""
analyze_seed_convergence_variance.py

Seed convergence analysis using per-seed empirical variance.
No reference mean needed — avoids the bias of the reference-based approach.

Method
------
For each sample i, compute the sample std s_i across K=20 seeds.
The expected relative error of a k-seed mean vs the true (infinite-seed) mean is:

    rSE_i(k) = norm_F(s_i) / (sqrt(k) * norm_F(mu_i))

where norm_F is the Frobenius norm over (T, ns, na).
This is the Coefficient of Variation (CV) of the k-seed mean.

Assumptions
-----------
  - Seeds are i.i.d. (satisfied: independent random draws, same model)
  - Finite variance (satisfied: root lengths are bounded)
  - No normality required for the point estimate or quantile bands.
  - Normality IS implicitly assumed if ±1 SE is interpreted as a 68% CI.
    We instead use empirical P10/P25/P75/P90 bands (non-parametric).

Bootstrap validation
--------------------
At k = 1, 3, 5, 7, 10, 14, 20 we also compute bootstrap SE by resampling
k seeds from the K=20 pool (with replacement) 500 times per sample.
Bootstrap SE should agree with s/sqrt(k) if the iid assumption holds.

Extrapolation
-------------
The 1/sqrt(k) curve naturally extrapolates to any k without new simulations.
Dashed lines show predicted rSE for k = 21..50.

Outputs
-------
  sbi/figures/seed_convergence_variance.png         -- main 3-panel figure
  sbi/figures/seed_convergence_variance_quintile.png -- per-quintile detail (5 panels)
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

# K=20 dataset (tensors_k20 required)
TENSORS_SUBDIR = "tensors_k20"
K = 20
N_EXTRAP = 50        # extrapolate curve up to this k
BOOTSTRAP_B = 500    # bootstrap resamples per (sample, k)
BOOTSTRAP_KS = [1, 3, 5, 7, 10, 14, 20]  # k values where bootstrap is computed

RNG = np.random.default_rng(42)

# ── 1. Load data ──────────────────────────────────────────────────────────────

all_X    = []
all_meta = []

for cfg in CONFIGS:
    cfg_dir  = os.path.join(REPO_ROOT, "sbi", "data", "multi_seed", f"config_{cfg}")
    npz_path = os.path.join(cfg_dir, TENSORS_SUBDIR, "X_raw.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found. Run sbi/run_extend_multiseed_all_configs.sh first."
        )
    X  = np.load(npz_path)["X"]          # (N, K, T, ns, na, nc)
    df = pd.read_csv(os.path.join(cfg_dir, "samples_filtered.csv"))
    df["config"] = cfg
    all_X.append(X)
    all_meta.append(df)

X_all  = np.concatenate(all_X, axis=0)   # (N, K, T, ns, na, nc)
df_all = pd.concat(all_meta, ignore_index=True)

N, K_actual, T, ns, na, nc = X_all.shape
assert K_actual == K, f"Expected K={K}, got {K_actual}"
print(f"Loaded {N} samples × K={K} seeds  shape={X_all.shape}")

# ── 2. Per-sample statistics across K seeds ───────────────────────────────────

# mu: (N, T, ns, na, nc)  — sample mean over K seeds (≈ true mean estimator)
# s:  (N, T, ns, na, nc)  — sample std over K seeds (ddof=1)
mu = X_all.mean(axis=1)
s  = X_all.std(axis=1, ddof=1)

# Frobenius-norm collapsed to (N, nc):
# norm_mu[i,c] = sqrt(mean over (T,ns,na) of mu[i,:,:,:,c]^2)
norm_mu = np.sqrt((mu**2).mean(axis=(1, 2, 3)))         # (N, nc)
norm_s  = np.sqrt((s**2).mean(axis=(1, 2, 3)))          # (N, nc)

# rSE_base[i, c] = norm_s[i,c] / norm_mu[i,c]  — relative std at k=1
# rSE at k seeds: rSE_base / sqrt(k)
rSE_base = norm_s / (norm_mu + 1e-8)   # (N, nc)

# For root-length channel: absolute SE at k=1 (in cm)
norm_s_abs = np.sqrt((s[..., 0]**2).mean(axis=(1, 2, 3)))   # (N,) in cm

# ── 3. Convergence curves: parametric (s/sqrt(k)) ────────────────────────────

ks = np.arange(1, K + 1)           # k = 1..20
ks_extrap = np.arange(K + 1, N_EXTRAP + 1)   # k = 21..50

# rSE_k[i, c, k_idx] = rSE_base[i,c] / sqrt(k)
rSE_k = rSE_base[:, :, np.newaxis] / np.sqrt(ks)[np.newaxis, np.newaxis, :]
# shape: (N, nc, K)

abs_se_k = norm_s_abs[:, np.newaxis] / np.sqrt(ks)[np.newaxis, :]
# shape: (N, K)

# Extrapolation
rSE_extrap = rSE_base[:, :, np.newaxis] / np.sqrt(ks_extrap)[np.newaxis, np.newaxis, :]
abs_se_extrap = norm_s_abs[:, np.newaxis] / np.sqrt(ks_extrap)[np.newaxis, :]

# ── 4. Bootstrap validation ───────────────────────────────────────────────────

print("\nComputing bootstrap SE at selected k values...")

# X_flat[i, k, e] — flatten (T,ns,na,nc) → e, then norm per (T,ns,na) per channel
# For efficiency: collapse (T,ns,na) into a single spatial axis
X_flat = X_all.reshape(N, K, T * ns * na, nc)   # (N, K, E, nc)

# mu_flat: (N, E, nc)
mu_flat = X_flat.mean(axis=1)

# Bootstrap SE: for each sample i and k, draw B subsets of size k from K=20 seeds
# boot_rSE[i, k_idx, c] = bootstrap estimate of rSE at that k
boot_rSE = np.full((N, len(BOOTSTRAP_KS), nc), np.nan)

for ki, k_val in enumerate(BOOTSTRAP_KS):
    print(f"  Bootstrap k={k_val}...", end=" ", flush=True)
    boot_norms_s = np.zeros((N, BOOTSTRAP_B, nc))

    for b in range(BOOTSTRAP_B):
        idx = RNG.integers(0, K, size=(N, k_val))   # (N, k_val)
        # Gather seeds: boot_X[i, j, e, c] for j in idx[i]
        # Use advanced indexing
        i_idx = np.arange(N)[:, np.newaxis]         # (N, 1)
        boot_X = X_flat[i_idx, idx]                 # (N, k_val, E, nc)
        boot_mean = boot_X.mean(axis=1)             # (N, E, nc)
        diff = boot_mean - mu_flat                  # (N, E, nc)
        # Frobenius norm of diff per channel
        boot_norms_s[:, b, :] = np.sqrt((diff**2).mean(axis=1))   # (N, nc)

    # Bootstrap expected displacement = MEAN of bootstrap Frobenius norms
    # (std would measure norm stability, not expected displacement magnitude)
    boot_mean_frob = boot_norms_s.mean(axis=1)                     # (N, nc)
    norm_mu_flat = np.sqrt((mu_flat**2).mean(axis=1))              # (N, nc)
    boot_rSE[:, ki, :] = boot_mean_frob / (norm_mu_flat + 1e-8)
    print(f"mean rSE={boot_rSE[:, ki, 0].mean():.4f}  "
          f"(parametric: {rSE_base[:, 0].mean() / np.sqrt(k_val):.4f})")

# ── 5. g_axial quintiles ──────────────────────────────────────────────────────

df_all["g_axial_quintile"] = pd.qcut(
    df_all["g_axial"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
)
quintile_idx = {q: np.where(df_all["g_axial_quintile"] == q)[0]
                for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]}

# ── 6. Summary table ──────────────────────────────────────────────────────────

print("\n── Variance-based relative SE (mean ± std across N samples) ──")
print(f"{'k':>4}  {'root_len CV(%)':>16}  {'has_root CV(%)':>16}  "
      f"{'angle CV(%)':>13}  {'abs SE (cm)':>12}")
for ki, k_val in enumerate(ks):
    vals = [f"{rSE_k[:, c, ki].mean()*100:.2f} ± {rSE_k[:, c, ki].std()*100:.2f}"
            for c in range(nc)]
    print(f"{k_val:>4}  {vals[0]:>16}  {vals[1]:>16}  {vals[2]:>13}  "
          f"{abs_se_k[:, ki].mean():.4f} ± {abs_se_k[:, ki].std():.4f}")

# k needed for <10% and <5% CV (root length channel)
mean_rSE_base = rSE_base[:, 0].mean()
k_10pct = int(np.ceil((mean_rSE_base / 0.10) ** 2))
k_5pct  = int(np.ceil((mean_rSE_base / 0.05) ** 2))
print(f"\nMean base CV (root length, k=1): {mean_rSE_base*100:.1f}%")
print(f"k needed for <10% CV: {k_10pct}")
print(f"k needed for <5%  CV: {k_5pct}")

# ── 7. Plot settings ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

Q_COLORS = {"Q1": "#4393C3", "Q2": "#92C5DE", "Q3": "#B8D97C",
            "Q4": "#F4A582", "Q5": "#D6604D"}
CH_COLORS  = ["#2E6DA4", "#2CA02C", "#D95F02"]
CH_LABELS  = ["Root length (cm)", "Root presence", "Crossing angle"]
CH_MARKERS = ["o", "s", "^"]

ks_full   = np.concatenate([ks, ks_extrap])
all_ks_plot = np.arange(1, N_EXTRAP + 1)

# ── 8. Main figure ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                          gridspec_kw={"wspace": 0.38})
fig.patch.set_facecolor("white")

tick_step = 2

# ── Panel (a): relative SE by channel ────────────────────────────────────────
ax = axes[0]
for c in range(nc):
    # Parametric: mean rSE across samples at each k (observed range)
    m  = rSE_k[:, c, :].mean(axis=0)           # (K,)
    p25 = np.percentile(rSE_k[:, c, :], 25, axis=0)
    p75 = np.percentile(rSE_k[:, c, :], 75, axis=0)
    p10 = np.percentile(rSE_k[:, c, :], 10, axis=0)
    p90 = np.percentile(rSE_k[:, c, :], 90, axis=0)

    # Extrapolation
    m_ext = rSE_base[:, c].mean() / np.sqrt(ks_extrap)

    ax.plot(ks, m, color=CH_COLORS[c], marker=CH_MARKERS[c],
            markersize=4.5, lw=1.5, label=CH_LABELS[c])
    ax.fill_between(ks, p25, p75, color=CH_COLORS[c], alpha=0.18)
    ax.fill_between(ks, p10, p90, color=CH_COLORS[c], alpha=0.08)
    ax.plot(ks_extrap, m_ext, color=CH_COLORS[c], lw=1.0, ls="--", alpha=0.55)

    # Bootstrap dots
    for ki_b, k_val in enumerate(BOOTSTRAP_KS):
        bm = boot_rSE[:, ki_b, c].mean()
        ax.scatter(k_val, bm, marker="x", s=30, color=CH_COLORS[c],
                   linewidths=1.2, zorder=5)

ax.axhline(0.10, color="0.5", lw=0.7, ls=":", alpha=0.7)
ax.axhline(0.05, color="0.4", lw=0.7, ls=":", alpha=0.7)
ax.text(N_EXTRAP + 0.3, 0.10, "10%", fontsize=7, color="0.5", va="center")
ax.text(N_EXTRAP + 0.3, 0.05, "5%",  fontsize=7, color="0.4", va="center")
ax.axvline(7,  color="0.7", lw=0.7, ls="--", alpha=0.6)
ax.text(7.2, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 0.8,
        "K=7", fontsize=7, color="0.5")

ax.set_xlim(0.5, N_EXTRAP + 2)
ax.set_xticks(np.arange(2, N_EXTRAP + 1, tick_step))
ax.set_xlabel("Number of seeds (k)", fontsize=9)
ax.set_ylabel("CV of k-seed mean  (s / √k / μ)", fontsize=9)
ax.set_title("(a) Convergence by channel\n"
             "bands=P10/P25/P75/P90, ✕=bootstrap (no normality assumed)",
             loc="left", fontsize=8.5, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False)

# ── Panel (b): absolute SE for root length ────────────────────────────────────
ax = axes[1]
m    = abs_se_k.mean(axis=0)
p25  = np.percentile(abs_se_k, 25, axis=0)
p75  = np.percentile(abs_se_k, 75, axis=0)
p10  = np.percentile(abs_se_k, 10, axis=0)
p90  = np.percentile(abs_se_k, 90, axis=0)
m_ext = norm_s_abs.mean() / np.sqrt(ks_extrap)

ax.plot(ks, m, color=CH_COLORS[0], marker="o", markersize=4.5, lw=1.5)
ax.fill_between(ks, p25, p75, color=CH_COLORS[0], alpha=0.18)
ax.fill_between(ks, p10, p90, color=CH_COLORS[0], alpha=0.08)
ax.plot(ks_extrap, m_ext, color=CH_COLORS[0], lw=1.0, ls="--", alpha=0.55)
ax.axvline(7, color="0.7", lw=0.7, ls="--", alpha=0.6)

ax.set_xlim(0.5, N_EXTRAP + 2)
ax.set_xticks(np.arange(2, N_EXTRAP + 1, tick_step))
ax.set_xlabel("Number of seeds (k)", fontsize=9)
ax.set_ylabel("SE of mean root length (cm)", fontsize=9)
ax.set_title("(b) Absolute SE — root length\n"
             "Theoretical: s / √k  (bands=P10/P25/P75/P90)",
             loc="left", fontsize=8.5, fontweight="bold")
ax.axhline(0, color="0.5", lw=0.5, ls="--")

# Annotations at k=7 and k=20
for k_ann in [7, 20]:
    ki = k_ann - 1
    ax.annotate(f"k={k_ann}: {m[ki]:.3f} cm",
                xy=(k_ann, m[ki]),
                xytext=(k_ann + 1.5, m[ki] + 0.05),
                fontsize=7, color=CH_COLORS[0],
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))

# ── Panel (c): relative SE by g_axial quintile ────────────────────────────────
ax = axes[2]
for q, idx in quintile_idx.items():
    if len(idx) == 0:
        continue
    m = rSE_k[idx, 0, :].mean(axis=0)      # root length channel, mean over quintile
    m_ext = rSE_base[idx, 0].mean() / np.sqrt(ks_extrap)
    ax.plot(ks, m, color=Q_COLORS[q], marker="o", markersize=4,
            lw=1.4, label=f"{q} (n={len(idx)})")
    ax.plot(ks_extrap, m_ext, color=Q_COLORS[q], lw=0.9, ls="--", alpha=0.5)

ax.axhline(0.10, color="0.5", lw=0.7, ls=":", alpha=0.7)
ax.axvline(7, color="0.7", lw=0.7, ls="--", alpha=0.6)
ax.set_xlim(0.5, N_EXTRAP + 2)
ax.set_xticks(np.arange(2, N_EXTRAP + 1, tick_step))
ax.set_xlabel("Number of seeds (k)", fontsize=9)
ax.set_ylabel("CV of k-seed mean (root length)", fontsize=9)
ax.set_title("(c) Convergence by g_axial quintile\n"
             "(dashed = extrapolation beyond K=20)",
             loc="left", fontsize=8.5, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, title="g_axial (Q1=slow → Q5=fast)",
          title_fontsize=7)
ax.axhline(0, color="0.5", lw=0.5, ls="--")

fig.suptitle(
    f"Seed convergence — variance-based (iid, no normality assumed)\n"
    f"N={N} samples × K={K} seeds, dashed = 1/√k extrapolation to k={N_EXTRAP}",
    y=1.01, fontsize=10.5, fontweight="bold"
)

out_main = os.path.join(FIG_DIR, "seed_convergence_variance.png")
plt.savefig(out_main, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {out_main}")
plt.close()

# ── 9. Per-quintile detail figure ────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 5, figsize=(17, 4.0), sharey=True,
                            gridspec_kw={"wspace": 0.12})
fig2.patch.set_facecolor("white")

for qi, (q, idx) in enumerate(quintile_idx.items()):
    ax = axes2[qi]
    if len(idx) == 0:
        ax.set_visible(False)
        continue

    sub = rSE_k[idx, 0, :]         # (n_q, K)  — root length channel
    m   = sub.mean(axis=0)
    p25 = np.percentile(sub, 25, axis=0)
    p75 = np.percentile(sub, 75, axis=0)
    p10 = np.percentile(sub, 10, axis=0)
    p90 = np.percentile(sub, 90, axis=0)
    m_ext = rSE_base[idx, 0].mean() / np.sqrt(ks_extrap)

    # Individual sample traces (thinned)
    for i in idx[::max(1, len(idx) // 30)]:
        ax.plot(ks, rSE_k[i, 0, :], color=Q_COLORS[q], lw=0.3, alpha=0.3)

    ax.fill_between(ks, p10, p90, color=Q_COLORS[q], alpha=0.10)
    ax.fill_between(ks, p25, p75, color=Q_COLORS[q], alpha=0.22)
    ax.plot(ks, m, color=Q_COLORS[q], marker="o", markersize=4.5, lw=1.6)
    ax.plot(ks_extrap, m_ext, color=Q_COLORS[q], lw=1.0, ls="--", alpha=0.5)

    # Bootstrap validation dots
    for ki_b, k_val in enumerate(BOOTSTRAP_KS):
        bm = boot_rSE[idx, ki_b, 0].mean()
        ax.scatter(k_val, bm, marker="x", s=25, color=Q_COLORS[q],
                   linewidths=1.2, zorder=5)

    ax.axhline(0.10, color="0.5", lw=0.6, ls=":", alpha=0.7)
    ax.axvline(7, color="0.7", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(np.arange(2, N_EXTRAP + 1, tick_step))
    ax.set_xlim(0.5, N_EXTRAP + 2)
    ax.set_xlabel("k (seeds)", fontsize=8)
    if qi == 0:
        ax.set_ylabel("CV of k-seed mean (root length)", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.tick_params(labelsize=7)

    k_thresh_10 = int(np.ceil((rSE_base[idx, 0].mean() / 0.10) ** 2))
    ax.set_title(f"{q}  (n={len(idx)})\n"
                 f"k(CV<10%)≈{k_thresh_10}",
                 fontsize=8.5, fontweight="bold", color=Q_COLORS[q])

    ax.annotate(f"k=1: {m[0]:.3f}", xy=(1, m[0]),
                xytext=(1.5, m[0] * 1.05), fontsize=6.5, color=Q_COLORS[q])
    ax.annotate(f"k=20: {m[-1]:.3f}", xy=(20, m[-1]),
                xytext=(15, m[-1] + 0.03), fontsize=6.5, color=Q_COLORS[q])

fig2.suptitle(
    "Seed convergence by g_axial quintile — root length channel\n"
    "Variance-based (s/√k, iid-only); ✕=bootstrap; dashed=extrapolation; "
    "bands=P10/P25/P75/P90",
    y=1.03, fontsize=9.5, fontweight="bold"
)

out_q = os.path.join(FIG_DIR, "seed_convergence_variance_quintile.png")
plt.savefig(out_q, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_q}")
plt.close()

print("\nDone.")
print(f"\nKey result: K=7 CV = {rSE_k[:, 0, 6].mean()*100:.1f}%  "
      f"(root length; estimated vs true mean, not vs K=20)")
print(f"k needed for CV < 10%: {k_10pct}")
print(f"k needed for CV < 5%:  {k_5pct}")
