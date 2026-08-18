"""
plot_fig2_root_minirhizotron.py

Figure 2 — Root system architecture and minirhizotron observation principle.

  (a) 3D root system — Julia CairoMakie PNG rendered by render_fig2a_makie.jl
      (plants 4 & 20, config B2, day 84, true params).
  (b) Station FOV images — FovRenderer pixel-accurate PNGs, sky-facing azimuth (az=0).

Pre-requisites (run first):
    julia publication/scripts/render_fig2a_makie.jl
    julia publication/scripts/render_fov_images.jl

Run from project root:
    python3 publication/scripts/plot_fig2_root_minirhizotron.py
Output: publication/figures/fig2_root_minirhizotron.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FIG_OUT = os.path.join(ROOT, "publication", "figures", "fig2_root_minirhizotron.png")
SIM_DIR = os.path.join(ROOT, "output_fig2_tmp")
FOV_DIR = os.path.join(SIM_DIR, "fov_images")
PLT3D   = os.path.join(SIM_DIR, "plot3d", "root_day_084_2plants.png")

STATIONS = [5, 9, 13]
DAYS     = [56, 70, 84]   # every 14 days from day 56

def station_depth(sta_idx):
    """Vertical depth (cm) of station centre."""
    along = 2.5 + (sta_idx - 1) * 5.0
    return along / np.sqrt(2)

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 7.0))
gs  = fig.add_gridspec(1, 2, left=0.02, right=0.98,
                       top=0.86, bottom=0.04, wspace=0.28,
                       width_ratios=[1, 1])

ax_3d  = fig.add_subplot(gs[0, 0])
gs_mr  = gs[0, 1].subgridspec(len(STATIONS), len(DAYS), wspace=0.005, hspace=0.06)
axes_mr = [[fig.add_subplot(gs_mr[r, c]) for c in range(len(DAYS))]
            for r in range(len(STATIONS))]

# ── Panel (a): Julia CairoMakie 3D PNG ────────────────────────────────────────

if not os.path.exists(PLT3D):
    raise FileNotFoundError(
        f"3D plot PNG not found: {PLT3D}\n"
        "Run: julia publication/scripts/render_fig2a_makie.jl"
    )

img3d = mpimg.imread(PLT3D)
# Auto-crop white margins so the 3D plot fills the panel
_mask = np.any(img3d[:, :, :3] < 0.97, axis=2)
_rows = np.where(_mask.any(axis=1))[0]
_cols = np.where(_mask.any(axis=0))[0]
if len(_rows) and len(_cols):
    _pad = 15
    img3d = img3d[
        max(0, _rows[0] - _pad) : min(img3d.shape[0], _rows[-1] + _pad),
        max(0, _cols[0] - _pad) : min(img3d.shape[1], _cols[-1] + _pad),
    ]
ax_3d.imshow(img3d)
ax_3d.axis("off")


# ── Panel (b): FOV station images from FovRenderer ────────────────────────────

# Azimuth 0 (0° = sky-facing / up-side) has by far the most root pixels
# across all stations and days: 8440 px vs 1956/1964/7767 for az 1/2/3.
BEST_AZ = 0

def load_fov_image(sta_idx, day, az=BEST_AZ):
    """Load a single azimuth FOV image."""
    fname = f"d{day:03d}_st{sta_idx:02d}_az{az}.png"
    fpath = os.path.join(FOV_DIR, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"FOV image not found: {fpath}\n"
            "Run: julia publication/scripts/render_fov_images.jl"
        )
    return np.array(Image.open(fpath).convert("L"))

for r, sta_idx in enumerate(STATIONS):
    depth = station_depth(sta_idx)
    for c, day in enumerate(DAYS):
        ax = axes_mr[r][c]
        composite = load_fov_image(sta_idx, day)

        ax.imshow(composite, cmap="gray", vmin=0, vmax=255,
                  origin="lower", aspect="equal",
                  interpolation="nearest")
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#555555")
            sp.set_linewidth(0.6)

        if r == 0:
            ax.set_title(f"Day {day}", fontsize=12, fontweight="bold", pad=5)
        if c == 0:
            ax.set_ylabel(f"Station {sta_idx}\n({depth:.0f} cm depth)",
                          fontsize=10, labelpad=6, rotation=0,
                          ha="right", va="center")

# Axis-orientation label on bottom-left panel only
ax_bl = axes_mr[-1][0]
ax_bl.annotate("", xy=(0.55, 0.08), xytext=(0.05, 0.08),
               xycoords="axes fraction",
               arrowprops=dict(arrowstyle="->", color="orange", lw=1.2))
ax_bl.text(0.05, 0.15, "along tube", transform=ax_bl.transAxes,
           fontsize=8, color="orange", va="bottom")

# Both panel subtitles in figure coordinates so they share the same y height
FIG_Y = 0.975
fig.text(0.02, FIG_Y,
         "(a)  Root system & minirhizotron tube\n"
         "      Day 84, config B, nearest 10 plants are visualized",
         fontsize=11, fontweight="bold", va="top",
         transform=fig.transFigure)
fig.text(0.52, FIG_Y,
         "(b)  Minirhizotron station windows (5×5 cm, sky-facing azimuth)",
         fontsize=11, fontweight="bold", va="top",
         transform=fig.transFigure)

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
