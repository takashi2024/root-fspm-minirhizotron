"""
plot_figS1_tube_geometry.py

Figure S1 — Minirhizotron tube geometry.
Layout: 1 row × 2 columns.
  (a) Longitudinal section (y–z plane): tube at 45°, observation stations
  (b) Azimuth cross-section: four imaging windows (0°/90°/180°/270°)

Run from project root:
  python3 publication/scripts/plot_figS1_tube_geometry.py
Output: publication/figures/figS1_tube_geometry.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc

ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FIG_OUT = os.path.join(ROOT, "publication", "figures", "figS1_tube_geometry.png")
SBI_CFG = os.path.join(ROOT, "sbi", "configs")

TUBE_C = "#CC2222"
SURF_C = "#5A8C5A"
BOX_BG = "#F7F5F0"

def load_env(cfg):
    with open(os.path.join(SBI_CFG, f"Parameter_SBI_env_{cfg}.json")) as f:
        return json.load(f)

env_a  = load_env("A")
tube_a = env_a["minirhizotron"]
orig   = np.array(tube_a["tube_origin_cm"])
dirv   = np.array(tube_a["tube_direction"])
tlen   = tube_a["tube_length_cm"]
r_tube = tube_a.get("tube_outer_radius_cm", 3.0)
stations_a = tube_a["station_positions_cm"]

# ── Figure layout ─────────────────────────────────────────────────────────────

fig, (ax_long, ax_azim) = plt.subplots(1, 2, figsize=(14, 7),
                                        gridspec_kw={"width_ratios": [1.6, 1],
                                                     "wspace": 0.25})
fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.10)

# ── (a) Longitudinal section ──────────────────────────────────────────────────

perp_yz_norm = np.array([dirv[2], -dirv[1]]) / np.sqrt(dirv[2]**2 + dirv[1]**2)
cy0, cz0 = orig[1], orig[2]
cy1 = cy0 + tlen * dirv[1]
cz1 = cz0 + tlen * dirv[2]

ax_long.set_aspect("equal")
ax_long.set_xlim(-3, 103)
ax_long.set_ylim(-103, 5)

ax_long.add_patch(mpatches.FancyBboxPatch((0, -100), 100, 100,
    boxstyle="square,pad=0", linewidth=1.2,
    edgecolor="#555555", facecolor=BOX_BG))

ax_long.axhline(0, color=SURF_C, lw=1.0, ls="-", alpha=0.7, zorder=2)
ax_long.text(101, 0.5, "surface", fontsize=14, color=SURF_C, va="bottom")

ax_long.plot([cy0, cy1], [cz0, cz1], color=TUBE_C, lw=2.5, zorder=4)
for sign in [+1, -1]:
    ax_long.plot([cy0 + sign*r_tube*perp_yz_norm[0],
                  cy1 + sign*r_tube*perp_yz_norm[0]],
                 [cz0 + sign*r_tube*perp_yz_norm[1],
                  cz1 + sign*r_tube*perp_yz_norm[1]],
                 color=TUBE_C, lw=1.0, ls="--", alpha=0.45, zorder=3)

ax_long.add_patch(Arc((cy0, cz0), width=20, height=20, angle=0,
                      theta1=270, theta2=315, color="#666666", lw=1.2))
ax_long.text(cy0 - 6, -12, "45°", fontsize=15, color="#555555", ha="center")

for s_cm in stations_a:
    pty = cy0 + s_cm * dirv[1]
    ptz = cz0 + s_cm * dirv[2]
    ax_long.plot([pty - 2.2*perp_yz_norm[0], pty + 2.2*perp_yz_norm[0]],
                 [ptz - 2.2*perp_yz_norm[1], ptz + 2.2*perp_yz_norm[1]],
                 color=TUBE_C, lw=1.1, zorder=5)

STN_OFFSETS = {"s2": (9.0, 1.5), "s5": (3.5, 0), "s9": (3.5, 0), "s13": (3.5, 0)}
for s_cm, lbl in [(7.5, "s2"), (22.5, "s5"), (42.5, "s9"), (62.5, "s13")]:
    pty = cy0 + s_cm * dirv[1]
    ptz = cz0 + s_cm * dirv[2]
    dx, dz = STN_OFFSETS[lbl]
    ax_long.text(pty + dx, ptz + dz, f"{lbl} ({abs(ptz):.0f} cm)",
                 fontsize=14, color="#555555", va="center")

y_positions = sorted(set(p["pos_cm"][1] for p in env_a["planting"]["plants"]))
for yp in y_positions:
    ax_long.plot(yp, 0, "o", ms=6, color=SURF_C, zorder=3,
                 markeredgewidth=0.5, markeredgecolor="#224422")

ax_long.set_xlabel("Along-row direction — y (cm)", fontsize=18, labelpad=5)
ax_long.set_ylabel("Depth — z (cm)", fontsize=18, labelpad=5)
ax_long.tick_params(labelsize=16)
ax_long.set_xticks([0, 25, 50, 75, 100])
ax_long.set_yticks([0, -25, -50, -75, -100])
ax_long.set_yticklabels(["0", "25", "50", "75", "100"])
ax_long.spines["top"].set_visible(False)
ax_long.spines["right"].set_visible(False)
ax_long.legend(handles=[
    plt.Line2D([0],[0], color=TUBE_C, lw=2.2, label="Tube centreline (45°)"),
    plt.Line2D([0],[0], color=TUBE_C, lw=1.0, ls="--", alpha=0.6, label="Tube walls (±3 cm)"),
    plt.Line2D([0],[0], color=SURF_C, marker="o", ms=6, lw=0, label="Plant rows (projected)"),
], fontsize=15, loc="lower right", framealpha=0.9)
# title placed below via fig.text at a shared y level

# ── (b) Azimuth cross-section ─────────────────────────────────────────────────

R_INNER  = 3.0
SKIN     = 1.0
R_OUTER  = R_INNER + SKIN
WIN_HALF = 47.7

AZ_WINS = [
    (90,  "0°\n(up/sky side)",      "#8888CC", "#4444AA"),
    (0,   "90°\n(right)",           "#F4A460", "#996622"),
    (270, "180°\n(down/soil side)", "#AA66BB", "#773388"),
    (180, "270°\n(left)",           "#55BBBB", "#227777"),
]

ax_azim.set_aspect("equal")
pad = 3.5
ax_azim.set_xlim(-(R_OUTER + pad), R_OUTER + pad)
ax_azim.set_ylim(-(R_OUTER + pad), R_OUTER + pad)
ax_azim.set_facecolor("white")
ax_azim.spines["top"].set_visible(False)
ax_azim.spines["right"].set_visible(False)
ax_azim.tick_params(labelsize=15)
ax_azim.set_xlabel("Cross-section x'  (cm)", fontsize=17)
ax_azim.set_ylabel("Cross-section z'  (cm, up = sky)", fontsize=17)
ax_azim.grid(True, lw=0.35, alpha=0.4, color="0.75")

theta_full = np.linspace(0, 2 * np.pi, 360)

for mpl_deg, lbl, fc, tc in AZ_WINS:
    t1, t2 = mpl_deg - WIN_HALF, mpl_deg + WIN_HALF
    theta  = np.linspace(np.radians(t1), np.radians(t2), 120)
    px = np.concatenate([R_INNER * np.cos(theta),
                         R_OUTER * np.cos(theta[::-1])])
    py = np.concatenate([R_INNER * np.sin(theta),
                         R_OUTER * np.sin(theta[::-1])])
    ax_azim.fill(px, py, color=fc, alpha=0.55, zorder=2)
    ax_azim.plot(np.append(px, px[0]),
                 np.append(py, py[0]),
                 color=fc, lw=0.6, zorder=3)
    mid_r   = (R_INNER + R_OUTER) / 2
    mid_rad = np.radians(mpl_deg)
    ax_azim.text(mid_r * np.cos(mid_rad), mid_r * np.sin(mid_rad),
                 lbl.split("\n")[0], ha="center", va="center",
                 fontsize=13, color=tc, fontweight="bold", zorder=5)

ax_azim.plot(R_OUTER * np.cos(theta_full), R_OUTER * np.sin(theta_full),
             color="#555555", lw=1.2, zorder=4)
ax_azim.plot(R_INNER * np.cos(theta_full), R_INNER * np.sin(theta_full),
             color="#555555", lw=1.2, zorder=4)
ax_azim.fill(R_INNER * np.cos(theta_full), R_INNER * np.sin(theta_full),
             color="white", zorder=3)
ax_azim.plot(0, 0, "+", ms=7, color="#555555", lw=1.2, zorder=5)

AZ_TEXT = {
    90:  ("0°\n(sky)",    "center", "bottom"),
    0:   ("90°\n(right)", "left",   "center"),
    270: ("180°\n(soil)", "center", "top"),
    180: ("270°\n(left)", "right",  "center"),
}
text_r = R_OUTER + 1.1
for mpl_deg, lbl, fc, tc in AZ_WINS:
    mid_rad = np.radians(mpl_deg)
    ha, va = AZ_TEXT[mpl_deg][1], AZ_TEXT[mpl_deg][2]
    ax_azim.text(text_r * np.cos(mid_rad), text_r * np.sin(mid_rad),
                 AZ_TEXT[mpl_deg][0], ha=ha, va=va,
                 fontsize=15, color=tc, fontweight="bold", zorder=6)
# title placed below via fig.text at a shared y level

# ── Shared-height titles (align to the taller axis top) ──────────────────────

fig.canvas.draw()
pos_a = ax_long.get_position()
pos_b = ax_azim.get_position()
title_y = max(pos_a.y1, pos_b.y1) + 0.012

fig.text(pos_a.x0, title_y,
         "(a)  Tube geometry — longitudinal section",
         fontsize=17, fontweight="bold", va="bottom", ha="left",
         transform=fig.transFigure)
fig.text(pos_b.x0, title_y,
         "(b)  Tube cross-section — azimuth windows",
         fontsize=17, fontweight="bold", va="bottom", ha="left",
         transform=fig.transFigure)

# ── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {FIG_OUT}")
