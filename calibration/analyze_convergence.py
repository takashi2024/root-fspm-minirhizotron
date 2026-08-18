"""
analyze_convergence.py

Parses CMA-ES iteration logs for all 20 test case calibrations and assesses
whether 150 iterations is sufficient for convergence.

For each case, extracts the J-vs-iteration curve and computes:
  - J at iter 50, 75, 100, 125, 150
  - % improvement from iter 100→150 (key convergence indicator)
  - % improvement from iter 125→150
  - Final sigma (small sigma = converged)
  - Whether J was still declining at the end

Output:
  calibration/convergence_analysis.csv
  calibration/convergence_analysis.txt (human-readable)
"""

import csv
import math
import os
import re

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(CALIB_DIR, "logs", "test_cases")
TC_CSV    = os.path.join(CALIB_DIR, "test_cases_selected.csv")
OUT_CSV   = os.path.join(CALIB_DIR, "convergence_analysis.csv")
OUT_TXT   = os.path.join(CALIB_DIR, "convergence_analysis.txt")

# ── Load test case metadata ────────────────────────────────────────────────────
with open(TC_CSV, newline="") as f:
    tc_meta = {(r["config"], int(r["sample_id"])): r for r in csv.DictReader(f)}

# ── Parse one log file ─────────────────────────────────────────────────────────
# CMA-ES table row: "   N   fevals   J   sigma   axis_ratio   time"
ROW_RE = re.compile(r'^\s+(\d+)\s+(\d+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)')

def parse_log(path):
    """Returns list of (iter, J, sigma) tuples."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            m = ROW_RE.match(line)
            if m:
                it    = int(m.group(1))
                J     = float(m.group(3))
                sigma = float(m.group(4))
                # Skip penalty rows (J = 5e17)
                if J < 1e10:
                    rows.append((it, J, sigma))
    return rows

def best_J_at(rows, target_iter):
    """Best (minimum) J seen up to and including target_iter."""
    best = math.inf
    for it, J, _ in rows:
        if it <= target_iter:
            best = min(best, J)
    return best if best < math.inf else math.nan

def pct_improvement(j_from, j_to):
    if math.isnan(j_from) or math.isnan(j_to) or j_from < 1e-12:
        return math.nan
    return (j_from - j_to) / j_from * 100.0

# ── Analyse all cases ──────────────────────────────────────────────────────────

results = []
configs = sorted(os.listdir(os.path.join(CALIB_DIR, "configs", "test_cases")))

for cfg_file in configs:
    tag = cfg_file.replace(".json", "")
    # Parse config tag: tc_<config>_<sample_id>_s2
    parts = tag.split("_")
    # handle B2/D2 configs
    if parts[1] in ("B2", "D2"):
        config    = parts[1]
        sample_id = int(parts[2])
    else:
        config    = parts[1]
        sample_id = int(parts[2])

    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    rows = parse_log(log_path)

    if not rows:
        print(f"  WARNING: no iteration data found in {tag}.log")
        continue

    j50  = best_J_at(rows, 50)
    j75  = best_J_at(rows, 75)
    j100 = best_J_at(rows, 100)
    j125 = best_J_at(rows, 125)
    j150 = best_J_at(rows, 150)

    final_iter, final_J, final_sigma = rows[-1]

    impr_100_150 = pct_improvement(j100, j150)
    impr_125_150 = pct_improvement(j125, j150)
    impr_50_100  = pct_improvement(j50,  j100)

    # Was J still declining in last 10 iters? (compare last 10 vs prior 10)
    last10  = [J for it, J, _ in rows if it > final_iter - 10]
    prior10 = [J for it, J, _ in rows if final_iter - 20 < it <= final_iter - 10]
    still_declining = (min(last10) < min(prior10)) if last10 and prior10 else False

    meta = tc_meta.get((config, sample_id), {})
    quintile = meta.get("g_axial_quintile", "?")
    g_axial  = float(meta.get("g_axial", 0))

    results.append({
        "tag":           tag,
        "config":        config,
        "sample_id":     sample_id,
        "g_axial_quintile": quintile,
        "g_axial":       g_axial,
        "n_iters":       final_iter,
        "J_50":          j50,
        "J_75":          j75,
        "J_100":         j100,
        "J_125":         j125,
        "J_150":         j150,
        "final_J":       final_J,
        "final_sigma":   final_sigma,
        "impr_50_100_pct":  impr_50_100,
        "impr_100_150_pct": impr_100_150,
        "impr_125_150_pct": impr_125_150,
        "still_declining":  still_declining,
    })

    print(f"  {tag:22s}  Q{quintile[-1]}  "
          f"J@100={j100:.4f}  J@150={j150:.4f}  "
          f"impr100→150={impr_100_150:+.1f}%  "
          f"sigma={final_sigma:.2e}  "
          f"{'STILL DECLINING' if still_declining else 'plateaued'}")

# ── Write CSV ──────────────────────────────────────────────────────────────────
if results:
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows → {OUT_CSV}")

# ── Summary report ─────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 78)
lines.append("CMA-ES Convergence Analysis — 20 Test Cases (Scenario 2, maxiter=150)")
lines.append("=" * 78)
lines.append("")
lines.append(f"  {'Tag':<22}  Q  {'J@50':>7}  {'J@100':>7}  {'J@125':>7}  {'J@150':>7}  "
             f"{'↓100→150':>9}  {'↓125→150':>9}  {'sigma':>8}  {'status'}")
lines.append("-" * 110)

n_still   = 0
n_plateau = 0
for r in results:
    impr1 = f"{r['impr_100_150_pct']:+.1f}%" if not math.isnan(r['impr_100_150_pct']) else "  —"
    impr2 = f"{r['impr_125_150_pct']:+.1f}%" if not math.isnan(r['impr_125_150_pct']) else "  —"
    status = "DECLINING" if r["still_declining"] else "plateau"
    if r["still_declining"]:
        n_still += 1
    else:
        n_plateau += 1
    lines.append(
        f"  {r['tag']:<22}  {r['g_axial_quintile']:>2}  "
        f"{r['J_50']:>7.4f}  {r['J_100']:>7.4f}  {r['J_125']:>7.4f}  {r['J_150']:>7.4f}  "
        f"{impr1:>9}  {impr2:>9}  {r['final_sigma']:>8.2e}  {status}"
    )

lines.append("-" * 110)
lines.append("")

# Aggregate stats
valid = [r for r in results if not math.isnan(r["impr_100_150_pct"])]
if valid:
    mean_impr = sum(r["impr_100_150_pct"] for r in valid) / len(valid)
    max_impr  = max(r["impr_100_150_pct"] for r in valid)
    min_impr  = min(r["impr_100_150_pct"] for r in valid)
    lines.append(f"Improvement iter 100→150 (across {len(valid)} cases):")
    lines.append(f"  Mean: {mean_impr:+.1f}%   Max: {max_impr:+.1f}%   Min: {min_impr:+.1f}%")
    lines.append(f"  Still declining at iter 150: {n_still} / {len(results)} cases")
    lines.append(f"  Apparently plateaued:        {n_plateau} / {len(results)} cases")
    lines.append("")

valid2 = [r for r in results if not math.isnan(r["impr_125_150_pct"])]
if valid2:
    mean2 = sum(r["impr_125_150_pct"] for r in valid2) / len(valid2)
    lines.append(f"Improvement iter 125→150 (last 25 iters, across {len(valid2)} cases):")
    lines.append(f"  Mean: {mean2:+.1f}%")
    lines.append("")

lines.append("Verdict:")
if n_still > len(results) / 2:
    lines.append("  INSUFFICIENT — majority of cases still declining at iter 150.")
    lines.append("  Recommend maxiter=200-250 for future runs.")
elif mean_impr > 3.0:
    lines.append(f"  BORDERLINE — mean {mean_impr:.1f}% improvement in final 50 iters.")
    lines.append("  Some cases benefit from more iterations; consider maxiter=200.")
else:
    lines.append(f"  ADEQUATE — mean {mean_impr:.1f}% improvement in final 50 iters.")
    lines.append("  150 iterations appears sufficient for most cases.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n--- Convergence Summary ---")
print("\n".join(lines))
print(f"\nWrote → {OUT_TXT}")
