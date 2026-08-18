# plot_tube_layout_3tubes.jl
#
# Visualises the 3-tube minirhizotron layout relative to the plant grid.
# Box: 100 cm wide (x) × 100 cm deep into soil (y) × 150 cm tall (z).
# Plants: 8 rows × 6 columns, conventional rapeseed (12.5 cm row spacing, 16 cm within-row).
# Tubes at 45°, entering the soil surface (z=0) at x = 25, 50, 75 cm, y = 20 cm.
#
# Outputs two PNG files in calibration/diagnostics_mr/:
#   tube_layout_3tubes_plan_section.png  — plan view (x-y) + longitudinal section (y-z)
#   tube_layout_3tubes_front.png         — front view (x-z) showing all 3 tube trajectories
#
# Run from repo root:
#   julia calibration/plot_tube_layout_3tubes.jl

import Plots
using Printf

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
const W  = 100.0   # box width   (x, cm)
const D  = 100.0   # box depth into soil (y, cm)
const H  = 150.0   # box height  (z, cm) — z goes 0 to -H

const TUBE_ORIGIN_Y    = 20.0    # y entry position (cm from front face)
const TUBE_ORIGIN_Z    = 0.0     # z entry position (soil surface)
const TUBE_DIR_Y       = 0.7071067811865476
const TUBE_DIR_Z       = -0.7071067811865476
const TUBE_LENGTH      = 80.0    # cm
const TUBE_RADIUS      = 3.0     # outer radius, cm
const SKIN_DEPTH       = 1.0     # cm

const TUBE_X_CM        = [25.0, 50.0, 75.0]
const TUBE_COLORS      = [:dodgerblue, :firebrick, :forestgreen]
const TUBE_LABELS      = ["Tube 1 (x=25 cm)", "Tube 2 (x=50 cm)", "Tube 3 (x=75 cm)"]

const STATION_POS      = 2.5:5.0:72.5   # station midpoints along tube axis

# Plant positions (from-left x, y into soil)
const ROW_X = [6.25, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 93.75]
const COL_Y = [8.0, 24.0, 40.0, 56.0, 72.0, 88.0]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
# Point along tube at parameter t (cm from origin)
tube_point(tx, t) = (tx, TUBE_ORIGIN_Y + t * TUBE_DIR_Y, TUBE_ORIGIN_Z + t * TUBE_DIR_Z)

# ──────────────────────────────────────────────────────────────────────────────
# Panel 1: Plan view (x horizontal, y vertical — top-down)
# Shows plant grid, row positions, and tube footprints.
# ──────────────────────────────────────────────────────────────────────────────
function make_plan_panel()
    p = Plots.plot(;
        xlabel = "x — cross-row direction (cm)",
        ylabel = "y — along-row / into soil (cm)",
        title  = "Top-down plan view  (z=0 surface, looking down)",
        xlims  = (-2, W + 2),
        ylims  = (-2, D + 2),
        aspect_ratio = :equal,
        legend = :topright,
        legend_background_color = :transparent,
        size   = (600, 650)
    )

    # Box boundary
    Plots.plot!(p, [0, W, W, 0, 0], [0, 0, D, D, 0];
        color=:black, linewidth=2, label=false)

    # Row marker lines (dashed)
    for rx in ROW_X
        Plots.plot!(p, [rx, rx], [0, D];
            color=:lightgray, linestyle=:dash, linewidth=1, label=false)
    end

    # Plant positions
    px = [rx for rx in ROW_X for _ in COL_Y]
    py = [cy for _  in ROW_X for cy in COL_Y]
    Plots.scatter!(p, px, py;
        marker=:circle, ms=5, color=:gray, markerstrokewidth=0,
        label="Plants (48)")

    # Tube footprints in plan view
    for (tid, tx, col, lab) in zip(1:3, TUBE_X_CM, TUBE_COLORS, TUBE_LABELS)
        t_end = TUBE_LENGTH
        y0 = TUBE_ORIGIN_Y
        y1 = TUBE_ORIGIN_Y + t_end * TUBE_DIR_Y   # ≈ 61.6 cm

        # Tube centreline
        Plots.plot!(p, [tx, tx], [y0, y1];
            color=col, linewidth=2.5, label=lab)

        # Tube annulus band (outer radius projected to plan view)
        Plots.plot!(p, [tx - TUBE_RADIUS, tx - TUBE_RADIUS], [y0, y1];
            color=col, linewidth=0.8, linestyle=:dot, label=false)
        Plots.plot!(p, [tx + TUBE_RADIUS, tx + TUBE_RADIUS], [y0, y1];
            color=col, linewidth=0.8, linestyle=:dot, label=false)

        # Entry marker
        Plots.scatter!(p, [tx], [y0];
            marker=:dtriangle, ms=8, color=col, markerstrokewidth=0, label=false)

        # Station tick marks
        for t in STATION_POS
            ys = TUBE_ORIGIN_Y + t * TUBE_DIR_Y
            Plots.scatter!(p, [tx], [ys];
                marker=:hline, ms=6, color=col, markerstrokewidth=1.5, label=false)
        end

        # Label: clearance to nearest row
        if tid == 1
            left_row  = maximum(rx for rx in ROW_X if rx < tx)
            right_row = minimum(rx for rx in ROW_X if rx > tx)
            cl = round(tx - left_row - TUBE_RADIUS; digits=1)
            Plots.annotate!(p, tx, y1 + 3,
                Plots.text(@sprintf("%.1f cm\nclearance", cl), 7, col, :center))
        end
    end

    return p
end

# ──────────────────────────────────────────────────────────────────────────────
# Panel 2: Longitudinal section (y horizontal, z vertical — tube plane)
# All 3 tubes occupy the same y-z plane; shows 45° angle and station depths.
# ──────────────────────────────────────────────────────────────────────────────
function make_section_panel()
    p = Plots.plot(;
        xlabel = "y — along-row / into soil (cm)",
        ylabel = "z — depth below surface (cm)",
        title  = "Longitudinal section (y-z plane, same for all 3 tubes)",
        xlims  = (-5, D + 5),
        ylims  = (-H - 5, 5),
        yflip  = false,
        legend = :bottomleft,
        legend_background_color = :transparent,
        size   = (650, 600)
    )

    # Box boundary (y-z)
    Plots.plot!(p, [0, D, D, 0, 0], [0, 0, -H, -H, 0];
        color=:black, linewidth=2, label=false)

    # Soil surface reference
    Plots.plot!(p, [0, D], [0, 0];
        color=:saddlebrown, linewidth=1.5, linestyle=:dash, label="Soil surface (z=0)")

    # Plant stem stubs (projected to y-z plane)
    for cy in COL_Y
        Plots.plot!(p, [cy, cy], [0, -8];
            color=:green, linewidth=1, label=false)
    end
    Plots.scatter!(p, COL_Y, zeros(length(COL_Y));
        marker=:circle, ms=5, color=:green, markerstrokewidth=0,
        label="Plant rows (projected)")

    # Tube centreline (one representative — all tubes identical in y-z)
    y0 = TUBE_ORIGIN_Y
    z0 = TUBE_ORIGIN_Z
    y1 = y0 + TUBE_LENGTH * TUBE_DIR_Y
    z1 = z0 + TUBE_LENGTH * TUBE_DIR_Z
    Plots.plot!(p, [y0, y1], [z0, z1];
        color=:black, linewidth=3, label="Tube centreline (45°)")

    # Annulus edges (±(TUBE_RADIUS + SKIN_DEPTH) perpendicular to tube)
    perp_y =  TUBE_DIR_Z   # perpendicular to direction in y-z
    perp_z = -TUBE_DIR_Y
    for sgn in (-1, 1), r in [TUBE_RADIUS, TUBE_RADIUS + SKIN_DEPTH]
        oy = y0 + sgn * r * perp_y;  oz = z0 + sgn * r * perp_z
        ey = y1 + sgn * r * perp_y;  ez = z1 + sgn * r * perp_z
        Plots.plot!(p, [oy, ey], [oz, ez];
            color=:black, linewidth=0.7, linestyle=:dot, label=false)
    end

    # Station positions (coloured by station number — depth gradient)
    n_st = length(collect(STATION_POS))
    station_colors = Plots.cgrad(:viridis, n_st)
    for (si, t) in enumerate(STATION_POS)
        yt = y0 + t * TUBE_DIR_Y
        zt = z0 + t * TUBE_DIR_Z
        depth = -zt
        Plots.scatter!(p, [yt], [zt];
            marker=:circle, ms=5, color=station_colors[si],
            markerstrokewidth=0, label=false)
        if si in (1, 5, 10, 15)
            Plots.annotate!(p, yt + 2.5, zt,
                Plots.text(@sprintf("s%d\n%.0f cm", si, depth), 7, :left))
        end
    end

    # 45° angle annotation
    arc_r = 12.0
    θs = range(270, stop=315, length=50) .* (π / 180)
    Plots.plot!(p, y0 .+ arc_r .* cos.(θs), z0 .+ arc_r .* sin.(θs);
        color=:black, linewidth=1, label=false)
    Plots.annotate!(p, y0 + 14, -7, Plots.text("45°", 9, :left))

    return p
end

# ──────────────────────────────────────────────────────────────────────────────
# Panel 3: Front view (x horizontal, z vertical — looking along y-axis)
# ──────────────────────────────────────────────────────────────────────────────
function make_front_panel()
    p = Plots.plot(;
        xlabel = "x — cross-row direction (cm)",
        ylabel = "z — depth below surface (cm)",
        title  = "Front view (x-z), all 3 tubes projected",
        xlims  = (-2, W + 2),
        ylims  = (-H - 5, 5),
        legend = :topright,
        legend_background_color = :transparent,
        size   = (600, 650)
    )

    # Box boundary
    Plots.plot!(p, [0, W, W, 0, 0], [0, 0, -H, -H, 0];
        color=:black, linewidth=2, label=false)

    # Soil surface
    Plots.plot!(p, [0, W], [0, 0];
        color=:saddlebrown, linewidth=1.5, linestyle=:dash, label=false)

    # Plant row lines
    for (ri, rx) in enumerate(ROW_X)
        lab = ri == 1 ? "Plant rows" : false
        Plots.plot!(p, [rx, rx], [0, -H];
            color=:lightgray, linestyle=:dash, linewidth=1, label=lab)
    end

    # Row labels at bottom
    for rx in ROW_X
        Plots.annotate!(p, rx, -H - 3,
            Plots.text(@sprintf("%.0f", rx), 7, :center, :gray))
    end

    # Tube trajectories (in x-z, direction has no x component → each tube is a
    # vertical profile at constant x, from z=0 to z = -tube_length*|dir_z|)
    z_end = TUBE_LENGTH * TUBE_DIR_Z   # ≈ -56.6 cm
    for (tid, tx, col, lab) in zip(1:3, TUBE_X_CM, TUBE_COLORS, TUBE_LABELS)
        # Outer annulus in x (tube radius)
        Plots.plot!(p,
            [tx - TUBE_RADIUS, tx + TUBE_RADIUS,
             tx + TUBE_RADIUS, tx - TUBE_RADIUS,
             tx - TUBE_RADIUS],
            [0, 0, z_end, z_end, 0];
            color=col, linewidth=1.5, linestyle=:dash,
            fillrange=z_end, fillalpha=0.1, label=false)

        # Skin annulus
        ri = TUBE_RADIUS; ro = TUBE_RADIUS + SKIN_DEPTH
        Plots.plot!(p,
            [tx - ro, tx + ro, tx + ro, tx - ro, tx - ro],
            [0, 0, z_end, z_end, 0];
            color=col, linewidth=0.7, linestyle=:dot, label=false)

        # Centreline
        Plots.plot!(p, [tx, tx], [0, z_end];
            color=col, linewidth=2.5, label=lab)

        # Entry point
        Plots.scatter!(p, [tx], [0];
            marker=:dtriangle, ms=8, color=col, markerstrokewidth=0, label=false)

        # Station marks
        for t in STATION_POS
            zs = t * TUBE_DIR_Z
            Plots.scatter!(p, [tx], [zs];
                marker=:hline, ms=6, color=col, markerstrokewidth=1.5, label=false)
        end

        # Depth label at deepest station
        Plots.annotate!(p, tx + 3.5, z_end + 1,
            Plots.text(@sprintf("%.0f cm", -z_end), 8, col, :left))
    end

    # Clearance bracket for Tube 1
    tx1    = TUBE_X_CM[1]
    left_r = maximum(rx for rx in ROW_X if rx < tx1)
    cl     = tx1 - TUBE_RADIUS - left_r
    z_ann  = -5.0
    Plots.plot!(p, [left_r, tx1 - TUBE_RADIUS], [z_ann, z_ann];
        color=:black, linewidth=1, label=false)
    Plots.annotate!(p, (left_r + tx1 - TUBE_RADIUS) / 2, z_ann - 4,
        Plots.text(@sprintf("%.1f cm", cl), 7, :center))

    return p
end

# ──────────────────────────────────────────────────────────────────────────────
# Compose and save
# ──────────────────────────────────────────────────────────────────────────────
function main()
    diagdir = joinpath(@__DIR__, "diagnostics_mr")
    mkpath(diagdir)

    Plots.gr()

    # Figure 1: plan view + section
    p1 = make_plan_panel()
    p2 = make_section_panel()
    fig1 = Plots.plot(p1, p2; layout=(1, 2), size=(1300, 680),
        plot_title="Minirhizotron 3-tube layout — plan view and longitudinal section")
    out1 = joinpath(diagdir, "tube_layout_3tubes_plan_section.png")
    Plots.savefig(fig1, out1)
    @info "Saved" path=out1

    # Figure 2: front view
    p3  = make_front_panel()
    out2 = joinpath(diagdir, "tube_layout_3tubes_front.png")
    Plots.savefig(p3, out2)
    @info "Saved" path=out2

    println("\nLayout summary")
    println("  Box:  $(Int(W)) × $(Int(D)) × $(Int(H)) cm  (x width × y soil depth × z height)")
    println("  Tube: 45°, length=$(Int(TUBE_LENGTH)) cm, radius=$(TUBE_RADIUS) cm, skin=$(SKIN_DEPTH) cm")
    println("  Entry at y=$(TUBE_ORIGIN_Y) cm, z=0 (soil surface)")
    println("  Tube end: y≈$(round(TUBE_ORIGIN_Y + TUBE_LENGTH*TUBE_DIR_Y, digits=1)) cm, " *
            "z≈$(round(TUBE_LENGTH*TUBE_DIR_Z, digits=1)) cm depth")
    println()
    println("  x-pos  Left row  Right row  Left clearance  Right clearance")
    for tx in TUBE_X_CM
        lr = maximum(rx for rx in ROW_X if rx < tx; init=0.0)
        rr = minimum(rx for rx in ROW_X if rx > tx; init=W)
        cl = tx - TUBE_RADIUS - lr
        cr = rr - (tx + TUBE_RADIUS)
        @printf("  %5.1f   %7.2f   %8.2f   %13.2f   %14.2f  cm\n", tx, lr, rr, cl, cr)
    end
    println()
    println("  Station depths (cm below surface):")
    for (si, t) in enumerate(STATION_POS)
        depth = -t * TUBE_DIR_Z
        @printf("    s%02d  t=%.1f cm  → depth=%.2f cm\n", si, Float64(t), depth)
    end
end

main()
