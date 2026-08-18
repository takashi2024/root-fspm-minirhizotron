# plot_tube_layout_1tube_200cm.jl
#
# Visualises the single-tube minirhizotron layout for the 200 cm wide box.
# Box: 200 cm wide (x) × 200 cm deep into soil (y) × 100 cm tall (z).
# Plants: 12 cols × 9 rows, 25 cm bare border on all sides.
# Single tube at 45°, entering at x=100 cm (centre), y=72 cm, z=0.
# Tube scans y≈74..129 cm, centred at y≈100 cm (plot centre).
#
# Outputs PNG files in calibration/diagnostics_mr/:
#   tube_layout_1tube_200cm_plan_section.png
#   tube_layout_1tube_200cm_front.png
#
# Run from repo root:
#   julia calibration/plot_tube_layout_1tube_200cm.jl

import Plots
using Printf

const W  = 200.0
const D  = 200.0
const H  = 100.0

const TUBE_ORIGIN_Y = 72.0
const TUBE_ORIGIN_Z = 0.0
const TUBE_DIR_Y    = 0.7071067811865476
const TUBE_DIR_Z    = -0.7071067811865476
const TUBE_LENGTH   = 80.0
const TUBE_RADIUS   = 3.0
const SKIN_DEPTH    = 1.0

const TUBE_X_CM   = [100.0]
const TUBE_COLORS = [:firebrick]
const TUBE_LABELS = ["Tube (x=100 cm)"]

const STATION_POS = 2.5:5.0:72.5

# 12 cols × 9 rows, 25 cm bare border
const ROW_X = [31.25, 43.75, 56.25, 68.75, 81.25, 93.75,
               106.25, 118.75, 131.25, 143.75, 156.25, 168.75]
const COL_Y = [33.0, 49.0, 65.0, 81.0, 97.0, 113.0, 129.0, 145.0, 161.0]

function make_plan_panel()
    p = Plots.plot(;
        xlabel = "x — cross-row direction (cm)",
        ylabel = "y — along-row / into soil (cm)",
        title  = "Top-down plan view  (z=0 surface, looking down)",
        xlims  = (-5, W + 5),
        ylims  = (-5, D + 5),
        aspect_ratio = :equal,
        legend = :topright,
        legend_background_color = :transparent,
        size   = (700, 720)
    )

    # Box boundary
    Plots.plot!(p, [0, W, W, 0, 0], [0, 0, D, D, 0];
        color=:black, linewidth=2, label=false)

    # 25 cm border zone (shaded)
    border = 25.0
    Plots.plot!(p, [border, W-border, W-border, border, border],
                   [border, border, D-border, D-border, border];
        color=:lightgreen, linewidth=1, linestyle=:dash, label="25 cm border (no plants)")

    # Column marker lines
    for rx in ROW_X
        Plots.plot!(p, [rx, rx], [0, D];
            color=:lightgray, linestyle=:dash, linewidth=1, label=false)
    end

    # Plant positions
    px = [rx for rx in ROW_X for _ in COL_Y]
    py = [cy for _  in ROW_X for cy in COL_Y]
    Plots.scatter!(p, px, py;
        marker=:circle, ms=4, color=:gray, markerstrokewidth=0,
        label="Plants ($(length(px)))")

    for (tx, col, lab) in zip(TUBE_X_CM, TUBE_COLORS, TUBE_LABELS)
        y0 = TUBE_ORIGIN_Y
        y1 = TUBE_ORIGIN_Y + TUBE_LENGTH * TUBE_DIR_Y

        Plots.plot!(p, [tx, tx], [y0, y1];
            color=col, linewidth=3, label=lab)
        Plots.plot!(p, [tx - TUBE_RADIUS, tx - TUBE_RADIUS], [y0, y1];
            color=col, linewidth=0.8, linestyle=:dot, label=false)
        Plots.plot!(p, [tx + TUBE_RADIUS, tx + TUBE_RADIUS], [y0, y1];
            color=col, linewidth=0.8, linestyle=:dot, label=false)
        Plots.scatter!(p, [tx], [y0];
            marker=:dtriangle, ms=9, color=col, markerstrokewidth=0, label=false)

        for t in STATION_POS
            Plots.scatter!(p, [tx], [TUBE_ORIGIN_Y + t * TUBE_DIR_Y];
                marker=:hline, ms=5, color=col, markerstrokewidth=1.5, label=false)
        end

        left_row  = maximum(rx for rx in ROW_X if rx < tx)
        right_row = minimum(rx for rx in ROW_X if rx > tx)
        cl_left  = tx - TUBE_RADIUS - left_row
        cl_right = right_row - (tx + TUBE_RADIUS)
        z_ann = y1 + 6
        Plots.plot!(p, [left_row, tx - TUBE_RADIUS], [z_ann, z_ann];
            color=:black, linewidth=1, label=false)
        Plots.annotate!(p, (left_row + tx - TUBE_RADIUS) / 2, z_ann + 5,
            Plots.text(@sprintf("%.1f cm", cl_left), 8, :center))
        Plots.plot!(p, [tx + TUBE_RADIUS, right_row], [z_ann, z_ann];
            color=:black, linewidth=1, label=false)
        Plots.annotate!(p, (tx + TUBE_RADIUS + right_row) / 2, z_ann + 5,
            Plots.text(@sprintf("%.1f cm", cl_right), 8, :center))
    end

    return p
end

function make_section_panel()
    p = Plots.plot(;
        xlabel = "y — along-row / into soil (cm)",
        ylabel = "z — depth below surface (cm)",
        title  = "Longitudinal section (y-z plane)",
        xlims  = (-5, D + 5),
        ylims  = (-H - 5, 5),
        legend = :bottomleft,
        legend_background_color = :transparent,
        size   = (700, 600)
    )

    Plots.plot!(p, [0, D, D, 0, 0], [0, 0, -H, -H, 0];
        color=:black, linewidth=2, label=false)
    Plots.plot!(p, [0, D], [0, 0];
        color=:saddlebrown, linewidth=1.5, linestyle=:dash, label="Soil surface (z=0)")

    # 25 cm border zones
    for y_border in [25.0, D - 25.0]
        Plots.plot!(p, [y_border, y_border], [0, -H];
            color=:lightgreen, linewidth=1, linestyle=:dash, label=false)
    end

    for cy in COL_Y
        Plots.plot!(p, [cy, cy], [0, -8];
            color=:green, linewidth=1, label=false)
    end
    Plots.scatter!(p, COL_Y, zeros(length(COL_Y));
        marker=:circle, ms=5, color=:green, markerstrokewidth=0,
        label="Plant rows (projected)")

    y0 = TUBE_ORIGIN_Y; z0 = TUBE_ORIGIN_Z
    y1 = y0 + TUBE_LENGTH * TUBE_DIR_Y
    z1 = z0 + TUBE_LENGTH * TUBE_DIR_Z
    Plots.plot!(p, [y0, y1], [z0, z1];
        color=:black, linewidth=3, label="Tube centreline (45°)")

    perp_y =  TUBE_DIR_Z
    perp_z = -TUBE_DIR_Y
    for sgn in (-1, 1), r in [TUBE_RADIUS, TUBE_RADIUS + SKIN_DEPTH]
        Plots.plot!(p,
            [y0 + sgn*r*perp_y, y1 + sgn*r*perp_y],
            [z0 + sgn*r*perp_z, z1 + sgn*r*perp_z];
            color=:black, linewidth=0.7, linestyle=:dot, label=false)
    end

    n_st = length(collect(STATION_POS))
    station_colors = Plots.cgrad(:viridis, n_st)
    for (si, t) in enumerate(STATION_POS)
        yt = y0 + t * TUBE_DIR_Y
        zt = z0 + t * TUBE_DIR_Z
        Plots.scatter!(p, [yt], [zt];
            marker=:circle, ms=5, color=station_colors[si],
            markerstrokewidth=0, label=false)
        if si in (1, 5, 10, 15)
            Plots.annotate!(p, yt + 2.5, zt,
                Plots.text(@sprintf("s%d\n%.0f cm", si, -zt), 7, :left))
        end
    end

    arc_r = 12.0
    θs = range(270, stop=315, length=50) .* (π / 180)
    Plots.plot!(p, y0 .+ arc_r .* cos.(θs), z0 .+ arc_r .* sin.(θs);
        color=:black, linewidth=1, label=false)
    Plots.annotate!(p, y0 + 14, -7, Plots.text("45°", 9, :left))

    return p
end

function make_front_panel()
    p = Plots.plot(;
        xlabel = "x — cross-row direction (cm)",
        ylabel = "z — depth below surface (cm)",
        title  = "Front view (x-z), tube projected",
        xlims  = (-5, W + 5),
        ylims  = (-H - 5, 5),
        legend = :topright,
        legend_background_color = :transparent,
        size   = (700, 500)
    )

    Plots.plot!(p, [0, W, W, 0, 0], [0, 0, -H, -H, 0];
        color=:black, linewidth=2, label=false)
    Plots.plot!(p, [0, W], [0, 0];
        color=:saddlebrown, linewidth=1.5, linestyle=:dash, label=false)

    # 25 cm border zones
    for x_border in [25.0, W - 25.0]
        Plots.plot!(p, [x_border, x_border], [0, -H];
            color=:lightgreen, linewidth=1, linestyle=:dash, label=false)
    end

    for (ri, rx) in enumerate(ROW_X)
        Plots.plot!(p, [rx, rx], [0, -H];
            color=:lightgray, linestyle=:dash, linewidth=1,
            label=(ri == 1 ? "Plant rows" : false))
    end
    for (ri, rx) in enumerate(ROW_X)
        ri % 2 == 1 || continue   # label every other row to avoid crowding
        Plots.annotate!(p, rx, -H - 3,
            Plots.text(@sprintf("%.0f", rx), 6, :center, :gray))
    end

    z_end = TUBE_LENGTH * TUBE_DIR_Z
    for (tx, col, lab) in zip(TUBE_X_CM, TUBE_COLORS, TUBE_LABELS)
        ri = TUBE_RADIUS; ro = ri + SKIN_DEPTH
        Plots.plot!(p, [tx-ro, tx+ro, tx+ro, tx-ro, tx-ro],
                       [0, 0, z_end, z_end, 0];
            color=col, linewidth=0.7, linestyle=:dot, label=false)
        Plots.plot!(p, [tx-ri, tx+ri, tx+ri, tx-ri, tx-ri],
                       [0, 0, z_end, z_end, 0];
            color=col, linewidth=1.5, linestyle=:dash,
            fillrange=z_end, fillalpha=0.12, label=false)
        Plots.plot!(p, [tx, tx], [0, z_end];
            color=col, linewidth=3, label=lab)
        Plots.scatter!(p, [tx], [0];
            marker=:dtriangle, ms=9, color=col, markerstrokewidth=0, label=false)
        for t in STATION_POS
            Plots.scatter!(p, [tx], [t * TUBE_DIR_Z];
                marker=:hline, ms=5, color=col, markerstrokewidth=1.5, label=false)
        end
        Plots.annotate!(p, tx + 5, z_end + 1,
            Plots.text(@sprintf("%.0f cm depth", -z_end), 9, col, :left))

        left_row  = maximum(rx for rx in ROW_X if rx < tx)
        right_row = minimum(rx for rx in ROW_X if rx > tx)
        cl = tx - ri - left_row
        z_brk = -8.0
        Plots.plot!(p, [left_row, tx - ri], [z_brk, z_brk];
            color=:black, linewidth=1, label=false)
        Plots.annotate!(p, (left_row + tx - ri) / 2, z_brk - 5,
            Plots.text(@sprintf("%.1f cm", cl), 8, :center))
    end

    return p
end

function main()
    diagdir = joinpath(@__DIR__, "diagnostics_mr")
    mkpath(diagdir)
    Plots.gr()

    p1 = make_plan_panel()
    p2 = make_section_panel()
    fig1 = Plots.plot(p1, p2; layout=(1, 2), size=(1400, 720),
        plot_title="Single-tube minirhizotron layout (200 cm box) — plan view and longitudinal section")
    out1 = joinpath(diagdir, "tube_layout_1tube_200cm_plan_section.png")
    Plots.savefig(fig1, out1)
    @info "Saved" path=out1

    p3  = make_front_panel()
    out2 = joinpath(diagdir, "tube_layout_1tube_200cm_front.png")
    Plots.savefig(p3, out2)
    @info "Saved" path=out2

    tx = TUBE_X_CM[1]
    left_row  = maximum(rx for rx in ROW_X if rx < tx)
    right_row = minimum(rx for rx in ROW_X if rx > tx)
    println("\nLayout summary (single tube, 200 cm box)")
    println("  Box:  $(Int(W)) × $(Int(D)) × $(Int(H)) cm  (W × D × H)")
    println("  Plants: $(length(ROW_X)) cols × $(length(COL_Y)) rows = $(length(ROW_X)*length(COL_Y)) plants")
    println("  Plant x range: $(ROW_X[1])–$(ROW_X[end]) cm  ($(minimum(ROW_X)) cm from walls)")
    println("  Plant y range: $(COL_Y[1])–$(COL_Y[end]) cm")
    println("  Tube: x=$(tx) cm, y-origin=$(TUBE_ORIGIN_Y) cm")
    y_end = TUBE_ORIGIN_Y + TUBE_LENGTH * TUBE_DIR_Y
    println("  Tube scans y=$(TUBE_ORIGIN_Y)..$(round(y_end,digits=1)) cm, " *
            "centre y≈$(round((TUBE_ORIGIN_Y+y_end)/2,digits=1)) cm")
    @printf("  Left clearance:  %.2f cm  (row at x=%.2f)\n", tx - TUBE_RADIUS - left_row,  left_row)
    @printf("  Right clearance: %.2f cm  (row at x=%.2f)\n", right_row - (tx + TUBE_RADIUS), right_row)
end

main()
