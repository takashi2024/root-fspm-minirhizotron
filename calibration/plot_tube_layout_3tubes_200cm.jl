# plot_tube_layout_3tubes_200cm.jl
#
# Visualises the 3-tube minirhizotron layout for the 200 cm wide box.
# Box: 200 cm wide (x) × 200 cm deep into soil (y) × 100 cm tall (z).
# Plants: 12 cols × 9 rows, 25 cm bare border on all sides.
# Tubes at 45°, entering at x=50, 100, 150 cm, y=72 cm, z=0.
# Each tube scans y≈74..129 cm, centred at y≈100 cm (plot centre).
#
# Outputs PNG files in calibration/diagnostics_mr/:
#   tube_layout_3tubes_200cm_plan_section.png
#   tube_layout_3tubes_200cm_front.png
#
# Run from repo root:
#   julia calibration/plot_tube_layout_3tubes_200cm.jl

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

const TUBE_X_CM    = [75.0, 100.0, 125.0]
const TUBE_COLORS  = [:dodgerblue, :firebrick, :forestgreen]
const TUBE_LABELS  = ["Tube 1 (x=75 cm)", "Tube 2 (x=100 cm)", "Tube 3 (x=125 cm)"]

const STATION_POS = 2.5:5.0:72.5

const ROW_X = [31.25, 43.75, 56.25, 68.75, 81.25, 93.75,
               106.25, 118.75, 131.25, 143.75, 156.25, 168.75]
const COL_Y = [33.0, 49.0, 65.0, 81.0, 97.0, 113.0, 129.0, 145.0, 161.0]

tube_point(tx, t) = (tx, TUBE_ORIGIN_Y + t * TUBE_DIR_Y, TUBE_ORIGIN_Z + t * TUBE_DIR_Z)

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

    Plots.plot!(p, [0, W, W, 0, 0], [0, 0, D, D, 0];
        color=:black, linewidth=2, label=false)

    # 25 cm border zone
    border = 25.0
    Plots.plot!(p, [border, W-border, W-border, border, border],
                   [border, border, D-border, D-border, border];
        color=:lightgreen, linewidth=1.2, linestyle=:dash, label="25 cm border (no plants)")

    for rx in ROW_X
        Plots.plot!(p, [rx, rx], [0, D];
            color=:lightgray, linestyle=:dash, linewidth=1, label=false)
    end

    px = [rx for rx in ROW_X for _ in COL_Y]
    py = [cy for _  in ROW_X for cy in COL_Y]
    Plots.scatter!(p, px, py;
        marker=:circle, ms=4, color=:gray, markerstrokewidth=0,
        label="Plants ($(length(px)))")

    for (tid, tx, col, lab) in zip(1:3, TUBE_X_CM, TUBE_COLORS, TUBE_LABELS)
        y0 = TUBE_ORIGIN_Y
        y1 = TUBE_ORIGIN_Y + TUBE_LENGTH * TUBE_DIR_Y

        Plots.plot!(p, [tx, tx], [y0, y1];
            color=col, linewidth=2.5, label=lab)
        Plots.plot!(p, [tx - TUBE_RADIUS, tx - TUBE_RADIUS], [y0, y1];
            color=col, linewidth=0.8, linestyle=:dot, label=false)
        Plots.plot!(p, [tx + TUBE_RADIUS, tx + TUBE_RADIUS], [y0, y1];
            color=col, linewidth=0.8, linestyle=:dot, label=false)
        Plots.scatter!(p, [tx], [y0];
            marker=:dtriangle, ms=8, color=col, markerstrokewidth=0, label=false)
        for t in STATION_POS
            ys = TUBE_ORIGIN_Y + t * TUBE_DIR_Y
            Plots.scatter!(p, [tx], [ys];
                marker=:hline, ms=5, color=col, markerstrokewidth=1.5, label=false)
        end

        # Clearance label on tube 1 only
        if tid == 1
            left_row  = maximum(rx for rx in ROW_X if rx < tx)
            cl = tx - TUBE_RADIUS - left_row
            Plots.annotate!(p, tx - 8, y1 + 8,
                Plots.text(@sprintf("%.1f cm\nclearance", cl), 7, col, :center))
        end
    end

    return p
end

function make_section_panel()
    p = Plots.plot(;
        xlabel = "y — along-row / into soil (cm)",
        ylabel = "z — depth below surface (cm)",
        title  = "Longitudinal section (y-z plane, same for all 3 tubes)",
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
        oy = y0 + sgn*r*perp_y;  oz = z0 + sgn*r*perp_z
        ey = y1 + sgn*r*perp_y;  ez = z1 + sgn*r*perp_z
        Plots.plot!(p, [oy, ey], [oz, ez];
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
        title  = "Front view (x-z), all 3 tubes projected",
        xlims  = (-5, W + 5),
        ylims  = (-H - 5, 5),
        legend = :topright,
        legend_background_color = :transparent,
        size   = (750, 500)
    )

    Plots.plot!(p, [0, W, W, 0, 0], [0, 0, -H, -H, 0];
        color=:black, linewidth=2, label=false)
    Plots.plot!(p, [0, W], [0, 0];
        color=:saddlebrown, linewidth=1.5, linestyle=:dash, label=false)

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
        ri % 2 == 1 || continue
        Plots.annotate!(p, rx, -H - 3,
            Plots.text(@sprintf("%.0f", rx), 6, :center, :gray))
    end

    z_end = TUBE_LENGTH * TUBE_DIR_Z
    for (tid, tx, col, lab) in zip(1:3, TUBE_X_CM, TUBE_COLORS, TUBE_LABELS)
        Plots.plot!(p,
            [tx-TUBE_RADIUS, tx+TUBE_RADIUS, tx+TUBE_RADIUS, tx-TUBE_RADIUS, tx-TUBE_RADIUS],
            [0, 0, z_end, z_end, 0];
            color=col, linewidth=1.5, linestyle=:dash,
            fillrange=z_end, fillalpha=0.1, label=false)
        ri = TUBE_RADIUS; ro = ri + SKIN_DEPTH
        Plots.plot!(p, [tx-ro, tx+ro, tx+ro, tx-ro, tx-ro],
                       [0, 0, z_end, z_end, 0];
            color=col, linewidth=0.7, linestyle=:dot, label=false)
        Plots.plot!(p, [tx, tx], [0, z_end];
            color=col, linewidth=2.5, label=lab)
        Plots.scatter!(p, [tx], [0];
            marker=:dtriangle, ms=8, color=col, markerstrokewidth=0, label=false)
        for t in STATION_POS
            Plots.scatter!(p, [tx], [t * TUBE_DIR_Z];
                marker=:hline, ms=5, color=col, markerstrokewidth=1.5, label=false)
        end
        Plots.annotate!(p, tx + 4, z_end + 1,
            Plots.text(@sprintf("%.0f cm", -z_end), 8, col, :left))
    end

    # Clearance bracket for Tube 1
    tx1    = TUBE_X_CM[1]
    left_r = maximum(rx for rx in ROW_X if rx < tx1)
    cl     = tx1 - TUBE_RADIUS - left_r
    z_ann  = -8.0
    Plots.plot!(p, [left_r, tx1 - TUBE_RADIUS], [z_ann, z_ann];
        color=:black, linewidth=1, label=false)
    Plots.annotate!(p, (left_r + tx1 - TUBE_RADIUS) / 2, z_ann - 5,
        Plots.text(@sprintf("%.1f cm", cl), 7, :center))

    return p
end

function main()
    diagdir = joinpath(@__DIR__, "diagnostics_mr")
    mkpath(diagdir)
    Plots.gr()

    p1 = make_plan_panel()
    p2 = make_section_panel()
    fig1 = Plots.plot(p1, p2; layout=(1, 2), size=(1400, 720),
        plot_title="Minirhizotron 3-tube layout (200 cm box) — plan view and longitudinal section")
    out1 = joinpath(diagdir, "tube_layout_3tubes_200cm_plan_section.png")
    Plots.savefig(fig1, out1)
    @info "Saved" path=out1

    p3  = make_front_panel()
    out2 = joinpath(diagdir, "tube_layout_3tubes_200cm_front.png")
    Plots.savefig(p3, out2)
    @info "Saved" path=out2

    println("\nLayout summary (3 tubes, 200 cm box)")
    println("  Box:  $(Int(W)) × $(Int(D)) × $(Int(H)) cm  (W × D × H)")
    println("  Plants: $(length(ROW_X)) cols × $(length(COL_Y)) rows = $(length(ROW_X)*length(COL_Y)) plants")
    println("  Plant x range: $(ROW_X[1])–$(ROW_X[end]) cm  ($(minimum(ROW_X)) cm from walls)")
    println("  Plant y range: $(COL_Y[1])–$(COL_Y[end]) cm")
    y_end = TUBE_ORIGIN_Y + TUBE_LENGTH * TUBE_DIR_Y
    println("  Tubes: y-origin=$(TUBE_ORIGIN_Y) cm, scans y=$(TUBE_ORIGIN_Y)..$(round(y_end,digits=1)) cm")
    println()
    println("  x-pos  Left row  Right row  Left clearance  Right clearance  Wall dist")
    for tx in TUBE_X_CM
        lr = maximum(rx for rx in ROW_X if rx < tx; init=0.0)
        rr = minimum(rx for rx in ROW_X if rx > tx; init=W)
        cl = tx - TUBE_RADIUS - lr
        cr = rr - (tx + TUBE_RADIUS)
        wd = min(tx, W - tx)
        @printf("  %5.1f   %7.2f   %8.2f   %13.2f   %14.2f   %8.2f cm\n",
                tx, lr, rr, cl, cr, wd)
    end
end

main()
