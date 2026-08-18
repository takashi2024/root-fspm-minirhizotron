# plot_tube_layout_1tube.jl
#
# Visualises the single-tube minirhizotron layout relative to the plant grid.
# Box: 100 cm wide (x) × 100 cm deep into soil (y) × 150 cm tall (z).
# Plants: 8 rows × 6 columns, conventional rapeseed (12.5 cm row spacing, 16 cm within-row).
# Single tube at 45°, entering at x=50 cm, y=20 cm, z=0.
#
# Outputs three PNG files in calibration/diagnostics_mr/:
#   tube_layout_1tube_plan_section.png
#   tube_layout_1tube_front.png
#   tube_layout_1tube_azimuth_crosssection.png
#
# Run from repo root:
#   julia calibration/plot_tube_layout_1tube.jl

import Plots
using Printf

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
const W  = 100.0
const D  = 100.0
const H  = 150.0

const TUBE_ORIGIN_Y = 20.0
const TUBE_ORIGIN_Z = 0.0
const TUBE_DIR_Y    = 0.7071067811865476
const TUBE_DIR_Z    = -0.7071067811865476
const TUBE_LENGTH   = 80.0
const TUBE_RADIUS   = 3.0
const SKIN_DEPTH    = 1.0

const TUBE_X_CM   = [50.0]
const TUBE_COLORS = [:firebrick]
const TUBE_LABELS = ["Tube (x=50 cm)"]

const STATION_POS = 2.5:5.0:72.5

const ROW_X = [6.25, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 93.75]
const COL_Y = [8.0, 24.0, 40.0, 56.0, 72.0, 88.0]

const AZIMUTHS_DEG = [0.0, 90.0, 180.0, 270.0]
const AZ_COLORS    = [:royalblue, :darkorange, :purple, :teal]
const AZ_LABELS    = ["0° (up/sky side)", "90° (right)", "180° (down/soil side)", "270° (left)"]

# ──────────────────────────────────────────────────────────────────────────────
# Panel 1: Plan view (x-y, top-down)
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

    Plots.plot!(p, [0, W, W, 0, 0], [0, 0, D, D, 0];
        color=:black, linewidth=2, label=false)

    for rx in ROW_X
        Plots.plot!(p, [rx, rx], [0, D];
            color=:lightgray, linestyle=:dash, linewidth=1, label=false)
    end

    px = [rx for rx in ROW_X for _ in COL_Y]
    py = [cy for _  in ROW_X for cy in COL_Y]
    Plots.scatter!(p, px, py;
        marker=:circle, ms=5, color=:gray, markerstrokewidth=0,
        label="Plants (48)")

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
                marker=:hline, ms=6, color=col, markerstrokewidth=1.5, label=false)
        end

        left_row  = maximum(rx for rx in ROW_X if rx < tx)
        right_row = minimum(rx for rx in ROW_X if rx > tx)
        cl_left  = tx - TUBE_RADIUS - left_row
        cl_right = right_row - (tx + TUBE_RADIUS)

        # Clearance bracket — left side
        z_ann = y1 + 4
        Plots.plot!(p, [left_row, tx - TUBE_RADIUS], [z_ann, z_ann];
            color=:black, linewidth=1, label=false)
        Plots.annotate!(p, (left_row + tx - TUBE_RADIUS) / 2, z_ann + 3,
            Plots.text(@sprintf("%.1f cm", cl_left), 8, :center))
        # Clearance bracket — right side
        Plots.plot!(p, [tx + TUBE_RADIUS, right_row], [z_ann, z_ann];
            color=:black, linewidth=1, label=false)
        Plots.annotate!(p, (tx + TUBE_RADIUS + right_row) / 2, z_ann + 3,
            Plots.text(@sprintf("%.1f cm", cl_right), 8, :center))
    end

    return p
end

# ──────────────────────────────────────────────────────────────────────────────
# Panel 2: Longitudinal section (y-z)
# ──────────────────────────────────────────────────────────────────────────────
function make_section_panel()
    p = Plots.plot(;
        xlabel = "y — along-row / into soil (cm)",
        ylabel = "z — depth below surface (cm)",
        title  = "Longitudinal section (y-z plane)",
        xlims  = (-5, D + 5),
        ylims  = (-H - 5, 5),
        legend = :bottomleft,
        legend_background_color = :transparent,
        size   = (650, 600)
    )

    Plots.plot!(p, [0, D, D, 0, 0], [0, 0, -H, -H, 0];
        color=:black, linewidth=2, label=false)
    Plots.plot!(p, [0, D], [0, 0];
        color=:saddlebrown, linewidth=1.5, linestyle=:dash, label="Soil surface (z=0)")

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

# ──────────────────────────────────────────────────────────────────────────────
# Panel 3: Front view (x-z)
# ──────────────────────────────────────────────────────────────────────────────
function make_front_panel()
    p = Plots.plot(;
        xlabel = "x — cross-row direction (cm)",
        ylabel = "z — depth below surface (cm)",
        title  = "Front view (x-z), tube projected",
        xlims  = (-2, W + 2),
        ylims  = (-H - 5, 5),
        legend = :topright,
        legend_background_color = :transparent,
        size   = (600, 650)
    )

    Plots.plot!(p, [0, W, W, 0, 0], [0, 0, -H, -H, 0];
        color=:black, linewidth=2, label=false)
    Plots.plot!(p, [0, W], [0, 0];
        color=:saddlebrown, linewidth=1.5, linestyle=:dash, label=false)

    for (ri, rx) in enumerate(ROW_X)
        Plots.plot!(p, [rx, rx], [0, -H];
            color=:lightgray, linestyle=:dash, linewidth=1,
            label=(ri == 1 ? "Plant rows" : false))
    end
    for rx in ROW_X
        Plots.annotate!(p, rx, -H - 3,
            Plots.text(@sprintf("%.0f", rx), 7, :center, :gray))
    end

    z_end = TUBE_LENGTH * TUBE_DIR_Z
    for (tx, col, lab) in zip(TUBE_X_CM, TUBE_COLORS, TUBE_LABELS)
        ri = TUBE_RADIUS; ro = ri + SKIN_DEPTH

        Plots.plot!(p,
            [tx-ro, tx+ro, tx+ro, tx-ro, tx-ro],
            [0, 0, z_end, z_end, 0];
            color=col, linewidth=0.7, linestyle=:dot, label=false)
        Plots.plot!(p,
            [tx-ri, tx+ri, tx+ri, tx-ri, tx-ri],
            [0, 0, z_end, z_end, 0];
            color=col, linewidth=1.5, linestyle=:dash,
            fillrange=z_end, fillalpha=0.12, label=false)
        Plots.plot!(p, [tx, tx], [0, z_end];
            color=col, linewidth=3, label=lab)
        Plots.scatter!(p, [tx], [0];
            marker=:dtriangle, ms=9, color=col, markerstrokewidth=0, label=false)

        for t in STATION_POS
            Plots.scatter!(p, [tx], [t * TUBE_DIR_Z];
                marker=:hline, ms=6, color=col, markerstrokewidth=1.5, label=false)
        end

        Plots.annotate!(p, tx + 4, z_end + 1,
            Plots.text(@sprintf("%.0f cm depth", -z_end), 9, col, :left))

        left_row  = maximum(rx for rx in ROW_X if rx < tx)
        right_row = minimum(rx for rx in ROW_X if rx > tx)
        cl = tx - ri - left_row
        z_brk = -8.0
        Plots.plot!(p, [left_row, tx - ri], [z_brk, z_brk];
            color=:black, linewidth=1, label=false)
        Plots.annotate!(p, (left_row + tx - ri) / 2, z_brk - 4,
            Plots.text(@sprintf("%.1f cm", cl), 8, :center))
    end

    return p
end

# ──────────────────────────────────────────────────────────────────────────────
# Panel 4: Tube cross-section — azimuth windows
# Shows the 4 camera windows (0°/90°/180°/270°) around the tube annulus.
# ──────────────────────────────────────────────────────────────────────────────
function make_azimuth_panel()
    IMAGE_ARC_CM = 5.0
    half_arc_deg = (IMAGE_ARC_CM / (2π * TUBE_RADIUS)) * 180.0   # ≈ 47.75°

    p = Plots.plot(;
        xlabel = "cross-section x'  (cm)",
        ylabel = "cross-section z'  (cm, up = sky side)",
        title  = "Tube cross-section — azimuth windows\n" *
                 @sprintf("(r_outer=%.0f cm, skin=%.0f cm, arc window ±%.1f°)",
                          TUBE_RADIUS, SKIN_DEPTH, half_arc_deg),
        aspect_ratio = :equal,
        xlims = (-(TUBE_RADIUS + SKIN_DEPTH + 1.5), TUBE_RADIUS + SKIN_DEPTH + 1.5),
        ylims = (-(TUBE_RADIUS + SKIN_DEPTH + 1.5), TUBE_RADIUS + SKIN_DEPTH + 1.5),
        legend = :outertopright,
        legend_background_color = :transparent,
        size   = (550, 580)
    )

    # Outer and inner annulus rings
    θ_ring = range(0, 2π, length=200)
    for r in [TUBE_RADIUS, TUBE_RADIUS + SKIN_DEPTH]
        Plots.plot!(p, r .* cos.(θ_ring), r .* sin.(θ_ring);
            color=:black, linewidth=(r == TUBE_RADIUS ? 1.5 : 0.8),
            linestyle=(r == TUBE_RADIUS ? :solid : :dash),
            label=false)
    end

    # Azimuth windows (wedge sectors in the annulus)
    # Convention: 0° = upward (sky side), measured clockwise.
    # In the cross-section plane, 0° maps to +z' (up), 90° CW maps to +x' (right).
    # atan2 convention: angle from +z' axis, increasing CW = -atan2(x', z')
    # So azimuth = 0° → direction (x'=0, z'=+1)  → angle from +z' = 0
    #    azimuth = 90°→ direction (x'=+1, z'=0)   → angle from +z' = 90°
    for (az, col, lab) in zip(AZIMUTHS_DEG, AZ_COLORS, AZ_LABELS)
        az_rad    = az * π / 180.0
        half_rad  = half_arc_deg * π / 180.0
        θ_start   = π/2 - az_rad - half_rad   # math angle (CCW from +x)
        θ_end     = π/2 - az_rad + half_rad

        θs = range(θ_start, stop=θ_end, length=60)
        θs_rev = reverse(θs)
        xs = vcat(TUBE_RADIUS .* cos.(θs), (TUBE_RADIUS + SKIN_DEPTH) .* cos.(θs_rev))
        zs = vcat(TUBE_RADIUS .* sin.(θs), (TUBE_RADIUS + SKIN_DEPTH) .* sin.(θs_rev))

        Plots.plot!(p, [xs; xs[1]], [zs; zs[1]];
            color=col, linewidth=1.5,
            fillrange=0, fillalpha=0.35, fillcolor=col,
            label="$lab (±$(round(Int, half_arc_deg))°)")

        # Direction label
        cx = (TUBE_RADIUS + SKIN_DEPTH + 0.8) * cos(π/2 - az_rad)
        cz = (TUBE_RADIUS + SKIN_DEPTH + 0.8) * sin(π/2 - az_rad)
        Plots.annotate!(p, cx, cz, Plots.text("$(Int(az))°", 9, col, :center))
    end

    # Soil surface direction indicator
    Plots.annotate!(p, 0, -(TUBE_RADIUS + SKIN_DEPTH + 1.0),
        Plots.text("▼ soil", 8, :gray, :center))
    Plots.annotate!(p, 0,  (TUBE_RADIUS + SKIN_DEPTH + 1.0),
        Plots.text("▲ sky", 8, :gray, :center))

    # Overlap note
    overlap = 2 * half_arc_deg - 90.0
    Plots.annotate!(p, 0, -(TUBE_RADIUS + SKIN_DEPTH + 1.7),
        Plots.text(@sprintf("Adjacent windows overlap by %.1f°", overlap), 7, :gray, :center))

    return p
end

# ──────────────────────────────────────────────────────────────────────────────
# Compose and save
# ──────────────────────────────────────────────────────────────────────────────
function main()
    diagdir = joinpath(@__DIR__, "diagnostics_mr")
    mkpath(diagdir)

    Plots.gr()

    p1 = make_plan_panel()
    p2 = make_section_panel()
    fig1 = Plots.plot(p1, p2; layout=(1, 2), size=(1300, 680),
        plot_title="Single-tube minirhizotron layout — plan view and longitudinal section")
    out1 = joinpath(diagdir, "tube_layout_1tube_plan_section.png")
    Plots.savefig(fig1, out1)
    @info "Saved" path=out1

    p3  = make_front_panel()
    out2 = joinpath(diagdir, "tube_layout_1tube_front.png")
    Plots.savefig(p3, out2)
    @info "Saved" path=out2

    p4  = make_azimuth_panel()
    out3 = joinpath(diagdir, "tube_layout_1tube_azimuth_crosssection.png")
    Plots.savefig(p4, out3)
    @info "Saved" path=out3

    IMAGE_ARC_CM = 5.0
    half_arc_deg = (IMAGE_ARC_CM / (2π * TUBE_RADIUS)) * 180.0
    tx = TUBE_X_CM[1]
    left_row  = maximum(rx for rx in ROW_X if rx < tx)
    right_row = minimum(rx for rx in ROW_X if rx > tx)

    println("\nLayout summary (single tube)")
    println("  Box:  $(Int(W)) × $(Int(D)) × $(Int(H)) cm  (x width × y soil depth × z height)")
    println("  Tube: 45°, length=$(Int(TUBE_LENGTH)) cm, radius=$(TUBE_RADIUS) cm, skin=$(SKIN_DEPTH) cm")
    println("  Entry: x=$(tx) cm, y=$(TUBE_ORIGIN_Y) cm, z=0")
    println("  End:   y≈$(round(TUBE_ORIGIN_Y + TUBE_LENGTH*TUBE_DIR_Y, digits=1)) cm, " *
            "z≈$(round(TUBE_LENGTH*TUBE_DIR_Z, digits=1)) cm")
    @printf("  Left clearance:  %.2f cm  (row at x=%.2f)\n", tx - TUBE_RADIUS - left_row,  left_row)
    @printf("  Right clearance: %.2f cm  (row at x=%.2f)\n", right_row - (tx + TUBE_RADIUS), right_row)
    @printf("  Azimuth window:  ±%.2f°  (5 cm arc on %.0f cm radius)\n", half_arc_deg, TUBE_RADIUS)
    @printf("  Adjacent overlap: %.2f°\n", 2 * half_arc_deg - 90.0)
end

main()
