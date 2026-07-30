# LayoutDiagnostics.jl
# Generates static tube-layout diagnostic figures from simulation config.
# Enabled via "layout_diagnostics": {"enabled": true} in the parameter JSON.
# Uses Plots.jl (gr backend); guarded by _PLOTS_AVAILABLE[].

const _PLOTS_AVAILABLE = Ref(false)
try
    import Plots
    _PLOTS_AVAILABLE[] = true
catch
end

function save_layout_diagnostics(box::Rhizobox, mr::Minirhizotron,
                                  plants::Vector{Plant}, outdir::String)
    _PLOTS_AVAILABLE[] || begin
        @warn "layout_diagnostics: Plots.jl not available — skipping layout figures."
        return
    end
    mr.enabled || return

    mkpath(outdir)

    # ── Derived geometry ──────────────────────────────────────────────────────
    W = box.W; D = box.D; H = box.H

    # Tube in x_from_left coords (Config converts x_from_left → centred)
    tube_x_left = mr.origin[1] + W / 2
    tube_y0 = mr.origin[2]
    tube_z0 = mr.origin[3]
    tube_dy = mr.direction[2]
    tube_dz = mr.direction[3]
    tube_y1 = tube_y0 + mr.length_cm * tube_dy
    tube_z1 = tube_z0 + mr.length_cm * tube_dz

    # Plant base positions (first segment p0; centred x → x_from_left)
    plant_x = [pl.root.segments[1].p0[1] + W / 2 for pl in plants]
    plant_y = [pl.root.segments[1].p0[2]          for pl in plants]
    row_x   = sort(unique(round.(plant_x; digits=4)))
    col_y   = sort(unique(round.(plant_y; digits=4)))

    IMAGE_ARC_CM = mr.image_arc_cm
    R = mr.r_outer_cm; SK = mr.skin_depth_cm

    Plots.gr()

    # ── Panel 1: Top-down plan view (x-y) ────────────────────────────────────
    function make_plan_panel(; ghost::Bool=false)
        show_ghost = ghost && (box.periodic_x || box.periodic_y)
        x_lo = (show_ghost && box.periodic_x) ? -W       : -2.0
        x_hi = (show_ghost && box.periodic_x) ?  2.0 * W :  W + 2.0
        y_lo = (show_ghost && box.periodic_y) ? -D       : -2.0
        y_hi = (show_ghost && box.periodic_y) ?  2.0 * D :  D + 2.0
        panel_w = (show_ghost && box.periodic_x) ? 1400 : 600
        panel_h = (show_ghost && box.periodic_y) ? 1400 : 650

        p = Plots.plot(;
            xlabel="x — cross-row direction (cm)",
            ylabel="y — along-row / into soil (cm)",
            title="Top-down plan view  (z=0 surface, looking down)",
            xlims=(x_lo, x_hi), ylims=(y_lo, y_hi),
            aspect_ratio=:equal, legend=:topright,
            legend_background_color=:transparent, size=(panel_w, panel_h))

        # ── Ghost PBC repeat panels (draw first so main box is on top) ────────
        if show_ghost
            x_shifts = box.periodic_x ? (-W, 0.0, W) : (0.0,)
            y_shifts = box.periodic_y ? (-D, 0.0, D) : (0.0,)
            for sx in x_shifts, sy in y_shifts
                (sx == 0.0 && sy == 0.0) && continue   # skip main box
                Plots.plot!(p, [sx, sx+W, sx+W, sx, sx],
                               [sy, sy, sy+D, sy+D, sy];
                    color=:gray, linewidth=1.0, linestyle=:dash, label=false)
                for rx in row_x
                    Plots.plot!(p, [rx+sx, rx+sx], [sy, sy+D];
                        color=:lightgray, linestyle=:dash, linewidth=0.6, label=false)
                end
                Plots.scatter!(p, plant_x .+ sx, plant_y .+ sy;
                    marker=:circle, ms=3, color=:gray, markerstrokewidth=0,
                    alpha=0.35, label=false)
                Plots.plot!(p, [tube_x_left+sx, tube_x_left+sx], [tube_y0+sy, tube_y1+sy];
                    color=:firebrick, linewidth=1.5, linestyle=:dot, alpha=0.35, label=false)
            end
            # PBC boundary lines
            if box.periodic_x
                for xb in (0.0, W)
                    Plots.plot!(p, [xb, xb], [-D, 2*D];
                        color=:dodgerblue, linewidth=2.0, linestyle=:dash,
                        label=(xb == 0.0 ? "Periodic x-boundary (PBC)" : false))
                end
            end
            if box.periodic_y
                for yb in (0.0, D)
                    Plots.plot!(p, [-W, 2*W], [yb, yb];
                        color=:darkorange, linewidth=2.0, linestyle=:dash,
                        label=(yb == 0.0 ? "Periodic y-boundary (PBC)" : false))
                end
            end
        elseif box.periodic_x || box.periodic_y
            if box.periodic_x
                for xb in (0.0, W)
                    Plots.plot!(p, [xb, xb], [0, D];
                        color=:dodgerblue, linewidth=1.5, linestyle=:dash,
                        label=(xb == 0.0 ? "Periodic x-boundary (PBC)" : false))
                end
            end
            if box.periodic_y
                for yb in (0.0, D)
                    Plots.plot!(p, [0, W], [yb, yb];
                        color=:darkorange, linewidth=1.5, linestyle=:dash,
                        label=(yb == 0.0 ? "Periodic y-boundary (PBC)" : false))
                end
            end
        end

        # ── Main box ──────────────────────────────────────────────────────────
        Plots.plot!(p, [0, W, W, 0, 0], [0, 0, D, D, 0];
            color=:black, linewidth=2, label=false)

        if !box.periodic_x
            border = 0.0
            if !isempty(row_x)
                border = min(row_x[1], W - row_x[end])
                if border > 0.5
                    Plots.plot!(p,
                        [border, W-border, W-border, border, border],
                        [border, border, D-border, D-border, border];
                        color=:lightgreen, linewidth=1, linestyle=:dash,
                        label=@sprintf("%.0f cm border (no plants)", border))
                end
            end
        end

        for rx in row_x
            Plots.plot!(p, [rx, rx], [0, D];
                color=:lightgray, linestyle=:dash, linewidth=1, label=false)
        end
        Plots.scatter!(p, plant_x, plant_y;
            marker=:circle, ms=4, color=:gray, markerstrokewidth=0,
            label="Plants ($(length(plants)))")

        Plots.plot!(p, [tube_x_left, tube_x_left], [tube_y0, tube_y1];
            color=:firebrick, linewidth=3, label=@sprintf("Tube (x=%.0f cm)", tube_x_left))
        for sgn in (-1, 1)
            Plots.plot!(p, [tube_x_left + sgn*R, tube_x_left + sgn*R], [tube_y0, tube_y1];
                color=:firebrick, linewidth=0.8, linestyle=:dot, label=false)
        end
        Plots.scatter!(p, [tube_x_left], [tube_y0];
            marker=:dtriangle, ms=9, color=:firebrick, markerstrokewidth=0, label=false)
        for t in mr.station_positions_cm
            Plots.scatter!(p, [tube_x_left], [tube_y0 + t * tube_dy];
                marker=:hline, ms=5, color=:firebrick, markerstrokewidth=1.5, label=false)
        end

        # Clearance brackets
        left_rows  = filter(rx -> rx < tube_x_left - R, row_x)
        right_rows = filter(rx -> rx > tube_x_left + R, row_x)
        if !isempty(left_rows) && !isempty(right_rows)
            lr = maximum(left_rows); rr = minimum(right_rows)
            z_ann = tube_y1 + 4
            Plots.plot!(p, [lr, tube_x_left - R], [z_ann, z_ann];
                color=:black, linewidth=1, label=false)
            Plots.annotate!(p, (lr + tube_x_left - R) / 2, z_ann + 3,
                Plots.text(@sprintf("%.1f cm", tube_x_left - R - lr), 8, :center))
            Plots.plot!(p, [tube_x_left + R, rr], [z_ann, z_ann];
                color=:black, linewidth=1, label=false)
            Plots.annotate!(p, (tube_x_left + R + rr) / 2, z_ann + 3,
                Plots.text(@sprintf("%.1f cm", rr - (tube_x_left + R)), 8, :center))
        end
        return p
    end

    # ── Panel 2: Longitudinal section (y-z) ──────────────────────────────────
    function make_section_panel()
        p = Plots.plot(;
            xlabel="y — along-row / into soil (cm)",
            ylabel="z — depth below surface (cm)",
            title="Longitudinal section (y-z plane)",
            xlims=(-5, D + 5), ylims=(-H - 5, 5),
            legend=:bottomleft, legend_background_color=:transparent, size=(650, 600))

        Plots.plot!(p, [0, D, D, 0, 0], [0, 0, -H, -H, 0];
            color=:black, linewidth=2, label=false)
        Plots.plot!(p, [0, D], [0, 0];
            color=:saddlebrown, linewidth=1.5, linestyle=:dash, label="Soil surface (z=0)")

        for cy in col_y
            Plots.plot!(p, [cy, cy], [0, -8]; color=:green, linewidth=1, label=false)
        end
        Plots.scatter!(p, collect(col_y), zeros(length(col_y));
            marker=:circle, ms=5, color=:green, markerstrokewidth=0,
            label="Plant rows (projected)")

        Plots.plot!(p, [tube_y0, tube_y1], [tube_z0, tube_z1];
            color=:black, linewidth=3, label=@sprintf("Tube centreline (%.0f°)", abs(atand(tube_dz, tube_dy))))

        perp_y = tube_dz; perp_z = -tube_dy
        for sgn in (-1, 1), r in [R, R + SK]
            Plots.plot!(p,
                [tube_y0 + sgn*r*perp_y, tube_y1 + sgn*r*perp_y],
                [tube_z0 + sgn*r*perp_z, tube_z1 + sgn*r*perp_z];
                color=:black, linewidth=0.7, linestyle=:dot, label=false)
        end

        n_st = length(mr.station_positions_cm)
        station_colors = Plots.cgrad(:viridis, n_st)
        for (si, t) in enumerate(mr.station_positions_cm)
            yt = tube_y0 + t * tube_dy; zt = tube_z0 + t * tube_dz
            Plots.scatter!(p, [yt], [zt];
                marker=:circle, ms=5, color=station_colors[si],
                markerstrokewidth=0, label=false)
            if si in (1, 5, 10, 15)
                Plots.annotate!(p, yt + 2.5, zt,
                    Plots.text(@sprintf("s%d\n%.0f cm", si, -zt), 7, :left))
            end
        end

        arc_r = min(10.0, D * 0.12)
        angle_deg = atand(-tube_dz, tube_dy)
        θs = range(270, stop=270 + angle_deg, length=50) .* (π / 180)
        Plots.plot!(p, tube_y0 .+ arc_r .* cos.(θs), tube_z0 .+ arc_r .* sin.(θs);
            color=:black, linewidth=1, label=false)
        Plots.annotate!(p, tube_y0 + arc_r + 2, -arc_r * 0.4,
            Plots.text(@sprintf("%.0f°", angle_deg), 9, :left))
        return p
    end

    # ── Panel 3: Front view (x-z) ─────────────────────────────────────────────
    function make_front_panel(; ghost::Bool=false)
        show_ghost = ghost && box.periodic_x
        x_lo = show_ghost ? -W       : -2.0
        x_hi = show_ghost ?  2.0 * W :  W + 2.0

        p = Plots.plot(;
            xlabel="x — cross-row direction (cm)",
            ylabel="z — depth below surface (cm)",
            title="Front view (x-z), tube projected",
            xlims=(x_lo, x_hi), ylims=(-H - 5, 5),
            legend=:topright, legend_background_color=:transparent, size=(600, 550))

        # ── Ghost PBC repeat panels ───────────────────────────────────────────
        if show_ghost
            for shift in (-W, W)
                Plots.plot!(p, [shift, shift+W, shift+W, shift, shift],
                               [0, 0, -H, -H, 0];
                    color=:gray, linewidth=1.0, linestyle=:dash, label=false)
                for rx in row_x
                    Plots.plot!(p, [rx+shift, rx+shift], [0, -H];
                        color=:lightgray, linestyle=:dash, linewidth=0.6, label=false)
                end
                Plots.plot!(p, [tube_x_left+shift, tube_x_left+shift], [tube_z0, tube_z1];
                    color=:firebrick, linewidth=1.5, linestyle=:dot, alpha=0.35, label=false)
            end
            for xb in (0.0, W)
                Plots.plot!(p, [xb, xb], [0, -H];
                    color=:dodgerblue, linewidth=2.0, linestyle=:dash,
                    label=(xb == 0.0 ? "Periodic x-boundary" : false))
            end
        elseif box.periodic_x
            for xb in (0.0, W)
                Plots.plot!(p, [xb, xb], [0, -H];
                    color=:dodgerblue, linewidth=1.5, linestyle=:dash,
                    label=(xb == 0.0 ? "Periodic x-boundary" : false))
            end
        end

        # ── Main box ──────────────────────────────────────────────────────────
        Plots.plot!(p, [0, W, W, 0, 0], [0, 0, -H, -H, 0];
            color=:black, linewidth=2, label=false)
        Plots.plot!(p, [0, W], [0, 0];
            color=:saddlebrown, linewidth=1.5, linestyle=:dash, label=false)

        for (ri, rx) in enumerate(row_x)
            Plots.plot!(p, [rx, rx], [0, -H];
                color=:lightgray, linestyle=:dash, linewidth=1,
                label=(ri == 1 ? "Plant rows" : false))
        end
        for rx in row_x
            Plots.annotate!(p, rx, -H - 3,
                Plots.text(@sprintf("%.0f", rx), 7, :center, :gray))
        end

        ro = R + SK
        Plots.plot!(p,
            [tube_x_left-ro, tube_x_left+ro, tube_x_left+ro, tube_x_left-ro, tube_x_left-ro],
            [tube_z0, tube_z0, tube_z1, tube_z1, tube_z0];
            color=:firebrick, linewidth=0.7, linestyle=:dot, label=false)
        Plots.plot!(p,
            [tube_x_left-R, tube_x_left+R, tube_x_left+R, tube_x_left-R, tube_x_left-R],
            [tube_z0, tube_z0, tube_z1, tube_z1, tube_z0];
            color=:firebrick, linewidth=1.5, linestyle=:dash,
            fillrange=tube_z1, fillalpha=0.12, label=false)
        Plots.plot!(p, [tube_x_left, tube_x_left], [tube_z0, tube_z1];
            color=:firebrick, linewidth=3, label=@sprintf("Tube (x=%.0f cm)", tube_x_left))
        Plots.scatter!(p, [tube_x_left], [tube_z0];
            marker=:dtriangle, ms=9, color=:firebrick, markerstrokewidth=0, label=false)
        for t in mr.station_positions_cm
            Plots.scatter!(p, [tube_x_left], [tube_z0 + t * tube_dz];
                marker=:hline, ms=5, color=:firebrick, markerstrokewidth=1.5, label=false)
        end
        Plots.annotate!(p, tube_x_left + 4, tube_z1 + 1,
            Plots.text(@sprintf("%.0f cm depth", -tube_z1), 9, :firebrick, :left))

        left_rows  = filter(rx -> rx < tube_x_left - R, row_x)
        right_rows = filter(rx -> rx > tube_x_left + R, row_x)
        if !isempty(left_rows) && !isempty(right_rows)
            lr = maximum(left_rows)
            z_brk = tube_z0 - 8.0
            Plots.plot!(p, [lr, tube_x_left - R], [z_brk, z_brk];
                color=:black, linewidth=1, label=false)
            Plots.annotate!(p, (lr + tube_x_left - R) / 2, z_brk - 4,
                Plots.text(@sprintf("%.1f cm", tube_x_left - R - lr), 8, :center))
        end
        return p
    end

    # ── Panel 4: Azimuth cross-section ────────────────────────────────────────
    function make_azimuth_panel()
        isempty(mr.camera_azimuths_deg) && return nothing
        half_arc_deg = (IMAGE_ARC_CM / (2π * R)) * 180.0
        az_colors = [:royalblue, :darkorange, :purple, :teal, :green, :brown]
        az_labels = ["0° (up/sky side)", "90° (right)", "180° (down/soil side)",
                     "270° (left)", "45°", "135°"]

        p = Plots.plot(;
            xlabel="cross-section x'  (cm)",
            ylabel="cross-section z'  (cm, up = sky side)",
            title="Tube cross-section — azimuth windows\n" *
                  @sprintf("(r_outer=%.0f cm, skin=%.0f cm, arc window ±%.1f°)",
                           R, SK, half_arc_deg),
            aspect_ratio=:equal,
            xlims=(-(R + SK + 1.5), R + SK + 1.5),
            ylims=(-(R + SK + 1.5), R + SK + 1.5),
            legend=:outertopright, legend_background_color=:transparent, size=(550, 580))

        θ_ring = range(0, 2π, length=200)
        for r in [R, R + SK]
            Plots.plot!(p, r .* cos.(θ_ring), r .* sin.(θ_ring);
                color=:black, linewidth=(r == R ? 1.5 : 0.8),
                linestyle=(r == R ? :solid : :dash), label=false)
        end

        for (i, az) in enumerate(mr.camera_azimuths_deg)
            col = az_colors[mod1(i, length(az_colors))]
            lab = i <= length(az_labels) ? az_labels[i] : @sprintf("%.0f°", az)
            az_rad = az * π / 180.0; half_rad = half_arc_deg * π / 180.0
            θ_start = π/2 - az_rad - half_rad; θ_end = π/2 - az_rad + half_rad
            θs = range(θ_start, stop=θ_end, length=60); θs_rev = reverse(θs)
            xs = vcat(R .* cos.(θs), (R + SK) .* cos.(θs_rev))
            zs = vcat(R .* sin.(θs), (R + SK) .* sin.(θs_rev))
            Plots.plot!(p, [xs; xs[1]], [zs; zs[1]];
                color=col, linewidth=1.5,
                fillrange=0, fillalpha=0.35, fillcolor=col,
                label="$lab (±$(round(Int, half_arc_deg))°)")
            cx = (R + SK + 0.8) * cos(π/2 - az_rad)
            cz = (R + SK + 0.8) * sin(π/2 - az_rad)
            Plots.annotate!(p, cx, cz, Plots.text("$(round(Int, az))°", 9, col, :center))
        end

        Plots.annotate!(p, 0, -(R + SK + 1.0), Plots.text("▼ soil", 8, :gray, :center))
        Plots.annotate!(p, 0,  (R + SK + 1.0), Plots.text("▲ sky",  8, :gray, :center))
        overlap = 2 * half_arc_deg - 90.0
        if overlap > 0
            Plots.annotate!(p, 0, -(R + SK + 1.7),
                Plots.text(@sprintf("Adjacent windows overlap by %.1f°", overlap), 7, :gray, :center))
        end
        return p
    end

    # ── Save figures ──────────────────────────────────────────────────────────
    box_label = @sprintf("%.0f×%.0f%s box", W, D, box.periodic_x ? " PBC" : "")

    # Without ghost panels
    p1 = make_plan_panel(ghost=false)
    p2 = make_section_panel()
    fig1 = Plots.plot(p1, p2; layout=(1, 2), size=(1300, 680),
        plot_title="Minirhizotron layout ($box_label) — plan view and longitudinal section")
    Plots.savefig(fig1, joinpath(outdir, "layout_plan_section.png"))

    p3 = make_front_panel(ghost=false)
    Plots.savefig(p3, joinpath(outdir, "layout_front.png"))

    # With ghost PBC repeat panels (only when periodic_x)
    if box.periodic_x || box.periodic_y
        p1g = make_plan_panel(ghost=true)
        fig_w = (box.periodic_x && box.periodic_y) ? 2800 : (box.periodic_x ? 2200 : 1300)
        fig_h = box.periodic_y ? 1000 : 680
        fig1g = Plots.plot(p1g, p2; layout=(1, 2), size=(fig_w, fig_h),
            plot_title="Minirhizotron layout ($box_label) — plan view with PBC repeats and longitudinal section")
        Plots.savefig(fig1g, joinpath(outdir, "layout_plan_section_pbc_repeat.png"))

        p3g = make_front_panel(ghost=true)
        Plots.savefig(p3g, joinpath(outdir, "layout_front_pbc_repeat.png"))
    end

    p4 = make_azimuth_panel()
    if p4 !== nothing
        Plots.savefig(p4, joinpath(outdir, "layout_azimuth_crosssection.png"))
    end

    @info "Layout diagnostics saved." outdir=outdir
end
