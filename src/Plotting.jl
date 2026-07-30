# Auto-split from root_fspm_rhizobox_multiplant_SRL.jl
# File: Plotting.jl

# Draw a Minirhizotron tube as cross-section rings + 4 lengthwise lines + dashed axis.
function plot_tube!(ax, mr::Minirhizotron)
    mr.enabled || return

    d = mr.direction
    ref = abs(d[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    u = normalize_vec(cross(Vector{Float64}(d), ref))
    v = normalize_vec(cross(Vector{Float64}(d), u))

    r   = mr.r_outer_cm
    col = (:royalblue, 0.55)

    # Cross-section circles (9 evenly spaced along tube)
    n_circ = 36
    θ = range(0, 2π; length=n_circ + 1)
    for t in range(0.0, mr.length_cm; length=9)
        ctr = mr.origin .+ t .* d
        pts = [Point3f((ctr .+ r .* (cos(a) .* u .+ sin(a) .* v))...) for a in θ]
        lines!(ax, pts; color=col, linewidth=1.2)
    end

    # Four lengthwise lines connecting first and last rings
    for a in range(0, 2π; length=5)[1:end-1]
        off = r .* (cos(a) .* u .+ sin(a) .* v)
        lines!(ax,
            [Point3f((mr.origin .+ off)...), Point3f((mr.origin .+ mr.length_cm .* d .+ off)...)];
            color=col, linewidth=1.2)
    end

    # Dashed centre axis
    lines!(ax,
        [Point3f(mr.origin...), Point3f((mr.origin .+ mr.length_cm .* d)...)];
        color=:royalblue, linewidth=1.5, linestyle=:dash)

    # Entry-point marker
    scatter!(ax, [Point3f(mr.origin...)]; markersize=12, color=:royalblue)
end

function plot_tube_2d!(ax, mr::Minirhizotron;
                       project_to_plate::Bool=true,
                       y_max_cm::Float64=0.5)
    mr.enabled || return
    d = mr.direction
    O = mr.origin
    L = mr.length_cm
    R = mr.r_outer_cm

    # Determine t range along tube axis to display
    local t0, t1
    if project_to_plate
        t0, t1 = 0.0, L
    elseif abs(d[2]) < 1e-10
        (0.0 <= O[2] <= y_max_cm) || return
        t0, t1 = 0.0, L
    else
        ta = (0.0      - O[2]) / d[2]
        tb = (y_max_cm - O[2]) / d[2]
        t0 = max(0.0, min(ta, tb))
        t1 = min(L,   max(ta, tb))
        t0 >= t1 && return
    end

    # Perpendicular basis in 3D (same cross-product trick as plot_tube!)
    ref = abs(d[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    u = normalize_vec(cross(Vector{Float64}(d), ref))
    v = normalize_vec(cross(Vector{Float64}(d), u))

    # Silhouette: find α where u[2]*cos(α) + v[2]*sin(α) = 0
    u_y = u[2]; v_y = v[2]
    (abs(u_y) < 1e-10 && abs(v_y) < 1e-10) && return  # tube along Y → dot in XZ projection
    α1 = atan(-u_y, v_y)
    dx = R * (cos(α1)*u[1] + sin(α1)*v[1])
    dz = R * (cos(α1)*(-u[3]) + sin(α1)*(-v[3]))

    # Axis endpoints projected to (x, depth = -z)
    px0 = O[1] + t0*d[1];  pz0 = -(O[3] + t0*d[3])
    px1 = O[1] + t1*d[1];  pz1 = -(O[3] + t1*d[3])

    col = (:royalblue, 0.65)
    lines!(ax, [px0+dx, px1+dx], [pz0+dz, pz1+dz]; color=col, linewidth=1.5)
    lines!(ax, [px0-dx, px1-dx], [pz0-dz, pz1-dz]; color=col, linewidth=1.5)
    lines!(ax, [px0+dx, px0-dx], [pz0+dz, pz0-dz]; color=col, linewidth=1.2)
    lines!(ax, [px1+dx, px1-dx], [pz1+dz, pz1-dz]; color=col, linewidth=1.2)
    lines!(ax, [px0, px1], [pz0, pz1]; color=:royalblue, linewidth=1.5, linestyle=:dash)
end

function save_root_figure(plants::Vector{Plant}, box::Rhizobox, day::Int;
                          outdir::String="output_fspm_rhizobox",
                          minirhizotron::Union{Nothing,Minirhizotron}=nothing)
    _MAKIE_AVAILABLE[] || error("Makie not available. Install CairoMakie or GLMakie, then rerun. (save_every must be 0 otherwise)")
    fig = Figure(size=(900, 900), figure_padding=(10, 10, 150, 10))
    Label(fig[0, 1], @sprintf("Day %d | Plants: %d", day, length(plants)), fontsize=20, tellwidth=false)
    ax = Axis3(fig[1, 1], xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)", aspect=:data)
    xlims!(ax, -box.W/2, box.W/2)
    ylims!(ax, 0, box.D)
    zlims!(ax, -box.H, 0)

    for pl in plants
        for seg in pl.root.segments
            # Skip segments that cross a periodic boundary (visual wrap artefact)
            box.periodic_x && abs(seg.p1[1] - seg.p0[1]) > box.W / 2 && continue
            box.periodic_y && abs(seg.p1[2] - seg.p0[2]) > box.D / 2 && continue
            box.periodic_z && abs(seg.p1[3] - seg.p0[3]) > box.H / 2 && continue
            depth = -seg.p1[3]
            lines!(ax, [Point3f(seg.p0...), Point3f(seg.p1...)], linewidth=2,
                   color=depth, colormap=:viridis, colorrange=(0, box.H))
        end
    end
    Colorbar(fig[1, 2], colormap=:viridis, colorrange=(0, box.H), label="Depth (cm)")

    if minirhizotron !== nothing
        plot_tube!(ax, minirhizotron)
    end

    mkpath(outdir)
    fname = joinpath(outdir, "root_day_$(lpad(day, 3, '0')).png")
    save(fname, fig)
    return fname
end

function save_root_figure_2d(plants::Vector{Plant}, box::Rhizobox, day::Int;
                             outdir::String="output_fspm_rhizobox",
                             y_max_cm::Float64=box.D,
                             project_to_plate::Bool=true,
                             linewidth::Float64=0.8,
                             yreversed::Bool=true,
                             tube_axis_only::Bool=false,
                             minirhizotron::Union{Nothing,Minirhizotron}=nothing)

    _MAKIE_AVAILABLE[] || error("Makie not available. Install CairoMakie or GLMakie, then rerun.")

    mkpath(outdir)
    fname = joinpath(outdir, "root2d_day_$(lpad(day, 3, '0')).png")

    pbc_label = box.periodic_x ? " PBC" : ""
    box_label = @sprintf("%dx%d%s", Int(round(box.W)), Int(round(box.D)), pbc_label)

    fig = Figure(size=(900, 900))
    ax = Axis(fig[1, 1],
        xlabel="X (cm)",
        ylabel="Depth (cm)",
        title=@sprintf("Day %d | %s | plate view (X–Depth)", day, box_label),
        yreversed=yreversed)

    xlims!(ax, -box.W/2, box.W/2)
    ylims!(ax, 0, box.H)

    seg_color = (:black, 0.35)
    for pl in plants
        for seg in pl.root.segments
            box.periodic_x && abs(seg.p1[1] - seg.p0[1]) > box.W / 2 && continue
            box.periodic_y && abs(seg.p1[2] - seg.p0[2]) > box.D / 2 && continue
            box.periodic_z && abs(seg.p1[3] - seg.p0[3]) > box.H / 2 && continue
            if project_to_plate
                x0 = seg.p0[1]; d0 = -seg.p0[3]
                x1 = seg.p1[1]; d1 = -seg.p1[3]
                lines!(ax, [x0, x1], [d0, d1]; linewidth=linewidth, color=seg_color)
            else
                q0, q1, inside = clip_segment_to_y_slab(seg.p0, seg.p1; y_min=0.0, y_max=y_max_cm)
                inside || continue
                x0 = q0[1]; d0 = -q0[3]
                x1 = q1[1]; d1 = -q1[3]
                lines!(ax, [x0, x1], [d0, d1]; linewidth=linewidth, color=seg_color)
            end
        end
    end

    if minirhizotron !== nothing
        if tube_axis_only
            mr = minirhizotron
            O = mr.origin; d = mr.direction; L = mr.length_cm
            px = [O[1] + t*d[1] for t in (0.0, L)]
            pz = [-(O[3] + t*d[3]) for t in (0.0, L)]
            lines!(ax, px, pz; color=:royalblue, linewidth=1.5, linestyle=:dash)
        else
            plot_tube_2d!(ax, minirhizotron; project_to_plate=project_to_plate, y_max_cm=y_max_cm)
        end
    end

    save(fname, fig)
end

function save_tube_unrolled_figure(plants::Vector{Plant}, mr::Minirhizotron, day::Int;
                                    outdir::String="output_minirhizotron/root2d")
    _MAKIE_AVAILABLE[] || error("Makie not available.")
    mkpath(outdir)
    fname = joinpath(outdir, "root2d_day_$(lpad(day, 3, '0')).png")

    d   = mr.direction
    ref = abs(d[1]) < 0.9 ? [1.0,0.0,0.0] : [0.0,1.0,0.0]
    u   = normalize_vec(cross(Vector{Float64}(d), ref))
    v   = normalize_vec(cross(Vector{Float64}(d), u))
    rlo = mr.r_outer_cm
    rhi = mr.r_outer_cm + mr.skin_depth_cm
    step = mr.sample_step_cm
    arc_max = π * mr.r_outer_cm

    fig = Figure(size=(1200, 500))
    ax  = Axis(fig[1,1],
               xlabel="Along tube (cm)",
               ylabel="Arc position (cm)",
               title=@sprintf("Day %d | Unrolled tube view", day))
    xlims!(ax, 0.0, mr.length_cm)
    ylims!(ax, -arc_max, arc_max)

    for pl in plants
        for seg in pl.root.segments
            vseg = seg.p1 .- seg.p0
            L = _norm(vseg)
            L < 1e-12 && continue
            n = max(2, Int(ceil(L / step)))
            t_buf   = Float64[]
            arc_buf = Float64[]
            for k in 0:n
                p  = seg.p0 .+ (k/n) .* vseg
                dp = p .- mr.origin
                t  = dot(dp, d)
                if !(0.0 <= t <= mr.length_cm)
                    if length(t_buf) >= 2; lines!(ax, t_buf, arc_buf; linewidth=1.5); end
                    empty!(t_buf); empty!(arc_buf)
                    continue
                end
                foot   = mr.origin .+ t .* d
                radial = p .- foot
                r      = _norm(radial)
                if !(rlo <= r <= rhi)
                    if length(t_buf) >= 2; lines!(ax, t_buf, arc_buf; linewidth=1.5); end
                    empty!(t_buf); empty!(arc_buf)
                    continue
                end
                ru = dot(radial, u); rv = dot(radial, v)
                arc_new = atan(rv, ru) * mr.r_outer_cm
                # Break line at seam wrap-around (arc jumps by > half circumference)
                if !isempty(arc_buf) && abs(arc_new - arc_buf[end]) > arc_max
                    if length(t_buf) >= 2; lines!(ax, t_buf, arc_buf; linewidth=1.5); end
                    empty!(t_buf); empty!(arc_buf)
                end
                push!(t_buf, t)
                push!(arc_buf, arc_new)
            end
            if length(t_buf) >= 2
                lines!(ax, t_buf, arc_buf; linewidth=1.5)
            end
        end
    end

    # Station image windows: filled blue rectangles (along-tube extent ± half image height)
    half_h = mr.image_along_cm / 2.0
    for s_pos in mr.station_positions_cm
        vspan!(ax, s_pos - half_h, s_pos + half_h; color=(:royalblue, 0.10))
        vlines!(ax, [s_pos - half_h, s_pos + half_h];
                color=(:royalblue, 0.45), linewidth=0.8, linestyle=:dash)
    end

    # Camera azimuth windows: shaded horizontal bands with seam-wrapping
    if !isempty(mr.camera_azimuths_deg)
        half_arc = mr.image_arc_cm / 2.0
        cam_colours = [:red, :green, :orange, :purple, :brown]
        for (ci, az_deg) in enumerate(mr.camera_azimuths_deg)
            az_norm = mod(az_deg + 180.0, 360.0) - 180.0   # map to [−180, +180]
            arc_ctr = deg2rad(az_norm) * mr.r_outer_cm
            col = cam_colours[mod1(ci, length(cam_colours))]
            a_lo = arc_ctr - half_arc
            a_hi = arc_ctr + half_arc
            # Main band (clipped)
            lo_c = clamp(a_lo, -arc_max, arc_max)
            hi_c = clamp(a_hi, -arc_max, arc_max)
            if lo_c < hi_c
                hspan!(ax, lo_c, hi_c; color=(col, 0.15))
                hlines!(ax, [lo_c, hi_c]; color=(col, 0.55), linewidth=1.2, linestyle=:dash)
            end
            # Seam overflow: top → wrap to bottom
            if a_hi > arc_max
                ov = a_hi - arc_max
                hspan!(ax, -arc_max, -arc_max + ov; color=(col, 0.15))
                hlines!(ax, [-arc_max + ov]; color=(col, 0.55), linewidth=1.2, linestyle=:dash)
            end
            # Seam overflow: bottom → wrap to top
            if a_lo < -arc_max
                ov = -arc_max - a_lo
                hspan!(ax, arc_max - ov, arc_max; color=(col, 0.15))
                hlines!(ax, [arc_max - ov]; color=(col, 0.55), linewidth=1.2, linestyle=:dash)
            end
        end
    end

    save(fname, fig)
    return fname
end
