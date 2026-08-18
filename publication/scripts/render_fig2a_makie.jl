"""
render_fig2a_makie.jl

Renders panel (a) of Figure 2 using CairoMakie: the 10 plants nearest to the
minirhizotron tube axis (plants 4–13, column 1, y ≈ 22–78 cm, x = −25 cm).

Root segments are coloured by depth with the viridis colormap — matching the
aesthetic of save_root_figure() in src/Plotting.jl.  All root orders are shown;
order-2 laterals use reduced alpha to avoid clutter.

Output: output_fig2_tmp/plot3d/root_day_084_2plants.png

Run from project root:
    julia publication/scripts/render_fig2a_makie.jl
"""

const PROJ = abspath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJ, "src", "Deps.jl"))
include(joinpath(PROJ, "src", "Types.jl"))
include(joinpath(PROJ, "src", "Geometry.jl"))
include(joinpath(PROJ, "src", "DensityRegulation.jl"))
include(joinpath(PROJ, "src", "GrowthStep.jl"))
include(joinpath(PROJ, "src", "MetricsAndOutput.jl"))
include(joinpath(PROJ, "src", "Plotting.jl"))
include(joinpath(PROJ, "src", "Config.jl"))
include(joinpath(PROJ, "src", "Simulation.jl"))

const ENV_JSON  = joinpath(PROJ, "calibration", "fig1_env_D2_281_2mm.json")
const OUTDIR    = joinpath(PROJ, "output_fig2_tmp", "plot3d")
# 10 nearest plants to tube axis: column-1 plants alongside the tube y-extent
const PLANT_IDS = 1:10

mkpath(OUTDIR)

println("Simulating to day 84...")
cfg = load_environment_and_plants(ENV_JSON)
simulate!(cfg.plants, cfg.box, cfg.reg;
    days          = 84,
    seed          = cfg.seed,
    save_every    = 0,
    outdir        = cfg.outdir,
    minirhizotron = cfg.mr,
    mr_csv_path   = joinpath(cfg.outdir, "fov_tube_fig2a.csv"),
    show_progress = true)

sel = cfg.plants[collect(PLANT_IDS)]

# ── Figure ────────────────────────────────────────────────────────────────────

fig = Figure(size=(900, 900), figure_padding=(10, 10, 60, 10))

ax = Axis3(fig[1, 1];
    xlabel         = "x (cm)",
    ylabel         = "y (cm)",
    zlabel         = "z (cm)",
    xlabelsize     = 18,
    ylabelsize     = 18,
    zlabelsize     = 18,
    xticklabelsize = 14,
    yticklabelsize = 14,
    zticklabelsize = 14,
    aspect         = :data)

box = cfg.box
X_MIN = -65.0; X_MAX = 5.0
Y_MIN = 0.0;   Y_MAX = box.D * 0.85
Z_MIN = -box.H * 0.90; Z_MAX = 0.0

xlims!(ax, X_MIN, X_MAX)
ylims!(ax, Y_MIN, Y_MAX)
zlims!(ax, Z_MIN, Z_MAX)

# Depth-based viridis colouring — same aesthetic as save_root_figure().
# Order 2 is skipped (too numerous; hides the tube).
# Order-1 laterals rendered with thin lines to show branching structure.
ORDER_LW = Dict{Int,Float32}(0 => 2.5f0, 1 => 0.6f0)

out_of_range = p -> p[1] < X_MIN || p[1] > X_MAX ||
                    p[2] < Y_MIN || p[2] > Y_MAX ||
                    p[3] < Z_MIN || p[3] > Z_MAX

# Azimuth-aware depth split.
# Camera depth: dist(P) = P_x·cos(az) + P_y·sin(az); larger = farther = behind.
# Compare each segment midpoint's dist() to the tube centroid's dist().
az_rad        = Float64(ax.azimuth[])
cos_az, sin_az = cos(az_rad), sin(az_rad)
tube_x        = Float64(cfg.mr.origin[1])   # −12.5 (constant along tube)
tube_y_ctr    = cfg.mr.origin[2] + (cfg.mr.length_cm / 2.0) * cfg.mr.direction[2]
tube_dist_ref = tube_x * cos_az + tube_y_ctr * sin_az

ORDER_LW_BG = Dict{Int,Float32}(0 => 2.0f0, 1 => 0.5f0, 2 => 0.25f0)
ORDER_LW2   = Dict{Int,Float32}(0 => 2.5f0, 1 => 0.6f0,  2 => 0.35f0)

# Pass 1 — all 32 plants, all orders, faint gray (structural context)
for pl in cfg.plants
    for seg in pl.root.segments
        box.periodic_x && abs(seg.p1[1] - seg.p0[1]) > box.W / 2 && continue
        box.periodic_y && abs(seg.p1[2] - seg.p0[2]) > box.D / 2 && continue
        (out_of_range(seg.p0) && out_of_range(seg.p1)) && continue
        lw = get(ORDER_LW_BG, seg.order, 0.25f0)
        lines!(ax, [Point3f(seg.p0...), Point3f(seg.p1...)];
               color = (:gray55, 0.18f0), linewidth = lw)
    end
end

# Pass 2a — focal plants, segments BEHIND the tube (dist ≥ tube_dist_ref)
# drawn before tube so the solid cylinder occludes them
for pl in sel
    for seg in pl.root.segments
        let mx = (seg.p0[1]+seg.p1[1])/2, my = (seg.p0[2]+seg.p1[2])/2
            mx*cos_az + my*sin_az < tube_dist_ref && continue  # skip if in front
        end
        box.periodic_x && abs(seg.p1[1] - seg.p0[1]) > box.W / 2 && continue
        box.periodic_y && abs(seg.p1[2] - seg.p0[2]) > box.D / 2 && continue
        (out_of_range(seg.p0) && out_of_range(seg.p1)) && continue
        depth = -seg.p1[3]
        lw    = get(ORDER_LW2, seg.order, 0.35f0)
        lines!(ax, [Point3f(seg.p0...), Point3f(seg.p1...)];
               color = depth, colormap = :viridis,
               colorrange = (0.0, box.H), linewidth = lw)
    end
end

# Tube: solid opaque cylinder (drawn between the two focal passes so it occludes
# roots behind it and is itself occluded by roots in front)
let
    mr  = cfg.mr
    d   = mr.direction
    ref = abs(d[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    u   = normalize_vec(cross(Vector{Float64}(d), ref))
    v   = normalize_vec(cross(Vector{Float64}(d), u))
    r   = mr.r_outer_cm

    nθ = 48; nL = 24
    θs = range(0, 2π; length=nθ+1)
    ts = range(0.0, mr.length_cm; length=nL)
    xmat = Matrix{Float32}(undef, nL, nθ+1)
    ymat = Matrix{Float32}(undef, nL, nθ+1)
    zmat = Matrix{Float32}(undef, nL, nθ+1)
    for (i, ti) in enumerate(ts), (j, a) in enumerate(θs)
        pt = mr.origin .+ ti .* d .+ r .* (cos(a) .* u .+ sin(a) .* v)
        xmat[i,j] = Float32(pt[1])
        ymat[i,j] = Float32(pt[2])
        zmat[i,j] = Float32(pt[3])
    end
    cmat = fill(RGBAf(0.855f0, 0.647f0, 0.125f0, 0.88f0), (nL, nθ+1))
    surface!(ax, xmat, ymat, zmat; color=cmat, shading=NoShading)

    col = (:goldenrod4, 1.0)
    θ = range(0, 2π; length=37)
    for t in range(0.0, mr.length_cm; length=9)
        ctr = mr.origin .+ t .* d
        pts = [Point3f((ctr .+ r .* (cos(a) .* u .+ sin(a) .* v))...) for a in θ]
        lines!(ax, pts; color=col, linewidth=1.0)
    end
    for a in range(0, 2π; length=5)[1:end-1]
        off = r .* (cos(a) .* u .+ sin(a) .* v)
        lines!(ax,
            [Point3f((mr.origin .+ off)...),
             Point3f((mr.origin .+ mr.length_cm .* d .+ off)...)];
            color=col, linewidth=1.0)
    end
end

# Pass 2b — focal plants, segments IN FRONT of the tube (dist < tube_dist_ref)
# drawn after tube so they appear on top of the cylinder
for pl in sel
    for seg in pl.root.segments
        let mx = (seg.p0[1]+seg.p1[1])/2, my = (seg.p0[2]+seg.p1[2])/2
            mx*cos_az + my*sin_az >= tube_dist_ref && continue  # skip if behind
        end
        box.periodic_x && abs(seg.p1[1] - seg.p0[1]) > box.W / 2 && continue
        box.periodic_y && abs(seg.p1[2] - seg.p0[2]) > box.D / 2 && continue
        (out_of_range(seg.p0) && out_of_range(seg.p1)) && continue
        depth = -seg.p1[3]
        lw    = get(ORDER_LW2, seg.order, 0.35f0)
        lines!(ax, [Point3f(seg.p0...), Point3f(seg.p1...)];
               color = depth, colormap = :viridis,
               colorrange = (0.0, box.H), linewidth = lw)
    end
end

# Station window markers — 5×5 cm red rectangles on sky-facing (azimuth 0) tube surface
let
    mr  = cfg.mr
    d   = mr.direction
    ref = abs(d[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    u   = normalize_vec(cross(Vector{Float64}(d), ref))
    v   = normalize_vec(cross(Vector{Float64}(d), u))
    sky = -u                         # azimuth-0 outward normal (sky-facing)
    r   = mr.r_outer_cm
    ha  = mr.image_along_cm / 2.0
    hc  = mr.image_arc_cm   / 2.0

    for si in [5, 9, 13]
        along = mr.station_positions_cm[si]
        ctr   = mr.origin .+ along .* d
        wctr  = ctr .+ r .* sky
        c1 = Point3f((wctr .+ ha .* d .+ hc .* v)...)
        c2 = Point3f((wctr .+ ha .* d .- hc .* v)...)
        c3 = Point3f((wctr .- ha .* d .- hc .* v)...)
        c4 = Point3f((wctr .- ha .* d .+ hc .* v)...)
        lines!(ax, [c1, c2, c3, c4, c1]; color=:red, linewidth=2.5)
        lbl_pos = Point3f((wctr .+ [0.0, 4.0, 4.0])...)
        text!(ax, lbl_pos; text="St.$si",
              color=:red, fontsize=16, align=(:center, :bottom))
    end
end

# Camera: rotate to show plants from a more frontal angle, low elevation
ax.azimuth[]   = 0.15π
ax.elevation[] = 0.15π

fname = joinpath(OUTDIR, "root_day_084_2plants.png")
save(fname, fig)
println("Saved → $(abspath(fname))")
