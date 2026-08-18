# generate_mock_mr_obs_3tubes_200cm.jl
#
# Generates mock minirhizotron observations for THREE tubes in the 200 cm wide box.
# Box: W=200, D=200, H=100. Plants: 25 cm bare border on all sides.
#   Plant zone: x_from_left 31.25-168.75 (12 cols, 12.5 cm spacing)
#               y 33-161 cm (9 rows, 16 cm spacing) → 108 plants total.
# Tubes at x=50, 100, 150 cm (from left) — all ≥50 cm from walls, no boundary suppression.
# Tube origin y=72 cm → scans y≈74..123 cm, centred at y≈100 cm (plot centre).
#
# Output: calibration/data/mr_obs_stations_3tubes_200cm.csv
# Columns: tube_id,date,station_idx,along_tube_cm,depth_cm,azimuth_deg,root_length_cm
#
# Run: julia calibration/generate_mock_mr_obs_3tubes_200cm.jl  (from repo root)

using JSON
using JSON3
using CSV
using DataFrames
using Dates
using Random
using Printf
using Statistics

# ----------------------------
# Settings
# ----------------------------
const SOWING_DATE  = Date(2025, 4, 1)
const ACQ_OFFSETS  = 7:7:91         # days after sowing: 7, 14, 21, ..., 91 (weekly)
const MOCK_SEED       = 42
const NOISE_CV        = 0.20
const SIGMA_ANGLE_DEG = 5.0   # per-segment Gaussian noise on crossing angle

# Three tube x-positions in the 200 cm wide box (x_from_left convention)
const TUBE_X_CM = [75.0, 100.0, 125.0]
const BOX_W_CM  = 200.0

# ----------------------------
# Include model code
# ----------------------------
include(joinpath(@__DIR__, "..", "src", "Deps.jl"))
include(joinpath(@__DIR__, "..", "src", "Types.jl"))
include(joinpath(@__DIR__, "..", "src", "Geometry.jl"))
include(joinpath(@__DIR__, "..", "src", "DensityRegulation.jl"))
include(joinpath(@__DIR__, "..", "src", "GrowthStep.jl"))
include(joinpath(@__DIR__, "..", "src", "MetricsAndOutput.jl"))
include(joinpath(@__DIR__, "..", "src", "Plotting.jl"))
include(joinpath(@__DIR__, "..", "src", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "Simulation.jl"))

# ----------------------------
# JSON helpers
# ----------------------------
read_json(path)  = open(path, "r") do io; JSON.parse(String(read(io))); end
function write_json_pretty(path, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

# Build a Minirhizotron at a given from-left x position (200 cm box).
function make_tube(x_from_left_cm::Float64; W_cm::Float64=BOX_W_CM)
    origin    = [x_from_left_cm - W_cm/2, 72.0, 0.0]
    direction = normalize_vec([0.0, 0.7071067811865476, -0.7071067811865476])
    return Minirhizotron(
        true, origin, direction, 80.0, 3.0, 1.0, 5.0, 5.0,
        collect(2.5:5.0:72.5), 0.1, "tube_stations_daily.csv",
        [0.0, 90.0, 180.0, 270.0]
    )
end

"""
Per-window (station×azimuth) accumulation of length, noisy mean angle, and noisy histogram.
Returns (lengths, angle_sum, hist_sum). Each segment contributes at a noise-perturbed angle.
"""
function acquire_window_angles(plants::Vector{Plant}, mr::Minirhizotron)
    ns = length(mr.station_positions_cm)
    na = length(mr.camera_azimuths_deg)
    half_h       = mr.image_along_cm / 2.0
    half_arc_deg = (mr.image_arc_cm / (2π * mr.r_outer_cm)) * 180.0
    rlo  = mr.r_outer_cm
    rhi  = mr.r_outer_cm + mr.skin_depth_cm
    step = mr.sample_step_cm

    lengths   = zeros(Float64, ns, na)
    angle_sum = zeros(Float64, ns, na)
    hist_sum  = zeros(Float64, ns, na, N_ANGLE_BINS)

    for pl in plants
        for seg in pl.root.segments
            v = seg.p1 .- seg.p0
            L = _norm(v)
            L < 1e-12 && continue
            seg_dir = v ./ L
            θ_true  = _segment_crossing_angle_deg(seg_dir, mr.direction)
            θ_noisy = clamp(θ_true + SIGMA_ANGLE_DEG * randn(), 0.0, 90.0)
            bin_idx = _angle_bin(θ_noisy)

            n  = max(1, Int(ceil(L / step)))
            dl = L / n
            for k in 0:(n-1)
                tmid = (k + 0.5) / n
                p    = seg.p0 .+ tmid .* v
                t, r = _tube_project(p, mr.origin, mr.direction)
                rlo <= r <= rhi || continue
                0.0 <= t <= mr.length_cm || continue
                si = 0
                for s in 1:ns
                    if abs(t - mr.station_positions_cm[s]) <= half_h; si = s; break; end
                end
                si == 0 && continue
                az = _tube_azimuth(p, mr.origin, mr.direction)
                ai = 0
                for a in 1:na
                    diff = abs(mod(az - mr.camera_azimuths_deg[a] + 180.0, 360.0) - 180.0)
                    if diff <= half_arc_deg; ai = a; break; end
                end
                ai == 0 && continue
                lengths[si, ai]           += dl
                angle_sum[si, ai]         += θ_noisy * dl
                hist_sum[si, ai, bin_idx] += dl
            end
        end
    end
    return lengths, angle_sum, hist_sum
end

function save_root_figure_3tubes(plants, box, day::Int, tubes; outdir::String)
    _MAKIE_AVAILABLE[] || return
    fig = Figure(size=(900, 900), figure_padding=(10, 10, 150, 10))
    Label(fig[0, 1], @sprintf("Day %d | Plants: %d | 3 tubes at x=75,100,125 cm", day, length(plants)),
          fontsize=17, tellwidth=false)
    ax = Axis3(fig[1, 1], xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)", aspect=:data)
    xlims!(ax, -box.W/2, box.W/2)
    ylims!(ax, 0, box.D)
    zlims!(ax, -box.H, 0)
    for pl in plants
        for seg in pl.root.segments
            depth = -seg.p1[3]
            lines!(ax, [Point3f(seg.p0...), Point3f(seg.p1...)], linewidth=2,
                   color=depth, colormap=:viridis, colorrange=(0, box.H))
        end
    end
    Colorbar(fig[1, 2], colormap=:viridis, colorrange=(0, box.H), label="Depth (cm)")
    for tube in tubes
        plot_tube!(ax, tube)
    end
    mkpath(outdir)
    fname = joinpath(outdir, "root_day_$(lpad(day, 3, '0')).png")
    save(fname, fig)
end

function main()
    Random.seed!(MOCK_SEED)
    σ_ln = sqrt(log(1 + NOISE_CV^2))

    tmpdir = joinpath(@__DIR__, "__mock_tmp_3tubes_200cm__")
    mkpath(tmpdir)

    n_sim_days     = maximum(ACQ_OFFSETS) + 1
    acq_simday_set = Set(Int(d) + 1 for d in ACQ_OFFSETS)
    simday_to_date = Dict(Int(d)+1 => SOWING_DATE + Day(d) for d in ACQ_OFFSETS)

    tubes = [make_tube(tx) for tx in TUBE_X_CM]

    # Load 200 cm box config with tube disabled (manual sampling loop)
    plot3d_dir = joinpath(tmpdir, "plot3d")
    mkpath(plot3d_dir)

    env = read_json(joinpath(@__DIR__, "Parameter_calib_minirhizotron_200cm.json"))
    env["species_params_files"]["VD"]       = joinpath(@__DIR__, "Parameter_VD_template.json")
    env["simulation"]["seed"]               = MOCK_SEED
    env["simulation"]["days"]               = n_sim_days
    env["simulation"]["outdir"]             = tmpdir
    env["simulation"]["plot3d_outdir"]      = plot3d_dir
    env["minirhizotron"]["enabled"]         = false
    env["facilitation"]["enabled"]          = false
    env["density_regulation"]["enabled"]    = false

    env_path = joinpath(tmpdir, "Parameter_tmp.json")
    write_json_pretty(env_path, env)
    cfg = load_environment_and_plants(env_path)

    all_rows = DataFrame(
        tube_id=Int[], date=String[], station_idx=Int[],
        along_tube_cm=Float64[], depth_cm=Float64[], azimuth_deg=Float64[],
        root_length_cm=Float64[], mean_crossing_angle_deg=Float64[],
        hist_bin0=Float64[], hist_bin1=Float64[], hist_bin2=Float64[],
        hist_bin3=Float64[], hist_bin4=Float64[], hist_bin5=Float64[]
    )

    R_dummy = zeros(1, 1)
    Random.seed!(MOCK_SEED)
    @info "Running 200 cm box simulation for 3 tubes simultaneously (x=75, 100, 125 cm)..."
    prog = Progress(n_sim_days; desc="Simulating days", dt=0.5)
    for day in 1:n_sim_days
        for pl in cfg.plants
            step_plant!(pl, cfg.box, cfg.reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end

        if day in acq_simday_set
            dt = simday_to_date[day]
            save_root_figure_3tubes(cfg.plants, cfg.box, day, tubes; outdir=plot3d_dir)
            for (tid, tube) in enumerate(tubes)
                ns = length(tube.station_positions_cm)
                na = length(tube.camera_azimuths_deg)
                lengths, angle_sum, hist_sum = acquire_window_angles(cfg.plants, tube)
                for si in 1:ns
                    s_pos    = tube.station_positions_cm[si]
                    p_sta    = tube.origin .+ s_pos .* tube.direction
                    depth_cm = -p_sta[3]
                    for ai in 1:na
                        az  = tube.camera_azimuths_deg[ai]
                        val = lengths[si, ai]
                        noisy_len = val > 0 ? val * exp(σ_ln * randn() - σ_ln^2/2) : 0.0
                        if val > 1e-12
                            mean_ang = angle_sum[si, ai] / val
                            bins     = ntuple(b -> hist_sum[si, ai, b] / val, N_ANGLE_BINS)
                        else
                            mean_ang = NaN
                            bins     = ntuple(_ -> NaN, N_ANGLE_BINS)
                        end
                        push!(all_rows, (tid, string(dt), si, s_pos, depth_cm, az,
                            max(0.0, noisy_len), mean_ang,
                            bins[1], bins[2], bins[3], bins[4], bins[5], bins[6]))
                    end
                end
            end
        end
        next!(prog)
    end

    out_csv = joinpath(@__DIR__, "data", "mr_obs_stations_3tubes_200cm.csv")
    sort!(all_rows, [:tube_id, :date, :station_idx, :azimuth_deg])
    CSV.write(out_csv, all_rows)
    @info "Wrote 200 cm 3-tube mock observations" path=out_csv nrow=nrow(all_rows)

    for tid in 1:length(TUBE_X_CM)
        @printf("\nTube %d (x=%.0f cm):\n", tid, TUBE_X_CM[tid])
        for (sd, dt) in sort(collect(simday_to_date))
            sub = all_rows[(all_rows.tube_id .== tid) .& (all_rows.date .== string(dt)), :]
            nrow(sub) > 0 && @printf("  %s (day %3d): %d rows, total=%.3f cm, mean_angle=%.1f°\n",
                dt, sd, nrow(sub), sum(sub.root_length_cm),
                mean(filter(!isnan, sub.mean_crossing_angle_deg)))
        end
    end
end

main()
