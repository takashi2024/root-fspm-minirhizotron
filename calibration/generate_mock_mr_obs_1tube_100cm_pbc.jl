# generate_mock_mr_obs_1tube_100cm_pbc.jl
#
# Generates mock minirhizotron observations for the 100 cm PBC box design.
# Single tube centred at x=50 cm (x=0 in centred coords), y-origin=21.716 cm.
#
# Output: calibration/data/mr_obs_stations_1tube_100cm_pbc.csv
# Columns: date, station_idx, along_tube_cm, depth_cm, azimuth_deg,
#          root_length_cm, mean_crossing_angle_deg,
#          hist_bin0..hist_bin5
#
# Run: julia calibration/generate_mock_mr_obs_1tube_100cm_pbc.jl  (from repo root)

using JSON
using JSON3
using CSV
using DataFrames
using Dates
using Random
using Printf
using ProgressMeter
using Statistics

# ----------------------------
# Settings
# ----------------------------
const SOWING_DATE     = Date(2025, 4, 1)
const ACQ_OFFSETS     = 7:7:91        # days after sowing: 7, 14, ..., 91 (weekly)
const MOCK_SEED       = 42
const NOISE_CV        = 0.20
const SIGMA_ANGLE_DEG = 5.0           # per-segment Gaussian noise on crossing angle

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
read_json(path::AbstractString) = open(path, "r") do io; JSON.parse(String(read(io))); end
function write_json_pretty(path::AbstractString, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

# Build the 1-tube Minirhizotron matching the calibration runner's _make_tube(50.0; W_cm=100.0).
# origin = [tube_x_from_left - W/2, tube_origin_y, 0] = [50-50, 21.716, 0] = [0, 21.716, 0]
function make_tube()
    origin    = [0.0, 21.716, 0.0]
    direction = normalize_vec([0.0, 0.7071067811865476, -0.7071067811865476])
    return Minirhizotron(
        true, origin, direction, 80.0, 3.0, 1.0, 5.0, 5.0,
        collect(2.5:5.0:72.5), 0.1, "tube_stations_daily.csv",
        [0.0, 90.0, 180.0, 270.0]
    )
end

"""
Per-window (station×azimuth) accumulation of length, noisy mean angle, and noisy histogram.
Returns (lengths, angle_sum, hist_sum) as ns×na, ns×na, ns×na×N_ANGLE_BINS arrays.
Each visible segment contributes its true-length dl at a noise-perturbed angle.
"""
function acquire_window_angles(plants::Vector{Plant}, mr::Minirhizotron)
    ns = length(mr.station_positions_cm)
    na = length(mr.camera_azimuths_deg)

    half_h       = mr.image_along_cm / 2.0
    half_arc_deg = (mr.image_arc_cm / (2π * mr.r_outer_cm)) * 180.0
    rlo          = mr.r_outer_cm
    rhi          = mr.r_outer_cm + mr.skin_depth_cm
    step         = mr.sample_step_cm

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
                    if abs(t - mr.station_positions_cm[s]) <= half_h
                        si = s; break
                    end
                end
                si == 0 && continue

                az = _tube_azimuth(p, mr.origin, mr.direction)
                ai = 0
                for a in 1:na
                    diff = abs(mod(az - mr.camera_azimuths_deg[a] + 180.0, 360.0) - 180.0)
                    if diff <= half_arc_deg
                        ai = a; break
                    end
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

function main()
    Random.seed!(MOCK_SEED)
    σ_ln = sqrt(log(1 + NOISE_CV^2))

    tmpdir = joinpath(@__DIR__, "__mock_tmp_1tube_100cm_pbc__")
    mkpath(tmpdir)

    n_sim_days     = maximum(ACQ_OFFSETS) + 1
    acq_simday_set = Set(Int(d) + 1 for d in ACQ_OFFSETS)
    simday_to_date = Dict(Int(d)+1 => SOWING_DATE + Day(d) for d in ACQ_OFFSETS)

    tube = make_tube()

    env = read_json(joinpath(@__DIR__, "Parameter_calib_minirhizotron_100cm_pbc.json"))
    env["species_params_files"]["VD"]    = joinpath(@__DIR__, "Parameter_VD_template.json")
    env["simulation"]["seed"]            = MOCK_SEED
    env["simulation"]["days"]            = n_sim_days
    env["simulation"]["outdir"]          = tmpdir
    env["simulation"]["save_every"]      = 0
    env["minirhizotron"]["enabled"]      = false
    env["facilitation"]["enabled"]       = false
    env["density_regulation"]["enabled"] = false

    env_path = joinpath(tmpdir, "Parameter_tmp.json")
    write_json_pretty(env_path, env)
    cfg = load_environment_and_plants(env_path)

    df_obs = DataFrame(
        date                    = String[],
        station_idx             = Int[],
        along_tube_cm           = Float64[],
        depth_cm                = Float64[],
        azimuth_deg             = Float64[],
        root_length_cm          = Float64[],
        mean_crossing_angle_deg = Float64[],
        hist_bin0               = Float64[],
        hist_bin1               = Float64[],
        hist_bin2               = Float64[],
        hist_bin3               = Float64[],
        hist_bin4               = Float64[],
        hist_bin5               = Float64[]
    )

    R_dummy = zeros(1, 1)
    Random.seed!(MOCK_SEED)
    @info "Running 100 cm PBC box simulation for 1 tube at x=50 cm (y_origin=21.716 cm)..."
    prog = Progress(n_sim_days; desc="Simulating days", dt=0.5)
    for day in 1:n_sim_days
        for pl in cfg.plants
            step_plant!(pl, cfg.box, cfg.reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end

        if day in acq_simday_set
            dt = simday_to_date[day]
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

                    push!(df_obs, (
                        string(dt), si, s_pos, depth_cm, az,
                        max(0.0, noisy_len), mean_ang,
                        bins[1], bins[2], bins[3], bins[4], bins[5], bins[6]
                    ))
                end
            end
        end
        next!(prog)
    end

    mkpath(joinpath(@__DIR__, "data"))
    out_csv = joinpath(@__DIR__, "data", "mr_obs_stations_1tube_100cm_pbc.csv")
    sort!(df_obs, [:date, :station_idx, :azimuth_deg])
    CSV.write(out_csv, df_obs)
    @info "Wrote 100 cm PBC mock observations (1 tube)" path=out_csv nrow=nrow(df_obs)

    for (sd, dt) in sort(collect(simday_to_date))
        sub = df_obs[df_obs.date .== string(dt), :]
        if nrow(sub) > 0
            valid_angles = filter(!isnan, sub.mean_crossing_angle_deg)
            ang_str = isempty(valid_angles) ? "NaN" : @sprintf("%.1f°", mean(valid_angles))
            @printf("  %s (day %3d): %d rows, total=%.3f cm, mean_angle=%s\n",
                dt, sd, nrow(sub), sum(skipmissing(sub.root_length_cm)), ang_str)
        end
    end
end

main()
