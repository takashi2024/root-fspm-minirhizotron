# generate_mock_mr_obs_3tubes.jl
#
# Generates mock minirhizotron observations for THREE tube replicates.
# Tubes are at x=25, 50, 75 cm — each centred between a pair of adjacent rows
# in the conventional rapeseed layout (12.5 cm row spacing, 3.25 cm clearance).
#
# Output: calibration/data/mr_obs_stations_3tubes.csv
# Columns: tube_id,date,station_idx,along_tube_cm,depth_cm,azimuth_deg,root_length_cm
#
# Run: julia calibration/generate_mock_mr_obs_3tubes.jl  (from repo root)

using JSON
using JSON3
using CSV
using DataFrames
using Dates
using Random
using Printf

# ----------------------------
# Settings
# ----------------------------
const SOWING_DATE  = Date(2025, 4, 1)
const ACQ_OFFSETS  = 30:14:90       # days after sowing
const MOCK_SEED    = 42
const NOISE_CV     = 0.20

# Three tube x-positions (cm from left wall, W=100)
# x=25: between row 2 (x=18.75) and row 3 (x=31.25) — 3.25 cm clearance each side
# x=50: between row 4 (x=43.75) and row 5 (x=56.25) — 3.25 cm clearance each side
# x=75: between row 6 (x=68.75) and row 7 (x=81.25) — 3.25 cm clearance each side
const TUBE_X_CM = [25.0, 50.0, 75.0]

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

# Build a Minirhizotron at a given from-left x position.
# Matches the geometry in Parameter_calib_minirhizotron.json.
function make_tube(x_from_left_cm::Float64; W_cm::Float64=100.0)
    origin    = [x_from_left_cm - W_cm/2, 20.0, 0.0]
    direction = normalize_vec([0.0, 0.7071067811865476, -0.7071067811865476])
    return Minirhizotron(
        true, origin, direction, 80.0, 3.0, 1.0, 5.0, 5.0,
        collect(2.5:5.0:72.5), 0.1, "tube_stations_daily.csv",
        [0.0, 90.0, 180.0, 270.0]
    )
end

function main()
    Random.seed!(MOCK_SEED)
    σ_ln = sqrt(log(1 + NOISE_CV^2))

    tmpdir = joinpath(@__DIR__, "__mock_tmp_3tubes__")
    mkpath(tmpdir)

    n_sim_days     = maximum(ACQ_OFFSETS) + 1
    acq_simday_set = Set(Int(d) + 1 for d in ACQ_OFFSETS)
    simday_to_date = Dict(Int(d)+1 => SOWING_DATE + Day(d) for d in ACQ_OFFSETS)

    # Build 3 tube objects — all share the same root system
    tubes = [make_tube(tx) for tx in TUBE_X_CM]

    # Load plants once with tube observation disabled.
    # Roots grow without tube boundary deflection, then all 3 tubes sample the
    # same realisation — matching the real installation where one root system
    # is observed by all tubes simultaneously.
    env = read_json(joinpath(@__DIR__, "Parameter_calib_minirhizotron.json"))
    env["species_params_files"]["VD"]        = joinpath(@__DIR__, "Parameter_VD_template.json")
    env["simulation"]["seed"]                = MOCK_SEED
    env["simulation"]["days"]                = n_sim_days
    env["simulation"]["outdir"]              = tmpdir
    env["simulation"]["plot3d_outdir"]       = joinpath(tmpdir, "plot3d")
    env["minirhizotron"]["enabled"]          = false
    env["facilitation"]["enabled"]           = false
    env["density_regulation"]["enabled"]     = false

    env_path = joinpath(tmpdir, "Parameter_tmp.json")
    write_json_pretty(env_path, env)
    cfg = load_environment_and_plants(env_path)

    all_rows = DataFrame(
        tube_id=Int[], date=String[], station_idx=Int[],
        along_tube_cm=Float64[], depth_cm=Float64[],
        azimuth_deg=Float64[], root_length_cm=Float64[]
    )

    R_dummy = zeros(1, 1)
    Random.seed!(MOCK_SEED)
    @info "Running single forward simulation for all 3 tubes simultaneously..."
    prog = Progress(n_sim_days; desc="Simulating days", dt=0.5)
    for day in 1:n_sim_days
        for pl in cfg.plants
            step_plant!(pl, cfg.box, cfg.reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end

        if day in acq_simday_set
            dt = simday_to_date[day]
            for (tid, tube) in enumerate(tubes)
                ns  = length(tube.station_positions_cm)
                na  = length(tube.camera_azimuths_deg)
                mat = tube_station_azimuth_lengths_cm(cfg.plants, tube)
                for si in 1:ns
                    s_pos    = tube.station_positions_cm[si]
                    p_sta    = tube.origin .+ s_pos .* tube.direction
                    depth_cm = -p_sta[3]
                    for ai in 1:na
                        az    = tube.camera_azimuths_deg[ai]
                        val   = mat[si, ai]
                        noisy = val > 0 ? val * exp(σ_ln * randn() - σ_ln^2/2) : 0.0
                        push!(all_rows, (tid, string(dt), si,
                            s_pos, depth_cm, az, max(0.0, noisy)))
                    end
                end
            end
        end
        next!(prog)
    end

    out_csv = joinpath(@__DIR__, "data", "mr_obs_stations_3tubes.csv")
    sort!(all_rows, [:tube_id, :date, :station_idx, :azimuth_deg])
    CSV.write(out_csv, all_rows)
    @info "Wrote 3-tube mock observations" path=out_csv nrow=nrow(all_rows)

    # Summary per tube per date
    for tid in 1:length(TUBE_X_CM)
        @printf("\nTube %d (x=%.0f cm):\n", tid, TUBE_X_CM[tid])
        for (sd, dt) in sort(collect(simday_to_date))
            sub = all_rows[(all_rows.tube_id .== tid) .& (all_rows.date .== string(dt)), :]
            nrow(sub) > 0 && @printf("  %s (day %3d): %d rows, total=%.3f cm\n",
                dt, sd, nrow(sub), sum(sub.root_length_cm))
        end
    end
end

main()
