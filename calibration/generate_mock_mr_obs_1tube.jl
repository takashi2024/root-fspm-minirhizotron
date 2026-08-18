# generate_mock_mr_obs_1tube.jl
#
# Generates mock minirhizotron observation data by running the forward simulation
# with the VD template parameters and adding lognormal noise (CV=20%) to simulate
# DL measurement error.
# Single tube at x=50 cm (centred, midway between row pairs 4 and 5).
#
# Output: calibration/data/mr_obs_stations_1tube.csv
# Columns: date,station_idx,along_tube_cm,depth_cm,azimuth_deg,root_length_cm
#
# Run: julia calibration/generate_mock_mr_obs_1tube.jl   (from repo root)

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
const SOWING_DATE = Date(2025, 4, 1)
const ACQ_OFFSETS = 30:14:90      # days after sowing: 30, 44, 58, 72, 86
const MOCK_SEED   = 42
const NOISE_CV    = 0.20

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
read_json(path::AbstractString)  = open(path, "r") do io; JSON.parse(String(read(io))); end
function write_json_pretty(path::AbstractString, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

function main()
    Random.seed!(MOCK_SEED)
    σ_ln = sqrt(log(1 + NOISE_CV^2))

    tmpdir = joinpath(@__DIR__, "__mock_tmp_1tube__")
    mkpath(tmpdir)

    env_path_base = joinpath(@__DIR__, "Parameter_calib_minirhizotron.json")
    env = read_json(env_path_base)

    # Point species file to calibration VD template (absolute path)
    env["species_params_files"]["VD"] = joinpath(@__DIR__, "Parameter_VD_template.json")
    env["simulation"]["seed"]   = MOCK_SEED
    env["simulation"]["days"]   = maximum(ACQ_OFFSETS) + 1
    env["simulation"]["outdir"] = tmpdir
    env["simulation"]["plot3d_outdir"] = joinpath(tmpdir, "plot3d")

    mr_csv_path = joinpath(tmpdir, "tube_stations_daily.csv")
    env["minirhizotron"]["csv_name"] = "tube_stations_daily.csv"

    env_tmp = joinpath(tmpdir, "Parameter_calib_tmp.json")
    write_json_pretty(env_tmp, env)

    @info "Running forward simulation for mock data..."
    cfg = load_environment_and_plants(env_tmp)

    simulate!(cfg.plants, cfg.box, cfg.reg;
        days             = maximum(ACQ_OFFSETS) + 1,
        seed             = MOCK_SEED,
        save_every       = 0,
        outdir           = cfg.outdir,
        plot3d_dir       = cfg.plot3d_dir,
        viz_enabled      = false,
        vm_enabled       = false,
        vg_enabled       = false,
        snapshot_enabled = false,
        show_progress    = true,
        minirhizotron    = cfg.mr,
        mr_csv_path      = mr_csv_path
    )

    @info "Reading simulation output..." path=mr_csv_path
    df_sim = CSV.read(mr_csv_path, DataFrame)

    # Acquisition days (sim day = days after sowing + 1)
    acq_simdays = [Int(d) + 1 for d in ACQ_OFFSETS]
    acq_dates   = [SOWING_DATE + Day(d) for d in ACQ_OFFSETS]
    simday_to_date = Dict(sd => dt for (sd, dt) in zip(acq_simdays, acq_dates))

    df_obs = DataFrame(
        date           = String[],
        station_idx    = Int[],
        along_tube_cm  = Float64[],
        depth_cm       = Float64[],
        azimuth_deg    = Float64[],
        root_length_cm = Float64[]
    )

    for sd in acq_simdays
        haskey(simday_to_date, sd) || continue
        dt = simday_to_date[sd]
        sub = df_sim[df_sim.day .== sd, :]
        nrow(sub) == 0 && continue

        for row in eachrow(sub)
            val   = Float64(row.length_cm)
            noisy = val > 0 ? val * exp(σ_ln * randn() - σ_ln^2 / 2) : 0.0
            noisy = max(0.0, noisy)
            push!(df_obs, (
                string(dt),
                Int(row.station_idx),
                Float64(row.along_tube_cm),
                Float64(row.depth_cm),
                Float64(row.azimuth_deg),
                noisy
            ))
        end
    end

    out_csv = joinpath(@__DIR__, "data", "mr_obs_stations_1tube.csv")
    CSV.write(out_csv, df_obs)
    @info "Wrote mock observations" path=out_csv nrow=nrow(df_obs) dates=length(acq_simdays)

    # Summary
    for (sd, dt) in sort(collect(simday_to_date))
        sub = df_obs[df_obs.date .== string(dt), :]
        if nrow(sub) > 0
            @printf("  %s (day %3d): %d rows, total=%.3f cm\n",
                dt, sd, nrow(sub), sum(sub.root_length_cm))
        end
    end
end

main()
