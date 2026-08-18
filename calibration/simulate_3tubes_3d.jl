# simulate_3tubes_3d.jl
#
# Forward simulation with the 3-tube layout and 3-tube calibrated parameters.
# Outputs 3D root figures (PNG) every SAVE_EVERY days for each tube position.
#
# Inputs:
#   calibration/ParameterVD_best_mr_3tubes.json
#   calibration/Parameter_calib_minirhizotron.json
#
# Outputs in calibration/simulation_3tubes_3d/:
#   tube1_x25/plot3d/root_day_XXX.png
#   tube2_x50/plot3d/root_day_XXX.png
#   tube3_x75/plot3d/root_day_XXX.png
#
# Run from repo root:
#   julia calibration/simulate_3tubes_3d.jl

using JSON
using JSON3
using Dates

const TUBE_X_CM  = [25.0, 50.0, 75.0]
const SIM_DAYS   = 90
const SAVE_EVERY = 5        # 3D figure every N days (18 frames over 90 days)
const SIM_SEED   = 42

include(joinpath(@__DIR__, "..", "src", "Deps.jl"))
include(joinpath(@__DIR__, "..", "src", "Types.jl"))
include(joinpath(@__DIR__, "..", "src", "Geometry.jl"))
include(joinpath(@__DIR__, "..", "src", "DensityRegulation.jl"))
include(joinpath(@__DIR__, "..", "src", "GrowthStep.jl"))
include(joinpath(@__DIR__, "..", "src", "MetricsAndOutput.jl"))
include(joinpath(@__DIR__, "..", "src", "Plotting.jl"))
include(joinpath(@__DIR__, "..", "src", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "Simulation.jl"))

read_json(path) = open(path, "r") do io; JSON.parse(String(read(io))); end
function write_json_pretty(path, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

function main()
    env_base    = read_json(joinpath(@__DIR__, "Parameter_calib_minirhizotron.json"))
    best_params = joinpath(@__DIR__, "ParameterVD_best_mr_3tubes.json")
    outroot     = joinpath(@__DIR__, "simulation_3tubes_3d")
    mkpath(outroot)

    for (tid, tx) in enumerate(TUBE_X_CM)
        label  = "tube$(tid)_x$(Int(tx))"
        outdir = joinpath(outroot, label)
        mkpath(outdir)

        env = deepcopy(env_base)
        env["species_params_files"]["VD"]      = best_params
        env["simulation"]["seed"]              = SIM_SEED
        env["simulation"]["days"]              = SIM_DAYS
        env["simulation"]["save_every"]        = SAVE_EVERY
        env["simulation"]["outdir"]            = outdir
        env["simulation"]["plot3d_outdir"]     = joinpath(outdir, "plot3d")
        env["minirhizotron"]["tube_origin_cm"] = [tx, 20.0, 0.0]
        env["facilitation"]["enabled"]         = false
        env["density_regulation"]["enabled"]   = false
        env["visualization_2d"]["enabled"]     = false

        env_path = joinpath(outdir, "env.json")
        write_json_pretty(env_path, env)

        @info "Running tube $tid (x=$(tx) cm)..."
        cfg = load_environment_and_plants(env_path)
        simulate!(cfg.plants, cfg.box, cfg.reg;
            days             = SIM_DAYS,
            seed             = SIM_SEED,
            save_every       = SAVE_EVERY,
            outdir           = cfg.outdir,
            plot3d_dir       = cfg.plot3d_dir,
            viz_enabled      = false,
            vm_enabled       = false,
            vg_enabled       = false,
            snapshot_enabled = false,
            show_progress    = true,
            minirhizotron    = cfg.mr,
            mr_csv_path      = joinpath(outdir, "tube_stations_daily.csv")
        )

        n_frames = length(filter(f -> endswith(f, ".png"), readdir(cfg.plot3d_dir)))
        @info "Tube $tid done" frames=n_frames outdir=cfg.plot3d_dir
    end

    println("\nAll done. Images saved to: $outroot")
    println("Each tube subfolder contains plot3d/root_day_XXX.png")
end

main()
