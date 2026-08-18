# generate_mock_visualizations_200cm.jl
#
# Runs the 200 cm box template simulation (seed=42) and saves:
#   - plot3d/root_day_XXX.png  : 3D root figure every 7 days
#   - root2d/root2d_day_XXX.png: 2D plate view (X–Depth) every 7 days
#
# Both include the minirhizotron tube at x=100 cm overlaid.
#
# Run: julia calibration/generate_mock_visualizations_200cm.jl  (from repo root)

using JSON
using Dates

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
    open(path, "w") do io; JSON.print(io, obj, 2); end
end

const N_DAYS   = 92     # cover day 91 (last weekly observation)
const SAVE_EVERY = 7    # save every 7 days
const MOCK_SEED  = 42
const OUTDIR   = joinpath(@__DIR__, "mock_visualizations_200cm")

env_path_base = joinpath(@__DIR__, "Parameter_calib_minirhizotron_200cm.json")
env = read_json(env_path_base)

env["species_params_files"]["VD"] = joinpath(@__DIR__, "Parameter_VD_template.json")
env["simulation"]["seed"]         = MOCK_SEED
env["simulation"]["days"]         = N_DAYS
env["simulation"]["save_every"]   = SAVE_EVERY
env["simulation"]["outdir"]       = OUTDIR
env["simulation"]["plot3d_outdir"] = joinpath(OUTDIR, "plot3d")

tmpdir   = joinpath(@__DIR__, "__mock_viz_tmp__")
mkpath(tmpdir)
env_tmp  = joinpath(tmpdir, "Parameter_viz_tmp.json")
write_json_pretty(env_tmp, env)

@info "Loading environment and plants..."
cfg = load_environment_and_plants(env_tmp)

plot3d_dir = joinpath(OUTDIR, "plot3d")
root2d_dir = joinpath(OUTDIR, "root2d")
mkpath(plot3d_dir)
mkpath(root2d_dir)

@info "Running simulation..." days=N_DAYS seed=MOCK_SEED save_every=SAVE_EVERY
simulate!(cfg.plants, cfg.box, cfg.reg;
    days              = N_DAYS,
    seed              = MOCK_SEED,
    save_every        = SAVE_EVERY,          # 3D plot every 7 days → plot3d/
    plot3d_dir        = plot3d_dir,
    viz_enabled       = true,
    viz_save_every    = SAVE_EVERY,          # 2D plate view every 7 days → root2d/
    viz_outdir        = root2d_dir,
    viz_project_to_plate = true,
    viz_y_max_cm      = Float64(cfg.box.D),  # show full soil depth
    viz_linewidth     = 1.5,
    vm_enabled        = false,
    vg_enabled        = false,
    snapshot_enabled  = false,
    show_progress     = true,
    minirhizotron     = cfg.mr,
    mr_csv_path       = joinpath(tmpdir, "tube_stations_daily.csv")
)

@info "Done." plot3d=plot3d_dir root2d=root2d_dir
@info "Files saved:" n_3d=length(filter(f->endswith(f,".png"), readdir(plot3d_dir))) \
                     n_2d=length(filter(f->endswith(f,".png"), readdir(root2d_dir)))
