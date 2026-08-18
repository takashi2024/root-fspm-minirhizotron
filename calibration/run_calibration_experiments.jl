# run_calibration_experiments.jl
#
# Sequential experiment runner: calibrates all 8 objective modes one after another.
# Reads config files from calibration/configs/calib_*.json.
# Writes a comparison summary to calibration/experiments/calibration_experiment_summary.csv.
#
# Usage:
#   julia calibration/run_calibration_experiments.jl
#
# To run a subset, pass config file names as arguments:
#   julia calibration/run_calibration_experiments.jl calib_multi_objective.json calib_replicate_multi.json

using JSON
using JSON3
using CSV
using DataFrames
using Printf

configs_dir  = joinpath(@__DIR__, "configs")
output_base  = joinpath(@__DIR__, "experiments")
summary_csv  = joinpath(output_base, "calibration_experiment_summary.csv")

# Discover configs
all_configs = sort(filter(f -> startswith(f, "calib_") && endswith(f, ".json"),
                          readdir(configs_dir)))

# If specific files were passed as CLI args, filter to those
if !isempty(ARGS)
    requested = Set(ARGS)
    all_configs = filter(c -> c in requested, all_configs)
    isempty(all_configs) && error("None of the requested configs found in $configs_dir\nAvailable: $(join(readdir(configs_dir), ", "))")
end

isempty(all_configs) && error("No config files found in $configs_dir")

println("═"^62)
@printf("  Flexible Calibration Experiment Runner\n")
@printf("  Running %d experiment(s):\n", length(all_configs))
for c in all_configs
    println("    • $c")
end
println("═"^62)

summary_rows = DataFrame(
    objective          = String[],
    tube_setup         = String[],
    best_J             = Float64[],
    diag_rmse_cm       = Float64[],
    diag_r2            = Float64[],
    diag_pbias_pct     = Float64[],
    obs_zero_fraction  = Float64[],
    runtime_minutes    = Float64[],
    calibration_seeds  = String[],
    sigma0             = Float64[],
    maxiter            = Int[],
    exp_dir            = String[]
)

for (idx, cfg_file) in enumerate(all_configs)
    cfg_path = joinpath(configs_dir, cfg_file)
    println("\n" * "─"^62)
    @printf("  [%d/%d] Running: %s\n", idx, length(all_configs), cfg_file)
    println("─"^62)

    t0 = time()
    local exp_dir_out = ""
    try
        # Run via subprocess using the same Julia binary as the parent process
        julia_bin = joinpath(Sys.BINDIR, "julia")
        script = joinpath(@__DIR__, "main_Optimization_mr_flexible.jl")
        run(`$julia_bin $script $cfg_path`)
        t_elapsed = time() - t0

        # Parse run_summary.json from the most recent experiment matching this config
        obj_name = begin
            d = open(cfg_path, "r") do io; JSON.parse(String(read(io))); end
            get(d, "calibration_objective", splitext(cfg_file)[1])
        end
        # Find the newest experiment directory for this objective
        exp_dirs = filter(d -> startswith(d, obj_name),
                          readdir(output_base; join=false))
        if isempty(exp_dirs)
            @warn "No experiment directory found for $obj_name after run"
            continue
        end
        newest_dir = joinpath(output_base, sort(exp_dirs)[end])
        summary_path = joinpath(newest_dir, "run_summary.json")
        if !isfile(summary_path)
            @warn "run_summary.json not found in $newest_dir"
            continue
        end
        s = open(summary_path, "r") do io; JSON.parse(String(read(io))); end

        push!(summary_rows, (
            String(get(s, "objective",         obj_name)),
            String(get(s, "tube_setup",        "1tube")),
            Float64(get(s, "best_J",            NaN)),
            Float64(get(s, "diag_rmse",         NaN)),
            Float64(get(s, "diag_r2",           NaN)),
            Float64(get(s, "diag_pbias",        NaN)),
            Float64(get(s, "obs_zero_fraction", NaN)),
            Float64(get(s, "runtime_seconds",   t_elapsed)) / 60,
            join(get(s, "calibration_seeds", Int[]), ","),
            Float64(get(s, "sigma0",   0.1)),
            Int(get(s, "maxiter", 100)),
            newest_dir
        ))
        exp_dir_out = newest_dir
        @printf("  ✓ Completed in %.1f min → %s\n", t_elapsed/60, newest_dir)

    catch e
        @warn "Experiment failed for $cfg_file: $e"
        push!(summary_rows, (
            cfg_file, "?", NaN, NaN, NaN, NaN, NaN, (time()-t0)/60, "", NaN, 0, exp_dir_out
        ))
    end
end

# Write summary CSV
mkpath(output_base)
CSV.write(summary_csv, summary_rows)

println("\n" * "═"^62)
println("  Experiment summary")
println("═"^62)
@printf("  %-40s  %8s  %8s  %8s\n", "Objective", "Best J", "R²", "PBIAS%")
println("  " * "-"^58)
for r in eachrow(summary_rows)
    @printf("  %-40s  %8.3f  %8.3f  %7.1f%%\n",
            r.objective, r.best_J, r.diag_r2, r.diag_pbias_pct)
end
println("═"^62)
@printf("  Summary saved → %s\n", summary_csv)
