#!/usr/bin/env julia
# run_layout_diagnostics_only.jl
#
# Runs layout diagnostics for one or more SBI environment configs WITHOUT
# running the full simulation. Only plant positions (set at load time) matter.
#
# Usage:
#   julia sbi/julia/run_layout_diagnostics_only.jl <config.json> [config2.json ...]
#   julia sbi/julia/run_layout_diagnostics_only.jl  # defaults to all 4 configs

const REPO_ROOT = dirname(dirname(dirname(abspath(@__FILE__))))

# Include all src/ modules needed by Config.jl and LayoutDiagnostics.jl
for f in ["Deps.jl", "Types.jl", "Geometry.jl", "DensityRegulation.jl",
          "GrowthStep.jl", "MetricsAndOutput.jl", "Config.jl",
          "LayoutDiagnostics.jl"]
    include(joinpath(REPO_ROOT, "src", f))
end

default_configs = [
    joinpath(REPO_ROOT, "sbi", "configs", "Parameter_SBI_env_A.json"),
    joinpath(REPO_ROOT, "sbi", "configs", "Parameter_SBI_env_B.json"),
    joinpath(REPO_ROOT, "sbi", "configs", "Parameter_SBI_env_C.json"),
    joinpath(REPO_ROOT, "sbi", "configs", "Parameter_SBI_env_D.json"),
]

config_paths = length(ARGS) >= 1 ? ARGS : default_configs

for cfg_path in config_paths
    @info "Running layout diagnostics" config=cfg_path
    cfg     = load_environment_and_plants(cfg_path)
    outdir  = cfg.layout_diag_outdir
    mkpath(outdir)
    save_layout_diagnostics(cfg.box, cfg.mr, cfg.plants, outdir)
    @info "Done" outdir=outdir n_plants=length(cfg.plants)
end
