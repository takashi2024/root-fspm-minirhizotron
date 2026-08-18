#!/usr/bin/env julia
# sample_prior_only.jl
#
# Generates the parameter sample table (N=500) with growth diagnostics,
# WITHOUT running any simulations. Use this to review parameter coverage
# before committing to full dataset generation.
#
# Usage (from repo root):
#   julia sbi/julia/sample_prior_only.jl [sbi/configs/sbi_config.json]
#
# Output:
#   sbi/data/samples_preview.csv   (or outdir from config)

using JSON
using Printf
using Statistics

const REPO_ROOT = dirname(dirname(dirname(abspath(@__FILE__))))

include(joinpath(@__DIR__, "PriorSampler.jl"))

config_path = length(ARGS) >= 1 ? ARGS[1] :
              joinpath(REPO_ROOT, "sbi", "configs", "sbi_config.json")

cfg    = JSON.parsefile(config_path)
outdir = joinpath(REPO_ROOT, cfg["outdir"])
mkpath(outdir)

N          = cfg["n_samples"]
seeds_start = get(cfg, "seeds_start", 1000)
sim_days    = get(cfg, "days", 90)
n_axial     = get(cfg, "n_axial_roots", 1)
max_order1  = get(cfg, "max_order1_per_plant", 20)

@info "Sampling prior: N=$N, sim_days=$sim_days"
df = sample_prior(N;
    rng_seed             = 0,
    seeds_start          = seeds_start,
    n_axial_roots        = n_axial,
    max_order1_per_plant = max_order1,
    sim_days             = sim_days,
    min_feasible_depth_cm = MIN_FEASIBLE_DEPTH_CM,
)

out_path = joinpath(outdir, "samples_preview.csv")
save_samples_csv(df, out_path)
@info "Saved $(nrow(df)) rows → $out_path"

# Print a summary
println("\n=== Parameter + depth diagnostic summary (N=$(nrow(df))) ===")
for col in [:elongation_rate_min, :elongation_rate_max, :elongation_logistic_k,
            :elongation_logistic_t50, :depth_at_inflection_cm, :max_depth_day90_cm]
    vals = df[!, col]
    @printf("  %-28s  min=%6.2f  median=%6.2f  max=%6.2f\n",
            string(col), minimum(vals), median(vals), maximum(vals))
end
println()
n_feas = sum(df.feasible)
@printf("  Feasible (max_depth >= %.0f cm): %d / %d (%.0f%%)\n",
        MIN_FEASIBLE_DEPTH_CM, n_feas, nrow(df), 100*n_feas/nrow(df))
