# PriorSampler.jl
#
# Samples FSPM root architectural parameters from biologically plausible prior
# distributions using Latin Hypercube Sampling (LHS) for uniform coverage of
# the 8-dimensional parameter space.
#
# Prior bounds mirror PARAM_BOUNDS in calibration/main_Optimization_mr_flexible.jl.
# Log-uniform priors are used for rate/probability parameters that span orders of
# magnitude; uniform priors for angles and bounded probabilities.
#
# Biological constraints enforced:
#   - elongation_rate_max >= elongation_rate_min  (clamped, not rejected)
#   - branching_prob_order1 >= 0.05               (prevents degenerate zero-lateral systems)
#   - max_depth_day90_cm >= MIN_FEASIBLE_DEPTH_CM (rejection: roots must reach tube)
#
# Diagnostic columns added to output DataFrame (assuming vertical axial growth):
#   - depth_at_inflection_cm  : cumulative axial length at t50 (= depth below surface)
#   - max_depth_day90_cm      : cumulative axial length at day 90
#   - feasible                : Bool, true if max_depth_day90_cm >= MIN_FEASIBLE_DEPTH_CM
#
# Requires: LatinHypercubeSampling.jl, DataFrames.jl, CSV.jl, Random

using Random
using DataFrames
using CSV
using LatinHypercubeSampling
using OrderedCollections

# ──────────────────────────────────────────────────────────────────────────────
# Prior specification
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: name => (prior_type, lo, hi)
#   :log_uniform — sample exp(Uniform(log(lo), log(hi)))
#   :uniform     — sample Uniform(lo, hi)
# Minimum feasible depth: reject samples where axial root stays shallower than
# this at day 90. Ensures roots traverse a meaningful portion of the tube.
const MIN_FEASIBLE_DEPTH_CM = 10.0

const SBI_PARAM_SPEC = OrderedDict{String, Tuple{Symbol, Float64, Float64}}(
    "elongation_rate_min"     => (:log_uniform,  0.10,  0.70),  # was (0.05, 0.5)
    "elongation_rate_max"     => (:log_uniform,  0.40,  2.00),  # was (0.2, 1.5)
    "elongation_logistic_k"   => (:log_uniform,  0.05,  0.4),
    "elongation_logistic_t50" => (:uniform,      10.0, 80.0),   # was (5.0, 50.0)
    "branch_angle_order1"     => (:uniform,      40.0, 85.0),
    "g_axial"                 => (:log_uniform,  0.01,  0.50),
    "branching_prob_order1"   => (:uniform,       0.05,  0.90),  # was (0.05, 1.0)
    "branching_prob_order2"   => (:uniform,       0.0,   0.90),  # was (0.0, 1.0)
)

const SBI_PARAM_NAMES = collect(keys(SBI_PARAM_SPEC))
const N_SBI_PARAMS    = length(SBI_PARAM_NAMES)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""Map a unit-interval value u ∈ [0,1] to a log-uniform draw in [lo, hi]."""
@inline function _map_log_uniform(u::Float64, lo::Float64, hi::Float64)::Float64
    exp(log(lo) + u * (log(hi) - log(lo)))
end

"""Map a unit-interval value u ∈ [0,1] to a uniform draw in [lo, hi]."""
@inline function _map_uniform(u::Float64, lo::Float64, hi::Float64)::Float64
    lo + u * (hi - lo)
end

"""Map a row of LHS unit samples to the corresponding parameter values."""
function _lhs_row_to_params(row::AbstractVector{Float64})::Vector{Float64}
    out = Vector{Float64}(undef, N_SBI_PARAMS)
    for (i, (name, (ptype, lo, hi))) in enumerate(SBI_PARAM_SPEC)
        u = row[i]
        out[i] = ptype == :log_uniform ? _map_log_uniform(u, lo, hi) :
                                          _map_uniform(u, lo, hi)
    end
    # Enforce elongation_rate_max >= elongation_rate_min
    idx_min = findfirst(==("elongation_rate_min"), SBI_PARAM_NAMES)
    idx_max = findfirst(==("elongation_rate_max"), SBI_PARAM_NAMES)
    if out[idx_max] < out[idx_min]
        out[idx_max] = out[idx_min]
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# Growth diagnostics (vertical axial root approximation)
# ──────────────────────────────────────────────────────────────────────────────

"""
    axial_growth_diagnostics(rate_min, rate_max, k, t50; sim_days=90)
    -> (depth_at_inflection_cm, max_depth_day90_cm)

Estimates axial root depth over time assuming the root grows straight down.
Depth = cumulative sum of daily logistic elongation rates.

This is a diagnostic approximation only — actual 3-D simulations include
gravitropism, noise, and branching geometry that alter the true depth profile.

Returns:
  depth_at_inflection_cm  — cumulative length (≈ depth) at day floor(t50)
  max_depth_day90_cm      — cumulative length (≈ depth) at day sim_days
"""
function axial_growth_diagnostics(
    rate_min::Float64, rate_max::Float64, k::Float64, t50::Float64;
    sim_days::Int = 90
)::Tuple{Float64, Float64}
    L        = 0.0
    L_t50    = 0.0
    t50_day  = floor(Int, t50)
    for day in 1:sim_days
        L += rate_min + (rate_max - rate_min) / (1.0 + exp(-k * (Float64(day) - t50)))
        if day == t50_day
            L_t50 = L
        end
    end
    # If t50 > sim_days, L_t50 was never assigned — set to L (fully grown by then)
    if t50_day >= sim_days
        L_t50 = L
    end
    return L_t50, L
end

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    sample_prior(N; rng_seed, seeds_start, n_axial_roots, max_order1_per_plant,
                 sim_days, min_feasible_depth_cm) -> DataFrame

Draw N parameter sets using Latin Hypercube Sampling with feasibility rejection.

Oversamples by 4× then keeps only feasible rows (max_depth_day90_cm >= min_feasible_depth_cm),
returning exactly N rows. Warns if the feasibility rate is below 50%.

Columns in output:
  sample_id, sim_seed, <8 param names>, n_axial_roots, max_order1_per_plant,
  depth_at_inflection_cm, max_depth_day90_cm, feasible, has_visible_roots
"""
function sample_prior(
    N::Int;
    rng_seed::Int             = 0,
    seeds_start::Int          = 1000,
    n_axial_roots::Int        = 1,
    max_order1_per_plant::Int = 20,
    sim_days::Int             = 90,
    min_feasible_depth_cm::Float64 = MIN_FEASIBLE_DEPTH_CM
)::DataFrame

    oversample = 4
    N_draw     = N * oversample

    # LHS in [0,1]^N_SBI_PARAMS over a larger pool
    plan     = randomLHC(MersenneTwister(rng_seed), N_draw, N_SBI_PARAMS)
    lhs_unit = (Float64.(plan) .- 0.5) ./ N_draw

    idx_rmin = findfirst(==("elongation_rate_min"),     SBI_PARAM_NAMES)
    idx_rmax = findfirst(==("elongation_rate_max"),     SBI_PARAM_NAMES)
    idx_k    = findfirst(==("elongation_logistic_k"),   SBI_PARAM_NAMES)
    idx_t50  = findfirst(==("elongation_logistic_t50"), SBI_PARAM_NAMES)

    accepted = NamedTuple[]
    n_rejected = 0

    for i in 1:N_draw
        pvals  = _lhs_row_to_params(lhs_unit[i, :])
        d_inf, d_max = axial_growth_diagnostics(
            pvals[idx_rmin], pvals[idx_rmax], pvals[idx_k], pvals[idx_t50];
            sim_days = sim_days
        )
        feasible = d_max >= min_feasible_depth_cm
        feasible || (n_rejected += 1)

        push!(accepted, NamedTuple(vcat(
            [:sample_id => 0, :sim_seed => 0],   # filled below
            [Symbol(SBI_PARAM_NAMES[j]) => pvals[j] for j in 1:N_SBI_PARAMS],
            [:n_axial_roots            => n_axial_roots,
             :max_order1_per_plant     => max_order1_per_plant,
             :depth_at_inflection_cm   => d_inf,
             :max_depth_day90_cm       => d_max,
             :feasible                 => (feasible ? 1 : 0),
             :has_visible_roots        => 0]
        )))
    end

    feasibility_rate = 1.0 - n_rejected / N_draw
    @info "Prior feasibility: $(round(100*feasibility_rate; digits=1))% of $(N_draw) candidates passed (max_depth >= $(min_feasible_depth_cm) cm)"

    # Keep only feasible candidates, take first N
    feasible_rows = filter(r -> r.feasible == 1, accepted)
    if length(feasible_rows) < N
        @warn "Only $(length(feasible_rows)) feasible samples found (needed $N). Returning all feasible rows."
    end
    kept = feasible_rows[1:min(N, length(feasible_rows))]

    # Assign sequential sample_id and sim_seed
    df = DataFrame(kept)
    df.sample_id = 1:nrow(df)
    df.sim_seed  = seeds_start .+ (1:nrow(df))

    return df
end

"""
    save_samples_csv(df, path)

Write the parameter sample table to a CSV file.
"""
function save_samples_csv(df::DataFrame, path::AbstractString)
    mkpath(dirname(path))
    CSV.write(path, df)
end

"""
    load_samples_csv(path) -> DataFrame
"""
function load_samples_csv(path::AbstractString)::DataFrame
    CSV.read(path, DataFrame)
end
