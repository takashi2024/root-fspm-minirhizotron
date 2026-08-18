#!/usr/bin/env julia
# explore_skin_depth.jl
#
# Compares root visibility at two skin depths (1.0 cm vs 0.2 cm) without
# re-running any simulation twice. Each parameter set is simulated once to
# day 84 with a single seed; then tube_station_azimuth_lengths_cm is called
# twice from the same grown root state, using a reconstructed Minirhizotron
# with skin_depth_cm = 0.2 instead of 1.0.
#
# Usage (from repo root):
#   julia sbi/julia/explore_skin_depth.jl [N_PER_CONFIG]
#
# N_PER_CONFIG: how many samples to draw per planting config (default 30).
# Total runtime ≈ N_PER_CONFIG × 4 × 5-6 seconds ≈ 10 min for default N=30.
#
# Output: printed summary table + CSV at sbi/data/skin_depth_explore.csv

using JSON, Printf, CSV, DataFrames, Statistics, Random

const REPO_ROOT = dirname(dirname(dirname(abspath(@__FILE__))))

for f in ["Deps", "Types", "Geometry", "DensityRegulation",
          "GrowthStep", "MetricsAndOutput", "Config"]
    include(joinpath(REPO_ROOT, "src", "$(f).jl"))
end
include(joinpath(@__DIR__, "SimRunner.jl"))

# ── Configuration ──────────────────────────────────────────────────────────────

const CONFIGS = [
    ("A",  "sbi/configs/sbi_config_A.json"),
    ("B2", "sbi/configs/sbi_config_B2.json"),
    ("C",  "sbi/configs/sbi_config_C.json"),
    ("D2", "sbi/configs/sbi_config_D2.json"),
]

const SKIN_DEPTHS = [1.0, 0.2]   # cm; 1.0 = current, 0.2 = proposed
const EVAL_DAY    = 84
const EVAL_SEED   = 9001          # separate from dataset seeds (≥ 7001) and CMA-ES (≤ 99)
const MIN_WINDOWS = 10            # identifiability threshold used in test case selection

# ── Helper: construct a Minirhizotron with a different skin depth ───────────────

function with_skin_depth(mr::Minirhizotron, d::Float64)::Minirhizotron
    Minirhizotron(
        mr.enabled,
        mr.origin,
        mr.direction,
        mr.length_cm,
        mr.r_outer_cm,
        d,
        mr.image_along_cm,
        mr.image_arc_cm,
        mr.station_positions_cm,
        mr.sample_step_cm,
        mr.csv_name,
        mr.camera_azimuths_deg,
    )
end

# ── Helper: grow plants to EVAL_DAY, return (plants, mr) ──────────────────────

function grow_to_eval_day(
    params::NamedTuple,
    base_env_path::String,
    vd_template_path::String,
    tmp_dir::String,
)
    sp_path  = joinpath(tmp_dir, "sp_explore.json")
    env_path = joinpath(tmp_dir, "env_explore.json")
    _write_candidate_species_json(vd_template_path, params, sp_path)
    _write_candidate_env_json(base_env_path, sp_path, EVAL_SEED, EVAL_DAY,
                               tmp_dir, env_path)

    cfg = load_environment_and_plants(env_path)
    Random.seed!(EVAL_SEED)
    R_dummy = zeros(Float64, 1, 1)
    for day in 1:EVAL_DAY
        for pl in cfg.plants
            step_plant!(pl, cfg.box, cfg.reg, R_dummy, R_dummy, 1, 1;
                        mature_branching=false, facilitation=nothing,
                        facil_gates=nothing, minirhizotron=nothing)
        end
    end

    rm(sp_path,  force=true)
    rm(env_path, force=true)
    return cfg.plants, cfg.mr
end

# ── Main ───────────────────────────────────────────────────────────────────────

function main(n_per_config::Int)
    results = DataFrame(
        config        = String[],
        sample_id     = Int[],
        g_axial       = Float64[],
        bp1           = Float64[],
        skin_depth_cm = Float64[],
        occ_windows   = Int[],
        mean_rld      = Float64[],
        max_rld       = Float64[],
    )

    tmp_base = joinpath(REPO_ROOT, "sbi", "data", "_explore_tmp_")
    mkpath(tmp_base)

    for (cfg_tag, cfg_rel) in CONFIGS
        cfg_path = joinpath(REPO_ROOT, cfg_rel)
        cfg_json = open(cfg_path, "r") do io; JSON.parse(String(read(io))); end

        base_env_path    = joinpath(REPO_ROOT, cfg_json["base_env_json"])
        vd_template_path = joinpath(REPO_ROOT, cfg_json["vd_template_json"])

        data_dir  = joinpath(REPO_ROOT, "sbi", "data", "multi_seed", "config_$(cfg_tag)")
        csv_path  = joinpath(data_dir, "samples_filtered.csv")
        df        = CSV.read(csv_path, DataFrame)
        df        = df[1:min(n_per_config, nrow(df)), :]

        n_cfg = nrow(df)
        @printf("\n── Config %s: %d samples ──────────────────────\n", cfg_tag, n_cfg)

        tmp_dir = joinpath(tmp_base, cfg_tag)
        mkpath(tmp_dir)

        for (i, row) in enumerate(eachrow(df))
            params = (;
                elongation_rate_min     = Float64(row.elongation_rate_min),
                elongation_rate_max     = Float64(row.elongation_rate_max),
                elongation_logistic_k   = Float64(row.elongation_logistic_k),
                elongation_logistic_t50 = Float64(row.elongation_logistic_t50),
                branch_angle_order1     = Float64(row.branch_angle_order1),
                g_axial                 = Float64(row.g_axial),
                branching_prob_order1   = Float64(row.branching_prob_order1),
                branching_prob_order2   = Float64(row.branching_prob_order2),
                n_axial_roots           = Int(row.n_axial_roots),
                max_order1_per_plant    = Int(row.max_order1_per_plant),
            )

            plants, mr_orig = grow_to_eval_day(params, base_env_path,
                                               vd_template_path, tmp_dir)

            ns = length(mr_orig.station_positions_cm)
            na = length(mr_orig.camera_azimuths_deg)

            for sd in SKIN_DEPTHS
                mr_sd  = with_skin_depth(mr_orig, sd)
                lens   = tube_station_azimuth_lengths_cm(plants, mr_sd)  # ns × na matrix
                occ    = count(x -> x > 1e-6, lens)
                mean_r = occ > 0 ? mean(lens[lens .> 1e-6]) : 0.0
                max_r  = maximum(lens)

                push!(results, (
                    cfg_tag,
                    Int(row.sample_id),
                    Float64(row.g_axial),
                    Float64(row.branching_prob_order1),
                    sd,
                    occ,
                    mean_r,
                    max_r,
                ))
            end

            @printf("  [%d/%d] s%04d  g=%.3f  bp1=%.2f  occ@1cm=%d  occ@2mm=%d\n",
                    i, n_cfg, Int(row.sample_id),
                    Float64(row.g_axial), Float64(row.branching_prob_order1),
                    results[end-1, :occ_windows],
                    results[end,   :occ_windows])
        end
    end

    rm(tmp_base; recursive=true, force=true)

    # ── Summary statistics ────────────────────────────────────────────────────

    println()
    println("=" ^ 72)
    println("Skin Depth Comparison — Root Visibility Reduction")
    @printf("  Evaluation day : %d\n", EVAL_DAY)
    @printf("  Min windows    : %d  (identifiability threshold)\n", MIN_WINDOWS)
    println("=" ^ 72)

    for (cfg_tag, _) in CONFIGS
        sub = filter(r -> r.config == cfg_tag, results)
        if nrow(sub) == 0; continue; end

        println("\nConfig $(cfg_tag):")
        println("  skin_depth │ zero-obs │ <10-win  │ ≥10-win  │ mean occ │ mean RLD (cm)")
        println("  ───────────┼──────────┼──────────┼──────────┼──────────┼──────────────")

        for sd in SKIN_DEPTHS
            s = filter(r -> r.skin_depth_cm == sd, sub)
            n = nrow(s)
            n_zero  = count(r -> r.occ_windows == 0, eachrow(s))
            n_few   = count(r -> 0 < r.occ_windows < MIN_WINDOWS, eachrow(s))
            n_ok    = count(r -> r.occ_windows >= MIN_WINDOWS, eachrow(s))
            mean_occ = mean(s.occ_windows)
            rld_ok   = filter(r -> r.occ_windows > 0, eachrow(s))
            mean_rld = length(rld_ok) > 0 ? mean(r.mean_rld for r in rld_ok) : 0.0
            @printf("  %8.1f cm │ %5d(%3.0f%%)│ %5d(%3.0f%%)│ %5d(%3.0f%%)│ %8.1f │ %10.3f\n",
                    sd,
                    n_zero, 100*n_zero/n,
                    n_few,  100*n_few/n,
                    n_ok,   100*n_ok/n,
                    mean_occ,
                    mean_rld)
        end
    end

    # ── Reduction ratio across all configs ───────────────────────────────────

    println()
    println("Overall reduction in occupied windows (2 mm vs 1 cm):")
    paired = innerjoin(
        filter(r -> r.skin_depth_cm == 1.0, results)[:, [:config, :sample_id, :occ_windows]],
        filter(r -> r.skin_depth_cm == 0.2, results)[:, [:config, :sample_id, :occ_windows]],
        on=[:config, :sample_id], makeunique=true,
    )
    rename!(paired, :occ_windows => :occ_1cm, :occ_windows_1 => :occ_2mm)
    ratios = [r.occ_2mm / max(r.occ_1cm, 1) for r in eachrow(paired)]
    @printf("  Median retention : %.1f%%\n", 100 * median(ratios))
    @printf("  Mean retention   : %.1f%%\n", 100 * mean(ratios))
    @printf("  Cases with 0 win at 2mm but ≥1 at 1cm : %d\n",
            count(r -> r.occ_2mm == 0 && r.occ_1cm > 0, eachrow(paired)))
    @printf("  Cases with ≥10 win at 1cm but <10 at 2mm : %d\n",
            count(r -> r.occ_1cm >= MIN_WINDOWS && r.occ_2mm < MIN_WINDOWS, eachrow(paired)))

    # ── Save CSV ──────────────────────────────────────────────────────────────

    out_csv = joinpath(REPO_ROOT, "sbi", "data", "skin_depth_explore.csv")
    CSV.write(out_csv, results)
    @printf("\nDetailed results → %s\n", out_csv)
end

# ── Entry point ────────────────────────────────────────────────────────────────

n_per_config = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
@printf("Exploring skin depth effect: N=%d samples per config × 4 configs\n", n_per_config)
@printf("Simulation seed: %d  |  Evaluation day: %d\n\n", EVAL_SEED, EVAL_DAY)
main(n_per_config)
