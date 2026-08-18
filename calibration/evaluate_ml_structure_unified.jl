# evaluate_ml_structure_unified.jl
#
# Unified structural similarity evaluation for all scenarios and k values.
# Replaces the patchwork of per-scenario scripts and uses consistent true params.
#
# Usage:
#   julia calibration/evaluate_ml_structure_unified.jl <mode> [tag]
#
# mode is one of:
#   s2_k1    — Scenario 2, k=1 obs: CMA-ES (tc_v2_*_s2)        + TabICL-MS-k1
#   s2_k7    — Scenario 2, k=7 obs: CMA-ES (tc_v2_*_s2ms)      + TabICL-MS-v2
#   s2_k14   — Scenario 2, k=14 obs: CMA-ES (tc_v2_*_s2ms_k14) + TabICL-MS-k14
#   s1_k1    — Scenario 1, k=1 obs: CMA-ES (tc_v2_*_s1)        + TabICL-MS-k1
#   s1_k7    — Scenario 1, k=7 obs: CMA-ES (tc_v2_*_s1ms)      + TabICL-MS-v2
#   s1_k14   — Scenario 1, k=14 obs: CMA-ES (tc_v2_*_s1ms_k14) + TabICL-MS-k14
#
# [tag]  optional — single-case mode, e.g. "tc_v2_A_254_s2ms"
#        (writes per-case CSV to ml_struct_tmp_<mode>/<tag>.csv; skips aggregate)
#
# Notes:
#   - All modes use N=20 test cases (no EXCLUDED filter)
#   - k=1 uses TabICL-MS-k1 (consistent pipeline across k values)
#   - True params come from test_cases_selected.csv (all 8 params correct)
#   - S2 modes: elongation fixed at true values (warmstart); 4 arch params predicted
#   - S1 modes: all 8 params predicted by CMA-ES and TabICL-MS
#   - Per-case CSVs are skipped if already present in the tmp dir
#   - Run in parallel using a shell loop for speed

ENV["GKSwstype"] = "nul"
include(joinpath(@__DIR__, "main_Optimization_mr_flexible.jl"))

using Printf, Statistics, Random, CSV, DataFrames

# ── Mode configuration ────────────────────────────────────────────────────────

const SIM_DAY    = 84
const VOXEL_CM   = 5.0
const RLD_THRESH = 0.01

const ARCH_PARAMS = ["branch_angle_order1", "g_axial",
                     "branching_prob_order1", "branching_prob_order2"]

const ALL_PARAMS  = ["elongation_rate_min", "elongation_rate_max",
                     "elongation_logistic_k", "elongation_logistic_t50",
                     "branch_angle_order1", "g_axial",
                     "branching_prob_order1", "branching_prob_order2"]

struct ModeConfig
    exp_suffix     ::String   # CMA-ES experiment dir suffix (e.g. "_s2ms")
    tabicl_csv     ::String   # relative path to TabICL predictions CSV
    tabicl_scen    ::String   # scenario string to filter TabICL CSV
    param_names    ::Vector{String}  # params to predict (4 or 8)
    cmae_method    ::String   # method label for output
    tabicl_method  ::String   # method label for output
    tmp_dir        ::String   # per-case CSV tmp directory name
    out_csv        ::String   # final aggregated CSV filename
end

const TABICL_K7_CSV = joinpath("sbi", "data", "combined",
    "test_case_eval_multiseed_v2", "results_tabicl_multiseed_v2.csv")
const TABICL_K14_CSV = joinpath("sbi", "data", "combined",
    "test_case_eval_multiseed_v2_k14", "results_tabicl_ms_v2_k14.csv")
const TABICL_K1_CSV = joinpath("sbi", "data", "combined",
    "test_case_eval_multiseed_v2_k1", "results_tabicl_ms_v2_k1.csv")

const MODES = Dict(
    "s2_k1"  => ModeConfig("_s2",       TABICL_K1_CSV,  "Scen2: elong known",
                            ARCH_PARAMS, "CMA-ES-S2",        "TabICL-MS-k1",
                            "ml_struct_tmp_v2_s2_k1",   "comparison_structure_metrics_v2_s2_k1.csv"),
    "s2_k7"  => ModeConfig("_s2ms",     TABICL_K7_CSV,  "Scen2: elong known",
                            ARCH_PARAMS, "CMA-ES-S2MS",      "TabICL-MS-v2",
                            "ml_struct_tmp_v2_s2ms",    "comparison_structure_metrics_v2_s2ms.csv"),
    "s2_k14" => ModeConfig("_s2ms_k14", TABICL_K14_CSV, "Scen2: elong known",
                            ARCH_PARAMS, "CMA-ES-S2MS-k14",  "TabICL-MS-k14",
                            "ml_struct_tmp_v2_s2ms_k14","comparison_structure_metrics_v2_s2ms_k14.csv"),
    "s1_k1"  => ModeConfig("_s1",       TABICL_K1_CSV,  "Scen1: all unknown",
                            ALL_PARAMS,  "CMA-ES-S1",        "TabICL-MS-k1",
                            "ml_struct_tmp_v2_s1_k1",   "comparison_structure_metrics_v2_s1_k1.csv"),
    "s1_k7"  => ModeConfig("_s1ms",     TABICL_K7_CSV,  "Scen1: all unknown",
                            ALL_PARAMS,  "CMA-ES-S1MS",      "TabICL-MS-v2",
                            "ml_struct_tmp_v2_s1ms",    "comparison_structure_metrics_v2_s1ms.csv"),
    "s1_k14" => ModeConfig("_s1ms_k14", TABICL_K14_CSV, "Scen1: all unknown",
                            ALL_PARAMS,  "CMA-ES-S1MS-k14",  "TabICL-MS-k14",
                            "ml_struct_tmp_v2_s1ms_k14","comparison_structure_metrics_v2_s1ms_k14.csv"),
)

# ── Parse arguments ───────────────────────────────────────────────────────────

length(ARGS) < 1 && error("Usage: julia evaluate_ml_structure_unified.jl <mode> [tag]")

const MODE_KEY  = ARGS[1]
const SINGLE_TAG = length(ARGS) >= 2 ? ARGS[2] : nothing

haskey(MODES, MODE_KEY) || error("Unknown mode '$MODE_KEY'. Valid: $(join(sort(collect(keys(MODES))), ", "))")

const MC = MODES[MODE_KEY]
is_s2 = startswith(MODE_KEY, "s2")

@printf("Mode: %s  |  CMA-ES: %s  |  TabICL: %s\n", MODE_KEY, MC.cmae_method, MC.tabicl_method)

# ── Metric helpers ─────────────────────────────────────────────────────────────

rmse_rld(pred, true_)  = sqrt(mean((pred .- true_) .^ 2))

function rmse_angle(ca, ta, cr, tr)
    valid = (cr .> RLD_THRESH) .& (tr .> RLD_THRESH) .& .!isnan.(ca) .& .!isnan.(ta)
    sum(valid) == 0 && return NaN
    sqrt(mean((ca[valid] .- ta[valid]) .^ 2))
end

rmse_null_rld(tr) = (m = mean(tr); sqrt(mean((tr .- m) .^ 2)))

function rmse_null_angle(ta, tr)
    valid = (tr .> RLD_THRESH) .& .!isnan.(ta)
    sum(valid) == 0 && return NaN
    v = ta[valid]; sqrt(mean((v .- mean(v)) .^ 2))
end

skill(r, r0) = (isnan(r) || isnan(r0) || r0 < 1e-12) ? NaN : 1.0 - r / r0

# ── Simulation + extraction ────────────────────────────────────────────────────

function run_sim_extract(env_json_path, species_json_path, seed, nx, nz, box_W, box_H, tmp_dir)
    runid  = string("ms", seed, "_", abs(hash(species_json_path * string(seed))) % 10^9)
    outdir = joinpath(tmp_dir, runid)
    mkpath(outdir)

    env = _read_json(env_json_path)
    env["species_params_files"]["VD"] = species_json_path
    env["simulation"]["seed"]            = seed
    env["simulation"]["days"]            = SIM_DAY
    env["simulation"]["outdir"]          = outdir
    env["facilitation"]["enabled"]       = false
    env["density_regulation"]["enabled"] = false
    env["minirhizotron"]["enabled"]      = false

    env_path = joinpath(outdir, "env.json")
    _write_json_pretty(env_path, env)

    cfg = load_environment_and_plants(env_path)
    R_dummy = zeros(1, 1)
    Random.seed!(seed)
    for _ in 1:SIM_DAY
        for pl in cfg.plants
            step_plant!(pl, cfg.box, cfg.reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end
    end

    rld_acc  = zeros(nx, nz)
    awsum    = zeros(nx, nz)
    alen     = zeros(nx, nz)

    for pl in cfg.plants
        for seg in pl.root.segments
            dp = seg.p1 .- seg.p0
            dl = sqrt(dp[1]^2 + dp[2]^2 + dp[3]^2)
            dl > 1e-9 || continue
            mid = (seg.p0 .+ seg.p1) .* 0.5
            ix = clamp(floor(Int, (mid[1] + box_W * 0.5) / VOXEL_CM) + 1, 1, nx)
            iz = clamp(floor(Int, (-mid[3])              / VOXEL_CM) + 1, 1, nz)
            rld_acc[ix, iz] += dl
            dir_z = dp[3] / dl
            θ = acos(clamp(abs(dir_z), 0.0, 1.0)) * 180.0 / π
            awsum[ix, iz]   += θ * dl
            alen[ix, iz]    += dl
        end
    end

    rm(outdir; recursive=true, force=true)

    ang = fill(NaN, nx, nz)
    for i in 1:nx, j in 1:nz
        alen[i, j] > RLD_THRESH && (ang[i, j] = awsum[i, j] / alen[i, j])
    end
    rld_acc, ang
end

# ── Accumulator ───────────────────────────────────────────────────────────────

mutable struct Acc
    rld  ::Matrix{Float64}
    aws  ::Matrix{Float64}
    alen ::Matrix{Float64}
end
Acc(nx, nz) = Acc(zeros(nx,nz), zeros(nx,nz), zeros(nx,nz))

function update!(a::Acc, rld, ang)
    a.rld .+= rld
    for i in 1:size(rld,1), j in 1:size(rld,2)
        !isnan(ang[i,j]) && (a.aws[i,j] += ang[i,j]*rld[i,j]; a.alen[i,j] += rld[i,j])
    end
end

function finalise(a::Acc, K)
    mr = a.rld ./ K
    ma = fill(NaN, size(mr)...)
    for i in 1:size(mr,1), j in 1:size(mr,2)
        a.alen[i,j] > RLD_THRESH && (ma[i,j] = a.aws[i,j] / a.alen[i,j])
    end
    mr, ma
end

function metrics(mr_p, ma_p, mr_t, ma_t)
    r_rld  = rmse_rld(mr_p, mr_t)
    r0_rld = rmse_null_rld(mr_t)
    r_ang  = rmse_angle(ma_p, ma_t, mr_p, mr_t)
    r0_ang = rmse_null_angle(ma_t, mr_t)
    skill(r_rld, r0_rld), skill(r_ang, r0_ang), sum(mr_p .> RLD_THRESH)
end

# ── Load data ─────────────────────────────────────────────────────────────────

calib_dir  = @__DIR__
repo_dir   = dirname(calib_dir)
tc_csv     = joinpath(calib_dir, "test_cases_selected_v2.csv")
tc_df      = CSV.read(tc_csv, DataFrame)   # all 20 cases

tabicl_csv_path = joinpath(repo_dir, MC.tabicl_csv)
isfile(tabicl_csv_path) || error("TabICL CSV not found: $tabicl_csv_path")
tab_df_full = CSV.read(tabicl_csv_path, DataFrame)
tab_df = filter(r -> r.scenario == MC.tabicl_scen, tab_df_full)
@printf("TabICL rows loaded (%s): %d\n", MC.tabicl_scen, nrow(tab_df))

configs_dir  = joinpath(calib_dir, "configs", "test_cases_v2")
exps_dir     = joinpath(calib_dir, "experiments")
tmp_out_dir  = joinpath(calib_dir, MC.tmp_dir)
mkpath(tmp_out_dir)

# ── Main loop ─────────────────────────────────────────────────────────────────

out_rows = []

rows_to_process = if SINGLE_TAG !== nothing
    # Derive config/sample_id from tag by stripping known suffix
    parts = split(SINGLE_TAG, "_")
    filter(r -> "tc_v2_$(r.config)_$(r.sample_id)$(MC.exp_suffix)" == SINGLE_TAG, eachrow(tc_df))
else
    eachrow(tc_df)
end

for row in rows_to_process
    cfg_name  = row.config
    sample_id = row.sample_id
    tag       = "tc_v2_$(cfg_name)_$(sample_id)$(MC.exp_suffix)"

    per_case_csv = joinpath(tmp_out_dir, "$(tag).csv")
    if isfile(per_case_csv) && SINGLE_TAG === nothing
        @printf("SKIP (done): %s\n", tag)
        df_pc = CSV.read(per_case_csv, DataFrame)
        append!(out_rows, eachrow(df_pc))
        continue
    end

    exp_dir     = joinpath(exps_dir, tag)
    best_path   = joinpath(exp_dir, "best_params.json")
    config_path = joinpath(configs_dir, "$(tag).json")
    warmstart_path = joinpath(calib_dir, "data", "test_cases",
                               "warmstart_$(cfg_name)_$(sample_id).json")

    if !isfile(best_path)
        @printf("SKIP %s — best_params.json not found\n", tag); continue
    end
    if !isfile(config_path)
        @printf("SKIP %s — config JSON not found\n", tag); continue
    end
    if !isfile(warmstart_path)
        @printf("SKIP %s — warmstart JSON not found\n", tag); continue
    end

    # TabICL predictions for this case
    tab_rows = filter(r -> r.config == cfg_name && r.sample_id == sample_id, tab_df)
    if nrow(tab_rows) == 0
        @printf("SKIP %s — TabICL prediction not found\n", tag); continue
    end
    θ_tab = zeros(Float64, length(MC.param_names))
    for (i, p) in enumerate(MC.param_names)
        pr = filter(r -> r.parameter == p, tab_rows)
        θ_tab[i] = nrow(pr) > 0 ? Float64(pr[1, :pred_value]) : NaN
    end
    if any(isnan.(θ_tab))
        @printf("SKIP %s — TabICL missing parameter(s)\n", tag); continue
    end

    tc_config = _read_json(config_path)
    env_json  = String(tc_config["env_json"])
    seeds     = Int.(tc_config["calibration_seeds"])

    env_raw = _read_json(env_json)
    box_W   = Float64(env_raw["rhizobox"]["W"])
    box_H   = Float64(env_raw["rhizobox"]["H"])
    nx      = ceil(Int, box_W / VOXEL_CM)
    nz      = ceil(Int, box_H / VOXEL_CM)

    # True params (all 8 from test_cases_selected.csv)
    θ_true_all = Float64[Float64(row[p]) for p in ALL_PARAMS]
    # Arch-only true params (for display)
    θ_true_arch = Float64[Float64(row[p]) for p in ARCH_PARAMS]

    # CMA-ES predicted params
    bp = _read_json(best_path)
    θ_cal = Float64[Float64(get(bp, p, NaN)) for p in MC.param_names]
    if any(isnan.(θ_cal))
        @printf("SKIP %s — best_params.json missing parameter(s)\n", tag); continue
    end

    tmp_dir = joinpath(exp_dir, "__mlstruct_unified_$(MODE_KEY)__")
    mkpath(tmp_dir)

    true_spe = joinpath(tmp_dir, "VD_true.json")
    cal_spe  = joinpath(tmp_dir, "VD_cmae.json")
    tab_spe  = joinpath(tmp_dir, "VD_tabicl.json")

    # TRUE: always use all 8 params
    _write_candidate_VD_json(warmstart_path, true_spe, θ_true_all;
        param_names=ALL_PARAMS, n_axial_roots=1, max_order1_per_plant=20)
    # CMA-ES: predicted params (4 for S2, 8 for S1); warmstart provides elongation for S2
    _write_candidate_VD_json(warmstart_path, cal_spe, θ_cal;
        param_names=MC.param_names, n_axial_roots=1, max_order1_per_plant=20)
    # TabICL: same structure as CMA-ES
    _write_candidate_VD_json(warmstart_path, tab_spe, θ_tab;
        param_names=MC.param_names, n_axial_roots=1, max_order1_per_plant=20)

    @printf("\n[%s] %d×%d voxels  %d seeds\n", tag, nx, nz, length(seeds))
    if is_s2
        @printf("  TRUE:  angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f\n",
                θ_true_arch[1], θ_true_arch[2], θ_true_arch[3], θ_true_arch[4])
        @printf("  %s: angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f\n",
                MC.cmae_method, θ_cal[1], θ_cal[2], θ_cal[3], θ_cal[4])
        @printf("  %s: angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f\n",
                MC.tabicl_method, θ_tab[1], θ_tab[2], θ_tab[3], θ_tab[4])
    else
        @printf("  TRUE:  angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f  elong=[%.3f,%.3f]\n",
                θ_true_arch[1], θ_true_arch[2], θ_true_arch[3], θ_true_arch[4],
                θ_true_all[1], θ_true_all[2])
        arch_cal  = [get(bp, p, NaN) for p in ARCH_PARAMS]
        arch_tab  = θ_tab[5:8]
        elong_cal = θ_cal[1:2]
        elong_tab = θ_tab[1:2]
        @printf("  %s: angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f  elong=[%.3f,%.3f]\n",
                MC.cmae_method, arch_cal[1], arch_cal[2], arch_cal[3], arch_cal[4],
                elong_cal[1], elong_cal[2])
        @printf("  %s: angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f  elong=[%.3f,%.3f]\n",
                MC.tabicl_method, arch_tab[1], arch_tab[2], arch_tab[3], arch_tab[4],
                elong_tab[1], elong_tab[2])
    end
    flush(stdout)

    acc_t = Acc(nx, nz); acc_c = Acc(nx, nz); acc_b = Acc(nx, nz)

    for (si, seed) in enumerate(seeds)
        @printf("  seed %d (%d/%d)\r", seed, si, length(seeds)); flush(stdout)
        rld_t, ang_t = run_sim_extract(env_json, true_spe, seed, nx, nz, box_W, box_H, tmp_dir)
        rld_c, ang_c = run_sim_extract(env_json, cal_spe,  seed, nx, nz, box_W, box_H, tmp_dir)
        rld_b, ang_b = run_sim_extract(env_json, tab_spe,  seed, nx, nz, box_W, box_H, tmp_dir)
        update!(acc_t, rld_t, ang_t)
        update!(acc_c, rld_c, ang_c)
        update!(acc_b, rld_b, ang_b)
    end
    println()

    rm(tmp_dir; recursive=true, force=true)

    K = length(seeds)
    mr_t, ma_t = finalise(acc_t, K)
    mr_c, ma_c = finalise(acc_c, K)
    mr_b, ma_b = finalise(acc_b, K)

    ss_rld_c, ss_ang_c, nocc_c = metrics(mr_c, ma_c, mr_t, ma_t)
    ss_rld_b, ss_ang_b, nocc_b = metrics(mr_b, ma_b, mr_t, ma_t)
    nocc_t = sum(mr_t .> RLD_THRESH)

    @printf("  %s: SS_rld=%+.3f  SS_ang=%+.3f\n",
            MC.cmae_method, ss_rld_c, ss_ang_c)
    @printf("  %s: SS_rld=%+.3f  SS_ang=%+.3f\n",
            MC.tabicl_method, ss_rld_b, ss_ang_b)

    # Arch-params display values (always 4, regardless of scenario)
    arch_c = is_s2 ? θ_cal : θ_cal[5:8]
    arch_b = is_s2 ? θ_tab : θ_tab[5:8]

    new_rows = []
    for (method, ss_r, ss_a, nocc, arch_θ) in [
            (MC.cmae_method,   ss_rld_c, ss_ang_c, nocc_c, arch_c),
            (MC.tabicl_method, ss_rld_b, ss_ang_b, nocc_b, arch_b),
        ]
        push!(new_rows, (
            method=method, config=cfg_name, sample_id=sample_id,
            g_axial_quintile=row.g_axial_quintile,
            true_g_axial=Float64(row.g_axial),
            windows_with_roots=row.windows_with_roots,
            ss_rld=ss_r, ss_angle=ss_a,
            n_occ_true=nocc_t, n_occ_pred=nocc,
            pred_branch_angle=arch_θ[1], pred_g_axial=arch_θ[2],
            pred_bp1=arch_θ[3], pred_bp2=arch_θ[4],
        ))
    end

    CSV.write(per_case_csv, DataFrame(new_rows))
    @printf("  Wrote → %s\n", per_case_csv)
    append!(out_rows, new_rows)
end

# ── Output ─────────────────────────────────────────────────────────────────────

df_out = DataFrame(out_rows)

if SINGLE_TAG !== nothing
    @printf("\nDone: %s\n", SINGLE_TAG)
else
    out_csv = joinpath(calib_dir, MC.out_csv)
    CSV.write(out_csv, df_out)
    n_cases = nrow(df_out) ÷ 2
    @printf("\nWrote %d rows (%d cases) → %s\n", nrow(df_out), n_cases, out_csv)

    println("\n" * "="^74)
    @printf("Structural Similarity — %s vs %s (N=%d)\n",
            MC.cmae_method, MC.tabicl_method, n_cases)
    println("="^62)
    @printf("  %-18s  %-4s  %4s  %2s  %8s  %8s\n",
            "Method","cfg","#","Q","SS_rld","SS_ang")
    println("-"^62)
    for r in eachrow(df_out)
        @printf("  %-18s  %-4s  %4d  %2s  %+8.3f  %+8.3f\n",
                r.method, r.config, r.sample_id, r.g_axial_quintile,
                r.ss_rld, r.ss_angle)
    end
    println("-"^62)
    println("\nMethod means:")
    for method in [MC.cmae_method, MC.tabicl_method]
        sub = filter(r -> r.method == method, eachrow(df_out))
        isempty(sub) && continue
        ss_r = mean(filter(!isnan, [r.ss_rld  for r in sub]))
        ss_a = mean(filter(!isnan, [r.ss_angle for r in sub]))
        @printf("  %-18s  SS_rld=%+.3f  SS_angle=%+.3f\n",
                method, ss_r, ss_a)
    end
end
