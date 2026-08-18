# evaluate_ml_structure_s2ms.jl
#
# Structural similarity comparison: CMA-ES S2MS vs TabICL-multiseed (Scen2).
#
# For each of the 18 S2MS test cases, runs three simulation sets at SIM_DAY=84
# with K=12 seeds:
#   (a) TRUE parameters         → ground-truth 3D structure
#   (b) CMA-ES S2MS calibrated  → from calibration/experiments/tc_*_s2ms/best_params.json
#   (c) TabICL-multiseed Scen2  → from sbi/data/combined/test_case_eval_multiseed/results_tabicl_multiseed.csv
#                                  (scenario == "Scen2: elong known", 4 arch params)
#
# Both CMA-ES and TabICL use true elongation (from warmstart JSON) — fair comparison.
#
# CLI: julia calibration/evaluate_ml_structure_s2ms.jl [tag]
#   tag  e.g. "tc_A_254_s2ms" — single-case mode (writes ml_struct_tmp_s2ms/<tag>.csv)
#   (no arg) — all 18 cases sequentially

ENV["GKSwstype"] = "nul"
include(joinpath(@__DIR__, "main_Optimization_mr_flexible.jl"))

using Printf, Statistics, Random, CSV, DataFrames

const SIM_DAY    = 84
const VOXEL_CM   = 5.0
const RLD_THRESH = 0.01

const ARCH_PARAMS = ["branch_angle_order1", "g_axial",
                     "branching_prob_order1", "branching_prob_order2"]

const EXCLUDED = Set([("A", 497), ("C", 100)])

const SINGLE_TAG = length(ARGS) >= 1 ? ARGS[1] : nothing

# ── Metric helpers ─────────────────────────────────────────────────────────────

function chamfer_distance(A::Matrix{Float64}, B::Matrix{Float64})::Float64
    (size(A, 2) == 0 || size(B, 2) == 0) && return NaN
    d_A2B = mean(minimum(sum((A[:, i] .- B) .^ 2; dims=1)) for i in 1:size(A, 2))
    d_B2A = mean(minimum(sum((B[:, j] .- A) .^ 2; dims=1)) for j in 1:size(B, 2))
    sqrt((d_A2B + d_B2A) / 2.0)
end

rmse_rld(pred, true_) = sqrt(mean((pred .- true_) .^ 2))

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

    pts_list = Vector{Float64}[]
    rld_acc  = zeros(nx, nz)
    awsum    = zeros(nx, nz)
    alen     = zeros(nx, nz)

    for pl in cfg.plants
        for seg in pl.root.segments
            dp = seg.p1 .- seg.p0
            dl = sqrt(dp[1]^2 + dp[2]^2 + dp[3]^2)
            dl > 1e-9 || continue
            mid = (seg.p0 .+ seg.p1) .* 0.5
            push!(pts_list, mid)
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

    pts = isempty(pts_list) ? zeros(3, 0) : reduce(hcat, pts_list)
    ang = fill(NaN, nx, nz)
    for i in 1:nx, j in 1:nz
        alen[i, j] > RLD_THRESH && (ang[i, j] = awsum[i, j] / alen[i, j])
    end
    pts, rld_acc, ang
end

# ── Accumulator ───────────────────────────────────────────────────────────────

mutable struct Acc
    rld  ::Matrix{Float64}
    aws  ::Matrix{Float64}
    alen ::Matrix{Float64}
    chs  ::Vector{Float64}
end
Acc(nx, nz) = Acc(zeros(nx,nz), zeros(nx,nz), zeros(nx,nz), Float64[])

function update!(a::Acc, pts, rld, ang, pts_true)
    a.rld .+= rld
    for i in 1:size(rld,1), j in 1:size(rld,2)
        !isnan(ang[i,j]) && (a.aws[i,j] += ang[i,j]*rld[i,j]; a.alen[i,j] += rld[i,j])
    end
    push!(a.chs, chamfer_distance(pts, pts_true))
end

function finalise(a::Acc, K)
    mr = a.rld ./ K
    ma = fill(NaN, size(mr)...)
    for i in 1:size(mr,1), j in 1:size(mr,2)
        a.alen[i,j] > RLD_THRESH && (ma[i,j] = a.aws[i,j] / a.alen[i,j])
    end
    mr, ma, filter(!isnan, a.chs)
end

function metrics(mr_p, ma_p, ch_v, mr_t, ma_t)
    r_rld  = rmse_rld(mr_p, mr_t)
    r0_rld = rmse_null_rld(mr_t)
    r_ang  = rmse_angle(ma_p, ma_t, mr_p, mr_t)
    r0_ang = rmse_null_angle(ma_t, mr_t)
    skill(r_rld, r0_rld), skill(r_ang, r0_ang),
    isempty(ch_v) ? NaN : mean(ch_v),
    sum(mr_p .> RLD_THRESH)
end

# ── Load data ─────────────────────────────────────────────────────────────────

calib_dir = @__DIR__
tc_csv    = joinpath(calib_dir, "test_cases_selected.csv")
tc_df     = CSV.read(tc_csv, DataFrame)
# Filter to 18 cases (exclude slow cases)
tc_df = filter(r -> !((r.config, r.sample_id) in EXCLUDED), tc_df)

tabicl_csv = joinpath(dirname(calib_dir), "sbi", "data", "combined",
                       "test_case_eval_multiseed", "results_tabicl_multiseed.csv")
tab_df  = CSV.read(tabicl_csv, DataFrame)
tab_s2  = filter(r -> r.scenario == "Scen2: elong known", tab_df)
@printf("TabICL-multiseed Scen2 rows loaded: %d\n", nrow(tab_s2))

configs_dir = joinpath(calib_dir, "configs", "test_cases")
exps_dir    = joinpath(calib_dir, "experiments")
tmp_out_dir = joinpath(calib_dir, "ml_struct_tmp_s2ms")
mkpath(tmp_out_dir)

# ── Main loop ─────────────────────────────────────────────────────────────────

out_rows = []

rows_to_process = SINGLE_TAG !== nothing ?
    filter(r -> "tc_$(r.config)_$(r.sample_id)_s2ms" == SINGLE_TAG, eachrow(tc_df)) :
    eachrow(tc_df)

for row in rows_to_process
    cfg_name  = row.config
    sample_id = row.sample_id
    tag       = "tc_$(cfg_name)_$(sample_id)_s2ms"

    exp_dir        = joinpath(exps_dir, tag)
    best_path      = joinpath(exp_dir, "best_params.json")
    config_path    = joinpath(configs_dir, "$(tag).json")
    warmstart_path = joinpath(calib_dir, "data", "test_cases",
                               "warmstart_$(cfg_name)_$(sample_id).json")

    if !isfile(best_path)
        @printf("SKIP %s — best_params.json not found\n", tag); continue
    end

    # TabICL-multiseed Scen2 predictions for this case (4 arch params)
    tab_rows = filter(r -> r.config == cfg_name && r.sample_id == sample_id, tab_s2)
    if nrow(tab_rows) == 0
        @printf("SKIP %s — TabICL-multiseed Scen2 prediction not found\n", tag); continue
    end
    θ_tab = zeros(Float64, length(ARCH_PARAMS))
    for (i, p) in enumerate(ARCH_PARAMS)
        pr = filter(r -> r.parameter == p, tab_rows)
        θ_tab[i] = nrow(pr) > 0 ? Float64(pr[1, :pred_value]) : 0.0
    end

    tc_config = _read_json(config_path)
    env_json  = String(tc_config["env_json"])
    seeds     = Int.(tc_config["calibration_seeds"])

    env_raw = _read_json(env_json)
    box_W   = Float64(env_raw["rhizobox"]["W"])
    box_H   = Float64(env_raw["rhizobox"]["H"])
    nx      = ceil(Int, box_W / VOXEL_CM)
    nz      = ceil(Int, box_H / VOXEL_CM)

    θ_true = Float64[Float64(row[p]) for p in ARCH_PARAMS]
    bp     = _read_json(best_path)
    θ_cal  = Float64[Float64(bp[p]) for p in ARCH_PARAMS]

    tmp_dir = joinpath(exp_dir, "__mlstruct_s2ms_tmp__")
    mkpath(tmp_dir)

    true_spe = joinpath(tmp_dir, "VD_true.json")
    cal_spe  = joinpath(tmp_dir, "VD_cmae_s2ms.json")
    tab_spe  = joinpath(tmp_dir, "VD_tabicl_ms.json")

    _write_candidate_VD_json(warmstart_path, true_spe, θ_true;
        param_names=ARCH_PARAMS, n_axial_roots=1, max_order1_per_plant=20)
    _write_candidate_VD_json(warmstart_path, cal_spe, θ_cal;
        param_names=ARCH_PARAMS, n_axial_roots=1, max_order1_per_plant=20)
    _write_candidate_VD_json(warmstart_path, tab_spe, θ_tab;
        param_names=ARCH_PARAMS, n_axial_roots=1, max_order1_per_plant=20)

    @printf("\n[%s] %d×%d voxels  %d seeds\n", tag, nx, nz, length(seeds))
    @printf("  TRUE:        angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f\n",
            θ_true[1], θ_true[2], θ_true[3], θ_true[4])
    @printf("  CMA-ES-S2MS: angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f\n",
            θ_cal[1], θ_cal[2], θ_cal[3], θ_cal[4])
    @printf("  TabICL-MS:   angle=%.1f  g=%.4f  bp1=%.3f  bp2=%.3f\n",
            θ_tab[1], θ_tab[2], θ_tab[3], θ_tab[4])
    flush(stdout)

    acc_t = Acc(nx, nz); acc_c = Acc(nx, nz); acc_b = Acc(nx, nz)

    for (si, seed) in enumerate(seeds)
        @printf("  seed %d (%d/%d)\r", seed, si, length(seeds)); flush(stdout)
        pts_t, rld_t, ang_t = run_sim_extract(env_json, true_spe, seed, nx, nz, box_W, box_H, tmp_dir)
        pts_c, rld_c, ang_c = run_sim_extract(env_json, cal_spe,  seed, nx, nz, box_W, box_H, tmp_dir)
        pts_b, rld_b, ang_b = run_sim_extract(env_json, tab_spe,  seed, nx, nz, box_W, box_H, tmp_dir)
        update!(acc_t, pts_t, rld_t, ang_t, pts_t)
        update!(acc_c, pts_c, rld_c, ang_c, pts_t)
        update!(acc_b, pts_b, rld_b, ang_b, pts_t)
    end
    println()

    rm(tmp_dir; recursive=true, force=true)

    K = length(seeds)
    mr_t, ma_t, _    = finalise(acc_t, K)
    mr_c, ma_c, ch_c = finalise(acc_c, K)
    mr_b, ma_b, ch_b = finalise(acc_b, K)

    ss_rld_c, ss_ang_c, chf_c, nocc_c = metrics(mr_c, ma_c, ch_c, mr_t, ma_t)
    ss_rld_b, ss_ang_b, chf_b, nocc_b = metrics(mr_b, ma_b, ch_b, mr_t, ma_t)
    nocc_t = sum(mr_t .> RLD_THRESH)

    @printf("  CMA-ES-S2MS: SS_rld=%+.3f  SS_ang=%+.3f  Chamfer=%.3f cm\n", ss_rld_c, ss_ang_c, chf_c)
    @printf("  TabICL-MS:   SS_rld=%+.3f  SS_ang=%+.3f  Chamfer=%.3f cm\n", ss_rld_b, ss_ang_b, chf_b)

    for (method, ss_r, ss_a, chf, nocc, θ) in [
            ("CMA-ES-S2MS", ss_rld_c, ss_ang_c, chf_c, nocc_c, θ_cal),
            ("TabICL-MS",   ss_rld_b, ss_ang_b, chf_b, nocc_b, θ_tab),
        ]
        push!(out_rows, (
            method=method, config=cfg_name, sample_id=sample_id,
            g_axial_quintile=row.g_axial_quintile,
            true_g_axial=Float64(row.g_axial),
            windows_with_roots=row.windows_with_roots,
            ss_rld=ss_r, ss_angle=ss_a, chamfer_cm=chf,
            n_occ_true=nocc_t, n_occ_pred=nocc,
            pred_branch_angle=θ[1], pred_g_axial=θ[2],
            pred_bp1=θ[3], pred_bp2=θ[4],
        ))
    end

    per_case_csv = joinpath(tmp_out_dir, "$(tag).csv")
    CSV.write(per_case_csv, DataFrame(out_rows[end-1:end]))
    @printf("  Wrote → %s\n", per_case_csv)
end

# ── Output ─────────────────────────────────────────────────────────────────────

df_out = DataFrame(out_rows)

if SINGLE_TAG !== nothing
    @printf("\nDone: %s\n", SINGLE_TAG)
else
    out_csv = joinpath(calib_dir, "comparison_structure_metrics_s2ms.csv")
    CSV.write(out_csv, df_out)
    @printf("\nWrote %d rows → %s\n", nrow(df_out), out_csv)

    println("\n" * "="^72)
    println("Structural Similarity — CMA-ES-S2MS vs TabICL-multiseed (Scen2, N=18)")
    println("="^72)
    @printf("  %-14s  %-4s  %4s  %2s  %8s  %8s  %8s\n",
            "Method","cfg","#","Q","SS_rld","SS_ang","Chamfer")
    println("-"^72)
    for r in eachrow(df_out)
        @printf("  %-14s  %-4s  %4d  %2s  %+8.3f  %+8.3f  %8.3f\n",
                r.method, r.config, r.sample_id, r.g_axial_quintile,
                r.ss_rld, r.ss_angle, r.chamfer_cm)
    end
    println("-"^72)
    println("\nMethod means:")
    for method in ["CMA-ES-S2MS", "TabICL-MS"]
        sub = filter(r -> r.method == method, eachrow(df_out))
        isempty(sub) && continue
        ss_r = mean(filter(!isnan, [r.ss_rld   for r in sub]))
        ss_a = mean(filter(!isnan, [r.ss_angle  for r in sub]))
        chf  = mean(filter(!isnan, [r.chamfer_cm for r in sub]))
        @printf("  %-14s  SS_rld=%+.3f  SS_angle=%+.3f  Chamfer=%.3f cm\n",
                method, ss_r, ss_a, chf)
    end
end
