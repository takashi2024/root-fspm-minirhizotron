# evaluate_test_case_structure.jl
#
# Computes structural similarity metrics (SS_rld, SS_angle, Chamfer) for the 20
# Scenario 2 test case calibrations. Requires calibrations to have completed.
#
# For each test case, runs two sets of simulations at SIM_DAY:
#   (a) true parameters  → "ground truth" 3D root structure
#   (b) calibrated best_params → "predicted" 3D root structure
# Then computes voxel-based RMSE / skill scores and Chamfer distance.
#
# Reads per-test-case true params from test_cases_selected.csv (not hardcoded).
#
# Run: julia calibration/evaluate_test_case_structure.jl  (from repo root)
# Output:
#   calibration/test_case_structure_metrics.csv

ENV["GKSwstype"] = "nul"
include(joinpath(@__DIR__, "main_Optimization_mr_flexible.jl"))

using Printf, Statistics, Random, CSV, DataFrames

const SIM_DAY    = 84     # last snapshot day in SBI dataset
const VOXEL_CM   = 5.0
const RLD_THRESH = 0.01

const CALIB_PARAMS_TC = ["branch_angle_order1", "g_axial",
                          "branching_prob_order1", "branching_prob_order2"]

# ── Metric helpers (same as voxel_rld_mode_comparison.jl) ─────────────────────

function chamfer_distance(A::Matrix{Float64}, B::Matrix{Float64})::Float64
    (size(A, 2) == 0 || size(B, 2) == 0) && return NaN
    d_A2B = mean(minimum(sum((A[:, i] .- B) .^ 2; dims=1)) for i in 1:size(A, 2))
    d_B2A = mean(minimum(sum((B[:, j] .- A) .^ 2; dims=1)) for j in 1:size(B, 2))
    sqrt((d_A2B + d_B2A) / 2.0)
end

function rmse_rld_tc(calib::Matrix{Float64}, true_::Matrix{Float64})::Float64
    sqrt(mean((calib .- true_) .^ 2))
end

function rmse_angle_tc(ca::Matrix{Float64}, ta::Matrix{Float64},
                       cr::Matrix{Float64}, tr::Matrix{Float64})::Float64
    valid = (cr .> RLD_THRESH) .& (tr .> RLD_THRESH) .& .!isnan.(ca) .& .!isnan.(ta)
    sum(valid) == 0 && return NaN
    sqrt(mean((ca[valid] .- ta[valid]) .^ 2))
end

function rmse_null_rld_tc(tr::Matrix{Float64})::Float64
    m = mean(tr); sqrt(mean((tr .- m) .^ 2))
end

function rmse_null_angle_tc(ta::Matrix{Float64}, tr::Matrix{Float64})::Float64
    valid = (tr .> RLD_THRESH) .& .!isnan.(ta)
    sum(valid) == 0 && return NaN
    v = ta[valid]; sqrt(mean((v .- mean(v)) .^ 2))
end

skill_tc(r, r0) = (isnan(r) || isnan(r0) || r0 < 1e-12) ? NaN : 1.0 - r / r0

# ── Simulation + extraction ────────────────────────────────────────────────────

function run_sim_extract_tc(env_json_path::String, species_json_path::String,
                             seed::Int, nx::Int, nz::Int,
                             box_W::Float64, box_H::Float64,
                             tmp_dir::String)
    runid  = string("tc_s", seed, "_", abs(hash(species_json_path * string(seed))) % 10^9)
    outdir = joinpath(tmp_dir, runid)
    mkpath(outdir)

    env = _read_json(env_json_path)
    env["species_params_files"]["VD"] = species_json_path
    env["simulation"]["seed"]          = seed
    env["simulation"]["days"]          = SIM_DAY
    env["simulation"]["outdir"]        = outdir
    env["facilitation"]["enabled"]     = false
    env["density_regulation"]["enabled"] = false
    env["minirhizotron"]["enabled"]    = false

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

    pts_list  = Vector{Float64}[]
    rld_acc   = zeros(nx, nz)
    awsum_acc = zeros(nx, nz)
    alen_acc  = zeros(nx, nz)

    for pl in cfg.plants
        for seg in pl.root.segments
            dp = seg.p1 .- seg.p0
            dl = sqrt(dp[1]^2 + dp[2]^2 + dp[3]^2)
            dl > 1e-9 || continue
            mid = (seg.p0 .+ seg.p1) .* 0.5
            push!(pts_list, mid)
            ix = clamp(floor(Int, (mid[1] + box_W * 0.5) / VOXEL_CM) + 1, 1, nx)
            iz = clamp(floor(Int, (-mid[3])              / VOXEL_CM) + 1, 1, nz)
            rld_acc[ix, iz]   += dl
            dir_z = dp[3] / dl
            θ = acos(clamp(abs(dir_z), 0.0, 1.0)) * 180.0 / π
            awsum_acc[ix, iz] += θ * dl
            alen_acc[ix, iz]  += dl
        end
    end

    rm(outdir; recursive=true, force=true)

    pts = isempty(pts_list) ? zeros(3, 0) : reduce(hcat, pts_list)
    angle_grid = fill(NaN, nx, nz)
    for i in 1:nx, j in 1:nz
        alen_acc[i, j] > RLD_THRESH && (angle_grid[i, j] = awsum_acc[i, j] / alen_acc[i, j])
    end
    pts, rld_acc, angle_grid
end

# ── Load test cases CSV ────────────────────────────────────────────────────────

calib_dir = @__DIR__
tc_csv    = joinpath(calib_dir, "test_cases_selected.csv")
tc_df     = CSV.read(tc_csv, DataFrame)

configs_dir = joinpath(calib_dir, "configs", "test_cases")
exps_dir    = joinpath(calib_dir, "experiments")

# ── Main evaluation loop ───────────────────────────────────────────────────────

out_rows = []

for row in eachrow(tc_df)
    cfg_name  = row.config
    sample_id = row.sample_id
    tag       = "tc_$(cfg_name)_$(sample_id)_s2"

    exp_dir        = joinpath(exps_dir, tag)
    best_path      = joinpath(exp_dir, "best_params.json")
    config_path    = joinpath(configs_dir, "$(tag).json")
    warmstart_path = joinpath(calib_dir, "data", "test_cases",
                               "warmstart_$(cfg_name)_$(sample_id).json")

    if !isfile(best_path)
        @printf("SKIP %s — best_params.json not found\n", tag)
        continue
    end

    tc_config  = _read_json(config_path)
    env_json   = String(tc_config["env_json"])
    seeds      = Int.(tc_config["calibration_seeds"])

    env_raw    = _read_json(env_json)
    box_W      = Float64(env_raw["rhizobox"]["W"])
    box_H      = Float64(env_raw["rhizobox"]["H"])
    nx         = ceil(Int, box_W / VOXEL_CM)
    nz         = ceil(Int, box_H / VOXEL_CM)

    # True architectural params
    θ_true = Float64[Float64(row[p]) for p in CALIB_PARAMS_TC]
    # Calibrated params
    bp     = _read_json(best_path)
    θ_cal  = Float64[Float64(bp[p]) for p in CALIB_PARAMS_TC]

    tmp_dir = joinpath(exp_dir, "__struct_tmp__")
    mkpath(tmp_dir)

    # Write true and calibrated species JSONs
    true_spe_path = joinpath(tmp_dir, "VD_true.json")
    cal_spe_path  = joinpath(tmp_dir, "VD_calibrated.json")
    _write_candidate_VD_json(warmstart_path, true_spe_path, θ_true;
        param_names=CALIB_PARAMS_TC, n_axial_roots=1, max_order1_per_plant=20)
    _write_candidate_VD_json(warmstart_path, cal_spe_path, θ_cal;
        param_names=CALIB_PARAMS_TC, n_axial_roots=1, max_order1_per_plant=20)

    @printf("\n[%s] box=%.0f×%.0f → %d×%d voxels, %d seeds @ day %d\n",
            tag, box_W, box_H, nx, nz, length(seeds), SIM_DAY)
    flush(stdout)

    rld_sum_t  = zeros(nx, nz); rld_sum_c  = zeros(nx, nz)
    awsum_t    = zeros(nx, nz); awsum_c    = zeros(nx, nz)
    alen_t     = zeros(nx, nz); alen_c     = zeros(nx, nz)
    chamfer_v  = Float64[]

    for (si, seed) in enumerate(seeds)
        @printf("    seed %d (%d/%d)\n", seed, si, length(seeds)); flush(stdout)
        pts_t, rld_t, ang_t = run_sim_extract_tc(env_json, true_spe_path,
                                                   seed, nx, nz, box_W, box_H, tmp_dir)
        pts_c, rld_c, ang_c = run_sim_extract_tc(env_json, cal_spe_path,
                                                   seed, nx, nz, box_W, box_H, tmp_dir)
        rld_sum_t .+= rld_t; rld_sum_c .+= rld_c
        for i in 1:nx, j in 1:nz
            if !isnan(ang_t[i, j])
                awsum_t[i, j] += ang_t[i, j] * rld_t[i, j]
                alen_t[i, j]  += rld_t[i, j]
            end
            if !isnan(ang_c[i, j])
                awsum_c[i, j] += ang_c[i, j] * rld_c[i, j]
                alen_c[i, j]  += rld_c[i, j]
            end
        end
        push!(chamfer_v, chamfer_distance(pts_c, pts_t))
    end

    rm(tmp_dir; recursive=true, force=true)

    K = length(seeds)
    mr_t = rld_sum_t ./ K; mr_c = rld_sum_c ./ K
    ma_t = fill(NaN, nx, nz); ma_c = fill(NaN, nx, nz)
    for i in 1:nx, j in 1:nz
        alen_t[i, j] > RLD_THRESH && (ma_t[i, j] = awsum_t[i, j] / alen_t[i, j])
        alen_c[i, j] > RLD_THRESH && (ma_c[i, j] = awsum_c[i, j] / alen_c[i, j])
    end

    r_rld = rmse_rld_tc(mr_c, mr_t);   r0_rld = rmse_null_rld_tc(mr_t)
    r_ang = rmse_angle_tc(ma_c, ma_t, mr_c, mr_t); r0_ang = rmse_null_angle_tc(ma_t, mr_t)
    ss_rld = skill_tc(r_rld, r0_rld); ss_ang = skill_tc(r_ang, r0_ang)

    valid_ch = filter(!isnan, chamfer_v)
    chamfer_mean = isempty(valid_ch) ? NaN : mean(valid_ch)

    n_occ_t = sum(mr_t .> RLD_THRESH)
    n_occ_c = sum(mr_c .> RLD_THRESH)

    @printf("  RMSE_rld=%.4f  SS_rld=%+.3f  RMSE_ang=%.2f°  SS_ang=%+.3f  Chamfer=%.3f cm\n",
            r_rld, ss_rld, r_ang, ss_ang, chamfer_mean)

    push!(out_rows, (
        config=cfg_name, sample_id=sample_id,
        g_axial_quintile=row.g_axial_quintile,
        true_g_axial=Float64(row.g_axial),
        rmse_rld=r_rld, ss_rld=ss_rld,
        rmse_angle=r_ang, ss_angle=ss_ang,
        chamfer_cm=chamfer_mean,
        n_occ_true=n_occ_t, n_occ_calib=n_occ_c,
        n_seeds=K,
    ))
end

# ── Write CSV ──────────────────────────────────────────────────────────────────

out_csv = joinpath(calib_dir, "test_case_structure_metrics.csv")
df_out  = DataFrame(out_rows)
CSV.write(out_csv, df_out)
@printf("\nWrote %d rows → %s\n", nrow(df_out), out_csv)

# ── Summary table ──────────────────────────────────────────────────────────────

println("\n" * "="^72)
println("Structural Similarity — Test Cases Scenario 2")
println("="^72)
@printf("  %-7s %4s  %2s  %7s  %8s  %7s  %8s  %8s\n",
        "config", "#", "Q", "g_axial", "SS_rld", "SS_ang", "Chamfer", "n_occ_c")
println("-"^72)
for r in eachrow(df_out)
    @printf("  %-7s %4d  %2s  %7.4f  %+8.3f  %+7.3f  %8.3f  %8d\n",
            r.config, r.sample_id, r.g_axial_quintile, r.true_g_axial,
            r.ss_rld, r.ss_angle, r.chamfer_cm, r.n_occ_calib)
end
println("-"^72)
@printf("  Mean:                        %+8.3f  %+7.3f  %8.3f\n",
        mean(df_out.ss_rld), mean(df_out.ss_angle), mean(df_out.chamfer_cm))
