# voxel_rld_mode_comparison.jl
#
# Level 2b structural comparison: full 3D root system WITHOUT minirhizotron tube filter.
# Computes both voxelized RLD metrics AND Chamfer in one simulation pass.
#
# ── L2/RMSE calculation on voxels ────────────────────────────────────────────────────
#
# After averaging across K seeds, each experiment has two 2D grids (nx × nz):
#
#   mean_rld[ix,iz]   = avg root length per VOXEL_CM × VOXEL_CM voxel (cm), K-seed avg
#   mean_angle[ix,iz] = length-weighted mean zenith angle per voxel (degrees), K-seed avg
#                       zenith angle: 0°=downward axial, 90°=horizontal lateral
#                       captures branch_angle_order1 and g_axial effects directly
#
#   Δrld   = mean_rld_calib   - mean_rld_true        (cm, all voxels)
#   Δangle = mean_angle_calib - mean_angle_true       (deg, valid voxels only:
#                                                       both rld > RLD_THRESH)
#
#   RMSE_rld   = sqrt( mean(Δrld²)   )   cm   — comparable across box sizes (mean, not sum)
#   RMSE_angle = sqrt( mean(Δangle²) )   deg
#
# Null baseline: uniform predictor (every voxel = global mean).
# Skill scores (dimensionless, analogous to R²):
#   SS_rld   = 1 - RMSE_rld   / RMSE_null_rld
#   SS_angle = 1 - RMSE_angle / RMSE_null_angle
#
# Run: julia calibration/voxel_rld_mode_comparison.jl  (from repo root)

ENV["GKSwstype"] = "nul"
include(joinpath(@__DIR__, "main_Optimization_mr_flexible.jl"))

using Printf, Statistics, Random

const SIM_DAY    = 90
const VOXEL_CM   = 5.0
const RLD_THRESH = 0.01   # cm — minimum root length to consider a voxel occupied
const W_ANGLE    = 0.3    # weight for angle term (matches w_angle_hist in calibration)

const CALIB_PARAMS_SET = Set(["branch_angle_order1", "g_axial",
                               "branching_prob_order1", "branching_prob_order2"])
const TRUE_PARAMS = Dict(
    "branch_angle_order1"   => 80.0,
    "g_axial"               => 0.05,
    "branching_prob_order1" => 0.20,
    "branching_prob_order2" => 0.30,
)

function classify_group(config_file::String)
    cfg = lowercase(config_file)
    occursin("100cm_pbc", cfg) && occursin("angle_hist", cfg) && return "B: 100cm_pbc+angle_hist"
    occursin("angle_hist", cfg)                                && return "C: 200cm+angle_hist"
    return "A: 200cm_length_only"
end

# ── Chamfer distance ───────────────────────────────────────────────────────────────────
function chamfer_distance(A::Matrix{Float64}, B::Matrix{Float64})::Float64
    (size(A, 2) == 0 || size(B, 2) == 0) && return NaN
    d_A2B = mean(minimum(sum((A[:, i] .- B) .^ 2; dims=1)) for i in 1:size(A, 2))
    d_B2A = mean(minimum(sum((B[:, j] .- A) .^ 2; dims=1)) for j in 1:size(B, 2))
    sqrt((d_A2B + d_B2A) / 2.0)
end

# ── RMSE / skill score helpers ────────────────────────────────────────────────────────
function rmse_rld(calib::Matrix{Float64}, true_::Matrix{Float64})::Float64
    sqrt(mean((calib .- true_) .^ 2))
end

function rmse_angle(calib_a::Matrix{Float64}, true_a::Matrix{Float64},
                    calib_r::Matrix{Float64}, true_r::Matrix{Float64})::Float64
    valid = (calib_r .> RLD_THRESH) .& (true_r .> RLD_THRESH) .&
            .!isnan.(calib_a) .& .!isnan.(true_a)
    sum(valid) == 0 && return NaN
    sqrt(mean((calib_a[valid] .- true_a[valid]) .^ 2))
end

function rmse_null_rld(true_r::Matrix{Float64})::Float64
    m = mean(true_r)
    sqrt(mean((true_r .- m) .^ 2))
end

function rmse_null_angle(true_a::Matrix{Float64}, true_r::Matrix{Float64})::Float64
    valid = (true_r .> RLD_THRESH) .& .!isnan.(true_a)
    sum(valid) == 0 && return NaN
    v = true_a[valid]
    sqrt(mean((v .- mean(v)) .^ 2))
end

skill(r, r0) = (isnan(r) || isnan(r0) || r0 < 1e-12) ? NaN : 1.0 - r / r0

# ── Simulation + extraction (plants stay local — no type-inference leakage) ────────────
#
# Returns: (pts, rld_grid, mean_angle_grid)
#   pts        :: Matrix{Float64}  (3 × n_segments) midpoints for Chamfer
#   rld_grid   :: Matrix{Float64}  (nx × nz)  root length per voxel (cm)
#   angle_grid :: Matrix{Float64}  (nx × nz)  length-weighted mean zenith angle (deg)
#                                              NaN where no roots present
#
# box_W, box_H, nx, nz are passed in so the return type is always the same
# regardless of the simulation parameters — prevents Julia type-inference explosion.
#
function run_sim_extract(env_json_path::String, species_json_path::String,
                         seed::Int, nx::Int, nz::Int,
                         box_W::Float64, box_H::Float64,
                         tmp_dir::String)
    runid  = string("vx_s", seed, "_", abs(hash(species_json_path * string(seed))) % 10^9)
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

    # ── Extract inside this function so Vector{Plant} never escapes ────────────
    pts_list = Vector{Float64}[]
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

            # Voxel index: x ∈ [-W/2, W/2] → ix; z ∈ [-H, 0] → iz
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

# ── Collect 4-param experiment dirs ───────────────────────────────────────────────────

exp_base  = joinpath(@__DIR__, "experiments")
calib_dir = @__DIR__
repo_root = dirname(calib_dir)

latest = Dict{Tuple{String,String}, Any}()
for entry in readdir(exp_base; join=true)
    isdir(entry) || continue
    rs_path = joinpath(entry, "run_summary.json")
    bp_path = joinpath(entry, "best_params.json")
    (isfile(rs_path) && isfile(bp_path)) || continue
    rs = _read_json(rs_path)
    cp = get(rs, "calibration_params", nothing)
    cp === nothing && continue
    Set(String.(cp)) == CALIB_PARAMS_SET || continue
    obj   = get(rs, "objective", "?")
    grp   = classify_group(get(rs, "config_file", ""))
    key   = (grp, obj)
    dname = basename(entry)
    if !haskey(latest, key) || dname > latest[key]["_dirname"]
        rs["_dirname"] = dname; rs["_dir"] = entry; rs["_group"] = grp
        latest[key] = rs
    end
end

isempty(latest) && (println("No 4-param runs found."); exit(0))

# ── Main loop ──────────────────────────────────────────────────────────────────────────

results = []

for ((grp, obj), rs) in latest
    exp_dir = rs["_dir"]

    config_path_raw = get(rs, "config_file", "")
    config_path = let cp = String(config_path_raw)
        isabspath(cp)                   ? cp :
        isfile(joinpath(calib_dir, cp)) ? joinpath(calib_dir, cp) :
        isfile(joinpath(repo_root, cp)) ? joinpath(repo_root, cp) : cp
    end
    isfile(config_path) || (println("SKIP $obj: config not found"); continue)

    config        = _read_json(config_path)
    env_raw       = String(get(config, "env_json", "Parameter_calib_minirhizotron.json"))
    env_json      = isabspath(env_raw) ? env_raw : joinpath(calib_dir, env_raw)
    isfile(env_json) || (println("SKIP $obj: env_json not found"); continue)

    template_raw  = String(get(config, "warm_start_json", "Parameter_VD_template.json"))
    template_json = isabspath(template_raw) ? template_raw : joinpath(calib_dir, template_raw)
    isfile(template_json) || (println("SKIP $obj: template not found"); continue)

    calib_params = String.(get(rs, "calibration_params",
                               ["branch_angle_order1","g_axial",
                                "branching_prob_order1","branching_prob_order2"]))
    seeds  = Int.(get(rs, "calibration_seeds", [42]))
    n_ax   = Int(get(rs, "n_axial_roots", 1))
    n_mo1  = Int(get(rs, "max_order1_per_plant", 20))

    # Read box dimensions directly from env JSON (no simulation needed)
    env_raw_data = _read_json(env_json)
    box_W = Float64(env_raw_data["rhizobox"]["W"])
    box_H = Float64(env_raw_data["rhizobox"]["H"])
    nx    = ceil(Int, box_W / VOXEL_CM)
    nz    = ceil(Int, box_H / VOXEL_CM)

    bp = _read_json(joinpath(exp_dir, "best_params.json"))
    θ_calib = Float64[Float64(bp[p]) for p in calib_params]
    θ_true  = Float64[TRUE_PARAMS[p] for p in calib_params]

    tmp_dir = joinpath(exp_dir, "__voxel_tmp__")
    mkpath(tmp_dir)

    template_species_json = joinpath(tmp_dir, "VD_template.json")
    _write_candidate_VD_json(template_json, template_species_json, θ_true;
        param_names=calib_params, n_axial_roots=n_ax, max_order1_per_plant=n_mo1)
    calib_species_json = joinpath(tmp_dir, "VD_calibrated.json")
    _write_candidate_VD_json(template_json, calib_species_json, θ_calib;
        param_names=calib_params, n_axial_roots=n_ax, max_order1_per_plant=n_mo1)

    @printf("\n[%s | %s]\n", grp, obj)
    @printf("  %s\n", basename(exp_dir))
    @printf("  Box: %.0f×%.0f cm → %d×%d voxels (%.0f cm each)\n",
            box_W, box_H, nx, nz, VOXEL_CM)
    @printf("  Running %d seeds × 2 sims at day %d...\n", length(seeds), SIM_DAY)
    flush(stdout)

    # Accumulators (plain arrays — no struct types in outer scope)
    rld_sum_true    = zeros(nx, nz)
    rld_sum_calib   = zeros(nx, nz)
    awsum_true      = zeros(nx, nz)   # length-weighted angle sum across seeds
    awsum_calib     = zeros(nx, nz)
    alen_true       = zeros(nx, nz)   # total length contributing to angle
    alen_calib      = zeros(nx, nz)
    chamfer_seeds   = Float64[]
    npts_true_all   = Int[]
    npts_calib_all  = Int[]

    for (si, seed) in enumerate(seeds)
        @printf("    seed %d (%d/%d)\n", seed, si, length(seeds))
        flush(stdout)

        pts_t, rld_t, ang_t = run_sim_extract(env_json, template_species_json,
                                               seed, nx, nz, box_W, box_H, tmp_dir)
        pts_c, rld_c, ang_c = run_sim_extract(env_json, calib_species_json,
                                               seed, nx, nz, box_W, box_H, tmp_dir)

        rld_sum_true  .+= rld_t
        rld_sum_calib .+= rld_c

        # Accumulate angle with length weighting (sum across seeds; normalize at end)
        for i in 1:nx, j in 1:nz
            if !isnan(ang_t[i, j])
                awsum_true[i, j] += ang_t[i, j] * rld_t[i, j]
                alen_true[i, j]  += rld_t[i, j]
            end
            if !isnan(ang_c[i, j])
                awsum_calib[i, j] += ang_c[i, j] * rld_c[i, j]
                alen_calib[i, j]  += rld_c[i, j]
            end
        end

        push!(chamfer_seeds,  chamfer_distance(pts_c, pts_t))
        push!(npts_true_all,  size(pts_t, 2))
        push!(npts_calib_all, size(pts_c, 2))
    end

    rm(tmp_dir; recursive=true, force=true)

    K = length(seeds)
    mean_rld_true  = rld_sum_true  ./ K
    mean_rld_calib = rld_sum_calib ./ K

    mean_ang_true  = fill(NaN, nx, nz)
    mean_ang_calib = fill(NaN, nx, nz)
    for i in 1:nx, j in 1:nz
        alen_true[i, j]  > RLD_THRESH && (mean_ang_true[i, j]  = awsum_true[i, j]  / alen_true[i, j])
        alen_calib[i, j] > RLD_THRESH && (mean_ang_calib[i, j] = awsum_calib[i, j] / alen_calib[i, j])
    end

    # Metrics
    r_rld   = rmse_rld(mean_rld_calib, mean_rld_true)
    r_ang   = rmse_angle(mean_ang_calib, mean_ang_true, mean_rld_calib, mean_rld_true)
    r0_rld  = rmse_null_rld(mean_rld_true)
    r0_ang  = rmse_null_angle(mean_ang_true, mean_rld_true)
    ss_rld  = skill(r_rld,  r0_rld)
    ss_ang  = skill(r_ang,  r0_ang)
    ch_mean = mean(filter(!isnan, chamfer_seeds))
    ch_std  = std(filter(!isnan,  chamfer_seeds))

    n_occ_true  = sum(mean_rld_true  .> RLD_THRESH)
    n_occ_calib = sum(mean_rld_calib .> RLD_THRESH)

    @printf("  Occupied voxels: true=%d  calib=%d  (of %d)\n",
            n_occ_true, n_occ_calib, nx * nz)
    @printf("  RMSE_rld=%.4f cm   (null=%.4f, SS_rld=%+.3f)\n", r_rld, r0_rld, ss_rld)
    @printf("  RMSE_angle=%.2f°   (null=%.2f°, SS_angle=%+.3f)\n", r_ang, r0_ang, ss_ang)
    @printf("  Chamfer=%.3f±%.3f cm  (pts/seed: true=%d calib=%d)\n",
            ch_mean, ch_std,
            round(Int, mean(npts_true_all)), round(Int, mean(npts_calib_all)))
    flush(stdout)

    mean_pct_err = mean(abs(100.0 * (Float64(bp[p]) - TRUE_PARAMS[p]) / TRUE_PARAMS[p])
                        for p in keys(TRUE_PARAMS) if haskey(bp, p))
    r2ms   = Float64(get(rs, "diag_r2_multiseed",   NaN))
    rmsems = Float64(get(rs, "diag_rmse_multiseed", NaN))

    push!(results, (
        group      = grp,
        obj        = obj,
        rmse_rld   = r_rld,
        ss_rld     = ss_rld,
        rmse_angle = r_ang,
        ss_angle   = ss_ang,
        chamfer    = ch_mean,
        chamfer_std = ch_std,
        r2_ms      = r2ms,
        rmse_ms    = rmsems,
        pct_err    = mean_pct_err,
        n_occ_true  = n_occ_true,
        n_occ_calib = n_occ_calib,
        n_voxels    = nx * nz,
    ))
end

# ── Summary table ──────────────────────────────────────────────────────────────────────

sort!(results, by = r -> (r.group, r.rmse_rld))

println()
for grp in unique(r.group for r in results)
    grs = filter(r -> r.group == grp, results)
    W = 135
    println(repeat('═', W))
    println("  $grp")
    println(repeat('═', W))
    @printf("  %-40s  %10s  %9s  %11s  %10s  %10s  %10s\n",
            "Objective", "RMSE_rld", "SS_rld", "RMSE_angle°", "SS_angle",
            "Chamfer", "Mean|%err|")
    println(repeat('─', W))
    for r in grs
        @printf("  %-40s  %10.4f  %9.3f  %11.2f  %10.3f  %10.3f  %10.1f%%\n",
                r.obj,
                r.rmse_rld, r.ss_rld,
                r.rmse_angle, r.ss_angle,
                r.chamfer, r.pct_err)
    end
    println(repeat('─', W))
end

println()
println("Metrics:")
println("  RMSE_rld   = sqrt(mean(Δrld²))   per $(VOXEL_CM)cm voxel (cm) — all voxels")
println("  RMSE_angle = sqrt(mean(Δangle²)) per voxel (degrees) — voxels with roots in both systems")
println("  Zenith angle: 0°=downward, 90°=horizontal. Captures g_axial and branch_angle effects.")
println("  SS = 1 - RMSE/RMSE_null  (null=uniform predictor; 1=perfect, 0=null baseline)")
println("  Chamfer recomputed for consistency check (should match chamfer_mode_comparison.jl).")
println()
