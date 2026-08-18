# chamfer_mode_comparison.jl
#
# Computes Chamfer distance between calibrated and template (true) root systems
# for all 4-param experiment dirs, averaged across K calibration seeds.
#
# Level 2b structural comparison: full 3D root system WITHOUT minirhizotron tube filter.
# Day 90 = final observation day (most mature root system).
#
# For each seed k (paired):
#   - Simulate 90 days with TEMPLATE (true) params → 3D point cloud P_true_k
#   - Simulate 90 days with CALIBRATED params      → 3D point cloud P_calib_k
#   - Chamfer(P_calib_k, P_true_k)
# Then: Chamfer_multiseed = mean_k Chamfer_k   [Option 1, seed-stable]
#
# Note: Both simulations are stochastic and use the same seed per pair, so random
# variation is controlled while parameter differences drive the Chamfer distance.
#
# Run: julia calibration/chamfer_mode_comparison.jl  (from repo root)

ENV["GKSwstype"] = "nul"
include(joinpath(@__DIR__, "main_Optimization_mr_flexible.jl"))

using Printf, Statistics, Random

const SIM_DAY = 90

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

# ── Extract 3D point cloud (segment midpoints) from plants ────────────────────
function extract_midpoints(plants)::Matrix{Float64}
    pts = Vector{Float64}[]
    for pl in plants
        for seg in pl.root.segments
            push!(pts, (seg.p0 .+ seg.p1) .* 0.5)
        end
    end
    isempty(pts) ? zeros(3, 0) : reduce(hcat, pts)
end

# ── Symmetric Chamfer distance (sqrt of mean squared nearest-neighbour dist) ──
function chamfer_distance(A::Matrix{Float64}, B::Matrix{Float64})::Float64
    (size(A, 2) == 0 || size(B, 2) == 0) && return NaN
    # A → B
    d_A2B = mean(minimum(sum((A[:, i] .- B) .^ 2; dims=1)) for i in 1:size(A, 2))
    # B → A
    d_B2A = mean(minimum(sum((B[:, j] .- A) .^ 2; dims=1)) for j in 1:size(B, 2))
    sqrt((d_A2B + d_B2A) / 2.0)   # cm
end

# ── Run full 3D simulation, return segment midpoint cloud at SIM_DAY ──────────
function run_sim_3d(env_json_path::String, species_json_path::String,
                    seed::Int, n_axial::Int, max_o1::Int,
                    tmp_dir::String)::Matrix{Float64}
    runid  = string("ch_s", seed, "_", abs(hash(species_json_path * string(seed))) % 10^9)
    outdir = joinpath(tmp_dir, runid)
    mkpath(outdir)

    env = _read_json(env_json_path)
    env["species_params_files"]["VD"] = species_json_path
    env["simulation"]["seed"]         = seed
    env["simulation"]["days"]         = SIM_DAY
    env["simulation"]["outdir"]       = outdir
    env["facilitation"]["enabled"]    = false
    env["density_regulation"]["enabled"] = false
    env["minirhizotron"]["enabled"]   = false

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

    pts = extract_midpoints(cfg.plants)
    rm(outdir; recursive=true, force=true)
    pts
end

# ── Collect 4-param experiment dirs ──────────────────────────────────────────
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
    obj  = get(rs, "objective", "?")
    grp  = classify_group(get(rs, "config_file", ""))
    key  = (grp, obj)
    dname = basename(entry)
    if !haskey(latest, key) || dname > latest[key]["_dirname"]
        rs["_dirname"] = dname; rs["_dir"] = entry; rs["_group"] = grp
        latest[key] = rs
    end
end

isempty(latest) && (println("No 4-param runs found."); exit(0))

# ── Compute Chamfer for each experiment ──────────────────────────────────────
results = []

for ((grp, obj), rs) in latest
    exp_dir = rs["_dir"]
    bp_path = joinpath(exp_dir, "best_params.json")

    # Resolve config and env paths
    config_path_raw = get(rs, "config_file", "")
    config_path = let cp = String(config_path_raw)
        isabspath(cp)                   ? cp :
        isfile(joinpath(calib_dir, cp)) ? joinpath(calib_dir, cp) :
        isfile(joinpath(repo_root, cp)) ? joinpath(repo_root, cp) : cp
    end
    isfile(config_path) || (println("SKIP $obj: config not found"); continue)

    config   = _read_json(config_path)
    env_raw  = String(get(config, "env_json", "Parameter_calib_minirhizotron.json"))
    env_json = isabspath(env_raw) ? env_raw : joinpath(calib_dir, env_raw)
    isfile(env_json) || (println("SKIP $obj: env_json not found"); continue)

    template_raw = String(get(config, "warm_start_json", "Parameter_VD_template.json"))
    template_json = isabspath(template_raw) ? template_raw : joinpath(calib_dir, template_raw)
    isfile(template_json) || (println("SKIP $obj: template not found"); continue)

    calib_params = String.(get(rs, "calibration_params",
                               ["branch_angle_order1","g_axial",
                                "branching_prob_order1","branching_prob_order2"]))
    seeds  = Int.(get(rs, "calibration_seeds", [42]))
    n_ax   = Int(get(rs, "n_axial_roots", 1))
    n_mo1  = Int(get(rs, "max_order1_per_plant", 20))

    # Build calibrated species JSON from best_params
    bp = _read_json(bp_path)
    θ_calib = Float64[Float64(bp[p]) for p in calib_params]

    tmp_dir = joinpath(exp_dir, "__chamfer_tmp__")
    mkpath(tmp_dir)

    # Write species JSONs (once each, reused across seeds)
    # Template: true parameter values with correct n_axial / max_o1
    θ_true = Float64[TRUE_PARAMS[p] for p in calib_params]
    template_species_json = joinpath(tmp_dir, "VD_template.json")
    _write_candidate_VD_json(template_json, template_species_json, θ_true;
        param_names=calib_params, n_axial_roots=n_ax, max_order1_per_plant=n_mo1)
    # Calibrated: best-fit parameter values
    calib_species_json = joinpath(tmp_dir, "VD_calibrated.json")
    _write_candidate_VD_json(template_json, calib_species_json, θ_calib;
        param_names=calib_params, n_axial_roots=n_ax, max_order1_per_plant=n_mo1)

    @printf("\n[%s | %s]\n", grp, obj)
    @printf("  %s\n", basename(exp_dir))
    @printf("  Running %d seeds × 2 sims (template + calibrated) at day %d...\n",
            length(seeds), SIM_DAY)

    chamfer_per_seed = Float64[]
    n_pts_true_all   = Int[]
    n_pts_calib_all  = Int[]

    for (si, seed) in enumerate(seeds)
        @printf("    seed %d (%d/%d)\n", seed, si, length(seeds))
        pts_true  = run_sim_3d(env_json, template_species_json, seed, n_ax, n_mo1, tmp_dir)
        pts_calib = run_sim_3d(env_json, calib_species_json, seed, n_ax, n_mo1, tmp_dir)
        ch = chamfer_distance(pts_calib, pts_true)
        push!(chamfer_per_seed, ch)
        push!(n_pts_true_all,  size(pts_true,  2))
        push!(n_pts_calib_all, size(pts_calib, 2))
    end

    rm(tmp_dir; recursive=true, force=true)

    ch_mean = mean(filter(!isnan, chamfer_per_seed))
    ch_std  = std(filter(!isnan,  chamfer_per_seed))
    np_true  = round(Int, mean(n_pts_true_all))
    np_calib = round(Int, mean(n_pts_calib_all))

    @printf("  Chamfer (mean ± std): %.3f ± %.3f cm\n", ch_mean, ch_std)
    @printf("  Points/seed: true=%d  calib=%d\n", np_true, np_calib)

    # Parameter recovery
    mean_pct_err = mean(abs(100.0 * (Float64(bp[p]) - TRUE_PARAMS[p]) / TRUE_PARAMS[p])
                        for p in keys(TRUE_PARAMS) if haskey(bp, p))
    r2ms  = Float64(get(rs, "diag_r2_multiseed",   NaN))
    rmsems = Float64(get(rs, "diag_rmse_multiseed", NaN))

    push!(results, (
        group       = grp,
        obj         = obj,
        chamfer     = ch_mean,
        chamfer_std = ch_std,
        r2_ms       = r2ms,
        rmse_ms     = rmsems,
        pct_err     = mean_pct_err,
    ))
end

# ── Summary table ─────────────────────────────────────────────────────────────
sort!(results, by = r -> (r.group, r.chamfer))
println()
for grp in unique(r.group for r in results)
    grs = filter(r -> r.group == grp, results)
    W = 110
    println(repeat('═', W))
    println("  $grp")
    println(repeat('═', W))
    @printf("  %-40s  %10s  %10s  %8s  %9s  %10s\n",
            "Objective", "Chamfer(cm)", "Std(cm)", "R²(ms)", "RMSE(ms)", "Mean|%err|")
    println(repeat('─', W))
    for r in grs
        @printf("  %-40s  %10.3f  %10.3f  %8.3f  %9.3f  %10.1f%%\n",
                r.obj, r.chamfer, r.chamfer_std, r.r2_ms, r.rmse_ms, r.pct_err)
    end
    println(repeat('─', W))
end

println("\nNote: Chamfer = sqrt(mean squared nearest-neighbour distance), symmetric.")
println("      Lower = calibrated root system more structurally similar to template (true).")
println("      Seed-stable: E[Chamfer(calib_k, true_k)] averaged over paired seeds.")
println()
