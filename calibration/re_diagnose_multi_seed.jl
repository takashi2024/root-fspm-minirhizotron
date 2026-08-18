# re_diagnose_multi_seed.jl
#
# Re-runs the post-calibration diagnostic using all calibration seeds (averaged)
# for existing experiment directories, WITHOUT re-running CMA-ES.
#
# For each experiment dir it:
#   1. Loads best_params.json + run_summary.json + original config JSON
#   2. Runs one simulation per calibration seed with the best params
#   3. Averages predicted root-length per window across seeds
#   4. Computes R², RMSE, PBIAS on the seed-averaged predictions
#   5. Writes obs_vs_pred_multiseed.png and best_obs_pred_stations_multiseed.csv
#   6. Appends diag_r2_multiseed / diag_rmse_multiseed / diag_pbias_multiseed
#      / diag_n_seeds to run_summary.json
#
# Usage (from repo root):
#   julia calibration/re_diagnose_multi_seed.jl                      # auto-find all 4-param runs
#   julia calibration/re_diagnose_multi_seed.jl <exp_dir> [<exp_dir> ...]

ENV["GKSwstype"] = "nul"
include(joinpath(@__DIR__, "main_Optimization_mr_flexible.jl"))

using Printf, Statistics

const CALIB_PARAMS_SET = Set(["branch_angle_order1", "g_axial",
                               "branching_prob_order1", "branching_prob_order2"])

function classify_group(config_file::String)
    cfg = lowercase(config_file)
    occursin("100cm_pbc", cfg) && occursin("angle_hist", cfg) && return "B: 100cm_pbc+angle_hist"
    occursin("angle_hist", cfg)                                && return "C: 200cm+angle_hist"
    return "A: 200cm_length_only"
end

# ── Find all 4-param experiment dirs (latest per group×objective) ────────────
function find_target_dirs(exp_base::String)
    latest = Dict{Tuple{String,String}, Tuple{String,String}}()  # (group,obj) => (dirname, dirpath)
    for entry in readdir(exp_base; join=true)
        isdir(entry) || continue
        rs_path = joinpath(entry, "run_summary.json")
        isfile(rs_path) || continue
        rs = _read_json(rs_path)
        cp = get(rs, "calibration_params", nothing)
        cp === nothing && continue
        Set(String.(cp)) == CALIB_PARAMS_SET || continue
        obj   = get(rs, "objective", "?")
        grp   = classify_group(get(rs, "config_file", ""))
        key   = (grp, obj)
        dname = basename(entry)
        if !haskey(latest, key) || dname > latest[key][1]
            latest[key] = (dname, entry)
        end
    end
    return [v[2] for v in values(latest)]
end

# ── Multi-seed averaged diagnostic for 1-tube setup ─────────────────────────
function rediag_1tube(exp_dir::String)
    rs_path = joinpath(exp_dir, "run_summary.json")
    bp_path = joinpath(exp_dir, "best_params.json")
    isfile(rs_path) || (println("  SKIP: no run_summary.json"); return)
    isfile(bp_path) || (println("  SKIP: no best_params.json");  return)

    rs = _read_json(rs_path)
    bp = _read_json(bp_path)

    # Already done?
    if haskey(rs, "diag_r2_multiseed")
        @printf("  SKIP: already has diag_r2_multiseed = %.3f\n", rs["diag_r2_multiseed"])
        return
    end

    config_path_raw = get(rs, "config_file", nothing)
    config_path_raw === nothing && (println("  SKIP: no config_file in run_summary"); return)
    # Resolve relative paths: try as-is, then relative to calibration dir, then repo root
    calib_dir  = @__DIR__
    repo_root  = dirname(calib_dir)
    config_path = let cp = String(config_path_raw)
        isabspath(cp)             ? cp :
        isfile(joinpath(calib_dir, cp))  ? joinpath(calib_dir, cp) :
        isfile(joinpath(repo_root, cp))  ? joinpath(repo_root, cp) :
        cp   # will fail below with a clear message
    end
    isfile(config_path) || (println("  SKIP: config_file not found: $config_path"); return)
    config = _read_json(config_path)

    # ── Geometry / paths from config ─────────────────────────────────────────
    _env_raw  = get(config, "env_json", "Parameter_calib_minirhizotron.json")
    env_json  = isabspath(String(_env_raw)) ? String(_env_raw) :
                joinpath(calib_dir, String(_env_raw))
    env_raw   = _read_json(env_json)
    tube_origin_y_cm = Float64(get(get(env_raw, "minirhizotron", Dict()),
                                   "tube_origin_cm", [0.0, 20.0, 0.0])[2])
    box_width_cm     = Float64(get(config, "box_width_cm", 100.0))
    tube_x_cm        = Float64(get(config, "tube_x_1tube_cm", 50.0))

    _obs_raw  = get(config, "obs_csv_1tube", "data/mr_obs_stations_1tube.csv")
    obs_csv   = isabspath(String(_obs_raw)) ? String(_obs_raw) :
                joinpath(calib_dir, String(_obs_raw))
    isfile(obs_csv) || (println("  SKIP: obs CSV not found: $obs_csv"); return)

    calib_params = String.(get(rs, "calibration_params",
                               ["branch_angle_order1","g_axial",
                                "branching_prob_order1","branching_prob_order2"]))
    seeds        = Int.(get(rs, "calibration_seeds", [42]))
    n_ax         = Int(get(rs, "n_axial_roots", 1))
    n_mo1        = Int(get(rs, "max_order1_per_plant", 20))

    # θ from best_params.json
    θ = Float64[Float64(bp[p]) for p in calib_params]

    # Obs data
    used_days, obs_by_day, _, _, meta_by_day = _load_obs_1tube(obs_csv)
    isempty(used_days) && (println("  SKIP: no obs data"); return)

    # Temp dir for simulation output
    tmp_dir = joinpath(exp_dir, "__rediag_tmp__")

    # ── Run all seeds, accumulate ─────────────────────────────────────────────
    n_seeds = length(seeds)
    @printf("  Running %d seeds...\n", n_seeds)

    # acc[day] accumulates sum of predicted lengths across seeds
    acc = Dict{Int, Vector{Float64}}()
    cnt = Dict{Int, Int}()

    for (si, seed) in enumerate(seeds)
        @printf("    seed %d (%d/%d)\n", seed, si, n_seeds)
        result = _run_sim_1tube(env_json, bp_path, θ, used_days;
            seed=seed,
            n_axial_roots=n_ax,
            max_order1_per_plant=n_mo1,
            param_names=calib_params,
            tube_x_from_left_cm=tube_x_cm,
            box_width_cm=box_width_cm,
            tube_origin_y_cm=tube_origin_y_cm,
            tmp_dir=tmp_dir)

        for d in used_days
            preds = get(result.lengths, d, nothing)
            preds === nothing && continue
            if !haskey(acc, d)
                acc[d] = zeros(Float64, length(preds))
                cnt[d] = 0
            end
            acc[d] .+= preds
            cnt[d]  += 1
        end
    end

    # Average
    pred_avg = Dict{Int, Vector{Float64}}(
        d => acc[d] ./ cnt[d] for d in keys(acc))

    # ── Build obs-pred table ──────────────────────────────────────────────────
    df_long = _build_obs_pred_long(used_days, obs_by_day, meta_by_day, pred_avg)
    CSV.write(joinpath(exp_dir, "best_obs_pred_stations_multiseed.csv"), df_long)

    # ── Metrics (non-zero windows only, same as single-seed) ─────────────────
    x_all = Float64.(df_long.obs_cm)
    y_all = Float64.(df_long.pred_cm)
    nz    = (x_all .> 0.0) .| (y_all .> 0.0)
    xnz   = x_all[nz]; ynz = y_all[nz]
    m = isempty(xnz) ? nothing : _regression_metrics(xnz, ynz)

    if m !== nothing
        @printf("  R² (seed=42):        %.3f  →  R² (multi-seed avg): %.3f\n",
                get(rs, "diag_r2", NaN), m.r2)
        @printf("  RMSE:       %.3f cm  →  %.3f cm\n",
                get(rs, "diag_rmse", NaN), m.rmse)
        @printf("  PBIAS:      %.1f%%    →  %.1f%%\n",
                get(rs, "diag_pbias", NaN), m.pbias)
    end

    # ── Scatter plot ──────────────────────────────────────────────────────────
    obj_label = get(rs, "objective", "?")
    _save_scatter_png(df_long, joinpath(exp_dir, "obs_vs_pred_multiseed.png");
        title="$obj_label (multi-seed avg, n=$n_seeds): Obs vs Pred")

    # ── Update run_summary.json ───────────────────────────────────────────────
    rs["diag_r2_multiseed"]    = m !== nothing ? m.r2    : NaN
    rs["diag_rmse_multiseed"]  = m !== nothing ? m.rmse  : NaN
    rs["diag_pbias_multiseed"] = m !== nothing ? m.pbias : NaN
    rs["diag_n_seeds"]         = n_seeds
    delete!(rs, "_dir"); delete!(rs, "_dirname")
    delete!(rs, "_best_params"); delete!(rs, "_group")
    _write_json_pretty(rs_path, rs)

    # Cleanup temp dir
    isdir(tmp_dir) && rm(tmp_dir; recursive=true)
end

# ── Main ─────────────────────────────────────────────────────────────────────
exp_base = joinpath(@__DIR__, "experiments")

target_dirs = if isempty(ARGS)
    println("Auto-finding 4-param experiment dirs in $exp_base ...")
    found = find_target_dirs(exp_base)
    # Only Groups A and B
    filter(d -> begin
        rs = _read_json(joinpath(d, "run_summary.json"))
        grp = classify_group(get(rs, "config_file", ""))
        grp in ("A: 200cm_length_only", "B: 100cm_pbc+angle_hist")
    end, found)
else
    String.(ARGS)
end

sort!(target_dirs)
println("Found $(length(target_dirs)) experiment dirs to re-diagnose.\n")

for (i, d) in enumerate(target_dirs)
    obj = get(_read_json(joinpath(d, "run_summary.json")), "objective", "?")
    grp = classify_group(get(_read_json(joinpath(d, "run_summary.json")), "config_file", ""))
    @printf("[%d/%d] %s | %s\n", i, length(target_dirs), grp, obj)
    @printf("        %s\n", basename(d))
    try
        rediag_1tube(d)
    catch e
        println("  ERROR: ", e)
    end
    println()
end

println("Done. Run summarize_4param_results.jl to see updated metrics.")
