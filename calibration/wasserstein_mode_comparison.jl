# wasserstein_mode_comparison.jl
#
# Computes Wasserstein-based structural similarity metrics across all 4-param experiment
# dirs, using multi-seed averaged predictions (seed-stable, butterfly-effect-free).
#
# Metrics per experiment:
#   W_depth   : 1D Wasserstein on root-length-weighted depth distribution (cm)
#   W_angle   : 1D Wasserstein on root-length-weighted angle distribution (deg)
#   SS_depth  : Structural Skill Score = 1 - W_depth / W_null_depth
#   SS_angle  : Structural Skill Score = 1 - W_angle / W_null_angle
#
# Null baseline: uniform predictor — every window gets predicted length = mean(obs)
# (represents knowing nothing about spatial structure).
#
# Both metrics are computed on multi-seed averaged pred_cm, so they are not affected
# by seed-to-seed stochastic variation (same reasoning as diag_r2_multiseed).
#
# Wasserstein cross-sample comparability:
#   Raw W distances are NOT comparable across different test samples (different scale).
#   SS_depth / SS_angle normalize by the null baseline and are dimensionless (like R²).
#   For within-experiment paired comparisons across modes, raw W is valid.
#
# Run: julia calibration/wasserstein_mode_comparison.jl  (from repo root)

using CSV, DataFrames, JSON, Printf, Statistics, LinearAlgebra

const CALIB_PARAMS = Set(["branch_angle_order1", "g_axial",
                           "branching_prob_order1", "branching_prob_order2"])

read_json(p) = open(p, "r") do io; JSON.parse(String(read(io))); end

function classify_group(config_file::String)
    cfg = lowercase(config_file)
    occursin("100cm_pbc", cfg) && occursin("angle_hist", cfg) && return "B: 100cm_pbc+angle_hist"
    occursin("angle_hist", cfg)                                && return "C: 200cm+angle_hist"
    return "A: 200cm_length_only"
end

# ── 1D Wasserstein (exact, via CDF area) ─────────────────────────────────────
# locations: where each mass is located (e.g. depth in cm, angle in deg)
# weights:   how much mass is at each location (root length); need not sum to 1
function wasserstein_1d(locs_p::Vector{Float64}, weights_p::Vector{Float64},
                        locs_q::Vector{Float64}, weights_q::Vector{Float64})::Float64
    (isempty(locs_p) || isempty(locs_q)) && return NaN
    sum(weights_p) < 1e-12 || sum(weights_q) < 1e-12 && return NaN

    # Normalize to probability distributions
    wp = weights_p ./ sum(weights_p)
    wq = weights_q ./ sum(weights_q)

    # Merge and sort all unique locations
    all_locs = sort(unique(vcat(locs_p, locs_q)))
    n = length(all_locs)

    # Build empirical CDFs
    cdf_p = zeros(n); cdf_q = zeros(n)
    for (i, loc) in enumerate(all_locs)
        cdf_p[i] = sum(wp[locs_p .<= loc])
        cdf_q[i] = sum(wq[locs_q .<= loc])
    end

    # Integrate |CDF_p - CDF_q| via trapezoid rule
    W = 0.0
    for i in 1:(n - 1)
        dx = all_locs[i+1] - all_locs[i]
        W += 0.5 * (abs(cdf_p[i] - cdf_q[i]) + abs(cdf_p[i+1] - cdf_q[i+1])) * dx
    end
    W
end

# ── Structural Skill Score ────────────────────────────────────────────────────
# SS = 1 - W(method) / W(null)
# null: uniform predictor — every window gets predicted length = mean(obs)
function skill_score(W_method::Float64, W_null::Float64)::Float64
    (isnan(W_method) || isnan(W_null) || W_null < 1e-12) && return NaN
    1.0 - W_method / W_null
end

# ── Load obs angle data (for angle-based Wasserstein) ─────────────────────────
# Returns Dict: (station_idx, along_tube_cm, azimuth_deg) => length-weighted mean angle
function load_obs_angles(obs_csv::String)::Dict{Tuple{Int,Float64,Float64}, Float64}
    isfile(obs_csv) || return Dict()
    df = CSV.read(obs_csv, DataFrame)
    hasproperty(df, :mean_crossing_angle_deg) || return Dict()

    angle_map = Dict{Tuple{Int,Float64,Float64}, Tuple{Float64,Float64}}()
    for row in eachrow(df)
        key = (Int(row.station_idx), Float64(row.along_tube_cm), Float64(row.azimuth_deg))
        len = Float64(row.root_length_cm)
        ang = row.mean_crossing_angle_deg
        (ismissing(ang) || isnan(ang) || len <= 0) && continue
        prev = get(angle_map, key, (0.0, 0.0))
        angle_map[key] = (prev[1] + len * Float64(ang), prev[2] + len)
    end

    Dict(k => v[1] / v[2] for (k, v) in angle_map if v[2] > 0)
end

# ── Find all 4-param experiment dirs (latest per group × objective) ───────────
exp_base  = joinpath(@__DIR__, "experiments")
calib_dir = @__DIR__
repo_root = dirname(calib_dir)

latest = Dict{Tuple{String,String}, Any}()
for entry in readdir(exp_base; join=true)
    isdir(entry) || continue
    rs_path = joinpath(entry, "run_summary.json")
    ms_path = joinpath(entry, "best_obs_pred_stations_multiseed.csv")
    (isfile(rs_path) && isfile(ms_path)) || continue
    rs = read_json(rs_path)
    cp = get(rs, "calibration_params", nothing)
    cp === nothing && continue
    Set(String.(cp)) == CALIB_PARAMS || continue
    obj  = get(rs, "objective", "?")
    grp  = classify_group(get(rs, "config_file", ""))
    key  = (grp, obj)
    dname = basename(entry)
    if !haskey(latest, key) || dname > latest[key]["_dirname"]
        rs["_dirname"] = dname
        rs["_dir"]     = entry
        rs["_group"]   = grp
        latest[key] = rs
    end
end

isempty(latest) && (println("No 4-param runs with multiseed CSV found."); exit(0))

# ── Compute Wasserstein metrics for each experiment ───────────────────────────
results = []

for ((grp, obj), rs) in latest
    exp_dir = rs["_dir"]
    ms_path = joinpath(exp_dir, "best_obs_pred_stations_multiseed.csv")
    df = CSV.read(ms_path, DataFrame)

    # Aggregation strategy:
    #   :final  → use only the last observation day (mature root system snapshot)
    #   :all    → sum across all days (overall structural tendency)
    #   :mean   → mean across all days
    AGG_MODE = :final

    df_use = if AGG_MODE == :final
        max_day = maximum(df.day)
        filter(r -> r.day == max_day, df)
    else
        df
    end

    agg_fn = AGG_MODE == :mean ? (x -> mean(x)) : (x -> sum(x))
    df_agg = combine(groupby(df_use, [:station_idx, :along_tube_cm, :depth_cm, :azimuth_deg]),
                     :obs_cm  => agg_fn => :obs_total,
                     :pred_cm => agg_fn => :pred_total)

    depths    = Float64.(df_agg.depth_cm)
    obs_len   = Float64.(df_agg.obs_total)
    pred_len  = Float64.(df_agg.pred_total)

    # ── Depth-based Wasserstein ───────────────────────────────────────────────
    # null: uniform predictor → each window gets mean(obs)
    n_windows   = length(obs_len)
    null_len    = fill(mean(obs_len), n_windows)

    W_depth     = wasserstein_1d(depths, obs_len,   depths, pred_len)
    W_null_d    = wasserstein_1d(depths, obs_len,   depths, null_len)
    SS_d        = skill_score(W_depth, W_null_d)

    # ── Angle-based Wasserstein ───────────────────────────────────────────────
    # Load obs angle data; pair by window key
    config_path_raw = get(rs, "config_file", "")
    config_path = let cp = String(config_path_raw)
        isabspath(cp)                   ? cp :
        isfile(joinpath(calib_dir, cp)) ? joinpath(calib_dir, cp) :
        isfile(joinpath(repo_root, cp)) ? joinpath(repo_root, cp) : cp
    end
    obs_csv = if isfile(config_path)
        cfg = read_json(config_path)
        raw = get(cfg, "obs_csv_1tube", "data/mr_obs_stations_1tube.csv")
        p = String(raw)
        isabspath(p) ? p : joinpath(calib_dir, p)
    else
        ""
    end

    angle_map = load_obs_angles(obs_csv)

    W_angle = NaN; W_null_a = NaN; SS_a = NaN
    if !isempty(angle_map)
        angles_obs = Float64[]; obs_for_angle = Float64[]; pred_for_angle = Float64[]
        for row in eachrow(df_agg)
            key = (Int(row.station_idx), Float64(row.along_tube_cm), Float64(row.azimuth_deg))
            haskey(angle_map, key) || continue
            ang = angle_map[key]
            push!(angles_obs,    ang)
            push!(obs_for_angle, row.obs_total)
            push!(pred_for_angle, row.pred_total)
        end

        if !isempty(angles_obs)
            null_a   = fill(mean(obs_for_angle), length(angles_obs))
            W_angle  = wasserstein_1d(angles_obs, obs_for_angle,  angles_obs, pred_for_angle)
            W_null_a = wasserstein_1d(angles_obs, obs_for_angle,  angles_obs, null_a)
            SS_a     = skill_score(W_angle, W_null_a)
        end
    end

    # Parameter recovery (from run_summary or best_params)
    bp_path = joinpath(exp_dir, "best_params.json")
    mean_pct_err = NaN
    if isfile(bp_path)
        bp = read_json(bp_path)
        true_vals = Dict("branch_angle_order1" => 80.0, "g_axial" => 0.05,
                         "branching_prob_order1" => 0.20, "branching_prob_order2" => 0.30)
        errs = [abs(100.0 * (Float64(bp[p]) - true_vals[p]) / true_vals[p])
                for p in keys(true_vals) if haskey(bp, p)]
        isempty(errs) || (mean_pct_err = mean(errs))
    end

    r2ms  = get(rs, "diag_r2_multiseed",  NaN)
    rmsems = get(rs, "diag_rmse_multiseed", NaN)

    push!(results, (
        group    = grp,
        obj      = obj,
        r2_ms    = Float64(r2ms),
        rmse_ms  = Float64(rmsems),
        W_depth  = W_depth,
        SS_depth = SS_d,
        W_angle  = W_angle,
        SS_angle = SS_a,
        pct_err  = mean_pct_err,
    ))
end

sort!(results, by = r -> (r.group, r.W_depth))

# ── Print results table ───────────────────────────────────────────────────────
for grp in unique(r.group for r in results)
    grs = filter(r -> r.group == grp, results)
    W = 130
    println("\n", repeat('═', W))
    println("  $grp")
    println(repeat('═', W))
    @printf("  %-40s  %8s  %9s  %9s  %9s  %9s  %9s  %10s\n",
            "Objective", "R²(ms)", "RMSE(ms)", "W_depth", "SS_depth",
            "W_angle", "SS_angle", "Mean|%err|")
    println(repeat('─', W))
    for r in grs
        @printf("  %-40s  %8.3f  %9.3f  %9.3f  %9.3f  %9s  %9s  %10s\n",
                r.obj,
                r.r2_ms, r.rmse_ms,
                r.W_depth, r.SS_depth,
                isnan(r.W_angle)  ? "  —" : @sprintf("%9.3f", r.W_angle),
                isnan(r.SS_angle) ? "  —" : @sprintf("%9.3f", r.SS_angle),
                isnan(r.pct_err)  ? "  —" : @sprintf("%10.1f%%", r.pct_err))
    end
    println(repeat('─', W))
end

# ── Cross-mode Wasserstein correlation with parameter recovery ────────────────
println("\n", repeat('─', 70))
println("  Pearson correlation across modes (within each group):")
println(repeat('─', 70))
for grp in unique(r.group for r in results)
    grs = filter(r -> r.group == grp, results)
    length(grs) < 3 && continue
    valid = filter(r -> !isnan(r.W_depth) && !isnan(r.pct_err), grs)
    length(valid) < 3 && continue
    ws = [r.W_depth for r in valid]
    pe = [r.pct_err for r in valid]
    # Pearson r
    ws_c = ws .- mean(ws); pe_c = pe .- mean(pe)
    r_val = dot(ws_c, pe_c) / (sqrt(sum(ws_c.^2)) * sqrt(sum(pe_c.^2)) + 1e-12)
    @printf("  %s:  r(W_depth, |%%err|) = %+.3f\n", grp, r_val)
end
println(repeat('─', 70))

println("\nNote: SS > 0 = better than uniform predictor; SS ≈ 1 = perfect structural recovery.")
println("      W_depth in cm, W_angle in degrees.")
println("      Rows sorted by W_depth (ascending = structurally closer to observed).")
println()
