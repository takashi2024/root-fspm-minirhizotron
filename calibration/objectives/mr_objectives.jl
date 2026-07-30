# mr_objectives.jl
#
# All 8 calibration objective functions for minirhizotron tube data.
# Pure computation — no simulation code, no src/ includes.
#
# Supported objective modes:
#   1. rmse_baseline                  — plain SSE / n_windows (includes zero-zero)
#   2. weighted_rmse_zero_downweighted — per-window weights; optional log1p
#   3. balanced_zero_sampling         — random subsample of zeros to match n_positives × ratio
#   4. two_part_hurdle                — balanced-accuracy (presence) + log1p RMSE (intensity)
#   5. summary_feature_loss           — normalized error on 6 aggregate features
#   6. multi_objective_zero_aware     — 5-component weighted multi-objective
#   7. replicate_seed_average         — average predictions across seeds, then weighted RMSE
#   8. replicate_seed_multi_objective — average predictions across seeds, then multi-objective

using Statistics
using Random
using DataFrames
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# Configuration struct  (immutable; created by load_objective_config)
# ─────────────────────────────────────────────────────────────────────────────
struct CalibrationObjectiveConfig
    objective::String

    # Mode 2: weighted RMSE
    zero_weight::Float64
    positive_weight::Float64
    use_log1p::Bool
    loss_metric::String           # "rmse" or "mae"

    # Mode 3: balanced zero sampling
    zero_to_positive_ratio::Float64
    sampling_seed::Int
    resample_each_generation::Bool

    # Mode 4 / 6: presence detection threshold (cm)
    presence_threshold::Float64

    # Mode 4: two-part hurdle component weights
    w_presence::Float64
    w_intensity::Float64

    # Mode 5 / 6 / 8: depth binning & rooting depth
    depth_bin_edges_cm::Vector{Float64}
    rooting_depth_threshold_cm::Float64

    # Mode 5: summary feature component weights
    w_total::Float64
    w_depth::Float64
    w_time::Float64
    w_occupancy::Float64
    w_rooting_depth::Float64
    w_center_depth::Float64

    # Mode 6 / 8: multi-objective component weights
    w_total_root::Float64
    w_depth_profile::Float64
    w_presence_absence::Float64
    w_positive_intensity::Float64
    w_rooting_depth_mo::Float64

    # Modes 7 / 8: replicate seed prediction averaging
    replicate_seeds::Vector{Int}
    replicate_aggregation::String  # "mean" (only option currently)

    # Modes 1–6: outer seed loop (loss averaged across these seeds)
    calibration_seeds::Vector{Int}

    # Angle-aware components (all modes; disabled by default)
    w_angle_mean::Float64          # weight for mean-crossing-angle loss (0.0 = off)
    w_angle_hist::Float64          # weight for histogram angle loss (0.0 = off; recommended)
    angle_normalize_deg::Float64   # denominator for mean-angle MSE normalization (45.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Constructor helpers
# ─────────────────────────────────────────────────────────────────────────────
function default_objective_config()::CalibrationObjectiveConfig
    CalibrationObjectiveConfig(
        "rmse_baseline",
        0.1, 1.0, false, "rmse",
        2.0, 42, false,
        0.0,
        0.5, 0.5,
        [0.0, 10.0, 20.0, 30.0, 40.0, 60.0], 0.05,
        0.25, 0.25, 0.15, 0.15, 0.10, 0.10,
        0.30, 0.25, 0.20, 0.15, 0.10,
        [11, 22, 99, 88], "mean",
        [11, 22, 99, 88],
        0.0, 0.0, 45.0
    )
end

function load_objective_config(d::Dict)::CalibrationObjectiveConfig
    oc = get(d, "objective_config", Dict{String,Any}())
    gf(key, default) = Float64(get(oc, key, default))
    gi(key, default) = Int(get(oc, key, default))
    gb(key, default) = Bool(get(oc, key, default))
    gs(key, default) = String(get(oc, key, default))

    base_seeds = Int.(get(d, "calibration_seeds", [11, 22, 99, 88]))
    rep_seeds  = Int.(get(oc, "replicate_seeds", base_seeds))
    dbe = get(oc, "depth_bin_edges_cm", [0.0, 10.0, 20.0, 30.0, 40.0, 60.0])

    CalibrationObjectiveConfig(
        gs("objective", get(d, "calibration_objective", "rmse_baseline")),
        gf("zero_weight", 0.1),
        gf("positive_weight", 1.0),
        gb("use_log1p", false),
        gs("loss_metric", "rmse"),
        gf("zero_to_positive_ratio", 2.0),
        gi("sampling_seed", 42),
        gb("resample_each_generation", false),
        gf("presence_threshold", 0.0),
        gf("w_presence", 0.5),
        gf("w_intensity", 0.5),
        Float64.(dbe),
        gf("rooting_depth_threshold_cm", 0.05),
        gf("w_total", 0.25),
        gf("w_depth", 0.25),
        gf("w_time", 0.15),
        gf("w_occupancy", 0.15),
        gf("w_rooting_depth", 0.10),
        gf("w_center_depth", 0.10),
        gf("w_total_root", 0.30),
        gf("w_depth_profile", 0.25),
        gf("w_presence_absence", 0.20),
        gf("w_positive_intensity", 0.15),
        gf("w_rooting_depth_mo", 0.10),
        rep_seeds,
        gs("replicate_aggregation", "mean"),
        base_seeds,
        gf("w_angle_mean", 0.0),
        gf("w_angle_hist", 0.0),
        gf("angle_normalize_deg", 45.0)
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Zero-inflation statistics
# ─────────────────────────────────────────────────────────────────────────────
"""
Compute zero-inflation statistics from the observation DataFrame.
The DataFrame must have columns: `root_length_cm`, `depth_cm`, `date`.
Returns a NamedTuple used by mr_diagnostics.jl.
"""
function compute_zero_inflation_stats(obs_df::DataFrame)
    v       = Float64.(obs_df.root_length_cm)
    n_total = length(v)
    n_zero  = count(x -> x == 0.0, v)
    n_pos   = n_total - n_zero
    pos_v   = filter(x -> x > 0, v)
    pos_range = isempty(pos_v) ? (0.0, 0.0) : (minimum(pos_v), maximum(pos_v))

    # By depth bin (10 cm bins)
    depth_edges = [0.0, 10.0, 20.0, 30.0, 40.0, 60.0]
    n_bins = length(depth_edges) - 1
    bin_labels   = ["$(Int(depth_edges[i]))-$(Int(depth_edges[i+1])) cm" for i in 1:n_bins]
    bin_n_zero   = zeros(Int, n_bins)
    bin_n_pos    = zeros(Int, n_bins)
    for (r, obs) in zip(eachrow(obs_df), v)
        d = Float64(r.depth_cm)
        b = searchsortedlast(depth_edges, d)
        b = clamp(b, 1, n_bins)
        if obs == 0.0
            bin_n_zero[b] += 1
        else
            bin_n_pos[b] += 1
        end
    end
    by_depth = DataFrame(
        depth_bin  = bin_labels,
        n_zero     = bin_n_zero,
        n_positive = bin_n_pos,
        n_total    = bin_n_zero .+ bin_n_pos,
        zero_frac  = [n == 0 ? NaN : nz / n
                      for (nz, n) in zip(bin_n_zero, bin_n_zero .+ bin_n_pos)]
    )

    # By date
    dates      = sort(unique(obs_df.date))
    date_nzero = Int[]
    date_npos  = Int[]
    for dt in dates
        rows = obs_df[obs_df.date .== dt, :]
        push!(date_nzero, count(x -> x == 0.0, rows.root_length_cm))
        push!(date_npos,  count(x -> x > 0.0,  rows.root_length_cm))
    end
    by_date = DataFrame(
        date       = dates,
        n_zero     = date_nzero,
        n_positive = date_npos,
        n_total    = date_nzero .+ date_npos,
        zero_frac  = [n == 0 ? NaN : nz / n
                      for (nz, n) in zip(date_nzero, date_nzero .+ date_npos)]
    )

    println("  Zero-inflation summary:")
    @printf("    n_total = %d  |  n_zero = %d  |  n_positive = %d  |  zero_frac = %.1f%%\n",
            n_total, n_zero, n_pos, 100.0 * n_zero / max(1, n_total))
    if !isempty(pos_v)
        @printf("    Positive range: %.3f – %.3f cm\n", pos_range[1], pos_range[2])
    end

    return (n_total=n_total, n_zero=n_zero, n_positive=n_pos,
            zero_frac=n_zero / max(1, n_total),
            pos_range=pos_range, by_depth=by_depth, by_date=by_date)
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: balanced index pre-computation
# ─────────────────────────────────────────────────────────────────────────────
"""
Pre-compute balanced index subsets per day before the CMA-ES loop.
Returns Dict{Int, Vector{Int}}: day => sorted indices to keep.
"""
function prepare_balanced_indices(obs_by_day::Dict{Int,Vector{Float64}},
                                  used_days::Vector{Int},
                                  cfg::CalibrationObjectiveConfig)
    rng = MersenneTwister(cfg.sampling_seed)
    out = Dict{Int, Vector{Int}}()
    for d in used_days
        obs     = get(obs_by_day, d, Float64[])
        isempty(obs) && continue
        pos_idx  = findall(o -> o > cfg.presence_threshold, obs)
        zero_idx = findall(o -> o <= cfg.presence_threshold, obs)
        n_keep   = round(Int, cfg.zero_to_positive_ratio * length(pos_idx))
        n_keep   = min(n_keep, length(zero_idx))
        if isempty(pos_idx)
            # No positives: keep 10% of zeros to avoid zero-gradient on all-zero days
            n_keep = max(1, round(Int, 0.10 * length(zero_idx)))
        end
        keep_zero = n_keep > 0 ? shuffle!(rng, copy(zero_idx))[1:n_keep] : Int[]
        out[d] = sort(vcat(pos_idx, keep_zero))
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Summary features (used by modes 5 & 6)
# ─────────────────────────────────────────────────────────────────────────────
struct SummaryFeatures
    total_root_cm::Float64
    depth_profile::Vector{Float64}      # summed root by depth bin
    time_profile::Vector{Float64}       # total root per day
    occupancy_rate::Float64             # fraction of windows with root > threshold
    max_rooting_depth_cm::Float64       # deepest window with root > threshold
    center_of_mass_depth_cm::Float64    # root-length-weighted mean depth
end

function compute_summary_features(obs_by_day::Dict{Int,Vector{Float64}},
                                  meta_by_day::Dict{Int,DataFrame},
                                  used_days::Vector{Int},
                                  cfg::CalibrationObjectiveConfig)::SummaryFeatures
    n_bins  = length(cfg.depth_bin_edges_cm) - 1
    dp      = zeros(Float64, n_bins)
    tp      = zeros(Float64, length(used_days))
    total   = 0.0
    n_all   = 0
    n_pos   = 0
    dw_sum  = 0.0
    dw_tot  = 0.0
    max_d   = 0.0

    for (ti, day) in enumerate(used_days)
        obs  = get(obs_by_day,  day, Float64[])
        meta = get(meta_by_day, day, DataFrame())
        isempty(obs) && continue
        has_depth = hasproperty(meta, :depth_cm) && nrow(meta) == length(obs)

        for i in eachindex(obs)
            v = max(0.0, obs[i])
            total    += v
            tp[ti]   += v
            n_all    += 1
            v > cfg.rooting_depth_threshold_cm && (n_pos += 1)

            depth = has_depth ? Float64(meta.depth_cm[i]) : 0.0
            bin   = searchsortedlast(cfg.depth_bin_edges_cm, depth)
            bin   = clamp(bin, 1, n_bins)
            dp[bin] += v

            dw_sum += v * depth
            dw_tot += v
            if v > cfg.rooting_depth_threshold_cm
                max_d = max(max_d, depth)
            end
        end
    end

    occ = n_all > 0 ? n_pos / n_all : 0.0
    com = dw_tot > 0 ? dw_sum / dw_tot : 0.0

    return SummaryFeatures(total, dp, tp, occ, max_d, com)
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: rmse_baseline
# Plain SSE including zero-zero pairs, divided by n_windows per day then averaged
# ─────────────────────────────────────────────────────────────────────────────
function loss_rmse_baseline(obs_by_day::Dict{Int,Vector{Float64}},
                             pred_by_day::Dict{Int,Vector{Float64}},
                             used_days::Vector{Int})::Float64
    J = 0.0; n = 0
    for d in used_days
        obs  = get(obs_by_day,  d, Float64[])
        pred = get(pred_by_day, d, Float64[])
        (isempty(obs) || length(obs) != length(pred)) && continue
        J += sum((pred .- obs).^2) / length(obs)
        n += 1
    end
    return n > 0 ? J / n : 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: weighted_rmse_zero_downweighted
# Per-window weight: positive_weight for obs > presence_threshold, else zero_weight
# ─────────────────────────────────────────────────────────────────────────────
function loss_weighted_rmse(obs_by_day::Dict{Int,Vector{Float64}},
                             pred_by_day::Dict{Int,Vector{Float64}},
                             used_days::Vector{Int},
                             cfg::CalibrationObjectiveConfig)::Float64
    Js = Float64[]
    for d in used_days
        obs  = get(obs_by_day,  d, Float64[])
        pred = get(pred_by_day, d, Float64[])
        (isempty(obs) || length(obs) != length(pred)) && continue

        w  = [o > cfg.presence_threshold ? cfg.positive_weight : cfg.zero_weight for o in obs]
        sw = sum(w)
        sw == 0 && continue

        if cfg.use_log1p
            obs_t  = max.(obs,  0.0)
            pred_t = max.(pred, 0.0)
            if cfg.loss_metric == "mae"
                push!(Js, sum(w .* abs.(log1p.(pred_t) .- log1p.(obs_t))) / sw)
            else
                push!(Js, sqrt(sum(w .* (log1p.(pred_t) .- log1p.(obs_t)).^2) / sw))
            end
        else
            if cfg.loss_metric == "mae"
                push!(Js, sum(w .* abs.(pred .- obs)) / sw)
            else
                push!(Js, sqrt(sum(w .* (pred .- obs).^2) / sw))
            end
        end
    end
    return isempty(Js) ? 0.0 : mean(Js)
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: balanced_zero_sampling
# Uses pre-computed balanced_idx (computed once before CMA-ES loop)
# ─────────────────────────────────────────────────────────────────────────────
function loss_balanced_sampling(obs_by_day::Dict{Int,Vector{Float64}},
                                 pred_by_day::Dict{Int,Vector{Float64}},
                                 used_days::Vector{Int},
                                 balanced_idx::Dict{Int,Vector{Int}})::Float64
    J = 0.0; n = 0
    for d in used_days
        obs  = get(obs_by_day,  d, Float64[])
        pred = get(pred_by_day, d, Float64[])
        idx  = get(balanced_idx, d, Int[])
        (isempty(idx) || length(obs) != length(pred)) && continue
        o = obs[idx]; p = pred[idx]
        isempty(o) && continue
        J += sum((p .- o).^2) / length(o)
        n += 1
    end
    return n > 0 ? J / n : 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode 4: two_part_hurdle
# Component 1: balanced accuracy for presence/absence detection
# Component 2: log1p RMSE on sections where obs > presence_threshold
# ─────────────────────────────────────────────────────────────────────────────
function loss_two_part_hurdle(obs_by_day::Dict{Int,Vector{Float64}},
                               pred_by_day::Dict{Int,Vector{Float64}},
                               used_days::Vector{Int},
                               cfg::CalibrationObjectiveConfig)::Float64
    all_obs  = Float64[]
    all_pred = Float64[]
    for d in used_days
        obs  = get(obs_by_day,  d, Float64[])
        pred = get(pred_by_day, d, Float64[])
        (isempty(obs) || length(obs) != length(pred)) && continue
        append!(all_obs,  obs)
        append!(all_pred, pred)
    end
    isempty(all_obs) && return 0.0

    thr    = cfg.presence_threshold
    obs_b  = all_obs  .> thr
    pred_b = all_pred .> thr

    tp = count(obs_b  .& pred_b)
    tn = count(.!obs_b .& .!pred_b)
    fp = count(.!obs_b .& pred_b)
    fn = count(obs_b  .& .!pred_b)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    presence_loss = 1.0 - (sensitivity + specificity) / 2.0

    pos_obs = all_obs[obs_b]
    pos_pred = all_pred[obs_b]
    intensity_loss = if isempty(pos_obs)
        0.0
    else
        sqrt(mean((log1p.(max.(pos_pred, 0.0)) .- log1p.(max.(pos_obs, 0.0))).^2))
    end

    return cfg.w_presence * presence_loss + cfg.w_intensity * intensity_loss
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode 5: summary_feature_loss
# Normalized relative error on 6 aggregate statistics
# ─────────────────────────────────────────────────────────────────────────────
function loss_summary_features(obs_by_day::Dict{Int,Vector{Float64}},
                                pred_by_day::Dict{Int,Vector{Float64}},
                                meta_by_day::Dict{Int,DataFrame},
                                used_days::Vector{Int},
                                cfg::CalibrationObjectiveConfig)::Float64
    obs_f  = compute_summary_features(obs_by_day,  meta_by_day, used_days, cfg)
    pred_f = compute_summary_features(pred_by_day, meta_by_day, used_days, cfg)
    ε = 1e-6

    # Total root: relative absolute error
    e_total = abs(obs_f.total_root_cm - pred_f.total_root_cm) /
              max(obs_f.total_root_cm, ε)

    # Depth profile: normalized RMSE of fractional profiles
    obs_dp_sum  = max(sum(obs_f.depth_profile),  ε)
    pred_dp_sum = max(sum(pred_f.depth_profile), ε)
    obs_dp_n    = obs_f.depth_profile  ./ obs_dp_sum
    pred_dp_n   = pred_f.depth_profile ./ pred_dp_sum
    e_depth = sqrt(mean((obs_dp_n .- pred_dp_n).^2))

    # Time profile: normalized RMSE of fractional profiles
    obs_tp_sum  = max(sum(obs_f.time_profile),  ε)
    pred_tp_sum = max(sum(pred_f.time_profile), ε)
    obs_tp_n    = obs_f.time_profile  ./ obs_tp_sum
    pred_tp_n   = pred_f.time_profile ./ pred_tp_sum
    e_time = sqrt(mean((obs_tp_n .- pred_tp_n).^2))

    # Occupancy: absolute error (already in [0,1])
    e_occ = abs(obs_f.occupancy_rate - pred_f.occupancy_rate)

    # Max rooting depth: relative absolute error
    e_rd = abs(obs_f.max_rooting_depth_cm - pred_f.max_rooting_depth_cm) /
           max(obs_f.max_rooting_depth_cm, ε)

    # Center of mass depth: relative absolute error
    e_com = abs(obs_f.center_of_mass_depth_cm - pred_f.center_of_mass_depth_cm) /
            max(obs_f.center_of_mass_depth_cm, ε)

    return (cfg.w_total    * e_total +
            cfg.w_depth    * e_depth +
            cfg.w_time     * e_time  +
            cfg.w_occupancy * e_occ  +
            cfg.w_rooting_depth * e_rd +
            cfg.w_center_depth  * e_com)
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode 6: multi_objective_zero_aware
# 5 components: total root, depth profile, presence/absence (balanced acc),
#               positive intensity (log1p), max rooting depth
# Also used for mode 8 (replicate_seed_multi_objective) on averaged predictions
# ─────────────────────────────────────────────────────────────────────────────
function loss_multi_objective(obs_by_day::Dict{Int,Vector{Float64}},
                               pred_by_day::Dict{Int,Vector{Float64}},
                               meta_by_day::Dict{Int,DataFrame},
                               used_days::Vector{Int},
                               cfg::CalibrationObjectiveConfig)::Float64
    ε = 1e-6
    obs_f  = compute_summary_features(obs_by_day,  meta_by_day, used_days, cfg)
    pred_f = compute_summary_features(pred_by_day, meta_by_day, used_days, cfg)

    # 1. Total root (relative error)
    e_total = abs(obs_f.total_root_cm - pred_f.total_root_cm) /
              max(obs_f.total_root_cm, ε)

    # 2. Depth profile (normalized RMSE)
    obs_dp_sum  = max(sum(obs_f.depth_profile),  ε)
    pred_dp_sum = max(sum(pred_f.depth_profile), ε)
    obs_dp_n    = obs_f.depth_profile  ./ obs_dp_sum
    pred_dp_n   = pred_f.depth_profile ./ pred_dp_sum
    e_depth = sqrt(mean((obs_dp_n .- pred_dp_n).^2))

    # 3. Presence / absence (balanced accuracy, all days combined)
    all_obs  = Float64[]
    all_pred = Float64[]
    for d in used_days
        o = get(obs_by_day,  d, Float64[])
        p = get(pred_by_day, d, Float64[])
        (isempty(o) || length(o) != length(p)) && continue
        append!(all_obs, o); append!(all_pred, p)
    end
    thr  = cfg.presence_threshold
    ob   = all_obs  .> thr
    pb   = all_pred .> thr
    tp   = count(ob  .& pb)
    tn   = count(.!ob .& .!pb)
    fp   = count(.!ob .& pb)
    fn   = count(ob  .& .!pb)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    e_presence = 1.0 - (sens + spec) / 2.0

    # 4. Positive intensity (log1p RMSE on sections where obs > threshold)
    pos_mask   = all_obs .> thr
    e_intensity = if count(pos_mask) > 0
        sqrt(mean((log1p.(max.(all_pred[pos_mask], 0.0)) .-
                   log1p.(max.(all_obs[pos_mask],  0.0))).^2))
    else
        0.0
    end

    # 5. Max rooting depth (relative error)
    e_rd = abs(obs_f.max_rooting_depth_cm - pred_f.max_rooting_depth_cm) /
           max(obs_f.max_rooting_depth_cm, ε)

    return (cfg.w_total_root       * e_total     +
            cfg.w_depth_profile    * e_depth      +
            cfg.w_presence_absence * e_presence   +
            cfg.w_positive_intensity * e_intensity +
            cfg.w_rooting_depth_mo * e_rd)
end

# ─────────────────────────────────────────────────────────────────────────────
# Angle loss functions
# ─────────────────────────────────────────────────────────────────────────────

# Normalized MSE on length-weighted mean crossing angle. Skips NaN windows.
function loss_angle_mean(obs_angle_by_day::Dict{Int,Vector{Float64}},
                          pred_angle_by_day::Dict{Int,Vector{Float64}},
                          used_days::Vector{Int};
                          normalize_deg::Float64=45.0)::Float64
    J = 0.0; n = 0; norm2 = normalize_deg^2
    for d in used_days
        oa = get(obs_angle_by_day,  d, Float64[])
        pa = get(pred_angle_by_day, d, Float64[])
        (isempty(oa) || length(oa) != length(pa)) && continue
        for i in eachindex(oa)
            (isnan(oa[i]) || isnan(pa[i])) && continue
            J += (pa[i] - oa[i])^2 / norm2
            n += 1
        end
    end
    return n > 0 ? J / n : 0.0
end

# MSE on normalized angle histograms (6 bins). Skips windows where either
# obs or pred has no visible roots (NaN row). Each row must sum to 1.0.
function loss_angle_histogram(obs_hist_by_day::Dict{Int,Matrix{Float64}},
                               pred_hist_by_day::Dict{Int,Matrix{Float64}},
                               used_days::Vector{Int})::Float64
    J = 0.0; n = 0
    for d in used_days
        Oh = get(obs_hist_by_day,  d, nothing)
        Ph = get(pred_hist_by_day, d, nothing)
        (Oh === nothing || Ph === nothing) && continue
        size(Oh) == size(Ph) || continue
        nb = size(Oh, 2)
        for w in 1:size(Oh, 1)
            any(isnan, view(Oh, w, :)) && continue
            any(isnan, view(Ph, w, :)) && continue
            J += sum((view(Ph, w, :) .- view(Oh, w, :)).^2) / nb
            n += 1
        end
    end
    return n > 0 ? J / n : 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Top-level dispatch
# All modes go through compute_loss. For modes 7 and 8, the caller
# must pass already-averaged pred_by_day and specify a base_mode.
# balanced_idx is only consulted when objective == "balanced_zero_sampling".
# ─────────────────────────────────────────────────────────────────────────────
function compute_loss(obs_by_day::Dict{Int,Vector{Float64}},
                      pred_by_day::Dict{Int,Vector{Float64}},
                      meta_by_day::Dict{Int,DataFrame},
                      used_days::Vector{Int},
                      cfg::CalibrationObjectiveConfig;
                      balanced_idx::Dict{Int,Vector{Int}} = Dict{Int,Vector{Int}}(),
                      base_mode::String = "")::Float64

    mode = isempty(base_mode) ? cfg.objective : base_mode

    if mode == "rmse_baseline"
        return loss_rmse_baseline(obs_by_day, pred_by_day, used_days)

    elseif mode in ("weighted_rmse_zero_downweighted", "log1p_uniform")
        return loss_weighted_rmse(obs_by_day, pred_by_day, used_days, cfg)

    elseif mode == "balanced_zero_sampling"
        return loss_balanced_sampling(obs_by_day, pred_by_day, used_days, balanced_idx)

    elseif mode == "two_part_hurdle"
        return loss_two_part_hurdle(obs_by_day, pred_by_day, used_days, cfg)

    elseif mode == "summary_feature_loss"
        return loss_summary_features(obs_by_day, pred_by_day, meta_by_day, used_days, cfg)

    elseif mode in ("multi_objective_zero_aware", "replicate_seed_multi_objective")
        return loss_multi_objective(obs_by_day, pred_by_day, meta_by_day, used_days, cfg)

    elseif mode == "replicate_seed_average"
        # After averaging, evaluate with weighted RMSE
        return loss_weighted_rmse(obs_by_day, pred_by_day, used_days, cfg)

    else
        error("Unknown objective mode: '$mode'")
    end
end
