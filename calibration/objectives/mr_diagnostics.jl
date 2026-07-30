# mr_diagnostics.jl
#
# Post-calibration diagnostic plots and reports for minirhizotron flexible calibration.
# Assumes mr_objectives.jl has already been included (provides CalibrationObjectiveConfig,
# compute_zero_inflation_stats, SummaryFeatures, compute_summary_features).
#
# Usage:
#   include("objectives/mr_diagnostics.jl")
#   save_zero_inflation_report(obs_df, experiment_dir)
#   save_calibration_diagnostics(obs_pred_df, cfg, experiment_dir)

using CSV
using DataFrames
using Statistics
using Printf
import Plots
ENV["GKSwstype"] = "nul"   # headless GR rendering — no display required

# ─────────────────────────────────────────────────────────────────────────────
# Zero-inflation report
# Writes:
#   zero_inflation_stats.csv
#   zero_inflation_histogram.png  (log-x distribution of all positive lengths)
#   zero_inflation_by_depth.png   (zero fraction bar chart by depth bin)
#   zero_inflation_by_date.png    (zero fraction bar chart by acquisition date)
# ─────────────────────────────────────────────────────────────────────────────
function save_zero_inflation_report(obs_df::DataFrame, outdir::AbstractString)
    mkpath(outdir)
    zi = compute_zero_inflation_stats(obs_df)

    # Summary CSV
    CSV.write(joinpath(outdir, "zero_inflation_by_depth.csv"), zi.by_depth)
    CSV.write(joinpath(outdir, "zero_inflation_by_date.csv"),  zi.by_date)

    summary_df = DataFrame(
        metric = ["n_total", "n_zero", "n_positive", "zero_fraction",
                  "positive_min_cm", "positive_max_cm"],
        value  = [zi.n_total, zi.n_zero, zi.n_positive,
                  zi.zero_frac, zi.pos_range[1], zi.pos_range[2]]
    )
    CSV.write(joinpath(outdir, "zero_inflation_stats.csv"), summary_df)

    Plots.gr()

    # Histogram of positive root lengths (log10 x-axis)
    pos_v = filter(x -> x > 0, Float64.(obs_df.root_length_cm))
    if !isempty(pos_v)
        p_hist = Plots.histogram(log10.(pos_v);
            bins=25, label=false,
            xlabel="log₁₀(root length cm)", ylabel="Count",
            title=@sprintf("Positive root length distribution (n=%d / %d total, %.0f%% zeros)",
                           zi.n_positive, zi.n_total, 100*zi.zero_frac))
        Plots.savefig(p_hist, joinpath(outdir, "zero_inflation_histogram.png"))
    end

    # Bar chart by depth bin
    db   = zi.by_depth
    zf   = replace(db.zero_frac, NaN => 0.0)
    p_d  = Plots.bar(db.depth_bin, zf .* 100;
        label=false, xlabel="Depth", ylabel="Zero fraction (%)",
        title="Zero fraction by depth bin",
        rotation=45, bar_width=0.7, ylims=(0, 100))
    Plots.savefig(p_d, joinpath(outdir, "zero_inflation_by_depth.png"))

    # Bar chart by date
    bd    = zi.by_date
    zfd   = replace(bd.zero_frac, NaN => 0.0)
    dates_str = string.(bd.date)
    p_dt  = Plots.bar(dates_str, zfd .* 100;
        label=false, xlabel="Date", ylabel="Zero fraction (%)",
        title="Zero fraction by acquisition date",
        rotation=45, bar_width=0.7, ylims=(0, 100))
    Plots.savefig(p_dt, joinpath(outdir, "zero_inflation_by_date.png"))

    println("  Zero-inflation report saved → $outdir")
end

# ─────────────────────────────────────────────────────────────────────────────
# Calibration diagnostics
# obs_pred_df must have columns: obs_cm, pred_cm, depth_cm, day, azimuth_deg
# Writes:
#   obs_vs_pred_scatter.png
#   obs_vs_pred_scatter_log.png    (log1p axes)
#   depth_profile_comparison.png   (mean obs & pred by depth bin, lines per date)
#   time_profile_comparison.png    (total root obs vs pred by date)
#   occupancy_by_depth.png         (positive fraction obs vs pred by depth bin)
#   presence_confusion_matrix.csv
#   summary_features_comparison.csv
#   objective_components.csv
# ─────────────────────────────────────────────────────────────────────────────
function save_calibration_diagnostics(obs_pred_df::DataFrame,
                                      cfg::CalibrationObjectiveConfig,
                                      outdir::AbstractString)
    mkpath(outdir)
    Plots.gr()

    obs  = Float64.(obs_pred_df.obs_cm)
    pred = Float64.(obs_pred_df.pred_cm)
    pred = max.(pred, 0.0)

    # ── Scatter: all points ──────────────────────────────────────────────────
    nz_mask = (obs .> 0.0) .| (pred .> 0.0)
    x = obs[nz_mask]; y = pred[nz_mask]
    if !isempty(x)
        lo = min(minimum(x), minimum(y)); hi = max(maximum(x), maximum(y))
        rng_v = max(hi - lo, 1e-6)
        p_sc = Plots.scatter(x, y; ms=2, label=false, alpha=0.6,
            xlabel="Observed (cm)", ylabel="Predicted (cm)",
            title="Obs vs Pred (non-zero windows)")
        Plots.plot!(p_sc, [lo, hi], [lo, hi]; label="1:1", color=:black)
        Plots.savefig(p_sc, joinpath(outdir, "obs_vs_pred_scatter.png"))

        # Log1p axes version
        xl = log1p.(x); yl = log1p.(y)
        lol = min(minimum(xl), minimum(yl)); hil = max(maximum(xl), maximum(yl))
        p_log = Plots.scatter(xl, yl; ms=2, label=false, alpha=0.6,
            xlabel="log(1 + Observed) cm", ylabel="log(1 + Predicted) cm",
            title="Obs vs Pred — log1p scale")
        Plots.plot!(p_log, [lol, hil], [lol, hil]; label="1:1", color=:black)
        Plots.savefig(p_log, joinpath(outdir, "obs_vs_pred_scatter_log.png"))
    end

    # ── Depth profile comparison ────────────────────────────────────────────
    if hasproperty(obs_pred_df, :depth_cm) && hasproperty(obs_pred_df, :day)
        depth_edges = cfg.depth_bin_edges_cm
        n_bins  = length(depth_edges) - 1
        bin_mid = [(depth_edges[i] + depth_edges[i+1]) / 2 for i in 1:n_bins]
        days_u  = sort(unique(obs_pred_df.day))
        colors_d = length(days_u) > 1 ? Plots.distinguishable_colors(length(days_u)) :
                   [:blue]

        p_dp = Plots.plot(; xlabel="Mean root length (cm)", ylabel="Depth midpoint (cm)",
            title="Depth profile: obs (—) vs pred (--)",
            yflip=true, legend=:bottomright, legend_background_color=:transparent)

        for (ci, d) in enumerate(days_u)
            sub = obs_pred_df[obs_pred_df.day .== d, :]
            obs_bins  = zeros(n_bins)
            pred_bins = zeros(n_bins)
            cnt_bins  = zeros(Int, n_bins)
            for r in eachrow(sub)
                b = searchsortedlast(depth_edges, Float64(r.depth_cm))
                b = clamp(b, 1, n_bins)
                obs_bins[b]  += Float64(r.obs_cm)
                pred_bins[b] += max(Float64(r.pred_cm), 0.0)
                cnt_bins[b]  += 1
            end
            cnt_bins[cnt_bins .== 0] .= 1
            col = colors_d isa Vector ? colors_d[ci] : colors_d
            Plots.plot!(p_dp, obs_bins  ./ cnt_bins, bin_mid;
                label="day=$d obs",  color=col, linestyle=:solid)
            Plots.plot!(p_dp, pred_bins ./ cnt_bins, bin_mid;
                label="day=$d pred", color=col, linestyle=:dash)
        end
        Plots.savefig(p_dp, joinpath(outdir, "depth_profile_comparison.png"))

        # ── Time profile ─────────────────────────────────────────────────────
        obs_total  = [sum(max.(obs_pred_df[obs_pred_df.day .== d, :obs_cm], 0.0))
                      for d in days_u]
        pred_total = [sum(max.(obs_pred_df[obs_pred_df.day .== d, :pred_cm], 0.0))
                      for d in days_u]
        p_tp = Plots.plot(days_u, obs_total;  label="Observed",  marker=:circle,
            xlabel="Simulation day", ylabel="Total root length (cm)",
            title="Total root length by date")
        Plots.plot!(p_tp, days_u, pred_total; label="Predicted", marker=:square,
            linestyle=:dash)
        Plots.savefig(p_tp, joinpath(outdir, "time_profile_comparison.png"))

        # ── Occupancy by depth ───────────────────────────────────────────────
        thr = cfg.presence_threshold
        obs_occ  = zeros(n_bins)
        pred_occ = zeros(n_bins)
        cnt_occ  = zeros(Int, n_bins)
        for r in eachrow(obs_pred_df)
            b = searchsortedlast(depth_edges, Float64(r.depth_cm))
            b = clamp(b, 1, n_bins)
            obs_occ[b]  += Float64(r.obs_cm)  > thr ? 1 : 0
            pred_occ[b] += max(Float64(r.pred_cm), 0.0) > thr ? 1 : 0
            cnt_occ[b]  += 1
        end
        cnt_occ[cnt_occ .== 0] .= 1
        p_oc = Plots.bar(bin_mid, obs_occ ./ cnt_occ .* 100;
            bar_width=4, label="Observed",  alpha=0.7,
            xlabel="Depth midpoint (cm)", ylabel="Occupancy (%)",
            title="Fraction of windows with root > $(thr) cm")
        Plots.bar!(p_oc, bin_mid .+ 2, pred_occ ./ cnt_occ .* 100;
            bar_width=4, label="Predicted", alpha=0.7)
        Plots.savefig(p_oc, joinpath(outdir, "occupancy_by_depth.png"))
    end

    # ── Presence confusion matrix CSV ────────────────────────────────────────
    thr  = cfg.presence_threshold
    ob   = obs  .> thr
    pb   = pred .> thr
    tp   = count(ob  .& pb)
    tn   = count(.!ob .& .!pb)
    fp   = count(.!ob .& pb)
    fn   = count(ob  .& .!pb)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv  = tp / max(1, tp + fp)
    npv  = tn / max(1, tn + fn)
    cm_df = DataFrame(
        metric = ["TP", "TN", "FP", "FN",
                  "sensitivity", "specificity", "PPV", "NPV",
                  "balanced_accuracy"],
        value  = Float64.([tp, tn, fp, fn, sens, spec, ppv, npv, (sens+spec)/2])
    )
    CSV.write(joinpath(outdir, "presence_confusion_matrix.csv"), cm_df)

    # ── Summary features comparison CSV ────────────────────────────────────
    if hasproperty(obs_pred_df, :depth_cm) && hasproperty(obs_pred_df, :day)
        days_u = sort(unique(obs_pred_df.day))
        obs_bd  = Dict(d => Float64.(obs_pred_df[obs_pred_df.day .== d, :obs_cm])
                       for d in days_u)
        pred_bd = Dict(d => max.(Float64.(obs_pred_df[obs_pred_df.day .== d, :pred_cm]), 0.0)
                       for d in days_u)
        meta_bd = Dict(d => DataFrame(depth_cm=Float64.(obs_pred_df[obs_pred_df.day .== d, :depth_cm]))
                       for d in days_u)
        obs_f  = compute_summary_features(obs_bd,  meta_bd, days_u, cfg)
        pred_f = compute_summary_features(pred_bd, meta_bd, days_u, cfg)
        sf_df = DataFrame(
            feature   = ["total_root_cm", "occupancy_rate",
                         "max_rooting_depth_cm", "center_of_mass_depth_cm"],
            observed  = [obs_f.total_root_cm,  obs_f.occupancy_rate,
                         obs_f.max_rooting_depth_cm, obs_f.center_of_mass_depth_cm],
            predicted = [pred_f.total_root_cm, pred_f.occupancy_rate,
                         pred_f.max_rooting_depth_cm, pred_f.center_of_mass_depth_cm]
        )
        CSV.write(joinpath(outdir, "summary_features_comparison.csv"), sf_df)
    end

    println("  Calibration diagnostics saved → $outdir")
end
