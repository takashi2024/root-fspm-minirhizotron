# validate_comparison.jl
#
# Held-out validation comparison: 1-tube vs 3-tube calibration.
#
# Uses VALIDATION_SEEDS that were NOT used during calibration (11,22) or obs
# generation (42), giving an unbiased estimate of prediction quality for each
# approach.  Reports mean ± std of fit metrics across 5 seeds — the std
# reduction for 3-tube quantifies the benefit of multiple tube replicates.
#
# Inputs:
#   calibration/ParameterVD_best_mr.json          (1-tube calibrated params)
#   calibration/ParameterVD_best_mr_3tubes.json   (3-tube calibrated params)
#   calibration/data/mr_obs_stations_1tube.csv
#   calibration/data/mr_obs_stations_3tubes.csv
#   calibration/Parameter_calib_minirhizotron.json (base env)
#   calibration/Parameter_VD_template.json
#
# Outputs in calibration/diagnostics_mr/comparison/:
#   validation_metrics_1tube.csv
#   validation_metrics_3tubes.csv
#   validation_summary.csv
#   validation_scatter_1tube.png
#   validation_scatter_3tubes.png
#   validation_boxplot_rmse.png
#   validation_boxplot_pbias.png
#   validation_boxplot_r2.png
#
# Run from repo root:
#   julia calibration/validate_comparison.jl

using JSON
using JSON3
using CSV
using DataFrames
using Statistics
using Dates
using Random
using Printf
import Plots

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
const VALIDATION_SEEDS   = (33, 44, 55, 66, 77)
const CALIBRATION_SEEDS  = (11, 22)
const SOWING_DATE        = Date(2025, 4, 1)
const TUBE_X_CM_3TUBES   = [25.0, 50.0, 75.0]
const TUBE2_IDX          = 2   # index of x=50 cm tube in TUBE_X_CM_3TUBES

# ──────────────────────────────────────────────────────────────────────────────
# Include model code
# ──────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "Deps.jl"))
include(joinpath(@__DIR__, "..", "src", "Types.jl"))
include(joinpath(@__DIR__, "..", "src", "Geometry.jl"))
include(joinpath(@__DIR__, "..", "src", "DensityRegulation.jl"))
include(joinpath(@__DIR__, "..", "src", "GrowthStep.jl"))
include(joinpath(@__DIR__, "..", "src", "MetricsAndOutput.jl"))
include(joinpath(@__DIR__, "..", "src", "Plotting.jl"))
include(joinpath(@__DIR__, "..", "src", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "Simulation.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────────────────────
read_json(path::AbstractString) = open(path, "r") do io; JSON.parse(String(read(io))); end
function write_json_pretty(path::AbstractString, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

# ──────────────────────────────────────────────────────────────────────────────
# Date helper
# ──────────────────────────────────────────────────────────────────────────────
date_to_simday(d::Date) = Dates.value(d - SOWING_DATE) + 1

# ──────────────────────────────────────────────────────────────────────────────
# Read calibration metadata (J and seeds) written by calibration scripts
# ──────────────────────────────────────────────────────────────────────────────
function read_calibration_info(path::AbstractString)
    d = read_json(path)
    J     = Float64(get(d, "calibration_J",     NaN))
    seeds = get(d, "calibration_seeds", Int[])
    return J, seeds
end

# ──────────────────────────────────────────────────────────────────────────────
# Read calibrated θ from best-param JSON
# ──────────────────────────────────────────────────────────────────────────────
function read_theta_from_json(path::AbstractString)
    d = read_json(path)
    θ = [
        Float64(get(d, "elongation_rate_min",     0.05)),
        Float64(get(d, "elongation_rate_max",     0.70)),
        Float64(get(d, "elongation_logistic_k",   0.12)),
        Float64(get(d, "elongation_logistic_t50", 20.0)),
        Float64(get(d, "branch_angle_order1",     80.0)),
        Float64(get(d, "g_axial",                  0.05)),
        Float64(get(d, "branching_prob_order1",    0.20)),
        Float64(get(d, "branching_prob_order2",    0.30)),
    ]
    n_axial    = Int(get(d, "n_axial_roots",        1))
    max_order1 = Int(get(d, "max_order1_per_plant", 20))
    return θ, n_axial, max_order1
end

# ──────────────────────────────────────────────────────────────────────────────
# Observation loaders (copied verbatim from calibration scripts)
# ──────────────────────────────────────────────────────────────────────────────
function load_obs_by_simday(obs_csv::AbstractString)
    df = CSV.read(obs_csv, DataFrame)
    df.date = Date.(string.(df.date))
    sort!(df, [:date, :station_idx, :azimuth_deg])
    obs_by_day  = Dict{Int, Vector{Float64}}()
    meta_by_day = Dict{Int, DataFrame}()
    for subdate in groupby(df, :date)
        d  = subdate.date[1]
        sd = date_to_simday(d)
        sd >= 1 || continue
        obs_by_day[sd] = Float64.(subdate.root_length_cm)
        meta_by_day[sd] = DataFrame(
            station_idx   = Int.(subdate.station_idx),
            along_tube_cm = Float64.(subdate.along_tube_cm),
            depth_cm      = Float64.(subdate.depth_cm),
            azimuth_deg   = Float64.(subdate.azimuth_deg)
        )
    end
    return sort(collect(keys(obs_by_day))), obs_by_day, meta_by_day
end

function load_obs_by_simday_3tubes(obs_csv::AbstractString)
    df = CSV.read(obs_csv, DataFrame)
    df.date = Date.(string.(df.date))
    sort!(df, [:tube_id, :date, :station_idx, :azimuth_deg])
    obs_by_tube_day = Dict{Int, Dict{Int, Vector{Float64}}}()
    meta_by_day     = Dict{Int, DataFrame}()
    all_days        = Set{Int}()
    for sub_tube in groupby(df, :tube_id)
        tid = sub_tube.tube_id[1]
        obs_by_tube_day[tid] = Dict{Int, Vector{Float64}}()
        for sub_date in groupby(sub_tube, :date)
            d  = sub_date.date[1]
            sd = date_to_simday(d)
            sd >= 1 || continue
            obs_by_tube_day[tid][sd] = Float64.(sub_date.root_length_cm)
            push!(all_days, sd)
            if tid == 1 && !haskey(meta_by_day, sd)
                meta_by_day[sd] = DataFrame(
                    station_idx   = Int.(sub_date.station_idx),
                    along_tube_cm = Float64.(sub_date.along_tube_cm),
                    depth_cm      = Float64.(sub_date.depth_cm),
                    azimuth_deg   = Float64.(sub_date.azimuth_deg)
                )
            end
        end
    end
    return sort(collect(all_days)), obs_by_tube_day, meta_by_day
end

# ──────────────────────────────────────────────────────────────────────────────
# Candidate JSON writer
# ──────────────────────────────────────────────────────────────────────────────
function write_candidate_VD_json(template_path::AbstractString,
                                 out_path::AbstractString,
                                 θ::Vector{Float64};
                                 n_axial_roots::Int,
                                 max_order1_per_plant::Int)
    d = read_json(template_path)
    d["root_elongation_mode"]    = "logistic"
    d["elongation_rate_min"]     = θ[1]
    d["elongation_rate_max"]     = θ[2]
    d["elongation_logistic_k"]   = θ[3]
    d["elongation_logistic_t50"] = θ[4]
    d["branch_angle_order1"]     = θ[5]
    d["g_axial"]                 = θ[6]
    d["branching_prob_order1"]   = θ[7]
    d["branching_prob_order2"]   = θ[8]
    d["n_axial_roots"]           = n_axial_roots
    d["max_order1_per_plant"]    = max_order1_per_plant
    write_json_pretty(out_path, d)
end

# ──────────────────────────────────────────────────────────────────────────────
# Simulation runners (adapted from calibration scripts; use validation tmp dir)
# ──────────────────────────────────────────────────────────────────────────────
const VALIDATION_TMP = joinpath(@__DIR__, "__calib_tmp_mr_validation__")

# Build a Minirhizotron at a given from-left x position.
function make_tube(x_from_left_cm::Float64; W_cm::Float64=100.0)
    origin    = [x_from_left_cm - W_cm/2, 20.0, 0.0]
    direction = normalize_vec([0.0, 0.7071067811865476, -0.7071067811865476])
    return Minirhizotron(
        true, origin, direction, 80.0, 3.0, 1.0, 5.0, 5.0,
        collect(2.5:5.0:72.5), 0.1, "tube_stations_daily.csv",
        [0.0, 90.0, 180.0, 270.0]
    )
end

function run_validation_1tube(env_json_path::AbstractString,
                               vd_template_path::AbstractString,
                               θ::Vector{Float64},
                               need_days::Vector{Int};
                               seed::Int,
                               n_axial_roots::Int,
                               max_order1_per_plant::Int)
    mkpath(VALIDATION_TMP)
    runid  = string("v1t_s", seed, "_", abs(hash(θ)) % 10^8)
    outdir = joinpath(VALIDATION_TMP, runid)
    mkpath(outdir)

    vd_cand = joinpath(outdir, "VD.json")
    write_candidate_VD_json(vd_template_path, vd_cand, θ;
        n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)

    env = read_json(env_json_path)
    env["species_params_files"]["VD"]        = vd_cand
    env["simulation"]["seed"]                = seed
    env["simulation"]["days"]                = maximum(need_days)
    env["simulation"]["outdir"]              = outdir
    env["simulation"]["plot3d_outdir"]       = joinpath(outdir, "plot3d")
    env["facilitation"]["enabled"]           = false
    env["density_regulation"]["enabled"]     = false
    env["visualization_2d"]["enabled"]       = false
    env["minirhizotron"]["csv_name"]         = "tube_stations_daily.csv"

    mr_csv   = joinpath(outdir, "tube_stations_daily.csv")
    env_path = joinpath(outdir, "env.json")
    write_json_pretty(env_path, env)

    cfg = load_environment_and_plants(env_path)
    simulate!(cfg.plants, cfg.box, cfg.reg;
        days=maximum(need_days), seed=seed, save_every=0,
        outdir=cfg.outdir, plot3d_dir=cfg.plot3d_dir,
        viz_enabled=false, vm_enabled=false, vg_enabled=false,
        snapshot_enabled=false, show_progress=false,
        minirhizotron=cfg.mr, mr_csv_path=mr_csv)

    df = CSV.read(mr_csv, DataFrame)
    sort!(df, [:day, :station_idx, :azimuth_deg])
    out = Dict{Int, Vector{Float64}}()
    for sd in need_days
        sub = df[df.day .== sd, :]
        out[sd] = nrow(sub) > 0 ? Float64.(sub.length_cm) : Float64[]
    end
    return out
end

function run_validation_3tubes(env_json_path::AbstractString,
                                vd_template_path::AbstractString,
                                θ::Vector{Float64},
                                need_days::Vector{Int};
                                seed::Int,
                                n_axial_roots::Int,
                                max_order1_per_plant::Int)
    mkpath(VALIDATION_TMP)
    runid  = string("v3t_s", seed, "_", abs(hash(θ)) % 10^8)
    outdir = joinpath(VALIDATION_TMP, runid)
    mkpath(outdir)

    vd_cand = joinpath(outdir, "VD.json")
    write_candidate_VD_json(vd_template_path, vd_cand, θ;
        n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)

    env = read_json(env_json_path)
    env["species_params_files"]["VD"]      = vd_cand
    env["simulation"]["seed"]              = seed
    env["simulation"]["days"]              = maximum(need_days)
    env["simulation"]["outdir"]            = outdir
    env["simulation"]["plot3d_outdir"]     = joinpath(outdir, "plot3d")
    env["facilitation"]["enabled"]         = false
    env["density_regulation"]["enabled"]   = false
    env["minirhizotron"]["enabled"]        = false

    env_path = joinpath(outdir, "env.json")
    write_json_pretty(env_path, env)

    cfg      = load_environment_and_plants(env_path)
    plants   = cfg.plants
    box      = cfg.box
    reg      = cfg.reg
    tubes    = [make_tube(tx) for tx in TUBE_X_CM_3TUBES]
    need_set = Set(need_days)
    results  = Dict{Int, Dict{Int, Vector{Float64}}}(
        tid => Dict{Int, Vector{Float64}}() for tid in 1:length(TUBE_X_CM_3TUBES))

    R_dummy = zeros(1, 1)
    Random.seed!(seed)
    for day in 1:maximum(need_days)
        for pl in plants
            step_plant!(pl, box, reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end
        if day in need_set
            for (tid, tube) in enumerate(tubes)
                ns  = length(tube.station_positions_cm)
                na  = length(tube.camera_azimuths_deg)
                mat = tube_station_azimuth_lengths_cm(plants, tube)
                results[tid][day] = [mat[si, ai] for si in 1:ns for ai in 1:na]
            end
        end
    end
    return results
end

# ──────────────────────────────────────────────────────────────────────────────
# Metrics (identical to calibration scripts)
# ──────────────────────────────────────────────────────────────────────────────
function regression_metrics(x::Vector{Float64}, y::Vector{Float64})
    n    = length(x)
    n < 2 && return (rmse=NaN, r2=NaN, ccc=NaN, kge=NaN, pbias=NaN)
    rmse = sqrt(mean((y .- x).^2))
    x̄    = mean(x); ȳ = mean(y)
    sst  = sum((x .- x̄).^2)
    sse  = sum((y .- x).^2)
    r2   = sst == 0 ? NaN : 1.0 - sse / sst
    vx   = sum((x .- x̄).^2) / (n-1)
    vy   = sum((y .- ȳ).^2) / (n-1)
    cxy  = sum((x .- x̄) .* (y .- ȳ)) / (n-1)
    ccc_d = vx + vy + (x̄ - ȳ)^2
    ccc  = ccc_d == 0 ? NaN : (2cxy) / ccc_d
    r_p  = (vx == 0 || vy == 0) ? NaN : cxy / sqrt(vx * vy)
    β    = x̄ == 0 ? NaN : ȳ / x̄
    σx   = sqrt(max(vx, 0.0)); σy = sqrt(max(vy, 0.0))
    γ    = (x̄ == 0 || ȳ == 0 || σx == 0) ? NaN : (σy / ȳ) / (σx / x̄)
    kge  = (isnan(r_p) || isnan(β) || isnan(γ)) ? NaN :
           1.0 - sqrt((r_p-1)^2 + (β-1)^2 + (γ-1)^2)
    pbias = sum(x) == 0 ? NaN : 100.0 * sum(x .- y) / sum(x)
    return (rmse=rmse, r2=r2, ccc=ccc, kge=kge, pbias=pbias)
end

# ──────────────────────────────────────────────────────────────────────────────
# Flatten obs/pred across days into one pair of vectors
# ──────────────────────────────────────────────────────────────────────────────
function flatten_1tube(used_days, obs_by_day, sim_by_day)
    all_obs  = Float64[]
    all_pred = Float64[]
    for d in used_days
        obs  = get(obs_by_day, d, Float64[])
        pred = get(sim_by_day, d, Float64[])
        length(obs) == length(pred) > 0 || continue
        # Mirror loss function: exclude zero-zero windows
        nz = (obs .> 0.0) .| (pred .> 0.0)
        append!(all_obs,  obs[nz])
        append!(all_pred, pred[nz])
    end
    return all_obs, all_pred
end

function flatten_3tubes(used_days, obs_by_tube_day, sim_by_tube)
    all_obs  = Float64[]
    all_pred = Float64[]
    all_tid  = Int[]
    for tid in 1:length(TUBE_X_CM_3TUBES)
        obs_day = get(obs_by_tube_day, tid, Dict{Int,Vector{Float64}}())
        sim_day = get(sim_by_tube,     tid, Dict{Int,Vector{Float64}}())
        for d in used_days
            obs  = get(obs_day, d, Float64[])
            pred = get(sim_day, d, Float64[])
            length(obs) == length(pred) > 0 || continue
            # Mirror loss function: exclude zero-zero windows
            nz = (obs .> 0.0) .| (pred .> 0.0)
            append!(all_obs,  obs[nz])
            append!(all_pred, pred[nz])
            append!(all_tid,  fill(tid, sum(nz)))
        end
    end
    return all_obs, all_pred, all_tid
end

# ──────────────────────────────────────────────────────────────────────────────
# Per-seed validation runners
# ──────────────────────────────────────────────────────────────────────────────
function validate_1tube(env_json, vd_template, θ1, n_ax1, mo1,
                        used_days1, obs_by_day1)
    rows = DataFrame(seed=Int[], rmse=Float64[], r2=Float64[],
                     ccc=Float64[], kge=Float64[], pbias=Float64[])
    for s in VALIDATION_SEEDS
        @info "  1-tube seed=$s"
        sim = run_validation_1tube(env_json, vd_template, θ1, used_days1;
            seed=s, n_axial_roots=n_ax1, max_order1_per_plant=mo1)
        obs_v, pred_v = flatten_1tube(used_days1, obs_by_day1, sim)
        m = regression_metrics(obs_v, pred_v)
        push!(rows, (s, m.rmse, m.r2, m.ccc, m.kge, m.pbias))
    end
    return rows
end

function validate_3tubes(env_json, vd_template, θ3, n_ax3, mo3,
                         used_days3, obs_by_tube_day3,
                         obs_by_day1, used_days1)
    rows_agg  = DataFrame(seed=Int[], rmse=Float64[], r2=Float64[],
                          ccc=Float64[], kge=Float64[], pbias=Float64[])
    rows_pt   = DataFrame(seed=Int[], tube_id=Int[], tube_x_cm=Float64[],
                          rmse=Float64[], r2=Float64[], ccc=Float64[],
                          kge=Float64[], pbias=Float64[])
    # fair: 3-tube model at x=50 vs same 1-tube obs used by 1-tube model
    rows_fair = DataFrame(seed=Int[], rmse=Float64[], r2=Float64[],
                          ccc=Float64[], kge=Float64[], pbias=Float64[])
    for s in VALIDATION_SEEDS
        @info "  3-tube seed=$s"
        sim = run_validation_3tubes(env_json, vd_template, θ3, used_days3;
            seed=s, n_axial_roots=n_ax3, max_order1_per_plant=mo3)
        # aggregate across all 3 tubes
        obs_v, pred_v, _ = flatten_3tubes(used_days3, obs_by_tube_day3, sim)
        m = regression_metrics(obs_v, pred_v)
        push!(rows_agg, (s, m.rmse, m.r2, m.ccc, m.kge, m.pbias))
        # per-tube metrics
        for (tid, tx) in enumerate(TUBE_X_CM_3TUBES)
            od = get(obs_by_tube_day3, tid, Dict{Int,Vector{Float64}}())
            sd = get(sim, tid, Dict{Int,Vector{Float64}}())
            ov = Float64[]; pv = Float64[]
            for d in used_days3
                o = get(od, d, Float64[]); p = get(sd, d, Float64[])
                length(o) == length(p) > 0 || continue
                append!(ov, o); append!(pv, p)
            end
            isempty(ov) && continue
            mt = regression_metrics(ov, pv)
            push!(rows_pt, (s, tid, tx, mt.rmse, mt.r2, mt.ccc, mt.kge, mt.pbias))
        end
        # fair: tube_id=TUBE2_IDX (x=50 cm) vs 1-tube obs
        sim_t2 = get(sim, TUBE2_IDX, Dict{Int,Vector{Float64}}())
        ov2, pv2 = flatten_1tube(used_days1, obs_by_day1, sim_t2)
        if !isempty(ov2)
            mf = regression_metrics(ov2, pv2)
            push!(rows_fair, (s, mf.rmse, mf.r2, mf.ccc, mf.kge, mf.pbias))
        end
    end
    return rows_agg, rows_pt, rows_fair
end

# In-sample fit on calibration seeds (11, 22) — diagnose overfitting vs bad calibration.
function validate_insample_1tube(env_json, vd_template, θ1, n_ax1, mo1,
                                  used_days1, obs_by_day1)
    rows = DataFrame(seed=Int[], rmse=Float64[], r2=Float64[],
                     ccc=Float64[], kge=Float64[], pbias=Float64[])
    for s in CALIBRATION_SEEDS
        sim = run_validation_1tube(env_json, vd_template, θ1, used_days1;
            seed=s, n_axial_roots=n_ax1, max_order1_per_plant=mo1)
        obs_v, pred_v = flatten_1tube(used_days1, obs_by_day1, sim)
        m = regression_metrics(obs_v, pred_v)
        push!(rows, (s, m.rmse, m.r2, m.ccc, m.kge, m.pbias))
    end
    return rows
end

function validate_insample_3tubes(env_json, vd_template, θ3, n_ax3, mo3,
                                   used_days3, obs_by_tube_day3)
    rows = DataFrame(seed=Int[], rmse=Float64[], r2=Float64[],
                     ccc=Float64[], kge=Float64[], pbias=Float64[])
    for s in CALIBRATION_SEEDS
        sim = run_validation_3tubes(env_json, vd_template, θ3, used_days3;
            seed=s, n_axial_roots=n_ax3, max_order1_per_plant=mo3)
        obs_v, pred_v, _ = flatten_3tubes(used_days3, obs_by_tube_day3, sim)
        m = regression_metrics(obs_v, pred_v)
        push!(rows, (s, m.rmse, m.r2, m.ccc, m.kge, m.pbias))
    end
    return rows
end

# ──────────────────────────────────────────────────────────────────────────────
# Save scatter PNG (all seeds overlaid, coloured by seed)
# ──────────────────────────────────────────────────────────────────────────────
function save_scatter_png(env_json, vd_template, θ, n_ax, mo,
                          used_days, obs_by_day, obs_by_tube_day,
                          is_3tube::Bool, outpath::AbstractString)
    Plots.gr()
    seed_colors = [:royalblue, :firebrick, :forestgreen, :darkorange, :purple]

    p = Plots.plot(; xlabel="Observed (cm)", ylabel="Predicted (cm)",
        title = is_3tube ? "Validation scatter — 3-tube (5 seeds)" :
                           "Validation scatter — 1-tube (5 seeds)",
        legend=:bottomright, legend_background_color=:transparent)

    all_obs_combined  = Float64[]
    all_pred_combined = Float64[]

    for (si, s) in enumerate(VALIDATION_SEEDS)
        if is_3tube
            sim = run_validation_3tubes(env_json, vd_template, θ, used_days;
                seed=s, n_axial_roots=n_ax, max_order1_per_plant=mo)
            obs_v, pred_v, _ = flatten_3tubes(used_days, obs_by_tube_day, sim)
        else
            sim = run_validation_1tube(env_json, vd_template, θ, used_days;
                seed=s, n_axial_roots=n_ax, max_order1_per_plant=mo)
            obs_v, pred_v = flatten_1tube(used_days, obs_by_day, sim)
        end
        isempty(obs_v) && continue
        append!(all_obs_combined,  obs_v)
        append!(all_pred_combined, pred_v)
        Plots.scatter!(p, obs_v, pred_v; ms=2, alpha=0.5,
            color=seed_colors[si], label="seed=$s")
    end

    if !isempty(all_obs_combined)
        lo = min(minimum(all_obs_combined), minimum(all_pred_combined))
        hi = max(maximum(all_obs_combined), maximum(all_pred_combined))
        Plots.plot!(p, [lo, hi], [lo, hi]; color=:black, label="1:1")
        m = regression_metrics(all_obs_combined, all_pred_combined)
        rng = max(hi - lo, 1e-6)
        x0  = lo + 0.02 * rng; ytop = hi - 0.04 * rng
        st  = @sprintf("ALL seeds combined:\nR²=%.3f  RMSE=%.3f cm\nCCC=%.3f  PBIAS=%.1f%%",
                       m.r2, m.rmse, m.ccc, m.pbias)
        Plots.annotate!(p, (x0, ytop, Plots.text(st, 8, :left)))
    end
    Plots.savefig(p, outpath)
end

# ──────────────────────────────────────────────────────────────────────────────
# Boxplot for one metric
# ──────────────────────────────────────────────────────────────────────────────
function save_metric_boxplot(df1::DataFrame, df3::DataFrame,
                              col::Symbol, ylabel::AbstractString,
                              title::AbstractString, outpath::AbstractString)
    Plots.gr()
    v1 = Float64.(filter(isfinite, df1[!, col]))
    v3 = Float64.(filter(isfinite, df3[!, col]))

    p = Plots.plot(; title=title, ylabel=ylabel,
        xticks=(1:2, ["1-tube", "3-tube"]),
        legend=:topright, legend_background_color=:transparent,
        xlims=(0.5, 2.5))

    # Individual points
    Plots.scatter!(p, fill(1, length(v1)), v1; ms=7, color=:firebrick,
        markerstrokewidth=0, label="1-tube seeds", alpha=0.8)
    Plots.scatter!(p, fill(2, length(v3)), v3; ms=7, color=:dodgerblue,
        markerstrokewidth=0, label="3-tube seeds", alpha=0.8)

    # Mean lines
    if !isempty(v1)
        m1 = mean(v1)
        Plots.plot!(p, [0.7, 1.3], [m1, m1]; color=:firebrick, linewidth=2.5,
            label=@sprintf("1-tube mean=%.3f", m1))
    end
    if !isempty(v3)
        m3 = mean(v3)
        Plots.plot!(p, [1.7, 2.3], [m3, m3]; color=:dodgerblue, linewidth=2.5,
            label=@sprintf("3-tube mean=%.3f", m3))
    end

    # Std error bars
    if !isempty(v1) && length(v1) > 1
        m1 = mean(v1); sd1 = std(v1)
        Plots.plot!(p, [1, 1], [m1-sd1, m1+sd1]; color=:firebrick, linewidth=1.5,
            label=false)
    end
    if !isempty(v3) && length(v3) > 1
        m3 = mean(v3); sd3 = std(v3)
        Plots.plot!(p, [2, 2], [m3-sd3, m3+sd3]; color=:dodgerblue, linewidth=1.5,
            label=false)
    end

    Plots.savefig(p, outpath)
end

# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
function save_summary_csv(df1::DataFrame, df3::DataFrame, outpath::AbstractString)
    metrics = [:rmse, :r2, :ccc, :kge, :pbias]
    rows    = DataFrame(metric=String[], mean_1tube=Float64[], std_1tube=Float64[],
                        mean_3tubes=Float64[], std_3tubes=Float64[])
    for col in metrics
        v1 = Float64.(filter(isfinite, df1[!, col]))
        v3 = Float64.(filter(isfinite, df3[!, col]))
        push!(rows, (
            string(col),
            isempty(v1) ? NaN : mean(v1),
            isempty(v1) ? NaN : std(v1),
            isempty(v3) ? NaN : mean(v3),
            isempty(v3) ? NaN : std(v3)
        ))
    end
    CSV.write(outpath, rows)
    return rows
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
function main()
    env_json    = joinpath(@__DIR__, "Parameter_calib_minirhizotron.json")
    vd_template = joinpath(@__DIR__, "Parameter_VD_template.json")
    best_1tube  = joinpath(@__DIR__, "ParameterVD_best_mr_1tube.json")
    best_3tubes = joinpath(@__DIR__, "ParameterVD_best_mr_3tubes.json")
    obs_1tube   = joinpath(@__DIR__, "data", "mr_obs_stations_1tube.csv")
    obs_3tubes  = joinpath(@__DIR__, "data", "mr_obs_stations_3tubes.csv")
    outdir      = joinpath(@__DIR__, "diagnostics_mr", "comparison")
    mkpath(outdir)

    # Load calibrated parameters and calibration metadata
    θ1, n_ax1, mo1   = read_theta_from_json(best_1tube)
    θ3, n_ax3, mo3   = read_theta_from_json(best_3tubes)
    calib_J1, calib_seeds1 = read_calibration_info(best_1tube)
    calib_J3, calib_seeds3 = read_calibration_info(best_3tubes)
    @info "1-tube params" n_axial_roots=n_ax1 max_order1=mo1 calib_J=calib_J1
    @info "3-tube params" n_axial_roots=n_ax3 max_order1=mo3 calib_J=calib_J3

    # Load observations
    used_days1, obs_by_day1, _   = load_obs_by_simday(obs_1tube)
    used_days3, obs_by_tube3, _  = load_obs_by_simday_3tubes(obs_3tubes)
    @info "Obs loaded" days_1tube=used_days1 days_3tubes=used_days3

    # ── Validate 1-tube ──────────────────────────────────────────────────────
    @info "Validating 1-tube calibration..."
    df1 = validate_1tube(env_json, vd_template, θ1, n_ax1, mo1,
                         used_days1, obs_by_day1)
    CSV.write(joinpath(outdir, "validation_metrics_1tube.csv"), df1)
    @info "1-tube done" df1

    # ── Validate 3-tube (aggregate + per-tube + fair x50) ───────────────────
    @info "Validating 3-tube calibration (held-out)..."
    df3_agg, df3_pt, df3_fair = validate_3tubes(
        env_json, vd_template, θ3, n_ax3, mo3,
        used_days3, obs_by_tube3, obs_by_day1, used_days1)
    CSV.write(joinpath(outdir, "validation_metrics_3tubes.csv"),          df3_agg)
    CSV.write(joinpath(outdir, "validation_metrics_3tubes_per_tube.csv"), df3_pt)
    CSV.write(joinpath(outdir, "validation_metrics_fair_x50.csv"),        df3_fair)
    @info "3-tube done" df3_agg

    # ── In-sample check (calibration seeds 11, 22) ──────────────────────────
    @info "In-sample check on calibration seeds..."
    df_is1 = validate_insample_1tube(env_json, vd_template, θ1, n_ax1, mo1,
                                      used_days1, obs_by_day1)
    df_is3 = validate_insample_3tubes(env_json, vd_template, θ3, n_ax3, mo3,
                                       used_days3, obs_by_tube3)
    CSV.write(joinpath(outdir, "validation_metrics_insample_1tube.csv"),  df_is1)
    CSV.write(joinpath(outdir, "validation_metrics_insample_3tubes.csv"), df_is3)
    @info "In-sample done" df_is1 df_is3

    # ── Summary CSV (aggregate) ──────────────────────────────────────────────
    summary = save_summary_csv(df1, df3_agg, joinpath(outdir, "validation_summary.csv"))

    # ── Metric plots ──────────────────────────────────────────────────────────
    Plots.gr()
    # Aggregate (not directly comparable — different n_obs and spatial extent)
    save_metric_boxplot(df1, df3_agg, :rmse,  "RMSE (cm)", "RMSE — 1-tube vs 3-tube (all positions)",
        joinpath(outdir, "validation_boxplot_rmse.png"))
    save_metric_boxplot(df1, df3_agg, :pbias, "PBIAS (%)", "PBIAS — 1-tube vs 3-tube (all positions)",
        joinpath(outdir, "validation_boxplot_pbias.png"))
    save_metric_boxplot(df1, df3_agg, :r2,    "R²",        "R² — 1-tube vs 3-tube (all positions)",
        joinpath(outdir, "validation_boxplot_r2.png"))
    # Fair comparison (both models at x=50 cm, same 1-tube obs)
    save_metric_boxplot(df1, df3_fair, :rmse, "RMSE (cm)", "RMSE — fair comparison (both at x=50 cm)",
        joinpath(outdir, "validation_boxplot_rmse_fair.png"))
    save_metric_boxplot(df1, df3_fair, :r2,   "R²",        "R² — fair comparison (both at x=50 cm)",
        joinpath(outdir, "validation_boxplot_r2_fair.png"))
    @info "Metric plots saved"

    # ── Scatter plots ────────────────────────────────────────────────────────
    @info "Generating scatter plots..."
    save_scatter_png(env_json, vd_template, θ1, n_ax1, mo1,
                     used_days1, obs_by_day1, nothing,
                     false, joinpath(outdir, "validation_scatter_1tube.png"))
    save_scatter_png(env_json, vd_template, θ3, n_ax3, mo3,
                     used_days3, nothing, obs_by_tube3,
                     true,  joinpath(outdir, "validation_scatter_3tubes.png"))
    @info "Scatter plots saved"

    # ── Console summary ───────────────────────────────────────────────────────
    units = Dict(:rmse=>" cm", :r2=>"   ", :ccc=>"   ", :kge=>"   ", :pbias=>" %")

    # A. Per-tube breakdown for 3-tube (spatial heterogeneity check)
    println("\n" * "="^72)
    println("  A. 3-tube per-tube breakdown  (mean ± std across $(length(VALIDATION_SEEDS)) held-out seeds)")
    println("     Unequal fit across tube positions reveals spatial imbalance in the loss.")
    println("="^72)
    @printf("  %-9s  %19s  %19s  %19s\n", "Metric", "Tube1(x=25cm)", "Tube2(x=50cm)", "Tube3(x=75cm)")
    println("-"^72)
    for metric in [:rmse, :r2, :kge, :pbias]
        u = get(units, metric, "")
        @printf("  %-9s", string(metric) * u)
        for tid in 1:3
            v = Float64.(filter(isfinite,
                df3_pt[df3_pt.tube_id .== tid, metric]))
            @printf("  %7.3f ± %6.3f   ",
                isempty(v) ? NaN : mean(v),
                isempty(v) ? NaN : std(v))
        end
        println()
    end
    println("="^72)

    # B. Fair comparison: both models at x=50 cm, same 1-tube obs
    println("\n" * "="^72)
    println("  B. FAIR comparison — both models at x=50 cm, same 1-tube obs")
    println("     Like-for-like: identical spatial position and identical observations.")
    println("="^72)
    @printf("  %-8s  %22s  %22s\n", "Metric", "1-tube (mean ± std)", "3-tube @x50 (mean ± std)")
    println("-"^72)
    for metric in [:rmse, :r2, :ccc, :kge, :pbias]
        u  = get(units, metric, "")
        v1 = Float64.(filter(isfinite, df1[!, metric]))
        vf = Float64.(filter(isfinite, df3_fair[!, metric]))
        @printf("  %-8s  %9.3f ± %8.3f%-3s  %9.3f ± %8.3f%s\n",
            string(metric),
            isempty(v1) ? NaN : mean(v1), isempty(v1) ? NaN : std(v1), u,
            isempty(vf) ? NaN : mean(vf), isempty(vf) ? NaN : std(vf), u)
    end
    println("="^72)

    # C. (reserved — aggregate table moved to E below)

    # D. Calibration summary (J + in-sample fit on training seeds)
    println("\n" * "="^72)
    println("  D. Calibration summary")
    println("="^72)
    @printf("  %-22s  %22s  %22s\n", "", "1-tube", "3-tube")
    println("-"^72)
    @printf("  %-22s  %22s  %22s\n", "Calibration seeds",
        isempty(calib_seeds1) ? string(CALIBRATION_SEEDS) : string(Tuple(calib_seeds1)),
        isempty(calib_seeds3) ? string(CALIBRATION_SEEDS) : string(Tuple(calib_seeds3)))
    @printf("  %-22s  %22d  %22d\n", "max_order1_per_plant", mo1, mo3)
    @printf("  %-22s  %22.2f  %22.2f\n", "CMA-ES best J",
        isnan(calib_J1) ? -9999.0 : calib_J1,
        isnan(calib_J3) ? -9999.0 : calib_J3)
    println("-"^72)
    println("  In-sample fit on calibration seeds ($(CALIBRATION_SEEDS)):")
    for metric in [:rmse, :r2, :kge, :pbias]
        u   = get(units, metric, "")
        vi1 = Float64.(filter(isfinite, df_is1[!, metric]))
        vi3 = Float64.(filter(isfinite, df_is3[!, metric]))
        @printf("  %-22s  %9.3f ± %8.3f%-3s  %9.3f ± %8.3f%s\n",
            string(metric),
            isempty(vi1) ? NaN : mean(vi1), isempty(vi1) ? NaN : std(vi1), u,
            isempty(vi3) ? NaN : mean(vi3), isempty(vi3) ? NaN : std(vi3), u)
    end
    println("="^72)

    # E. Full held-out validation (out-of-sample, aggregated for context)
    println("\n" * "="^72)
    println("  E. Out-of-sample held-out validation  (seeds=$(VALIDATION_SEEDS))")
    println("     [B] is the fair comparison; [A] shown for reference only")
    println("="^72)
    @printf("  %-8s  %22s  %22s  %22s\n", "Metric",
        "1-tube (mean ± std)", "3-tube agg (mean ± std)", "3-tube @x50 (mean ± std)")
    println("-"^72)
    for metric in [:rmse, :r2, :kge, :pbias]
        u  = get(units, metric, "")
        v1 = Float64.(filter(isfinite, df1[!, metric]))
        v3 = Float64.(filter(isfinite, df3_agg[!, metric]))
        vf = Float64.(filter(isfinite, df3_fair[!, metric]))
        @printf("  %-8s  %9.3f ± %6.3f%-3s  %9.3f ± %6.3f%-3s  %9.3f ± %6.3f%s\n",
            string(metric),
            isempty(v1) ? NaN : mean(v1), isempty(v1) ? NaN : std(v1), u,
            isempty(v3) ? NaN : mean(v3), isempty(v3) ? NaN : std(v3), u,
            isempty(vf) ? NaN : mean(vf), isempty(vf) ? NaN : std(vf), u)
    end
    println("="^72)

    println("\nOutputs saved to: $outdir")
end

main()
