# main_Optimization_mr_flexible.jl
#
# Flexible CMA-ES calibration driver for minirhizotron data.
# Reads all settings (objective mode, seeds, tube setup, etc.) from a JSON config.
# Writes output to calibration/experiments/{objective}_{timestamp}/.
#
# Usage:
#   julia calibration/main_Optimization_mr_flexible.jl calibration/configs/calib_multi_objective.json
#
# Config format: see calibration/configs/calib_rmse_baseline.json for a template.
# Supported tube_setup values: "1tube" | "3tubes"
#
# NOTE: This script does NOT modify the existing calibration scripts.
#       It uses the 3-tube manual-day-loop simulation approach for both tube setups
#       (minirhizotron=nothing; in-memory sampling via tube_station_azimuth_lengths_cm).

using JSON
using JSON3
using CSV
using DataFrames
using Statistics
using Dates
using Random
using Printf
using CMAEvolutionStrategy

import Plots
ENV["GKSwstype"] = "nul"   # headless GR rendering — no display required

# ─────────────────────────────────────────────────────────────────────────────
# Parameter metadata — names, defaults, bounds
# ─────────────────────────────────────────────────────────────────────────────
const PARAM_NAMES_ALL = [
    "elongation_rate_min", "elongation_rate_max", "elongation_logistic_k",
    "elongation_logistic_t50", "branch_angle_order1", "g_axial",
    "branching_prob_order1", "branching_prob_order2"
]
const PARAM_DEFAULTS = Dict{String,Float64}(
    "elongation_rate_min"     => 0.05,
    "elongation_rate_max"     => 0.70,
    "elongation_logistic_k"   => 0.12,
    "elongation_logistic_t50" => 20.0,
    "branch_angle_order1"     => 80.0,
    "g_axial"                 => 0.05,
    "branching_prob_order1"   => 0.20,
    "branching_prob_order2"   => 0.30,
)
const PARAM_BOUNDS = Dict{String,Tuple{Float64,Float64}}(
    "elongation_rate_min"     => (0.001, 1.0),
    "elongation_rate_max"     => (0.001, 1.5),
    "elongation_logistic_k"   => (0.05,  0.4),
    "elongation_logistic_t50" => (5.0,  50.0),
    "branch_angle_order1"     => (40.0, 85.0),
    "g_axial"                 => (0.01,  0.50),
    "branching_prob_order1"   => (0.05,  1.0),
    "branching_prob_order2"   => (0.0,   1.0),
)

# ─────────────────────────────────────────────────────────────────────────────
# Objective functions + diagnostics
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "objectives", "mr_objectives.jl"))
include(joinpath(@__DIR__, "objectives", "mr_diagnostics.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Model code
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "Deps.jl"))
include(joinpath(@__DIR__, "..", "src", "Types.jl"))
include(joinpath(@__DIR__, "..", "src", "Geometry.jl"))
include(joinpath(@__DIR__, "..", "src", "DensityRegulation.jl"))
include(joinpath(@__DIR__, "..", "src", "GrowthStep.jl"))
include(joinpath(@__DIR__, "..", "src", "MetricsAndOutput.jl"))
include(joinpath(@__DIR__, "..", "src", "Plotting.jl"))
include(joinpath(@__DIR__, "..", "src", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "Simulation.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
_read_json(path::AbstractString) =
    open(path, "r") do io; JSON.parse(String(read(io))); end

function _write_json_pretty(path::AbstractString, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

# ─────────────────────────────────────────────────────────────────────────────
# Sowing date (shared by all minirhizotron calibration scripts)
# ─────────────────────────────────────────────────────────────────────────────
const _SOWING_DATE = Date(2025, 4, 1)

function _date_to_simday(d::Date)
    return Dates.value(d - _SOWING_DATE) + 1
end

# ─────────────────────────────────────────────────────────────────────────────
# Observation loaders
# ─────────────────────────────────────────────────────────────────────────────
function _load_obs_1tube(obs_csv::AbstractString)
    df = CSV.read(obs_csv, DataFrame)
    df.date = Date.(string.(df.date))
    sort!(df, [:date, :station_idx, :azimuth_deg])

    has_mean = hasproperty(df, :mean_crossing_angle_deg)
    has_hist = hasproperty(df, :hist_bin0)
    _to_f(v) = ismissing(v) ? NaN : Float64(v)

    obs_by_day            = Dict{Int, Vector{Float64}}()
    obs_angle_mean_by_day = Dict{Int, Vector{Float64}}()
    obs_angle_hist_by_day = Dict{Int, Matrix{Float64}}()
    meta_by_day           = Dict{Int, DataFrame}()

    for subdate in groupby(df, :date)
        sd  = _date_to_simday(subdate.date[1])
        sd >= 1 || continue
        obs_by_day[sd] = Float64.(subdate.root_length_cm)
        if has_mean
            obs_angle_mean_by_day[sd] = [_to_f(v) for v in subdate.mean_crossing_angle_deg]
        end
        if has_hist
            obs_angle_hist_by_day[sd] = hcat(
                [_to_f.(subdate[!, Symbol("hist_bin$(b-1)")]) for b in 1:N_ANGLE_BINS]...
            )
        end
        meta_by_day[sd] = DataFrame(
            station_idx   = Int.(subdate.station_idx),
            along_tube_cm = Float64.(subdate.along_tube_cm),
            depth_cm      = Float64.(subdate.depth_cm),
            azimuth_deg   = Float64.(subdate.azimuth_deg)
        )
    end
    used_days = sort(collect(keys(obs_by_day)))
    return used_days, obs_by_day, obs_angle_mean_by_day, obs_angle_hist_by_day, meta_by_day
end

function _load_obs_3tubes(obs_csv::AbstractString)
    df = CSV.read(obs_csv, DataFrame)
    df.date = Date.(string.(df.date))
    sort!(df, [:tube_id, :date, :station_idx, :azimuth_deg])

    has_mean = hasproperty(df, :mean_crossing_angle_deg)
    has_hist = hasproperty(df, :hist_bin0)
    _to_f(v) = ismissing(v) ? NaN : Float64(v)

    obs_by_tube_day            = Dict{Int, Dict{Int, Vector{Float64}}}()
    obs_angle_mean_by_tube_day = Dict{Int, Dict{Int, Vector{Float64}}}()
    obs_angle_hist_by_tube_day = Dict{Int, Dict{Int, Matrix{Float64}}}()
    meta_by_day                = Dict{Int, DataFrame}()
    all_days                   = Set{Int}()

    for sub_tube in groupby(df, :tube_id)
        tid = sub_tube.tube_id[1]
        obs_by_tube_day[tid]            = Dict{Int, Vector{Float64}}()
        obs_angle_mean_by_tube_day[tid] = Dict{Int, Vector{Float64}}()
        obs_angle_hist_by_tube_day[tid] = Dict{Int, Matrix{Float64}}()
        for sub_date in groupby(sub_tube, :date)
            sd = _date_to_simday(sub_date.date[1])
            sd >= 1 || continue
            obs_by_tube_day[tid][sd] = Float64.(sub_date.root_length_cm)
            if has_mean
                obs_angle_mean_by_tube_day[tid][sd] =
                    [_to_f(v) for v in sub_date.mean_crossing_angle_deg]
            end
            if has_hist
                obs_angle_hist_by_tube_day[tid][sd] = hcat(
                    [_to_f.(sub_date[!, Symbol("hist_bin$(b-1)")]) for b in 1:N_ANGLE_BINS]...
                )
            end
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
    used_days = sort(collect(all_days))
    return used_days, obs_by_tube_day, obs_angle_mean_by_tube_day,
           obs_angle_hist_by_tube_day, meta_by_day
end

# ─────────────────────────────────────────────────────────────────────────────
# Parameter JSON writer
# θ = [elong_min, elong_max, logistic_k, logistic_t50,
#      branch_angle_order1, g_axial, branching_prob_order1, branching_prob_order2]
# ─────────────────────────────────────────────────────────────────────────────
function _write_candidate_VD_json(template_path::AbstractString,
                                  out_path::AbstractString,
                                  θ::Vector{Float64};
                                  param_names::Vector{String}=PARAM_NAMES_ALL,
                                  n_axial_roots::Int,
                                  max_order1_per_plant::Int)
    d = _read_json(template_path)
    d["root_elongation_mode"] = "logistic"
    for (i, name) in enumerate(param_names)
        d[name] = θ[i]
    end
    d["n_axial_roots"]        = n_axial_roots
    d["max_order1_per_plant"] = max_order1_per_plant
    _write_json_pretty(out_path, d)
end

function _extract_theta(json_dict::Dict,
                        param_names::Vector{String}=PARAM_NAMES_ALL)::Vector{Float64}
    return [Float64(get(json_dict, p, PARAM_DEFAULTS[p])) for p in param_names]
end

# ─────────────────────────────────────────────────────────────────────────────
# Tube construction (same geometry as in existing scripts)
# ─────────────────────────────────────────────────────────────────────────────
function _make_tube(x_from_left_cm::Float64; W_cm::Float64=100.0, tube_origin_y_cm::Float64=20.0, skin_depth_cm::Float64=1.0)
    origin    = [x_from_left_cm - W_cm/2, tube_origin_y_cm, 0.0]
    direction = normalize_vec([0.0, 0.7071067811865476, -0.7071067811865476])
    return Minirhizotron(
        true, origin, direction, 80.0, 3.0, skin_depth_cm, 5.0, 5.0,
        collect(2.5:5.0:72.5), 0.1, "tube_stations_daily.csv",
        [0.0, 90.0, 180.0, 270.0]
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Simulation runners
# Both use manual day loop + in-memory tube sampling (minirhizotron=nothing).
# ─────────────────────────────────────────────────────────────────────────────

"""
Run one simulation for a single tube.
Returns Dict{Int, Vector{Float64}}: sim_day => station-azimuth vector (station-major).
"""
function _run_sim_1tube(env_json_path::AbstractString,
                        vd_template_path::AbstractString,
                        θ::Vector{Float64},
                        need_days::Vector{Int};
                        seed::Int,
                        n_axial_roots::Int,
                        max_order1_per_plant::Int,
                        param_names::Vector{String}=PARAM_NAMES_ALL,
                        tube_x_from_left_cm::Float64 = 50.0,
                        box_width_cm::Float64 = 100.0,
                        tube_origin_y_cm::Float64 = 20.0,
                        tmp_dir::AbstractString)
    mkpath(tmp_dir)
    runid  = string("s", seed, "_", abs(hash(θ)) % 10^9)
    outdir = joinpath(tmp_dir, "out_$(runid)")
    mkpath(outdir)

    vd_cand = joinpath(outdir, "VD.json")
    _write_candidate_VD_json(vd_template_path, vd_cand, θ;
        param_names=param_names,
        n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)

    env = _read_json(env_json_path)
    skin_depth_from_env = Float64(get(get(env, "minirhizotron", Dict()), "skin_depth_cm", 1.0))
    env["species_params_files"]["VD"]     = vd_cand
    env["simulation"]["seed"]             = seed
    env["simulation"]["days"]             = maximum(need_days)
    env["simulation"]["outdir"]           = outdir
    env["simulation"]["plot3d_outdir"]    = joinpath(outdir, "plot3d")
    env["facilitation"]["enabled"]        = false
    env["density_regulation"]["enabled"]  = false
    env["minirhizotron"]["enabled"]       = false

    env_path = joinpath(outdir, "env.json")
    _write_json_pretty(env_path, env)

    cfg_sim  = load_environment_and_plants(env_path)
    tube     = _make_tube(tube_x_from_left_cm; W_cm=box_width_cm, tube_origin_y_cm=tube_origin_y_cm, skin_depth_cm=skin_depth_from_env)
    ns       = length(tube.station_positions_cm)
    na       = length(tube.camera_azimuths_deg)
    need_set        = Set(need_days)
    lengths_res     = Dict{Int, Vector{Float64}}()
    angles_mean_res = Dict{Int, Vector{Float64}}()
    angles_hist_res = Dict{Int, Matrix{Float64}}()
    R_dummy         = zeros(1, 1)

    Random.seed!(seed)
    for day in 1:maximum(need_days)
        for pl in cfg_sim.plants
            step_plant!(pl, cfg_sim.box, cfg_sim.reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end
        if day in need_set
            len_mat  = tube_station_azimuth_lengths_cm(cfg_sim.plants, tube)
            mean_mat = tube_station_azimuth_angles_cm(cfg_sim.plants, tube)
            hist_arr = tube_station_azimuth_angle_histograms(cfg_sim.plants, tube)
            lengths_res[day]     = [len_mat[si, ai]  for si in 1:ns for ai in 1:na]
            angles_mean_res[day] = [mean_mat[si, ai] for si in 1:ns for ai in 1:na]
            angles_hist_res[day] = vcat([reshape(hist_arr[si, ai, :], 1, N_ANGLE_BINS)
                                         for si in 1:ns for ai in 1:na]...)
        end
    end
    return (lengths=lengths_res, angles_mean=angles_mean_res, angles_hist=angles_hist_res)
end

"""
Run one simulation for 3 tubes simultaneously.
Returns Dict{Int, Dict{Int, Vector{Float64}}}: tube_id => (day => vector).
"""
function _run_sim_3tubes(env_json_path::AbstractString,
                         vd_template_path::AbstractString,
                         θ::Vector{Float64},
                         need_days::Vector{Int};
                         seed::Int,
                         n_axial_roots::Int,
                         max_order1_per_plant::Int,
                         param_names::Vector{String}=PARAM_NAMES_ALL,
                         tube_x_cm_3tubes::Vector{Float64}=[25.0, 50.0, 75.0],
                         box_width_cm::Float64=100.0,
                         tube_origin_y_cm::Float64=20.0,
                         tmp_dir::AbstractString)
    mkpath(tmp_dir)
    runid  = string("s", seed, "_", abs(hash(θ)) % 10^9)
    outdir = joinpath(tmp_dir, "out3t_$(runid)")
    mkpath(outdir)

    vd_cand = joinpath(outdir, "VD.json")
    _write_candidate_VD_json(vd_template_path, vd_cand, θ;
        param_names=param_names,
        n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)

    env = _read_json(env_json_path)
    skin_depth_from_env = Float64(get(get(env, "minirhizotron", Dict()), "skin_depth_cm", 1.0))
    env["species_params_files"]["VD"]     = vd_cand
    env["simulation"]["seed"]             = seed
    env["simulation"]["days"]             = maximum(need_days)
    env["simulation"]["outdir"]           = outdir
    env["simulation"]["plot3d_outdir"]    = joinpath(outdir, "plot3d")
    env["facilitation"]["enabled"]        = false
    env["density_regulation"]["enabled"]  = false
    env["minirhizotron"]["enabled"]       = false

    env_path = joinpath(outdir, "env.json")
    _write_json_pretty(env_path, env)

    cfg_sim  = load_environment_and_plants(env_path)
    tubes    = [_make_tube(tx; W_cm=box_width_cm, tube_origin_y_cm=tube_origin_y_cm, skin_depth_cm=skin_depth_from_env) for tx in tube_x_cm_3tubes]
    ns       = length(tubes[1].station_positions_cm)
    na       = length(tubes[1].camera_azimuths_deg)
    need_set        = Set(need_days)
    n_tubes         = length(tubes)
    lengths_res     = Dict{Int, Dict{Int, Vector{Float64}}}(tid => Dict() for tid in 1:n_tubes)
    angles_mean_res = Dict{Int, Dict{Int, Vector{Float64}}}(tid => Dict() for tid in 1:n_tubes)
    angles_hist_res = Dict{Int, Dict{Int, Matrix{Float64}}}(tid => Dict() for tid in 1:n_tubes)
    R_dummy         = zeros(1, 1)

    Random.seed!(seed)
    for day in 1:maximum(need_days)
        for pl in cfg_sim.plants
            step_plant!(pl, cfg_sim.box, cfg_sim.reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end
        if day in need_set
            for (tid, tube) in enumerate(tubes)
                len_mat  = tube_station_azimuth_lengths_cm(cfg_sim.plants, tube)
                mean_mat = tube_station_azimuth_angles_cm(cfg_sim.plants, tube)
                hist_arr = tube_station_azimuth_angle_histograms(cfg_sim.plants, tube)
                lengths_res[tid][day]     = [len_mat[si, ai]  for si in 1:ns for ai in 1:na]
                angles_mean_res[tid][day] = [mean_mat[si, ai] for si in 1:ns for ai in 1:na]
                angles_hist_res[tid][day] = vcat([reshape(hist_arr[si, ai, :], 1, N_ANGLE_BINS)
                                                   for si in 1:ns for ai in 1:na]...)
            end
        end
    end
    return (lengths=lengths_res, angles_mean=angles_mean_res, angles_hist=angles_hist_res)
end

# ─────────────────────────────────────────────────────────────────────────────
# Parameter bounds check (same bounds as existing scripts)
# ─────────────────────────────────────────────────────────────────────────────
function _check_bounds(θ::Vector{Float64},
                       param_names::Vector{String}=PARAM_NAMES_ALL)::Bool
    for (i, name) in enumerate(param_names)
        lo, hi = PARAM_BOUNDS[name]
        (θ[i] < lo || θ[i] > hi) && return false
    end
    i_min = findfirst(==("elongation_rate_min"), param_names)
    i_max = findfirst(==("elongation_rate_max"), param_names)
    if i_min !== nothing && i_max !== nothing
        θ[i_max] < θ[i_min] && return false
    end
    return true
end

# ─────────────────────────────────────────────────────────────────────────────
# Loss closure factories
# ─────────────────────────────────────────────────────────────────────────────

"""
Build loss closure for a single tube.
For modes 7 & 8: averages predictions across replicate_seeds, then evaluates loss once.
For modes 1–6: evaluates loss per calibration_seed, then averages losses.
"""
function make_loss_flexible_1tube(used_days::Vector{Int},
                                   obs_by_day::Dict{Int,Vector{Float64}},
                                   meta_by_day::Dict{Int,DataFrame},
                                   env_json_path::AbstractString,
                                   vd_template_path::AbstractString;
                                   n_axial_roots::Int,
                                   max_order1_per_plant::Int,
                                   cfg::CalibrationObjectiveConfig,
                                   param_names::Vector{String}=PARAM_NAMES_ALL,
                                   tube_x_from_left_cm::Float64=50.0,
                                   box_width_cm::Float64=100.0,
                                   tube_origin_y_cm::Float64=20.0,
                                   obs_angle_mean_by_day::Dict{Int,Vector{Float64}}=Dict{Int,Vector{Float64}}(),
                                   obs_angle_hist_by_day::Dict{Int,Matrix{Float64}}=Dict{Int,Matrix{Float64}}(),
                                   tmp_dir::AbstractString)
    # Determine n_windows from first available day
    n_windows = isempty(used_days) ? 60 :
        length(get(obs_by_day, used_days[1], zeros(60)))

    # Pre-compute balanced indices (mode 3 only; empty dict for all other modes)
    balanced_idx = cfg.objective == "balanced_zero_sampling" ?
        prepare_balanced_indices(obs_by_day, used_days, cfg) :
        Dict{Int, Vector{Int}}()

    is_replicate_mode = cfg.objective in ("replicate_seed_average",
                                           "replicate_seed_multi_objective")

    function loss(θ::AbstractVector{<:Real})
        θ = Float64.(θ)
        _check_bounds(θ, param_names) || return 1e18

        w_m = cfg.w_angle_mean
        w_h = cfg.w_angle_hist

        if is_replicate_mode
            # Average predictions across replicate seeds (parallel)
            seeds_r   = cfg.replicate_seeds
            results_r = Vector{Any}(undef, length(seeds_r))
            Threads.@threads for i in eachindex(seeds_r)
                results_r[i] = _run_sim_1tube(env_json_path, vd_template_path, θ, used_days;
                    seed=seeds_r[i], n_axial_roots=n_axial_roots,
                    max_order1_per_plant=max_order1_per_plant,
                    param_names=param_names,
                    tube_x_from_left_cm=tube_x_from_left_cm, box_width_cm=box_width_cm,
                    tube_origin_y_cm=tube_origin_y_cm,
                    tmp_dir=tmp_dir)
            end
            pred_sum = Dict(d => zeros(n_windows) for d in used_days)
            am_sum   = Dict(d => zeros(n_windows) for d in used_days)
            am_cnt   = Dict(d => zeros(Int, n_windows) for d in used_days)
            ah_sum   = Dict(d => zeros(n_windows, N_ANGLE_BINS) for d in used_days)
            ah_cnt   = Dict(d => zeros(Int, n_windows) for d in used_days)
            n_valid  = 0
            for result in results_r
                for d in used_days
                    v = get(result.lengths, d, Float64[])
                    length(v) == n_windows || continue
                    pred_sum[d] .+= v
                    n_valid += 1
                    if w_m > 0
                        av = get(result.angles_mean, d, Float64[])
                        length(av) == n_windows || continue
                        for w in 1:n_windows
                            isnan(av[w]) && continue
                            am_sum[d][w] += av[w]; am_cnt[d][w] += 1
                        end
                    end
                    if w_h > 0
                        ah = get(result.angles_hist, d, Matrix{Float64}(undef,0,0))
                        size(ah) == (n_windows, N_ANGLE_BINS) || continue
                        for w in 1:n_windows
                            any(isnan, view(ah, w, :)) && continue
                            ah_sum[d][w, :] .+= view(ah, w, :); ah_cnt[d][w] += 1
                        end
                    end
                end
            end
            n_valid == 0 && return 1e9
            n_per_day = max(1, n_valid ÷ length(used_days))
            pred_avg  = Dict(d => clamp.(pred_sum[d] ./ n_per_day, 0.0, Inf) for d in used_days)
            len_loss  = compute_loss(obs_by_day, pred_avg, meta_by_day, used_days, cfg;
                                     balanced_idx=balanced_idx)
            mean_loss = w_m > 0 ? begin
                pred_am = Dict{Int,Vector{Float64}}(d => [am_cnt[d][w] > 0 ?
                    am_sum[d][w]/am_cnt[d][w] : NaN for w in 1:n_windows] for d in used_days)
                loss_angle_mean(obs_angle_mean_by_day, pred_am, used_days;
                                normalize_deg=cfg.angle_normalize_deg)
            end : 0.0
            hist_loss = w_h > 0 ? begin
                pred_ah = Dict{Int,Matrix{Float64}}(d => begin
                    mat = Matrix{Float64}(undef, n_windows, N_ANGLE_BINS)
                    for w in 1:n_windows
                        if ah_cnt[d][w] > 0
                            mat[w, :] = ah_sum[d][w, :] ./ ah_cnt[d][w]
                        else
                            mat[w, :] .= NaN
                        end
                    end
                    mat
                end for d in used_days)
                loss_angle_histogram(obs_angle_hist_by_day, pred_ah, used_days)
            end : 0.0
            return (1 - w_m - w_h) * len_loss + w_m * mean_loss + w_h * hist_loss
        else
            # Average loss across calibration seeds (parallel)
            seeds_c = cfg.calibration_seeds
            Js      = Vector{Float64}(undef, length(seeds_c))
            Threads.@threads for i in eachindex(seeds_c)
                result = _run_sim_1tube(env_json_path, vd_template_path, θ, used_days;
                    seed=seeds_c[i], n_axial_roots=n_axial_roots,
                    max_order1_per_plant=max_order1_per_plant,
                    param_names=param_names,
                    tube_x_from_left_cm=tube_x_from_left_cm, box_width_cm=box_width_cm,
                    tube_origin_y_cm=tube_origin_y_cm,
                    tmp_dir=tmp_dir)
                pred_by_day = Dict(d => clamp.(get(result.lengths, d, zeros(n_windows)), 0.0, Inf)
                                   for d in used_days)
                len_J = compute_loss(obs_by_day, pred_by_day, meta_by_day, used_days, cfg;
                                     balanced_idx=balanced_idx)
                m_J = w_m > 0 ? begin
                    pred_am = Dict{Int,Vector{Float64}}(d =>
                        get(result.angles_mean, d, fill(NaN, n_windows)) for d in used_days)
                    loss_angle_mean(obs_angle_mean_by_day, pred_am, used_days;
                                    normalize_deg=cfg.angle_normalize_deg)
                end : 0.0
                h_J = w_h > 0 ? begin
                    pred_ah = Dict{Int,Matrix{Float64}}(d =>
                        get(result.angles_hist, d, fill(NaN, n_windows, N_ANGLE_BINS))
                        for d in used_days)
                    loss_angle_histogram(obs_angle_hist_by_day, pred_ah, used_days)
                end : 0.0
                Js[i] = (1 - w_m - w_h) * len_J + w_m * m_J + w_h * h_J
            end
            return isempty(Js) ? 1e9 : mean(Js)
        end
    end

    return loss
end

"""
Build loss closure for 3 tubes.
If aggregate_tubes=false (default): loss = mean over seeds of mean over tubes of
  compute_loss(obs_tube_i, pred_tube_i).
If aggregate_tubes=true: observations and predictions are averaged element-wise across
  all 3 tube positions before computing loss. This reduces stochastic noise in both the
  calibration target and the model prediction, giving a much cleaner signal.
"""
function make_loss_flexible_3tubes(used_days::Vector{Int},
                                    obs_by_tube_day::Dict{Int,Dict{Int,Vector{Float64}}},
                                    meta_by_day::Dict{Int,DataFrame},
                                    env_json_path::AbstractString,
                                    vd_template_path::AbstractString;
                                    n_axial_roots::Int,
                                    max_order1_per_plant::Int,
                                    cfg::CalibrationObjectiveConfig,
                                    param_names::Vector{String}=PARAM_NAMES_ALL,
                                    tube_x_cm_3tubes::Vector{Float64}=[25.0,50.0,75.0],
                                    box_width_cm::Float64=100.0,
                                    tube_origin_y_cm::Float64=20.0,
                                    aggregate_tubes::Bool=false,
                                    obs_angle_mean_by_tube_day::Dict{Int,Dict{Int,Vector{Float64}}}=Dict{Int,Dict{Int,Vector{Float64}}}(),
                                    obs_angle_hist_by_tube_day::Dict{Int,Dict{Int,Matrix{Float64}}}=Dict{Int,Dict{Int,Matrix{Float64}}}(),
                                    tmp_dir::AbstractString)
    n_tubes   = length(tube_x_cm_3tubes)
    tube_ids  = sort(collect(keys(obs_by_tube_day)))
    first_tid = first(tube_ids)
    n_windows = isempty(used_days) ? 60 :
        length(get(get(obs_by_tube_day, first_tid, Dict()), used_days[1], zeros(60)))

    # Pre-compute tube-averaged observations (used when aggregate_tubes=true)
    obs_avg_by_day = Dict(d => begin
        vecs = [get(get(obs_by_tube_day, tid, Dict()), d, zeros(n_windows)) for tid in tube_ids]
        mean(vecs)
    end for d in used_days)

    # Balanced indices — per-tube (aggregate_tubes=false) or on averaged obs (true)
    balanced_idx_by_tube = Dict{Int, Dict{Int, Vector{Int}}}()
    balanced_idx_agg     = Dict{Int, Vector{Int}}()
    if cfg.objective == "balanced_zero_sampling"
        if aggregate_tubes
            balanced_idx_agg = prepare_balanced_indices(obs_avg_by_day, used_days, cfg)
        else
            for tid in tube_ids
                balanced_idx_by_tube[tid] =
                    prepare_balanced_indices(obs_by_tube_day[tid], used_days, cfg)
            end
        end
    end

    is_replicate_mode = cfg.objective in ("replicate_seed_average",
                                           "replicate_seed_multi_objective")

    # Pre-compute tube-averaged angle observations for aggregate mode (NaN-aware)
    obs_am_avg_by_day = Dict{Int, Vector{Float64}}(d => begin
        s = zeros(n_windows); c = zeros(Int, n_windows)
        for tid in tube_ids
            av = get(get(obs_angle_mean_by_tube_day, tid, Dict()), d, Float64[])
            length(av) == n_windows || continue
            for w in 1:n_windows; isnan(av[w]) && continue; s[w] += av[w]; c[w] += 1; end
        end
        [c[w] > 0 ? s[w]/c[w] : NaN for w in 1:n_windows]
    end for d in used_days)

    obs_ah_avg_by_day = Dict{Int, Matrix{Float64}}(d => begin
        s = zeros(n_windows, N_ANGLE_BINS); c = zeros(Int, n_windows)
        for tid in tube_ids
            ah = get(get(obs_angle_hist_by_tube_day, tid, Dict()), d, Matrix{Float64}(undef,0,0))
            size(ah) == (n_windows, N_ANGLE_BINS) || continue
            for w in 1:n_windows
                any(isnan, view(ah, w, :)) && continue
                s[w, :] .+= view(ah, w, :); c[w] += 1
            end
        end
        mat = Matrix{Float64}(undef, n_windows, N_ANGLE_BINS)
        for w in 1:n_windows
            if c[w] > 0; mat[w, :] = s[w, :] ./ c[w]
            else; mat[w, :] .= NaN; end
        end
        mat
    end for d in used_days)

    function loss(θ::AbstractVector{<:Real})
        θ = Float64.(θ)
        _check_bounds(θ, param_names) || return 1e18

        w_m = cfg.w_angle_mean
        w_h = cfg.w_angle_hist

        # Helper: build averaged angle dicts from per-result accumulators
        function _avg_am(am_s, am_c)
            Dict{Int,Vector{Float64}}(d => [am_c[d][w] > 0 ? am_s[d][w]/am_c[d][w] : NaN
                                            for w in 1:n_windows] for d in used_days)
        end
        function _avg_ah(ah_s, ah_c)
            Dict{Int,Matrix{Float64}}(d => begin
                mat = Matrix{Float64}(undef, n_windows, N_ANGLE_BINS)
                for w in 1:n_windows
                    if ah_c[d][w] > 0; mat[w, :] = ah_s[d][w, :] ./ ah_c[d][w]
                    else; mat[w, :] .= NaN; end
                end
                mat
            end for d in used_days)
        end

        if aggregate_tubes
            # ── Aggregated mode: average obs AND pred across tubes ────────────
            seeds = is_replicate_mode ? cfg.replicate_seeds : cfg.calibration_seeds
            if is_replicate_mode
                results_r = Vector{Any}(undef, length(seeds))
                Threads.@threads for i in eachindex(seeds)
                    results_r[i] = _run_sim_3tubes(env_json_path, vd_template_path, θ, used_days;
                        seed=seeds[i], n_axial_roots=n_axial_roots,
                        max_order1_per_plant=max_order1_per_plant,
                        param_names=param_names,
                        tube_x_cm_3tubes=tube_x_cm_3tubes, box_width_cm=box_width_cm,
                        tube_origin_y_cm=tube_origin_y_cm, tmp_dir=tmp_dir)
                end
                pred_sum = Dict(d => zeros(n_windows) for d in used_days)
                am_sum   = Dict(d => zeros(n_windows) for d in used_days)
                am_cnt   = Dict(d => zeros(Int, n_windows) for d in used_days)
                ah_sum   = Dict(d => zeros(n_windows, N_ANGLE_BINS) for d in used_days)
                ah_cnt   = Dict(d => zeros(Int, n_windows) for d in used_days)
                n_valid  = 0
                for result in results_r
                    for tid in 1:n_tubes, d in used_days
                        lv = get(get(result.lengths, tid, Dict()), d, Float64[])
                        length(lv) == n_windows || continue
                        pred_sum[d] .+= lv; n_valid += 1
                        if w_m > 0
                            av = get(get(result.angles_mean, tid, Dict()), d, Float64[])
                            length(av) == n_windows || continue
                            for w in 1:n_windows; isnan(av[w]) && continue
                                am_sum[d][w] += av[w]; am_cnt[d][w] += 1; end
                        end
                        if w_h > 0
                            ah = get(get(result.angles_hist, tid, Dict()), d, Matrix{Float64}(undef,0,0))
                            size(ah) == (n_windows, N_ANGLE_BINS) || continue
                            for w in 1:n_windows; any(isnan, view(ah, w, :)) && continue
                                ah_sum[d][w, :] .+= view(ah, w, :); ah_cnt[d][w] += 1; end
                        end
                    end
                end
                n_valid == 0 && return 1e9
                n_total  = max(1, n_valid ÷ length(used_days))
                pred_avg = Dict(d => clamp.(pred_sum[d] ./ n_total, 0.0, Inf) for d in used_days)
                len_loss = compute_loss(obs_avg_by_day, pred_avg, meta_by_day, used_days, cfg;
                                        balanced_idx=balanced_idx_agg)
                m_J = w_m > 0 ? loss_angle_mean(obs_am_avg_by_day, _avg_am(am_sum, am_cnt),
                                                  used_days; normalize_deg=cfg.angle_normalize_deg) : 0.0
                h_J = w_h > 0 ? loss_angle_histogram(obs_ah_avg_by_day,
                                                       _avg_ah(ah_sum, ah_cnt), used_days) : 0.0
                return (1 - w_m - w_h) * len_loss + w_m * m_J + w_h * h_J
            else
                Js = Vector{Float64}(undef, length(seeds))
                Threads.@threads for i in eachindex(seeds)
                    result = _run_sim_3tubes(env_json_path, vd_template_path, θ, used_days;
                        seed=seeds[i], n_axial_roots=n_axial_roots,
                        max_order1_per_plant=max_order1_per_plant,
                        param_names=param_names,
                        tube_x_cm_3tubes=tube_x_cm_3tubes, box_width_cm=box_width_cm,
                        tube_origin_y_cm=tube_origin_y_cm, tmp_dir=tmp_dir)
                    ps_i   = Dict(d => zeros(n_windows) for d in used_days)
                    am_s_i = Dict(d => zeros(n_windows) for d in used_days)
                    am_c_i = Dict(d => zeros(Int, n_windows) for d in used_days)
                    ah_s_i = Dict(d => zeros(n_windows, N_ANGLE_BINS) for d in used_days)
                    ah_c_i = Dict(d => zeros(Int, n_windows) for d in used_days)
                    nv_i   = 0
                    for tid in 1:n_tubes, d in used_days
                        lv = get(get(result.lengths, tid, Dict()), d, Float64[])
                        length(lv) == n_windows || continue
                        ps_i[d] .+= lv; nv_i += 1
                        if w_m > 0
                            av = get(get(result.angles_mean, tid, Dict()), d, Float64[])
                            length(av) == n_windows || continue
                            for w in 1:n_windows; isnan(av[w]) && continue
                                am_s_i[d][w] += av[w]; am_c_i[d][w] += 1; end
                        end
                        if w_h > 0
                            ah = get(get(result.angles_hist, tid, Dict()), d, Matrix{Float64}(undef,0,0))
                            size(ah) == (n_windows, N_ANGLE_BINS) || continue
                            for w in 1:n_windows; any(isnan, view(ah, w, :)) && continue
                                ah_s_i[d][w, :] .+= view(ah, w, :); ah_c_i[d][w] += 1; end
                        end
                    end
                    nt_i     = max(1, nv_i ÷ length(used_days))
                    pred_avg = Dict(d => clamp.(ps_i[d] ./ nt_i, 0.0, Inf) for d in used_days)
                    lJ = compute_loss(obs_avg_by_day, pred_avg, meta_by_day, used_days, cfg;
                                      balanced_idx=balanced_idx_agg)
                    mJ = w_m > 0 ? loss_angle_mean(obs_am_avg_by_day, _avg_am(am_s_i, am_c_i),
                                                    used_days; normalize_deg=cfg.angle_normalize_deg) : 0.0
                    hJ = w_h > 0 ? loss_angle_histogram(obs_ah_avg_by_day,
                                                         _avg_ah(ah_s_i, ah_c_i), used_days) : 0.0
                    Js[i] = (1 - w_m - w_h) * lJ + w_m * mJ + w_h * hJ
                end
                return isempty(Js) ? 1e9 : mean(Js)
            end

        else
            # ── Per-tube mode: loss averaged across seeds × tubes ─────────────
            if is_replicate_mode
                seeds_r   = cfg.replicate_seeds
                results_r = Vector{Any}(undef, length(seeds_r))
                Threads.@threads for i in eachindex(seeds_r)
                    results_r[i] = _run_sim_3tubes(env_json_path, vd_template_path, θ, used_days;
                        seed=seeds_r[i], n_axial_roots=n_axial_roots,
                        max_order1_per_plant=max_order1_per_plant,
                        param_names=param_names,
                        tube_x_cm_3tubes=tube_x_cm_3tubes, box_width_cm=box_width_cm,
                        tube_origin_y_cm=tube_origin_y_cm, tmp_dir=tmp_dir)
                end
                pred_sum = Dict(tid => Dict(d => zeros(n_windows) for d in used_days) for tid in 1:n_tubes)
                am_sum_t = Dict(tid => Dict(d => zeros(n_windows) for d in used_days) for tid in 1:n_tubes)
                am_cnt_t = Dict(tid => Dict(d => zeros(Int, n_windows) for d in used_days) for tid in 1:n_tubes)
                ah_sum_t = Dict(tid => Dict(d => zeros(n_windows, N_ANGLE_BINS) for d in used_days) for tid in 1:n_tubes)
                ah_cnt_t = Dict(tid => Dict(d => zeros(Int, n_windows) for d in used_days) for tid in 1:n_tubes)
                n_valid = 0
                for result in results_r
                    for tid in 1:n_tubes
                        for d in used_days
                            lv = get(get(result.lengths, tid, Dict()), d, Float64[])
                            length(lv) == n_windows || continue
                            pred_sum[tid][d] .+= lv; n_valid += 1
                            if w_m > 0
                                av = get(get(result.angles_mean, tid, Dict()), d, Float64[])
                                length(av) == n_windows || continue
                                for w in 1:n_windows; isnan(av[w]) && continue
                                    am_sum_t[tid][d][w] += av[w]; am_cnt_t[tid][d][w] += 1; end
                            end
                            if w_h > 0
                                ah = get(get(result.angles_hist, tid, Dict()), d, Matrix{Float64}(undef,0,0))
                                size(ah) == (n_windows, N_ANGLE_BINS) || continue
                                for w in 1:n_windows; any(isnan, view(ah, w, :)) && continue
                                    ah_sum_t[tid][d][w, :] .+= view(ah, w, :); ah_cnt_t[tid][d][w] += 1; end
                            end
                        end
                    end
                end
                n_valid == 0 && return 1e9
                n_per_tube_day = max(1, n_valid ÷ (n_tubes * length(used_days)))
                Js_tube = Float64[]
                for tid in tube_ids
                    obs_day  = get(obs_by_tube_day, tid, Dict{Int,Vector{Float64}}())
                    pred_avg = Dict(d => clamp.(pred_sum[tid][d] ./ n_per_tube_day, 0.0, Inf) for d in used_days)
                    bidx = get(balanced_idx_by_tube, tid, Dict{Int,Vector{Int}}())
                    lJ = compute_loss(obs_day, pred_avg, meta_by_day, used_days, cfg; balanced_idx=bidx)
                    pred_am_t = Dict{Int,Vector{Float64}}(d => [am_cnt_t[tid][d][w] > 0 ?
                        am_sum_t[tid][d][w]/am_cnt_t[tid][d][w] : NaN for w in 1:n_windows] for d in used_days)
                    pred_ah_t = Dict{Int,Matrix{Float64}}(d => begin
                        mat = Matrix{Float64}(undef, n_windows, N_ANGLE_BINS)
                        for w in 1:n_windows
                            if ah_cnt_t[tid][d][w] > 0; mat[w, :] = ah_sum_t[tid][d][w, :] ./ ah_cnt_t[tid][d][w]
                            else; mat[w, :] .= NaN; end
                        end; mat
                    end for d in used_days)
                    obs_am_t = get(obs_angle_mean_by_tube_day, tid, Dict{Int,Vector{Float64}}())
                    obs_ah_t = get(obs_angle_hist_by_tube_day, tid, Dict{Int,Matrix{Float64}}())
                    mJ = w_m > 0 ? loss_angle_mean(obs_am_t, pred_am_t, used_days;
                                                    normalize_deg=cfg.angle_normalize_deg) : 0.0
                    hJ = w_h > 0 ? loss_angle_histogram(obs_ah_t, pred_ah_t, used_days) : 0.0
                    push!(Js_tube, (1 - w_m - w_h) * lJ + w_m * mJ + w_h * hJ)
                end
                return isempty(Js_tube) ? 1e9 : mean(Js_tube)
            else
                seeds_c = cfg.calibration_seeds
                Js_seed = Vector{Float64}(undef, length(seeds_c))
                Threads.@threads for i in eachindex(seeds_c)
                    result = _run_sim_3tubes(env_json_path, vd_template_path, θ, used_days;
                        seed=seeds_c[i], n_axial_roots=n_axial_roots,
                        max_order1_per_plant=max_order1_per_plant,
                        param_names=param_names,
                        tube_x_cm_3tubes=tube_x_cm_3tubes, box_width_cm=box_width_cm,
                        tube_origin_y_cm=tube_origin_y_cm, tmp_dir=tmp_dir)
                    Js_tube = Float64[]
                    for tid in tube_ids
                        obs_day  = get(obs_by_tube_day, tid, Dict{Int,Vector{Float64}}())
                        pred_day = Dict(d => clamp.(get(get(result.lengths, tid, Dict()), d,
                                                        zeros(n_windows)), 0.0, Inf) for d in used_days)
                        bidx = get(balanced_idx_by_tube, tid, Dict{Int,Vector{Int}}())
                        lJ = compute_loss(obs_day, pred_day, meta_by_day, used_days, cfg; balanced_idx=bidx)
                        obs_am_t = get(obs_angle_mean_by_tube_day, tid, Dict{Int,Vector{Float64}}())
                        obs_ah_t = get(obs_angle_hist_by_tube_day, tid, Dict{Int,Matrix{Float64}}())
                        pred_am_t = Dict{Int,Vector{Float64}}(d => get(get(result.angles_mean, tid, Dict()),
                                                                        d, fill(NaN, n_windows)) for d in used_days)
                        pred_ah_t = Dict{Int,Matrix{Float64}}(d => get(get(result.angles_hist, tid, Dict()),
                                                                        d, fill(NaN, n_windows, N_ANGLE_BINS)) for d in used_days)
                        mJ = w_m > 0 ? loss_angle_mean(obs_am_t, pred_am_t, used_days;
                                                        normalize_deg=cfg.angle_normalize_deg) : 0.0
                        hJ = w_h > 0 ? loss_angle_histogram(obs_ah_t, pred_ah_t, used_days) : 0.0
                        push!(Js_tube, (1 - w_m - w_h) * lJ + w_m * mJ + w_h * hJ)
                    end
                    Js_seed[i] = isempty(Js_tube) ? 1e9 : mean(Js_tube)
                end
                return isempty(Js_seed) ? 1e9 : mean(Js_seed)
            end
        end
    end

    return loss
end

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
function _regression_metrics(x::Vector{Float64}, y::Vector{Float64})
    n    = length(x)
    rmse = sqrt(mean((y .- x).^2))
    x̄    = mean(x); ȳ = mean(y)
    sst  = sum((x .- x̄).^2)
    r2   = sst == 0 ? NaN : 1.0 - sum((y .- x).^2) / sst
    vx   = sum((x .- x̄).^2) / max(1, n-1)
    vy   = sum((y .- ȳ).^2) / max(1, n-1)
    cxy  = sum((x .- x̄) .* (y .- ȳ)) / max(1, n-1)
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

# ─────────────────────────────────────────────────────────────────────────────
# Long-format obs-pred builder (1-tube)
# ─────────────────────────────────────────────────────────────────────────────
function _build_obs_pred_long(used_days::Vector{Int},
                               obs_by_day::Dict{Int,Vector{Float64}},
                               meta_by_day::Dict{Int,DataFrame},
                               sim_by_day::Dict{Int,Vector{Float64}})
    rows = DataFrame(day=Int[], station_idx=Int[], along_tube_cm=Float64[],
                     depth_cm=Float64[], azimuth_deg=Float64[],
                     obs_cm=Float64[], pred_cm=Float64[])
    for d in used_days
        obs  = get(obs_by_day,  d, Float64[])
        pred = get(sim_by_day,  d, Float64[])
        meta = get(meta_by_day, d, DataFrame())
        (isempty(obs) || isempty(pred) || length(obs) != length(pred)) && continue
        for i in 1:length(obs)
            push!(rows, (d,
                Int(meta.station_idx[i]), Float64(meta.along_tube_cm[i]),
                Float64(meta.depth_cm[i]), Float64(meta.azimuth_deg[i]),
                obs[i], pred[i]))
        end
    end
    return rows
end

# ─────────────────────────────────────────────────────────────────────────────
# Scatter PNG (non-zero windows; annotations)
# ─────────────────────────────────────────────────────────────────────────────
function _save_scatter_png(df_long::DataFrame, outpath::AbstractString;
                            title::AbstractString="Obs vs Pred")
    Plots.gr()
    dfp  = df_long[(df_long.obs_cm .> 0.0) .| (df_long.pred_cm .> 0.0), :]
    isempty(dfp) && return
    x   = Float64.(dfp.obs_cm); y = Float64.(dfp.pred_cm)
    m   = _regression_metrics(x, y)
    lo  = min(minimum(x), minimum(y)); hi = max(maximum(x), maximum(y))
    rng = max(hi - lo, 1e-6)
    x̄   = mean(x); ȳ = mean(y)
    sxx = sum((x .- x̄).^2); sxy = sum((x .- x̄) .* (y .- ȳ))
    b   = sxx == 0 ? NaN : sxy / sxx; a = ȳ - b * x̄

    p = Plots.scatter(x, y; ms=2, label=false, alpha=0.6,
        xlabel="Observed (cm)", ylabel="Predicted (cm)", title=title,
        legend=:bottomright, legend_background_color=:transparent)
    Plots.plot!(p, [lo, hi], [lo, hi]; label="1:1", color=:black)
    if !isnan(a) && !isnan(b)
        Plots.plot!(p, [lo, hi], [a+b*lo, a+b*hi]; label="OLS fit", linestyle=:dash)
    end
    eq = isnan(a) ? "" : @sprintf("OLS: y = %.3f + %.3f x", a, b)
    st = @sprintf("R² = %.3f\nRMSE = %.3f cm\nCCC = %.3f\nKGE = %.3f\nPBIAS = %.1f%%",
                  m.r2, m.rmse, m.ccc, m.kge, m.pbias)
    x0 = lo + 0.02*rng; ytop = hi - 0.04*rng
    Plots.annotate!(p, (x0, ytop,          Plots.text(eq, 9, :left)))
    Plots.annotate!(p, (x0, ytop-0.18*rng, Plots.text(st, 9, :left)))
    Plots.savefig(p, outpath)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main calibration runner (called from entry point or experiment runner)
# ─────────────────────────────────────────────────────────────────────────────
function run_flexible_calibration(config_path::AbstractString)
    config = _read_json(config_path)
    cfg    = load_objective_config(config)

    tube_setup  = String(get(config, "tube_setup", "1tube"))
    maxiter     = Int(get(config, "maxiter", 100))
    sigma0      = Float64(get(config, "sigma0", 0.1))
    grid_n_ax   = Int.(get(config, "grid_n_axial",    [1]))
    grid_mo1    = Int.(get(config, "grid_max_order1", [20]))
    output_base = String(get(config, "output_dir",
                             joinpath(@__DIR__, "experiments")))
    warm_start  = get(config, "warm_start_json", nothing)
    warm_start  = warm_start === nothing ? nothing : String(warm_start)
    calib_params_raw = get(config, "calibration_params", nothing)
    calib_params = calib_params_raw === nothing ? PARAM_NAMES_ALL : String.(calib_params_raw)

    # env_json and geometry can be overridden in the config
    _env_json_raw = get(config, "env_json", "Parameter_calib_minirhizotron.json")
    env_json = isabspath(String(_env_json_raw)) ? String(_env_json_raw) :
               joinpath(@__DIR__, String(_env_json_raw))
    env_raw = _read_json(env_json)
    tube_origin_y_cm = Float64(get(get(env_raw, "minirhizotron", Dict()),
                                   "tube_origin_cm", [0.0, 20.0, 0.0])[2])
    box_width_cm      = Float64(get(config, "box_width_cm", 100.0))
    tube_x_1tube_cm   = Float64(get(config, "tube_x_1tube_cm", 50.0))
    tube_x_3tubes_raw = get(config, "tube_x_positions_cm", [25.0, 50.0, 75.0])
    tube_x_3tubes     = Float64.(tube_x_3tubes_raw)
    aggregate_tubes   = Bool(get(config, "aggregate_tubes", false))

    # Observation CSV paths can also be overridden
    _obs_1tube_raw  = get(config, "obs_csv_1tube",  "data/mr_obs_stations_1tube.csv")
    _obs_3tube_raw  = get(config, "obs_csv_3tubes", "data/mr_obs_stations_3tubes.csv")
    obs_csv_1tube   = isabspath(String(_obs_1tube_raw)) ? String(_obs_1tube_raw) :
                      joinpath(@__DIR__, String(_obs_1tube_raw))
    obs_csv_3tubes  = isabspath(String(_obs_3tube_raw)) ? String(_obs_3tube_raw) :
                      joinpath(@__DIR__, String(_obs_3tube_raw))

    vd_template = joinpath(@__DIR__, "Parameter_VD_template.json")

    # Experiment output directory — use experiment_tag if provided, else timestamp
    _exp_tag = get(config, "experiment_tag", nothing)
    ts       = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    exp_name = _exp_tag !== nothing ? String(_exp_tag) : string(cfg.objective, "_", ts)
    exp_dir  = joinpath(output_base, exp_name)
    # Each run gets its own tmp_dir — avoids race conditions when parallel runs
    # start from the same warm-start θ₀ and generate identical hash-based subdirs.
    tmp_dir  = joinpath(@__DIR__, "__calib_tmp_mr_flexible__", exp_name)
    mkpath(exp_dir)

    @info "Flexible calibration" objective=cfg.objective tube_setup=tube_setup aggregate_tubes=aggregate_tubes exp_dir=exp_dir

    # Load observations
    obs_csv = tube_setup == "3tubes" ? obs_csv_3tubes : obs_csv_1tube
    isfile(obs_csv) || error("Observation CSV not found: $obs_csv")

    obs_df = CSV.read(obs_csv, DataFrame)
    obs_df.date = Date.(string.(obs_df.date))

    println("\n─── Zero-inflation diagnostics ─────────────────────────────")
    save_zero_inflation_report(obs_df, exp_dir)

    if tube_setup == "3tubes"
        used_days, obs_by_tube_day, obs_angle_mean_by_tube_day,
            obs_angle_hist_by_tube_day, meta_by_day = _load_obs_3tubes(obs_csv)
        @info "Loaded 3-tube observations" days=used_days n_tubes=length(tube_x_3tubes) tube_x_cm=tube_x_3tubes
    else
        used_days, obs_by_day, obs_angle_mean_by_day,
            obs_angle_hist_by_day, meta_by_day = _load_obs_1tube(obs_csv)
        @info "Loaded 1-tube observations" days=used_days n_per_day=length(obs_by_day[used_days[1]])
    end

    # Warm start — resolve relative paths against the script directory
    one_tube_best = joinpath(@__DIR__, "ParameterVD_best_mr_1tube.json")
    warm_start_resolved = if warm_start === nothing
        nothing
    elseif isabspath(warm_start)
        warm_start
    else
        joinpath(@__DIR__, warm_start)
    end
    src = if warm_start_resolved !== nothing && isfile(warm_start_resolved)
        warm_start_resolved
    elseif isfile(one_tube_best)
        one_tube_best
    else
        vd_template
    end
    θ0 = _extract_theta(_read_json(src), calib_params)
    @info "θ0 source" path=src

    t_start  = time()
    best_J   = Inf
    best_θ   = nothing
    best_int = nothing   # (n_axial_roots, max_order1_per_plant)

    for n_ax in grid_n_ax, m_o1 in grid_mo1
        @info "Grid point" n_axial_roots=n_ax max_order1_per_plant=m_o1

        loss_fn = if tube_setup == "3tubes"
            make_loss_flexible_3tubes(used_days, obs_by_tube_day, meta_by_day,
                env_json, vd_template;
                n_axial_roots=n_ax, max_order1_per_plant=m_o1,
                cfg=cfg, param_names=calib_params,
                tube_x_cm_3tubes=tube_x_3tubes, box_width_cm=box_width_cm,
                tube_origin_y_cm=tube_origin_y_cm,
                aggregate_tubes=aggregate_tubes,
                obs_angle_mean_by_tube_day=obs_angle_mean_by_tube_day,
                obs_angle_hist_by_tube_day=obs_angle_hist_by_tube_day,
                tmp_dir=tmp_dir)
        else
            make_loss_flexible_1tube(used_days, obs_by_day, meta_by_day,
                env_json, vd_template;
                n_axial_roots=n_ax, max_order1_per_plant=m_o1,
                cfg=cfg, param_names=calib_params,
                tube_x_from_left_cm=tube_x_1tube_cm, box_width_cm=box_width_cm,
                tube_origin_y_cm=tube_origin_y_cm,
                obs_angle_mean_by_day=obs_angle_mean_by_day,
                obs_angle_hist_by_day=obs_angle_hist_by_day,
                tmp_dir=tmp_dir)
        end

        @info "Starting CMA-ES" maxiter=maxiter sigma0=sigma0
        result = CMAEvolutionStrategy.minimize(loss_fn, θ0, sigma0;
                                               maxiter=maxiter, verbosity=1)
        θb = CMAEvolutionStrategy.xbest(result)
        Jb = CMAEvolutionStrategy.fbest(result)
        @info "Grid result" n_ax=n_ax m_o1=m_o1 J=Jb

        if Jb < best_J
            best_J   = Jb
            best_θ   = θb
            best_int = (n_axial_roots=n_ax, max_order1_per_plant=m_o1)
        end
    end

    runtime_s = time() - t_start
    @info "Optimization complete" best_J=best_J runtime_min=runtime_s/60

    # ── Save best parameters ─────────────────────────────────────────────────
    # Use warm_start_json as the base if provided (Scenario 2: contains true
    # elongation values that were fixed during calibration). Fall back to the
    # template for Scenario 1 where no warm_start is given.
    best_path = joinpath(exp_dir, "best_params.json")
    _write_candidate_VD_json(warm_start !== nothing ? warm_start : vd_template,
        best_path, best_θ;
        param_names=calib_params,
        n_axial_roots=best_int.n_axial_roots,
        max_order1_per_plant=best_int.max_order1_per_plant)
    d_bp = _read_json(best_path)
    d_bp["calibration_J"]     = best_J
    d_bp["calibration_seeds"] = collect(cfg.calibration_seeds)
    d_bp["objective"]         = cfg.objective
    d_bp["tube_setup"]        = tube_setup
    _write_json_pretty(best_path, d_bp)

    # ── Diagnostics run (seed = 42) ──────────────────────────────────────────
    println("\n─── Diagnostics (seed=42) ──────────────────────────────────")
    if tube_setup == "3tubes"
        diag_result = _run_sim_3tubes(env_json, vd_template, best_θ, used_days;
            seed=42, n_axial_roots=best_int.n_axial_roots,
            max_order1_per_plant=best_int.max_order1_per_plant,
            param_names=calib_params,
            tube_x_cm_3tubes=tube_x_3tubes, box_width_cm=box_width_cm,
            tube_origin_y_cm=tube_origin_y_cm,
            tmp_dir=tmp_dir)
        sim_diag      = diag_result.lengths
        diag_tube_ids = sort(collect(keys(obs_by_tube_day)))
        diag_n_win    = length(get(get(obs_by_tube_day, diag_tube_ids[1], Dict()),
                                   used_days[1], zeros(60)))
        if aggregate_tubes
            pred_avg_diag = Dict(d => clamp.(mean([get(get(sim_diag, tid, Dict()), d,
                                                       zeros(diag_n_win))
                                                   for tid in diag_tube_ids]), 0.0, Inf)
                                 for d in used_days)
            obs_avg_diag  = Dict(d => mean([get(get(obs_by_tube_day, tid, Dict()), d,
                                                zeros(diag_n_win))
                                            for tid in diag_tube_ids])
                                 for d in used_days)
            df_long = _build_obs_pred_long(used_days, obs_avg_diag, meta_by_day, pred_avg_diag)
        else
            centre_tid = argmin(abs.(tube_x_3tubes .- box_width_cm/2))
            sim_t2 = get(sim_diag, centre_tid, Dict{Int,Vector{Float64}}())
            df_long = _build_obs_pred_long(used_days,
                get(obs_by_tube_day, centre_tid, Dict{Int,Vector{Float64}}()),
                meta_by_day, sim_t2)
        end
    else
        diag_result = _run_sim_1tube(env_json, vd_template, best_θ, used_days;
            seed=42, n_axial_roots=best_int.n_axial_roots,
            max_order1_per_plant=best_int.max_order1_per_plant,
            param_names=calib_params,
            tube_x_from_left_cm=tube_x_1tube_cm, box_width_cm=box_width_cm,
            tube_origin_y_cm=tube_origin_y_cm,
            tmp_dir=tmp_dir)
        df_long = _build_obs_pred_long(used_days, obs_by_day, meta_by_day,
                                       diag_result.lengths)
    end

    CSV.write(joinpath(exp_dir, "best_obs_pred_stations.csv"), df_long)
    _save_scatter_png(df_long, joinpath(exp_dir, "obs_vs_pred_all.png");
        title="$(cfg.objective) $(tube_setup): Obs vs Pred")
    save_calibration_diagnostics(df_long, cfg, exp_dir)

    # ── Summary metrics ──────────────────────────────────────────────────────
    x_all = Float64.(df_long.obs_cm); y_all = Float64.(df_long.pred_cm)
    nz    = (x_all .> 0.0) .| (y_all .> 0.0)
    xnz = x_all[nz]; ynz = y_all[nz]
    m_all = isempty(xnz) ? nothing : _regression_metrics(xnz, ynz)

    zi_frac = count(x_all .== 0.0) / max(1, length(x_all))

    # ── Run summary JSON ─────────────────────────────────────────────────────
    run_summary = Dict{String, Any}(
        "objective"          => cfg.objective,
        "tube_setup"         => tube_setup,
        "best_J"             => best_J,
        "runtime_seconds"    => runtime_s,
        "n_axial_roots"      => best_int.n_axial_roots,
        "max_order1_per_plant" => best_int.max_order1_per_plant,
        "calibration_seeds"  => collect(cfg.calibration_seeds),
        "sigma0"             => sigma0,
        "maxiter"            => maxiter,
        "config_file"        => config_path,
        "obs_zero_fraction"  => zi_frac,
        "diag_rmse"          => m_all === nothing ? NaN : m_all.rmse,
        "diag_r2"            => m_all === nothing ? NaN : m_all.r2,
        "diag_pbias"         => m_all === nothing ? NaN : m_all.pbias,
        "best_params_path"   => best_path,
        "calibration_params" => calib_params
    )
    _write_json_pretty(joinpath(exp_dir, "run_summary.json"), run_summary)

    println(repeat('─', 62))
    @printf("  Objective:     %s\n", cfg.objective)
    @printf("  Tube setup:    %s\n", tube_setup)
    @printf("  Best J:        %.4f\n", best_J)
    if m_all !== nothing
        @printf("  Diag RMSE:     %.3f cm\n", m_all.rmse)
        @printf("  Diag R²:       %.3f\n",    m_all.r2)
        @printf("  Diag PBIAS:    %.1f%%\n",  m_all.pbias)
    end
    @printf("  Runtime:       %.1f min\n", runtime_s / 60)
    @printf("  Output dir:    %s\n", exp_dir)
    println(repeat('─', 62))

    @printf("\nBest θ:\n")
    for (i, pname) in enumerate(calib_params)
        @printf("  %-30s= %.4f\n", pname, best_θ[i])
    end
    @printf("  %-30s= %d\n",   "n_axial_roots", best_int.n_axial_roots)
    @printf("  %-30s= %d\n",   "max_order1_per_plant", best_int.max_order1_per_plant)

    return exp_dir, best_J, best_θ, best_int
end

# ─────────────────────────────────────────────────────────────────────────────
# Entry point (skipped when this file is include()d by the experiment runner)
# ─────────────────────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    isempty(ARGS) && error(
        "Usage: julia calibration/main_Optimization_mr_flexible.jl <config.json>")
    run_flexible_calibration(ARGS[1])
end
