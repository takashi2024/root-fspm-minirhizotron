# analyse_seed_convergence.jl
#
# Evaluates how many seed replicates are needed for stable R² in the diagnostic.
# Given best_params (or the template VD), runs N simulations with seeds 1..N,
# accumulates the running average prediction, and records R²(k) after each
# additional seed k.
#
# The result informs how many calibration_seeds should be used in future runs.
# IMPORTANT: run this BEFORE the full calibration to set calibration_seeds
# correctly — use --template mode with the VD template if no calibration
# exists yet.
#
# Usage — two modes:
#
#   Mode A: post-calibration (verify that enough seeds were used)
#     julia calibration/analyse_seed_convergence.jl <experiment_dir> [N_seeds]
#
#   Mode B: pre-calibration (determine K before running calibration)
#     julia calibration/analyse_seed_convergence.jl \
#       --template <vd_json> --config <calib_config_json> [N_seeds]
#
# Defaults: N_seeds = 30
#
# Outputs (written to <out_dir>/seed_convergence/):
#   convergence_metrics.csv      — k, r2, rmse, ccc, kge, pbias for each k
#   convergence_r2.png           — R²(k) and RMSE(k) vs seed count
#   convergence_r2_stability.png — rolling σ(R²) vs k (stability indicator)

using JSON
using JSON3
using CSV
using DataFrames
using Statistics
using Dates
using Random
using Printf

import Plots

# ─────────────────────────────────────────────────────────────────────────────
# Source files
# ─────────────────────────────────────────────────────────────────────────────
const _CALIB_DIR = @__DIR__
include(joinpath(_CALIB_DIR, "..", "src", "Deps.jl"))
include(joinpath(_CALIB_DIR, "..", "src", "Types.jl"))
include(joinpath(_CALIB_DIR, "..", "src", "Geometry.jl"))
include(joinpath(_CALIB_DIR, "..", "src", "DensityRegulation.jl"))
include(joinpath(_CALIB_DIR, "..", "src", "GrowthStep.jl"))
include(joinpath(_CALIB_DIR, "..", "src", "MetricsAndOutput.jl"))
include(joinpath(_CALIB_DIR, "..", "src", "Plotting.jl"))
include(joinpath(_CALIB_DIR, "..", "src", "Config.jl"))
include(joinpath(_CALIB_DIR, "..", "src", "Simulation.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
_read_json(path::AbstractString) =
    open(path, "r") do io; JSON.parse(String(read(io))); end

function _write_json_pretty(path::AbstractString, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

# ─────────────────────────────────────────────────────────────────────────────
# Sowing date
# ─────────────────────────────────────────────────────────────────────────────
const _SOWING_DATE = Date(2025, 4, 1)
_date_to_simday(d::Date) = Dates.value(d - _SOWING_DATE) + 1

# ─────────────────────────────────────────────────────────────────────────────
# Observation loaders
# ─────────────────────────────────────────────────────────────────────────────
function _load_obs_1tube(obs_csv::AbstractString)
    df = CSV.read(obs_csv, DataFrame)
    df.date = Date.(string.(df.date))
    sort!(df, [:date, :station_idx, :azimuth_deg])
    obs_by_day  = Dict{Int, Vector{Float64}}()
    meta_by_day = Dict{Int, DataFrame}()
    for subdate in groupby(df, :date)
        sd = _date_to_simday(subdate.date[1])
        sd >= 1 || continue
        obs_by_day[sd] = Float64.(subdate.root_length_cm)
        meta_by_day[sd] = DataFrame(
            station_idx   = Int.(subdate.station_idx),
            along_tube_cm = Float64.(subdate.along_tube_cm),
            depth_cm      = Float64.(subdate.depth_cm),
            azimuth_deg   = Float64.(subdate.azimuth_deg))
    end
    return sort(collect(keys(obs_by_day))), obs_by_day, meta_by_day
end

function _load_obs_3tubes(obs_csv::AbstractString)
    df = CSV.read(obs_csv, DataFrame)
    df.date = Date.(string.(df.date))
    sort!(df, [:tube_id, :date, :station_idx, :azimuth_deg])
    obs_by_tube_day = Dict{Int, Dict{Int, Vector{Float64}}}()
    meta_by_day     = Dict{Int, DataFrame}()
    all_days        = Set{Int}()
    for subg in groupby(df, [:tube_id, :date])
        tid = Int(subg.tube_id[1])
        sd  = _date_to_simday(subg.date[1])
        sd >= 1 || continue
        push!(all_days, sd)
        haskey(obs_by_tube_day, tid) || (obs_by_tube_day[tid] = Dict{Int, Vector{Float64}}())
        obs_by_tube_day[tid][sd] = Float64.(subg.root_length_cm)
        if !haskey(meta_by_day, sd)
            meta_by_day[sd] = DataFrame(
                station_idx   = Int.(subg.station_idx),
                along_tube_cm = Float64.(subg.along_tube_cm),
                depth_cm      = Float64.(subg.depth_cm),
                azimuth_deg   = Float64.(subg.azimuth_deg))
        end
    end
    return sort(collect(all_days)), obs_by_tube_day, meta_by_day
end

# ─────────────────────────────────────────────────────────────────────────────
# Tube construction
# ─────────────────────────────────────────────────────────────────────────────
function _make_tube(x_from_left_cm::Float64; W_cm::Float64=100.0, tube_origin_y_cm::Float64=20.0)
    origin    = [x_from_left_cm - W_cm/2, tube_origin_y_cm, 0.0]
    direction = normalize_vec([0.0, 0.7071067811865476, -0.7071067811865476])
    return Minirhizotron(
        true, origin, direction, 80.0, 3.0, 1.0, 5.0, 5.0,
        collect(2.5:5.0:72.5), 0.1, "tube_stations_daily.csv",
        [0.0, 90.0, 180.0, 270.0])
end

# ─────────────────────────────────────────────────────────────────────────────
# Parameter vector writer / extractor
# ─────────────────────────────────────────────────────────────────────────────
function _write_candidate_VD_json(template_path::AbstractString,
                                  out_path::AbstractString,
                                  θ::Vector{Float64};
                                  n_axial_roots::Int,
                                  max_order1_per_plant::Int)
    d = _read_json(template_path)
    d["root_elongation_mode"]     = "logistic"
    d["elongation_rate_min"]      = θ[1]
    d["elongation_rate_max"]      = θ[2]
    d["elongation_logistic_k"]    = θ[3]
    d["elongation_logistic_t50"]  = θ[4]
    d["branch_angle_order1"]      = θ[5]
    d["g_axial"]                  = θ[6]
    d["branching_prob_order1"]    = θ[7]
    d["branching_prob_order2"]    = θ[8]
    d["n_axial_roots"]            = n_axial_roots
    d["max_order1_per_plant"]     = max_order1_per_plant
    _write_json_pretty(out_path, d)
end

function _extract_theta(d::Dict)::Vector{Float64}
    return [
        Float64(get(d, "elongation_rate_min",     0.05)),
        Float64(get(d, "elongation_rate_max",      0.70)),
        Float64(get(d, "elongation_logistic_k",    0.12)),
        Float64(get(d, "elongation_logistic_t50",  20.0)),
        Float64(get(d, "branch_angle_order1",       80.0)),
        Float64(get(d, "g_axial",                   0.05)),
        Float64(get(d, "branching_prob_order1",     0.20)),
        Float64(get(d, "branching_prob_order2",     0.30)),
    ]
end

# ─────────────────────────────────────────────────────────────────────────────
# Simulation runners
# ─────────────────────────────────────────────────────────────────────────────
function _run_sim_1tube(env_json_path::AbstractString,
                        vd_template_path::AbstractString,
                        θ::Vector{Float64},
                        need_days::Vector{Int};
                        seed::Int,
                        n_axial_roots::Int,
                        max_order1_per_plant::Int,
                        tube_x_from_left_cm::Float64=50.0,
                        box_width_cm::Float64=100.0,
                        tube_origin_y_cm::Float64=20.0,
                        tmp_dir::AbstractString)
    mkpath(tmp_dir)
    runid  = string("s", seed, "_", abs(hash(θ)) % 10^9)
    outdir = joinpath(tmp_dir, "out_$(runid)")
    mkpath(outdir)
    vd_cand = joinpath(outdir, "VD.json")
    _write_candidate_VD_json(vd_template_path, vd_cand, θ;
        n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)
    env = _read_json(env_json_path)
    env["species_params_files"]["VD"]    = vd_cand
    env["simulation"]["seed"]            = seed
    env["simulation"]["days"]            = maximum(need_days)
    env["simulation"]["outdir"]          = outdir
    env["simulation"]["plot3d_outdir"]   = joinpath(outdir, "plot3d")
    env["facilitation"]["enabled"]       = false
    env["density_regulation"]["enabled"] = false
    env["minirhizotron"]["enabled"]      = false
    env_path = joinpath(outdir, "env.json")
    _write_json_pretty(env_path, env)
    cfg_sim  = load_environment_and_plants(env_path)
    tube     = _make_tube(tube_x_from_left_cm; W_cm=box_width_cm, tube_origin_y_cm=tube_origin_y_cm)
    ns       = length(tube.station_positions_cm)
    na       = length(tube.camera_azimuths_deg)
    need_set = Set(need_days)
    results  = Dict{Int, Vector{Float64}}()
    R_dummy  = zeros(1, 1)
    Random.seed!(seed)
    for day in 1:maximum(need_days)
        for pl in cfg_sim.plants
            step_plant!(pl, cfg_sim.box, cfg_sim.reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end
        if day in need_set
            mat = tube_station_azimuth_lengths_cm(cfg_sim.plants, tube)
            results[day] = [mat[si, ai] for si in 1:ns for ai in 1:na]
        end
    end
    return results
end

function _run_sim_3tubes(env_json_path::AbstractString,
                         vd_template_path::AbstractString,
                         θ::Vector{Float64},
                         need_days::Vector{Int};
                         seed::Int,
                         n_axial_roots::Int,
                         max_order1_per_plant::Int,
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
        n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)
    env = _read_json(env_json_path)
    env["species_params_files"]["VD"]    = vd_cand
    env["simulation"]["seed"]            = seed
    env["simulation"]["days"]            = maximum(need_days)
    env["simulation"]["outdir"]          = outdir
    env["simulation"]["plot3d_outdir"]   = joinpath(outdir, "plot3d")
    env["facilitation"]["enabled"]       = false
    env["density_regulation"]["enabled"] = false
    env["minirhizotron"]["enabled"]      = false
    env_path = joinpath(outdir, "env.json")
    _write_json_pretty(env_path, env)
    cfg_sim  = load_environment_and_plants(env_path)
    tubes    = [_make_tube(tx; W_cm=box_width_cm, tube_origin_y_cm=tube_origin_y_cm) for tx in tube_x_cm_3tubes]
    ns       = length(tubes[1].station_positions_cm)
    na       = length(tubes[1].camera_azimuths_deg)
    need_set = Set(need_days)
    results  = Dict{Int, Dict{Int, Vector{Float64}}}(
        tid => Dict{Int, Vector{Float64}}() for tid in 1:length(tubes))
    R_dummy  = zeros(1, 1)
    Random.seed!(seed)
    for day in 1:maximum(need_days)
        for pl in cfg_sim.plants
            step_plant!(pl, cfg_sim.box, cfg_sim.reg, R_dummy, R_dummy, 1, 1;
                mature_branching=false, facilitation=nothing,
                facil_gates=nothing, minirhizotron=nothing)
        end
        if day in need_set
            for (tid, tube) in enumerate(tubes)
                mat = tube_station_azimuth_lengths_cm(cfg_sim.plants, tube)
                results[tid][day] = [mat[si, ai] for si in 1:ns for ai in 1:na]
            end
        end
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
function _regression_metrics(x::Vector{Float64}, y::Vector{Float64})
    n     = length(x)
    rmse  = sqrt(mean((y .- x).^2))
    x̄     = mean(x); ȳ = mean(y)
    sst   = sum((x .- x̄).^2)
    r2    = sst == 0 ? NaN : 1.0 - sum((y .- x).^2) / sst
    vx    = sum((x .- x̄).^2) / max(1, n-1)
    vy    = sum((y .- ȳ).^2) / max(1, n-1)
    cxy   = sum((x .- x̄) .* (y .- ȳ)) / max(1, n-1)
    ccc_d = vx + vy + (x̄ - ȳ)^2
    ccc   = ccc_d == 0 ? NaN : (2cxy) / ccc_d
    r_p   = (vx == 0 || vy == 0) ? NaN : cxy / sqrt(vx * vy)
    β     = x̄ == 0 ? NaN : ȳ / x̄
    σx    = sqrt(max(vx, 0.0)); σy = sqrt(max(vy, 0.0))
    γ     = (x̄ == 0 || ȳ == 0 || σx == 0) ? NaN : (σy / ȳ) / (σx / x̄)
    kge   = (isnan(r_p) || isnan(β) || isnan(γ)) ? NaN :
            1.0 - sqrt((r_p-1)^2 + (β-1)^2 + (γ-1)^2)
    pbias = sum(x) == 0 ? NaN : 100.0 * sum(x .- y) / sum(x)
    return (rmse=rmse, r2=r2, ccc=ccc, kge=kge, pbias=pbias)
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared convergence loop (called by both public entry points)
# ─────────────────────────────────────────────────────────────────────────────
function _do_convergence_analysis(;
    θ::Vector{Float64},
    n_axial_roots::Int,
    max_order1_per_plant::Int,
    tube_setup::String,
    tube_x_1tube::Float64,
    tube_x_3tubes::Vector{Float64},
    tube_ids::Vector{Int},
    env_json::String,
    box_width_cm::Float64,
    tube_origin_y_cm::Float64,
    obs_target::Dict{Int,Vector{Float64}},
    used_days::Vector{Int},
    n_windows::Int,
    n_seeds::Int,
    n_calib_seeds::Int,
    out_dir::String,
    title_prefix::String)

    vd_template = joinpath(_CALIB_DIR, "Parameter_VD_template.json")
    tmp_dir = joinpath(out_dir, "__conv_tmp__")
    mkpath(tmp_dir)

    obs_vec = Float64[]
    for d in used_days
        append!(obs_vec, get(obs_target, d, zeros(n_windows)))
    end
    nz_mask = (obs_vec .> 0.0)
    @printf("  Observation windows : %d total, %d non-zero (%.1f%%)\n",
            length(obs_vec), sum(nz_mask), 100*mean(nz_mask))

    println("\n─── Running $n_seeds seeds for convergence analysis ────────────────")
    pred_sum = Dict(d => zeros(n_windows) for d in used_days)
    rows = DataFrame(k=Int[], seed=Int[], r2_all=Float64[], rmse_all=Float64[],
                     ccc_all=Float64[], kge_all=Float64[], pbias_all=Float64[],
                     r2_nz=Float64[], rmse_nz=Float64[], ccc_nz=Float64[])

    for k in 1:n_seeds
        seed = k

        if tube_setup == "3tubes"
            sim_by_tube = _run_sim_3tubes(env_json, vd_template, θ, used_days;
                seed=seed, n_axial_roots=n_axial_roots,
                max_order1_per_plant=max_order1_per_plant,
                tube_x_cm_3tubes=tube_x_3tubes, box_width_cm=box_width_cm,
                tube_origin_y_cm=tube_origin_y_cm, tmp_dir=tmp_dir)
            for d in used_days
                vecs = [get(get(sim_by_tube, tid, Dict()), d, zeros(n_windows))
                        for tid in tube_ids]
                valid = filter(v -> length(v) == n_windows, vecs)
                isempty(valid) && continue
                pred_sum[d] .+= mean(valid)
            end
        else
            sim = _run_sim_1tube(env_json, vd_template, θ, used_days;
                seed=seed, n_axial_roots=n_axial_roots,
                max_order1_per_plant=max_order1_per_plant,
                tube_x_from_left_cm=tube_x_1tube, box_width_cm=box_width_cm,
                tube_origin_y_cm=tube_origin_y_cm, tmp_dir=tmp_dir)
            for d in used_days
                v = get(sim, d, Float64[])
                length(v) == n_windows || continue
                pred_sum[d] .+= v
            end
        end

        pred_avg_vec = Float64[]
        for d in used_days
            append!(pred_avg_vec, clamp.(pred_sum[d] ./ k, 0.0, Inf))
        end

        m_all = _regression_metrics(obs_vec, pred_avg_vec)
        obs_nz  = obs_vec[nz_mask]; pred_nz = pred_avg_vec[nz_mask]
        m_nz = isempty(obs_nz) ? (r2=NaN, rmse=NaN, ccc=NaN, kge=NaN, pbias=NaN) :
               _regression_metrics(obs_nz, pred_nz)

        push!(rows, (k, seed, m_all.r2, m_all.rmse, m_all.ccc, m_all.kge, m_all.pbias,
                     m_nz.r2, m_nz.rmse, m_nz.ccc))

        @printf("  k=%2d  seed=%2d  R²(all)=%+.3f  RMSE(all)=%.4f  R²(nz)=%+.3f  RMSE(nz)=%.4f\n",
                k, seed, m_all.r2, m_all.rmse, m_nz.r2, m_nz.rmse)
    end

    CSV.write(joinpath(out_dir, "convergence_metrics.csv"), rows)

    w = 5
    r2_all   = rows.r2_all
    r2_nz    = rows.r2_nz
    rmse_all = rows.rmse_all
    ks = rows.k

    rolling_std_r2 = fill(NaN, length(ks))
    for i in w:length(ks)
        rolling_std_r2[i] = std(r2_all[i-w+1:i])
    end

    Plots.gr()
    p1 = Plots.plot(ks, r2_all;
        label="R² (all windows)", lw=2, color=:steelblue,
        xlabel="Number of seeds averaged (k)", ylabel="R²",
        title="$title_prefix\nR² convergence vs seed count",
        legend=:bottomright)
    Plots.plot!(p1, ks, r2_nz; label="R² (non-zero windows only)", lw=2,
        color=:darkorange, linestyle=:dash)
    Plots.hline!(p1, [0.0]; label=false, color=:black, linestyle=:dot, lw=1)
    Plots.vline!(p1, [n_calib_seeds]; label="calibration_seeds ($n_calib_seeds)",
        color=:red, linestyle=:dash, lw=1.5)
    Plots.savefig(p1, joinpath(out_dir, "convergence_r2.png"))

    p2 = Plots.plot(ks, rmse_all;
        label="RMSE (all windows)", lw=2, color=:steelblue,
        xlabel="Number of seeds averaged (k)", ylabel="RMSE (cm)",
        title="$title_prefix\nRMSE convergence vs seed count",
        legend=:topright)
    Plots.plot!(p2, ks, rows.rmse_nz; label="RMSE (non-zero windows only)", lw=2,
        color=:darkorange, linestyle=:dash)
    Plots.vline!(p2, [n_calib_seeds]; label="calibration_seeds ($n_calib_seeds)",
        color=:red, linestyle=:dash, lw=1.5)
    Plots.savefig(p2, joinpath(out_dir, "convergence_rmse.png"))

    valid = .!isnan.(rolling_std_r2)
    p3 = Plots.plot(ks[valid], rolling_std_r2[valid];
        label="Rolling σ(R²), window=$w", lw=2, color=:purple,
        xlabel="Number of seeds averaged (k)", ylabel="σ(R²)",
        title="$title_prefix\nR² stability (rolling std, window=$w seeds)")
    Plots.hline!(p3, [0.01]; label="σ=0.01 threshold", color=:gray, linestyle=:dot)
    Plots.vline!(p3, [n_calib_seeds]; label="calibration_seeds ($n_calib_seeds)",
        color=:red, linestyle=:dash, lw=1.5)
    Plots.savefig(p3, joinpath(out_dir, "convergence_r2_stability.png"))

    # Find convergence point
    conv_k = nothing; consec = 0
    for i in 1:length(ks)
        isnan(rolling_std_r2[i]) && continue
        if rolling_std_r2[i] < 0.01
            consec += 1
            consec >= 3 && (conv_k = ks[i-2]; break)
        else
            consec = 0
        end
    end

    println("\n─── Convergence Summary ─────────────────────────────────────────")
    @printf("  Source          : %s\n", title_prefix)
    @printf("  Tube setup      : %s\n", tube_setup)
    @printf("  Seeds evaluated : %d (k=1..%d)\n", n_seeds, n_seeds)
    @printf("  calibration_seeds used/planned : %d\n", n_calib_seeds)
    @printf("  R²(k=%d) all    : %+.3f\n", n_seeds, r2_all[end])
    @printf("  R²(k=%d) nz     : %+.3f\n", n_seeds, r2_nz[end])
    if conv_k !== nothing
        @printf("  Convergence at  : k≈%d (rolling σ(R²) < 0.01)\n", conv_k)
        if conv_k > n_calib_seeds
            @printf("  *** WARNING: R² stabilises at k≈%d but calibration_seeds=%d — increase it\n",
                    conv_k, n_calib_seeds)
        else
            @printf("  OK: calibration_seeds=%d ≥ convergence point k≈%d\n",
                    n_calib_seeds, conv_k)
        end
    else
        @printf("  Convergence not reached within %d seeds (consider increasing N_seeds)\n", n_seeds)
    end
    println("\nOutputs written to: $out_dir")

    try; rm(tmp_dir; recursive=true); catch; end
    return rows
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode A: post-calibration (from completed experiment directory)
# ─────────────────────────────────────────────────────────────────────────────
function run_convergence_analysis(exp_dir::AbstractString, n_seeds::Int)
    best_path    = joinpath(exp_dir, "best_params.json")
    summary_path = joinpath(exp_dir, "run_summary.json")
    isfile(best_path)    || error("best_params.json not found in $exp_dir")
    isfile(summary_path) || error("run_summary.json not found in $exp_dir")

    bp      = _read_json(best_path)
    summary = _read_json(summary_path)
    config  = let cf = String(get(summary, "config_file", ""))
                  isfile(cf) ? _read_json(cf) : Dict{String,Any}()
              end

    θ                    = _extract_theta(bp)
    n_axial_roots        = Int(get(bp, "n_axial_roots", 1))
    max_order1_per_plant = Int(get(bp, "max_order1_per_plant", 20))
    tube_setup           = String(get(bp, "tube_setup", "1tube"))
    n_calib_seeds        = length(get(bp, "calibration_seeds", [11, 22, 99, 88]))

    _env_raw  = get(config, "env_json", "Parameter_calib_minirhizotron_200cm.json")
    env_json  = isabspath(String(_env_raw)) ? String(_env_raw) :
                joinpath(_CALIB_DIR, String(_env_raw))
    env_raw   = _read_json(env_json)
    tube_origin_y_cm = Float64(get(get(env_raw, "minirhizotron", Dict()),
                                    "tube_origin_cm", [0.0, 20.0, 0.0])[2])
    box_width_cm = Float64(get(config, "box_width_cm", 100.0))

    out_dir = joinpath(exp_dir, "seed_convergence")
    mkpath(out_dir)

    tube_x_1tube  = 100.0
    tube_x_3tubes = [75.0, 100.0, 125.0]
    tube_ids      = Int[]
    obs_target    = Dict{Int,Vector{Float64}}()
    used_days     = Int[]
    n_windows     = 60

    if tube_setup == "3tubes"
        obs_csv = String(get(config, "obs_csv_3tubes",
                              joinpath(_CALIB_DIR, "data", "mr_obs_stations_3tubes_200cm.csv")))
        obs_csv = isabspath(obs_csv) ? obs_csv : joinpath(_CALIB_DIR, obs_csv)
        used_days, obs_3tube, _ = _load_obs_3tubes(obs_csv)
        tube_x_3tubes = Float64.(get(config, "tube_x_positions_cm", [75.0, 100.0, 125.0]))
        tube_ids  = sort(collect(keys(obs_3tube)))
        n_windows = length(get(get(obs_3tube, tube_ids[1], Dict()), used_days[1], zeros(60)))
        obs_target = Dict(d => mean([get(get(obs_3tube, tid, Dict()), d, zeros(n_windows))
                                     for tid in tube_ids])
                          for d in used_days)
    else
        obs_csv = String(get(config, "obs_csv_1tube",
                              joinpath(_CALIB_DIR, "data", "mr_obs_stations_1tube_200cm.csv")))
        obs_csv = isabspath(obs_csv) ? obs_csv : joinpath(_CALIB_DIR, obs_csv)
        used_days, obs_1tube, _ = _load_obs_1tube(obs_csv)
        tube_x_1tube = Float64(get(config, "tube_x_1tube_cm", 100.0))
        n_windows  = length(get(obs_1tube, used_days[1], zeros(60)))
        obs_target = obs_1tube
    end

    @info "Mode A: post-calibration convergence check" exp_dir=basename(exp_dir) tube_setup=tube_setup

    _do_convergence_analysis(;
        θ=θ, n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant,
        tube_setup=tube_setup, tube_x_1tube=tube_x_1tube, tube_x_3tubes=tube_x_3tubes,
        tube_ids=tube_ids, env_json=env_json, box_width_cm=box_width_cm,
        tube_origin_y_cm=tube_origin_y_cm, obs_target=obs_target,
        used_days=used_days, n_windows=n_windows,
        n_seeds=n_seeds, n_calib_seeds=n_calib_seeds,
        out_dir=out_dir, title_prefix=basename(exp_dir))
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode B: pre-calibration (from template VD + calib config, no best_params needed)
# ─────────────────────────────────────────────────────────────────────────────
function run_convergence_analysis_from_template(vd_json::AbstractString,
                                                 config_path::AbstractString,
                                                 n_seeds::Int,
                                                 out_dir::AbstractString)
    isfile(vd_json)     || error("VD JSON not found: $vd_json")
    isfile(config_path) || error("Calibration config not found: $config_path")

    vd     = _read_json(vd_json)
    config = _read_json(config_path)

    θ                    = _extract_theta(vd)
    n_axial_roots        = Int(get(vd, "n_axial_roots", 1))
    max_order1_per_plant = Int(get(vd, "max_order1_per_plant", 20))
    tube_setup           = String(get(config, "tube_setup", "1tube"))
    # planned calibration seeds — read from config so the plot marks the right line
    n_calib_seeds        = length(get(config, "calibration_seeds", [11, 22, 99, 88]))

    _env_raw  = get(config, "env_json", "Parameter_calib_minirhizotron_200cm.json")
    env_json  = isabspath(String(_env_raw)) ? String(_env_raw) :
                joinpath(_CALIB_DIR, String(_env_raw))
    env_raw   = _read_json(env_json)
    tube_origin_y_cm = Float64(get(get(env_raw, "minirhizotron", Dict()),
                                    "tube_origin_cm", [0.0, 20.0, 0.0])[2])
    box_width_cm = Float64(get(config, "box_width_cm", 100.0))

    mkpath(out_dir)

    tube_x_1tube  = 100.0
    tube_x_3tubes = [75.0, 100.0, 125.0]
    tube_ids      = Int[]
    obs_target    = Dict{Int,Vector{Float64}}()
    used_days     = Int[]
    n_windows     = 60

    if tube_setup == "3tubes"
        obs_csv = String(get(config, "obs_csv_3tubes",
                              joinpath(_CALIB_DIR, "data", "mr_obs_stations_3tubes_200cm.csv")))
        obs_csv = isabspath(obs_csv) ? obs_csv : joinpath(_CALIB_DIR, obs_csv)
        used_days, obs_3tube, _ = _load_obs_3tubes(obs_csv)
        tube_x_3tubes = Float64.(get(config, "tube_x_positions_cm", [75.0, 100.0, 125.0]))
        tube_ids  = sort(collect(keys(obs_3tube)))
        n_windows = length(get(get(obs_3tube, tube_ids[1], Dict()), used_days[1], zeros(60)))
        obs_target = Dict(d => mean([get(get(obs_3tube, tid, Dict()), d, zeros(n_windows))
                                     for tid in tube_ids])
                          for d in used_days)
    else
        obs_csv = String(get(config, "obs_csv_1tube",
                              joinpath(_CALIB_DIR, "data", "mr_obs_stations_1tube_200cm.csv")))
        obs_csv = isabspath(obs_csv) ? obs_csv : joinpath(_CALIB_DIR, obs_csv)
        used_days, obs_1tube, _ = _load_obs_1tube(obs_csv)
        tube_x_1tube = Float64(get(config, "tube_x_1tube_cm", 100.0))
        n_windows  = length(get(obs_1tube, used_days[1], zeros(60)))
        obs_target = obs_1tube
    end

    @info "Mode B: pre-calibration convergence check (template VD)" vd=basename(vd_json) config=basename(config_path) tube_setup=tube_setup

    _do_convergence_analysis(;
        θ=θ, n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant,
        tube_setup=tube_setup, tube_x_1tube=tube_x_1tube, tube_x_3tubes=tube_x_3tubes,
        tube_ids=tube_ids, env_json=env_json, box_width_cm=box_width_cm,
        tube_origin_y_cm=tube_origin_y_cm, obs_target=obs_target,
        used_days=used_days, n_windows=n_windows,
        n_seeds=n_seeds, n_calib_seeds=n_calib_seeds,
        out_dir=out_dir, title_prefix="template: $(basename(vd_json))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
const _USAGE = """
analyse_seed_convergence.jl — determine how many seeds are needed for stable R²

Mode A — post-calibration (verify seed count used was sufficient):
  julia calibration/analyse_seed_convergence.jl <experiment_dir> [N_seeds]

Mode B — pre-calibration (determine K BEFORE running calibration):
  julia calibration/analyse_seed_convergence.jl \\
    --template <vd_json> --config <calib_config_json> [--out <out_dir>] [N_seeds]

Arguments:
  experiment_dir   Path to a completed experiment (contains best_params.json)
  --template       Path to a VD parameter JSON (e.g. Parameter_VD_template.json)
  --config         Path to a calibration config JSON (e.g. configs/calib_200cm_1tube.json)
  --out            Output directory for Mode B (default: calibration/seed_convergence_pretrial/)
  N_seeds          Number of seeds to evaluate (default: 30)

Recommended workflow:
  1. Run Mode B with the template VD and your planned calibration config
  2. Read the convergence plot — find the elbow k where σ(R²) < 0.01
  3. Set calibration_seeds in your config to k values
  4. Run calibration
  5. Optionally run Mode A to confirm
"""

function _main()
    if length(ARGS) < 1
        println(_USAGE)
        exit(1)
    end

    if ARGS[1] == "--template"
        # Mode B: --template <vd> --config <cfg> [--out <dir>] [N]
        local vd_json     = ""
        local config_path = ""
        local out_dir     = joinpath(_CALIB_DIR, "seed_convergence_pretrial")
        local n_seeds     = 30
        local i = 1
        while i <= length(ARGS)
            if ARGS[i] == "--template" && i < length(ARGS)
                vd_json = ARGS[i+1]; i += 2
            elseif ARGS[i] == "--config" && i < length(ARGS)
                config_path = ARGS[i+1]; i += 2
            elseif ARGS[i] == "--out" && i < length(ARGS)
                out_dir = ARGS[i+1]; i += 2
            else
                try n_seeds = parse(Int, ARGS[i]); catch; end
                i += 1
            end
        end
        isempty(vd_json)     && error("--template requires a path argument")
        isempty(config_path) && error("--config requires a path argument")
        run_convergence_analysis_from_template(vd_json, config_path, n_seeds, out_dir)
    else
        # Mode A: <experiment_dir> [N_seeds]
        local exp_dir = ARGS[1]
        local n_seeds = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30
        run_convergence_analysis(exp_dir, n_seeds)
    end
end

_main()
