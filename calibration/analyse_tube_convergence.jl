# analyse_tube_convergence.jl
#
# Evaluates how many tube positions are needed for stable R² in the obs-pred comparison.
# This is the spatial-averaging counterpart to analyse_seed_convergence.jl:
#
#   Seed convergence  → reduces stochastic variance in PREDICTIONS
#   Tube convergence  → reduces stochastic variance in OBSERVATIONS
#
# The analysis runs two N-tube simulations in one pass:
#   1. "Ground truth" obs: template VD params (Parameter_VD_template.json) + MOCK_SEED
#   2. "Prediction":       calibrated params   (best_params.json)           + MOCK_SEED
#
# N tubes are evenly distributed across the valid plant zone. For m=1..N, the
# first m tubes are averaged and R²(m) is computed. The elbow in R²(m) vs m
# indicates the minimum number of physical tubes needed for a stable signal.
#
# Usage:
#   julia calibration/analyse_tube_convergence.jl <experiment_dir> [N_tubes [MOCK_SEED]]
#
# Defaults: N_tubes=15, MOCK_SEED=42
#
# Outputs (written to <experiment_dir>/tube_convergence/):
#   tube_positions.csv          — x_from_left_cm for each tube ID 1..N
#   convergence_metrics.csv     — m, r2_all, rmse_all, r2_nz, rmse_nz, ccc_all for each m
#   convergence_r2.png          — R²(m) and RMSE(m) vs number of tubes averaged
#   convergence_r2_stability.png — rolling σ(R²) vs m (stability indicator)

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
# Parameter vector writer
# ─────────────────────────────────────────────────────────────────────────────
function _write_candidate_VD_json(template_path::AbstractString,
                                  out_path::AbstractString,
                                  θ::Vector{Float64};
                                  n_axial_roots::Int,
                                  max_order1_per_plant::Int)
    d = _read_json(template_path)
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
# N-tube simulation runner
# Returns Dict{tube_id => Dict{day => Vector{Float64}}}
# ─────────────────────────────────────────────────────────────────────────────
function _run_sim_Ntubes(env_json_path::AbstractString,
                          vd_json_path::AbstractString,
                          need_days::Vector{Int};
                          seed::Int,
                          tube_x_cm::Vector{Float64},
                          box_width_cm::Float64,
                          tube_origin_y_cm::Float64,
                          tmp_dir::AbstractString,
                          label::AbstractString="run")
    mkpath(tmp_dir)
    outdir = joinpath(tmp_dir, "out_$(label)_s$(seed)")
    mkpath(outdir)

    env = _read_json(env_json_path)
    env["species_params_files"]["VD"]    = vd_json_path
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
    tubes    = [_make_tube(tx; W_cm=box_width_cm, tube_origin_y_cm=tube_origin_y_cm)
                for tx in tube_x_cm]
    n_tubes  = length(tubes)
    ns       = length(tubes[1].station_positions_cm)
    na       = length(tubes[1].camera_azimuths_deg)
    need_set = Set(need_days)
    results  = Dict{Int, Dict{Int, Vector{Float64}}}(
        tid => Dict{Int, Vector{Float64}}() for tid in 1:n_tubes)
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
# Observation dates (same schedule as mock generators)
# ─────────────────────────────────────────────────────────────────────────────
const _ACQ_OFFSETS = 30:14:90   # days after sowing: 30, 44, 58, 72, 86

function _acq_days()
    days  = [Int(d) + 1 for d in _ACQ_OFFSETS]
    dates = Dict(Int(d)+1 => _SOWING_DATE + Dates.Day(d) for d in _ACQ_OFFSETS)
    return days, dates
end

# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────
function run_tube_convergence_analysis(exp_dir::AbstractString,
                                        n_tubes::Int,
                                        mock_seed::Int)
    isdir(exp_dir) || error("Experiment directory not found: $exp_dir")

    best_path    = joinpath(exp_dir, "best_params.json")
    summary_path = joinpath(exp_dir, "run_summary.json")
    isfile(best_path)    || error("best_params.json not found in $exp_dir")
    isfile(summary_path) || error("run_summary.json not found in $exp_dir")

    bp      = _read_json(best_path)
    summary = _read_json(summary_path)

    θ                   = _extract_theta(bp)
    n_axial_roots       = Int(get(bp, "n_axial_roots", 1))
    max_order1_per_plant = Int(get(bp, "max_order1_per_plant", 20))
    config_file         = String(get(summary, "config_file", ""))
    config              = isfile(config_file) ? _read_json(config_file) : Dict{String,Any}()

    _env_raw  = get(config, "env_json", "Parameter_calib_minirhizotron_200cm.json")
    env_json  = isabspath(String(_env_raw)) ? String(_env_raw) :
                joinpath(_CALIB_DIR, String(_env_raw))
    isfile(env_json) || error("env_json not found: $env_json")

    vd_template = joinpath(_CALIB_DIR, "Parameter_VD_template.json")
    isfile(vd_template) || error("VD template not found: $vd_template")

    box_width_cm = Float64(get(config, "box_width_cm", 200.0))
    border_cm    = 25.0  # plant-free border width

    env_raw = _read_json(env_json)
    tube_origin_y_cm = Float64(get(get(env_raw, "minirhizotron", Dict()),
                                    "tube_origin_cm", [0.0, 20.0, 0.0])[2])

    # ── Tube positions: evenly spaced inside plant zone ──────────────────────
    # Plant zone: x_from_left ∈ [border_cm, box_width_cm - border_cm]
    x_min = border_cm
    x_max = box_width_cm - border_cm
    tube_x_cm = n_tubes == 1 ? [box_width_cm / 2.0] :
                [x_min + (x_max - x_min) * (i-1) / (n_tubes-1) for i in 1:n_tubes]

    out_dir = joinpath(exp_dir, "tube_convergence")
    mkpath(out_dir)
    tmp_dir = joinpath(out_dir, "__conv_tmp__")
    mkpath(tmp_dir)

    # Save tube positions
    df_pos = DataFrame(tube_id=1:n_tubes, x_from_left_cm=tube_x_cm)
    CSV.write(joinpath(out_dir, "tube_positions.csv"), df_pos)

    used_days, simday_to_date = _acq_days()
    n_windows = 60  # 15 stations × 4 azimuths

    println("\n─── Tube convergence analysis ──────────────────────────────────────")
    @printf("  Experiment       : %s\n", basename(exp_dir))
    @printf("  Box width        : %.0f cm  (plant zone: %.0f–%.0f cm)\n",
            box_width_cm, x_min, x_max)
    @printf("  Tube origin y    : %.1f cm\n", tube_origin_y_cm)
    @printf("  N tubes          : %d\n", n_tubes)
    @printf("  Tube x positions : %s cm\n",
            join([@sprintf("%.1f", x) for x in tube_x_cm], ", "))
    @printf("  Mock seed        : %d\n", mock_seed)
    @printf("  Acquisition days : %s\n", join(string.(used_days), ", "))

    # ── Simulation 1: ground truth obs (template params + mock_seed) ─────────
    println("\n[1/2] Running ground-truth simulation (template VD, seed=$mock_seed)...")
    obs_by_tube = _run_sim_Ntubes(env_json, vd_template, used_days;
        seed=mock_seed, tube_x_cm=tube_x_cm, box_width_cm=box_width_cm,
        tube_origin_y_cm=tube_origin_y_cm, tmp_dir=tmp_dir, label="obs")

    # ── Simulation 2: predictions (best_params + mock_seed) ──────────────────
    println("[2/2] Running prediction simulation (best_params, seed=$mock_seed)...")
    vd_best = joinpath(tmp_dir, "VD_best.json")
    _write_candidate_VD_json(vd_template, vd_best, θ;
        n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)
    pred_by_tube = _run_sim_Ntubes(env_json, vd_best, used_days;
        seed=mock_seed, tube_x_cm=tube_x_cm, box_width_cm=box_width_cm,
        tube_origin_y_cm=tube_origin_y_cm, tmp_dir=tmp_dir, label="pred")

    # ── Print per-tube totals (sanity check) ─────────────────────────────────
    println("\n─── Per-tube signal check ──────────────────────────────────────────")
    @printf("  %-6s  %-8s  %-12s  %-12s\n", "tube", "x_cm", "obs_total", "pred_total")
    for tid in 1:n_tubes
        obs_tot  = sum(sum(get(obs_by_tube[tid],  d, zeros(n_windows))) for d in used_days)
        pred_tot = sum(sum(get(pred_by_tube[tid], d, zeros(n_windows))) for d in used_days)
        @printf("  %-6d  %-8.1f  %-12.3f  %-12.3f\n",
                tid, tube_x_cm[tid], obs_tot, pred_tot)
    end

    # ── Build flattened obs/pred vectors per tube ────────────────────────────
    flat_obs  = Dict{Int, Vector{Float64}}()
    flat_pred = Dict{Int, Vector{Float64}}()
    for tid in 1:n_tubes
        o = Float64[]; p = Float64[]
        for d in used_days
            append!(o, get(obs_by_tube[tid],  d, zeros(n_windows)))
            append!(p, get(pred_by_tube[tid], d, zeros(n_windows)))
        end
        flat_obs[tid]  = o
        flat_pred[tid] = clamp.(p, 0.0, Inf)
    end
    total_windows = length(used_days) * n_windows

    # ── Convergence loop: accumulate tubes 1..m ──────────────────────────────
    println("\n─── R²(m) for m = 1 to $n_tubes tubes ─────────────────────────────")
    obs_sum  = zeros(total_windows)
    pred_sum = zeros(total_windows)
    rows = DataFrame(m=Int[], tube_id=Int[], x_from_left_cm=Float64[],
                     r2_all=Float64[], rmse_all=Float64[], ccc_all=Float64[],
                     kge_all=Float64[], pbias_all=Float64[],
                     r2_nz=Float64[], rmse_nz=Float64[], ccc_nz=Float64[])

    for m in 1:n_tubes
        obs_sum  .+= flat_obs[m]
        pred_sum .+= flat_pred[m]

        obs_avg  = obs_sum  ./ m
        pred_avg = pred_sum ./ m

        nz_mask = (obs_avg .> 0.0)
        m_all = _regression_metrics(obs_avg, pred_avg)
        obs_nz  = obs_avg[nz_mask]
        pred_nz = pred_avg[nz_mask]
        m_nz = isempty(obs_nz) ?
               (r2=NaN, rmse=NaN, ccc=NaN, kge=NaN, pbias=NaN) :
               _regression_metrics(obs_nz, pred_nz)

        push!(rows, (m, m, tube_x_cm[m],
                     m_all.r2, m_all.rmse, m_all.ccc, m_all.kge, m_all.pbias,
                     m_nz.r2, m_nz.rmse, m_nz.ccc))

        @printf("  m=%2d  x=%.1f  R²(all)=%+.3f  RMSE(all)=%.4f  R²(nz)=%+.3f  RMSE(nz)=%.4f  nz=%.1f%%\n",
                m, tube_x_cm[m], m_all.r2, m_all.rmse,
                m_nz.r2, m_nz.rmse,
                100 * mean(nz_mask))
    end

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = joinpath(out_dir, "convergence_metrics.csv")
    CSV.write(csv_path, rows)

    # ── Rolling std (window=3 since N may be small) ───────────────────────────
    w = min(3, n_tubes ÷ 3)
    r2_all  = rows.r2_all
    r2_nz   = rows.r2_nz
    rmse_all = rows.rmse_all
    ms = rows.m

    rolling_std_r2 = fill(NaN, length(ms))
    for i in w:length(ms)
        rolling_std_r2[i] = std(r2_all[i-w+1:i])
    end

    n_calib_tubes = length(get(config, "tube_x_positions_cm",
                               get(config, "tube_x_1tube_cm", [100.0])))
    tube_label = n_calib_tubes == 1 ? "1 tube used" : "$n_calib_tubes tubes used"

    # ── Plot 1: R²(m) vs m ─────────────────────────────────────────────────
    Plots.gr()
    p1 = Plots.plot(ms, r2_all;
        label="R² (all windows)", lw=2, color=:steelblue,
        xlabel="Number of tubes averaged (m)", ylabel="R²",
        title="$(basename(exp_dir))\nR² convergence vs tube count\n(obs=template+seed$(mock_seed), pred=best_params+seed$(mock_seed))",
        legend=:bottomright,
        xticks=1:n_tubes)
    Plots.plot!(p1, ms, r2_nz; label="R² (non-zero windows only)", lw=2,
        color=:darkorange, linestyle=:dash)
    Plots.hline!(p1, [0.0]; label=false, color=:black, linestyle=:dot, lw=1)
    Plots.vline!(p1, [n_calib_tubes]; label=tube_label, color=:red,
        linestyle=:dash, lw=1.5)
    Plots.savefig(p1, joinpath(out_dir, "convergence_r2.png"))

    # ── Plot 2: RMSE(m) vs m ───────────────────────────────────────────────
    p2 = Plots.plot(ms, rmse_all;
        label="RMSE (all windows)", lw=2, color=:steelblue,
        xlabel="Number of tubes averaged (m)", ylabel="RMSE (cm)",
        title="$(basename(exp_dir))\nRMSE convergence vs tube count",
        legend=:topright, xticks=1:n_tubes)
    Plots.plot!(p2, ms, rows.rmse_nz; label="RMSE (non-zero windows only)", lw=2,
        color=:darkorange, linestyle=:dash)
    Plots.vline!(p2, [n_calib_tubes]; label=tube_label, color=:red,
        linestyle=:dash, lw=1.5)
    Plots.savefig(p2, joinpath(out_dir, "convergence_rmse.png"))

    # ── Plot 3: Rolling std of R² ─────────────────────────────────────────
    valid = .!isnan.(rolling_std_r2)
    if any(valid)
        p3 = Plots.plot(ms[valid], rolling_std_r2[valid];
            label="Rolling σ(R²), window=$w", lw=2, color=:purple,
            xlabel="Number of tubes averaged (m)", ylabel="σ(R²)",
            title="$(basename(exp_dir))\nR² stability (rolling std, window=$w tubes)",
            xticks=1:n_tubes)
        Plots.hline!(p3, [0.01]; label="σ=0.01 threshold", color=:gray, linestyle=:dot)
        Plots.vline!(p3, [n_calib_tubes]; label=tube_label, color=:red,
            linestyle=:dash, lw=1.5)
        Plots.savefig(p3, joinpath(out_dir, "convergence_r2_stability.png"))
    end

    # ── Convergence point ────────────────────────────────────────────────────
    conv_m = nothing
    consec = 0
    for i in 1:length(ms)
        isnan(rolling_std_r2[i]) && continue
        if rolling_std_r2[i] < 0.01
            consec += 1
            consec >= 2 && (conv_m = ms[i - 1]; break)
        else
            consec = 0
        end
    end

    println("\n─── Convergence Summary ─────────────────────────────────────────")
    @printf("  Experiment       : %s\n", basename(exp_dir))
    @printf("  Tubes evaluated  : %d (m=1..%d)\n", n_tubes, n_tubes)
    @printf("  Tubes in calib.  : %d\n", n_calib_tubes)
    @printf("  R²(m=1)  all     : %+.3f\n", r2_all[1])
    @printf("  R²(m=%d) all     : %+.3f\n", n_tubes, r2_all[end])
    @printf("  R²(m=1)  nz      : %+.3f\n", r2_nz[1])
    @printf("  R²(m=%d) nz      : %+.3f\n", n_tubes, r2_nz[end])
    Δr2 = r2_all[end] - r2_all[1]
    @printf("  ΔR² (m=1→%d)     : %+.3f  (gain from spatial averaging)\n", n_tubes, Δr2)
    if conv_m !== nothing
        @printf("  Convergence at   : m≈%d tubes (rolling σ(R²) < 0.01)\n", conv_m)
        if conv_m > n_calib_tubes
            @printf("  WARNING: calibration used %d tube(s) but R² stabilises at m≈%d\n",
                    n_calib_tubes, conv_m)
        else
            @printf("  OK: calibration tube count (%d) meets or exceeds convergence (m≈%d)\n",
                    n_calib_tubes, conv_m)
        end
    else
        @printf("  Convergence not reached within %d tubes\n", n_tubes)
    end
    println("\nOutputs written to: $out_dir")

    try; rm(tmp_dir; recursive=true); catch; end

    return rows
end

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if length(ARGS) < 1
    println("""
Usage: julia calibration/analyse_tube_convergence.jl <experiment_dir> [N_tubes [MOCK_SEED]]

  experiment_dir  Path to a completed calibration experiment directory
                  (must contain best_params.json and run_summary.json)
  N_tubes         Number of tube positions to evaluate (default: 15)
  MOCK_SEED       Random seed for both obs and pred simulations (default: 42)

N tubes are evenly distributed across the valid plant zone (inside the 25 cm
border on each side of the box). Both the ground-truth obs (template VD) and
the prediction (best_params) are run with the same seed so that tube-position
spatial variance is the only source of difference.

Example:
  julia calibration/analyse_tube_convergence.jl \\
    calibration/experiments/replicate_seed_average_20260617_172848 15 42
""")
    exit(1)
end

exp_dir  = ARGS[1]
n_tubes  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 15
mock_seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 42

run_tube_convergence_analysis(exp_dir, n_tubes, mock_seed)
