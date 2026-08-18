# main_Optimization_minirhizotron_1tube.jl
#
# CMA-ES calibration of virtual dicot root parameters against minirhizotron
# tube-station observations.  Observations come from deep-learning analysis of
# 5×5 cm image windows (5 cm along-tube × 5 cm arc) at four cardinal azimuths
# (0°, 90°, 180°, 270°) sampled every 5 cm along the tube.
# Single tube at x=50 cm (centred, midway between row pairs 4 and 5).
#
# Inputs:
#   calibration/data/mr_obs_stations_1tube.csv  — DL root lengths (or mock)
#   calibration/Parameter_calib_minirhizotron.json
#   calibration/Parameter_VD_template.json
#
# Outputs:
#   ParameterVD_best_mr_1tube.json
#   diagnostics_mr/1tube/best_obs_pred_stations.csv
#   diagnostics_mr/1tube/best_objective_summary_mr.csv
#   diagnostics_mr/1tube/best_obs_vs_pred_ALL.png
#   diagnostics_mr/1tube/depth_profile_day_XXX.png  (one per acquisition date)
#
# Run from repo root:
#   julia calibration/main_Optimization_minirhizotron_1tube.jl

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

# ----------------------------
# Settings  (edit as needed)
# ----------------------------
const SOWING_DATE     = Date(2025, 4, 1)   # first sim day = sowing date
const SEEDS           = (11, 22, 99, 88)   # 4 seeds; excludes obs seed (42) and validation seeds (33,44,55,66,77)
const MAXITER         = 100
const SIGMA0          = 0.1
const GRID_N_AXIAL    = [1]
const GRID_MAX_ORDER1 = [20]               # known optimal from previous grid search; fixed to match 3-tube

# ----------------------------
# Include model code
# ----------------------------
include(joinpath(@__DIR__, "..", "src", "Deps.jl"))
include(joinpath(@__DIR__, "..", "src", "Types.jl"))
include(joinpath(@__DIR__, "..", "src", "Geometry.jl"))
include(joinpath(@__DIR__, "..", "src", "DensityRegulation.jl"))
include(joinpath(@__DIR__, "..", "src", "GrowthStep.jl"))
include(joinpath(@__DIR__, "..", "src", "MetricsAndOutput.jl"))
include(joinpath(@__DIR__, "..", "src", "Plotting.jl"))
include(joinpath(@__DIR__, "..", "src", "Config.jl"))
include(joinpath(@__DIR__, "..", "src", "Simulation.jl"))

# ----------------------------
# JSON helpers
# ----------------------------
read_json(path::AbstractString)  = open(path, "r") do io; JSON.parse(String(read(io))); end
function write_json_pretty(path::AbstractString, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

# ----------------------------
# Date / simday helpers
# ----------------------------
function date_to_simday(d::Date)
    return Dates.value(d - SOWING_DATE) + 1
end

# ----------------------------
# Observation loader
# ----------------------------
# Returns:
#   used_days  :: Vector{Int}              — sorted sim days with observations
#   obs_by_day :: Dict{Int, Vector{Float64}} — sim_day => vector of root_length_cm
#                 sorted by (station_idx ASC, azimuth_deg ASC)
function load_obs_by_simday(obs_csv::AbstractString)
    df = CSV.read(obs_csv, DataFrame)
    df.date = Date.(string.(df.date))

    # sort by (date, station_idx, azimuth_deg) for deterministic ordering
    sort!(df, [:date, :station_idx, :azimuth_deg])

    obs_by_day  = Dict{Int, Vector{Float64}}()
    meta_by_day = Dict{Int, DataFrame}()   # kept for diagnostics

    for subdate in groupby(df, :date)
        d   = subdate.date[1]
        sd  = date_to_simday(d)
        sd >= 1 || continue
        obs_by_day[sd]  = Float64.(subdate.root_length_cm)
        meta_by_day[sd] = DataFrame(
            station_idx   = Int.(subdate.station_idx),
            along_tube_cm = Float64.(subdate.along_tube_cm),
            depth_cm      = Float64.(subdate.depth_cm),
            azimuth_deg   = Float64.(subdate.azimuth_deg)
        )
    end

    used_days = sort(collect(keys(obs_by_day)))
    return used_days, obs_by_day, meta_by_day
end

# ----------------------------
# Candidate parameter writer
# θ = [elong_rate_min, elong_rate_max, logistic_k, logistic_t50,
#      branch_angle_order1, g_axial, branching_prob_order1, branching_prob_order2]
# ----------------------------
function write_candidate_VD_json(template_path::AbstractString,
                                 out_path::AbstractString,
                                 θ::Vector{Float64};
                                 n_axial_roots::Int,
                                 max_order1_per_plant::Int)
    d = read_json(template_path)

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

    write_json_pretty(out_path, d)
    return nothing
end

# ----------------------------
# Run one candidate simulation
# Returns Dict{Int, Vector{Float64}}: sim_day => station lengths
# sorted by (station_idx ASC, azimuth_deg ASC) to match load_obs_by_simday
# ----------------------------
function run_candidate_stations(env_json_path::AbstractString,
                                vd_template_path::AbstractString,
                                θ::Vector{Float64},
                                need_days::Vector{Int};
                                seed::Int,
                                n_axial_roots::Int,
                                max_order1_per_plant::Int)

    tmp = joinpath(@__DIR__, "__calib_tmp_mr_1tube__")
    mkpath(tmp)

    runid  = string(seed, "_", abs(hash(θ)) % 10^9)
    outdir = joinpath(tmp, "out_seed$(seed)_$runid")
    mkpath(outdir)

    vd_cand = joinpath(outdir, "Parameter_VD_candidate.json")
    write_candidate_VD_json(vd_template_path, vd_cand, θ;
        n_axial_roots=n_axial_roots,
        max_order1_per_plant=max_order1_per_plant
    )

    env = read_json(env_json_path)
    env["species_params_files"]["VD"] = vd_cand
    env["simulation"]["seed"]         = seed
    env["simulation"]["days"]         = maximum(need_days)
    env["simulation"]["outdir"]       = outdir
    env["simulation"]["plot3d_outdir"] = joinpath(outdir, "plot3d")
    env["facilitation"]["enabled"]    = false
    env["density_regulation"]["enabled"] = false
    env["visualization_2d"]["enabled"]   = false

    mr_csv_path = joinpath(outdir, "tube_stations_daily.csv")
    env["minirhizotron"]["csv_name"]  = "tube_stations_daily.csv"

    env_path = joinpath(outdir, "Parameter_candidate.json")
    write_json_pretty(env_path, env)

    cfg = load_environment_and_plants(env_path)
    simulate!(cfg.plants, cfg.box, cfg.reg;
        days             = maximum(need_days),
        seed             = seed,
        save_every       = 0,
        outdir           = cfg.outdir,
        plot3d_dir       = cfg.plot3d_dir,
        viz_enabled      = false,
        vm_enabled       = false,
        vg_enabled       = false,
        snapshot_enabled = false,
        show_progress    = false,
        minirhizotron    = cfg.mr,
        mr_csv_path      = mr_csv_path
    )

    df = CSV.read(mr_csv_path, DataFrame)
    sort!(df, [:day, :station_idx, :azimuth_deg])

    out = Dict{Int, Vector{Float64}}()
    for sd in need_days
        sub = df[df.day .== sd, :]
        out[sd] = nrow(sub) > 0 ? Float64.(sub.length_cm) : Float64[]
    end
    return out
end

# ----------------------------
# Loss function
# SSE over (station, azimuth) × days, averaged over days and seeds
# ----------------------------
function make_loss_minirhizotron(used_days::Vector{Int},
                                 obs_by_day::Dict{Int,Vector{Float64}},
                                 env_json_path::AbstractString,
                                 vd_template_path::AbstractString;
                                 n_axial_roots::Int,
                                 max_order1_per_plant::Int)

    function loss(θ::AbstractVector{<:Real})
        θ = Float64.(θ)

        # Parameter bounds
        (θ[1] <= 0.0 || θ[1] > 1.0)  && return 1e18   # elongation_rate_min
        (θ[2] <= 0.0 || θ[2] > 1.5)  && return 1e18   # elongation_rate_max
        (θ[2] < θ[1])                  && return 1e18   # enforce rmax >= rmin
        (θ[3] < 0.05 || θ[3] > 0.4)  && return 1e18   # logistic_k
        (θ[4] < 5.0  || θ[4] > 50.0) && return 1e18   # logistic_t50
        (θ[5] < 40.0 || θ[5] > 85.0) && return 1e18   # branch_angle_order1
        (θ[6] < 0.01 || θ[6] > 0.50) && return 1e18   # g_axial
        (θ[7] < 0.05 || θ[7] > 1.0)  && return 1e18   # branching_prob_order1 (>0 prevents elimination artefact)
        (θ[8] < 0.0  || θ[8] > 1.0)  && return 1e18   # branching_prob_order2

        Js = Float64[]
        for s in SEEDS
            sim = run_candidate_stations(env_json_path, vd_template_path, θ, used_days;
                seed=s, n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)

            J = 0.0
            n_days = 0
            for d in used_days
                pred = get(sim, d, Float64[])
                obs  = get(obs_by_day, d, Float64[])
                (isempty(pred) || isempty(obs) || length(pred) != length(obs)) && continue
                # Exclude zero-zero windows: they carry no information and dominate
                # the SSE when observations are sparse (many empty station-azimuth pairs).
                nz = (obs .> 0.0) .| (pred .> 0.0)
                any(nz) || continue
                J += sum((pred[nz] .- obs[nz]).^2) / sum(nz)
                n_days += 1
            end
            push!(Js, n_days > 0 ? J / n_days : 0.0)
        end
        return mean(Js)
    end

    return loss
end

# ----------------------------
# Metrics
# ----------------------------
function regression_metrics(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    rmse = sqrt(mean((y .- x).^2))
    x̄   = mean(x);  ȳ = mean(y)
    sst  = sum((x .- x̄).^2)
    sse  = sum((y .- x).^2)
    r2   = sst == 0 ? NaN : 1.0 - sse / sst
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

# ----------------------------
# Diagnostics: obs-pred CSV
# ----------------------------
function build_obs_pred_long(used_days::Vector{Int},
                             obs_by_day::Dict{Int,Vector{Float64}},
                             meta_by_day::Dict{Int,DataFrame},
                             sim_by_day::Dict{Int,Vector{Float64}})
    rows = DataFrame(
        day=Int[], station_idx=Int[], along_tube_cm=Float64[],
        depth_cm=Float64[], azimuth_deg=Float64[],
        obs_cm=Float64[], pred_cm=Float64[]
    )
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

# ----------------------------
# Diagnostics: scatter PNG
# ----------------------------
function save_scatter_png_mr(df_long::DataFrame, outpath::AbstractString)
    Plots.gr()
    dfp = df_long[(df_long.obs_cm .> 0.0) .| (df_long.pred_cm .> 0.0), :]
    isempty(dfp) && return
    x = Float64.(dfp.obs_cm); y = Float64.(dfp.pred_cm)
    m  = regression_metrics(x, y)
    x̄  = mean(x); ȳ = mean(y)
    sxx = sum((x .- x̄).^2); sxy = sum((x .- x̄) .* (y .- ȳ))
    b = sxx == 0 ? NaN : sxy / sxx; a = ȳ - b * x̄
    lo = min(minimum(x), minimum(y)); hi = max(maximum(x), maximum(y))
    rng = max(hi - lo, 1e-6)

    p = Plots.scatter(x, y; ms=2, label=false,
        xlabel="Observed (cm)", ylabel="Predicted (cm)",
        title="Obs vs Pred — all stations × dates × azimuths",
        legend=:bottomright, legend_background_color=:transparent)
    Plots.plot!(p, [lo, hi], [lo, hi]; label="1:1")
    if !isnan(a) && !isnan(b)
        Plots.plot!(p, [lo, hi], [a+b*lo, a+b*hi]; label="OLS fit")
    end
    eq  = isnan(a) ? "" : @sprintf("OLS: y = %.3f + %.3f x", a, b)
    st  = @sprintf("R² = %.3f\nRMSE = %.3f cm\nCCC = %.3f\nKGE = %.3f\nPBIAS = %.1f%%",
                   m.r2, m.rmse, m.ccc, m.kge, m.pbias)
    x0 = lo + 0.02*rng; ytop = hi - 0.04*rng
    Plots.annotate!(p, (x0, ytop,           Plots.text(eq, 9, :left)))
    Plots.annotate!(p, (x0, ytop-0.18*rng,  Plots.text(st, 9, :left)))
    Plots.savefig(p, outpath)
end

# ----------------------------
# Diagnostics: depth profiles (one PNG per date)
# ----------------------------
function save_depth_profile_pngs(df_long::DataFrame, outdir::AbstractString)
    mkpath(outdir)
    Plots.gr()
    azimuths = sort(unique(df_long.azimuth_deg))
    colors   = Plots.distinguishable_colors(length(azimuths))

    for subday in groupby(df_long, :day)
        d = subday.day[1]

        p = Plots.plot(; xlabel="Root length (cm)", ylabel="Depth (cm)",
            title="Depth profile — day $d",
            yflip=true,
            legend=:bottomright, legend_background_color=:transparent)

        for (ci, az) in enumerate(azimuths)
            sub = subday[subday.azimuth_deg .== az, :]
            sort!(sub, :depth_cm)
            Plots.plot!(p, sub.obs_cm,  sub.depth_cm;
                label="az=$(Int(az))° obs",  color=colors[ci], linestyle=:solid)
            Plots.plot!(p, sub.pred_cm, sub.depth_cm;
                label="az=$(Int(az))° pred", color=colors[ci], linestyle=:dash)
        end

        outpng = joinpath(outdir, @sprintf("depth_profile_day_%03d.png", d))
        Plots.savefig(p, outpng)
    end
end

# ----------------------------
# Diagnostics: objective summary CSV
# ----------------------------
function save_objective_summary_mr(df_long::DataFrame, outcsv::AbstractString)
    rows = DataFrame(day=Int[], n=Int[], rmse=Float64[], r2=Float64[],
                     ccc=Float64[], kge=Float64[], pbias=Float64[])
    for subday in groupby(df_long, :day)
        d   = subday.day[1]
        dfp = subday[(subday.obs_cm .> 0.0) .| (subday.pred_cm .> 0.0), :]
        x = Float64.(dfp.obs_cm); y = Float64.(dfp.pred_cm)
        if isempty(x)
            push!(rows, (d, 0, NaN, NaN, NaN, NaN, NaN))
        else
            m = regression_metrics(x, y)
            push!(rows, (d, length(x), m.rmse, m.r2, m.ccc, m.kge, m.pbias))
        end
    end

    # Overall row
    vals(col) = [v for v in col if isfinite(v)]
    push!(rows, (-1, sum(rows.n),
        isempty(vals(rows.rmse))  ? NaN : mean(vals(rows.rmse)),
        isempty(vals(rows.r2))    ? NaN : mean(vals(rows.r2)),
        isempty(vals(rows.ccc))   ? NaN : mean(vals(rows.ccc)),
        isempty(vals(rows.kge))   ? NaN : mean(vals(rows.kge)),
        isempty(vals(rows.pbias)) ? NaN : mean(vals(rows.pbias))))
    CSV.write(outcsv, rows)
end

# ----------------------------
# Main
# ----------------------------
function main()
    obs_csv      = joinpath(@__DIR__, "data", "mr_obs_stations_1tube.csv")
    env_json     = joinpath(@__DIR__, "Parameter_calib_minirhizotron.json")
    vd_template  = joinpath(@__DIR__, "Parameter_VD_template.json")
    best_path    = joinpath(@__DIR__, "ParameterVD_best_mr_1tube.json")

    isfile(obs_csv) || error("Observation CSV not found: $obs_csv\nRun generate_mock_mr_obs_1tube.jl first (generates mr_obs_stations_1tube.csv).")

    used_days, obs_by_day, meta_by_day = load_obs_by_simday(obs_csv)
    @info "Loaded observations" days=used_days n_per_day=length(obs_by_day[used_days[1]])

    # Warm-start from best json if it exists, else from template
    src = isfile(best_path) ? best_path : vd_template
    H0  = read_json(src)
    @info "θ0 source" path=src
    θ0 = [
        Float64(get(H0, "elongation_rate_min",     0.05)),
        Float64(get(H0, "elongation_rate_max",     0.70)),
        Float64(get(H0, "elongation_logistic_k",   0.12)),
        Float64(get(H0, "elongation_logistic_t50", 20.0)),
        Float64(get(H0, "branch_angle_order1",     80.0)),
        Float64(get(H0, "g_axial",                  0.05)),
        Float64(get(H0, "branching_prob_order1",    0.20)),
        Float64(get(H0, "branching_prob_order2",    0.30)),
    ]

    best_J   = Inf
    best_θ   = nothing
    best_cfg = nothing

    for n_ax in GRID_N_AXIAL, m_o1 in GRID_MAX_ORDER1
        @info "Grid point" n_axial_roots=n_ax max_order1_per_plant=m_o1

        loss = make_loss_minirhizotron(used_days, obs_by_day, env_json, vd_template;
            n_axial_roots=n_ax, max_order1_per_plant=m_o1)

        @info "Starting CMA-ES" θ0=θ0 σ0=SIGMA0
        result = CMAEvolutionStrategy.minimize(loss, θ0, SIGMA0; maxiter=MAXITER, verbosity=1)
        θbest  = CMAEvolutionStrategy.xbest(result)
        Jbest  = CMAEvolutionStrategy.fbest(result)

        @info "Grid result" n_axial_roots=n_ax max_order1_per_plant=m_o1 J=Jbest

        if Jbest < best_J
            best_J   = Jbest
            best_θ   = θbest
            best_cfg = (n_axial_roots=n_ax, max_order1_per_plant=m_o1)
        end
    end

    @info "Best overall" J=best_J cfg=best_cfg θ=best_θ

    # Save best parameters
    write_candidate_VD_json(vd_template, best_path, best_θ;
        n_axial_roots=best_cfg.n_axial_roots, max_order1_per_plant=best_cfg.max_order1_per_plant)
    # Annotate with calibration metadata so validate_comparison.jl can read them
    d_best = read_json(best_path)
    d_best["calibration_J"]     = best_J
    d_best["calibration_seeds"] = collect(SEEDS)
    write_json_pretty(best_path, d_best)
    @info "Saved best parameters" path=best_path

    # Diagnostics run (deterministic seed)
    @info "Running diagnostics simulation..."
    sim_best = run_candidate_stations(env_json, vd_template, best_θ, used_days;
        seed=42, n_axial_roots=best_cfg.n_axial_roots,
        max_order1_per_plant=best_cfg.max_order1_per_plant)

    df_long  = build_obs_pred_long(used_days, obs_by_day, meta_by_day, sim_best)
    diagdir  = joinpath(@__DIR__, "diagnostics_mr", "1tube")
    mkpath(diagdir)

    CSV.write(joinpath(diagdir, "best_obs_pred_stations.csv"), df_long)
    save_scatter_png_mr(df_long, joinpath(diagdir, "best_obs_vs_pred_ALL.png"))
    save_depth_profile_pngs(df_long, diagdir)
    save_objective_summary_mr(df_long, joinpath(diagdir, "best_objective_summary_mr.csv"))

    @info "Saved diagnostics" dir=diagdir

    @printf("\nBest θ:\n")
    @printf("  elongation_rate_min       = %.4f\n", best_θ[1])
    @printf("  elongation_rate_max       = %.4f\n", best_θ[2])
    @printf("  elongation_logistic_k     = %.4f\n", best_θ[3])
    @printf("  elongation_logistic_t50   = %.4f\n", best_θ[4])
    @printf("  branch_angle_order1       = %.4f\n", best_θ[5])
    @printf("  g_axial                   = %.4f\n", best_θ[6])
    @printf("  branching_prob_order1     = %.4f\n", best_θ[7])
    @printf("  branching_prob_order2     = %.4f\n", best_θ[8])
    @printf("  n_axial_roots             = %d\n",   best_cfg.n_axial_roots)
    @printf("  max_order1_per_plant      = %d\n",   best_cfg.max_order1_per_plant)
end

main()
