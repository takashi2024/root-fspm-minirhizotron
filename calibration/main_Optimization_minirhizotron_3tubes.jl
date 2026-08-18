# main_Optimization_minirhizotron_3tubes.jl
#
# CMA-ES calibration of virtual dicot root parameters against minirhizotron
# observations from THREE tube replicates installed at x = 25, 50, 75 cm.
#
# Motivation: a single tube samples <1% of soil volume, so which roots happen
# to intersect the annulus is highly seed-dependent (stochastic).  Running
# three spatially-separated replicate tubes and averaging their signals before
# computing the loss dramatically reduces this noise, enabling the optimizer to
# see the true parameter gradient rather than spatial-sampling variance.
#
# Strategy: for each candidate θ evaluation, ONE simulation is run per seed;
# all 3 tubes sample the same root realisation, matching the real installation.
# Loss = mean over seeds of mean over tubes of SSE/n_days.
#
# Inputs:
#   calibration/data/mr_obs_stations_3tubes.csv  — per-tube DL root lengths
#   calibration/Parameter_calib_minirhizotron.json   (base env — tube origin patched)
#   calibration/Parameter_VD_template.json
#
# Outputs:
#   calibration/ParameterVD_best_mr_3tubes.json
#   calibration/diagnostics_mr/3tubes/
#
# Run from repo root:
#   julia calibration/main_Optimization_minirhizotron_3tubes.jl

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
const SOWING_DATE     = Date(2025, 4, 1)
const SEEDS           = (11, 22, 99, 88)   # 4 seeds; excludes obs seed (42) and validation seeds (33,44,55,66,77)
const MAXITER         = 100
const SIGMA0          = 0.1
const GRID_N_AXIAL    = [1]
const GRID_MAX_ORDER1 = [20]               # fixed to match 1-tube — removes discrete trapping of optimizer

# Tube x-positions (cm from left wall).  All at y=5 cm, same angle as the
# single-tube config.  Each is centred between a row pair in the conventional
# layout (12.5 cm row spacing), giving 3.25 cm clearance on each side.
const TUBE_X_CM = [25.0, 50.0, 75.0]

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
read_json(path::AbstractString) = open(path, "r") do io; JSON.parse(String(read(io))); end
function write_json_pretty(path::AbstractString, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

# ----------------------------
# Date / simday helper
# ----------------------------
function date_to_simday(d::Date)
    return Dates.value(d - SOWING_DATE) + 1
end

# ----------------------------
# Observation loader (3-tube)
# ----------------------------
# Returns:
#   used_days       :: Vector{Int}
#   obs_by_tube_day :: Dict{Int, Dict{Int, Vector{Float64}}}
#                      tube_id => (sim_day => root_length vector)
#                      Each vector is sorted by (station_idx ASC, azimuth_deg ASC).
#   meta_by_day     :: Dict{Int, DataFrame}  — from tube 1 (for diagnostics coords)
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

    used_days = sort(collect(all_days))
    return used_days, obs_by_tube_day, meta_by_day
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

# ----------------------------
# Run ONE simulation; all 3 tubes sample the same root realisation.
# Returns Dict{Int, Dict{Int, Vector{Float64}}}:
#   tube_id => (sim_day => station×azimuth vector, station-major order)
# ----------------------------
function run_candidate_stations_3tubes(env_json_path::AbstractString,
                                       vd_template_path::AbstractString,
                                       θ::Vector{Float64},
                                       need_days::Vector{Int};
                                       seed::Int,
                                       n_axial_roots::Int,
                                       max_order1_per_plant::Int)

    tmp    = joinpath(@__DIR__, "__calib_tmp_mr_3tubes__")
    mkpath(tmp)
    runid  = string("s", seed, "_", abs(hash(θ)) % 10^9)
    outdir = joinpath(tmp, "out_$(runid)")
    mkpath(outdir)

    vd_cand = joinpath(outdir, "Parameter_VD_candidate.json")
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

    env_path = joinpath(outdir, "Parameter_candidate.json")
    write_json_pretty(env_path, env)

    cfg      = load_environment_and_plants(env_path)
    plants   = cfg.plants
    box      = cfg.box
    reg      = cfg.reg
    tubes    = [make_tube(tx) for tx in TUBE_X_CM]
    need_set = Set(need_days)
    results  = Dict{Int, Dict{Int, Vector{Float64}}}(
        tid => Dict{Int, Vector{Float64}}() for tid in 1:length(TUBE_X_CM))

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

# ----------------------------
# Loss function (3-tube)
# Loss = mean over seeds of { mean over tubes of { SSE(pred, obs) / n_days } }
# ----------------------------
function make_loss_minirhizotron_3tubes(used_days::Vector{Int},
                                        obs_by_tube_day::Dict{Int,Dict{Int,Vector{Float64}}},
                                        env_json_path::AbstractString,
                                        vd_template_path::AbstractString;
                                        n_axial_roots::Int,
                                        max_order1_per_plant::Int)
    n_tubes = length(TUBE_X_CM)

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
        (θ[7] < 0.05 || θ[7] > 1.0)  && return 1e18   # branching_prob_order1
        (θ[8] < 0.0  || θ[8] > 1.0)  && return 1e18   # branching_prob_order2

        Js_seed = Vector{Float64}(undef, length(SEEDS))
        Threads.@threads for i in eachindex(SEEDS)
            sim_by_tube = run_candidate_stations_3tubes(
                env_json_path, vd_template_path, θ, used_days;
                seed=SEEDS[i], n_axial_roots=n_axial_roots, max_order1_per_plant=max_order1_per_plant)

            Js_tube = Float64[]
            for tid in 1:n_tubes
                obs_day = get(obs_by_tube_day, tid, Dict{Int,Vector{Float64}}())
                sim_day = get(sim_by_tube,     tid, Dict{Int,Vector{Float64}}())

                J      = 0.0
                n_days = 0
                for d in used_days
                    pred = get(sim_day, d, Float64[])
                    obs  = get(obs_day, d, Float64[])
                    (isempty(pred) || isempty(obs) || length(pred) != length(obs)) && continue
                    # Exclude zero-zero windows: they carry no information and dominate
                    # the SSE when observations are sparse (many empty station-azimuth pairs).
                    nz = (obs .> 0.0) .| (pred .> 0.0)
                    any(nz) || continue
                    J += sum((pred[nz] .- obs[nz]).^2) / sum(nz)
                    n_days += 1
                end
                push!(Js_tube, n_days > 0 ? J / n_days : 0.0)
            end
            Js_seed[i] = mean(Js_tube)
        end
        return mean(Js_seed)
    end

    return loss
end

# ----------------------------
# Metrics
# ----------------------------
function regression_metrics(x::Vector{Float64}, y::Vector{Float64})
    n    = length(x)
    rmse = sqrt(mean((y .- x).^2))
    x̄    = mean(x); ȳ = mean(y)
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
# Build long-format obs-pred table (all tubes combined)
# ----------------------------
function build_obs_pred_long_3tubes(used_days::Vector{Int},
                                    obs_by_tube_day::Dict{Int,Dict{Int,Vector{Float64}}},
                                    meta_by_day::Dict{Int,DataFrame},
                                    sim_by_tube::Dict{Int,Dict{Int,Vector{Float64}}})
    rows = DataFrame(
        tube_id=Int[], day=Int[],
        station_idx=Int[], along_tube_cm=Float64[],
        depth_cm=Float64[], azimuth_deg=Float64[],
        obs_cm=Float64[], pred_cm=Float64[]
    )
    for tid in sort(collect(keys(obs_by_tube_day)))
        obs_day = obs_by_tube_day[tid]
        sim_day = get(sim_by_tube, tid, Dict{Int,Vector{Float64}}())
        for d in used_days
            obs  = get(obs_day, d, Float64[])
            pred = get(sim_day, d, Float64[])
            meta = get(meta_by_day, d, DataFrame())
            (isempty(obs) || isempty(pred) || length(obs) != length(pred)) && continue
            for i in 1:length(obs)
                push!(rows, (
                    tid, d,
                    Int(meta.station_idx[i]), Float64(meta.along_tube_cm[i]),
                    Float64(meta.depth_cm[i]), Float64(meta.azimuth_deg[i]),
                    obs[i], pred[i]
                ))
            end
        end
    end
    return rows
end

# ----------------------------
# Scatter PNG (all tubes, colour-coded by tube_id)
# ----------------------------
function save_scatter_png_mr_3tubes(df_long::DataFrame, outpath::AbstractString)
    Plots.gr()
    dfp = df_long[(df_long.obs_cm .> 0.0) .| (df_long.pred_cm .> 0.0), :]
    isempty(dfp) && return

    colors = [:blue, :red, :green]
    labels = ["Tube 1 (x=25)", "Tube 2 (x=50)", "Tube 3 (x=75)"]
    x_all  = Float64.(dfp.obs_cm); y_all = Float64.(dfp.pred_cm)
    m      = regression_metrics(x_all, y_all)
    lo = min(minimum(x_all), minimum(y_all))
    hi = max(maximum(x_all), maximum(y_all))
    rng = max(hi - lo, 1e-6)

    p = Plots.plot(; xlabel="Observed (cm)", ylabel="Predicted (cm)",
        title="Obs vs Pred — 3 tubes × all stations × dates × azimuths",
        legend=:bottomright, legend_background_color=:transparent)

    for (tid, col, lab) in zip(1:3, colors, labels)
        dft = dfp[dfp.tube_id .== tid, :]
        isempty(dft) && continue
        Plots.scatter!(p, Float64.(dft.obs_cm), Float64.(dft.pred_cm);
            ms=2, label=lab, color=col, alpha=0.6)
    end

    Plots.plot!(p, [lo, hi], [lo, hi]; label="1:1", color=:black, linestyle=:solid)

    x̄ = mean(x_all); ȳ = mean(y_all)
    sxx = sum((x_all .- x̄).^2); sxy = sum((x_all .- x̄) .* (y_all .- ȳ))
    b = sxx == 0 ? NaN : sxy / sxx; a = ȳ - b * x̄
    if !isnan(a) && !isnan(b)
        Plots.plot!(p, [lo, hi], [a+b*lo, a+b*hi]; label="OLS fit", color=:black, linestyle=:dash)
    end

    eq  = isnan(a) ? "" : @sprintf("OLS: y = %.3f + %.3f x", a, b)
    st  = @sprintf("R² = %.3f\nRMSE = %.3f cm\nCCC = %.3f\nKGE = %.3f\nPBIAS = %.1f%%",
                   m.r2, m.rmse, m.ccc, m.kge, m.pbias)
    x0 = lo + 0.02*rng; ytop = hi - 0.04*rng
    Plots.annotate!(p, (x0, ytop,          Plots.text(eq, 8, :left)))
    Plots.annotate!(p, (x0, ytop-0.18*rng, Plots.text(st, 8, :left)))
    Plots.savefig(p, outpath)
end

# ----------------------------
# Depth profiles — one PNG per date, 3 subplots (one per tube)
# ----------------------------
function save_depth_profile_pngs_3tubes(df_long::DataFrame, outdir::AbstractString)
    mkpath(outdir)
    Plots.gr()
    azimuths = sort(unique(df_long.azimuth_deg))
    colors   = Plots.distinguishable_colors(length(azimuths))

    for sub_day in groupby(df_long, :day)
        d = sub_day.day[1]
        plots = []
        for tid in 1:3
            sub_t = sub_day[sub_day.tube_id .== tid, :]
            pt = Plots.plot(; xlabel="Root length (cm)", ylabel="Depth (cm)",
                title="Tube $tid (x=$(Int(TUBE_X_CM[tid]))cm) — day $d",
                yflip=true, legend=:bottomright, legend_background_color=:transparent)
            for (ci, az) in enumerate(azimuths)
                sub = sub_t[sub_t.azimuth_deg .== az, :]
                isempty(sub) && continue
                sort!(sub, :depth_cm)
                Plots.plot!(pt, sub.obs_cm,  sub.depth_cm;
                    label="az=$(Int(az))° obs",  color=colors[ci], linestyle=:solid)
                Plots.plot!(pt, sub.pred_cm, sub.depth_cm;
                    label="az=$(Int(az))° pred", color=colors[ci], linestyle=:dash)
            end
            push!(plots, pt)
        end
        p = Plots.plot(plots...; layout=(1, 3), size=(1500, 600))
        outpng = joinpath(outdir, @sprintf("depth_profile_day_%03d.png", d))
        Plots.savefig(p, outpng)
    end
end

# ----------------------------
# Objective summary CSV (per day, averaged over tubes)
# ----------------------------
function save_objective_summary_mr_3tubes(df_long::DataFrame, outcsv::AbstractString)
    rows = DataFrame(day=Int[], n_total=Int[], rmse=Float64[], r2=Float64[],
                     ccc=Float64[], kge=Float64[], pbias=Float64[])
    for sub_day in groupby(df_long, :day)
        d   = sub_day.day[1]
        dfp = sub_day[(sub_day.obs_cm .> 0.0) .| (sub_day.pred_cm .> 0.0), :]
        x = Float64.(dfp.obs_cm); y = Float64.(dfp.pred_cm)
        if isempty(x)
            push!(rows, (d, 0, NaN, NaN, NaN, NaN, NaN))
        else
            m = regression_metrics(x, y)
            push!(rows, (d, length(x), m.rmse, m.r2, m.ccc, m.kge, m.pbias))
        end
    end
    vals(col) = [v for v in col if isfinite(v)]
    push!(rows, (-1, sum(rows.n_total),
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
    obs_csv     = joinpath(@__DIR__, "data", "mr_obs_stations_3tubes.csv")
    env_json    = joinpath(@__DIR__, "Parameter_calib_minirhizotron.json")
    vd_template = joinpath(@__DIR__, "Parameter_VD_template.json")
    best_path   = joinpath(@__DIR__, "ParameterVD_best_mr_3tubes.json")

    isfile(obs_csv) || error("3-tube observation CSV not found: $obs_csv\nRun generate_mock_mr_obs_3tubes.jl first.")

    used_days, obs_by_tube_day, meta_by_day = load_obs_by_simday_3tubes(obs_csv)
    @info "Loaded 3-tube observations" days=used_days n_tubes=length(TUBE_X_CM)
    for tid in 1:length(TUBE_X_CM)
        first_day = used_days[1]
        n = length(get(get(obs_by_tube_day, tid, Dict()), first_day, Float64[]))
        @info "  Tube $tid" x_cm=TUBE_X_CM[tid] n_per_day=n
    end

    # Warm-start: use 3-tube best if exists, then single-tube best, then template
    src = if isfile(best_path)
        best_path
    elseif isfile(joinpath(@__DIR__, "ParameterVD_best_mr_1tube.json"))
        joinpath(@__DIR__, "ParameterVD_best_mr_1tube.json")
    else
        vd_template
    end
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

        loss = make_loss_minirhizotron_3tubes(
            used_days, obs_by_tube_day, env_json, vd_template;
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

    write_candidate_VD_json(vd_template, best_path, best_θ;
        n_axial_roots=best_cfg.n_axial_roots, max_order1_per_plant=best_cfg.max_order1_per_plant)
    # Annotate with calibration metadata so validate_comparison.jl can read them
    d_best = read_json(best_path)
    d_best["calibration_J"]     = best_J
    d_best["calibration_seeds"] = collect(SEEDS)
    write_json_pretty(best_path, d_best)
    @info "Saved best parameters" path=best_path

    # Diagnostics run (deterministic seed)
    @info "Running diagnostics simulations (3 tubes, seed=42)..."
    sim_by_tube_diag = run_candidate_stations_3tubes(
        env_json, vd_template, best_θ, used_days;
        seed=42, n_axial_roots=best_cfg.n_axial_roots,
        max_order1_per_plant=best_cfg.max_order1_per_plant)

    df_long  = build_obs_pred_long_3tubes(
        used_days, obs_by_tube_day, meta_by_day, sim_by_tube_diag)
    diagdir  = joinpath(@__DIR__, "diagnostics_mr", "3tubes")
    mkpath(diagdir)

    CSV.write(joinpath(diagdir, "best_obs_pred_stations.csv"), df_long)
    save_scatter_png_mr_3tubes(df_long, joinpath(diagdir, "best_obs_vs_pred_ALL.png"))
    save_depth_profile_pngs_3tubes(df_long, diagdir)
    save_objective_summary_mr_3tubes(df_long, joinpath(diagdir, "best_objective_summary_mr_3tubes.csv"))

    @info "Saved diagnostics" dir=diagdir

    @printf("\nBest θ (3-tube calibration):\n")
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
