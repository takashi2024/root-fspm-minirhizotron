# summarize_4param_results.jl
#
# Reads all run_summary.json + best_params.json from experiments/
# that used 4-param calibration, and prints grouped comparison tables:
#   Group A: 200cm box, 1-tube, length-only loss
#   Group B: 100cm PBC box, 1-tube, length + angle-histogram loss
#   Group C: 200cm box, 1-tube, length + angle-histogram loss (partial)
#
# Within each group, only the latest run per objective mode is shown.
#
# Run: julia calibration/summarize_4param_results.jl  (from repo root)

using JSON, Printf, Dates, Statistics

read_json(p) = open(p, "r") do io; JSON.parse(String(read(io))); end

const CALIB_PARAMS = ["branch_angle_order1", "g_axial",
                      "branching_prob_order1", "branching_prob_order2"]

const TRUE_PARAMS = Dict(
    "branch_angle_order1"   => 80.0,
    "g_axial"               => 0.05,
    "branching_prob_order1" => 0.20,
    "branching_prob_order2" => 0.30,
)

const PARAM_SHORT = Dict(
    "branch_angle_order1"   => "ba_o1",
    "g_axial"               => "g_axial",
    "branching_prob_order1" => "bp_o1",
    "branching_prob_order2" => "bp_o2",
)

exp_base = joinpath(@__DIR__, "experiments")
isdir(exp_base) || error("experiments/ directory not found")

function classify_group(config_file::String)
    cfg = lowercase(config_file)
    if occursin("100cm_pbc", cfg) && occursin("angle_hist", cfg)
        return "B: 100cm_pbc+angle_hist"
    elseif occursin("angle_hist", cfg)
        return "C: 200cm+angle_hist"
    else
        return "A: 200cm_length_only"
    end
end

# Collect all 4-param runs
all_runs = []
for dir in readdir(exp_base; join=true)
    isdir(dir) || continue
    rs_path = joinpath(dir, "run_summary.json")
    bp_path = joinpath(dir, "best_params.json")
    isfile(rs_path) || continue
    rs = read_json(rs_path)
    calib_ps = get(rs, "calibration_params", nothing)
    calib_ps === nothing && continue
    Set(String.(calib_ps)) == Set(CALIB_PARAMS) || continue
    rs["_dir"]         = dir
    rs["_dirname"]     = basename(dir)
    rs["_best_params"] = isfile(bp_path) ? read_json(bp_path) : Dict()
    rs["_group"]       = classify_group(get(rs, "config_file", ""))
    push!(all_runs, rs)
end

isempty(all_runs) && (println("No 4-param runs found in experiments/"); exit(0))

# Keep only the latest run per (group, objective)
latest = Dict{Tuple{String,String}, Any}()
for r in all_runs
    key = (r["_group"], get(r, "objective", "?"))
    if !haskey(latest, key) || r["_dirname"] > latest[key]["_dirname"]
        latest[key] = r
    end
end

runs_by_group = Dict{String, Vector{Any}}()
for ((_grp, _obj), r) in latest
    g = r["_group"]
    haskey(runs_by_group, g) || (runs_by_group[g] = [])
    push!(runs_by_group[g], r)
end
for v in values(runs_by_group)
    sort!(v, by = r -> get(r, "objective", ""))
end

all_objectives = sort(unique(get(r, "objective", "?") for r in values(latest)))
groups = sort(collect(keys(runs_by_group)))

# ── Helper: print one table ───────────────────────────────────────────────────
function print_metrics_table(group_runs, group_label)
    has_multi = any(haskey(r, "diag_r2_multiseed") for r in group_runs)
    W = has_multi ? 140 : 100
    println("\n", repeat('═', W))
    println("  $group_label")
    println(repeat('═', W))
    if has_multi
        @printf("  %-40s  %8s  %8s  %10s  %10s  %11s  %9s\n",
                "Objective", "best_J", "R²(s42)", "R²(multi)", "RMSE(s42)", "RMSE(multi)", "PBIAS(%)")
    else
        @printf("  %-40s  %8s  %8s  %9s  %9s\n",
                "Objective", "best_J", "R²", "RMSE(cm)", "PBIAS(%)")
    end
    println(repeat('─', W))
    for r in group_runs
        obj   = get(r, "objective",  "?")
        bestJ = get(r, "best_J",     NaN)
        r2    = get(r, "diag_r2",    NaN)
        rmse  = get(r, "diag_rmse",  NaN)
        pbias = get(r, "diag_pbias", NaN)
        if has_multi
            r2m   = get(r, "diag_r2_multiseed",   NaN)
            rmsem = get(r, "diag_rmse_multiseed",  NaN)
            @printf("  %-40s  %8.4f  %8.3f  %10.3f  %10.3f  %11.3f  %9.1f\n",
                    obj, bestJ, r2, r2m, rmse, rmsem, pbias)
        else
            @printf("  %-40s  %8.4f  %8.3f  %9.3f  %9.1f\n", obj, bestJ, r2, rmse, pbias)
        end
    end
    println(repeat('─', W))
end

function print_params_table(group_runs, group_label)
    W = 115
    println("\n", repeat('═', W))
    println("  $group_label — parameter recovery")
    println(repeat('═', W))
    let parts = [@sprintf("  %-40s", "Objective")]
        for p in CALIB_PARAMS
            push!(parts, @sprintf("  %17s", PARAM_SHORT[p]))
        end
        println(join(parts))
    end
    println(repeat('─', W))
    let parts = [@sprintf("  %-40s", "── TRUE (template) ──")]
        for p in CALIB_PARAMS
            push!(parts, @sprintf("  %17.4f", TRUE_PARAMS[p]))
        end
        println(join(parts))
    end
    println(repeat('─', W))
    for r in group_runs
        obj = get(r, "objective", "?")
        bp  = r["_best_params"]
        let parts = [@sprintf("  %-40s", obj)]
            for p in CALIB_PARAMS
                val = get(bp, p, NaN)
                err_pct = isnan(val) ? NaN : 100.0 * (val - TRUE_PARAMS[p]) / TRUE_PARAMS[p]
                push!(parts, @sprintf("  %8.4f(%+6.1f%%)", val, err_pct))
            end
            println(join(parts))
        end
    end
    println(repeat('─', W))
end

# ── Print all groups ──────────────────────────────────────────────────────────
for g in groups
    grs = runs_by_group[g]
    print_metrics_table(grs, g)
end

for g in groups
    grs = runs_by_group[g]
    print_params_table(grs, g)
end

# ── Side-by-side R² comparison across groups for shared objectives ────────────
has_multi_any = any(haskey(r, "diag_r2_multiseed") for r in values(latest))
println("\n", repeat('═', 130))
if has_multi_any
    println("  SIDE-BY-SIDE R² COMPARISON  [s42 = seed-42 only | ms = multi-seed avg]")
else
    println("  SIDE-BY-SIDE R² COMPARISON (seed=42 diagnostic)")
end
println(repeat('═', 130))
shared_objs = sort(collect(all_objectives))
gcols = groups
let parts = [@sprintf("  %-38s", "Objective")]
    for g in gcols
        if has_multi_any
            push!(parts, @sprintf("  %13s %13s", g[1:min(end,12)]*"(s42)", g[1:min(end,12)]*"(ms)"))
        else
            push!(parts, @sprintf("  %22s", g))
        end
    end
    println(join(parts))
end
println(repeat('─', 130))
for obj in shared_objs
    let parts = [@sprintf("  %-38s", obj)]
        for g in gcols
            key = (g, obj)
            if haskey(latest, key)
                r = latest[key]
                r2   = get(r, "diag_r2", NaN)
                r2ms = get(r, "diag_r2_multiseed", NaN)
                if has_multi_any
                    s42_str = isnan(r2)   ? "      —" : @sprintf("%+7.3f", r2)
                    ms_str  = isnan(r2ms) ? "      —" : @sprintf("%+7.3f", r2ms)
                    push!(parts, @sprintf("  %13s %13s", s42_str, ms_str))
                else
                    push!(parts, @sprintf("  %22.3f", r2))
                end
            else
                if has_multi_any
                    push!(parts, @sprintf("  %13s %13s", "—", "—"))
                else
                    push!(parts, @sprintf("  %22s", "—"))
                end
            end
        end
        println(join(parts))
    end
end
println(repeat('─', 130))

# ── Mean |% error| summary ───────────────────────────────────────────────────
println("\n", repeat('─', 70))
println("  Mean |% error| across 4 params (parameter recovery score):")
println(repeat('─', 70))
@printf("  %-40s  %-25s  %8s\n", "Objective", "Group", "Mean|%err|")
println(repeat('─', 70))
scored = []
for ((g, obj), r) in latest
    bp   = r["_best_params"]
    errs = [abs(100.0 * (get(bp, p, NaN) - TRUE_PARAMS[p]) / TRUE_PARAMS[p])
            for p in CALIB_PARAMS]
    valid = filter(!isnan, errs)
    isempty(valid) && continue
    push!(scored, (obj, g, mean(valid)))
end
sort!(scored, by=x->(x[2], x[3]))
for (obj, g, mae) in scored
    @printf("  %-40s  %-25s  %8.1f%%\n", obj, g, mae)
end
println(repeat('─', 70))
println()
