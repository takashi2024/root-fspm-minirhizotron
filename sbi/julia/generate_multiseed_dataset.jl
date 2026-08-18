#!/usr/bin/env julia
# generate_multiseed_dataset.jl
#
# Generates K independent seed realizations for each non-zero-root sample in an
# existing SBI config, enabling multi-tube / multi-seed aggregation strategies.
#
# Filtering: samples where has_visible_roots == 1 in the existing samples.csv.
#
# Seeds: fixed set [7001..7001+K-1], independent of per-config seeds_start ranges
#        (existing SBI: 1000–6500; CMA-ES: ≤99 → no conflicts).
#
# Outputs (under sbi/data/multi_seed/config_<tag>/):
#   samples_filtered.csv            — filtered parameter table (one row per case)
#   tensors/
#     X_raw.npz   (N, K, T, ns, na, 3) float32 — raw per-seed summary tensors
#     X_mean.npz  (N, T, ns, na, 3)    float32 — mean across seeds
#     X_std.npz   (N, T, ns, na, 3)    float32 — std across seeds
#     y.npz       (N, 8)               float64 — true parameter values
#     param_names.txt
#     seed_list.txt                            — the K seeds used
#   per_sample/
#     s{sample_id}.npy  (K, T, ns, na, 3)     — per-sample checkpoint (resume support)
#
# The per_sample/ checkpoints are preserved so assembly can be re-run without
# re-simulating. Delete per_sample/ manually after confirming the NPZ files.
#
# Usage (from repo root):
#   julia --threads 8 sbi/julia/generate_multiseed_dataset.jl <sbi_config.json> [K]
#
# K defaults to 7. Passing K=1 reproduces the original single-seed format.

using JSON, Printf, NPZ, CSV, DataFrames, Statistics
using Base.Threads

const REPO_ROOT = dirname(dirname(dirname(abspath(@__FILE__))))

for f in ["Deps", "Types", "Geometry", "DensityRegulation",
          "GrowthStep", "MetricsAndOutput", "Config"]
    include(joinpath(REPO_ROOT, "src", "$(f).jl"))
end
include(joinpath(@__DIR__, "FovRenderer.jl"))
include(joinpath(@__DIR__, "PriorSampler.jl"))
include(joinpath(@__DIR__, "SimRunner.jl"))

# Fixed seeds for all multi-seed runs — offset far above all existing SBI ranges
const MULTISEED_BASE = 7001

# ── Config loading ─────────────────────────────────────────────────────────────

function load_cfg(config_path::String)
    abs_path = isabspath(config_path) ? config_path : joinpath(REPO_ROOT, config_path)
    open(abs_path, "r") do io; JSON.parse(String(read(io))); end
end

# ── Run one sample with a specific seed override ───────────────────────────────

function run_one_sample_seed(
    sample_id::Int,
    params::NamedTuple,
    seed::Int,
    base_env_path::String,
    vd_template_path::String,
    tmp_dir::String;
    days::Int,
    snapshot_days::Vector{Int},
    img_height::Int,
    img_width::Int,
    render_step_cm::Float64
)
    mkpath(tmp_dir)
    sp_path  = joinpath(tmp_dir, @sprintf("sp_%04d_s%07d.json", sample_id, seed))
    env_path = joinpath(tmp_dir, @sprintf("env_%04d_s%07d.json", sample_id, seed))

    _write_candidate_species_json(vd_template_path, params, sp_path)
    _write_candidate_env_json(base_env_path, sp_path, seed, days, tmp_dir, env_path)

    cfg = load_environment_and_plants(env_path)
    mr  = cfg.mr
    @assert mr.enabled "Minirhizotron must be enabled in base env"

    ns = length(mr.station_positions_cm)
    na = length(mr.camera_azimuths_deg)
    T  = length(snapshot_days)

    summary = zeros(Float32, T, ns, na, 3)
    Random.seed!(seed)
    snap_idx = 1
    R_dummy  = zeros(Float64, 1, 1)

    for day in 1:days
        for pl in cfg.plants
            step_plant!(pl, cfg.box, cfg.reg, R_dummy, R_dummy, 1, 1;
                        mature_branching=false, facilitation=nothing,
                        facil_gates=nothing, minirhizotron=nothing)
        end
        if snap_idx <= T && day == snapshot_days[snap_idx]
            summary[snap_idx, :, :, :] = _extract_summary(cfg.plants, mr)
            snap_idx += 1
        end
    end

    rm(sp_path,  force=true)
    rm(env_path, force=true)

    return summary   # (T, ns, na, 3)
end

# ── Main ───────────────────────────────────────────────────────────────────────

function main(config_path::String, K::Int)
    cfg = load_cfg(config_path)

    base_env_path    = joinpath(REPO_ROOT, cfg["base_env_json"])
    vd_template_path = joinpath(REPO_ROOT, cfg["vd_template_json"])
    src_outdir       = joinpath(REPO_ROOT, cfg["outdir"])
    days             = Int(cfg["days"])
    snapshot_days    = Int.(cfg["snapshot_days"])
    img_height       = Int(cfg["img_height"])
    img_width        = Int(cfg["img_width"])
    render_step_cm   = Float64(cfg["render_step_cm"])
    n_axial_roots    = Int(cfg["n_axial_roots"])
    max_order1       = Int(cfg["max_order1_per_plant"])

    # Derive config tag from outdir name (e.g. "sbi/data/config_A" → "A")
    cfg_tag = replace(basename(src_outdir), "config_" => "")

    outdir = if haskey(cfg, "multi_seed_outdir")
        joinpath(REPO_ROOT, cfg["multi_seed_outdir"])
    else
        joinpath(REPO_ROOT, "sbi", "data", "multi_seed", "config_$(cfg_tag)")
    end
    tensors_dir   = joinpath(outdir, "tensors")
    per_sample_dir = joinpath(outdir, "per_sample")
    tmp_root      = joinpath(outdir, "_tmp_")
    mkpath(tensors_dir)
    mkpath(per_sample_dir)
    mkpath(tmp_root)

    multi_seeds = collect(MULTISEED_BASE : MULTISEED_BASE + K - 1)

    println("=" ^ 65)
    println("Multi-seed Dataset Generator")
    @printf("  Config tag   : %s\n", cfg_tag)
    @printf("  K seeds      : %d  %s\n", K, multi_seeds)
    @printf("  Threads      : %d\n", nthreads())
    @printf("  Source dir   : %s\n", src_outdir)
    @printf("  Output dir   : %s\n", outdir)
    println("=" ^ 65)

    # ── 1. Load and filter samples ─────────────────────────────────────────────

    samples_src = joinpath(src_outdir, "samples.csv")
    df_all = CSV.read(samples_src, DataFrame)
    df = filter(r -> coalesce(r.has_visible_roots, 0.0) == 1.0, df_all)
    N  = nrow(df)
    @printf("\nFiltered: %d / %d samples have visible roots\n\n", N, nrow(df_all))

    # Infer ns, na from base env
    base_env = _read_json(base_env_path)
    ns = length(base_env["minirhizotron"]["station_positions_cm"])
    na = length(base_env["minirhizotron"]["camera_azimuths_deg"])
    T  = length(snapshot_days)

    # ── 2. Run simulations (threaded over samples) ─────────────────────────────

    println("[2/3] Running simulations  (K=$K seeds × N=$N samples × T=$T snapshots)...")
    println("      Per-sample checkpoints → $per_sample_dir")
    println()

    results = Vector{Array{Float32,5}}(undef, N)   # each: (K, T, ns, na, 3)
    progress = Threads.Atomic{Int}(0)

    @threads for i in 1:N
        row       = df[i, :]
        sample_id = Int(row.sample_id)
        chk_path  = joinpath(per_sample_dir, @sprintf("s%04d.npy", sample_id))

        if isfile(chk_path)
            raw = NPZ.npzread(chk_path)   # (K, T, ns, na, 3)
            results[i] = raw
            Threads.atomic_add!(progress, 1)
            @printf("  [%d/%d] s%04d SKIPPED (checkpoint exists)\n",
                    progress[], N, sample_id)
            continue
        end

        params = (;
            elongation_rate_min     = Float64(row.elongation_rate_min),
            elongation_rate_max     = Float64(row.elongation_rate_max),
            elongation_logistic_k   = Float64(row.elongation_logistic_k),
            elongation_logistic_t50 = Float64(row.elongation_logistic_t50),
            branch_angle_order1     = Float64(row.branch_angle_order1),
            g_axial                 = Float64(row.g_axial),
            branching_prob_order1   = Float64(row.branching_prob_order1),
            branching_prob_order2   = Float64(row.branching_prob_order2),
            n_axial_roots           = Int(row.n_axial_roots),
            max_order1_per_plant    = Int(row.max_order1_per_plant),
        )

        tmp_dir = joinpath(tmp_root, "s$(sample_id)")
        mkpath(tmp_dir)

        raw = zeros(Float32, K, T, ns, na, 3)
        for (ki, seed) in enumerate(multi_seeds)
            summ = run_one_sample_seed(
                sample_id, params, seed,
                base_env_path, vd_template_path, tmp_dir;
                days=days, snapshot_days=snapshot_days,
                img_height=img_height, img_width=img_width,
                render_step_cm=render_step_cm
            )
            raw[ki, :, :, :, :] = summ
        end

        rm(tmp_dir; recursive=true, force=true)

        # Write per-sample checkpoint
        NPZ.npzwrite(chk_path, raw)
        results[i] = raw

        n_done = Threads.atomic_add!(progress, 1) + 1
        @printf("  [%d/%d] s%04d done\n", n_done, N, sample_id)
    end

    rm(tmp_root; recursive=true, force=true)

    # ── 3. Assemble and save tensors ───────────────────────────────────────────

    println("\n[3/3] Assembling tensors...")

    X_raw  = zeros(Float32, N, K, T, ns, na, 3)
    X_mean = zeros(Float32, N, T, ns, na, 3)
    X_std  = zeros(Float32, N, T, ns, na, 3)
    y      = zeros(Float64, N, length(SBI_PARAM_NAMES))

    for i in 1:N
        X_raw[i, :, :, :, :, :] = results[i]
        for t in 1:T, s in 1:ns, a in 1:na, c in 1:3
            vals = results[i][:, t, s, a, c]
            X_mean[i, t, s, a, c] = mean(vals)
            X_std[i,  t, s, a, c] = std(vals; corrected=false)
        end
        for (j, pname) in enumerate(SBI_PARAM_NAMES)
            y[i, j] = Float64(df[i, Symbol(pname)])
        end
    end

    npz_raw  = joinpath(tensors_dir, "X_raw.npz")
    npz_mean = joinpath(tensors_dir, "X_mean.npz")
    npz_std  = joinpath(tensors_dir, "X_std.npz")
    npz_y    = joinpath(tensors_dir, "y.npz")

    NPZ.npzwrite(npz_raw,  Dict("X" => X_raw))
    NPZ.npzwrite(npz_mean, Dict("X" => X_mean))
    NPZ.npzwrite(npz_std,  Dict("X" => X_std))
    NPZ.npzwrite(npz_y,    Dict("y" => y))

    open(joinpath(tensors_dir, "param_names.txt"), "w") do io
        for p in SBI_PARAM_NAMES; println(io, p); end
    end
    open(joinpath(tensors_dir, "seed_list.txt"), "w") do io
        for s in multi_seeds; println(io, s); end
    end

    CSV.write(joinpath(outdir, "samples_filtered.csv"), df)

    println()
    @printf("  X_raw  : %s  %s\n", npz_raw,  string(size(X_raw)))
    @printf("  X_mean : %s  %s\n", npz_mean, string(size(X_mean)))
    @printf("  X_std  : %s  %s\n", npz_std,  string(size(X_std)))
    @printf("  y      : %s  %s\n", npz_y,    string(size(y)))
    println("\nDone.")
    println("Note: per_sample/ checkpoints preserved for re-assembly. Delete when confirmed.")
end

# ── Entry point ────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --threads N generate_multiseed_dataset.jl <sbi_config.json> [K=7]")
        exit(1)
    end
    K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
    main(ARGS[1], K)
end
