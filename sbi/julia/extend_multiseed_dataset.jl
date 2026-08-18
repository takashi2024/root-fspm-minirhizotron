#!/usr/bin/env julia
# extend_multiseed_dataset.jl
#
# Extends an existing K=7 multi-seed dataset by simulating K_ext=13 additional
# seeds (7008–7020), producing a merged K=20 dataset.
#
# Directory layout (under sbi/data/multi_seed/config_<tag>/):
#   per_sample/            — original K=7 checkpoints (UNTOUCHED)
#     s{sample_id}.npy       shape (7, T, ns, na, 3)
#   per_sample_ext/        — K_ext=13 new checkpoints (this script writes these)
#     s{sample_id}.npy       shape (13, T, ns, na, 3)
#   tensors/               — original K=7 tensors (UNTOUCHED)
#   tensors_k20/           — merged K=20 tensors (this script writes these)
#     X_raw.npz              (N, 20, T, ns, na, 3) float32
#     X_mean.npz             (N, T, ns, na, 3)     float32
#     X_std.npz              (N, T, ns, na, 3)     float32
#     y.npz                  (N, 8)                float64
#     param_names.txt
#     seed_list.txt          all 20 seeds used
#
# Resume: per_sample_ext/s{id}.npy checkpoints are skipped if they exist.
# Re-assembly (step 3) always re-runs from checkpoints.
#
# Usage (from repo root):
#   julia --threads 6 sbi/julia/extend_multiseed_dataset.jl <sbi_config.json> [K_ext=13]

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

const MULTISEED_BASE_ORIG = 7001   # original seeds: 7001–7007
const MULTISEED_BASE_EXT  = 7008   # extension seeds: 7008–7020
const K_ORIG              = 7

# ── Config loading ─────────────────────────────────────────────────────────────

function load_cfg(config_path::String)
    abs_path = isabspath(config_path) ? config_path : joinpath(REPO_ROOT, config_path)
    open(abs_path, "r") do io; JSON.parse(String(read(io))); end
end

# ── Simulate one sample at a given seed (identical to generate_multiseed_dataset.jl) ──

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

# ── Main ──────────────────────────────────────────────────────────────────────

function main(config_path::String, K_ext::Int)
    cfg = load_cfg(config_path)

    base_env_path    = joinpath(REPO_ROOT, cfg["base_env_json"])
    vd_template_path = joinpath(REPO_ROOT, cfg["vd_template_json"])
    src_outdir       = joinpath(REPO_ROOT, cfg["outdir"])
    days             = Int(cfg["days"])
    snapshot_days    = Int.(cfg["snapshot_days"])
    img_height       = Int(cfg["img_height"])
    img_width        = Int(cfg["img_width"])
    render_step_cm   = Float64(cfg["render_step_cm"])

    cfg_tag = replace(basename(src_outdir), "config_" => "")

    # multi_seed_outdir key allows v2 configs to redirect to multi_seed_v2/
    if haskey(cfg, "multi_seed_outdir")
        ms_dir = joinpath(REPO_ROOT, cfg["multi_seed_outdir"])
    else
        ms_dir = joinpath(REPO_ROOT, "sbi", "data", "multi_seed", "config_$(cfg_tag)")
    end
    K_total_label   = K_ORIG + K_ext
    per_sample_orig = joinpath(ms_dir, "per_sample")
    per_sample_ext  = joinpath(ms_dir, "per_sample_ext")
    tensors_kN      = joinpath(ms_dir, "tensors_k$(K_total_label)")
    tmp_root        = joinpath(ms_dir, "_tmp_ext_")
    mkpath(per_sample_ext)
    mkpath(tensors_kN)
    mkpath(tmp_root)

    ext_seeds  = collect(MULTISEED_BASE_EXT : MULTISEED_BASE_EXT + K_ext - 1)
    orig_seeds = collect(MULTISEED_BASE_ORIG : MULTISEED_BASE_ORIG + K_ORIG - 1)
    all_seeds  = vcat(orig_seeds, ext_seeds)
    K_total    = K_ORIG + K_ext

    println("=" ^ 65)
    println("Multi-seed Dataset Extender")
    @printf("  Config tag    : %s\n", cfg_tag)
    @printf("  Orig seeds    : K=%d  %s\n", K_ORIG, orig_seeds)
    @printf("  Ext  seeds    : K=%d  %s\n", K_ext,  ext_seeds)
    @printf("  Total K       : %d\n", K_total)
    @printf("  Threads       : %d\n", nthreads())
    @printf("  ms_dir        : %s\n", ms_dir)
    println("=" ^ 65)

    # ── Validate: per_sample/ must exist and contain K=7 checkpoints ──────────

    if !isdir(per_sample_orig)
        error("Original checkpoints not found: $per_sample_orig\n" *
              "Run generate_multiseed_dataset.jl first.")
    end

    # ── 1. Load filtered samples from existing samples_filtered.csv ───────────

    samples_csv = joinpath(ms_dir, "samples_filtered.csv")
    if !isfile(samples_csv)
        error("samples_filtered.csv not found at $samples_csv. " *
              "Run generate_multiseed_dataset.jl first.")
    end
    df = CSV.read(samples_csv, DataFrame)
    N  = nrow(df)

    base_env = _read_json(base_env_path)
    ns = length(base_env["minirhizotron"]["station_positions_cm"])
    na = length(base_env["minirhizotron"]["camera_azimuths_deg"])
    T  = length(snapshot_days)

    @printf("\nSamples to extend: %d\n\n", N)

    # ── 2. Simulate K_ext new seeds per sample ────────────────────────────────

    println("[2/3] Simulating extension seeds  (K_ext=$K_ext × N=$N)...")
    println("      Checkpoints → $per_sample_ext")
    println()

    ext_results = Vector{Array{Float32,5}}(undef, N)   # each: (K_ext, T, ns, na, 3)
    progress    = Threads.Atomic{Int}(0)

    @threads for i in 1:N
        row       = df[i, :]
        sample_id = Int(row.sample_id)
        chk_path  = joinpath(per_sample_ext, @sprintf("s%04d.npy", sample_id))

        if isfile(chk_path)
            raw = NPZ.npzread(chk_path)   # (K_ext, T, ns, na, 3)
            ext_results[i] = raw
            Threads.atomic_add!(progress, 1)
            @printf("  [%d/%d] s%04d SKIPPED (ext checkpoint exists)\n",
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

        raw = zeros(Float32, K_ext, T, ns, na, 3)
        for (ki, seed) in enumerate(ext_seeds)
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
        NPZ.npzwrite(chk_path, raw)
        ext_results[i] = raw

        n_done = Threads.atomic_add!(progress, 1) + 1
        @printf("  [%d/%d] s%04d done\n", n_done, N, sample_id)
    end

    rm(tmp_root; recursive=true, force=true)

    # ── 3. Merge orig + ext checkpoints and assemble K=20 tensors ─────────────

    println("\n[3/3] Merging K=$K_ORIG + K=$K_ext → K=$K_total and assembling tensors...")

    X_raw  = zeros(Float32, N, K_total, T, ns, na, 3)
    X_mean = zeros(Float32, N, T, ns, na, 3)
    X_std  = zeros(Float32, N, T, ns, na, 3)
    y      = zeros(Float64, N, length(SBI_PARAM_NAMES))

    for i in 1:N
        row       = df[i, :]
        sample_id = Int(row.sample_id)

        orig_chk = joinpath(per_sample_orig, @sprintf("s%04d.npy", sample_id))
        if !isfile(orig_chk)
            error("Original checkpoint missing for s$sample_id: $orig_chk")
        end
        orig_raw = NPZ.npzread(orig_chk)   # (7, T, ns, na, 3)
        ext_raw  = ext_results[i]           # (K_ext, T, ns, na, 3)

        merged = cat(orig_raw, ext_raw; dims=1)   # (K_total, T, ns, na, 3)
        X_raw[i, :, :, :, :, :] = merged

        for t in 1:T, s in 1:ns, a in 1:na, c in 1:3
            vals = merged[:, t, s, a, c]
            X_mean[i, t, s, a, c] = mean(vals)
            X_std[i,  t, s, a, c] = std(vals; corrected=false)
        end

        for (j, pname) in enumerate(SBI_PARAM_NAMES)
            y[i, j] = Float64(df[i, Symbol(pname)])
        end
    end

    npz_raw  = joinpath(tensors_kN, "X_raw.npz")
    npz_mean = joinpath(tensors_kN, "X_mean.npz")
    npz_std  = joinpath(tensors_kN, "X_std.npz")
    npz_y    = joinpath(tensors_kN, "y.npz")

    NPZ.npzwrite(npz_raw,  Dict("X" => X_raw))
    NPZ.npzwrite(npz_mean, Dict("X" => X_mean))
    NPZ.npzwrite(npz_std,  Dict("X" => X_std))
    NPZ.npzwrite(npz_y,    Dict("y" => y))

    open(joinpath(tensors_kN, "param_names.txt"), "w") do io
        for p in SBI_PARAM_NAMES; println(io, p); end
    end
    open(joinpath(tensors_kN, "seed_list.txt"), "w") do io
        for s in all_seeds; println(io, s); end
    end

    println()
    @printf("  X_raw  : %s  %s\n", npz_raw,  string(size(X_raw)))
    @printf("  X_mean : %s  %s\n", npz_mean, string(size(X_mean)))
    @printf("  X_std  : %s  %s\n", npz_std,  string(size(X_std)))
    @printf("  y      : %s  %s\n", npz_y,    string(size(y)))
    println()
    println("Seeds used (all $K_total):")
    println("  orig: $(orig_seeds)")
    println("  ext : $(ext_seeds)")
    println()
    println("Done.")
    println("Note: per_sample/ (K=7) and per_sample_ext/ (K=$K_ext) checkpoints preserved.")
    @printf("      tensors/ (K=7) unchanged.  tensors_k%d/ (K=%d) written.\n", K_total, K_total)
end

# ── Entry point ────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --threads N extend_multiseed_dataset.jl <sbi_config.json> [K_ext=13]")
        exit(1)
    end
    K_ext = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 13
    main(ARGS[1], K_ext)
end
