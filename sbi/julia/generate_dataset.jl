#!/usr/bin/env julia
# generate_dataset.jl
#
# CLI entry point for the SBI synthetic data generator.
#
# Usage (from repo root):
#   julia sbi/julia/generate_dataset.jl sbi/configs/sbi_config.json
#
# Outputs (all under sbi_config["outdir"], default "sbi/data"):
#   samples.csv                     — sampled parameter table with viability flag
#   images/s{id}_d{day}_st{si}_az{ai}.png  — B&W FOV images
#   tensors/X_fov.npz               — per-window input tensor  (N_win, T, H, W) uint8
#   tensors/X_fov_window_ids.npz    — window index array       (N_win, 3) int32
#   tensors/X_summary.npz           — summary feature tensor   (N, T, ns, na, 3) float32
#   tensors/y.npz                   — true parameter values    (N, 8) float64
#   tensors/param_names.txt         — one parameter name per line
#
# The per-window format (Option B) stores one row per (simulation × station × azimuth)
# pair.  Each row has T=11 temporal frames stacked as channels.  This matches the
# expected CNN input shape (batch, T=11, H=64, W=64).

using JSON
using Printf
using NPZ

# ──────────────────────────────────────────────────────────────────────────────
# Repo root detection
# ──────────────────────────────────────────────────────────────────────────────
const REPO_ROOT = dirname(dirname(dirname(abspath(@__FILE__))))

# ──────────────────────────────────────────────────────────────────────────────
# Include model source (same pattern as calibration scripts)
# ──────────────────────────────────────────────────────────────────────────────
for f in ["Deps", "Types", "Geometry", "DensityRegulation",
          "GrowthStep", "MetricsAndOutput", "Config"]
    include(joinpath(REPO_ROOT, "src", "$(f).jl"))
end
# SBI modules
include(joinpath(@__DIR__, "FovRenderer.jl"))
include(joinpath(@__DIR__, "PriorSampler.jl"))
include(joinpath(@__DIR__, "SimRunner.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
function main(config_path::String)
    cfg_path_abs = isabspath(config_path) ? config_path :
                   joinpath(REPO_ROOT, config_path)
    cfg = open(cfg_path_abs, "r") do io; JSON.parse(String(read(io))); end

    base_env_path    = joinpath(REPO_ROOT, cfg["base_env_json"])
    vd_template_path = joinpath(REPO_ROOT, cfg["vd_template_json"])
    outdir           = joinpath(REPO_ROOT, cfg["outdir"])
    mkpath(outdir)

    N                 = Int(cfg["n_samples"])
    seeds_start       = Int(cfg["seeds_start"])
    rng_seed          = Int(get(cfg, "rng_seed", 0))
    days              = Int(cfg["days"])
    snapshot_days     = Int.(cfg["snapshot_days"])
    img_height        = Int(cfg["img_height"])
    img_width         = Int(cfg["img_width"])
    render_step_cm    = Float64(cfg["render_step_cm"])
    n_axial_roots     = Int(cfg["n_axial_roots"])
    max_order1        = Int(cfg["max_order1_per_plant"])

    println("=" ^ 60)
    println("SBI Dataset Generator")
    println("  Config:      $cfg_path_abs")
    println("  N samples:   $N")
    println("  Days:        $days")
    println("  Snapshots:   $(length(snapshot_days)) × $(snapshot_days)")
    println("  Image size:  $(img_height)×$(img_width)")
    println("  Output dir:  $outdir")
    println("=" ^ 60)

    # ── 1. Sample parameters ──────────────────────────────────────────────────
    println("\n[1/3] Sampling $N parameter sets via Latin Hypercube...")
    df = sample_prior(N; rng_seed=rng_seed, seeds_start=seeds_start,
                      n_axial_roots=n_axial_roots,
                      max_order1_per_plant=max_order1)
    samples_csv = joinpath(outdir, "samples.csv")
    save_samples_csv(df, samples_csv)
    println("      Saved: $samples_csv")

    # ── 2. Run simulations ────────────────────────────────────────────────────
    println("\n[2/3] Running $N simulations...")
    X_images, X_summary, y, df = run_batch(
        df, base_env_path, vd_template_path, outdir;
        days=days, snapshot_days=snapshot_days,
        img_height=img_height, img_width=img_width,
        render_step_cm=render_step_cm, save_png=true, show_progress=true
    )

    n_visible = sum(coalesce.(df.has_visible_roots, false))
    println("      Simulations with visible roots: $n_visible / $N")
    save_samples_csv(df, samples_csv)  # re-save with updated has_visible_roots

    # ── 3. Assemble and save NPZ tensors ──────────────────────────────────────
    println("\n[3/3] Assembling tensors...")
    tensors_dir = joinpath(outdir, "tensors")
    mkpath(tensors_dir)

    T_snaps = size(X_images, 2)
    ns      = size(X_images, 3)
    na      = size(X_images, 4)
    N_win   = N * ns * na

    # Per-window tensor: flatten (N, T, ns, na, H, W) → (N_win, T, H, W)
    # Row ordering: sample outer, station middle, azimuth inner
    X_win     = zeros(UInt8,   N_win, T_snaps, img_height, img_width)
    win_ids   = zeros(Int32,   N_win, 3)  # (sample_id, station_idx, azimuth_idx)
    y_win     = zeros(Float32, N_win, size(y, 2))  # labels broadcast to each window

    row_idx = 1
    for i in 1:N
        sid = Int32(df.sample_id[i])
        for si in 1:ns
            for ai in 1:na
                X_win[row_idx, :, :, :]  = X_images[i, :, si, ai, :, :]
                win_ids[row_idx, :]      = Int32[sid, si, ai]
                y_win[row_idx, :]        = Float32.(y[i, :])
                row_idx += 1
            end
        end
    end

    npz_fov     = joinpath(tensors_dir, "X_fov.npz")
    npz_win_ids = joinpath(tensors_dir, "X_fov_window_ids.npz")
    npz_summary = joinpath(tensors_dir, "X_summary.npz")
    npz_y       = joinpath(tensors_dir, "y.npz")
    pnames_txt  = joinpath(tensors_dir, "param_names.txt")

    NPZ.npzwrite(npz_fov,     Dict("X" => X_win))
    NPZ.npzwrite(npz_win_ids, Dict("window_ids" => win_ids))
    NPZ.npzwrite(npz_summary, Dict("X" => X_summary))
    NPZ.npzwrite(npz_y,       Dict("y" => Float32.(y)))
    open(pnames_txt, "w") do io
        for p in SBI_PARAM_NAMES; println(io, p); end
    end

    println("      X_fov    : $(npz_fov)  $(size(X_win))")
    println("      X_summary: $(npz_summary)  $(size(X_summary))")
    println("      y        : $(npz_y)  $(size(y))")
    println("\nDone.")
end

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("Usage: julia generate_dataset.jl <path/to/sbi_config.json>")
        exit(1)
    end
    main(ARGS[1])
end
