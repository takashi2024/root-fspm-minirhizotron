# SimRunner.jl
#
# Runs the FSPM forward model for one sampled parameter set and extracts
# B&W FOV images + summary statistics at each snapshot day.
#
# Pattern mirrors _run_sim_1tube() in calibration/main_Optimization_mr_flexible.jl:
#   1. Write a candidate species JSON patching the VD template
#   2. Write a temporary env JSON pointing to the candidate species file
#   3. load_environment_and_plants() to build plants/box/reg
#   4. Manual day loop with step_plant!()
#   5. At each snapshot day: render images + extract summary metrics
#
# Does NOT include Simulation.jl or Plotting.jl — this manual loop replaces them.
#
# Dependencies (included by generate_dataset.jl before this file):
#   Deps, Types, Geometry, DensityRegulation, GrowthStep, MetricsAndOutput, Config
#   FovRenderer, PriorSampler

using JSON
using JSON3
using Printf
using Random

# ──────────────────────────────────────────────────────────────────────────────
# JSON helpers (same pattern as calibration scripts)
# ──────────────────────────────────────────────────────────────────────────────

_read_json(path::AbstractString) =
    open(path, "r") do io; Dict(JSON.parse(String(read(io)))); end

function _write_json_pretty(path::AbstractString, obj)
    open(path, "w") do io; JSON3.write(io, obj; indent=2); write(io, "\n"); end
end

# ──────────────────────────────────────────────────────────────────────────────
# Build a candidate species JSON by patching the VD template
# ──────────────────────────────────────────────────────────────────────────────

function _write_candidate_species_json(
    vd_template_path::String,
    params::NamedTuple,
    out_path::String
)
    d = _read_json(vd_template_path)
    # Patch the 8 SBI parameters (key names must match Config.jl exactly)
    d["root_elongation_mode"]    = "logistic"
    d["elongation_rate_min"]     = params.elongation_rate_min
    d["elongation_rate_max"]     = params.elongation_rate_max
    d["elongation_logistic_k"]   = params.elongation_logistic_k
    d["elongation_logistic_t50"] = params.elongation_logistic_t50
    d["branch_angle_order1"]     = params.branch_angle_order1
    d["g_axial"]                 = params.g_axial
    d["branching_prob_order1"]   = params.branching_prob_order1
    d["branching_prob_order2"]   = params.branching_prob_order2
    # Fixed structural params
    d["n_axial_roots"]           = params.n_axial_roots
    d["max_order1_per_plant"]    = params.max_order1_per_plant
    _write_json_pretty(out_path, d)
end

# ──────────────────────────────────────────────────────────────────────────────
# Build a temporary env JSON disabling all slow outputs
# ──────────────────────────────────────────────────────────────────────────────

function _write_candidate_env_json(
    base_env_path::String,
    species_json_path::String,
    sim_seed::Int,
    days::Int,
    tmp_dir::String,
    out_path::String
)
    env = _read_json(base_env_path)
    env["species_params_files"]["VD"] = abspath(species_json_path)
    env["simulation"]["seed"]         = sim_seed
    env["simulation"]["days"]         = days
    env["simulation"]["outdir"]       = tmp_dir
    env["simulation"]["save_every"]   = 0
    # Disable all csv/plot outputs — we extract in-memory
    env["visualization_2d"]["enabled"]    = false
    env["visualization_plate"]["enabled"] = false
    env["visible_metrics"]["enabled"]     = false
    env["visible_grid_daily"]["enabled"]  = false
    env["daily_snapshots"]["enabled"]     = false
    env["bdepth_metrics"]["enabled"]      = false
    # Keep minirhizotron enabled for geometry (but we sample in-memory, not via csv)
    _write_json_pretty(out_path, env)
end

# ──────────────────────────────────────────────────────────────────────────────
# Summary metric extraction for one snapshot
# ──────────────────────────────────────────────────────────────────────────────
# Returns (ns × na × 3) Float32 array:
#   channel 0: root length density (cm) — log1p-scaled later in Python
#   channel 1: root presence (0 or 1)
#   channel 2: mean crossing angle / 90  (∈ [0,1])

function _extract_summary(plants::Vector{Plant}, mr::Minirhizotron)
    ns     = length(mr.station_positions_cm)
    na     = length(mr.camera_azimuths_deg)
    result = zeros(Float32, ns, na, 3)

    lengths = tube_station_azimuth_lengths_cm(plants, mr)
    angles  = tube_station_azimuth_angles_cm(plants, mr)

    for si in 1:ns, ai in 1:na
        result[si, ai, 1] = Float32(lengths[si, ai])
        result[si, ai, 2] = lengths[si, ai] > 1e-6 ? 1f0 : 0f0
        ang = angles[si, ai]
        result[si, ai, 3] = isnan(ang) ? 0f0 : Float32(clamp(ang / 90.0, 0.0, 1.0))
    end
    return result
end

# ──────────────────────────────────────────────────────────────────────────────
# Core: run one sample
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_one_sample(sample_id, params, base_env_path, vd_template_path,
                   tmp_dir, img_dir;
                   days, snapshot_days, img_height, img_width, render_step_cm,
                   save_png)
    -> (images, summary, total_visible_length_cm)

`images`  :: Array{UInt8, 5}  shape (T, ns, na, H, W)
`summary` :: Array{Float32, 4} shape (T, ns, na, 3)
`total_visible_length_cm` :: Float64  (0 → all-null; stored as has_visible_roots flag)
"""
function run_one_sample(
    sample_id::Int,
    params::NamedTuple,
    base_env_path::String,
    vd_template_path::String,
    tmp_dir::String,
    img_dir::String;
    days::Int                = 90,
    snapshot_days::Vector{Int} = [14,21,28,35,42,49,56,63,70,77,84],
    img_height::Int          = 64,
    img_width::Int           = 64,
    render_step_cm::Float64  = 0.05,
    save_png::Bool           = true
)
    mkpath(tmp_dir)
    mkpath(img_dir)

    sp_path  = joinpath(tmp_dir, @sprintf("sbi_species_%04d.json", sample_id))
    env_path = joinpath(tmp_dir, @sprintf("sbi_env_%04d.json",     sample_id))

    _write_candidate_species_json(vd_template_path, params, sp_path)
    _write_candidate_env_json(base_env_path, sp_path, params.sim_seed, days, tmp_dir, env_path)

    cfg = load_environment_and_plants(env_path)
    mr  = cfg.mr
    @assert mr.enabled "Minirhizotron must be enabled in base env config"

    ns = length(mr.station_positions_cm)
    na = length(mr.camera_azimuths_deg)
    T  = length(snapshot_days)

    images  = zeros(UInt8,    T, ns, na, img_height, img_width)
    summary = zeros(Float32,  T, ns, na, 3)

    Random.seed!(params.sim_seed)
    snap_idx  = 1
    R_dummy   = zeros(Float64, 1, 1)
    nx_dummy  = 1
    nz_dummy  = 1

    for day in 1:days
        for pl in cfg.plants
            step_plant!(pl, cfg.box, cfg.reg, R_dummy, R_dummy, nx_dummy, nz_dummy;
                        mature_branching = false,
                        facilitation     = nothing)
        end

        if snap_idx <= T && day == snapshot_days[snap_idx]
            fov_tensor = render_and_save_all_fov(
                cfg.plants, mr, day, sample_id, img_dir;
                img_height    = img_height,
                img_width     = img_width,
                render_step_cm = render_step_cm,
                save_png      = save_png
            )
            images[snap_idx, :, :, :, :]  = fov_tensor
            summary[snap_idx, :, :, :]    = _extract_summary(cfg.plants, mr)
            snap_idx += 1
        end
    end

    total_vis = sum(summary[:, :, :, 1])   # sum of rld across all windows/days
    return images, summary, Float64(total_vis)
end

# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_batch(df_params, base_env_path, vd_template_path, outdir;
              days, snapshot_days, img_height, img_width, render_step_cm,
              save_png, show_progress)
    -> (X_images, X_summary, y, df_params_updated)

X_images :: Array{UInt8, 6}   (N, T, ns, na, H, W)
X_summary :: Array{Float32, 5} (N, T, ns, na, 3)
y         :: Matrix{Float64}   (N, 8)  — raw param values in SBI_PARAM_NAMES order
"""
function run_batch(
    df_params::DataFrame,
    base_env_path::String,
    vd_template_path::String,
    outdir::String;
    days::Int                  = 90,
    snapshot_days::Vector{Int} = [14,21,28,35,42,49,56,63,70,77,84],
    img_height::Int            = 64,
    img_width::Int             = 64,
    render_step_cm::Float64    = 0.05,
    save_png::Bool             = true,
    show_progress::Bool        = true
)
    N  = nrow(df_params)
    T  = length(snapshot_days)
    # Determine ns/na from base env (parse once)
    base_env = _read_json(base_env_path)
    ns = length(base_env["minirhizotron"]["station_positions_cm"])
    na = length(base_env["minirhizotron"]["camera_azimuths_deg"])

    img_dir = joinpath(outdir, "images")
    tmp_dir = joinpath(outdir, "_sbi_tmp_")
    mkpath(img_dir)
    mkpath(tmp_dir)

    X_images  = zeros(UInt8,   N, T, ns, na, img_height, img_width)
    X_summary = zeros(Float32, N, T, ns, na, 3)
    y         = zeros(Float64, N, length(SBI_PARAM_NAMES))

    for (i, row) in enumerate(eachrow(df_params))
        show_progress && @printf("  [%d/%d] sample_id=%d ...\n", i, N, row.sample_id)

        params = (;
            elongation_rate_min     = Float64(row.elongation_rate_min),
            elongation_rate_max     = Float64(row.elongation_rate_max),
            elongation_logistic_k   = Float64(row.elongation_logistic_k),
            elongation_logistic_t50 = Float64(row.elongation_logistic_t50),
            branch_angle_order1     = Float64(row.branch_angle_order1),
            g_axial                 = Float64(row.g_axial),
            branching_prob_order1   = Float64(row.branching_prob_order1),
            branching_prob_order2   = Float64(row.branching_prob_order2),
            sample_id               = Int(row.sample_id),
            sim_seed                = Int(row.sim_seed),
            n_axial_roots           = Int(row.n_axial_roots),
            max_order1_per_plant    = Int(row.max_order1_per_plant),
        )

        imgs, summ, total_vis = run_one_sample(
            Int(row.sample_id), params,
            base_env_path, vd_template_path,
            tmp_dir, img_dir;
            days=days, snapshot_days=snapshot_days,
            img_height=img_height, img_width=img_width,
            render_step_cm=render_step_cm, save_png=save_png
        )

        X_images[i, :, :, :, :, :]  = imgs
        X_summary[i, :, :, :, :]    = summ
        for (j, pname) in enumerate(SBI_PARAM_NAMES)
            y[i, j] = Float64(row[pname])
        end

        # Update viability flag in the dataframe
        df_params[i, :has_visible_roots] = total_vis > 0.0
    end

    # Clean up temp JSONs
    rm(tmp_dir; recursive=true, force=true)

    return X_images, X_summary, y, df_params
end
