# Auto-split from root_fspm_rhizobox_multiplant_SRL.jl
# File: Config.jl

# Load one species parameter JSON into SpeciesParams.
# Backward-compatible: new key names are tried first; old key names are accepted as fallbacks.
# Old name → new name mapping (both accepted in JSON):
#   elongation_rate        → elongation_rate_axial
#   branching_prob         → branching_prob_order1
#   branch_angle           → branch_angle_order1
#   max_tip_age            → max_tip_age_axial
#   elongation_rate_major  → elongation_rate_order1
#   max_tip_age_major      → max_tip_age_order1
#   branching_prob_minor   → branching_prob_order2
#   branch_angle_minor     → branch_angle_order2
#   elongation_rate_minor  → elongation_rate_order2
#   max_tip_age_minor      → max_tip_age_order2
#   minor_start_depth_cm   → order2_start_depth_cm
#   minor_end_depth_cm     → order2_end_depth_cm
#   g_primary              → g_axial
#   noise_primary          → noise_axial
#   n_basal_roots          → n_axial_roots
#   basal_spread_deg       → axial_spread_deg
#   basal_jitter_sd_deg    → axial_jitter_sd_deg
#   max_major_branches     → max_order1_per_plant
function load_species_params(path::AbstractString)
    d = _read_json(path)

    architecture_type = String(get(d, "architecture_type", "generic_multi_axial"))
    name = String(get(d, "name", "Unknown"))

    # --- Order 0 (axial roots) ---
    elong_axial        = Float64(get(d, "elongation_rate_axial",   get(d, "elongation_rate",   1.0)))
    branching_prob_order1 = Float64(get(d, "branching_prob_order1", get(d, "branching_prob",   0.05)))
    branch_angle_order1   = Float64(get(d, "branch_angle_order1",   get(d, "branch_angle",    60.0)))
    maxage_axial       = Int(get(d, "max_tip_age_axial",            get(d, "max_tip_age",       60)))

    # --- Order 1 (first-order laterals) ---
    elong_order1          = Float64(get(d, "elongation_rate_order1", get(d, "elongation_rate_major", elong_axial)))
    branching_prob_order2 = Float64(get(d, "branching_prob_order2",  get(d, "branching_prob_minor",  0.25)))
    branch_angle_order2   = Float64(get(d, "branch_angle_order2",    get(d, "branch_angle_minor",    branch_angle_order1)))
    maxage_order1         = Int(get(d, "max_tip_age_order1",         get(d, "max_tip_age_major",     maxage_axial)))

    # --- Order 2+ (second-order laterals / fine laterals) ---
    elong_order2  = Float64(get(d, "elongation_rate_order2", get(d, "elongation_rate_minor", 0.6 * elong_axial)))
    maxage_order2 = Int(get(d, "max_tip_age_order2",         get(d, "max_tip_age_minor",     max(10, round(Int, 0.5 * maxage_axial)))))

    # --- Logistic elongation mode ---
    root_elongation_mode = Symbol(lowercase(String(get(d, "root_elongation_mode", "constant"))))

    elong_min = Float64(get(d, "elongation_rate_min", elong_axial))
    elong_max = Float64(get(d, "elongation_rate_max", elong_axial))
    elong_k   = Float64(get(d, "elongation_logistic_k",   0.2))
    elong_t50 = Float64(get(d, "elongation_logistic_t50", 10.0))

    # --- Depth window for order-2 lateral emergence ---
    order2_start = Float64(get(d, "order2_start_depth_cm", get(d, "minor_start_depth_cm", 3.0)))
    order2_end   = Float64(get(d, "order2_end_depth_cm",   get(d, "minor_end_depth_cm",  25.0)))

    # --- Global controls ---
    g_axial      = Float64(get(d, "g_axial",      get(d, "g_primary",     0.4)))
    g_lateral    = Float64(get(d, "g_lateral",    0.15))
    gravitropism = Float64(get(d, "gravitropism", 1.0))

    srl          = Float64(get(d, "srl_m_per_g",  150.0))
    noise_axial  = Float64(get(d, "noise_axial",  get(d, "noise_primary",  0.05)))
    noise_lateral= Float64(get(d, "noise_lateral", 0.15))

    # --- Axial root initialization ---
    n_axial          = Int(get(d, "n_axial_roots",       get(d, "n_basal_roots",      1)))
    axial_spread     = Float64(get(d, "axial_spread_deg", get(d, "basal_spread_deg",  0.0)))
    axial_jitter_sd  = Float64(get(d, "axial_jitter_sd_deg", get(d, "basal_jitter_sd_deg", 2.0)))

    # --- Caps ---
    max_order1 = Int(get(d, "max_order1_per_plant", get(d, "max_major_branches", 6)))

    return SpeciesParams(
        architecture_type,
        name,
        root_elongation_mode,
        elong_min, elong_max, elong_k, elong_t50,
        elong_axial, branching_prob_order1, branch_angle_order1, maxage_axial,
        elong_order1, branching_prob_order2, branch_angle_order2, maxage_order1,
        elong_order2, maxage_order2,
        order2_start, order2_end,
        gravitropism, srl,
        noise_axial, noise_lateral,
        g_axial, g_lateral,
        n_axial, axial_spread, axial_jitter_sd,
        max_order1
    )
end

@inline x_from_left_to_centered(x_from_left::Real, W::Real) = Float64(x_from_left) - Float64(W)/2

function _parse_minirhizotron(mr::Dict, W::Float64)
    enabled = Bool(get(mr, "enabled", false))

    origin_raw = Vector{Float64}(Float64.(mr["tube_origin_cm"]))
    # x coord uses the same x_from_left convention as plant positions
    origin = [x_from_left_to_centered(origin_raw[1], W), origin_raw[2], origin_raw[3]]

    if haskey(mr, "tube_direction")
        raw = Vector{Float64}(Float64.(mr["tube_direction"]))
        direction = normalize_vec(raw)
    else
        az_deg = Float64(get(mr, "tube_azimuth_deg", 0.0))
        el_deg = Float64(get(mr, "tube_angle_from_horizontal_deg", 45.0))
        az = deg2rad(az_deg)
        el = deg2rad(el_deg)
        # x=across-row, y=plot-depth, z=vertical (down = negative z)
        direction = normalize_vec([sin(az)*cos(el), cos(az)*cos(el), -sin(el)])
    end

    length_cm     = Float64(get(mr, "tube_length_cm", 80.0))
    r_outer_cm    = Float64(get(mr, "tube_outer_radius_cm", 3.5))
    skin_depth_cm = Float64(get(mr, "skin_depth_cm", 0.3))
    image_along   = Float64(get(mr, "image_along_cm", 1.35))
    image_arc     = Float64(get(mr, "image_arc_cm", 5.2))
    sample_step   = Float64(get(mr, "sample_step_cm", 0.1))
    csv_name      = String(get(mr, "csv_name", "tube_stations_daily.csv"))

    if haskey(mr, "station_positions_cm")
        station_pos = sort(Vector{Float64}(Float64.(mr["station_positions_cm"])))
    else
        ns    = Int(get(mr, "n_stations", 10))
        first = Float64(get(mr, "station_first_cm", 5.0))
        last  = Float64(get(mr, "station_last_cm", length_cm - 5.0))
        station_pos = ns > 1 ? collect(range(first, last; length=ns)) : [first]
    end

    cam_azimuths = Vector{Float64}(Float64.(get(mr, "camera_azimuths_deg", Float64[])))

    return Minirhizotron(enabled, origin, direction, length_cm, r_outer_cm, skin_depth_cm,
                         image_along, image_arc, station_pos, sample_step, csv_name, cam_azimuths)
end

function _default_minirhizotron()
    return Minirhizotron(false, [0.0,0.0,0.0], [0.0,1.0,0.0], 0.0, 3.5, 0.3,
                         1.35, 5.2, Float64[], 0.1, "tube_stations_daily.csv", Float64[])
end

function load_environment_and_plants(env_path::AbstractString)
    env = _read_json(env_path)

    rb = env["rhizobox"]
    H = Float64(rb["H"])
    W = Float64(rb["W"])
    D = Float64(rb["D"])

    angle_from_horizontal = Float64(get(rb, "angle_from_horizontal_deg", 60.0))
    tilt_from_vertical = deg2rad(90.0 - angle_from_horizontal)

    periodic_x = Bool(get(rb, "periodic_x", false))
    periodic_y = Bool(get(rb, "periodic_y", false))
    periodic_z = Bool(get(rb, "periodic_z", false))
    box = Rhizobox(H, W, D, tilt_from_vertical, periodic_x, periodic_y, periodic_z)


    fac = get(env, "facilitation", Dict{String,Any}())
    fac_enabled = Bool(get(fac, "enabled", false))
    fac_focal = String(get(fac, "focal_species", "H"))
    fac_source = String(get(fac, "source_species", "B"))
    fac_strength = Float64(get(fac, "strength", 0.0))
    fac_radius_cm = Float64(get(fac, "radius_cm", 5.0))
    fac_depth_max_cm = Float64(get(fac, "depth_max_cm", 20.0))
    fac_depth_peak_cm = Float64(get(fac, "depth_peak_cm", min(10.0, 0.5*fac_depth_max_cm)))
    kernel_str = lowercase(String(get(fac, "kernel", "gaussian")))
    fac_kernel = kernel_str == "linear"      ? :linear :
                kernel_str == "exponential" ? :exponential :
                                            :gaussian
    trigger_depth_cm  = Float64(get(fac, "trigger_depth_cm", 10.0))
    trigger_delay_days = Int(get(fac, "trigger_delay_days", 0))
    trigger_ramp_days  = Int(get(fac, "trigger_ramp_days", 0))

    dr = get(env, "density_regulation", Dict{String,Any}())
    mode_str = lowercase(String(get(dr, "mode", "all")))
    mode = mode_str == "intra" ? :intra : (mode_str == "inter" ? :inter : :all)

    reg = DensityRegulation(
        Bool(get(dr, "enabled", false)),
        mode,
        Float64(get(dr, "grid_dx_cm", 2.0)),
        Float64(get(dr, "grid_dz_cm", 2.0)),
        Float64(get(dr, "sample_step_cm", 0.2)),
        Float64(get(dr, "radius_cm", 4.0)),
        Float64(get(dr, "alpha_elong", 1.0)),
        Float64(get(dr, "alpha_branch", 1.0)),
        Float64(get(dr, "min_factor", 0.2))
    )

    sim = env["simulation"]
    days = Int(get(sim, "days", 50))
    seed = Int(get(sim, "seed", 42))
    save_every = Int(get(sim, "save_every", 0))
    outdir = String(get(sim, "outdir", "output_fspm_rhizobox"))
    plot3d_dir = String(get(sim, "plot3d_outdir", joinpath(outdir, "plot3d")))
    mature_branching = Bool(get(sim, "mature_branching", false))
    step_cm = Float64(get(sim, "export_step_cm", get(sim, "export_point_step_cm", 0.1)))

    viz = get(env, "visualization_2d", Dict{String,Any}())
    viz_enabled = Bool(get(viz, "enabled", false))
    viz_save_every = Int(get(viz, "save_every", 0))
    viz_project_to_plate = Bool(get(viz, "project_to_plate", true))
    viz_y_max_cm = Float64(get(viz, "y_max_cm", 0.5))
    viz_linewidth = Float64(get(viz, "linewidth", 0.8))
    viz_outdir = String(get(viz, "outdir", outdir))

    vp = get(env, "visualization_plate", Dict{String,Any}())
    viz_plate_enabled   = Bool(get(vp, "enabled",    false))
    viz_plate_save_every = Int(get(vp, "save_every", viz_save_every))
    viz_plate_outdir    = String(get(vp, "outdir",   joinpath(outdir, "viz_plate")))

    vm = get(env, "visible_metrics", Dict{String,Any}())
    vm_enabled = Bool(get(vm, "enabled", false))
    vm_y_max_cm = Float64(get(vm, "y_max_cm", viz_y_max_cm))
    vm_csv_name = String(get(vm, "csv_name", "visible_length_daily.csv"))

    vg = get(env, "visible_grid_daily", Dict{String,Any}())
    vg_enabled = Bool(get(vg, "enabled", false))
    vg_every = Int(get(vg, "save_every", 1))
    vg_method_str = lowercase(String(get(vg, "method", "exact")))
    vg_method = vg_method_str == "raster" ? :raster : :exact
    vg_dx = Float64(get(vg, "dx", 2.0))
    vg_dz = Float64(get(vg, "dz", 2.0))
    vg_sample_step_cm = Float64(get(vg, "sample_step_cm", 0.2))
    vg_raster_step_cm = Float64(get(vg, "raster_step_cm", 0.2))
    vg_y_min_cm = Float64(get(vg, "y_min_cm", 0.0))
    vg_y_max_cm = Float64(get(vg, "y_max_cm", vm_y_max_cm))
    vg_csv_name = String(get(vg, "csv_name", "visible_grid_daily.csv"))

    ds = get(env, "daily_snapshots", Dict{String,Any}())
    snapshot_enabled = Bool(get(ds, "enabled", false))
    snapshot_nrows = Int(get(ds, "nrows", 6))
    snapshot_ncols = Int(get(ds, "ncols", 6))
    snapshot_y_max_cm = Float64(get(ds, "y_max_cm", 0.5))
    snapshot_step_cm = Float64(get(ds, "step_cm", 0.2))
    snapshot_csv_name = String(get(ds, "csv_name", "daily_tiles.csv"))

    bd = get(env, "bdepth_metrics", Dict{String,Any}())
    bdepth_enabled = Bool(get(bd, "enabled", false))
    bdepth_species = String(get(bd, "species", "B"))
    bdepth_csv_name = String(get(bd, "csv_name", "daily_B_max_depth_cm.csv"))

    spfiles = env["species_params_files"]
    spmap = Dict{String,SpeciesParams}()
    for (k, v) in spfiles
        spmap[String(k)] = load_species_params(String(v))
    end

    plants_cfg = env["planting"]["plants"]
    plants = Plant[]
    for (i, pcfg_any) in enumerate(plants_cfg)
        pcfg = Dict(pcfg_any)
        species = String(pcfg["species"])
        pos = pcfg["pos_cm"]
        x = x_from_left_to_centered(pos[1], W)
        y = Float64(pos[2])
        z = Float64(pos[3])
        start = [x, y, z]
        sp = spmap[species]
        rs = initialize_root_plant(i, start, sp)
        obs_raw = get(pcfg, "obs_root_biomass_g", nothing)
        obs = obs_raw === nothing ? nothing : Float64(obs_raw)
        push!(plants, Plant(i, species, rs, sp, obs, 0))
    end

    
    # collect source plant positions in box coords (i,x,z)
    source_xz = Tuple{Int,Float64,Float64}[]
    for (i, pcfg_any) in enumerate(plants_cfg)
        pcfg = Dict(pcfg_any)
        species = String(pcfg["species"])
        if species == fac_source
            pos = pcfg["pos_cm"]
            x = x_from_left_to_centered(pos[1], W)
            z = Float64(pos[3])
            push!(source_xz, (i, x, z))
        end
    end

    # @info "Facilitation sources" n=length(source_xz) source_xz=source_xz # temporary on --> will be deleted. 

    facil = Facilitation(fac_enabled, fac_focal, fac_source, fac_strength, fac_radius_cm,
                         fac_depth_max_cm, fac_depth_peak_cm, fac_kernel, source_xz,
                         trigger_depth_cm, trigger_delay_days, trigger_ramp_days)

    mr_raw = get(env, "minirhizotron", nothing)
    mr = mr_raw !== nothing ? _parse_minirhizotron(Dict{String,Any}(mr_raw), W) : _default_minirhizotron()

    ld = get(env, "layout_diagnostics", Dict{String,Any}())
    layout_diag_enabled = Bool(get(ld, "enabled", false))
    layout_diag_outdir  = String(get(ld, "outdir", joinpath(outdir, "layout_diagnostics")))

    return (box=box, plants=plants, days=days, seed=seed, save_every=save_every,
            outdir=outdir, plot3d_dir=plot3d_dir, mature_branching=mature_branching, reg=reg, facil=facil,
            mr=mr,
            export_step_cm=step_cm,
            viz_enabled=viz_enabled,
            viz_save_every=viz_save_every,
            viz_project_to_plate=viz_project_to_plate,
            viz_y_max_cm=viz_y_max_cm,
            viz_linewidth=viz_linewidth,
            viz_outdir=viz_outdir,
            vm_enabled=vm_enabled,
            vm_y_max_cm=vm_y_max_cm,
            vm_csv_name=vm_csv_name,
            vg_enabled=vg_enabled,
            vg_every=vg_every,
            vg_method=vg_method,
            vg_dx=vg_dx,
            vg_dz=vg_dz,
            vg_sample_step_cm=vg_sample_step_cm,
            vg_raster_step_cm=vg_raster_step_cm,
            vg_y_min_cm=vg_y_min_cm,
            vg_y_max_cm=vg_y_max_cm,
            vg_csv_name=vg_csv_name,
            snapshot_enabled=snapshot_enabled,
            snapshot_nrows=snapshot_nrows,
            snapshot_ncols=snapshot_ncols,
            snapshot_y_max_cm=snapshot_y_max_cm,
            snapshot_step_cm=snapshot_step_cm,
            snapshot_csv_name=snapshot_csv_name,
            bdepth_enabled=bdepth_enabled,
            bdepth_species=bdepth_species,
            bdepth_csv_name=bdepth_csv_name,
            layout_diag_enabled=layout_diag_enabled,
            layout_diag_outdir=layout_diag_outdir,
            viz_plate_enabled=viz_plate_enabled,
            viz_plate_save_every=viz_plate_save_every,
            viz_plate_outdir=viz_plate_outdir)
end
