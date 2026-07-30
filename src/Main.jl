# Auto-split from root_fspm_rhizobox_multiplant_SRL.jl
# File: Main.jl

function main(param_path::AbstractString)
    cfg = load_environment_and_plants(param_path)

    outdir     = cfg.outdir
    plot3d_dir = cfg.plot3d_dir
    box        = cfg.box
    reg        = cfg.reg
    plants     = cfg.plants
    facil      = cfg.facil

    @info "Loaded config" H=box.H W=box.W D=box.D tilt_from_vertical_deg=rad2deg(box.tilt_from_vertical) nplants=length(plants)

    mkpath(outdir)
    mkpath(plot3d_dir)

    simulate!(plants, box, reg;
        days=cfg.days,
        seed=cfg.seed,
        save_every=cfg.save_every,
        outdir=outdir,
        plot3d_dir=plot3d_dir,   # only add if simulate! supports it
        mature_branching=cfg.mature_branching,
        viz_enabled=cfg.viz_enabled,
        viz_save_every=cfg.viz_save_every,
        viz_project_to_plate=cfg.viz_project_to_plate,
        viz_y_max_cm=cfg.viz_y_max_cm,
        viz_linewidth=cfg.viz_linewidth,
        viz_outdir=cfg.viz_outdir,
        vm_enabled=cfg.vm_enabled,
        vm_y_max_cm=cfg.vm_y_max_cm,
        vm_csv_path=joinpath(outdir, cfg.vm_csv_name),
        vg_enabled=cfg.vg_enabled,
        vg_every=cfg.vg_every,
        vg_method=cfg.vg_method,
        vg_dx=cfg.vg_dx,
        vg_dz=cfg.vg_dz,
        vg_sample_step_cm=cfg.vg_sample_step_cm,
        vg_raster_step_cm=cfg.vg_raster_step_cm,
        vg_y_min_cm=cfg.vg_y_min_cm,
        vg_y_max_cm=cfg.vg_y_max_cm,
        vg_csv_path=joinpath(outdir, cfg.vg_csv_name),

        snapshot_enabled=cfg.snapshot_enabled,
        snapshot_nrows=cfg.snapshot_nrows,
        snapshot_ncols=cfg.snapshot_ncols,
        snapshot_y_max_cm=cfg.snapshot_y_max_cm,
        snapshot_step_cm=cfg.snapshot_step_cm,
        snapshot_csv_path=joinpath(outdir, cfg.snapshot_csv_name),

        bdepth_enabled = cfg.bdepth_enabled,
        bdepth_species = cfg.bdepth_species,
        bdepth_csv_path = joinpath(outdir, cfg.bdepth_csv_name),
        facilitation=cfg.facil,
        minirhizotron=cfg.mr,
        mr_csv_path=joinpath(outdir, cfg.mr.csv_name),
        viz_plate_enabled=cfg.viz_plate_enabled,
        viz_plate_save_every=cfg.viz_plate_save_every,
        viz_plate_outdir=cfg.viz_plate_outdir
    )

    export_segments_csv(plants, joinpath(outdir, "segments_all.csv"))
    export_pointcloud_xyz(plants, joinpath(outdir, "pointcloud_all.xyz"); step_cm=cfg.export_step_cm, add_fields=true)
    # export_pointcloud_ply(joinpath(outdir, "pointcloud_all.xyz"), plants)
    export_root_summary_csv(plants, joinpath(outdir, "root_summary.csv"))

    # Visible grid (dx=1, dz=1) + per-plant rows
    # export_visible_grid_csv(
    #     plants, box, joinpath(outdir, "visible_grid_dx1_dz1_exact.csv");
    #     method=:exact,
    #     include_byplant=true,
    #     y_min_cm=0.0, y_max_cm=cfg.vm_y_max_cm,
    #     dx=1.0, dz=1.0,
    #     sample_step_cm=0.2
    # )

    # Optional raster method (requires ImageMorphology)
    # export_visible_grid_csv(
    #     plants, box, joinpath(outdir, "visible_grid_dx1_dz1_raster.csv");
    #     method=:raster,
    #     include_byplant=true,
    #     y_min_cm=0.0, y_max_cm=cfg.vm_y_max_cm,
    #     dx=1.0, dz=1.0,
    #     raster_step_cm=0.2
    # )

    seg_dir = joinpath(outdir, "segments")
    pc_dir  = joinpath(outdir, "pointcloud")
    mkpath(seg_dir); mkpath(pc_dir)
    for pl in plants
        export_segments_csv([pl], joinpath(seg_dir, "segments_plant$(pl.id)_$(pl.species).csv"))
        export_pointcloud_xyz([pl], joinpath(pc_dir, "pointcloud_plant$(pl.id)_$(pl.species).xyz"); step_cm=cfg.export_step_cm, add_fields=true)
        # export_pointcloud_ply(joinpath(pc_dir, "pointcloud_plant$(pl.id)_$(pl.species).ply"), [pl])
    end

    if cfg.layout_diag_enabled
        save_layout_diagnostics(box, cfg.mr, plants, cfg.layout_diag_outdir)
    end

    total_segments = sum(length(pl.root.segments) for pl in plants)
    total_tips = sum(length(pl.root.tips) for pl in plants)
    @info "Simulation finished." total_segments=total_segments total_tips=total_tips outdir=outdir
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "Parameter.json")
    main(param_path)
end
