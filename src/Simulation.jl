# File: Simulation.jl

function simulate!(plants::Vector{Plant}, box::Rhizobox, reg::DensityRegulation;
                   days::Int=50, seed::Int=42,
                   save_every::Int=0,
                   outdir::String="output_fspm_rhizobox",
                   plot3d_dir::String = joinpath(outdir, "plot3d"),
                   mature_branching::Bool=false,
                   viz_enabled::Bool=false,
                   viz_save_every::Int=0,
                   viz_project_to_plate::Bool=true,
                   viz_y_max_cm::Float64=0.5,
                   viz_linewidth::Float64=2.0,
                   viz_outdir::String=outdir,
                   vm_enabled::Bool=false,
                   vm_y_max_cm::Float64=0.5,
                   vm_csv_path::String=joinpath(outdir, "visible_length_daily.csv"),
                   vg_enabled::Bool=false,
                   vg_every::Int=1,
                   vg_method::Symbol=:exact,
                   vg_dx::Float64=2.0,
                   vg_dz::Float64=2.0,
                   vg_sample_step_cm::Float64=0.2,
                   vg_raster_step_cm::Float64=0.2,
                   vg_y_min_cm::Float64=0.0,
                   vg_y_max_cm::Float64=0.5,
                   vg_csv_path::String=joinpath(outdir, "visible_grid_daily.csv"),
                   # --- daily tile snapshots (for calibration) ---
                   snapshot_enabled::Bool=false,
                   snapshot_nrows::Int=6,
                   snapshot_ncols::Int=6,
                   snapshot_y_max_cm::Float64=0.5,
                   snapshot_step_cm::Float64=0.2,
                   snapshot_csv_path::String=joinpath(outdir, "daily_tiles.csv"),
                   # --- Root depth estimation (for facilitation) ---
                   bdepth_enabled::Bool=false,
                   bdepth_species::String="B",
                   bdepth_csv_path::String=joinpath(outdir, "daily_B_max_depth_cm.csv"),
                   # --- progress bar ---
                   show_progress::Bool=true,
                   # --- facilitatin ---
                   facilitation::Union{Nothing,Facilitation}=nothing,
                   # --- minirhizotron tube observation ---
                   minirhizotron::Union{Nothing,Minirhizotron}=nothing,
                   mr_csv_path::String=joinpath(outdir, "tube_stations_daily.csv"),
                   # --- plate view (x-depth) output, independent of viz_enabled ---
                   viz_plate_enabled::Bool=false,
                   viz_plate_save_every::Int=0,
                   viz_plate_outdir::String=joinpath(outdir, "viz_plate"))

    Random.seed!(seed)
    mkpath(outdir)
    mkpath(plot3d_dir)

    # Ensure directories exist for snapshot output
    if snapshot_enabled
        mkpath(dirname(snapshot_csv_path))
    end

    # Overwrite (do not append) output CSVs if they already exist
    if snapshot_enabled
        for p in [snapshot_csv_path,
                  replace(snapshot_csv_path, ".csv" => "_wide.csv"),
                  replace(snapshot_csv_path, ".csv" => "_long.csv")]
            isfile(p) && rm(p; force=true)
        end
    end
    if vm_enabled && isfile(vm_csv_path)
        rm(vm_csv_path; force=true)
    end
    if vg_enabled && isfile(vg_csv_path)
        rm(vg_csv_path; force=true)
    end

    # Root depth estimation for species (to be used for driving facilitation)
    io_bdepth = nothing
    if bdepth_enabled
        mkpath(dirname(bdepth_csv_path))
        isfile(bdepth_csv_path) && rm(bdepth_csv_path; force=true)
        io_bdepth = open(bdepth_csv_path, "w")
        println(io_bdepth, "day,species,max_depth_cm")
    end

    if save_every > 0 && !_MAKIE_AVAILABLE[]
        error("save_every > 0 but Makie isn't available. Install CairoMakie or set save_every=0.")
    end

    io = nothing
    if vm_enabled
        io = open(vm_csv_path, "w")
        println(io, "day,scope,plant_id,visible_length_cm")
    end

    io_grid = nothing
    if vg_enabled
        io_grid = open(vg_csv_path, "w")
        println(io_grid, "day,method,ix,iz,x_cm,depth_cm,length_cm,area_cm2,rld_cm_per_cm2,slab_thickness_cm,rld_cm_per_cm3")
    end

    io_mr = nothing
    mr_active = minirhizotron !== nothing && minirhizotron.enabled && length(minirhizotron.station_positions_cm) > 0
    if mr_active
        isfile(mr_csv_path) && rm(mr_csv_path; force=true)
        d = dirname(mr_csv_path); d != "" && mkpath(d)
        io_mr = open(mr_csv_path, "w")
        if isempty(minirhizotron.camera_azimuths_deg)
            println(io_mr, "day,station_idx,along_tube_cm,x_cm,y_cm,depth_cm,length_cm")
        else
            println(io_mr, "day,station_idx,along_tube_cm,x_cm,y_cm,depth_cm,azimuth_deg,length_cm")
        end
    end

    b_reach_day = nothing  # ::Union{Nothing,Int}

    # # find B plants (indices)
    # B_ids = [i for (i,pl) in enumerate(plants) if pl.species == facilitation.source_species]
    # b_reach_day = fill(typemax(Int), length(B_ids))   # typemax means “not reached yet”
    b_reach_day = Dict{Int,Int}()   # pid => first day it reached trigger depth

    p = show_progress ? Progress(days; desc="Simulating days", dt=0.2) : nothing

    for day in 1:days

        # gate = 0.0
        # if facilitation !== nothing && facilitation.enabled
        #     # 1) compute current B max depth (cm, positive downward)
        #     Bmax = species_max_depth_cm(plants, facilitation.source_species)

        #     # 2) record the first day B reaches the trigger depth
        #     if b_reach_day === nothing && Bmax >= facilitation.trigger_depth_cm
        #         b_reach_day = day
        #     end

        #     # 3) Convert reach day -> gate with delay + optional ramp
        #     if b_reach_day !== nothing
        #         start_day = b_reach_day + facilitation.trigger_delay_days

        #         if day < start_day
        #             gate = 0.0
        #         elseif facilitation.trigger_ramp_days <= 0
        #             gate = 1.0
        #         else
        #             gate = clamp((day - start_day) / facilitation.trigger_ramp_days, 0.0, 1.0)
        #         end
        #     end
        # end

        # gates = zeros(Float64, length(B_ids))

        # for (k, idx) in enumerate(B_ids)
        #     dcm = plant_max_depth_cm(plants[idx])
        #     if b_reach_day[k] == typemax(Int) && dcm >= facilitation.trigger_depth_cm
        #         b_reach_day[k] = day
        #     end

        #     if b_reach_day[k] != typemax(Int)
        #         start_day = b_reach_day[k] + facilitation.trigger_delay_days
        #         if day < start_day
        #             gates[k] = 0.0
        #         elseif facilitation.trigger_ramp_days <= 0
        #             gates[k] = 1.0
        #         else
        #             gates[k] = clamp((day - start_day)/facilitation.trigger_ramp_days, 0.0, 1.0)
        #         end
        #     end
        # end

        gates_by_pid = nothing

        if facilitation !== nothing && facilitation.enabled
            gates_by_pid = Dict{Int,Float64}()

            for (pid, xs, zs) in facilitation.source_xz
                # compute B depth for that specific plant pid
                dcm = plant_max_depth_cm(plants[pid])

                # record first reach day
                if !haskey(b_reach_day, pid) && dcm >= facilitation.trigger_depth_cm
                    b_reach_day[pid] = day
                end

                gate_i = 0.0
                if haskey(b_reach_day, pid)
                    start_day = b_reach_day[pid] + facilitation.trigger_delay_days
                    if day < start_day
                        gate_i = 0.0
                    elseif facilitation.trigger_ramp_days <= 0
                        gate_i = 1.0
                    else
                        gate_i = clamp((day - start_day) / facilitation.trigger_ramp_days, 0.0, 1.0)
                    end
                end

                gates_by_pid[pid] = gate_i
            end
        end

        
        if reg.enabled
            R_all, R_by_species, nx, nz = build_rld_grids(plants, box, reg)
            for pl in plants
                R_self = get(R_by_species, pl.species, zeros(size(R_all)))
                step_plant!(pl, box, reg, R_all, R_self, nx, nz; mature_branching=mature_branching, facilitation=facilitation, facil_gates=gates_by_pid, minirhizotron=minirhizotron)
            end
        else
            R_all = zeros(1,1)
            nx = 1; nz = 1
            for pl in plants
                step_plant!(pl, box, reg, R_all, R_all, nx, nz; mature_branching=mature_branching, facilitation=facilitation, facil_gates=gates_by_pid, minirhizotron=minirhizotron)
            end
        end

        if save_every > 0 && day % save_every == 0
            save_root_figure(plants, box, day; outdir=plot3d_dir, minirhizotron=minirhizotron)
        end

        if viz_enabled && viz_save_every > 0 && day % viz_save_every == 0
            if mr_active
                save_tube_unrolled_figure(plants, minirhizotron, day; outdir=viz_outdir)
            else
                save_root_figure_2d(plants, box, day;
                    outdir=viz_outdir,
                    project_to_plate=viz_project_to_plate,
                    y_max_cm=viz_y_max_cm,
                    linewidth=viz_linewidth,
                    minirhizotron=minirhizotron)
            end
        end

        if viz_plate_enabled && viz_plate_save_every > 0 && day % viz_plate_save_every == 0
            save_root_figure_2d(plants, box, day;
                outdir=viz_plate_outdir,
                project_to_plate=true,
                linewidth=0.8,
                yreversed=false,
                tube_axis_only=true,
                minirhizotron=minirhizotron)
        end

        if vm_enabled
            append_daily_visible!(io, day, plants; y_max_cm=vm_y_max_cm)
        end

        if vg_enabled && (day % vg_every == 0)
            append_daily_visible_grid!(io_grid, day, plants, box;
                method=vg_method,
                y_min_cm=vg_y_min_cm, y_max_cm=vg_y_max_cm,
                dx=vg_dx, dz=vg_dz,
                sample_step_cm=vg_sample_step_cm,
                raster_step_cm=vg_raster_step_cm
            )
        end

        if snapshot_enabled
            tiles = visible_tiles_mm(plants, box;
                nrows=snapshot_nrows,
                ncols=snapshot_ncols,
                y_max_cm=snapshot_y_max_cm,
                step_cm=snapshot_step_cm
            )
            snapshot_csv_long = replace(snapshot_csv_path, ".csv" => "_long.csv")
            snapshot_csv_wide = replace(snapshot_csv_path, ".csv" => "_wide.csv")

            append_daily_tiles!(snapshot_csv_wide, day, tiles)
            append_daily_tiles_long!(snapshot_csv_long, day, tiles)
        end

        if bdepth_enabled
            dcm = species_max_depth_cm(plants, bdepth_species)
            println(io_bdepth, "$(day),$(bdepth_species),$(dcm)")
        end

        if mr_active
            append_daily_tube_stations!(io_mr, day, plants, minirhizotron)
        end

        show_progress && next!(p)
    end

    if vm_enabled
        close(io)
    end

    if vg_enabled
        close(io_grid)
    end

    if bdepth_enabled
        close(io_bdepth)
    end

    if mr_active
        close(io_mr)
    end

    return plants
end

# ----------------------------
# Main
# ----------------------------
