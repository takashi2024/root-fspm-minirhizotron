"""
render_fov_images.jl

Generates 5×5 cm minirhizotron station FOV images for Figure 2 using Julia's
FovRenderer (pixel-accurate root rasterization onto tube window geometry).

Runs a SINGLE B2 simulation to day 84 and renders FOV images at checkpoint days
[56, 70, 84] from within the same run.  Using a single run guarantees temporal
consistency: each checkpoint is a direct continuation of the previous, so root
pixels are strictly non-decreasing across time steps.

(The previous approach ran 3 separate simulate!() calls from scratch, which caused
apparent "disappearing roots" because each independent run produces slightly
different plant states — even with the same seed — due to accumulated floating-point
differences in file-I/O timing and other non-deterministic paths.)

Output: 36 PNG files in output_fig2_tmp/fov_images/
  Format: d{DAY:03d}_st{STATION:02d}_az{AZIMUTH}.png
  e.g.    d056_st05_az0.png, d084_st13_az3.png

Run from project root:
    julia publication/scripts/render_fov_images.jl
"""

const PROJ = abspath(joinpath(@__DIR__, "..", ".."))

include(joinpath(PROJ, "src", "Deps.jl"))
include(joinpath(PROJ, "src", "Types.jl"))
include(joinpath(PROJ, "src", "Geometry.jl"))
include(joinpath(PROJ, "src", "DensityRegulation.jl"))
include(joinpath(PROJ, "src", "GrowthStep.jl"))
include(joinpath(PROJ, "src", "MetricsAndOutput.jl"))
include(joinpath(PROJ, "src", "Plotting.jl"))
include(joinpath(PROJ, "src", "Config.jl"))
include(joinpath(PROJ, "src", "Simulation.jl"))
include(joinpath(PROJ, "sbi", "julia", "FovRenderer.jl"))

const ENV_JSON  = joinpath(PROJ, "calibration", "fig1_env_D2_281_2mm.json")
const FOV_DAYS  = Set([56, 70, 84])   # days to render (within single run)
const FOV_STNS  = [5, 9, 13]         # 1-based station indices
const IMG_H     = 256
const IMG_W     = 256
const STEP_CM   = 0.02               # finer than metric default for dense pixels
const OUTDIR    = joinpath(PROJ, "output_fig2_tmp", "fov_images")
const DAYS_MAX  = maximum(FOV_DAYS)

mkpath(OUTDIR)

cfg = load_environment_and_plants(ENV_JSON)

println("Single-run FOV rendering  seed=$(cfg.seed)  days 1–$(DAYS_MAX)")
println("Checkpoints: $(sort(collect(FOV_DAYS)))  |  stations: $FOV_STNS  |  azimuths: 0–3")
println()

Random.seed!(cfg.seed)

for day in 1:DAYS_MAX
    # One growth step per plant.
    # Mirrors the density-regulation-disabled branch of simulate!
    # (density_regulation.enabled = false in fig1_env_B2_143.json).
    R_all = zeros(1, 1)
    nx = 1;  nz = 1
    for pl in cfg.plants
        step_plant!(pl, cfg.box, cfg.reg, R_all, R_all, nx, nz;
                    mature_branching = false,
                    minirhizotron    = cfg.mr)
    end

    if day in FOV_DAYS
        println("  Day $day — rendering $(length(FOV_STNS) * 4) images...")
        for si in FOV_STNS
            for ai in 1:4
                img = render_fov_image_bw(cfg.plants, cfg.mr, si, ai;
                          img_height     = IMG_H,
                          img_width      = IMG_W,
                          render_step_cm = STEP_CM)
                fname = joinpath(OUTDIR,
                            "d$(lpad(day, 3, '0'))_st$(lpad(si, 2, '0'))_az$(ai - 1).png")
                save_fov_png(img, fname)
            end
        end
        println("  Day $day done → $(length(FOV_STNS) * 4) images saved")
    end
end

println()
println("All FOV images saved to $(abspath(OUTDIR))")
