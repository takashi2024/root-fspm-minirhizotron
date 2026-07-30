# FovRenderer.jl
#
# Renders a single 5×5 cm field-of-view (FOV) image as a binary (black-and-white)
# Matrix{UInt8} for one (station_idx, azimuth_idx) observation window.
#
# Projection geometry is IDENTICAL to tube_station_azimuth_lengths_cm in
# src/MetricsAndOutput.jl: same _tube_project, same _tube_azimuth, same station
# and azimuth window gate conditions.  The only difference is that instead of
# accumulating root length, we rasterise each visible root sample to a pixel.
#
# Coordinate layout of the output image:
#   Width  axis (columns): along-tube position t
#     column 1 = s_pos - image_along_cm/2   (near edge of station window)
#     column W = s_pos + image_along_cm/2   (far edge)
#   Height axis (rows):    arc offset from azimuth centre (cm on tube surface)
#     row 1 = -image_arc_cm/2  (bottom of azimuth window)
#     row H = +image_arc_cm/2  (top of azimuth window)
#
# Dependencies: Types.jl, MetricsAndOutput.jl (for _tube_project, _tube_azimuth, _norm)
# No Makie required.

using FileIO
using Images: Gray

"""
    render_fov_image_bw(plants, mr, station_idx, azimuth_idx;
                        img_height, img_width, render_step_cm) -> Matrix{UInt8}

Render a binary (0=background, 0xFF=root) image for the FOV window at
`(station_idx, azimuth_idx)`.  Uses finer sampling than the metric functions
(`render_step_cm=0.05` vs metric default 0.1 cm) to reduce pixel gaps on
thin roots.
"""
function render_fov_image_bw(
    plants::Vector{Plant},
    mr::Minirhizotron,
    station_idx::Int,
    azimuth_idx::Int;
    img_height::Int        = 64,
    img_width::Int         = 64,
    render_step_cm::Float64 = 0.05
)::Matrix{UInt8}

    img          = zeros(UInt8, img_height, img_width)
    s_pos        = mr.station_positions_cm[station_idx]
    az_center    = mr.camera_azimuths_deg[azimuth_idx]
    half_h       = mr.image_along_cm / 2.0
    # identical formula to tube_station_azimuth_lengths_cm
    half_arc_deg = (mr.image_arc_cm / (2π * mr.r_outer_cm)) * 180.0
    rlo          = mr.r_outer_cm
    rhi          = mr.r_outer_cm + mr.skin_depth_cm

    t_lo         = s_pos - half_h
    arc_half_cm  = mr.image_arc_cm / 2.0

    for pl in plants
        for seg in pl.root.segments
            v = seg.p1 .- seg.p0
            L = _norm(v)
            L < 1e-12 && continue
            n = max(2, Int(ceil(L / render_step_cm)))
            for k in 0:(n - 1)
                p    = seg.p0 .+ ((k + 0.5) / n) .* v
                t, r = _tube_project(p, mr.origin, mr.direction)

                # Gate 1: radial shell (identical to metric function)
                rlo <= r <= rhi || continue
                # Gate 2: station window (identical to metric function)
                abs(t - s_pos) <= half_h || continue
                # Gate 3: azimuth window (identical to metric function)
                az       = _tube_azimuth(p, mr.origin, mr.direction)
                az_diff  = mod(az - az_center + 180.0, 360.0) - 180.0
                abs(az_diff) <= half_arc_deg || continue

                # Map to pixel coordinates
                # Along-tube → column
                col = clamp(
                    floor(Int, (t - t_lo) / mr.image_along_cm * img_width) + 1,
                    1, img_width
                )
                # Arc offset (cm) → row
                arc_offset_cm = az_diff * (π / 180.0) * mr.r_outer_cm
                row = clamp(
                    floor(Int, (arc_offset_cm + arc_half_cm) / mr.image_arc_cm * img_height) + 1,
                    1, img_height
                )
                img[row, col] = 0xFF
            end
        end
    end
    return img
end

"""
    save_fov_png(img, path)

Save a Matrix{UInt8} FOV image as a grayscale PNG.
"""
function save_fov_png(img::Matrix{UInt8}, path::AbstractString)
    mkpath(dirname(path))
    save(path, Gray.(img ./ 255f0))
end

"""
    render_and_save_all_fov(plants, mr, day, sample_id, outdir;
                             img_height, img_width, render_step_cm)
                             -> Matrix{UInt8} (ns × na × H × W packed as 4-D Array)

Render B&W FOV images for all (station, azimuth) pairs at one snapshot day.
Saves PNGs and returns the raw uint8 tensor of shape (ns, na, H, W).
"""
function render_and_save_all_fov(
    plants::Vector{Plant},
    mr::Minirhizotron,
    day::Int,
    sample_id::Int,
    outdir::String;
    img_height::Int        = 64,
    img_width::Int         = 64,
    render_step_cm::Float64 = 0.05,
    save_png::Bool         = true
)
    ns  = length(mr.station_positions_cm)
    na  = length(mr.camera_azimuths_deg)
    out = zeros(UInt8, ns, na, img_height, img_width)

    for si in 1:ns
        for ai in 1:na
            img = render_fov_image_bw(plants, mr, si, ai;
                                      img_height=img_height,
                                      img_width=img_width,
                                      render_step_cm=render_step_cm)
            out[si, ai, :, :] = img
            if save_png
                fname = joinpath(outdir,
                    @sprintf("s%04d_d%03d_st%02d_az%d.png",
                             sample_id, day, si, ai - 1))
                save_fov_png(img, fname)
            end
        end
    end
    return out
end
