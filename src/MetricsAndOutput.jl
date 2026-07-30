# File: MetricsAndOutput.jl

using CSV
using DataFrames

function total_length_cm(root::RootSystem)
    s = 0.0
    for seg in root.segments
        s += segment_length_cm(seg)
    end
    return s
end

function predicted_root_biomass_g(pl::Plant)
    srl = pl.params.srl_m_per_g
    srl <= 0 && return NaN
    L_m = total_length_cm(pl.root) / 100.0
    return L_m / srl
end

function export_segments_csv(plants::Vector{Plant}, filepath::AbstractString)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        println(io, "plant_id,species,x0,y0,z0,x1,y1,z1,order,age,tip_age")
        for pl in plants
            for seg in pl.root.segments
                @printf(io, "%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d\n",
                        pl.id, pl.species,
                        seg.p0[1], seg.p0[2], seg.p0[3],
                        seg.p1[1], seg.p1[2], seg.p1[3],
                        seg.order, seg.age, seg.tip_age)
            end
        end
    end
    return filepath
end

function export_pointcloud_ply(path::AbstractString, pts::AbstractMatrix{<:Real})
    # pts: N×3 matrix (x,y,z)
    @assert size(pts, 2) == 3 "pts must be N×3"

    open(path, "w") do io
        n = size(pts, 1)
        # ASCII PLY (simple, robust)
        write(io, "ply\n")
        write(io, "format ascii 1.0\n")
        write(io, "element vertex $n\n")
        write(io, "property float x\n")
        write(io, "property float y\n")
        write(io, "property float z\n")
        write(io, "end_header\n")

        for i in 1:n
            x = Float64(pts[i,1]); y = Float64(pts[i,2]); z = Float64(pts[i,3])
            isfinite(x) && isfinite(y) && isfinite(z) || continue
            @printf(io, "%.6f %.6f %.6f\n", x, y, z)
        end
    end
end

function export_pointcloud_xyz(plants::Vector{Plant}, filepath::AbstractString; step_cm::Float64=0.1, add_fields::Bool=true)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        if add_fields
            println(io, "//X Y Z plant_id order age")
        else
            println(io, "//X Y Z")
        end
        for pl in plants
            for seg in pl.root.segments
                v = seg.p1 .- seg.p0
                L = _norm(v)
                L < 1e-12 && continue
                n = max(1, Int(floor(L / step_cm)))
                for k in 0:n
                    t = k / n
                    p = seg.p0 .+ t .* v
                    if add_fields
                        @printf(io, "%.6f %.6f %.6f %d %d %d\n", p[1], p[2], p[3], pl.id, seg.order, seg.age)
                    else
                        @printf(io, "%.6f %.6f %.6f\n", p[1], p[2], p[3])
                    end
                end
            end
        end
    end
    return filepath
end

function export_root_summary_csv(plants::Vector{Plant}, filepath::AbstractString)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        println(io, "plant_id,species,total_length_cm,pred_biomass_g,obs_biomass_g,rel_error")
        for pl in plants
            L = total_length_cm(pl.root)
            predB = predicted_root_biomass_g(pl)
            obsB = pl.obs_root_biomass_g === nothing ? NaN : pl.obs_root_biomass_g
            relerr = (isfinite(obsB) && obsB > 0 && isfinite(predB)) ? (predB - obsB) / obsB : NaN
            @printf(io, "%d,%s,%.6f,%.6f,%.6f,%.6f\n", pl.id, pl.species, L, predB, obsB, relerr)
        end
    end
    return filepath
end

# ============================================================
# Visible-root length in a near-glass slab: y ∈ [y_min, y_max]
# Exact clipping of each segment, then sum length inside.
# ============================================================
@inline function segment_length_cm_points(p0::Vector{Float64}, p1::Vector{Float64})
    return sqrt(sum(abs2, p1 .- p0))
end

function clip_segment_to_y_slab(p0::Vector{Float64}, p1::Vector{Float64}; y_min::Float64=0.0, y_max::Float64=0.5)
    y0 = p0[2]
    y1 = p1[2]
    dy = y1 - y0

    if abs(dy) < 1e-12
        if (y_min <= y0 <= y_max)
            return p0, p1, true
        else
            return p0, p1, false
        end
    end

    t_a = (y_min - y0) / dy
    t_b = (y_max - y0) / dy
    t_enter = min(t_a, t_b)
    t_exit  = max(t_a, t_b)

    t0 = max(0.0, t_enter)
    t1 = min(1.0, t_exit)

    if t1 <= t0
        return p0, p1, false
    end

    v = p1 .- p0
    q0 = p0 .+ t0 .* v
    q1 = p0 .+ t1 .* v
    return q0, q1, true
end

function visible_length_cm(plants::Vector{Plant}; y_max_cm::Float64=0.5, y_min_cm::Float64=0.0)
    total = 0.0
    for pl in plants
        for seg in pl.root.segments
            q0, q1, inside = clip_segment_to_y_slab(seg.p0, seg.p1; y_min=y_min_cm, y_max=y_max_cm)
            inside || continue
            total += segment_length_cm_points(q0, q1)
        end
    end
    return total
end

function visible_length_byplant(plants::Vector{Plant}; y_max_cm::Float64=0.5, y_min_cm::Float64=0.0)
    out = Dict{Int,Float64}()
    for pl in plants
        s = 0.0
        for seg in pl.root.segments
            q0, q1, inside = clip_segment_to_y_slab(seg.p0, seg.p1; y_min=y_min_cm, y_max=y_max_cm)
            inside || continue
            s += segment_length_cm_points(q0, q1)
        end
        out[pl.id] = s
    end
    return out
end

# ------------------------------------------------------------
# Visible-length grid (exact, midpoint accumulation)
# ------------------------------------------------------------
@inline function _bin_x_depth(x::Float64, z::Float64, box::Rhizobox, dx::Float64, dz::Float64, nx::Int, nz::Int)
    u = (x + box.W/2) / box.W
    ix = clamp(Int(floor(u * nx)) + 1, 1, nx)

    d = clamp(-z, 0.0, box.H)
    v = d / box.H
    iz = clamp(Int(floor(v * nz)) + 1, 1, nz)

    return ix, iz
end

function visible_length_grid(plants::Vector{Plant}, box::Rhizobox;
                             y_max_cm::Float64=0.5, y_min_cm::Float64=0.0,
                             dx::Float64=1.0, dz::Float64=1.0,
                             sample_step_cm::Float64=0.2,
                             include_byplant::Bool=false)

    nx = max(1, Int(ceil(box.W / dx)))
    nz = max(1, Int(ceil(box.H / dz)))
    L = zeros(Float64, nx, nz)
    L_byplant = include_byplant ? Dict{Int, Matrix{Float64}}() : Dict{Int, Matrix{Float64}}()

    if include_byplant
        for pl in plants
            L_byplant[pl.id] = zeros(Float64, nx, nz)
        end
    end

    for pl in plants
        for seg in pl.root.segments
            q0, q1, inside = clip_segment_to_y_slab(seg.p0, seg.p1; y_min=y_min_cm, y_max=y_max_cm)
            inside || continue

            v = q1 .- q0
            segL = sqrt(sum(abs2, v))
            segL < 1e-12 && continue

            n = max(1, Int(ceil(segL / sample_step_cm)))
            dl = segL / n

            for k in 0:(n-1)
                tmid = (k + 0.5) / n
                p = q0 .+ tmid .* v
                ix, iz = _bin_x_depth(p[1], p[3], box, dx, dz, nx, nz)
                L[ix, iz] += dl
                if include_byplant
                    L_byplant[pl.id][ix, iz] += dl
                end
            end
        end
    end

    return L, L_byplant, nx, nz
end

# ------------------------------------------------------------
# Rasterization + skeleton length (optional)
# ------------------------------------------------------------
function rasterize_visible_mask(plants::Vector{Plant}, box::Rhizobox;
    y_min_cm::Float64=0.0, y_max_cm::Float64=0.5,
    dx::Float64=0.1, dz::Float64=0.1,
    step_cm::Float64=0.05)

    nx = max(1, Int(ceil(box.W / dx)))
    nz = max(1, Int(ceil(box.H / dz)))
    mask = falses(nx, nz)

    @inline function bin_ix_iz(x::Float64, depth::Float64)
        u = (x + box.W/2) / box.W
        ix = clamp(Int(floor(u * nx)) + 1, 1, nx)
        v = depth / box.H
        iz = clamp(Int(floor(v * nz)) + 1, 1, nz)
        return ix, iz
    end

    for pl in plants
        for seg in pl.root.segments
            q0, q1, inside = clip_segment_to_y_slab(seg.p0, seg.p1; y_min=y_min_cm, y_max=y_max_cm)
            inside || continue

            x0 = q0[1]; d0 = -q0[3]
            x1 = q1[1]; d1 = -q1[3]

            vx = x1 - x0
            vd = d1 - d0
            L  = hypot(vx, vd)
            L < 1e-12 && continue

            n = max(1, Int(ceil(L / step_cm)))
            for k in 0:n
                t = k / n
                x = x0 + t*vx
                d = d0 + t*vd
                if 0.0 <= d <= box.H
                    ix, iz = bin_ix_iz(x, d)
                    mask[ix, iz] = true
                end
            end
        end
    end

    return mask, nx, nz
end

function skeleton_length_cm(skel::AbstractMatrix{Bool}, dx::Float64, dz::Float64)
    nx, nz = size(skel)
    L = 0.0

    for ix in 1:nx, iz in 1:nz
        skel[ix,iz] || continue

        if ix < nx && skel[ix+1, iz]
            L += dx
        end
        if iz < nz && skel[ix, iz+1]
            L += dz
        end
        if ix < nx && iz < nz && skel[ix+1, iz+1]
            L += hypot(dx, dz)
        end
        if ix > 1 && iz < nz && skel[ix-1, iz+1]
            L += hypot(dx, dz)
        end
    end

    return L
end

function visible_length_cm_raster(plants::Vector{Plant}, box::Rhizobox;
    y_min_cm::Float64=0.0, y_max_cm::Float64=0.5,
    dx::Float64=0.1, dz::Float64=0.1,
    step_cm::Float64=0.05)

    _IMG_AVAILABLE[] || error("ImageMorphology not available. Install with: julia -e \"import Pkg; Pkg.add(\\\"ImageMorphology\\\"); Pkg.add(\\\"Images\\\")\" ")

    mask, _, _ = rasterize_visible_mask(plants, box;
        y_min_cm=y_min_cm, y_max_cm=y_max_cm, dx=dx, dz=dz, step_cm=step_cm)

    skel = ImageMorphology.thinning(mask)
    return skeleton_length_cm(skel, dx, dz)
end

function visible_length_grid_raster(plants::Vector{Plant}, box::Rhizobox;
    y_min_cm::Float64=0.0, y_max_cm::Float64=0.5,
    dx::Float64=0.1, dz::Float64=0.1,
    step_cm::Float64=0.05,
    include_byplant::Bool=false)

    _IMG_AVAILABLE[] || error("ImageMorphology not available. Install with: julia -e \"import Pkg; Pkg.add(\\\"ImageMorphology\\\"); Pkg.add(\\\"Images\\\")\" ")

    # helper to compute Lgrid from a skeleton
    function _skeleton_to_Lgrid(skel::AbstractMatrix{Bool}, dx::Float64, dz::Float64)
        nx, nz = size(skel)
        Lgrid = zeros(Float64, nx, nz)
        for ix in 1:nx, iz in 1:nz
            skel[ix,iz] || continue
            if ix < nx && skel[ix+1, iz]
                Lgrid[ix,iz] += dx
            end
            if iz < nz && skel[ix, iz+1]
                Lgrid[ix,iz] += dz
            end
            if ix < nx && iz < nz && skel[ix+1, iz+1]
                Lgrid[ix,iz] += hypot(dx, dz)
            end
            if ix > 1 && iz < nz && skel[ix-1, iz+1]
                Lgrid[ix,iz] += hypot(dx, dz)
            end
        end
        return Lgrid
    end

    # ALL
    mask, nx, nz = rasterize_visible_mask(plants, box;
        y_min_cm=y_min_cm, y_max_cm=y_max_cm, dx=dx, dz=dz, step_cm=step_cm)
    skel = ImageMorphology.thinning(mask)
    L_all = _skeleton_to_Lgrid(skel, dx, dz)

    L_byplant = Dict{Int, Matrix{Float64}}()
    if include_byplant
        for pl in plants
            maskp, _, _ = rasterize_visible_mask([pl], box;
                y_min_cm=y_min_cm, y_max_cm=y_max_cm, dx=dx, dz=dz, step_cm=step_cm)
            skelp = ImageMorphology.thinning(maskp)
            L_byplant[pl.id] = _skeleton_to_Lgrid(skelp, dx, dz)
        end
    end

    return L_all, L_byplant, nx, nz
end

# ============================================================
# CSV export: visible-length grid
# ============================================================
function export_visible_grid_csv(plants::Vector{Plant}, box::Rhizobox, outpath::AbstractString;
    method::Symbol = :exact,
    include_byplant::Bool=false,
    y_min_cm::Float64=0.0, y_max_cm::Float64=0.5,
    dx::Float64=1.0, dz::Float64=1.0,
    sample_step_cm::Float64=0.2,
    raster_step_cm::Float64=0.2)

    mkpath(dirname(outpath))

    L_all = zeros(1,1)
    L_byplant = Dict{Int, Matrix{Float64}}()
    nx = 1; nz = 1

    if method == :exact
        L_all, L_byplant, nx, nz = visible_length_grid(plants, box;
            y_min_cm=y_min_cm, y_max_cm=y_max_cm,
            dx=dx, dz=dz,
            sample_step_cm=sample_step_cm,
            include_byplant=include_byplant)

    elseif method == :raster
        L_all, L_byplant, nx, nz = visible_length_grid_raster(plants, box;
            y_min_cm=y_min_cm, y_max_cm=y_max_cm,
            dx=dx, dz=dz,
            step_cm=raster_step_cm,
            include_byplant=include_byplant)

    else
        error("Unknown method=$method. Use :exact or :raster.")
    end

    area = dx * dz
    slabT = max(1e-12, y_max_cm - y_min_cm)
    vol = area * slabT

    # map plant id -> species for writing
    sp_of = Dict(pl.id => pl.species for pl in plants)

    open(outpath, "w") do io
        println(io, "method,scope,plant_id,species,ix,iz,x_cm,depth_cm,length_cm,area_cm2,rld_cm_per_cm2,slab_thickness_cm,rld_cm_per_cm3")

        # combined
        for iz in 1:nz, ix in 1:nx
            x_c = (-box.W/2) + (ix - 0.5) * dx
            depth_c = (iz - 0.5) * dz
            len = L_all[ix, iz]
            rld2 = len / area
            rld3 = len / vol
            @printf(io, "%s,all,,,%d,%d,%.6f,%.6f,%.6f,%.6f,%.10f,%.6f,%.10f\n",
                String(method), ix, iz, x_c, depth_c, len, area, rld2, slabT, rld3)
        end

        # per plant (optional)
        if include_byplant
            for (pid, M) in sort(collect(L_byplant); by=x->x[1])
                sp = get(sp_of, pid, "")
                for iz in 1:nz, ix in 1:nx
                    x_c = (-box.W/2) + (ix - 0.5) * dx
                    depth_c = (iz - 0.5) * dz
                    len = M[ix, iz]
                    rld2 = len / area
                    rld3 = len / vol
                    @printf(io, "%s,plant,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.10f,%.6f,%.10f\n",
                        String(method), pid, sp, ix, iz, x_c, depth_c, len, area, rld2, slabT, rld3)
                end
            end
        end
    end

    return outpath
end

# ============================================================
# Daily visible length export
# ============================================================
function append_daily_visible!(io, day::Int, plants::Vector{Plant}; y_max_cm::Float64=0.5, y_min_cm::Float64=0.0)
    vis_all = visible_length_cm(plants; y_max_cm=y_max_cm, y_min_cm=y_min_cm)
    bypl = visible_length_byplant(plants; y_max_cm=y_max_cm, y_min_cm=y_min_cm)

    for (pid, L) in sort(collect(bypl); by=x->x[1])
        println(io, "$day,plant,$pid,$L")
    end
    println(io, "$day,all,,$vis_all")
    return nothing
end

"""
    append_daily_visible_grid!(io, day, plants, box; method=:exact, ...)

Append a full x–depth visible-length grid for a given `day` to an *already-open* CSV `io`.

Output rows:
day,method,ix,iz,x_cm,depth_cm,length_cm,area_cm2,rld_cm_per_cm2,slab_thickness_cm,rld_cm_per_cm3
"""
function append_daily_visible_grid!(io, day::Int, plants::Vector{Plant}, box::Rhizobox;
    method::Symbol = :exact,
    y_min_cm::Float64=0.0, y_max_cm::Float64=0.5,
    dx::Float64=1.0, dz::Float64=1.0,
    sample_step_cm::Float64=0.2,
    raster_step_cm::Float64=0.2
)
    # NOTE: both visible_length_grid and visible_length_grid_raster return:
    #   (L_all::Matrix{Float64}, L_byplant::Dict{Int,Matrix{Float64}}, nx::Int, nz::Int)
    L_all, _L_byplant, nx, nz = if method == :exact
        visible_length_grid(plants, box;
            y_min_cm=y_min_cm, y_max_cm=y_max_cm,
            dx=dx, dz=dz,
            sample_step_cm=sample_step_cm
        )
    elseif method == :raster
        visible_length_grid_raster(plants, box;
            y_min_cm=y_min_cm, y_max_cm=y_max_cm,
            dx=dx, dz=dz,
            step_cm=raster_step_cm
        )
    else
        error("Unknown method=$method. Use :exact or :raster.")
    end

    area = dx * dz
    slabT = max(1e-12, y_max_cm - y_min_cm)
    vol = area * slabT

    for iz in 1:nz, ix in 1:nx
        x_c = (-box.W/2) + (ix - 0.5) * dx
        depth_c = (iz - 0.5) * dz
        len = L_all[ix, iz]
        rld2 = len / area
        rld3 = len / vol
        @printf(io, "%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.10f,%.6f,%.10f\n",
            day, String(method), ix, iz, x_c, depth_c, len, area, rld2, slabT, rld3)
    end
    return nothing
end

# ============================================================
# Daily snapshot helpers (tile-wise visible length in mm)
# ============================================================

"""
Compute visible root length per tile (mm) for a rhizobox plate.

- x is left-right (centered in model coordinates; -W/2 .. +W/2)
- z is vertical (0 at soil surface, negative downward)
- y is thickness into soil (we count only 0..y_max_cm)

We sample each segment by stepping along it with step_cm (cm).
Returns a Matrix{Float64} of size (nrows, ncols) in mm.
"""
function visible_tiles_mm(plants::Vector{Plant}, box::Rhizobox;
                          nrows::Int=6, ncols::Int=6,
                          y_max_cm::Float64=0.5,
                          step_cm::Float64=0.2)

    # tile extents
    # x in model is centered: [-W/2, +W/2]
    x_min = -box.W/2
    x_max =  box.W/2
    z_top = 0.0
    z_bot = -box.H

    dx = (x_max - x_min) / ncols
    dz = (z_top - z_bot) / nrows  # positive

    tiles = zeros(Float64, nrows, ncols)  # mm

    # helper: map point -> tile index
    @inline function tile_index(x::Float64, z::Float64)
        c = Int(floor((x - x_min) / dx)) + 1
        r = Int(floor((z_top - z) / dz)) + 1
        return r, c
    end

    for pl in plants
        segs = pl.root.segments
        for seg in segs
            p0 = seg.p0
            p1 = seg.p1

            # filter by visibility slab (both endpoints must be within, or we can be permissive)
            # We'll be permissive: sample points and check y each time.
            v = p1 .- p0
            L = sqrt(sum(v .^ 2))
            if L <= 1e-9
                continue
            end

            # number of samples along segment
            ns = max(1, Int(ceil(L / step_cm)))
            # step vector
            dv = v ./ ns
            # each step contributes length L/ns (cm) -> (mm)
            dlen_mm = (L / ns) * 10.0

            p = copy(p0)
            for i in 1:ns
                # midpoint sampling is a bit nicer; but endpoint is OK
                p .= p .+ dv

                y = Float64(p[2])
                if 0.0 <= y <= y_max_cm
                    x = Float64(p[1])
                    z = Float64(p[3])

                    r, c = tile_index(x, z)
                    if 1 <= r <= nrows && 1 <= c <= ncols
                        tiles[r, c] += dlen_mm
                    end
                end
            end
        end
    end

    return tiles
end

"""
Append one day of tile results to CSV.

Columns:
day,total_visible_mm,tile_r1c1_mm,...,tile_rNcM_mm
"""
function append_daily_tiles!(csv_path::AbstractString, day::Int, tiles::AbstractMatrix{<:Real})
    nrows, ncols = size(tiles)

    # write header if file doesn't exist
    if !isfile(csv_path)
        open(csv_path, "w") do io
            print(io, "day,total_visible_mm")
            for r in 1:nrows, c in 1:ncols
                print(io, ",tile_r$(r)c$(c)_mm")
            end
            println(io)
        end
    end

    total = sum(Float64.(tiles))

    open(csv_path, "a") do io
        print(io, day, ",", total)
        for r in 1:nrows, c in 1:ncols
            print(io, ",", Float64(tiles[r, c]))
        end
        println(io)
    end

    return nothing
end


function tiles_to_long_df(tiles::AbstractMatrix{<:Real}; day::Int, pot_id::Union{Missing,Int}=missing)
    nrows, ncols = size(tiles)

    pot_ids = Union{Missing,Int}[]
    days    = Int[]
    rows    = Int[]
    cols    = Int[]
    preds   = Float64[]

    for r in 1:nrows, c in 1:ncols
        push!(pot_ids, pot_id)
        push!(days, day)
        push!(rows, r)
        push!(cols, c)
        push!(preds, Float64(tiles[r,c]))
    end

    return DataFrame(
        pot_id = pot_ids,
        day    = days,
        row    = rows,
        col    = cols,
        pred_mm = preds
    )
end

function append_daily_tiles_long!(csv_path::AbstractString, day::Int, tiles::AbstractMatrix{<:Real};
                                  pot_id::Union{Missing,Int}=missing)
    df = tiles_to_long_df(tiles; day=day, pot_id=pot_id)

    mkpath(dirname(csv_path))

    if isfile(csv_path)
        CSV.write(csv_path, df; append=true, writeheader=false)
    else
        CSV.write(csv_path, df)
    end

    return nothing
end


# ----------------------------
# Helper to compute max depth
# ----------------------------

# For each species
function species_max_depth_cm(plants::Vector{Plant}, species::String)
    # depth = -z, in cm. Return 0 if no segments yet.
    maxd = 0.0
    for pl in plants
        pl.species == species || continue
        for seg in pl.root.segments
            # consider both endpoints (safe)
            maxd = max(maxd, -seg.p0[3], -seg.p1[3])
        end
    end
    return maxd
end

# For each plant
function plant_max_depth_cm(pl::Plant)
    maxd = 0.0
    for seg in pl.root.segments
        maxd = max(maxd, -seg.p0[3], -seg.p1[3])
    end
    return maxd
end


# ============================================================
# Minirhizotron tube visibility
# ============================================================

# Project point p onto the tube axis; return (t=along-axis cm, r=radial distance cm).
@inline function _tube_project(p::AbstractVector{<:Real},
                                origin::Vector{Float64}, dir::Vector{Float64})
    dp = p .- origin
    t  = dot(dp, dir)
    r  = _norm(p .- (origin .+ t .* dir))
    return t, r
end

# Total root length (cm) visible from the tube — all sample points within the
# annular shell [r_outer, r_outer + skin_depth] and within [0, tube_length].
function tube_visible_length_cm(plants::Vector{Plant}, mr::Minirhizotron)
    total = 0.0
    rlo   = mr.r_outer_cm
    rhi   = mr.r_outer_cm + mr.skin_depth_cm
    step  = mr.sample_step_cm
    for pl in plants
        for seg in pl.root.segments
            v = seg.p1 .- seg.p0
            L = _norm(v)
            L < 1e-12 && continue
            n  = max(1, Int(ceil(L / step)))
            dl = L / n
            for k in 0:(n-1)
                tmid = (k + 0.5) / n
                p = seg.p0 .+ tmid .* v
                t, r = _tube_project(p, mr.origin, mr.direction)
                if 0.0 <= t <= mr.length_cm && rlo <= r <= rhi
                    total += dl
                end
            end
        end
    end
    return total
end

# Per-station root length (cm). Station si captures roots within
# |t - station_positions_cm[si]| <= image_along_cm/2 in the tube vicinity.
function tube_station_lengths_cm(plants::Vector{Plant}, mr::Minirhizotron)
    ns = length(mr.station_positions_cm)
    ns == 0 && return Float64[]
    half_h = mr.image_along_cm / 2.0
    rlo    = mr.r_outer_cm
    rhi    = mr.r_outer_cm + mr.skin_depth_cm
    step   = mr.sample_step_cm
    lengths = zeros(Float64, ns)

    for pl in plants
        for seg in pl.root.segments
            v = seg.p1 .- seg.p0
            L = _norm(v)
            L < 1e-12 && continue
            n  = max(1, Int(ceil(L / step)))
            dl = L / n
            for k in 0:(n-1)
                tmid = (k + 0.5) / n
                p = seg.p0 .+ tmid .* v
                t, r = _tube_project(p, mr.origin, mr.direction)
                rlo <= r <= rhi || continue
                0.0 <= t <= mr.length_cm || continue
                for si in 1:ns
                    if abs(t - mr.station_positions_cm[si]) <= half_h
                        lengths[si] += dl
                        break  # count each sample point once
                    end
                end
            end
        end
    end
    return lengths
end

# Azimuth (0–360°) of a point relative to the tube cross-section.
# 0° = toward world sky (projected into cross-section plane); increases CW when viewed from outside.
function _tube_azimuth(p::Vector{Float64}, origin::Vector{Float64}, direction::Vector{Float64})
    dp     = p .- origin
    axial  = dot(dp, direction)
    radial = dp .- axial .* direction

    # e_up = component of world-up [0,0,1] perpendicular to tube axis
    world_up  = [0.0, 0.0, 1.0]
    e_up_raw  = world_up .- dot(world_up, direction) .* direction
    nu = _norm(e_up_raw)
    if nu < 1e-10
        # tube is vertical — use world y as fallback
        alt       = [0.0, 1.0, 0.0]
        e_up_raw  = alt .- dot(alt, direction) .* direction
        nu        = _norm(e_up_raw)
    end
    e_up = e_up_raw ./ nu

    # e_right = cross(direction, e_up) — completes right-hand frame
    e_right = [direction[2]*e_up[3] - direction[3]*e_up[2],
               direction[3]*e_up[1] - direction[1]*e_up[3],
               direction[1]*e_up[2] - direction[2]*e_up[1]]
    nr = _norm(e_right)
    nr < 1e-10 && return 0.0
    e_right = e_right ./ nr

    comp_up    = dot(radial, e_up)
    comp_right = dot(radial, e_right)
    return mod(atan(comp_right, comp_up) * 180.0 / π, 360.0)
end

# Per-station × per-azimuth root length (cm). Returns Matrix (ns × na).
# Each azimuth window spans ±(image_arc_cm / (2π r_outer) × 180°) around camera_azimuths_deg[ai].
function tube_station_azimuth_lengths_cm(plants::Vector{Plant}, mr::Minirhizotron)
    ns = length(mr.station_positions_cm)
    na = length(mr.camera_azimuths_deg)
    (ns == 0 || na == 0) && return zeros(Float64, 0, 0)

    half_h       = mr.image_along_cm / 2.0
    half_arc_deg = (mr.image_arc_cm / (2π * mr.r_outer_cm)) * 180.0
    rlo          = mr.r_outer_cm
    rhi          = mr.r_outer_cm + mr.skin_depth_cm
    step         = mr.sample_step_cm
    lengths      = zeros(Float64, ns, na)

    for pl in plants
        for seg in pl.root.segments
            v = seg.p1 .- seg.p0
            L = _norm(v)
            L < 1e-12 && continue
            n  = max(1, Int(ceil(L / step)))
            dl = L / n
            for k in 0:(n-1)
                tmid = (k + 0.5) / n
                p    = seg.p0 .+ tmid .* v
                t, r = _tube_project(p, mr.origin, mr.direction)
                rlo <= r <= rhi || continue
                0.0 <= t <= mr.length_cm || continue

                si = 0
                for s in 1:ns
                    if abs(t - mr.station_positions_cm[s]) <= half_h
                        si = s; break
                    end
                end
                si == 0 && continue

                az = _tube_azimuth(p, mr.origin, mr.direction)
                ai = 0
                for a in 1:na
                    diff = abs(mod(az - mr.camera_azimuths_deg[a] + 180.0, 360.0) - 180.0)
                    if diff <= half_arc_deg
                        ai = a; break
                    end
                end
                ai == 0 && continue

                lengths[si, ai] += dl
            end
        end
    end
    return lengths
end

# Bin edges for crossing-angle histograms (degrees). 6 bins: [0-15), [15-30), ..., [75-90].
const ANGLE_BIN_EDGES = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
const N_ANGLE_BINS    = 6

# Crossing angle of a segment relative to the tube axis (0°=parallel, 90°=perpendicular).
@inline function _segment_crossing_angle_deg(seg_dir::Vector{Float64},
                                              tube_dir::Vector{Float64})::Float64
    acos(clamp(abs(dot(seg_dir, tube_dir)), 0.0, 1.0)) * 180.0 / π
end

@inline function _angle_bin(θ_deg::Float64)::Int
    for b in 1:(N_ANGLE_BINS-1)
        θ_deg < ANGLE_BIN_EDGES[b+1] && return b
    end
    return N_ANGLE_BINS
end

"""
Length-weighted mean crossing angle per (station, azimuth) window.
Returns an ns×na Matrix{Float64}. Entries are NaN where no roots are visible.
"""
function tube_station_azimuth_angles_cm(plants::Vector{Plant}, mr::Minirhizotron)
    ns = length(mr.station_positions_cm)
    na = length(mr.camera_azimuths_deg)
    (ns == 0 || na == 0) && return fill(NaN, 0, 0)

    half_h       = mr.image_along_cm / 2.0
    half_arc_deg = (mr.image_arc_cm / (2π * mr.r_outer_cm)) * 180.0
    rlo          = mr.r_outer_cm
    rhi          = mr.r_outer_cm + mr.skin_depth_cm
    step         = mr.sample_step_cm
    angle_sum    = zeros(Float64, ns, na)
    length_sum   = zeros(Float64, ns, na)

    for pl in plants
        for seg in pl.root.segments
            v = seg.p1 .- seg.p0
            L = _norm(v)
            L < 1e-12 && continue
            n  = max(1, Int(ceil(L / step)))
            dl = L / n
            θ  = _segment_crossing_angle_deg(seg.direction, mr.direction)
            for k in 0:(n-1)
                tmid = (k + 0.5) / n
                p    = seg.p0 .+ tmid .* v
                t, r = _tube_project(p, mr.origin, mr.direction)
                rlo <= r <= rhi || continue
                0.0 <= t <= mr.length_cm || continue
                si = 0
                for s in 1:ns
                    if abs(t - mr.station_positions_cm[s]) <= half_h
                        si = s; break
                    end
                end
                si == 0 && continue
                az = _tube_azimuth(p, mr.origin, mr.direction)
                ai = 0
                for a in 1:na
                    diff = abs(mod(az - mr.camera_azimuths_deg[a] + 180.0, 360.0) - 180.0)
                    if diff <= half_arc_deg
                        ai = a; break
                    end
                end
                ai == 0 && continue
                angle_sum[si, ai]  += θ * dl
                length_sum[si, ai] += dl
            end
        end
    end
    result = fill(NaN, ns, na)
    for si in 1:ns, ai in 1:na
        length_sum[si, ai] > 1e-12 && (result[si, ai] = angle_sum[si, ai] / length_sum[si, ai])
    end
    return result
end

"""
Length-weighted normalized angle histogram per (station, azimuth) window.
Returns an ns×na×N_ANGLE_BINS Array{Float64,3}.
Each [si,ai,:] slice sums to 1.0 where roots are present, NaN where no roots.
Bins: [0-15), [15-30), [30-45), [45-60), [60-75), [75-90] degrees.
"""
function tube_station_azimuth_angle_histograms(plants::Vector{Plant}, mr::Minirhizotron)
    ns = length(mr.station_positions_cm)
    na = length(mr.camera_azimuths_deg)
    (ns == 0 || na == 0) && return fill(NaN, 0, 0, N_ANGLE_BINS)

    half_h       = mr.image_along_cm / 2.0
    half_arc_deg = (mr.image_arc_cm / (2π * mr.r_outer_cm)) * 180.0
    rlo          = mr.r_outer_cm
    rhi          = mr.r_outer_cm + mr.skin_depth_cm
    step         = mr.sample_step_cm
    hist_sum     = zeros(Float64, ns, na, N_ANGLE_BINS)
    length_sum   = zeros(Float64, ns, na)

    for pl in plants
        for seg in pl.root.segments
            v = seg.p1 .- seg.p0
            L = _norm(v)
            L < 1e-12 && continue
            n  = max(1, Int(ceil(L / step)))
            dl = L / n
            θ  = _segment_crossing_angle_deg(seg.direction, mr.direction)
            b  = _angle_bin(θ)
            for k in 0:(n-1)
                tmid = (k + 0.5) / n
                p    = seg.p0 .+ tmid .* v
                t, r = _tube_project(p, mr.origin, mr.direction)
                rlo <= r <= rhi || continue
                0.0 <= t <= mr.length_cm || continue
                si = 0
                for s in 1:ns
                    if abs(t - mr.station_positions_cm[s]) <= half_h
                        si = s; break
                    end
                end
                si == 0 && continue
                az = _tube_azimuth(p, mr.origin, mr.direction)
                ai = 0
                for a in 1:na
                    diff = abs(mod(az - mr.camera_azimuths_deg[a] + 180.0, 360.0) - 180.0)
                    if diff <= half_arc_deg
                        ai = a; break
                    end
                end
                ai == 0 && continue
                hist_sum[si, ai, b] += dl
                length_sum[si, ai]  += dl
            end
        end
    end
    result = fill(NaN, ns, na, N_ANGLE_BINS)
    for si in 1:ns, ai in 1:na
        lt = length_sum[si, ai]
        lt > 1e-12 && (result[si, ai, :] = hist_sum[si, ai, :] ./ lt)
    end
    return result
end

# Append one day of per-station root lengths to an open CSV io.
# 360° mode header:  day,station_idx,along_tube_cm,x_cm,y_cm,depth_cm,length_cm
# Azimuth mode header: day,station_idx,along_tube_cm,x_cm,y_cm,depth_cm,azimuth_deg,length_cm
function append_daily_tube_stations!(io, day::Int, plants::Vector{Plant}, mr::Minirhizotron)
    ns = length(mr.station_positions_cm)
    ns == 0 && return nothing

    if isempty(mr.camera_azimuths_deg)
        lengths = tube_station_lengths_cm(plants, mr)
        for si in 1:ns
            s_pos    = mr.station_positions_cm[si]
            p_sta    = mr.origin .+ s_pos .* mr.direction
            depth_cm = -p_sta[3]
            @printf(io, "%d,%d,%.4f,%.4f,%.4f,%.4f,%.8f\n",
                day, si, s_pos, p_sta[1], p_sta[2], depth_cm, lengths[si])
        end
    else
        lengths = tube_station_azimuth_lengths_cm(plants, mr)
        na = length(mr.camera_azimuths_deg)
        for si in 1:ns
            s_pos    = mr.station_positions_cm[si]
            p_sta    = mr.origin .+ s_pos .* mr.direction
            depth_cm = -p_sta[3]
            for ai in 1:na
                az = mr.camera_azimuths_deg[ai]
                @printf(io, "%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.8f\n",
                    day, si, s_pos, p_sta[1], p_sta[2], depth_cm, az, lengths[si, ai])
            end
        end
    end
    return nothing
end

# ----------------------------
# Optional plotting (Makie)
# ----------------------------
