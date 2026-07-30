# Auto-split from root_fspm_rhizobox_multiplant_SRL.jl
# File: DensityRegulation.jl

function build_rld_grid(plants::Vector{Plant}, box::Rhizobox, reg::DensityRegulation)
    dx = reg.grid_dx_cm
    dz = reg.grid_dz_cm
    nx, nz = _grid_size(box, dx, dz)

    L = zeros(Float64, nx, nz)

    for pl in plants
        for seg in pl.root.segments
            v = seg.p1 .- seg.p0
            segL = sqrt(sum(abs2, v))
            segL < 1e-12 && continue

            n = max(1, Int(ceil(segL / reg.sample_step_cm)))
            dl = segL / n

            for k in 0:(n-1)
                tmid = (k + 0.5) / n
                p = seg.p0 .+ tmid .* v
                ix, iz = _xz_to_idx(p[1], p[3], box, dx, dz, nx, nz)
                L[ix, iz] += dl
            end
        end
    end

    cell_area = dx * dz
    R = L ./ cell_area
    return R, nx, nz, cell_area
end

function build_rld_grids(plants::Vector{Plant}, box::Rhizobox, reg::DensityRegulation)
    R_all, nx, nz, _ = build_rld_grid(plants, box, reg)

    species = unique([pl.species for pl in plants])
    R_by_species = Dict{String, Matrix{Float64}}()
    for sp in species
        sp_plants = [pl for pl in plants if pl.species == sp]
        R_sp, _, _, _ = build_rld_grid(sp_plants, box, reg)
        R_by_species[sp] = R_sp
    end

    return R_all, R_by_species, nx, nz
end

function local_rld(R::Matrix{Float64}, x::Float64, z::Float64, box::Rhizobox, reg::DensityRegulation, nx::Int, nz::Int)
    dx = reg.grid_dx_cm
    dz = reg.grid_dz_cm
    r = reg.radius_cm

    ix0, iz0 = _xz_to_idx(x, z, box, dx, dz, nx, nz)

    rx = Int(ceil(r / dx))
    rz = Int(ceil(r / dz))

    s = 0.0
    w = 0.0
    for ix in max(1, ix0 - rx):min(nx, ix0 + rx)
        for iz in max(1, iz0 - rz):min(nz, iz0 + rz)
            xc = (-box.W/2) + (ix - 0.5) * dx
            dc = (iz - 0.5) * dz
            zc = -dc
            dist = sqrt((x - xc)^2 + (z - zc)^2)
            if dist <= r
                s += R[ix, iz]
                w += 1.0
            end
        end
    end
    return w > 0 ? (s / w) : 0.0
end

@inline function density_factor(rld::Float64; alpha::Float64, min_factor::Float64)
    f = 1.0 / (1.0 + alpha * max(rld, 0.0))
    return clamp(f, min_factor, 1.0)
end

# ----------------------------
# One day growth for one plant
# ----------------------------
