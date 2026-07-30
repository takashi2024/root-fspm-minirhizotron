# Auto-split from root_fspm_rhizobox_multiplant_SRL.jl
# File: Geometry.jl

# ----------------------------
# Vector helpers
# ----------------------------
@inline _norm(v::AbstractVector{<:Real}) = sqrt(sum(abs2, v))

@inline function normalize_vec(v::AbstractVector{<:Real}; eps::Float64=1e-12)
    n = _norm(v)
    return n < eps ? Vector{Float64}(v) : Vector{Float64}(v ./ n)
end

@inline safe_cross(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}) = cross(Vector{Float64}(a), Vector{Float64}(b))

rot_x(θ) = [
    1.0  0.0       0.0;
    0.0  cos(θ)  -sin(θ);
    0.0  sin(θ)   cos(θ)
]

function gravity_in_box(box::Rhizobox)
    R = rot_x(box.tilt_from_vertical)   # box -> world
    g_world = [0.0, 0.0, -1.0]
    g_box = transpose(R) * g_world
    return normalize_vec(g_box)
end

# ----------------------------
# Boundary handling
# ----------------------------
@inline function inside_box(p::AbstractVector{<:Real}, box::Rhizobox)
    x, y, z = p
    return (-box.W/2 ≤ x ≤ box.W/2) && (0.0 ≤ y ≤ box.D) && (-box.H ≤ z ≤ 0.0)
end

@inline function clamp_to_box(p::AbstractVector{<:Real}, box::Rhizobox)
    return [
        clamp(p[1], -box.W/2,  box.W/2),
        clamp(p[2],  0.0,      box.D),
        clamp(p[3], -box.H,    0.0)
    ]
end

function slide_direction(dir::Vector{Float64}, p_clamped::Vector{Float64}, p_raw::Vector{Float64}; eps::Float64=1e-12)
    d = copy(dir)
    if p_raw[1] != p_clamped[1]; d[1] = 0.0; end
    if p_raw[2] != p_clamped[2]; d[2] = 0.0; end
    if p_raw[3] != p_clamped[3]; d[3] = 0.0; end
    n = _norm(d)
    return n < eps ? normalize_vec(dir) : (d ./ n)
end

@inline _wrap_coord(v::Float64, lo::Float64, hi::Float64) = lo + mod(v - lo, hi - lo)

function bounded_grow(p0::Vector{Float64}, dir::Vector{Float64}, step_len::Float64, box::Rhizobox)
    p1_raw = p0 .+ dir .* step_len

    # Apply periodic wrapping before reflection check
    x = box.periodic_x ? _wrap_coord(p1_raw[1], -box.W/2, box.W/2) : p1_raw[1]
    y = box.periodic_y ? _wrap_coord(p1_raw[2],  0.0,     box.D)   : p1_raw[2]
    z = box.periodic_z ? _wrap_coord(p1_raw[3], -box.H,   0.0)     : p1_raw[3]
    p1_wrapped = [x, y, z]

    if inside_box(p1_wrapped, box)
        return p1_wrapped, dir   # direction unchanged through periodic boundary
    end
    # Non-periodic dimensions get reflection
    p1 = clamp_to_box(p1_wrapped, box)
    dir2 = slide_direction(dir, p1, p1_wrapped)
    return p1, dir2
end

function apply_tube_boundary(p1::Vector{Float64}, dir::Vector{Float64}, mr::Minirhizotron)
    dp = p1 .- mr.origin
    t  = dot(dp, mr.direction)
    # Outside tube length — no obstacle
    (t < 0.0 || t > mr.length_cm) && return p1, dir

    foot   = mr.origin .+ t .* mr.direction
    radial = p1 .- foot
    r      = _norm(radial)

    # Outside or on the tube wall — no collision
    r >= mr.r_outer_cm && return p1, dir

    # Clamp to tube surface
    n        = r > 1e-12 ? radial ./ r : [1.0, 0.0, 0.0]
    p1_fixed = foot .+ mr.r_outer_cm .* n

    # Remove inward radial component so root slides along tube exterior
    d_radial = dot(dir, n)
    dir_new  = d_radial < 0.0 ? normalize_vec(dir .- d_radial .* n) : dir

    return p1_fixed, dir_new
end

# ----------------------------
# Direction dynamics
# ----------------------------
function lateral_noise(dir::Vector{Float64}; σ::Float64=0.15, g::Vector{Float64}=[0.0,0.0,-1.0])
    d = normalize_vec(dir)
    g2 = normalize_vec(g)

    lateral = safe_cross(d, g2)
    if _norm(lateral) < 1e-8
        lateral = safe_cross(d, [1.0, 0.0, 0.0])
        if _norm(lateral) < 1e-8
            lateral = safe_cross(d, [0.0, 1.0, 0.0])
        end
    end
    lateral = normalize_vec(lateral)
    return normalize_vec(d .+ (σ * randn()) .* lateral)
end

function apply_gravitropism(dir::Vector{Float64}; g::Vector{Float64}, strength::Float64)
    d = normalize_vec(dir)
    return normalize_vec((1 - strength) .* d .+ strength .* normalize_vec(g))
end

function update_root_direction(dir::Vector{Float64}, order::Int, sp::SpeciesParams; g::Vector{Float64})
    # axial (order 0) and first-order laterals (order 1) share the same gravitropism and noise
    g_strength = (order in (0, 1)) ? (sp.g_axial * sp.gravitropism) : (sp.g_lateral * sp.gravitropism)
    noiseσ     = (order in (0, 1)) ?  sp.noise_axial                :  sp.noise_lateral
    d = apply_gravitropism(dir; g=g, strength=g_strength)
    d = lateral_noise(d; σ=noiseσ, g=g)
    return normalize_vec(d)
end

# ----------------------------
# Branching
# ----------------------------
function branch_direction(parent_dir::Vector{Float64}; angle_deg::Float64)
    θ = deg2rad(angle_deg)
    ϕ = 2π * rand()

    z = normalize_vec(parent_dir)
    ref = abs(z[3]) < 0.9 ? [0.0, 0.0, 1.0] : [0.0, 1.0, 0.0]
    x = normalize_vec(safe_cross(z, ref))
    y = safe_cross(z, x)

    d = cos(θ) .* z .+ sin(θ) .* (cos(ϕ) .* x .+ sin(ϕ) .* y)
    return normalize_vec(d)
end

# ----------------------------
# Initialization
# ----------------------------
function _rotate_about_y(v::AbstractVector{<:Real}, θ::Float64)
    x, y, z = Float64(v[1]), Float64(v[2]), Float64(v[3])
    c = cos(θ); s = sin(θ)
    return [c*x + s*z, y, -s*x + c*z]
end

function _rotate_about_x(v::AbstractVector{<:Real}, θ::Float64)
    x, y, z = Float64(v[1]), Float64(v[2]), Float64(v[3])
    c = cos(θ); s = sin(θ)
    return [x, c*y - s*z, s*y + c*z]
end

function initialize_root_plant(plant_id::Int, start_box::Vector{Float64}, sp::SpeciesParams)
    n = max(1, sp.n_axial_roots)

    max_x_deg     = max(0.0, sp.axial_spread_deg)
    max_y_deg     = max_x_deg
    jitter_sd_deg = sp.axial_jitter_sd_deg

    # yaw fan angles across the spread range
    yaw_angles = n == 1 ? [0.0] : collect(range(-max_x_deg, max_x_deg; length=n))
    yaw_angles = [clamp(a + randn()*jitter_sd_deg, -max_x_deg, max_x_deg) for a in yaw_angles]
    shuffle!(yaw_angles)

    p0 = Vector{Float64}(start_box)
    base = normalize_vec([0.0, 0.0, -1.0])

    segments = RootSegment[]
    tips = RootSegment[]

    for yaw_deg in yaw_angles
        # NEW: random pitch in y-direction
        pitch_deg = clamp((rand()*2 - 1) * max_y_deg + randn()*jitter_sd_deg, -max_y_deg, max_y_deg)

        dir0 = base
        dir0 = _rotate_about_y(dir0, deg2rad(Float64(yaw_deg)))
        dir0 = _rotate_about_x(dir0, deg2rad(Float64(pitch_deg)))
        dir0 = normalize_vec(dir0)

        # (optional but recommended) use bounded_grow instead of raw p0+...
        # p1 = p0 .+ dir0 .* sp.elongation_rate
        L0 = tip_elongation_rate(sp, 0, 0.0)
        p1 = p0 .+ dir0 .* L0
        seg = RootSegment(p0, p1, dir0, 0, 0, 0, plant_id)
        push!(segments, seg)
        push!(tips, seg)
    end

    return RootSystem(segments, tips)
end

# -------------------------------------------------------
# Density regulation (fine-scale) in x–z
# -------------------------------------------------------
@inline function _grid_size(box::Rhizobox, dx::Float64, dz::Float64)
    nx = max(1, Int(ceil(box.W / dx)))
    nz = max(1, Int(ceil(box.H / dz)))
    return nx, nz
end

@inline function _xz_to_idx(x::Float64, z::Float64, box::Rhizobox, dx::Float64, dz::Float64, nx::Int, nz::Int)
    u = (x + box.W/2) / box.W
    ix = clamp(Int(floor(u * nx)) + 1, 1, nx)

    d = clamp(-z, 0.0, box.H)
    v = d / box.H
    iz = clamp(Int(floor(v * nz)) + 1, 1, nz)

    return ix, iz
end
