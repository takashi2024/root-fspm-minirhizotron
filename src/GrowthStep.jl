# Auto-split from root_fspm_rhizobox_multiplant_SRL.jl
# File: GrowthStep.jl

@inline function logistic_rate(t::Real, rmin::Real, rmax::Real, k::Real, t50::Real)
    return Float64(rmin + (rmax - rmin) / (1 + exp(-k * (t - t50))))
end

@inline function tip_elongation_rate(sp::SpeciesParams, order::Int, t::Real)
    if sp.root_elongation_mode == :constant
        return order == 0 ? sp.elongation_rate_axial :
               (order == 1 ? sp.elongation_rate_order1 : sp.elongation_rate_order2)

    elseif sp.root_elongation_mode == :logistic
        if order == 0
            return logistic_rate(
                t,
                sp.elongation_rate_min,
                sp.elongation_rate_max,
                sp.elongation_logistic_k,
                sp.elongation_logistic_t50
            )
        elseif order == 1
            return logistic_rate(
                t,
                sp.elongation_rate_min,
                sp.elongation_rate_max,
                sp.elongation_logistic_k,
                sp.elongation_logistic_t50
            )
        else
            return logistic_rate(
                t,
                sp.elongation_rate_min,
                sp.elongation_rate_max,
                sp.elongation_logistic_k,
                sp.elongation_logistic_t50
            )
        end
    else
        error("Unknown root_elongation_mode: $(sp.root_elongation_mode)")
    end
end



# # ----------------------------
# # Facilitation: bias focal species growth toward source species hotspot (B-lite)
# # ----------------------------
# function apply_facilitation(dir::Vector{Float64}, pos::Vector{Float64}, plant_species::String,
#                             facil::Facilitation; gates_by_pid=nothing)
#     facil.enabled || return dir
#     plant_species == facil.focal_species || return dir
#     isempty(facil.source_xz) && return dir

#     depth_cm = -pos[3]
#     depth_cm <= facil.depth_max_cm || return dir

#     # zt = -facil.depth_peak_cm
#     R = facil.radius_cm
#     R <= 0 && return dir

#    # 5% rule
#     τ = 0.05
#     loginvτ = log(1/τ)                # ≈ log(20)
#     σ = R / sqrt(2*loginvτ)           # Gaussian scale so w(R)=0.05
#     L = R / loginvτ                   # Exponential scale so w(R)=0.05

#     u_sum = [0.0, 0.0, 0.0]
#     w_sum = 0.0


#     for (pid, xs, zs) in facil.source_xz
#         gate_i = get(gates_by_pid, pid, 0.0)   # assumes Dict
#         gate_i <= 0 && continue
#         dx = xs - pos[1]
#         # dz = zs - pos[3]
#         dz = -facil.depth_peak_cm - pos[3] # target depth in box coords (z is negative downward)
#         d  = sqrt(dx*dx + dz*dz)
#         d > 1e-9 || continue

#         w = if facil.kernel == :linear
#             max(0.0, 1.0 - d/R)
#         elseif facil.kernel == :gaussian
#             exp(-(d*d)/(2*σ*σ))
#         elseif facil.kernel == :exponential
#             exp(-d/L)
#         else
#             exp(-(d*d)/(2*σ*σ))
#         end

#         w *= gate_i
#         w <= 0 && continue
#         u = normalize_vec([dx, dir[2], dz])   # keeps current y component
#         u_sum .+= w .* u
#         w_sum += w
#     end

#     w_sum <= 0 && return dir
#     u_att = normalize_vec(u_sum)

#     # stabilize multi-source effect
#     w_eff = w_sum / max(1, length(facil.source_xz))
#     κ = clamp(facil.strength * w_eff, 0.0, 1.0)

#     κ == 0.0 && return dir
#     return normalize_vec((1-κ).*dir .+ κ.*u_att)
# end

# ----------------------------
# Facilitation: bias focal species growth toward source species hotspot (B-lite)
# Attraction acts only while the focal root is shallower than the cluster-root layer.
# Once the focal root reaches that layer (or deeper), normal gravitropism resumes.
# ----------------------------
function apply_facilitation(dir::Vector{Float64}, pos::Vector{Float64}, plant_species::String,
                            facil::Facilitation; gates_by_pid=nothing)
    facil.enabled || return dir
    plant_species == facil.focal_species || return dir
    isempty(facil.source_xz) && return dir

    depth_cm = -pos[3]
    depth_cm <= facil.depth_max_cm || return dir

    R = facil.radius_cm
    R <= 0 && return dir

    # target cluster-root depth in box coordinates (z is negative downward)
    z_target = -facil.depth_peak_cm

    # stop attraction once the focal root has reached or passed the cluster-root depth
    # (i.e. same depth or deeper than the target layer)
    pos[3] <= z_target && return dir

    # 5% rule
    τ = 0.05
    loginvτ = log(1 / τ)
    σ = R / sqrt(2 * loginvτ)   # Gaussian scale so w(R)=0.05
    L = R / loginvτ             # Exponential scale so w(R)=0.05

    u_sum = [0.0, 0.0, 0.0]
    w_sum = 0.0

    for (pid, xs, zs) in facil.source_xz
        gate_i = (gates_by_pid === nothing) ? 1.0 : get(gates_by_pid, pid, 0.0)
        gate_i <= 0 && continue

        dx = xs - pos[1]
        dz = z_target - pos[3]   # only attract toward the cluster-root depth while above it
        d  = sqrt(dx*dx + dz*dz)
        d > 1e-9 || continue

        w = if facil.kernel == :linear
            max(0.0, 1.0 - d / R)
        elseif facil.kernel == :gaussian
            exp(-(d*d) / (2 * σ*σ))
        elseif facil.kernel == :exponential
            exp(-d / L)
        else
            exp(-(d*d) / (2 * σ*σ))
        end

        w *= gate_i
        w <= 0 && continue

        # Steer in x-z plane toward attraction target, preserving current y momentum.
        # FIX 2026-03-20: previously normalize_vec([dx, dir[2], dz]) mixed raw spatial
        # distances (cm) with a dimensionless direction component, which caused the
        # y-component to be scaled near-zero whenever |dx| or |dz| > 1 cm.
        # Correct approach: normalize x-z direction first, then inject dir[2] before
        # the final normalization so all three components are on the same scale.
        u_xz = normalize_vec([dx, 0.0, dz])
        u    = normalize_vec([u_xz[1], dir[2], u_xz[3]])
        u_sum .+= w .* u
        w_sum += w
    end

    w_sum <= 0 && return dir
    u_att = normalize_vec(u_sum)

    # stabilize multi-source effect
    w_eff = w_sum / max(1, length(facil.source_xz))
    κ = clamp(facil.strength * w_eff, 0.0, 1.0)

    κ == 0.0 && return dir
    return normalize_vec((1 - κ) .* dir .+ κ .* u_att)
end



# Hierarchical growth + branching:
#   order 0 (axial roots)         : grow with elongation_rate_axial
#     → order 1 (first-order laterals)  via branching_prob_order1, capped by max_order1_per_plant
#     → order 2 (second-order laterals) via branching_prob_order2, only within depth window
#   order 1 (first-order laterals): grow with elongation_rate_order1
#     → order 2 (second-order laterals) via branching_prob_order2, only within depth window
#   order ≥ 2                     : grow with elongation_rate_order2, no further branching

function step_plant!(
    plant::Plant,
    box::Rhizobox,
    reg::DensityRegulation,
    R_all::Matrix{Float64},
    R_self::Matrix{Float64},
    nx::Int,
    nz::Int;
    mature_branching::Bool=false,
    facilitation::Union{Nothing,Facilitation}=nothing,
    facil_gates=nothing,
    minirhizotron::Union{Nothing,Minirhizotron}=nothing
)
    sp = plant.params
    g_box = gravity_in_box(box)

    # age plant
    plant.age += 1
    t_dev = Float64(plant.age)

    # age all segments
    for seg in plant.root.segments
        seg.age += 1
    end

    # count distinct order-1 lateral roots for cap (tips only, not total segments)
    order1_count = count(t -> t.order == 1, plant.root.tips)

    new_segments = RootSegment[]
    new_tips = RootSegment[]

    for tip in plant.root.tips
        tip.tip_age += 1

        # per-order max tip age
        max_age = tip.order == 0 ? sp.max_tip_age_axial : (tip.order == 1 ? sp.max_tip_age_order1 : sp.max_tip_age_order2)
        if tip.tip_age > max_age
            continue
        end

        # per-order elongation
        # base_elong = tip.order == 0 ? sp.elongation_rate : (tip.order == 1 ? sp.elongation_rate_major : sp.elongation_rate_minor)
        
        # per-order elongation, possibly logistic
        base_elong = tip_elongation_rate(sp, tip.order, t_dev)

        new_dir = update_root_direction(tip.direction, tip.order, sp; g=g_box)
        new_p0 = tip.p1
        
        if facilitation !== nothing
            new_dir = apply_facilitation(new_dir, new_p0, plant.species, facilitation; gates_by_pid=facil_gates) 
        end

        # local RLD for density regulation
        rld = 0.0
        if reg.enabled
            if reg.mode == :all
                rld = local_rld(R_all, new_p0[1], new_p0[3], box, reg, nx, nz)
            elseif reg.mode == :intra
                rld = local_rld(R_self, new_p0[1], new_p0[3], box, reg, nx, nz)
            elseif reg.mode == :inter
                r_all   = local_rld(R_all,  new_p0[1], new_p0[3], box, reg, nx, nz)
                r_intra = local_rld(R_self, new_p0[1], new_p0[3], box, reg, nx, nz)
                rld = max(0.0, r_all - r_intra)
            end
        end

        f_elong  = reg.enabled ? density_factor(rld; alpha=reg.alpha_elong,  min_factor=reg.min_factor) : 1.0
        f_branch = reg.enabled ? density_factor(rld; alpha=reg.alpha_branch, min_factor=reg.min_factor) : 1.0

        Lstep = base_elong * f_elong
        new_p1, new_dir2 = bounded_grow(new_p0, new_dir, Lstep, box)
        if minirhizotron !== nothing && minirhizotron.enabled
            new_p1, new_dir2 = apply_tube_boundary(new_p1, new_dir2, minirhizotron)
        end

        grown = RootSegment(new_p0, new_p1, new_dir2, 0, tip.order, tip.tip_age, plant.id)
        push!(new_segments, grown)
        push!(new_tips, grown)

        # depth gate for second-order laterals (applies to order 0→2 and order 1→2)
        depth_cm = -new_p0[3]
        in_order2_zone = (depth_cm >= sp.order2_start_depth_cm) && (depth_cm <= sp.order2_end_depth_cm)

        # --- branching rules ---
        if tip.tip_age > 5
            # order 0 → order 1 (first-order lateral), capped
            if tip.order == 0 && order1_count < sp.max_order1_per_plant
                if rand() < (sp.branching_prob_order1 * f_branch)
                    bdir = branch_direction(tip.direction; angle_deg=sp.branch_angle_order1)
                    b_p0 = tip.p0
                    # b_p1, bdir2 = bounded_grow(b_p0, bdir, sp.elongation_rate_major, box)
                    bL = tip_elongation_rate(sp, 1, t_dev)
                    b_p1, bdir2 = bounded_grow(b_p0, bdir, bL, box)
                    if minirhizotron !== nothing && minirhizotron.enabled
                        b_p1, bdir2 = apply_tube_boundary(b_p1, bdir2, minirhizotron)
                    end
                    branch = RootSegment(b_p0, b_p1, bdir2, 0, 1, 0, plant.id)
                    push!(new_segments, branch)
                    push!(new_tips, branch)
                    order1_count += 1
                end
            end

            # second-order laterals: from order 0 or 1, only within depth window
            if (tip.order == 0 || tip.order == 1) && in_order2_zone
                if rand() < (sp.branching_prob_order2 * f_branch)
                    bdir = branch_direction(tip.direction; angle_deg=sp.branch_angle_order2)
                    b_p0 = tip.p0
                    # b_p1, bdir2 = bounded_grow(b_p0, bdir, sp.elongation_rate_minor, box)
                    bL = tip_elongation_rate(sp, 2, t_dev)
                    b_p1, bdir2 = bounded_grow(b_p0, bdir, bL, box)
                    if minirhizotron !== nothing && minirhizotron.enabled
                        b_p1, bdir2 = apply_tube_boundary(b_p1, bdir2, minirhizotron)
                    end
                    branch = RootSegment(b_p0, b_p1, bdir2, 0, 2, 0, plant.id)
                    push!(new_segments, branch)
                    push!(new_tips, branch)
                end
            end
        end
    end

    # Optional mature branching (applies only to first-order laterals creating second-order laterals)
    if mature_branching
        for seg in plant.root.segments
            if seg.order != 1 || seg.age < 3
                continue
            end
            depth_cm = -seg.p0[3]
            in_order2_zone = (depth_cm >= sp.order2_start_depth_cm) && (depth_cm <= sp.order2_end_depth_cm)
            if in_order2_zone && rand() < (0.5 * sp.branching_prob_order2)
                bdir = branch_direction(seg.direction; angle_deg=sp.branch_angle_order2)
                b_p0 = seg.p0
                # b_p1, bdir2 = bounded_grow(b_p0, bdir, sp.elongation_rate_minor, box)
                bL = tip_elongation_rate(sp, 2, t_dev)
                b_p1, bdir2 = bounded_grow(b_p0, bdir, bL, box)
                if minirhizotron !== nothing && minirhizotron.enabled
                    b_p1, bdir2 = apply_tube_boundary(b_p1, bdir2, minirhizotron)
                end
                branch = RootSegment(b_p0, b_p1, bdir2, 0, 2, 0, plant.id)
                push!(new_segments, branch)
                push!(new_tips, branch)
            end
        end
    end

    append!(plant.root.segments, new_segments)
    plant.root.tips = new_tips
    return plant
end

# ----------------------------
# Export helpers: length/biomass
# ----------------------------
@inline function segment_length_cm(seg::RootSegment)
    return _norm(seg.p1 .- seg.p0)
end
