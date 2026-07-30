# Auto-split from root_fspm_rhizobox_multiplant_SRL.jl
# File: Types.jl

# ----------------------------
# Core root data structures
# ----------------------------
mutable struct RootSegment
    p0::Vector{Float64}
    p1::Vector{Float64}
    direction::Vector{Float64}   # unit (box coords)
    age::Int                     # segment age (days)
    order::Int                   # 0=axial, 1=first-order lateral, 2=second-order lateral, ...
    tip_age::Int                 # age of the active tip that produced this segment
    plant_id::Int
end

mutable struct RootSystem
    segments::Vector{RootSegment}
    tips::Vector{RootSegment}    # active growing tips (stored as the latest segment)
end

# ----------------------------
# Environment / regulation
# ----------------------------
struct DensityRegulation
    enabled::Bool
    mode::Symbol                 # :all | :intra | :inter
    grid_dx_cm::Float64
    grid_dz_cm::Float64
    sample_step_cm::Float64
    radius_cm::Float64
    alpha_elong::Float64
    alpha_branch::Float64
    min_factor::Float64
end

struct Rhizobox
    H::Float64
    W::Float64
    D::Float64
    tilt_from_vertical::Float64  # radians
    periodic_x::Bool             # wrap at x (left/right) walls instead of reflect
    periodic_y::Bool             # wrap at y (front/back) walls instead of reflect
    periodic_z::Bool             # wrap at z (bottom) wall instead of reflect
end

struct Facilitation
    enabled::Bool
    focal_species::String
    source_species::String
    strength::Float64
    radius_cm::Float64
    depth_max_cm::Float64
    depth_peak_cm::Float64
    kernel::Symbol              # linear | gaussian | exponential
    source_xz::Vector{Tuple{Int,Float64,Float64}}
    trigger_depth_cm::Float64
    trigger_delay_days::Int
    trigger_ramp_days::Int
end

# ----------------------------
# Minirhizotron observation tube
# ----------------------------
struct Minirhizotron
    enabled::Bool
    origin::Vector{Float64}          # cm, tube entry point [x,y,z] in box coords (x centred)
    direction::Vector{Float64}       # unit vector along tube axis (pointing into soil)
    length_cm::Float64               # usable tube length in soil (cm)
    r_outer_cm::Float64             # tube outer radius (cm)
    skin_depth_cm::Float64           # observable soil depth beyond tube wall (cm)
    image_along_cm::Float64         # image frame height along tube axis (cm)
    image_arc_cm::Float64           # camera FOV arc length on tube surface (cm)
    station_positions_cm::Vector{Float64}  # along-axis centres of image stations (sorted)
    sample_step_cm::Float64         # integration step for visibility computation (cm)
    csv_name::String
    camera_azimuths_deg::Vector{Float64}  # camera facing angles in cross-section plane (deg); empty = 360° view
end



# ----------------------------
# Species parameters (hierarchical architecture)
#
# Root order is a MODEL TOPOLOGY — not a biological origin term.
#   order 0 = axial root (the main framework axis — may represent a taproot,
#             seminal root, nodal/crown root, or generic basal axis depending on
#             architecture_type and species context)
#   order 1 = first-order lateral root (emerges from an axial root)
#   order 2 = second-order lateral root / fine lateral (emerges from order 1)
#
# Use architecture_type to document biological interpretation:
#   "generic_multi_axial" — multiple axial roots, no specific origin assumed
#   "dicot_taproot"       — one dominant primary root (radicle-derived); n_axial_roots=1
#   "monocot_fibrous"     — seminal + nodal/crown roots; n_axial_roots > 1
# ----------------------------
struct SpeciesParams
    architecture_type::String       # "generic_multi_axial" | "dicot_taproot" | "monocot_fibrous"
    name::String

    root_elongation_mode::Symbol    # :constant | :logistic
    elongation_rate_min::Float64
    elongation_rate_max::Float64
    elongation_logistic_k::Float64
    elongation_logistic_t50::Float64

    # Order 0 — axial roots (basal axes / framework roots)
    elongation_rate_axial::Float64
    branching_prob_order1::Float64   # order 0 → order 1 branching probability
    branch_angle_order1::Float64     # order 0 → order 1 angle (deg)
    max_tip_age_axial::Int           # order-0 max tip age (days)

    # Order 1 — first-order lateral roots
    elongation_rate_order1::Float64
    branching_prob_order2::Float64   # order 1 → order 2 branching probability
    branch_angle_order2::Float64     # order 1 → order 2 angle (deg)
    max_tip_age_order1::Int

    # Order 2+ — second-order lateral roots (fine laterals)
    elongation_rate_order2::Float64
    max_tip_age_order2::Int

    # Depth window for order-2 lateral emergence (applies to order 0→2 and order 1→2)
    order2_start_depth_cm::Float64
    order2_end_depth_cm::Float64

    # Global controls
    gravitropism::Float64
    srl_m_per_g::Float64
    noise_axial::Float64
    noise_lateral::Float64
    g_axial::Float64
    g_lateral::Float64

    # Axial root initialization
    n_axial_roots::Int              # number of order-0 axial roots initialized from plant base
    axial_spread_deg::Float64       # max fan angle (deg)
    axial_jitter_sd_deg::Float64    # angular jitter SD for axial root orientation (deg)

    # Caps
    max_order1_per_plant::Int       # per-plant cap on order-1 lateral count
end

mutable struct Plant
    id::Int
    species::String
    root::RootSystem
    params::SpeciesParams
    obs_root_biomass_g::Union{Nothing,Float64}
    age::Int
end
