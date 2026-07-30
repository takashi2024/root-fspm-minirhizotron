# Auto-split from root_fspm_rhizobox_multiplant_SRL.jl
# File: Deps.jl

# root_fspm_rhizobox_multiplant_SRL_v3.jl
# Multi-plant, multi-species segment-based root FSPM constrained to a tilted rhizobox.
# Julia 1.10.x compatible.
#
# Adds:
#   - Visible-root length in a near-glass slab (exact clipping in y)
#   - Visible-length grid export at arbitrary dx,dz
#   - Optional rasterization+skeletonization metric (requires ImageMorphology)
#   - export_visible_grid_csv(...; dx=1, dz=1, include_byplant=true)
#
# Coordinate system (BOX coords, cm):
#   x ∈ [-W/2, +W/2]  : width (horizontal along acrylic face)
#   y ∈ [0, D]        : thickness (y=0 is transparent plate; +y goes into soil)
#   z ∈ [-H, 0]       : depth (z=0 at soil surface; negative downward)

using LinearAlgebra
using Random
using Printf
using ProgressMeter

# ----------------------------
# JSON loading (top-level)
# ----------------------------
const _JSON_AVAILABLE = Ref(false)
try
    @eval using JSON
    _JSON_AVAILABLE[] = true
catch
    _JSON_AVAILABLE[] = false
end

function _read_json(path::AbstractString)
    _JSON_AVAILABLE[] || error("This script needs JSON.jl. Install once with: julia -e \"import Pkg; Pkg.add(\\\"JSON\\\")\" ")
    return JSON.parse(read(path, String))
end

# ----------------------------
# Optional Makie loading (top-level)
# ----------------------------
const _MAKIE_AVAILABLE = Ref(false)
const _MAKIE_BACKEND = Ref{Union{Nothing,Symbol}}(nothing)

try
    @eval using CairoMakie
    _MAKIE_AVAILABLE[] = true
    _MAKIE_BACKEND[] = :CairoMakie
catch
    try
        @eval using GLMakie
        _MAKIE_AVAILABLE[] = true
        _MAKIE_BACKEND[] = :GLMakie
    catch
        _MAKIE_AVAILABLE[] = false
        _MAKIE_BACKEND[] = nothing
    end
end

# ----------------------------
# Optional image morphology for raster skeleton metrics
# ----------------------------
const _IMG_AVAILABLE = Ref(false)
try
    @eval using ImageMorphology
    _IMG_AVAILABLE[] = true
catch
    _IMG_AVAILABLE[] = false
end

# ----------------------------
# Data structures
# ----------------------------
