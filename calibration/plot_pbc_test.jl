using CSV, DataFrames, CairoMakie

const SEG_CSV  = joinpath(@__DIR__, "__calib_tmp_mr_100cm_pbc__/out/segments_all.csv")
const OUT_3D   = joinpath(@__DIR__, "__calib_tmp_mr_100cm_pbc__/out/plot3d_pbc_day90.png")
const OUT_2D   = joinpath(@__DIR__, "__calib_tmp_mr_100cm_pbc__/out/root2d_pbc_day90.png")

# Box dims (must match config)
const W = 100.0; const D = 100.0; const H = 100.0

# Tube params (from config): origin [50,36,0], dir [0, 0.7071, -0.7071], len=80
const T_ORI = [50.0 - W/2, 36.0, 0.0]   # centred coords: x=0
const T_DIR = [0.0, 0.7071067811865476, -0.7071067811865476]
const T_LEN = 80.0
const T_R   = 3.0

df = CSV.read(SEG_CSV, DataFrame)

# ── 3-D plot ──────────────────────────────────────────────────────────────────
fig3 = Figure(size=(900, 900))
Label(fig3[0,1], "Day 90 | 100×100 PBC box | $(nrow(df)) segments", fontsize=18, tellwidth=false)
ax3 = Axis3(fig3[1,1], xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)", aspect=:data)
xlims!(ax3, -W/2, W/2); ylims!(ax3, 0, D); zlims!(ax3, -H, 0)

for row in eachrow(df)
    # Skip boundary-crossing segments
    abs(row.x1 - row.x0) > W/2 && continue
    depth = -row.z1
    lines!(ax3, [Point3f(row.x0, row.y0, row.z0), Point3f(row.x1, row.y1, row.z1)],
           linewidth=1.5, color=depth, colormap=:viridis, colorrange=(0, H))
end
Colorbar(fig3[1,2], colormap=:viridis, colorrange=(0,H), label="Depth (cm)")

# Draw tube axis
t_end = T_ORI .+ T_LEN .* T_DIR
lines!(ax3, [Point3f(T_ORI...), Point3f(t_end...)]; color=:royalblue, linewidth=2, linestyle=:dash)
scatter!(ax3, [Point3f(T_ORI...)]; markersize=12, color=:royalblue)

mkpath(dirname(OUT_3D))
save(OUT_3D, fig3)
println("Saved: $OUT_3D")

# ── 2-D plate view (project all roots onto x-z plane) ─────────────────────────
fig2 = Figure(size=(700, 800))
ax2 = Axis(fig2[1,1], xlabel="X (cm)", ylabel="Depth (cm)",
           title="Day 90 | 100×100 PBC | plate view (X–Depth)", yreversed=true)
xlims!(ax2, -W/2, W/2); ylims!(ax2, 0, H)

for row in eachrow(df)
    abs(row.x1 - row.x0) > W/2 && continue
    lines!(ax2, [row.x0, row.x1], [-row.z0, -row.z1]; linewidth=0.8, color=(:black, 0.3))
end

# Tube silhouette on plate
t_pts_x = [T_ORI[1] + s*T_DIR[1] for s in range(0, T_LEN; length=50)]
t_pts_z = [-( T_ORI[3] + s*T_DIR[3]) for s in range(0, T_LEN; length=50)]
lines!(ax2, t_pts_x, t_pts_z; color=:royalblue, linewidth=2, linestyle=:dash)

save(OUT_2D, fig2)
println("Saved: $OUT_2D")
