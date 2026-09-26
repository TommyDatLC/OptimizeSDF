import os
import time
import numpy as np
import pymeshlab
import pyvista as pv

OBJ_PATH = r"E:\Code\FinalProject\peeling_study\models_net\net_psb_1737_staircase.obj"
OPTIX_SDF_PATH = r"E:\Code\FinalProject\net_psb_1737_staircase_optix.sdf"  # dumped by C++ OptiX pipeline (smoothed)
OPT_PEELING = 164

print(f"Loading mesh: {OBJ_PATH}")
mesh_base = pv.read(OBJ_PATH).clean()
n_verts = mesh_base.n_points
print(f"Vertices: {n_verts}, Faces: {mesh_base.n_cells}")

def compute_sdf(obj_path, peeling):
    t0 = time.time()
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.apply_filter(
        "compute_scalar_by_shape_diameter_function_per_vertex_gpu",
        coneangle=150,
        numberrays=64,
        onprimitive=0,
        removeoutliers=False,
        peelingiteration=peeling
    )
    arr = np.array(ms.current_mesh().vertex_scalar_array())
    t1 = time.time()
    print(f"Computed SDF (peeling={peeling}) in {t1 - t0:.3f}s")
    return arr

print("Computing SDF at peeling=10...")
sdf_10 = compute_sdf(OBJ_PATH, 10)

print(f"Computing SDF at peeling={OPT_PEELING} (OPT)...")
sdf_opt = compute_sdf(OBJ_PATH, OPT_PEELING)

diff = np.abs(sdf_10 - sdf_opt)

print(f"Loading OptiX SDF: {OPTIX_SDF_PATH}")
optix_raw = np.loadtxt(OPTIX_SDF_PATH, delimiter=",")
sdf_optix = optix_raw[:, 1] if optix_raw.ndim > 1 else optix_raw
assert len(sdf_optix) == n_verts, f"OptiX SDF count {len(sdf_optix)} != verts {n_verts}"
print(f"OptiX SDF: min={sdf_optix.min():.4f} max={sdf_optix.max():.4f} mean={sdf_optix.mean():.4f} (smoothed dump)")

def minmax(v):
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

agree = np.abs(minmax(sdf_opt) - minmax(sdf_optix))
print(f"Agreement |PML_OPT-OptiX| (both min-max norm): mean={agree.mean():.4f} max={agree.max():.4f}")

# Color limits (PyMeshLab pair shares one scale; OptiX/agreement have their own, labeled)
both = np.concatenate([sdf_10, sdf_opt])
lo_sdf = float(np.min(both))
hi_sdf = float(np.percentile(both, 98))
hi_diff = float(np.percentile(diff, 98)) or float(np.max(diff))
hi_optix = float(np.percentile(sdf_optix, 98)) or float(np.max(sdf_optix))
hi_agree = float(np.percentile(agree, 98)) or float(np.max(agree))

# Setup 2x2 plotter
pl = pv.Plotter(shape=(2, 2), window_size=(1800, 1000), title="1737 staircase: PyMeshLab (10 vs 164) + OptiX SDF")

panels = [
    (0, 0, sdf_10, (lo_sdf, hi_sdf), "turbo", "PyMeshLab SDF (Peeling = 10)"),
    (0, 1, sdf_opt, (lo_sdf, hi_sdf), "turbo", f"PyMeshLab SDF (Peeling = {OPT_PEELING} OPT)"),
    (1, 0, sdf_optix, (0.0, hi_optix), "turbo", "OptiX SDF (smoothed dump)"),
    (1, 1, agree, (0.0, hi_agree), "inferno", "Agreement |PML_OPT - OptiX| (both min-max norm)"),
]

for r, c, vals, clim, cmap, title in panels:
    m = mesh_base.copy()
    m.point_data["SDF"] = vals
    pl.subplot(r, c)
    pl.add_mesh(
        m,
        scalars="SDF",
        cmap=cmap,
        clim=clim,
        show_edges=False,
        scalar_bar_args={"title": title, "vertical": True, "title_font_size": 11, "label_font_size": 9}
    )
    pl.add_text(title, font_size=12, position='upper_left', color='white', shadow=True)

# Add stats overlay on top-left panel
stats_text = (
    f"Model: net_psb_1737_staircase | Vertices: {n_verts}\n"
    f"Depth: 164 | PML peeled-diff: mean 0.00214 median 0.00083 max 0.29580\n"
    f"OptiX: mean {sdf_optix.mean():.4f} max {sdf_optix.max():.4f} | Agreement mean {agree.mean():.4f} max {agree.max():.4f}\n"
    f"[Controls: Rotate/Zoom syncs all panels | Link Sync ON]"
)
pl.subplot(0, 0)
pl.add_text(stats_text, font_size=9, position='lower_left', color='cyan', shadow=True)

pl.link_views()
print("Opening interactive preview window (close window when done)...")
pl.show()
