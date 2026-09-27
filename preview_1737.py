import os
import argparse
import numpy as np
import pymeshlab
import pyvista as pv

OBJ_PATH = r"E:\Code\FinalProject\peeling_study\models_net\net_psb_1737_staircase.obj"
OPT_PEELING = 164
ITERATIONS = [1, 4, 10, OPT_PEELING]
CACHE_DIR = r"E:\Code\FinalProject\peeling_study\models_net\cache_sdf"

def compute_or_load_sdf(obj_path, peeling, n_verts):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"1737_peeling_{peeling}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)

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
    arr = np.array(ms.current_mesh().vertex_scalar_array())[:n_verts]
    np.save(cache_file, arr)
    return arr

def main():
    parser = argparse.ArgumentParser(description="Preview Depth Peeling on 1737 Staircase (1, 4, 10, 164)")
    parser.add_argument("--save", type=str, default=None, help="Lưu ảnh screenshot")
    parser.add_argument("--off_screen", action="store_true", help="Chạy ẩn không mở cửa sổ GUI")
    args = parser.parse_args()

    mesh_base = pv.read(OBJ_PATH).clean()
    n_verts = mesh_base.n_points

    sdf_results = {it: compute_or_load_sdf(OBJ_PATH, it, n_verts) for it in ITERATIONS}

    # Thang màu đồng bộ cho cả 4 cửa sổ để dễ quan sát sự khác biệt
    all_vals = np.concatenate([sdf_results[it] for it in ITERATIONS])
    clim = (float(np.min(all_vals)), float(np.percentile(all_vals, 98)))

    pl = pv.Plotter(
        shape=(1, 4),
        window_size=(1920, 600),
        title="1737 Staircase: So sánh Depth Peeling (1, 4, 10 vs 164 iterations)",
        off_screen=args.off_screen
    )

    panel_configs = [
        (0, 1, "Peeling = 1 iteration"),
        (1, 4, "Peeling = 4 iterations"),
        (2, 10, "Peeling = 10 iterations (PyMeshLab Default)"),
        (3, OPT_PEELING, f"Peeling = {OPT_PEELING} iterations (Optimal / Ground Truth)"),
    ]

    for col_idx, it, title in panel_configs:
        pl.subplot(0, col_idx)
        m = mesh_base.copy()
        m.point_data["SDF"] = sdf_results[it]
        pl.add_mesh(
            m,
            scalars="SDF",
            cmap="turbo",
            clim=clim,
            show_edges=False,
            scalar_bar_args={
                "title": "SDF",
                "vertical": True,
                "title_font_size": 10,
                "label_font_size": 8
            }
        )
        pl.add_text(title, font_size=11, position="upper_left", color="white", shadow=True)

    # Đồng bộ góc quay camera cho cả 4 panel cùng một hàng
    pl.link_views()

    if args.save:
        pl.show(screenshot=args.save)
        print(f"Saved preview screenshot to: {args.save}")
    else:
        pl.show()

if __name__ == "__main__":
    main()
