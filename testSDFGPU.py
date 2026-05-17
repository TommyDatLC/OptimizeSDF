import os
# CỰC KỲ QUAN TRỌNG: Phải ép Qt chạy ẩn TRƯỚC KHI import pymeshlab
# Nếu import pymeshlab trước, phần lõi C++ đã lỡ khởi tạo giao diện mất rồi!
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pymeshlab


def main():
    print("Đang khởi tạo PyMeshLab...")
    ms = pymeshlab.MeshSet()

    # 1. Load mesh an toàn
    input_file = './Model/9.obj'
    try:
        ms.load_new_mesh(input_file)
        print(f"Đã load thành công: {input_file}")
    except Exception as e:
        print(f"Lỗi khi load file: {e}")
        return

    # 2. Chạy thuật toán và BẮT LOG C++
    print("Bắt đầu tính toán SDF (GPU version)...")
       # Gọi trực tiếp method như bạn viết là hoàn toàn hợp lệ trong phiên bản mới
    scalar = ms.compute_scalar_by_shape_diameter_function_per_vertex_gpu(
                coneangle = 150.0  # Nên truyền rành mạch số thực float để tránh C++ báo lỗi kiểu
            )
  sdf_values = ms.current_mesh().vertex_scalar_array()

    print("SDF values per vertex:", sdf_values)

    # Create a PyVista mesh from the original OBJ file
    mesh = pv.read(obj_file)

    # Convert vertex-based SDF values to face-based values by averaging
    face_centers = mesh.cell_centers().points
    face_sdf = np.zeros(mesh.n_faces)

    for i, cell in enumerate(mesh.faces.reshape((-1, 4))):
        vertex_ids = cell[1:]  # Skip the first number which is the number of vertices
        face_sdf[i] = np.mean(sdf_values[vertex_ids])

    # Assign SDF values to faces
    mesh.cell_data['sdf'] = face_sdf

    # Generate output filename based on input filename
    base_name = os.path.splitext(obj_file)[0]
    output_file = f"{base_name}.sdf"
    
    with open(output_file, 'w') as f:
        for face_id, sdf_value in enumerate(face_sdf):
            f.write(f"{face_id},{sdf_value}\n")

    # Create a plotter
    plotter = pv.Plotter()
    # Add the mesh to the plotter with SDF values as colors
    plotter.add_mesh(mesh, scalars='sdf', cmap='jet', show_edges=True)
    # Add a colorbar
    plotter.add_scalar_bar(title='SDF Values')
    # Show the visualization
    plotter.show()


if __name__ == "__main__":
    main()