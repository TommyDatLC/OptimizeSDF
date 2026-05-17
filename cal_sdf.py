import pymeshlab
import numpy as np
import pyvista as pv
import os
import time
import glob

def process_mesh(obj_file, hien_thi_3d=True):
    """Hàm xử lý một file mesh duy nhất"""
    print(f"Đang tính toán SDF cho '{obj_file}', vui lòng đợi...")
    
    # Load mesh
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(obj_file)
    except Exception as e:
        print(f"Lỗi khi đọc file {obj_file}: {e}")
        return

    # Bắt đầu đo thời gian chạy thuật toán
    start_time = time.perf_counter()
    
    # Compute the Shape Diameter Function (SDF)
    ms.apply_filter('compute_scalar_by_shape_diameter_function_per_vertex_gpu', 
                    coneangle = 150,numberrays  = 64)
                    
    # Kết thúc đo thời gian chạy
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    print(f"-> Thời gian chạy thuật toán: {execution_time:.4f} giây")
    
    # Lấy giá trị SDF
    sdf_values = ms.current_mesh().vertex_scalar_array()

    # Create a PyVista mesh từ file OBJ gốc
    mesh = pv.read(obj_file)

    # Convert vertex-based SDF values to face-based values by averaging
    face_sdf = np.zeros(mesh.n_faces)

    for i, cell in enumerate(mesh.faces.reshape((-1, 4))):
        vertex_ids = cell[1:]  # Bỏ qua con số đầu tiên (chỉ định số lượng đỉnh của mặt)
        face_sdf[i] = np.mean(sdf_values[vertex_ids])

    # Gán giá trị SDF vào faces
    mesh.cell_data['sdf'] = face_sdf

    # Tạo tên file output và lưu file .sdf
    base_name = os.path.splitext(obj_file)[0]
    output_file = f"{base_name}.sdf"
    
    with open(output_file, 'w') as f:
        for face_id, sdf_value in enumerate(face_sdf):
            f.write(f"{face_id},{sdf_value}\n")
    print(f"-> Đã lưu kết quả tại: {output_file}")

    # Hiển thị 3D (Nếu được bật)
    if hien_thi_3d:
        print("-> Đang mở cửa sổ 3D. Hãy đóng cửa sổ để chạy file tiếp theo...")
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars='sdf', cmap='jet', show_edges=True)
        plotter.add_scalar_bar(title='SDF Values')
        plotter.show()

def main():
    # Tên thư mục chứa các file model
    folder_path = "Model"
    
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(folder_path):
        print(f"Lỗi: Không tìm thấy thư mục '{folder_path}'.")
        return

    # Lấy danh sách tất cả các file có đuôi .obj trong thư mục Model
    # Dùng glob.glob kết hợp os.path.join sẽ tự động bắt định dạng chuẩn xác trên cả Windows/Mac/Linux
    search_pattern = os.path.join(folder_path, "*.obj")
    obj_files = glob.glob(search_pattern)

    if not obj_files:
        print(f"Không tìm thấy file .obj nào trong thư mục '{folder_path}'.")
        return

    print(f"Tìm thấy {len(obj_files)} file .obj. Bắt đầu xử lý...\n")
    print("=" * 50)

    # Chạy vòng lặp qua từng file
    for file_path in obj_files:
        # Nếu muốn máy tự động chạy hết không cần xem hình 3D, hãy đổi thành hien_thi_3d=False
        process_mesh(file_path, hien_thi_3d=True)
        print("=" * 50)
        
    print("Đã hoàn tất xử lý tất cả các file!")

if __name__ == "__main__":
    main()