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
    # ĐỒNG BỘ: Chỉnh cone_amplitude về 120 độ để khớp với C++ OptiX
    ms.apply_filter('compute_scalar_by_shape_diameter_function_per_vertex', 
                    cone_amplitude = 150)
                    
    # Kết thúc đo thời gian chạy
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    print(f"-> Thời gian chạy thuật toán: {execution_time:.4f} giây")
    
    # Lấy giá trị SDF (Mảng này có độ dài bằng số lượng Đỉnh)
    sdf_values = ms.current_mesh().vertex_scalar_array()

    # Create a PyVista mesh từ file OBJ gốc
    mesh = pv.read(obj_file)

    # ĐÃ SỬA: Gán giá trị SDF trực tiếp vào Đỉnh (Point Data) thay vì Mặt (Cell Data)
    # Điều này giúp PyVista nội suy màu mượt mà y hệt như Polyscope bên C++
    mesh.point_data['sdf'] = sdf_values

    # Tạo tên file output và lưu file .sdf theo dữ liệu Đỉnh
    base_name = os.path.splitext(obj_file)[0]
    output_file = f"{base_name}.sdf"
    
    with open(output_file, 'w') as f:
        # Ghi id của Đỉnh (Vertex) và giá trị SDF tương ứng
        for vertex_id, sdf_value in enumerate(sdf_values):
            f.write(f"{vertex_id},{sdf_value}\n")
    print(f"-> Đã lưu kết quả tại: {output_file}")

    # Hiển thị 3D (Nếu được bật)
    if hien_thi_3d:
        print("-> Đang mở cửa sổ 3D. Hãy đóng cửa sổ để chạy file tiếp theo...")
        plotter = pv.Plotter()
        
        # ĐỒNG BỘ: Dùng cmap 'turbo' và bật smooth_shading để giống hệt Polyscope
        plotter.add_mesh(mesh, scalars='sdf', cmap='turbo', show_edges=False, smooth_shading=True)
        plotter.add_scalar_bar(title='SDF Values (Per Vertex)')
        plotter.show()

def main():
    # Tên thư mục chứa các file model
    folder_path = "./Model"
    
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(folder_path):
        print(f"Lỗi: Không tìm thấy thư mục '{folder_path}'.")
        return

    # Lấy danh sách tất cả các file có đuôi .obj trong thư mục Model
    search_pattern = os.path.join(folder_path, "*.obj")
    obj_files = glob.glob(search_pattern)

    if not obj_files:
        print(f"Không tìm thấy file .obj nào trong thư mục '{folder_path}'.")
        return

    print(f"Tìm thấy {len(obj_files)} file .obj. Bắt đầu xử lý...\n")
    print("=" * 50)

    # Chạy vòng lặp qua từng file
    for file_path in obj_files:
        process_mesh(file_path, hien_thi_3d=True)
        print("=" * 50)
        
    print("Đã hoàn tất xử lý tất cả các file!")

if __name__ == "__main__":
    main()