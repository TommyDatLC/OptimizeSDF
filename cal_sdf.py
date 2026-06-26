import pymeshlab
import numpy as np
import pyvista as pv
import os
import time
import glob

def process_mesh(obj_file, optimal_peeling_iterations=10, hien_thi_3d=True):
    """Hàm xử lý một file mesh duy nhất"""
    print(f"Đang tính toán SDF cho '{obj_file}', vui lòng đợi...")
    
    # Load mesh bằng PyMeshLab
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(obj_file)
    except Exception as e:
        print(f"Lỗi khi đọc file {obj_file}: {e}")
        return

    # Bắt đầu đo thời gian chạy thuật toán trên GPU
    start_time = time.perf_counter()
    # BƯỚC 1: Sử dụng số lần Iteration chính xác đã được tính trước bằng Ray Tracing
    # Giá trị này đảm bảo vừa đủ để tia xuyên qua toàn bộ độ dày vật thể mà không bị lãng phí vòng lặp
    print(f"-> Sử dụng Optimal Peeling Iterations: {optimal_peeling_iterations}")

    # BƯỚC 2: Tính Shape Diameter Function (SDF) bằng tham số peelingiteration tối ưu
    ms.apply_filter('compute_scalar_by_shape_diameter_function_per_vertex_gpu', 
                    coneangle=150, numberrays=64,onprimitive = 0,removeoutliers = False, peelingiteration=optimal_peeling_iterations)
                    
    # Kết thúc đo thời gian chạy
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    print(f"-> Thời gian chạy thuật toán GPU (PyMeshLab): {execution_time:.4f} giây")
    
    # Lấy mảng giá trị SDF (Mỗi phần tử tương ứng với 1 Đỉnh/Vertex)
    sdf_values = ms.current_mesh().vertex_scalar_array()
    print(len(sdf_values))
    # ==========================================
    # LƯU KẾT QUẢ SDF CỦA ĐỈNH VÀO FILE
    # ==========================================
    base_name = os.path.splitext(obj_file)[0]
    output_file = f"{base_name}_pymeshlab.sdf"
    
    with open(output_file, 'w') as f:
        # Lặp qua mảng SDF của các đỉnh và ghi trực tiếp
        for vertex_id, sdf_value in enumerate(sdf_values):
            f.write(f"{vertex_id},{sdf_value}\n")
    print(f"-> Đã lưu kết quả ({len(sdf_values)} đỉnh) tại: {output_file}")

    # ==========================================
    # HIỂN THỊ 3D BẰNG PYVISTA (Theo Đỉnh)
    # ==========================================
    if hien_thi_3d:
        print("-> Đang mở cửa sổ 3D. Hãy đóng cửa sổ để chạy file tiếp theo...")
        
        # Load lại file bằng PyVista chỉ để lấy hình học vẽ lên màn hình
        mesh = pv.read(obj_file)

        # Gán trực tiếp mảng SDF vào dữ liệu ĐỈNH (point_data) thay vì MẶT (cell_data)
        # Điều này giúp màu sắc được nội suy mượt mà trên bề mặt
        mesh.point_data['sdf'] = sdf_values

        plotter = pv.Plotter()
        # Hiển thị mesh với bảng màu jet, ánh xạ theo giá trị sdf của các đỉnh
        plotter.add_mesh(mesh, scalars='sdf', cmap='jet', show_edges=True)
        plotter.add_scalar_bar(title='SDF Values (Vertices)')
        plotter.show()

def main():
    # Tên thư mục chứa các file model
    folder_path = "Model"
    
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
    
    # Từ điển tra cứu số Peeling Iteration tối thiểu cần thiết (tính toán bằng get_complexity.py)
    depth_complexities = {
        '112.obj': 4,
        '118.obj': 4,
        '158.obj': 4,
        '181.obj': 8,
        '360.obj': 20,
        '368.obj': 4,
        '369.obj': 8,
        '371.obj': 4,
        '400.obj': 6,
        '76.obj': 6,
        '9.obj': 8,
        'Leaf.obj': 8,
        'Torus.obj': 8,
        'Armadillo.obj': 6,
        'Bunny.obj': 6,
        'Dragon.obj': 10,
        'Lucy.obj': 9,
        'Nefertiti.obj': 14
    }

    # Chạy vòng lặp qua từng file
    for file_path in obj_files:
        optimal_iter = 10 # Force 10 peeling iterations for benchmark
        # Tắt hien_thi_3d=False để máy chạy tự động hàng loạt
        process_mesh(file_path, optimal_peeling_iterations=optimal_iter, hien_thi_3d=False)
        print("=" * 50)
        
    print("Đã hoàn tất xử lý tất cả các file bằng PyMeshLab!")

if __name__ == "__main__":
    main()