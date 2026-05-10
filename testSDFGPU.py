import os
# CỰC KỲ QUAN TRỌNG: Phải ép Qt chạy ẩn TRƯỚC KHI import pymeshlab
# Nếu import pymeshlab trước, phần lõi C++ đã lỡ khởi tạo giao diện mất rồi!
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pymeshlab
from wurlitzer import pipes # ĐÃ SỬA: Dùng pipes() thay vì sys_pipes()

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
    
    # Sử dụng pipes() và gán vào 2 biến để đọc dữ liệu sau khi chạy xong
    with pipes() as (out, err):
        try:
            # Gọi trực tiếp method như bạn viết là hoàn toàn hợp lệ trong phiên bản mới
            ms.compute_scalar_by_shape_diameter_function_per_vertex_gpu(
                coneangle = 150.0  # Nên truyền rành mạch số thực float để tránh C++ báo lỗi kiểu
            )
        except Exception as e:
            print(f"Python bắt được ngoại lệ: {e}")

    # 3. Đọc và in log từ bộ đệm của Wurlitzer
    stdout_log = out.read()
    stderr_log = err.read()

    if stdout_log:
        print("\n=== LOG TỪ C++ (STDOUT) ===")
        print(stdout_log)
    if stderr_log:
        print("\n=== LỖI TỪ C++ (STDERR) ===")
        print(stderr_log)

    # 4. Lưu kết quả
    output_file = 'output_mesh_with_sdf.ply'
    try:
        # Lưu ra file PLY là rất chuẩn, vì định dạng PLY hỗ trợ lưu giá trị vô hướng (scalar) cực tốt
        ms.save_current_mesh(output_file)
        print(f"\n-> Đã lưu kết quả thành công tại: {output_file}")
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")

if __name__ == "__main__":
    main()