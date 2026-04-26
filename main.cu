#include "Core/MatrixMemoryManager.cuh"
#include "NormalSDFOptix/SDFMain.cpp"
#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "Core/ClipSpaceConversion.hpp"
#include "Core/Matrix.cuh"
#include "Core/Model.cuh"
#include "Core/ModelHelper.cu"
#include "cmake-build-debug/_deps/polyscope-src/include/polyscope/polyscope.h"
#include "OptiX/optix.h"
#include "OptiX/optix_stubs.h"
#include <optix_function_table_definition.h>
#include <filesystem> // Yêu cầu C++17
namespace fs = std::filesystem;

int main() {
    // Khởi tạo các hệ thống
    optixInit();
    polyscope::init();

    std::string folderPath = "Model";

    // Kiểm tra xem thư mục có tồn tại không
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        std::cerr << "Lỗi: Không tìm thấy thư mục '" << folderPath << "'\n";
        return 1;
    }

    std::cout << "Bắt đầu quét thư mục: " << folderPath << "\n";

    // Cờ điều khiển việc có mở UI 3D hay không
    bool hien_thi_3d = false;

    // Lặp qua tất cả các file trong thư mục
    for (const auto& entry : fs::directory_iterator(folderPath)) {

        // Chỉ xử lý các file có phần mở rộng là .obj
        if (entry.is_regular_file() && entry.path().extension() == ".obj") {

            // Lấy đường dẫn tuyệt đối (hoặc tương đối) của file
            std::string filePath = entry.path().string();

            std::cout << "==================================================\n";
            std::cout << "Đang xử lý: " << filePath << "\n";

            // 1. Tải Model
            Model model = Model(filePath);
            // model.UpdateNormal();
            // model.ToClipSpace(v_data,p_data);
            // model.AddToScene();

            // 3. Chạy thuật toán tính SDF
            CaculatingSDFUsingOptix(model);

            // (Tuỳ chọn) NẾU BẠN CÓ HÀM LƯU FILE SDF, HÃY GỌI Ở ĐÂY
            // Lấy tên file gốc đổi đuôi .obj thành .sdf
            // std::string outputPath = entry.path().replace_extension(".sdf").string();
            // model.SaveSDF(outputPath);

            // 5. Hiển thị giao diện 3D
            if (hien_thi_3d) {
                std::cout << "-> Đang mở cửa sổ Polyscope. Hãy đóng cửa sổ để chạy file tiếp theo...\n";
                polyscope::show();

                // RẤT QUAN TRỌNG: Xóa dữ liệu 3D cũ đi sau khi đóng cửa sổ,
                // nếu không file sau sẽ bị vẽ đè lên file trước.
                polyscope::removeAllStructures();
            }
        }
    }

    std::cout << "==================================================\n";
    std::cout << "Đã hoàn tất xử lý tất cả các file!\n";

    return 0;
}