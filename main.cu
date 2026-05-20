#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda.h>
#include <cublas_v2.h>
#include <filesystem> // Yêu cầu C++17

#include "Core/MatrixMemoryManager.cuh"
#include "Core/ClipSpaceConversion.hpp"
#include "Core/Matrix.cuh"
#include "Core/Model.cuh"
#include "Core/ModelHelper.cu"
#include "src/Optix/interface.cu"
#include "src/OpenCL/interface.cpp"
#include "src/Compare.hpp"

#include "polyscope/polyscope.h"
#include "OptiX/optix.h"
#include "OptiX/optix_stubs.h"
#include <optix_function_table_definition.h>

namespace fs = std::filesystem;



// =========================================================================
// HÀM KHỞI TẠO VÀ CHẠY PIPELINE
// =========================================================================
void initialize()
{
    // Ép Console của Windows sử dụng bảng mã UTF-8 để hiển thị Tiếng Việt
    system("chcp 65001 > nul");

    // Khởi tạo các hệ thống
    optixInit();
    polyscope::init();

    std::string folderPath = "Model";

    // Kiểm tra xem thư mục có tồn tại không
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        std::cerr << "Lỗi: Không tìm thấy thư mục '" << folderPath << "'\n";
        return;
    }

    // ĐÃ MỞ: Đọc file shader PTX của OptiX phục vụ cho việc tính toán so sánh
    std::string optixPTX = readFile("E:/Code/FinalProject/cmake-build-default-visual-studio/OptixShaders.dir/Debug/SDFOptix.ptx");
    // std::string OpenCLShader = readFile("kernel.cl");
    // auto clEnv = InitOpenCLEnvironment(OpenCLShader);

    std::cout << "Bắt đầu quét thư mục: " << folderPath << "\n";

    // Đổi thành true để hiển thị Polyscope cửa sổ 3D
    bool hien_thi_3d = true;

    // Lặp qua tất cả các file trong thư mục
    for (const auto& entry : fs::directory_iterator(folderPath)) {

        // Chỉ xử lý các file có phần mở rộng là .obj
        if (entry.is_regular_file() && entry.path().extension() == ".obj") {

            std::string filePath = entry.path().string();

            std::cout << "==================================================\n";
            std::cout << "Đang xử lý: " << filePath << "\n";

            // -------------------------------------------------------------
            // HEAT MAP 1: TẢI KẾT QUẢ SDF TỪ PYMESHLAB (PYTHON)
            // -------------------------------------------------------------

            Model modelOptix(filePath);
            // Dự đoán tên file .sdf sinh ra bởi Python
            std::string base_name = entry.path().stem().string(); // Lấy tên không đuôi (VD: 112)
            std::string parent_dir = entry.path().parent_path().string(); // Lấy thư mục chứa nó (VD: Model)
            std::string sdfFilePath = parent_dir + "/" + base_name + "_pymeshlab.sdf";

            // Đọc file PyMeshLab SDF và gán vào modelPy
            LoadPyHeatMap(modelOptix, sdfFilePath);

            // Đăng ký modelPy lên Polyscope với tên riêng biệt
            modelOptix.AddToScene("PyMeshLab_SDF_Model", false);

            // -------------------------------------------------------------
            // HEAT MAP 2: TÍNH TOÁN SDF BẰNG C++ OPTIX (HARDWARE RAY TRACING)
            // -------------------------------------------------------------


            // Chạy tính toán bằng OptiX Shader
            CaculatingSDFUsingOptix(modelOptix, optixPTX);

            // -------------------------------------------------------------
            // HIỂN THỊ GIAO DIỆN ĐỐI CHIẾU 3D
            // -------------------------------------------------------------
            if (hien_thi_3d) {
                std::cout << "-> Đang mở cửa sổ Polyscope. Hãy đóng cửa sổ để chạy file tiếp theo...\n";
                polyscope::show();

                // RẤT QUAN TRỌNG: Xóa dữ liệu 3D cũ đi sau khi đóng cửa sổ
                polyscope::removeAllStructures();
            }
        }
    }

    std::cout << "==================================================\n";
    std::cout << "Đã hoàn tất xử lý tất cả các file!\n";

    // ReleaseOpenCLEnvironment(clEnv);
}

int main() {
    try {
        initialize();
    }
    catch (std::runtime_error& e) {
        std::cerr << "\n[LỖI NGHIÊM TRỌNG]: " << e.what() << "\n";
        system("pause");
    }
    return 0;
}