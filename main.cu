#include <string>
#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include <filesystem> // Yêu cầu C++17
#include "src/Optix/interface.cu"
#include "src/Compare.hpp"

#include "polyscope/polyscope.h"
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

    std::cout << "==================================================\n";
    std::cout << "BẮT ĐẦU KHỞI TẠO HỆ THỐNG TRUY VẤN OPTIX\n";
    OptixGlobalState optixState = InitializeOptixGlobalState(optixPTX);

    std::cout << "Bắt đầu quét thư mục: " << folderPath << "\n";

    // Đổi thành true để hiển thị Polyscope cửa sổ 3D
    bool hien_thi_3d = false;

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

            Model modelPy(filePath);
            // Dự đoán tên file .sdf sinh ra bởi Python
            std::string base_name = entry.path().stem().string(); // Lấy tên không đuôi (VD: 112)
            std::string parent_dir = entry.path().parent_path().string(); // Lấy thư mục chứa nó (VD: Model)
            std::string sdfFilePath = parent_dir + "/" + base_name + "_pymeshlab.sdf";

            // Đọc file PyMeshLab SDF và gán vào modelPy
            LoadPyHeatMap(modelPy, sdfFilePath);
            modelPy.SetShowHeatMap(true); // Bật hiển thị Heat Map cho PyMeshLab

            // Đăng ký modelPy lên Polyscope với tên riêng biệt
            modelPy.AddToScene("PyMeshLab_SDF_Model", false);

            // -------------------------------------------------------------
            // HEAT MAP 2: TÍNH TOÁN SDF BẰNG C++ OPTIX (HARDWARE RAY TRACING)
            // -------------------------------------------------------------

            Model modelOptix(filePath);
            // Chạy tính toán bằng OptiX Shader
            CaculatingSDFUsingOptix(modelOptix, optixState,64,150);

            // -------------------------------------------------------------
            // HEAT MAP 3: SO SÁNH SỰ KHÁC BIỆT (DIFFERENCE)
            // -------------------------------------------------------------
            Model modelDiff(filePath);
            const auto& attrPy = modelPy.GetVertexAttributes();
            const auto& attrOptix = modelOptix.GetVertexAttributes();

            if (attrPy.size() > 0 && attrPy.size() == attrOptix.size()) {
                for (size_t i = 0; i < attrPy.size(); i++) {
                    double diff = std::abs(attrOptix[i] - attrPy[i]);
                    modelDiff.AddHeatMapVertexForPreviewEngine(i, diff);
                }
                modelDiff.SetShowHeatMap(true);
                modelDiff.AddToScene("Difference_SDF_Model", false);
            } else {
                std::cout << "[CẢNH BÁO] Không thể tính Difference vì số lượng đỉnh của PyMeshLab (" << attrPy.size() 
                          << ") và Optix (" << attrOptix.size() << ") không khớp hoặc bằng 0!\n";
            }

            // -------------------------------------------------------------
            // HIỂN THỊ GIAO DIỆN ĐỐI CHIẾU 3D
            // -------------------------------------------------------------
            if (hien_thi_3d) {
                std::cout << "-> Dang mo cua so Polyscope. Hay dong cua so de chay file tiep theo...\n";
                polyscope::show();

                // RẤT QUAN TRỌNG: Xóa dữ liệu 3D cũ đi sau khi đóng cửa sổ
                polyscope::removeAllStructures();
            }
        }
    }

    DestroyOptixGlobalState(optixState);

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