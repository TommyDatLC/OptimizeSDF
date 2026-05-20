#include <string>
#include "../Core/Model.cuh"
// =========================================================================
// HÀM ĐỌC FILE SDF (SINH RA TỪ PYMESHLAB) VÀO MÔ HÌNH (ĐÃ SỬA LỖI ĐỌC FILE)
// =========================================================================
inline void LoadPyHeatMap(Model& model, const std::string& pymeshlab_sdf_file_directory)
{
    std::cout << "   [DEBUG] Đang cố gắng mở file: " << pymeshlab_sdf_file_directory << "\n";
    std::ifstream file(pymeshlab_sdf_file_directory);
    if (!file.is_open()) {
        std::cerr << "   [-] Cảnh báo: Không tìm thấy file! Hãy chắc chắn Python đã chạy và sinh file.\n";
        return;
    }

    std::string line;
    int vertexCount = 0;
    int lineCount = 0; // Đếm số dòng vật lý trong file Text

    while (std::getline(file, line)) {
        lineCount++;

        // Loại bỏ ký tự xuống dòng ẩn của Windows
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string id_str, val_str;

        if (std::getline(ss, id_str, ',') && std::getline(ss, val_str)) {
            try {
                int vertexIndex = std::stoi(id_str);
                double attribute = std::stod(val_str);

                // [DEBUG]: In thử 3 dòng đầu tiên ra để soi cấu trúc dữ liệu
                if (vertexCount < 3) {
                    std::cout << "      [DEBUG-READ] Đọc dòng " << lineCount
                              << " -> ID: " << vertexIndex << " | SDF Value: " << attribute << "\n";
                }

                model.AddHeatMapVertexForPreviewEngine(vertexIndex, attribute);
                vertexCount++;
            } catch (const std::exception& e) {
                std::cout << "      [DEBUG-ERROR] Lỗi parse dữ liệu tại dòng " << lineCount << ": " << line << "\n";
            }
        }
    }
}