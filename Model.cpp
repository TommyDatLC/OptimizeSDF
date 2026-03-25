//
// Created by tommydatlc on 3/25/26.
//

#define CUBLAS_API

#include "Model.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Include thư viện Polyscope
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"



// ---------------------------------------------------------
// 1. CONSTRUCTOR: Khởi tạo con trỏ cache và đọc file ngay lập tức
// ---------------------------------------------------------
Model::Model(std::string filename) {
    cacheVertexMatrix = nullptr;
    cacheIndicesMatrix = nullptr;
    ReadFromObjFile(filename);
}
// ---------------------------------------------------------
// 2. READ & SAVE OBJ
// ---------------------------------------------------------
void Model::ReadFromObjFile(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Lỗi: Không thể mở file " << filename << "\n";
        return;
    }

    Vertices.clear();
    VertexIndices.clear();
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float3 vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            Vertices.push_back(vertex);
        } else if (prefix == "f") {
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;

            auto getIndex = [](const std::string& token) {
                return std::stof(token.substr(0, token.find('/'))) - 1.0f;
            };

            float3 face;
            face.x = getIndex(v1);
            face.y = getIndex(v2);
            face.z = getIndex(v3);
            VertexIndices.push_back(face);
        }
    }
    file.close();
    std::cout << "Đọc thành công file: " << filename << " (" << Vertices.size() << " đỉnh)\n";
}

void Model::SaveObjFile(std::string filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Lỗi: Không thể lưu file " << filename << "\n";
        return;
    }

    for (const auto& v : Vertices) {
        file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }

    for (const auto& f : VertexIndices) {
        file << "f " << (f.x + 1) << " " << (f.y + 1) << " " << (f.z + 1) << "\n";
    }

    file.close();
    std::cout << "Đã lưu mô hình ra file: " << filename << "\n";
}

// ---------------------------------------------------------
// 3. XỬ LÝ MATRIX (CACHE & SET)
// ---------------------------------------------------------

Matrix& Model::GetVertexMatrix() {
    if (cacheVertexMatrix != nullptr) {
        return *cacheVertexMatrix;
    }

    cacheVertexMatrix = matrixMemMang.CreateMatrixPointer(4, Vertices.size());

    // Nạp dữ liệu vào ma trận (Bỏ comment và sửa lại hàm SetValue theo class Matrix của bạn)
    for (size_t col = 0; col < Vertices.size(); ++col) {
        cacheVertexMatrix->Set(0, col, Vertices[col].x);
        cacheVertexMatrix->Set(1, col, Vertices[col].y);
        cacheVertexMatrix->Set(2, col, Vertices[col].z);
        cacheVertexMatrix->Set(3, col, 1);
    }

    return *cacheVertexMatrix;
}

Matrix& Model::GetVertexIndicesMatrix() {
    if (cacheIndicesMatrix != nullptr) {
        return *cacheIndicesMatrix;
    }

    cacheIndicesMatrix = matrixMemMang.CreateMatrixPointer(3, VertexIndices.size());

    for (size_t col = 0; col < VertexIndices.size(); ++col) {
        cacheIndicesMatrix->Set(0, col, VertexIndices[col].x);
        cacheIndicesMatrix->Set(1, col, VertexIndices[col].y);
        cacheIndicesMatrix->Set(2, col, VertexIndices[col].z);
    }

    return *cacheIndicesMatrix;
}

void Model::SetVertexMatrix(Matrix& newVertex) {
    // 1. Lấy số lượng đỉnh từ chiều rộng của ma trận (mỗi cột là 1 đỉnh)
    int numVertices = newVertex.Width;

    // 2. BẮT BUỘC: Đồng bộ dữ liệu mới nhất từ GPU (Device) về CPU (Host) trước khi đọc
    newVertex.CopyToHost();

    // 3. Chuẩn bị vector
    Vertices.clear();
    Vertices.resize(numVertices);

    // 4. Cập nhật lại vector Vertices từ ma trận
    // Sử dụng hàm Get(h, w) - trong đó h (hàng) là x, y, z; w (cột) là index của đỉnh
    for (int col = 0; col < numVertices; ++col) {
        Vertices[col].x = newVertex.Get(0, col); // Hàng 0 là X
        Vertices[col].y = newVertex.Get(1, col); // Hàng 1 là Y
        Vertices[col].z = newVertex.Get(2, col); // Hàng 2 là Z
    }

    // 5. Reset lại cache pointer để lần sau gọi GetVertexMatrix(),
    // nó sẽ tự động nạp lại dữ liệu mới từ vector Vertices
    cacheVertexMatrix = nullptr;
}

// ---------------------------------------------------------
// 4. PREVIEW VỚI POLYSCOPE
// ---------------------------------------------------------
void Model::Preview() {
    polyscope::init();

    std::vector<std::array<double, 3>> pts;
    pts.reserve(Vertices.size());
    for(const auto& v : Vertices) {
        pts.push_back({(double)v.x, (double)v.y, (double)v.z});
    }

    std::vector<std::array<size_t, 3>> faces;
    faces.reserve(VertexIndices.size());
    for(const auto& f : VertexIndices) {
        faces.push_back({(size_t)f.x, (size_t)f.y, (size_t)f.z});
    }

    polyscope::registerSurfaceMesh("Preview Model", pts, faces);
    polyscope::show();
}