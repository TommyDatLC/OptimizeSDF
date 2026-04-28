//
// Created by tommydatlc on 3/25/26.
//

#define CUBLAS_API

#include "Model.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "ModelHelper.cu"
// Include thư viện Polyscope
#include "ClipSpaceConversion.hpp"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "MatrixMemoryManager.cuh"

// ---------------------------------------------------------
// 1. CONSTRUCTOR: Khởi tạo con trỏ cache và đọc file ngay lập tức
// ---------------------------------------------------------
Model::Model(std::string filename) {
    cacheVertexMatrix = nullptr;
    cacheIndicesMatrix = nullptr;
    vertexNormalMatrix = nullptr;
    faceNormalMatrix = nullptr;
    ReadFromObjFile(filename);
}

// ---------------------------------------------------------
// 2. READ & SAVE OBJ
// ---------------------------------------------------------
void Model::ReadFromObjFile(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::__throw_runtime_error("Lỗi: Không thể mở file");
    }

    vertex.clear();
    faces.clear();
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            std::array<double, 3> v;
            iss >> v[0] >> v[1] >> v[2];
            vertex.push_back(v);
        } else if (prefix == "f") {
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;

            auto getIndex = [](const std::string& token) {
                // Ép kiểu chuẩn xác sang size_t thay vì float như cũ
                return static_cast<size_t>(std::stoull(token.substr(0, token.find('/'))) - 1);
            };

            std::array<size_t, 3> f;
            f[0] = getIndex(v1);
            f[1] = getIndex(v2);
            f[2] = getIndex(v3);
            faces.push_back(f);
        }
    }
    file.close();
    std::cout << "Đọc thành công file: " << filename << " (" << vertex.size() << " đỉnh)\n";
}

void Model::SaveObjFile(std::string filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Lỗi: Không thể lưu file " << filename << "\n";
        return;
    }

    for (const auto& v : vertex) {
        file << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }

    for (const auto& f : faces) {
        // Cộng 1 lại khi lưu ra OBJ (OBJ file 1-indexed)
        file << "f " << (f[0] + 1) << " " << (f[1] + 1) << " " << (f[2] + 1) << "\n";
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

    cacheVertexMatrix = matrixMemMang.CreateMatrixPointer(4, vertex.size());

    for (size_t col = 0; col < vertex.size(); ++col) {
        cacheVertexMatrix->SetHost(0, col, static_cast<float>(vertex[col][0]));
        cacheVertexMatrix->SetHost(1, col, static_cast<float>(vertex[col][1]));
        cacheVertexMatrix->SetHost(2, col, static_cast<float>(vertex[col][2]));
        cacheVertexMatrix->SetHost(3, col, 1.0f);
    }
    return *cacheVertexMatrix;
}

Matrix& Model::GetVertexIndicesMatrix() {
    if (cacheIndicesMatrix != nullptr) {
        return *cacheIndicesMatrix;
    }

    cacheIndicesMatrix = matrixMemMang.CreateMatrixPointer(3, faces.size());

    for (size_t col = 0; col < faces.size(); ++col) {
        cacheIndicesMatrix->SetHost(0, col, static_cast<float>(faces[col][0]));
        cacheIndicesMatrix->SetHost(1, col, static_cast<float>(faces[col][1]));
        cacheIndicesMatrix->SetHost(2, col, static_cast<float>(faces[col][2]));
    }
    return *cacheIndicesMatrix;
}

Matrix & Model::GetVertexNormalMatrix() {
    return *vertexNormalMatrix;

}

void Model::SetVertexMatrix(Matrix& newVertex) {
    int numVertices = newVertex.Width;
    newVertex.CopyToHost();

    vertex.clear();
    vertex.resize(numVertices);

    for (int col = 0; col < numVertices; ++col) {
        vertex[col][0] = newVertex.GetHost(0, col); // X
        vertex[col][1] = newVertex.GetHost(1, col); // Y
        vertex[col][2] = newVertex.GetHost(2, col); // Z
    }
    cacheVertexMatrix = nullptr;
}

// ---------------------------------------------------------
// 4. THUẬT TOÁN & PREVIEW VỚI POLYSCOPE
// ---------------------------------------------------------
void Model::UpdateNormal() {
    Matrix& vertices = GetVertexMatrix();
    Matrix& indices = GetVertexIndicesMatrix();

    int numVertices = vertices.Width;
    int numFaces = indices.Width;

    MatrixMemoryManager matrixMemMang;

    if (vertexNormalMatrix == nullptr) {
        vertexNormalMatrix = matrixMemMang.CreateMatrixPointer(4, numVertices);
    }
    if (faceNormalMatrix == nullptr) {
        faceNormalMatrix = matrixMemMang.CreateMatrixPointer(4, numFaces);
    }

    vertices.CopyToDevice();
    indices.CopyToDevice();

    cudaMemset(vertexNormalMatrix->getDevicePtr(), 0, vertexNormalMatrix->GetSize());
    cudaMemset(faceNormalMatrix->getDevicePtr(), 0, faceNormalMatrix->GetSize());

    int blockSize = 256;
    int gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
    int gridSizeVertices = (numVertices + blockSize - 1) / blockSize;

    GPUNormalCaculation<<<gridSizeFaces, blockSize>>>(
        vertices.getDevicePtr(), vertices.Height,
        indices.getDevicePtr(), indices.Height, numFaces,
        faceNormalMatrix->getDevicePtr(),
        vertexNormalMatrix->getDevicePtr()
    );
    cudaDeviceSynchronize();

    GPUNormalizeVertexNormal<<<gridSizeVertices, blockSize>>>(
        vertexNormalMatrix->getDevicePtr(), numVertices
    );
    cudaDeviceSynchronize();

}

// MỚI HOÀN THIỆN: Gán đại lượng vô hướng (Scalar Attribute) cho đỉnh
void Model::AddHeatMapVertexForPreviewEngine(int vertexIndex, double attribute) {

    // Đảm bảo mảng lưu trữ có kích thước bằng với mảng đỉnh (vertex)
    if (vertexAttributes.size() != vertex.size()) {
        vertexAttributes.assign(vertex.size(), 0.0); // Cấp phát rỗng nếu chưa có
    }
    // 3. Cập nhật dữ liệu vào mảng
    if (vertexIndex < vertexAttributes.size()) {
        vertexAttributes[vertexIndex] = attribute;
    }
}



void Model::ToClipSpace(ViewData& V, PerspectiveCameraData& P) {
    Matrix& vertices = GetVertexMatrix();
    VertexClippingSpaceConversion(V, P, vertices);
    SetVertexMatrix(vertices);

    if (vertexNormalMatrix == nullptr || faceNormalMatrix == nullptr) {
        throw std::runtime_error("PointNormalMatrix is null. Call UpdateNormal() first.");
    }

    NormalClippingSpaceConversion(V, P, *vertexNormalMatrix, *faceNormalMatrix);

    vertices.CopyToHost();
    if (vertexNormalMatrix) vertexNormalMatrix->CopyToHost();
    if (faceNormalMatrix) faceNormalMatrix->CopyToHost();
}
void Model::AddToScene(std::string name, bool displayNormal) {
    // 1. Đăng ký bề mặt mesh vào Polyscope
    auto* psMesh = polyscope::registerSurfaceMesh(name, vertex, faces);

    // 2. Hiển thị Heat Map (Bản đồ nhiệt độ / SDF)
    if (vertexAttributes.size() == vertex.size()) {
        auto* heatMapQ = psMesh->addVertexScalarQuantity("SDF / Heat Map", vertexAttributes);
        heatMapQ->setEnabled(true);
        // Đổi sang "turbo" để giống hệt bản màu mặc định của C++ SDF
        heatMapQ->setColorMap("turbo");
    }

    // 3. Hiển thị Normal (World Space)
    if (displayNormal) {
        if (vertexNormalMatrix == nullptr || faceNormalMatrix == nullptr) {
            throw std::runtime_error("Normals have not been calculated yet. Please call UpdateNormal() first.");
        }

        // Kéo dữ liệu từ GPU về Host (RAM)
        vertexNormalMatrix->CopyToHost();
        faceNormalMatrix->CopyToHost();
        cudaDeviceSynchronize();

        // -----------------------------------------------------------------
        // HIỂN THỊ NORMAL TRONG WORLD SPACE: DÙNG VECTOR (MŨI TÊN 3D)
        // -----------------------------------------------------------------
        std::vector<std::array<double, 3>> vertexNormalsVector;
        vertexNormalsVector.reserve(vertex.size());

        for(size_t i = 0; i < vertex.size(); i++) {
            // Giữ nguyên giá trị thô [-1, 1] của vector trong không gian
            double nx = (double)vertexNormalMatrix->GetHost(0, i);
            double ny = (double)vertexNormalMatrix->GetHost(1, i);
            double nz = (double)vertexNormalMatrix->GetHost(2, i);

            vertexNormalsVector.push_back({nx, ny, nz});
        }

        std::vector<std::array<double, 3>> faceNormalsVector;
        faceNormalsVector.reserve(faces.size());

        for(size_t i = 0; i < faces.size(); i++) {
            // Giữ nguyên giá trị thô
            double nx = (double)faceNormalMatrix->GetHost(0, i);
            double ny = (double)faceNormalMatrix->GetHost(1, i);
            double nz = (double)faceNormalMatrix->GetHost(2, i);

            faceNormalsVector.push_back({nx, ny, nz});
        }

        // Đăng ký dưới dạng Mũi tên Vector (Vector Quantity)
        auto* vNormal = psMesh->addVertexVectorQuantity("Vertex Normals (World Space)", vertexNormalsVector);
        auto* fNormal = psMesh->addFaceVectorQuantity("Face Normals (World Space)", faceNormalsVector);

        // Bật hiển thị Vector của Mặt (Face) lên làm mặc định để dễ quan sát
        fNormal->setEnabled(true);

        // (Tùy chọn) Bạn có thể cấu hình thêm độ dài, màu sắc của mũi tên bằng code:
        // fNormal->setVectorColor({0.8, 0.2, 0.2});
        // fNormal->setVectorLengthScale(0.05);
    }
}