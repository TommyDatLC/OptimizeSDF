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
    VertexNormalMatrix = nullptr;
    FaceNormalMatrix = nullptr;
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

    if (VertexNormalMatrix == nullptr) {
        VertexNormalMatrix = matrixMemMang.CreateMatrixPointer(4, numVertices);
    }
    if (FaceNormalMatrix == nullptr) {
        FaceNormalMatrix = matrixMemMang.CreateMatrixPointer(4, numFaces);
    }

    vertices.CopyToDevice();
    indices.CopyToDevice();

    cudaMemset(VertexNormalMatrix->getDevicePtr(), 0, VertexNormalMatrix->GetSize());
    cudaMemset(FaceNormalMatrix->getDevicePtr(), 0, FaceNormalMatrix->GetSize());

    int blockSize = 256;
    int gridSizeFaces = (numFaces + blockSize - 1) / blockSize;
    int gridSizeVertices = (numVertices + blockSize - 1) / blockSize;

    GPUNormalCaculation<<<gridSizeFaces, blockSize>>>(
        vertices.getDevicePtr(), vertices.Height,
        indices.getDevicePtr(), indices.Height, numFaces,
        FaceNormalMatrix->getDevicePtr(),
        VertexNormalMatrix->getDevicePtr()
    );
    cudaDeviceSynchronize();

    GPUNormalizeVertexNormal<<<gridSizeVertices, blockSize>>>(
        VertexNormalMatrix->getDevicePtr(), numVertices
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

    if (VertexNormalMatrix == nullptr || FaceNormalMatrix == nullptr) {
        throw std::runtime_error("PointNormalMatrix is null. Call UpdateNormal() first.");
    }

    NormalClippingSpaceConversion(V, P, *VertexNormalMatrix, *FaceNormalMatrix);

    vertices.CopyToHost();
    if (VertexNormalMatrix) VertexNormalMatrix->CopyToHost();
    if (FaceNormalMatrix) FaceNormalMatrix->CopyToHost();
}

// Display in the engine
void Model::AddToScene(std::string name, bool displayNormal ) {
    // 1. Đăng ký bề mặt mesh
    auto* psMesh = polyscope::registerSurfaceMesh(name, vertex, faces);

    // 2. [TÍNH NĂNG MỚI]: Hiển thị Heat Map (Bản đồ nhiệt độ / Lỗi / SDF)
    if (vertexAttributes.size() == vertex.size()) {
        auto* heatMapQ = psMesh->addVertexScalarQuantity("Heat Map Attributes", vertexAttributes);
        heatMapQ->setEnabled(true);
        // Có thể đổi dải màu thành "turbo", "jet", hoặc "viridis" tùy thẩm mỹ
        heatMapQ->setColorMap("turbo");
    }

    // 3. Hiển thị Normal
    if (displayNormal) {
        if (VertexNormalMatrix == nullptr || FaceNormalMatrix == nullptr) {
            throw std::runtime_error("Normals have not been calculated yet. Please call UpdateNormal() first.");
        }

        VertexNormalMatrix->CopyToHost();
        FaceNormalMatrix->CopyToHost();
        cudaDeviceSynchronize();

        // -----------------------------------------------------------------
        // CÁCH CHUẨN XÁC ĐỂ XEM NORMAL TRONG CLIP SPACE: DÙNG MÀU SẮC (RGB)
        // -----------------------------------------------------------------
        std::vector<std::array<double, 3>> vertexColors;
        std::vector<std::array<double, 3>> vertexNormalsVector; // Giữ lại mũi tên nếu bạn vẫn muốn so sánh

        vertexColors.reserve(vertex.size());
        vertexNormalsVector.reserve(vertex.size());

        for(size_t i = 0; i < vertex.size(); i++) {
            double nx = (double)VertexNormalMatrix->GetHost(0, i);
            double ny = (double)VertexNormalMatrix->GetHost(1, i);
            double nz = (double)VertexNormalMatrix->GetHost(2, i);

            // Map giá trị từ [-1, 1] sang màu RGB [0, 1]
            vertexColors.push_back({(nx + 1.0) / 2.0, (ny + 1.0) / 2.0, (nz + 1.0) / 2.0});
            vertexNormalsVector.push_back({nx, ny, nz});
        }

        std::vector<std::array<double, 3>> faceColors;
        std::vector<std::array<double, 3>> faceNormalsVector;

        faceColors.reserve(faces.size());
        faceNormalsVector.reserve(faces.size());

        for(size_t i = 0; i < faces.size(); i++) {
            double nx = (double)FaceNormalMatrix->GetHost(0, i);
            double ny = (double)FaceNormalMatrix->GetHost(1, i);
            double nz = (double)FaceNormalMatrix->GetHost(2, i);

            faceColors.push_back({(nx + 1.0) / 2.0, (ny + 1.0) / 2.0, (nz + 1.0) / 2.0});
            faceNormalsVector.push_back({nx, ny, nz});
        }

        // Đăng ký màu sắc Normal (Màu Đỏ = X, Màu Xanh lá = Y, Màu Xanh dương = Z)
        auto* vColor = psMesh->addVertexColorQuantity("Clip Space Normal (Colors)", vertexColors);
        auto* fColor = psMesh->addFaceColorQuantity("Clip Space Face Normal (Colors)", faceColors);
        fColor->setEnabled(true); // Bật mặc định cái này để xem bề mặt!

        // Vẫn đăng ký mũi tên vector để bạn thấy rõ sự biến dạng của Clip Space
        psMesh->addVertexVectorQuantity("Distorted Vectors (Mũi tên méo)", vertexNormalsVector);
        psMesh->addFaceVectorQuantity("Distorted Face Vectors (Mũi tên méo)", faceNormalsVector);
    }
}