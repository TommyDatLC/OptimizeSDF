#define CUBLAS_API

#include "Model.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "ModelHelper.cu"
#include "ClipSpaceConversion.hpp"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "MatrixMemoryManager.cuh"

Model::Model(std::string filename) {
    cacheVertexMatrix = nullptr;
    cacheIndicesMatrix = nullptr;
    vertexNormalMatrix = nullptr;
    faceNormalMatrix = nullptr;
    ReadFromObjFile(filename);
}

void Model::ReadFromObjFile(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Lỗi: Không thể mở file");
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
    if (!file.is_open()) return;

    for (const auto& v : vertex) {
        file << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    for (const auto& f : faces) {
        file << "f " << (f[0] + 1) << " " << (f[1] + 1) << " " << (f[2] + 1) << "\n";
    }
    file.close();
}

Matrix& Model::GetVertexMatrix() {
    if (cacheVertexMatrix != nullptr) return *cacheVertexMatrix;

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
    if (cacheIndicesMatrix != nullptr) return *cacheIndicesMatrix;

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
        vertex[col][0] = newVertex.GetHost(0, col);
        vertex[col][1] = newVertex.GetHost(1, col);
        vertex[col][2] = newVertex.GetHost(2, col);
    }
    cacheVertexMatrix = nullptr;
}

void Model::UpdateNormal() {
    Matrix& vertices = GetVertexMatrix();
    Matrix& indices = GetVertexIndicesMatrix();

    int numVertices = vertices.Width;
    int numFaces = indices.Width;

    if (vertexNormalMatrix == nullptr) {
        vertexNormalMatrix = matrixMemMang.CreateMatrixPointer(4, numVertices);
    }
    if (faceNormalMatrix == nullptr) {
        faceNormalMatrix = matrixMemMang.CreateMatrixPointer(4, numFaces);
    }

    vertices.CopyToDevice();
    indices.CopyToDevice();

    // SỬA LỖI MŨI TÊN NORMAL BỊ MÉO: Ép buộc clear rác bằng byte chuẩn xác
    cudaMemset(vertexNormalMatrix->getDevicePtr(), 0, numVertices * 4 * sizeof(float));
    cudaMemset(faceNormalMatrix->getDevicePtr(), 0, numFaces * 4 * sizeof(float));

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

// Đã cập nhật kiểu dữ liệu tham số để khớp với file Model.cuh
void Model::AddHeatMapVertexForPreviewEngine(int vertexIndex, double attribute) {
    size_t idx = static_cast<size_t>(vertexIndex);
    if (vertexAttributes.size() != vertex.size()) {
        vertexAttributes.assign(vertex.size(), 0.0);
    }
    if (idx < vertexAttributes.size()) {
        vertexAttributes[idx] = attribute;
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
    auto* psMesh = polyscope::registerSurfaceMesh(name, vertex, faces);

    if (vertexAttributes.size() == vertex.size()) {
        auto* heatMapQ = psMesh->addVertexScalarQuantity("Heat Map Attributes", vertexAttributes);
        heatMapQ->setEnabled(true);
        heatMapQ->setColorMap("turbo");

        // CỨU TINH CỦA BẠN: Lệnh tự động bóp dải màu khít với Max/Min thực tế
        heatMapQ->resetMapRange();
    }

    if (displayNormal) {
        if (vertexNormalMatrix == nullptr || faceNormalMatrix == nullptr) return;

        vertexNormalMatrix->CopyToHost();
        faceNormalMatrix->CopyToHost();
        cudaDeviceSynchronize();

        std::vector<std::array<double, 3>> vertexNormalsVector;
        vertexNormalsVector.reserve(vertex.size());

        for(size_t i = 0; i < vertex.size(); i++) {
            double nx = (double)vertexNormalMatrix->GetHost(0, i);
            double ny = (double)vertexNormalMatrix->GetHost(1, i);
            double nz = (double)vertexNormalMatrix->GetHost(2, i);
            vertexNormalsVector.push_back({nx, ny, nz});
        }

        auto* vNormal = psMesh->addVertexVectorQuantity("Vertex Normals (World Space)", vertexNormalsVector);
        vNormal->setEnabled(true);
    }

}
 Matrix &Model::GetVertexNormalMatrix()
 {
    return *vertexNormalMatrix;
 }