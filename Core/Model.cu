#define CUBLAS_API

#include "Model.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "ModelHelper.cu"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

Model::Model(std::string filename) {
    cacheVertexMatrix = nullptr;
    cacheIndicesMatrix = nullptr;
    vertexNormalMatrix = nullptr;
    faceNormalMatrix = nullptr;
    ReadFromObjFile(filename);
}

void Model::ReadFromObjFile(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Lỗi: Không thể mở file");

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
            std::array<size_t, 3> f;
            std::string temp;
            for (int i = 0; i < 3; i++) {
                iss >> temp;
                size_t pos = temp.find('/');
                if (pos != std::string::npos) temp = temp.substr(0, pos);
                f[i] = std::stoull(temp) - 1; // OBJ file bắt đầu từ 1, chuyển về 0
            }
            faces.push_back(f);
        }
    }
}

Matrix<float>& Model::GetVertexMatrix() {
    if (cacheVertexMatrix == nullptr) {
        std::cout << "[DEBUG] Chuyển đổi dữ liệu Model (Vertex) sang Matrix...\n";
        std::cout << "[DEBUG] Tổng số đỉnh: " << vertex.size() << "\n";
        if (vertex.size() > 0) {
            std::cout << "[DEBUG] Đỉnh đầu tiên: (" << vertex[0][0] << ", " << vertex[0][1] << ", " << vertex[0][2] << ")\n";
        }
        // ROW-MAJOR: Chiều cao (Height) = Số đỉnh, Chiều rộng (Width) = 3 (X, Y, Z)
        cacheVertexMatrix = new Matrix<float>(vertex.size(), 3, nullptr);
        for(size_t i = 0; i < vertex.size(); i++) {
            cacheVertexMatrix->SetHost(i, 0, (float)vertex[i][0]);
            cacheVertexMatrix->SetHost(i, 1, (float)vertex[i][1]);
            cacheVertexMatrix->SetHost(i, 2, (float)vertex[i][2]);
        }
        cacheVertexMatrix->CopyToDevice();
    }
    return *cacheVertexMatrix;
}

Matrix<unsigned int>& Model::GetVertexIndicesMatrix() {
    if (cacheIndicesMatrix == nullptr) {
        std::cout << "[DEBUG] Chuyển đổi dữ liệu Model (Indices) sang Matrix...\n";
        std::cout << "[DEBUG] Tổng số mặt: " << faces.size() << "\n";
        if (faces.size() > 0) {
            std::cout << "[DEBUG] Mặt đầu tiên (indices): (" << faces[0][0] << ", " << faces[0][1] << ", " << faces[0][2] << ")\n";
        }
        // ROW-MAJOR: Chiều cao (Height) = Số mặt, Chiều rộng (Width) = 3 (V1, V2, V3)
        // Kiểu dữ liệu 'unsigned int' khớp chuẩn nhị phân 100% với uint3 của OptiX
        cacheIndicesMatrix = new Matrix<unsigned int>(faces.size(), 3, nullptr);
        for(size_t i = 0; i < faces.size(); i++) {
            cacheIndicesMatrix->SetHost(i, 0, (unsigned int)faces[i][0]);
            cacheIndicesMatrix->SetHost(i, 1, (unsigned int)faces[i][1]);
            cacheIndicesMatrix->SetHost(i, 2, (unsigned int)faces[i][2]);
        }
        cacheIndicesMatrix->CopyToDevice();
    }
    return *cacheIndicesMatrix;
}

void Model::UpdateNormal() {
    if (vertexNormalMatrix == nullptr) {
        vertexNormalMatrix = new Matrix<float>(vertex.size(), 3, nullptr);
    }
    cudaMemset(vertexNormalMatrix->getDevicePtr(), 0, vertexNormalMatrix->GetSize());

    // Đảm bảo dữ liệu đã nằm trên GPU
    GetVertexMatrix().CopyToDevice();
    GetVertexIndicesMatrix().CopyToDevice();

    std::cout << "[DEBUG] Đang tính toán Vertex Normal trên GPU...\n";

    int blockSize = 256;
    int gridSizeFaces = (faces.size() + blockSize - 1) / blockSize;
    std::cout << "[DEBUG] Cấu hình Kernel (Faces): Grid = " << gridSizeFaces << ", Block = " << blockSize << "\n";

    // Ép kiểu trực tiếp lấy con trỏ Raw đưa vào Kernel
    GPUNormalCaculation<<<gridSizeFaces, blockSize>>>(
        (const float3*)GetVertexMatrix().getDevicePtr(),
        (const uint3*)GetVertexIndicesMatrix().getDevicePtr(),
        faces.size(),
        (float3*)vertexNormalMatrix->getDevicePtr()
    );

    int gridSizeVerts = (vertex.size() + blockSize - 1) / blockSize;
    std::cout << "[DEBUG] Cấu hình Kernel (Vertices): Grid = " << gridSizeVerts << ", Block = " << blockSize << "\n";
    GPUNormalizeVertexNormal<<<gridSizeVerts, blockSize>>>(
        (float3*)vertexNormalMatrix->getDevicePtr(),
        vertex.size()
    );
    cudaDeviceSynchronize();
    
    vertexNormalMatrix->CopyToHost();
    if (vertex.size() > 0) {
        std::cout << "[DEBUG] Normal của đỉnh đầu tiên sau khi tính toán (GPU->Host): (" 
                  << vertexNormalMatrix->GetHost(0, 0) << ", " 
                  << vertexNormalMatrix->GetHost(0, 1) << ", " 
                  << vertexNormalMatrix->GetHost(0, 2) << ")\n";
    }
}

Matrix<float>& Model::GetVertexNormalMatrix() {
    if (vertexNormalMatrix == nullptr) UpdateNormal();
    return *vertexNormalMatrix;
}

void Model::AddHeatMapVertexForPreviewEngine(int index, double value) {
    if (vertexAttributes.size() != vertex.size()) {
        vertexAttributes.resize(vertex.size(), 0.0);
    }
    vertexAttributes[index] = value;
}

void Model::SetShowHeatMap(bool show) {
    showHeatMap = show;
}

void Model::AddToScene(std::string name, bool displayNormal) {
    auto* psMesh = polyscope::registerSurfaceMesh(name, vertex, faces);

    if (showHeatMap && vertexAttributes.size() == vertex.size()) {
        auto* heatMapQ = psMesh->addVertexScalarQuantity("SDF Heat Map", vertexAttributes);
        heatMapQ->setEnabled(true);
        heatMapQ->setColorMap("turbo");
        heatMapQ->resetMapRange();
    }

    if (displayNormal) {
        // Vertex Normals
        Matrix<float>& normalsMat = GetVertexNormalMatrix();
        normalsMat.CopyToHost(); // Đảm bảo dữ liệu normal có trên RAM
        std::vector<std::array<double, 3>> normalsList(vertex.size());
        for (size_t i = 0; i < vertex.size(); i++) {
            normalsList[i] = {
                (double)normalsMat.GetHost(i, 0),
                (double)normalsMat.GetHost(i, 1),
                (double)normalsMat.GetHost(i, 2)
            };
        }
        psMesh->addVertexVectorQuantity("normals", normalsList);

        // Face Normals
        std::vector<std::array<double, 3>> faceNormals(faces.size());
        for (size_t i = 0; i < faces.size(); i++) {
            auto v0 = vertex[faces[i][0]];
            auto v1 = vertex[faces[i][1]];
            auto v2 = vertex[faces[i][2]];
            
            double e1x = v1[0] - v0[0]; double e1y = v1[1] - v0[1]; double e1z = v1[2] - v0[2];
            double e2x = v2[0] - v0[0]; double e2y = v2[1] - v0[1]; double e2z = v2[2] - v0[2];
            
            double nx = e1y * e2z - e1z * e2y;
            double ny = e1z * e2x - e1x * e2z;
            double nz = e1x * e2y - e1y * e2x;
            
            double len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len > 1e-8) {
                faceNormals[i] = {nx/len, ny/len, nz/len};
            } else {
                faceNormals[i] = {0.0, 0.0, 1.0};
            }
        }
        psMesh->addFaceVectorQuantity("face_normals", faceNormals);
    }
}