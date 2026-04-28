#ifndef ModelHelper
#define ModelHelper

#include <device_launch_parameters.h>
#include "mathHelper.cu"

// Hàm hỗ trợ đọc từ ma trận (thay thế cho GetDevice an toàn trên GPU)
__device__ inline float GetD(const float* mem, int h, int w, int height) {
    return mem[w * height + h];
}

// Kernel tính toán: Chỉ nhận vào con trỏ float* và kích thước
__global__ inline void GPUNormalCaculation(
    const float* vertices, int vHeight,
    const float* indices, int iHeight, int numFaces,
    float* facesNormal, float* pointsNormal)
{
    int faceIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceIdx >= numFaces) return;

    // 1. Lấy chỉ số (index) của 3 đỉnh
    int i0 = static_cast<int>(GetD(indices, 0, faceIdx, iHeight));
    int i1 = static_cast<int>(GetD(indices, 1, faceIdx, iHeight));
    int i2 = static_cast<int>(GetD(indices, 2, faceIdx, iHeight));

    // 2. Lấy tọa độ 3 đỉnh từ mảng (x=0, y=1, z=2)
    float3 v0 = { GetD(vertices, 0, i0, vHeight), GetD(vertices, 1, i0, vHeight), GetD(vertices, 2, i0, vHeight) };
    float3 v1 = { GetD(vertices, 0, i1, vHeight), GetD(vertices, 1, i1, vHeight), GetD(vertices, 2, i1, vHeight) };
    float3 v2 = { GetD(vertices, 0, i2, vHeight), GetD(vertices, 1, i2, vHeight), GetD(vertices, 2, i2, vHeight) };

    // 3. Tính Face Normal (Weight theo diện tích tam giác)
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 rawCross = cross(edge1, edge2);
    
    // Normalize cho Face Normal
    float3 faceNorm = normalize(rawCross);

    // 4. Lưu Face Normal (ĐÃ SỬA: Chiều cao ma trận là 4, nên dùng * 4)
    facesNormal[faceIdx * 4 + 0] = faceNorm.x;
    facesNormal[faceIdx * 4 + 1] = faceNorm.y;
    facesNormal[faceIdx * 4 + 2] = faceNorm.z;
    facesNormal[faceIdx * 4 + 3] = 0.0f; // ĐÃ SỬA: Vector chỉ hướng thì W = 0

    // 5. Cộng dồn vào Vertex Normal (ĐÃ SỬA: BẮT BUỘC DÙNG * 4)
    atomicAdd(&pointsNormal[i0 * 4 + 0], rawCross.x);
    atomicAdd(&pointsNormal[i0 * 4 + 1], rawCross.y);
    atomicAdd(&pointsNormal[i0 * 4 + 2], rawCross.z);

    atomicAdd(&pointsNormal[i1 * 4 + 0], rawCross.x);
    atomicAdd(&pointsNormal[i1 * 4 + 1], rawCross.y);
    atomicAdd(&pointsNormal[i1 * 4 + 2], rawCross.z);

    atomicAdd(&pointsNormal[i2 * 4 + 0], rawCross.x);
    atomicAdd(&pointsNormal[i2 * 4 + 1], rawCross.y);
    atomicAdd(&pointsNormal[i2 * 4 + 2], rawCross.z);
}

// Kernel chuẩn hóa Normal của điểm
__global__ inline void GPUNormalizeVertexNormal(float* pointsNormal, int numVertices) {
    int vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexIdx >= numVertices) return;

    // ĐÃ SỬA: Dùng * 4 để lấy đúng bộ 3 giá trị x, y, z
    float3 rawNormal = {
        pointsNormal[vertexIdx * 4 + 0],
        pointsNormal[vertexIdx * 4 + 1],
        pointsNormal[vertexIdx * 4 + 2]
    };

    float3 norm = normalize(rawNormal);

    pointsNormal[vertexIdx * 4 + 0] = norm.x;
    pointsNormal[vertexIdx * 4 + 1] = norm.y;
    pointsNormal[vertexIdx * 4 + 2] = norm.z;
    pointsNormal[vertexIdx * 4 + 3] = 0.0f; // ĐÃ SỬA: Vector chỉ hướng thì W = 0
}

#endif