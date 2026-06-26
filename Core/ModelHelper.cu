#ifndef ModelHelper
#define ModelHelper

#include <device_launch_parameters.h>
#include "MathHelper.cu"

// =========================================================================
// KERNEL 1: TÍNH NORMAL MẶT PHẲNG VÀ CỘNG DỒN VÀO ĐỈNH
// Tối ưu hóa: Ép kiểu trực tiếp con trỏ Row-Major thành float3 và uint3
// =========================================================================
__global__ inline void GPUNormalCaculation(
    const float3* __restrict__ vertices,
    const uint3* __restrict__ indices,
    int numFaces,
    float3* __restrict__ pointsNormal)
{
    int faceIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceIdx >= numFaces) return;

    // Đọc trực tiếp cả 3 index của tam giác cùng lúc
    uint3 face = indices[faceIdx];

    // Đọc trực tiếp tọa độ 3 đỉnh của tam giác
    float3 v0 = vertices[face.x];
    float3 v1 = vertices[face.y];
    float3 v2 = vertices[face.z];

    // Tính vector cạnh và tích có hướng (Cross Product)
    float3 edge1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 edge2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 rawCross = cross(edge1, edge2);

    // Sử dụng ép kiểu con trỏ (float*) để thực hiện atomicAdd cho từng thành phần x, y, z
    // Mảng pointsNormal dạng Row-Major bản chất là 1 dải liên tục: x0,y0,z0, x1,y1,z1...
    float* rawPointNormals = (float*)pointsNormal;

    atomicAdd(&rawPointNormals[face.x * 3 + 0], rawCross.x);
    atomicAdd(&rawPointNormals[face.x * 3 + 1], rawCross.y);
    atomicAdd(&rawPointNormals[face.x * 3 + 2], rawCross.z);

    atomicAdd(&rawPointNormals[face.y * 3 + 0], rawCross.x);
    atomicAdd(&rawPointNormals[face.y * 3 + 1], rawCross.y);
    atomicAdd(&rawPointNormals[face.y * 3 + 2], rawCross.z);

    atomicAdd(&rawPointNormals[face.z * 3 + 0], rawCross.x);
    atomicAdd(&rawPointNormals[face.z * 3 + 1], rawCross.y);
    atomicAdd(&rawPointNormals[face.z * 3 + 2], rawCross.z);
}

// =========================================================================
// KERNEL 2: CHUẨN HÓA LẠI NORMAL CỦA ĐỈNH SAU KHI CỘNG DỒN
// =========================================================================
__global__ inline void GPUNormalizeVertexNormal(float3* pointsNormal, int numVertices) {
    int vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexIdx >= numVertices) return;

    float3 n = pointsNormal[vertexIdx];
    float length = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

    // An toàn tránh NaN nếu có đỉnh nằm cô lập
    if (length > 1e-8f) {
        pointsNormal[vertexIdx] = make_float3(n.x / length, n.y / length, n.z / length);
    } else {
        pointsNormal[vertexIdx] = make_float3(0.0f, 0.0f, 1.0f);
    }
}

#endif