#ifndef ModelHelper
#define ModelHelper

#include <device_launch_parameters.h>
#include "MathHelper.cu"

__device__ inline float GetD(const float* mem, int h, int w, int height) {
    return mem[w * height + h];
}

// KERNEL 1: Tính toán Normal của Mặt phẳng và Cộng dồn vào Đỉnh
__global__ inline void GPUNormalCaculation(
    const float* vertices, int vHeight,
    const float* indices, int iHeight, int numFaces,
    float* facesNormal, float* pointsNormal)
{
    int faceIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceIdx >= numFaces) return;

    int i0 = static_cast<int>(GetD(indices, 0, faceIdx, iHeight));
    int i1 = static_cast<int>(GetD(indices, 1, faceIdx, iHeight));
    int i2 = static_cast<int>(GetD(indices, 2, faceIdx, iHeight));

    float3 v0 = { GetD(vertices, 0, i0, vHeight), GetD(vertices, 1, i0, vHeight), GetD(vertices, 2, i0, vHeight) };
    float3 v1 = { GetD(vertices, 0, i1, vHeight), GetD(vertices, 1, i1, vHeight), GetD(vertices, 2, i1, vHeight) };
    float3 v2 = { GetD(vertices, 0, i2, vHeight), GetD(vertices, 1, i2, vHeight), GetD(vertices, 2, i2, vHeight) };

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 rawCross = cross(edge1, edge2);

    // AN TOÀN TRÁNH NaN: Kiểm tra độ dài trước khi Normalize
    float faceLen = sqrtf(rawCross.x*rawCross.x + rawCross.y*rawCross.y + rawCross.z*rawCross.z);
    float3 faceNorm = (faceLen > 1e-6f) ? make_float3(rawCross.x/faceLen, rawCross.y/faceLen, rawCross.z/faceLen) : make_float3(0,0,1);

    facesNormal[faceIdx * 4 + 0] = faceNorm.x;
    facesNormal[faceIdx * 4 + 1] = faceNorm.y;
    facesNormal[faceIdx * 4 + 2] = faceNorm.z;
    facesNormal[faceIdx * 4 + 3] = 0.0f;

    // Cộng dồn vào Normal của các đỉnh tạo nên mặt này
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

// KERNEL 2: Chuẩn hóa lại Normal của Đỉnh sau khi cộng dồn
__global__ inline void GPUNormalizeVertexNormal(float* pointsNormal, int numVertices) {
    int vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexIdx >= numVertices) return;

    float3 rawNormal = {
        pointsNormal[vertexIdx * 4 + 0],
        pointsNormal[vertexIdx * 4 + 1],
        pointsNormal[vertexIdx * 4 + 2]
    };

    // AN TOÀN TRÁNH NaN: Nếu đỉnh đứng cô lập, gán mặc định hướng Z
    float len = sqrtf(rawNormal.x*rawNormal.x + rawNormal.y*rawNormal.y + rawNormal.z*rawNormal.z);
    float3 norm;
    if (len > 1e-6f) {
        norm = make_float3(rawNormal.x / len, rawNormal.y / len, rawNormal.z / len);
    } else {
        norm = make_float3(0.0f, 0.0f, 1.0f);
    }

    pointsNormal[vertexIdx * 4 + 0] = norm.x;
    pointsNormal[vertexIdx * 4 + 1] = norm.y;
    pointsNormal[vertexIdx * 4 + 2] = norm.z;
    pointsNormal[vertexIdx * 4 + 3] = 0.0f;
}

#endif