#ifndef SDF_KERNELS_CUH
#define SDF_KERNELS_CUH

#include <cuda_runtime.h>
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <cub/cub.cuh>
#include <cmath>

// =========================================================================================
// ATOMIC MIN/MAX CHO FLOAT
// =========================================================================================
inline __device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
    int* address_as_int = (int*)addr;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

inline __device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    int* address_as_int = (int*)addr;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

// =========================================================================================
// KERNEL: BOUNDING BOX & NORMALIZATION
// =========================================================================================
inline __global__ void GPUComputeBoundingBox(const float3* vertices, int numVertices, float* d_minMax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    float3 v = vertices[idx];
    atomicMinFloat(&d_minMax[0], v.x); atomicMaxFloat(&d_minMax[1], v.x);
    atomicMinFloat(&d_minMax[2], v.y); atomicMaxFloat(&d_minMax[3], v.y);
    atomicMinFloat(&d_minMax[4], v.z); atomicMaxFloat(&d_minMax[5], v.z);
}

inline __global__ void GPUComputeSDFMinMax(const float* rawSDF, int numVertices, float* d_minMaxSDF) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    float val = rawSDF[idx];
    atomicMinFloat(&d_minMaxSDF[0], val);
    atomicMaxFloat(&d_minMaxSDF[1], val);
}

inline __global__ void GPUApplySDFNormalization(float* rawSDF, int numVertices, const float* d_minMaxSDF) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    
    float minSDF = d_minMaxSDF[0];
    float maxSDF = d_minMaxSDF[1];
    float alpha = 4.0f;
    float logAlphaPlus1 = logf(alpha + 1.0f);
    
    if (maxSDF - minSDF > 1e-6f) {
        float normalized = (rawSDF[idx] - minSDF) / (maxSDF - minSDF);
        rawSDF[idx] = logf(normalized * alpha + 1.0f) / logAlphaPlus1;
    } else {
        rawSDF[idx] = 0.0f;
    }
}

// =========================================================================================
// CÁC KERNEL HỖ TRỢ TÍNH TOÁN SDF BẰNG CUB
// =========================================================================================
inline __global__ void GPUGenerateOffsets(int* d_offsets, int numVertices, int raysPerPoint) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= numVertices) {
        d_offsets[idx] = idx * raysPerPoint;
    }
}

inline __global__ void GPUComputeRawSDF(
    float* d_distances,
    float* d_weights,
    const int* d_validCounts,
    float* d_rawSDF,
    int numVertices,
    int raysPerPoint) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    int validCount = d_validCounts[idx];
    if (validCount == 0) {
        d_rawSDF[idx] = 0.0f;
        return;
    }

    int baseIdx = idx * raysPerPoint;

    float totalWeightedDist = 0.0f;
    float totalWeight = 0.0f;

    // Không dùng Outlier Removal: Tính thẳng trung bình có trọng số của TẤT CẢ các tia
    for (int i = 0; i < validCount; i++) {
        float dist = d_distances[baseIdx + i];
        float weight = d_weights[baseIdx + i];
        
        totalWeightedDist += dist * weight;
        totalWeight += weight;
    }
    
    d_rawSDF[idx] = (totalWeight > 0.0f) ? (totalWeightedDist / totalWeight) : 0.0f;
}

inline __global__ void GPUGenerateEdges(const uint3* indices, int numFaces, uint64_t* d_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numFaces) return;
    uint3 face = indices[idx];
    int base = idx * 6;
    d_edges[base + 0] = ((uint64_t)face.x << 32) | face.y;
    d_edges[base + 1] = ((uint64_t)face.x << 32) | face.z;
    d_edges[base + 2] = ((uint64_t)face.y << 32) | face.x;
    d_edges[base + 3] = ((uint64_t)face.y << 32) | face.z;
    d_edges[base + 4] = ((uint64_t)face.z << 32) | face.x;
    d_edges[base + 5] = ((uint64_t)face.z << 32) | face.y;
}

inline __global__ void GPUExtractCSR(const uint64_t* d_uniqueEdges, int numUniqueEdges, int* d_nbrOffsets, int* d_nbrLists, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numUniqueEdges) return;
    
    uint64_t edge = d_uniqueEdges[idx];
    int src = (int)(edge >> 32);
    int dst = (int)(edge & 0xFFFFFFFF);
    
    d_nbrLists[idx] = dst;
    
    if (idx == 0) {
        d_nbrOffsets[src] = 0;
        for (int i = 0; i < src; i++) d_nbrOffsets[i] = 0;
    } else {
        uint64_t prevEdge = d_uniqueEdges[idx - 1];
        int prevSrc = (int)(prevEdge >> 32);
        if (src != prevSrc) {
            for (int i = prevSrc + 1; i <= src; i++) d_nbrOffsets[i] = idx;
        }
    }
    if (idx == numUniqueEdges - 1) {
        for (int i = src + 1; i <= numVertices; i++) d_nbrOffsets[i] = numUniqueEdges;
    }
}

// =========================================================================================
// KERNEL: LÀM MƯỢT DỊ HƯỚNG TRÊN GPU (ANISOTROPIC SMOOTHING)
// =========================================================================================
inline __global__ void AnisotropicSmoothingKernel(
    const float3* vertices, const float* sdfIn, float* sdfOut,
    const int* neighborOffsets, const int* neighborLists,
    int numVertices, float sigmaSpatial, float sigmaRange)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    float centerSDF = sdfIn[idx];
    float3 centerPos = vertices[idx];

    float sumWeights = 1.0f;
    float sumSDF = centerSDF * 1.0f;

    int start = neighborOffsets[idx];
    int end = neighborOffsets[idx + 1];

    for (int i = start; i < end; ++i) {
        int nIdx = neighborLists[i];
        float nSDF = sdfIn[nIdx];
        float3 nPos = vertices[nIdx];

        float dx = centerPos.x - nPos.x;
        float dy = centerPos.y - nPos.y;
        float dz = centerPos.z - nPos.z;
        float distSq = dx*dx + dy*dy + dz*dz;

        float valDiff = centerSDF - nSDF;
        float w = expf(-distSq / (2.0f * sigmaSpatial * sigmaSpatial)) *
                  expf(-(valDiff * valDiff) / (2.0f * sigmaRange * sigmaRange));

        sumWeights += w;
        sumSDF += nSDF * w;
    }
    sdfOut[idx] = sumSDF / sumWeights;
}

#endif // SDF_KERNELS_CUH
