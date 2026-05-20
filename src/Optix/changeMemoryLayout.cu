// KERNEL: Ép kiểu mảng Matrix (float) thành mảng float3 cho OptiX
inline __global__ void ConvertMatrixToFloat3(const float* matData, float3* outData, int numElements, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        outData[idx] = make_float3(
            matData[idx * height + 0],
            matData[idx * height + 1],
            matData[idx * height + 2]
        );
    }
}

// KERNEL: Ép kiểu mảng Matrix (float) thành mảng uint3 cho OptiX
inline __global__ void ConvertMatrixToUint3(const float* matData, uint3* outData, int numElements, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        outData[idx] = make_uint3(
            (unsigned int)matData[idx * height + 0],
            (unsigned int)matData[idx * height + 1],
            (unsigned int)matData[idx * height + 2]
        );
    }
}

// =========================================================================================
// KERNEL LÀM MƯỢT DỊ HƯỚNG TRÊN GPU (ANISOTROPIC SMOOTHING)
// Sử dụng cấu trúc danh sách kề dạng CSR (Compressed Sparse Row) để truy xuất cực nhanh
// =========================================================================================
inline __global__ void AnisotropicSmoothingKernel(
    const float3* vertices,
    const float* sdfIn,
    float* sdfOut,
    const int* neighborOffsets,
    const int* neighborLists,
    int numVertices,
    float sigmaSpatial,
    float sigmaRange)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    float centerSDF = sdfIn[idx];
    float3 centerPos = vertices[idx];

    float sumWeights = 1.0f; // Trọng số của chính đỉnh trung tâm = 1
    float sumSDF = centerSDF * 1.0f;

    // Lấy khoảng con trỏ trong danh sách láng giềng 1D của đỉnh 'idx'
    int start = neighborOffsets[idx];
    int end = neighborOffsets[idx + 1];

    for (int i = start; i < end; ++i) {
        int nIdx = neighborLists[i];
        float nSDF = sdfIn[nIdx];
        float3 nPos = vertices[nIdx];

        // 1. Trọng số Không gian (Sự xa gần của tam giác)
        float dx = centerPos.x - nPos.x;
        float dy = centerPos.y - nPos.y;
        float dz = centerPos.z - nPos.z;
        float distSq = dx*dx + dy*dy + dz*dz;

        // 2. Trọng số Tính năng (Sự chênh lệch giá trị SDF để bảo vệ ranh giới)
        float valDiff = centerSDF - nSDF;

        // 3. Công thức Bộ lọc Song phương (Bilateral Filter)
        float w = expf(-distSq / (2.0f * sigmaSpatial * sigmaSpatial)) *
                  expf(-(valDiff * valDiff) / (2.0f * sigmaRange * sigmaRange));

        sumWeights += w;
        sumSDF += nSDF * w;
    }

    sdfOut[idx] = sumSDF / sumWeights;
}