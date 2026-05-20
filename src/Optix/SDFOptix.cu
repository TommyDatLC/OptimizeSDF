// File: SDFOptix.cu
#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../Core/MathHelper.cu"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

// Giới hạn số tia để không làm tràn thanh ghi (Registers) khi dùng Insertion Sort
#define MAX_RAYS 128

struct alignas(8) Params {
    float3* vertices;
    float3* normals;
    float* outputSDF;
    OptixTraversableHandle bvhHandle;
    int raysPerPoint;
    float coneAngleRad;
};

extern "C" __constant__ Params params;

// =========================================================================
// THUẬT TOÁN HAMMERSLEY 2D (UNIFORM SAMPLING - KHÔNG NGẪU NHIÊN)
// Phân bố các điểm hoàn toàn đồng đều trên bề mặt nón.
// =========================================================================
__device__ inline float radicalInverse_VdC(unsigned int bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f; // Chia cho 0x100000000
}

__device__ inline float2 hammersley2d(unsigned int i, unsigned int N) {
    return make_float2(float(i) / float(N), radicalInverse_VdC(i));
}

// =========================================================================
// CHƯƠNG TRÌNH CHÍNH
// =========================================================================
extern "C" __global__ void __raygen__sdf_cone() {
    const uint3 idx = optixGetLaunchIndex();

    float3 origin = params.vertices[idx.x];
    float3 normal = params.normals[idx.x];

    // Trục tâm hình nón: Pháp tuyến hướng vào trong
    float3 invertedNormal = make_float3(-normal.x, -normal.y, -normal.z);

    // Hệ tọa độ cục bộ (TBN)
    float3 up = (fabsf(invertedNormal.z) < 0.999f) ? make_float3(0,0,1) : make_float3(1,0,0);
    float3 tangent = normalize(cross(up, invertedNormal));
    float3 bitangent = cross(invertedNormal, tangent);

    float distances[MAX_RAYS];
    float weights[MAX_RAYS];
    int validCount = 0;

    int rayLimit = (params.raysPerPoint > MAX_RAYS) ? MAX_RAYS : params.raysPerPoint;
    float cosThetaMax = cosf(params.coneAngleRad / 2.0f);

    // =====================================================================
    // BƯỚC 1: BẮN TIA UNIFORM VÀ GHI NHẬN MỌI KẾT QUẢ
    // =====================================================================
    for (int i = 0; i < rayLimit; ++i) {
        // Lấy tọa độ (u, v) đồng đều từ chuỗi Hammersley
        float2 uv = hammersley2d((unsigned int)i, (unsigned int)rayLimit);

        // Nội suy ra tọa độ trên hình nón
        float z = cosThetaMax + (1.0f - cosThetaMax) * uv.x;
        float phi = 2.0f * M_PIf * uv.y;
        float radius = sqrtf(fmaxf(0.0f, 1.0f - z*z));

        float localX = radius * cosf(phi);
        float localY = radius * sinf(phi);
        float localZ = z;

        // Xoay tia theo TBN
        float3 rayDirection = make_float3(
            localX * tangent.x + localY * bitangent.x + localZ * invertedNormal.x,
            localX * tangent.y + localY * bitangent.y + localZ * invertedNormal.y,
            localX * tangent.z + localY * bitangent.z + localZ * invertedNormal.z
        );
        rayDirection = normalize(rayDirection);

        uint32_t hitDistanceFloatAsInt = __float_as_uint(-1.0f);
        optixTrace(
            params.bvhHandle, origin, rayDirection,
            0.0001f, 1e16f, 0.0f,
            OptixVisibilityMask( 255 ), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0, hitDistanceFloatAsInt
        );

        float dist = __uint_as_float(hitDistanceFloatAsInt);

        // KHÔNG CHECK FALSE INTERSECTION: Chấp nhận mọi tia đâm trúng
        if (dist > 0.0f && dist < 1e15f) {
            distances[validCount] = dist;

            // Tính góc từ trục Z cục bộ (localZ = cos(góc))
            float angle = fmaxf(acosf(localZ), 0.001f);

            // TRỌNG SỐ: Nghịch đảo của góc (như bài báo Shapira 08)
            weights[validCount] = 1.0f / angle;

            validCount++;
        }
    }

    float finalSDF = 0.0f;

    // =====================================================================
    // BƯỚC 2: LỌC OUTLIER & TÍNH TRUNG BÌNH CÓ TRỌNG SỐ
    // =====================================================================
    if (validCount > 0) {
        // Sao chép sang mảng tạm để sắp xếp
        float sortedDists[MAX_RAYS];
        for (int i = 0; i < validCount; i++) sortedDists[i] = distances[i];

        // Thuật toán Insertion Sort (Nhỏ nhẹ, cực nhanh trên GPU Registers)
        for (int i = 1; i < validCount; i++) {
            float key = sortedDists[i];
            int j = i - 1;
            while (j >= 0 && sortedDists[j] > key) {
                sortedDists[j + 1] = sortedDists[j];
                j--;
            }
            sortedDists[j + 1] = key;
        }

        // Tìm Trung vị (Median)
        float median = (validCount % 2 == 0) ?
            (sortedDists[validCount / 2 - 1] + sortedDists[validCount / 2]) * 0.5f :
            sortedDists[validCount / 2];

        // Tính Độ lệch chuẩn (Standard Deviation)
        float sumSq = 0.0f;
        for (int i = 0; i < validCount; i++) {
            float diff = distances[i] - median;
            sumSq += diff * diff;
        }
        float stddev = sqrtf(sumSq / validCount);

        float totalWeightedDist = 0.0f;
        float totalWeight = 0.0f;

        // Lọc Outlier và Cộng dồn
        for (int i = 0; i < validCount; i++) {
            // LỌC 1-SIGMA: Vứt bỏ các tia có chiều dài nằm ngoài vùng an toàn
            if (fabsf(distances[i] - median) <= stddev) {
                totalWeightedDist += distances[i] * weights[i];
                totalWeight += weights[i];
            }
        }

        finalSDF = (totalWeight > 0.0f) ? (totalWeightedDist / totalWeight) : 0.0f;
    }

    params.outputSDF[idx.x] = finalSDF;
}

// CLOSEST HIT SHADER: TRẢ VỀ KHOẢNG CÁCH VA CHẠM
extern "C" __global__ void __closesthit__sdf() {
    float hit_t = optixGetRayTmax();
    optixSetPayload_0(__float_as_uint(hit_t));
}