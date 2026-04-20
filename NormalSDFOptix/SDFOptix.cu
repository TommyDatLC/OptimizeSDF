// File: SDFOptix.cu
#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../Core/mathHelper.cu" // Chứa các hàm float3 của bạn

// 1. ĐỊNH NGHĨA STRUCT PARAMS ĐỂ NHẬN DỮ LIỆU TỪ CPU
struct Params {
    float3* vertices;
    float3* normals;
    float* outputSDF;
    OptixTraversableHandle bvhHandle;
    int raysPerPoint;
    float coneAngle;
};

// Khai báo params trên vùng nhớ hằng (Constant Memory) của OptiX
extern "C" __constant__ Params params;

// 2. HÀM TẠO SỐ NGẪU NHIÊN TRÊN GPU (Đơn giản hóa bằng phép băm - Hash)
__device__ float2 GetRandomFloat2(unsigned int id, unsigned int sample) {
    // Thuật toán hash đơn giản tạo số giả ngẫu nhiên (Pseudo-random)
    unsigned int h = id * 1103515245 + sample * 12345;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    // Map về khoảng [0.0, 1.0]
    return make_float2((float)(h & 0xffffff) / (float)0xffffff,
                       (float)((h >> 8) & 0xffffff) / (float)0xffffff);
}

// 3. HÀM SINH TIA TRONG HÌNH NÓN (Uniform Cone Sampling)
__device__ float3 SampleCone(float3 normal, float2 randVal, float angleRad) {
    // randVal.x và randVal.y nằm trong khoảng [0, 1]
    float cosTheta = cosf(angleRad);
    float z = 1.0f - randVal.x * (1.0f - cosTheta);
    float radius = sqrtf(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * 3.14159265f * randVal.y;

    // Tọa độ cục bộ của tia trong không gian nón (trục Z hướng lên)
    float x = radius * cosf(phi);
    float y = radius * sinf(phi);

    // Tạo hệ tọa độ cục bộ (TBN) từ Normal để xoay nón về đúng hướng
    float3 up = (abs(normal.z) < 0.999f) ? make_float3(0,0,1) : make_float3(1,0,0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    // Xoay tia từ không gian cục bộ sang không gian thế giới
    return make_float3(
        tangent.x * x + bitangent.x * y + normal.x * z,
        tangent.y * x + bitangent.y * y + normal.y * z,
        tangent.z * x + bitangent.z * y + normal.z * z
    );
}

// =========================================================================
// RAYGEN SHADER: CHƯƠNG TRÌNH CHÍNH BẮN TIA
// =========================================================================
extern "C" __global__ void __raygen__sdf_cone() {
    const uint3 idx = optixGetLaunchIndex();

    float3 origin = params.vertices[idx.x];
    float3 normal = params.normals[idx.x];

    float totalDistance = 0.0f;
    int hitCount = 0;

    for (int i = 0; i < params.raysPerPoint; ++i) {
        float2 randomSeed = GetRandomFloat2(idx.x, i);
        float3 rayDirection = SampleCone(normal, randomSeed, params.coneAngle);

        unsigned int hitDistanceFloatAsInt;
        optixTrace(
            params.bvhHandle,
            origin,
            rayDirection,
            0.001f,
            1e16f,
            0.0f,
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0,
            hitDistanceFloatAsInt
        );

        float dist = __uint_as_float(hitDistanceFloatAsInt);

        if (dist > 0.0f && dist < 1e15f) {
            totalDistance += dist;
            hitCount++;
        }
    }

    float avgDistance = (hitCount > 0) ? (totalDistance / hitCount) : 0.0f;
    params.outputSDF[idx.x] = avgDistance;
}

// CLOSEST HIT SHADER: TRẢ VỀ KHOẢNG CÁCH VA CHẠM
extern "C" __global__ void __closesthit__sdf() {
    float hit_t = optixGetRayTmax();
    optixSetPayload_0(__float_as_uint(hit_t));
}