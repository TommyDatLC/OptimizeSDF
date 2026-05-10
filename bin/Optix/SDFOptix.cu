// File: SDFOptix.cu
#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../Core/MathHelper.cu" // Chứa các hàm float3 của bạn

struct Params {
    float3* vertices;
    float3* normals;
    float* outputSDF;
    float3* localRays;
    OptixTraversableHandle bvhHandle;
    int raysPerPoint;
    int padding;
};

extern "C" __constant__ Params params;

// =========================================================================
// RAYGEN SHADER: CHƯƠNG TRÌNH CHÍNH BẮN TIA
// =========================================================================
extern "C" __global__ void __raygen__sdf_cone() {
    const uint3 idx = optixGetLaunchIndex();

    float3 origin = params.vertices[idx.x];
    float3 normal = params.normals[idx.x];

    float3 invertedNormal = make_float3(-normal.x, -normal.y, -normal.z);

    float3 up = (fabsf(invertedNormal.z) < 0.999f) ? make_float3(0,0,1) : make_float3(1,0,0);
    float3 tangent = normalize(cross(up, invertedNormal));
    float3 bitangent = cross(invertedNormal, tangent);

    float totalDistance = 0.0f;
    int hitCount = 0;

    for (int i = 0; i < params.raysPerPoint; ++i) {
        float3 localRay = params.localRays[i];

        float3 rayDirection = make_float3(
            tangent.x * localRay.x + bitangent.x * localRay.y + invertedNormal.x * localRay.z,
            tangent.y * localRay.x + bitangent.y * localRay.y + invertedNormal.y * localRay.z,
            tangent.z * localRay.x + bitangent.z * localRay.y + invertedNormal.z * localRay.z
        );
        rayDirection = normalize(rayDirection);

        // FIX LỖI RÁC BỘ NHỚ: Khởi tạo giá trị mặc định là -1.0f (Biểu thị cho MISS)
        unsigned int hitDistanceFloatAsInt = __float_as_uint(-1.0f);

        optixTrace(
            params.bvhHandle,
            origin,
            rayDirection,
            0.0001f, 1e16f, 0.0f,
            OptixVisibilityMask( 255 ), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0,
            hitDistanceFloatAsInt
        );

        float dist = __uint_as_float(hitDistanceFloatAsInt);

        // AUTO-FIX INVERTED NORMAL: Nếu bắn vào trong mà trượt, thử bắn lộn ra ngoài!
        if (dist < 0.0f) {
            hitDistanceFloatAsInt = __float_as_uint(-1.0f); // Reset lại
            optixTrace(
                params.bvhHandle,
                origin,
                make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z), // Đảo hướng tia
                0.0001f, 1e16f, 0.0f,
                OptixVisibilityMask( 255 ), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0, 1, 0,
                hitDistanceFloatAsInt
            );
            dist = __uint_as_float(hitDistanceFloatAsInt);
        }

        // Chỉ cộng dồn nếu thực sự có va chạm hợp lệ
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