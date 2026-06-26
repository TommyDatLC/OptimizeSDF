// File: SDFOptix.cu
#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../Core/MathHelper.cu"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define MAX_RAYS 128

#include "OptixHostUtils.cuh"

extern "C" __constant__ Params params;

// =========================================================================
// RAYGEN SHADER: BẮN TIA VÀ TÍNH TOÁN SDF
// =========================================================================
extern "C" __global__ void __raygen__sdf_cone() {
    const uint3 idx = optixGetLaunchIndex();

    // Dữ liệu đã là Row-Major float3, gọi phát ăn luôn, không cần w
    float3 origin = params.vertices[idx.x];
    float3 normal = params.normals[idx.x];

    float3 invertedNormal = make_float3(-normal.x, -normal.y, -normal.z);

    float3 up = (fabsf(invertedNormal.z) < 0.999f) ? make_float3(0,0,1) : make_float3(1,0,0);
    float3 tangent = normalize(cross(up, invertedNormal));
    float3 bitangent = cross(invertedNormal, tangent);

    int rayLimit = (params.raysPerPoint > MAX_RAYS) ? MAX_RAYS : params.raysPerPoint;
    float cosThetaMax = cosf(params.coneAngleRad / 2.0f);

    int validCount = 0;
    int baseIdx = idx.x * rayLimit;

    for (int i = 0; i < rayLimit; ++i) {
        float2 uv = params.hammersleyUVs[i];

        float z = cosThetaMax + (1.0f - cosThetaMax) * uv.x;
        float phi = 2.0f * M_PIf * uv.y;
        float radius = sqrtf(fmaxf(0.0f, 1.0f - z*z));

        float localX = radius * cosf(phi);
        float localY = radius * sinf(phi);
        float localZ = z;

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

        if (dist > 0.0f && dist < 1e15f) {
            params.outputDistances[baseIdx + validCount] = dist;
            float angle = fmaxf(acosf(localZ), 0.001f);
            params.outputWeights[baseIdx + validCount] = 1.0f / angle;
            validCount++;
        }
    }

    // Fill invalid rays with large distances so they sort to the end if sorted as a fixed segment block
    for (int i = validCount; i < rayLimit; i++) {
        params.outputDistances[baseIdx + i] = 1e16f;
        params.outputWeights[baseIdx + i] = 0.0f;
    }

    params.validCounts[idx.x] = validCount;
}

extern "C" __global__ void __closesthit__sdf() {
    float hit_t = optixGetRayTmax();
    optixSetPayload_0(__float_as_uint(hit_t));
}