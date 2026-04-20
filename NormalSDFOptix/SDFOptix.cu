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
    // 1. Lật ngược Normal để đâm vào trong khối
    float3 invertedNormal = make_float3(-normal.x, -normal.y, -normal.z);

    // 2. TƯ DUY MỚI: Tưởng tượng một cái "Bia bắn cung" hình tròn phẳng
    // Đặt cái bia này cách điểm bắn đúng 1 mét (z = 1).
    // Dùng lượng giác cơ bản: Bán kính lớn nhất của bia = tan(góc_mở)
    float maxRadius = tanf(angleRad);

    // 3. CHỌN MỘT ĐIỂM NGẪU NHIÊN BÊN TRONG DIỆN TÍCH HÌNH TRÒN ĐÓ
    // BÍ THUẬT: Phải có Căn Bậc Hai sqrtf() thì đạn mới tản đều khắp mặt bia.
    // Nếu không có căn, đạn sẽ bị túm tụm dày đặc ở hồng tâm.
    float r = maxRadius * sqrtf(randVal.x);
    float phi = 2.0f * 3.14159265f * randVal.y;

    // Tọa độ của viên đạn TRÊN MẶT BIA PHẲNG
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = 1.0f; // Luôn nằm trên mặt phẳng z = 1

    // 4. CHUẨN HÓA LẠI ĐỘ DÀI
    // Tia đạn bắn từ mắt (0,0,0) tới điểm (x,y,1) trên bia sẽ dài hơn 1.
    // Ta BẮT BUỘC phải gọi normalize() để gọt nó về độ dài đúng bằng 1 cho OptiX.
    float3 localRay = normalize(make_float3(x, y, z));

    // 5. TẠO TBN VÀ XOAY TIA VÀO BỀ MẶT MÔ HÌNH (Giữ nguyên như cũ)
    float3 up = (abs(invertedNormal.z) < 0.999f) ? make_float3(0,0,1) : make_float3(1,0,0);
    float3 tangent = normalize(cross(up, invertedNormal));
    float3 bitangent = cross(invertedNormal, tangent);

    // Nhân ma trận TBN với tọa độ tia cục bộ
    return make_float3(
        tangent.x * localRay.x + bitangent.x * localRay.y + invertedNormal.x * localRay.z,
        tangent.y * localRay.x + bitangent.y * localRay.y + invertedNormal.y * localRay.z,
        tangent.z * localRay.x + bitangent.z * localRay.y + invertedNormal.z * localRay.z
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