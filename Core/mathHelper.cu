#ifndef MATH_HELPER
#define MATH_HELPER

#include <cmath>
#include <vector_types.h>
// Cấu trúc dữ liệu float3 cơ bản


// 1. Hàm trừ hai vector
inline float3 sub(const float3& a, const float3& b) {
    return float3{a.x - b.x, a.y - b.y, a.z - b.z};
}


inline __device__ __host__ float3 cross(const float3& a, const float3& b) {
    return float3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// 3. Hàm tính tích vô hướng (Dot Product)
// Kết quả là một giá trị vô hướng (float)
inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 4. Hàm chuẩn hóa vector (Normalize)
// Đưa độ dài vector về 1 đơn vị
inline __device__ __host__ float3 normalize(float3 v) {
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.000001f) {
        v.x /= len;
        v.y /= len;
        v.z /= len;
    }
    return v;
}

inline __device__ float3 operator-(const float3& a, const float3& b) {
    return float3(a.x - b.x, a.y - b.y, a.z - b.z );

}
#endif